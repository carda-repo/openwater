"""Bewaar de nieuwste rij per operator/speler/transactie, over alle CSV-delen.

Accounttransacties worden geordend op extraction_date, daarna _write_timestamp
en created_at. Bet- en sessietransacties op created_at, daarna _write_timestamp
en extraction_date. Ontbreekt de eerste tijd, dan gebruiken we de volgende.
Bij gelijke tijden (of zonder geldige tijd) wint de laatst gelezen rij.

De tijdelijke SQLite-index bewaart alleen sleutels, tijden en rijposities. Een
tweede streaming-pass schrijft de geselecteerde volledige rijen terug naar hun
oorspronkelijke bestanden. Parent-IDs en andere rijwaarden worden niet gewijzigd.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from functools import lru_cache
from pathlib import Path
import logging
import re
import sqlite3
import tempfile

import pandas as pd

from check_format import resolve_schema_key


TRANSACTION_TABLES = {
    "WOK_Player_Account_Transaction": ("transaction_id", None,
                                       ("extraction_date", "_write_timestamp", "created_at")),
    "WOK_Bet_Transaction": ("transactions_id", "WOK_Bet",
                             ("created_at", "_write_timestamp", "extraction_date")),
    "WOK_Game_Session_Transaction": ("transaction_id", "WOK_Game_Session",
                                      ("created_at", "_write_timestamp", "extraction_date")),
}
_TABLE_KEYS = [f"{name}{plural}.csv" for name in TRANSACTION_TABLES for plural in ("", "s")]
_MISSING = {"", "null", "none", "nan", "n/a"}
_NO_TIME = -(2**63)


def transaction_table_name(filename: str) -> str | None:
    """Herken ook vendor-prefixes, hoofdletters, meervoud en CSV-deelbestanden."""
    key, _ = resolve_schema_key(filename, _TABLE_KEYS)
    return key[:-4].removesuffix("s") if key else None


def _file_order(path: Path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def _read_options(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as stream:
        header = stream.readline()
    return {"sep": ";" if header.count(";") > header.count(",") else ",",
            "dtype": str, "keep_default_na": False, "na_filter": False,
            "encoding": "utf-8-sig"}


def _columns(path: Path) -> dict[str, str]:
    return {name.strip().lower(): name
            for name in pd.read_csv(path, nrows=0, **_read_options(path)).columns}


def _id(value: str) -> str:
    value = value.strip()
    return "" if value.lower() in _MISSING else value


def _times(chunk: pd.DataFrame, columns: dict, names: tuple) -> tuple[list, list, list]:
    """UTC-nanoseconden, onafhankelijk van pandas' automatisch gekozen tijdseenheid."""
    parsed = []
    for name in names:
        if name in columns:
            ts = pd.to_datetime(chunk[columns[name]], format="mixed", errors="coerce", utc=True)
            parsed.append(ts.dt.as_unit("ns").astype("int64"))
        else:
            parsed.append(pd.Series(_NO_TIME, index=chunk.index, dtype="int64"))
    primary = parsed[0].where(parsed[0] != _NO_TIME, parsed[1])
    primary = primary.where(primary != _NO_TIME, parsed[2])
    return primary.tolist(), parsed[1].tolist(), parsed[2].tolist()


def _index_parent_operators(conn: sqlite3.Connection, root: Path, parent: str | None,
                           chunksize: int) -> None:
    """Subtabellen hebben meestal geen operator_id; haal die via hun parent op."""
    conn.execute("CREATE TABLE parents (pk_id TEXT PRIMARY KEY, operator_id TEXT) WITHOUT ROWID")
    if parent is None:
        return
    for path in sorted(root.glob("*.csv"), key=_file_order):
        key, _ = resolve_schema_key(path.name, [parent + ".csv"])
        if not key:
            continue
        cols = _columns(path)
        if not {"pk_id", "operator_id"} <= cols.keys():
            continue
        for chunk in pd.read_csv(path, chunksize=chunksize,
                                 usecols=[cols["pk_id"], cols["operator_id"]],
                                 **_read_options(path)):
            conn.executemany("INSERT OR REPLACE INTO parents VALUES (?, ?)",
                             [(_id(pk), _id(op)) for pk, op in zip(chunk[cols["pk_id"]], chunk[cols["operator_id"]])
                              if _id(pk)])
    conn.commit()


def _deduplicate_table(root: Path, table: str, paths: list[Path], chunksize: int) -> dict:
    tx_column, parent_table, time_columns = TRANSACTION_TABLES[table]
    stats = {"table": table, "rows_read": 0, "rows_written": 0,
             "duplicates_removed": 0, "missing_ids": 0, "missing_timestamps": 0}
    with tempfile.TemporaryDirectory(prefix=".transaction_dedup_", dir=root) as work:
        with closing(sqlite3.connect(str(Path(work) / "index.sqlite"))) as conn:
            conn.execute("PRAGMA temp_store = FILE")
            conn.execute("""CREATE TABLE winners (
                operator_id TEXT, player_id TEXT, transaction_id TEXT,
                time INTEGER, tie_time INTEGER, last_time INTEGER,
                sequence INTEGER, file_no INTEGER, row_no INTEGER,
                PRIMARY KEY (operator_id, player_id, transaction_id)
            ) WITHOUT ROWID""")
            _index_parent_operators(conn, root, parent_table, chunksize)

            @lru_cache(maxsize=chunksize)
            def parent_operator(pk: str) -> str:
                row = conn.execute("SELECT operator_id FROM parents WHERE pk_id = ?", (pk,)).fetchone()
                return row[0] if row else ""

            headers = []
            for file_no, path in enumerate(paths):
                cols = _columns(path)
                headers.append(cols)
                # transactions_id is the export's plural spelling; accept the singular too.
                tx = cols.get(tx_column) or cols.get("transaction_id") or cols.get("transactions_id")
                player = cols.get("player_profile_id")
                if not player or not tx:
                    raise ValueError(f"{path.name}: player_profile_id en transactie-ID zijn vereist voor deduplicatie")
                op_col = cols.get("operator_id")
                parent_col = cols.get("wok_bet_pk_id" if parent_table == "WOK_Bet"
                                      else "wok_game_session_pk_id") if parent_table else None
                offset = 0
                for chunk in pd.read_csv(path, chunksize=chunksize, **_read_options(path)):
                    primary, secondary, tertiary = _times(chunk, cols, time_columns)
                    players = chunk[player].map(_id).tolist()
                    txids = chunk[tx].map(_id).tolist()
                    operators = chunk[op_col].map(_id).tolist() if op_col else [""] * len(chunk)
                    parents = chunk[parent_col].map(_id).tolist() if parent_col else [""] * len(chunk)
                    candidates = []
                    for i, (pid, tid) in enumerate(zip(players, txids)):
                        if not pid or not tid:
                            stats["missing_ids"] += 1
                            continue
                        if primary[i] == _NO_TIME:
                            stats["missing_timestamps"] += 1
                        operator = operators[i] or (parent_operator(parents[i]) if parents[i] else "")
                        candidates.append((operator, pid, tid, primary[i], secondary[i], tertiary[i],
                                           stats["rows_read"] + i, file_no, offset + i))
                    conn.executemany("""INSERT INTO winners VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (operator_id, player_id, transaction_id) DO UPDATE SET
                            time=excluded.time, tie_time=excluded.tie_time, last_time=excluded.last_time,
                            sequence=excluded.sequence, file_no=excluded.file_no, row_no=excluded.row_no
                        WHERE (excluded.time, excluded.tie_time, excluded.last_time, excluded.sequence)
                            > (winners.time, winners.tie_time, winners.last_time, winners.sequence)
                    """, candidates)
                    conn.commit()
                    stats["rows_read"] += len(chunk)
                    offset += len(chunk)

            conn.execute("CREATE INDEX winner_positions ON winners (file_no, row_no)")
            for file_no, path in enumerate(paths):
                cols = headers[file_no]
                tx = cols.get(tx_column) or cols.get("transaction_id") or cols.get("transactions_id")
                player = cols["player_profile_id"]
                output = Path(work) / f"{file_no}.csv"
                pd.read_csv(path, nrows=0, **_read_options(path)).to_csv(output, index=False)
                offset = 0
                for chunk in pd.read_csv(path, chunksize=chunksize, **_read_options(path)):
                    selected = {row[0] - offset for row in conn.execute(
                        "SELECT row_no FROM winners WHERE file_no = ? AND row_no >= ? AND row_no < ?",
                        (file_no, offset, offset + len(chunk)))}
                    keep = [i in selected or not _id(pid) or not _id(tid)
                            for i, (pid, tid) in enumerate(zip(chunk[player], chunk[tx]))]
                    result = chunk.loc[keep]
                    result.to_csv(output, mode="a", header=False, index=False)
                    stats["rows_written"] += len(result)
                    offset += len(chunk)
            # All outputs are ready before replacing any CSV of this table.
            for file_no, path in enumerate(paths):
                (Path(work) / f"{file_no}.csv").replace(path)
    stats["duplicates_removed"] = stats["rows_read"] - stats["rows_written"]
    return stats


def deduplicate_transactions(cleaned_dir: str | Path, *, chunksize: int = 200_000,
                             logger: logging.Logger | None = None) -> list[dict]:
    """Dedupliceer alleen de drie transactietabellen in één cleaned/operator-map.

    Zonder operator_id of beschikbare parent geldt de inputmap als operator-scope.
    Rijen zonder speler- of transactie-ID blijven afzonderlijk behouden.
    """
    if chunksize < 1:
        raise ValueError("chunksize moet positief zijn")
    root = Path(cleaned_dir)
    grouped = defaultdict(list)
    for path in sorted(root.glob("*.csv"), key=_file_order):
        table = transaction_table_name(path.name)
        if table:
            grouped[table].append(path)
    reports = []
    for table, paths in grouped.items():
        report = _deduplicate_table(root, table, paths, chunksize)
        reports.append(report)
        msg = (f"[DEDUP] {table}: gelezen={report['rows_read']:,}, behouden={report['rows_written']:,}, "
               f"verwijderd={report['duplicates_removed']:,}, ontbrekende IDs={report['missing_ids']:,}, "
               f"zonder geldige versietijd={report['missing_timestamps']:,}")
        if logger:
            logger.info(msg)
        else:
            print(msg, flush=True)
    return reports
