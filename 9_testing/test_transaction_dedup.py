"""Regression tests: newest transactions, global keys and unchanged join values."""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "1_cleaning"))
from clean_pipeline import clean_directory
from transaction_dedup import deduplicate_transactions, transaction_table_name

TABLES = [
    ("WOK_Player_Account_Transaction", "transaction_id"),
    ("WOK_Bet_Transaction", "transactions_id"),
    ("WOK_Game_Session_Transaction", "transaction_id"),
]


def _write(path, rows, sep=","):
    pd.DataFrame(rows).to_csv(path, sep=sep, index=False)


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _row(tx_col, pk, date, player="001", tx="0001", **extra):
    return {"pk_id": pk, "player_profile_id": player, tx_col: tx,
            "created_at": date, "extraction_date": date, **extra}


@pytest.mark.parametrize("table,tx_col", TABLES)
@pytest.mark.parametrize("chunksize", [1, 2, 100])
def test_newest_over_chunks_and_files_preserves_distinct_transactions(tmp_path, table, tx_col, chunksize):
    first = tmp_path / f"vendor_{table}_2.csv"
    second = tmp_path / f"vendor_{table}_10.csv"
    _write(first, [
        _row(tx_col, "old", "2026-01-01"),
        _row(tx_col, "other-player", "2026-01-01", player="002"),
        _row(tx_col, "other-tx", "2026-01-01", tx="0002"),
    ])
    _write(second, [
        _row(tx_col, "new", "2026-01-03"),
        _row(tx_col, "middle", "2026-01-02"),
    ])
    [report] = deduplicate_transactions(tmp_path, chunksize=chunksize)
    result = _rows(first) + _rows(second)
    assert {r["pk_id"] for r in result} == {"new", "other-player", "other-tx"}
    assert {(r["player_profile_id"], r[tx_col]) for r in result} == {
        ("001", "0001"), ("002", "0001"), ("001", "0002")}
    assert report["rows_read"] == 5
    assert report["duplicates_removed"] == 2
    before = [first.read_bytes(), second.read_bytes()]
    [again] = deduplicate_transactions(tmp_path, chunksize=chunksize)
    assert again["duplicates_removed"] == 0
    assert before == [first.read_bytes(), second.read_bytes()]
    assert not list(tmp_path.glob(".transaction_dedup_*"))


def test_account_uses_extraction_then_write_then_created_time(tmp_path):
    path = tmp_path / "WOK_Player_Account_Transaction.csv"
    _write(path, [
        _row("transaction_id", "winner", "2026-01-02", extraction_date="2026-02-02",
             _write_timestamp="2026-02-03"),
        _row("transaction_id", "older-extract", "2026-03-01", extraction_date="2026-02-01",
             _write_timestamp="2026-03-01"),
        _row("transaction_id", "older-write", "2026-03-01", extraction_date="2026-02-02",
             _write_timestamp="2026-02-02"),
    ])
    deduplicate_transactions(tmp_path, chunksize=1)
    assert _rows(path)[0]["pk_id"] == "winner"


@pytest.mark.parametrize("table,tx_col", TABLES[1:])
def test_relational_transactions_use_created_time(tmp_path, table, tx_col):
    path = tmp_path / f"{table}.csv"
    _write(path, [
        _row(tx_col, "new", "2026-02-02", extraction_date="2026-01-01"),
        _row(tx_col, "old", "2026-01-01", extraction_date="2026-03-03"),
    ])
    deduplicate_transactions(tmp_path, chunksize=1)
    assert _rows(path)[0]["pk_id"] == "new"


@pytest.mark.parametrize("table,tx_col", TABLES)
def test_equal_or_missing_times_keep_last_and_missing_ids_stay_separate(tmp_path, table, tx_col):
    path = tmp_path / f"{table}.csv"
    _write(path, [
        _row(tx_col, "first-tie", "2026-01-01"),
        _row(tx_col, "last-tie", "2026-01-01"),
        _row(tx_col, "first-undated", "invalid", tx="undated"),
        _row(tx_col, "last-undated", "", tx="undated"),
        _row(tx_col, "missing-player-1", "2026-01-01", player=""),
        _row(tx_col, "missing-player-2", "2026-01-02", player="NULL"),
        _row(tx_col, "missing-tx-1", "2026-01-01", tx=""),
        _row(tx_col, "missing-tx-2", "2026-01-02", tx="NULL"),
    ])
    [report] = deduplicate_transactions(tmp_path, chunksize=2)
    assert {r["pk_id"] for r in _rows(path)} == {
        "last-tie", "last-undated", "missing-player-1", "missing-player-2", "missing-tx-1", "missing-tx-2"}
    assert report["missing_ids"] == 4
    assert report["missing_timestamps"] == 2


@pytest.mark.parametrize("table,tx_col", TABLES)
def test_missing_primary_time_uses_fallback_and_valid_time_beats_undated(tmp_path, table, tx_col):
    path = tmp_path / f"{table}.csv"
    _write(path, [
        _row(tx_col, "valid", "2026-02-01", extraction_date=""),
        _row(tx_col, "old", "2026-01-01", extraction_date=""),
        _row(tx_col, "undated", "invalid", extraction_date="invalid"),
    ])
    deduplicate_transactions(tmp_path, chunksize=1)
    assert _rows(path)[0]["pk_id"] == "valid"


def test_timezones_and_submicrosecond_order(tmp_path):
    path = tmp_path / "WOK_Bet_Transaction.csv"
    _write(path, [
        _row("transactions_id", "new", "2026-01-01T12:00:00.000000002+01:00"),
        _row("transactions_id", "old", "2026-01-01T11:00:00.000000001Z"),
    ])
    deduplicate_transactions(tmp_path, chunksize=1)
    assert _rows(path)[0]["pk_id"] == "new"


@pytest.mark.parametrize("table,tx_col", TABLES)
def test_operators_are_separate_including_parent_lookup(tmp_path, table, tx_col):
    path = tmp_path / f"{table}.csv"
    rows = [
        _row(tx_col, "a-old", "2026-01-01"),
        _row(tx_col, "b", "2026-01-02"),
        _row(tx_col, "a-new", "2026-01-03"),
    ]
    if table == "WOK_Player_Account_Transaction":
        for row, op in zip(rows, ["a", "b", "a"]):
            row["operator_id"] = op
    else:
        parent = "WOK_Bet" if table == "WOK_Bet_Transaction" else "WOK_Game_Session"
        fk = "wok_bet_pk_id" if parent == "WOK_Bet" else "wok_game_session_pk_id"
        # Header order deliberately differs from the usecols order.
        _write(tmp_path / f"{parent}.csv", [
            {"operator_id": "a", "pk_id": "parent-a"},
            {"operator_id": "b", "pk_id": "parent-b"},
        ])
        for row, pk in zip(rows, ["parent-a", "parent-b", "parent-a"]):
            row[fk] = pk
    _write(path, rows)
    parent_before = {p: p.read_bytes() for p in tmp_path.glob("*.csv") if p != path}
    deduplicate_transactions(tmp_path, chunksize=1)
    assert {r["pk_id"] for r in _rows(path)} == {"a-new", "b"}
    assert all(p.read_bytes() == content for p, content in parent_before.items())


def test_pipeline_deduplicates_three_tables_before_labelling(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    for table, tx_col in TABLES:
        _write(raw / f"{table}.csv", [
            _row(tx_col, "new", "2026-01-03", player=" NA "),
            _row(tx_col, "old", "2026-01-01", player="NA"),
            _row(tx_col, "different", "2026-01-02", tx="0002"),
        ], sep=";")
    _write(raw / "WOK_Bet_Parts.csv", [{"pk_id": "part-1", "wok_bet_pk_id": "bet-1", "part_id": "0001"}], sep=";")
    original = {p: p.read_bytes() for p in raw.glob("*.csv")}
    labelled = []

    def label(directory, **kwargs):
        for table, _ in TABLES:
            result = _rows(Path(directory) / f"{table}.csv")
            assert {r["pk_id"] for r in result} == {"new", "different"}
            assert next(r for r in result if r["pk_id"] == "new")["player_profile_id"] == "NA"
        labelled.append(directory)

    monkeypatch.setattr("clean_pipeline.label_outliers", label)
    cleaned = clean_directory(raw, tmp_path / "output", chunksize=1)
    assert labelled == [str(cleaned)]
    assert all(p.read_bytes() == content for p, content in original.items())
    assert _rows(cleaned / "WOK_Bet_Parts.csv") == [{"pk_id": "part-1", "wok_bet_pk_id": "bet-1", "part_id": "0001"}]
    log = (raw / "logs" / "transaction_dedup.log").read_text()
    assert all(table in log for table, _ in TABLES)


@pytest.mark.parametrize("table,tx_col", TABLES[1:])
@pytest.mark.parametrize("sep", [",", ";"])
def test_pipeline_accepts_relational_transactions_and_first_chunk(tmp_path, table, tx_col, sep):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw / f"{table}.csv", [
        _row(tx_col, "old", "2026-01-01"),
        _row(tx_col, "new", "2026-01-03"),
    ], sep=sep)
    cleaned = clean_directory(raw, tmp_path / "output", chunksize=1,
                              only_first_chunk=True, do_label=False)
    assert [r["pk_id"] for r in _rows(cleaned / f"{table}.csv")] == ["old"]


def test_pipeline_deduplicates_legacy_pipe_quoted_account_transactions(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for record, date, amount in [("new", "2026-02-02T00:00:00Z", "-10.00"),
                                 ("old", "2026-02-01T00:00:00Z", "-20.00")]:
        rows.append({
            "Record_ID": record, "Extraction_Date": date, "Operator_ID": "a",
            "Data_Safe_ID": "ds-a", "Replaced_Record_ID": "",
            "Player_Profile_ID": "001", "Transaction_ID": "0001",
            "Transaction_Datetime": "2026-01-01T10:00:00Z", "Transaction_Amount": amount,
            "Transaction_Deposit_Instrument": "", "Transaction_Type": "STAKE",
            "Transaction_Status": "SUCCESSFUL",
        })
    with (raw / "WOK_Player_Account_Transaction.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), quotechar="|", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    cleaned = clean_directory(raw, tmp_path / "output", chunksize=1, do_label=False)
    [result] = _rows(cleaned / "WOK_Player_Account_Transaction.csv")
    assert result["Record_ID"] == "new"
    assert result["Player_Profile_ID"] == "001"
    assert result["Transaction_ID"] == "0001"
    assert float(result["Transaction_Amount"]) == -10.0


def test_account_legacy_camelcase_and_header_only_csv(tmp_path):
    path = tmp_path / "WOK_Player_Account_Transaction.csv"
    rows = [{"Player_Profile_ID": "001", "Transaction_ID": "0001", "Extraction_Date": date,
             "Record_ID": pk, "Transaction_Amount": amount}
            for pk, date, amount in [("new", "2026-02-02T00:00:00Z", "-10"),
                                    ("old", "2026-02-01T00:00:00Z", "-20")]]
    _write(path, rows)
    empty = tmp_path / "WOK_Bet_Transaction.csv"
    pd.DataFrame(columns=["player_profile_id", "transactions_id", "created_at"]).to_csv(empty, index=False)
    reports = deduplicate_transactions(tmp_path, chunksize=1)
    assert _rows(path) == [rows[0]]
    assert _rows(empty) == []
    assert sum(r["duplicates_removed"] for r in reports) == 1


def test_semicolon_cleaning_keeps_numeric_ids_and_normalizes_comma_decimals(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw / "WOK_Player_Account_Transaction.csv", [
        _row("transaction_id", "new", "2026-01-02", transaction_amount="-10,50"),
        _row("transaction_id", "old", "2026-01-01", transaction_amount="-20,50"),
        _row("transaction_id", "no-amount", "2026-01-01", tx="0002", transaction_amount=""),
    ], sep=";")
    cleaned = clean_directory(raw, tmp_path / "output", chunksize=10, do_label=False)
    result = _rows(cleaned / "WOK_Player_Account_Transaction.csv")
    assert result[0]["player_profile_id"] == "001"
    assert result[0]["transaction_id"] == "0001"
    assert float(result[0]["transaction_amount"]) == -10.5
    assert result[1]["transaction_amount"] == ""


def test_f0_counts_repeated_stake_once_after_cleaning(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        _row("transaction_id", "stake-old", "2026-01-01", tx="stake", transaction_amount=-10),
        _row("transaction_id", "stake-new", "2026-01-02", tx="stake", transaction_amount=-10),
        _row("transaction_id", "win", "2026-01-03", tx="win", transaction_amount=15),
    ]
    for r in rows:
        r.update(transaction_datetime="2026-01-01T10:00:00Z", transaction_status="SUCCESSFUL",
                 transaction_type="WINNING" if r["transaction_id"] == "win" else "STAKE")
    _write(raw / "WOK_Player_Account_Transaction.csv", rows, sep=";")
    cleaned = clean_directory(raw, tmp_path / "output", chunksize=1, do_label=False)
    monkeypatch.syspath_prepend(str(ROOT / "3_features"))
    fes = importlib.import_module("feature_engineering_spanish")
    out = fes.f0_net_winloss({"WOK_Player_Account_Transaction": [cleaned / "WOK_Player_Account_Transaction.csv"]})
    assert out["f0_net_winloss"].tolist() == [5.0]


@pytest.mark.parametrize("name,expected", [
    ("vendor_WOK_BET_TRANSACTIONS_12.csv", "WOK_Bet_Transaction"),
    ("WOK_Game_Session_Transaction_[2].csv", "WOK_Game_Session_Transaction"),
    ("WOK_Player_Account_Transaction_1_out.csv", "WOK_Player_Account_Transaction"),
    ("WOK_Bet_Parts.csv", None),
])
def test_table_names(name, expected):
    assert transaction_table_name(name) == expected
