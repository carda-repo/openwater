# ============================================================
# In dit blok wordt de manier waarop we CSV-bestanden inlezen volledig aangepast zodat het geschikt is voor streaming en grote datasets (zoals op Snellius).

# Voorheen werden alle CSV-bestanden in één keer als complete pandas DataFrames ingeladen. Dat werkt goed voor kleine bestanden, maar bij honderden miljoenen regels kan dit het geheugen snel opblazen. Daarom gebruiken we nu een aanpak waarbij:
# 	•	load_tables_local(...) alleen nog de padnamen (Paths) van de relevante bestanden ophaalt en teruggeeft, gegroepeerd per logisch WOK-bestand (bijvoorbeeld WOK_Game_Session). Er worden hier dus nog géén DataFrames ingeladen.
# 	•	iter_csv_chunks(...) leest deze bestanden vervolgens stapsgewijs in chunks (standaard 200.000 regels per keer) en levert pandas DataFrames op per chunk. Zo kunnen we rijen verwerken zonder alles tegelijk in het geheugen te hebben.

# In FILE_DEFINITIONS is voor elk logisch bestandstype vastgelegd:
# 	•	df_name: hoe het DataFrame later in de code heet;
# 	•	validation_key: welke schema-sleutel uit organisatie_SCHEMA_BY_FILE gebruikt moet worden om de inhoud te valideren.

# De functie _logical_name_from_path probeert uit een bestandsnaam de bijbehorende logische WOK-naam te halen (bijv. uit HardRock_WOK_Game_Session_1.csv → WOK_Game_Session). Op basis hiervan worden de bestanden gebucket per tabeltype.

# Deze structuur maakt het eenvoudiger om:
# 	•	bestanden snel te scannen en te koppelen aan de juiste tabel;
# 	•	streamingverwerking toe te passen zonder complexe aanpassingen in de rest van de code;
# 	•	consistent te blijven werken met de validatieschema’s.
# ============================================================

from pathlib import Path
from typing import Dict, List, Optional, Iterable, Generator, Tuple
import re
import pandas as pd


def _natural_sort_key(path: Path):
    """
    Natural sort key for file paths.
    Sorts _1, _2, ... _10, _11 correctly (numerically) instead of alphabetically.
    Example: file_1.csv, file_2.csv, file_10.csv instead of file_1.csv, file_10.csv, file_2.csv
    """
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split(r'(\d+)', str(path))]

# 1) File definitions (your reference of all possible filetypes)
FILE_DEFINITIONS = {
    'operator':                 {'df_name': 'operator_df',                 'validation_key': 'WOK_Operator.csv'},
    'complaint':                {'df_name': 'complaints_df',               'validation_key': 'WOK_Complaint.csv'},
    'bet':                      {'df_name': 'bet_df',                      'validation_key': 'WOK_Bet.csv'},
    'game_session':             {'df_name': 'game_session_df',             'validation_key': 'WOK_Game_Session.csv'},
    'game':                     {'df_name': 'game_df',                     'validation_key': 'WOK_Game.csv'},
    'intervention':             {'df_name': 'intervention_df',             'validation_key': 'WOK_Intervention.csv'},
    'player_limits':            {'df_name': 'player_limits_df',            'validation_key': 'WOK_Player_Limits.csv'},
    'net_deposit_threshold':    {'df_name': 'net_deposit_threshold_df',    'validation_key': 'WOK_Net_Deposit_Threshold.csv'},
    'player_account_transaction':{'df_name': 'player_account_transaction_df','validation_key': 'WOK_Player_Account_Transaction.csv'},
    'player_flags':             {'df_name': 'player_flags_df',             'validation_key': 'WOK_Player_Flags.csv'},
    'player_profile':           {'df_name': 'player_profile_df',           'validation_key': 'WOK_Player_Profile.csv'},
}


def _logical_name_from_path(p: Path, verbose: bool = False) -> Optional[str]:
    """
    Parse logical WOK_* table name from a filename like:
      <hash>_WOK_Game_Session_1.csv
      WOK_Game_Session.csv
      Casino_WOK_Game session.csv   (handles spaces → underscores)
    Returns: 'WOK_Game_Session' or None if not found.
    """
    stem = p.stem
    if stem.startswith("daily_"):
        return None
    s_norm = stem.replace(" ", "_")
    m = re.search(r'(WOK_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)', s_norm)
    if not m:
        if verbose:
            print(f"    ✖ No WOK_* pattern in: {p.name} (normalized: {s_norm})")
        return None
    logical = m.group(1)
    logical = re.sub(r'_\d+$', '', logical)  # drop trailing _<digits> split index
    if verbose:
        print(f"    ✓ Parsed logical: {logical}  (from {p.name})")
    return logical

def debug_scan_files(base_dir: Path, glob_pattern: str = "*.csv", recursive: bool = True, limit: int = 50):
    base_dir = Path(base_dir)
    print(f"\n🔎 DEBUG SCAN")
    print(f"Base dir: {base_dir}")
    print(f"Glob: {glob_pattern} | Recursive: {recursive}")

    files = sorted(base_dir.rglob(glob_pattern) if recursive else base_dir.glob(glob_pattern), key=_natural_sort_key)

    # Skip derived / helper exports like daily_*.csv (not real WOK tables)
    files = [f for f in files if not f.name.startswith("daily_")]

    # (optional) also skip the helper directory entirely
    files = [f for f in files if "_daily_timeseries" not in f.parts]

    print(f"Found {len(files)} file(s).")
    if not files:
        print("⚠️ No files found — check the path or use recursive=True.")
        return

    print(f"\nFirst {min(len(files), limit)} file(s):")
    for p in files[:limit]:
        _ = _logical_name_from_path(p, verbose=True)

def load_tables_local(
    base_dir: Path,
    logical_tables: List[str],
    usecols_by_table: Dict[str, List[str]] = None,  # kept for signature compatibility; not used here
    glob_pattern: str = "*.csv",
    recursive: bool = True,
    verbose: bool = True,
):
    """
    STREAMING VERSION (paths only):
    Discover CSV files for each logical table and return a dict:
        { logical_table: [Path(...), Path(...), ...] }
    plus missing flags for requested WOK_Game_Session, WOK_Bet and WOK_Complaint
    tables. verbose controls diagnostic output only.
    """
    base_dir = Path(base_dir)
    files = sorted(base_dir.rglob(glob_pattern) if recursive else base_dir.glob(glob_pattern), key=_natural_sort_key)

    # ✅ Skip certain directories to avoid test files and incomplete CSV schemas
    skip_dirs = ("cleaned", "archive", "example_files_json", "example_player_profile_for_labelling",
                 "testoutput", "test_output", "logs")
    if not base_dir.name.startswith(skip_dirs):
        files = [f for f in files if not any(
            part.startswith(skip_dirs) or part.startswith("test_")
            for part in f.parts
        )]

    if verbose:
        print(f"\n📂 load_tables_local()  [STREAMING: returns paths]")
        print(f"Base dir: {base_dir}")
        print(f"Glob: {glob_pattern} | Recursive: {recursive}")
        print(f"Total CSV candidates: {len(files)}")

    if not files:
        raise FileNotFoundError(f"No CSVs found under {base_dir} (recursive={recursive})")

    # Bucket op EXACTE logische naam = bestandsstam minus een chunk-suffix '_<cijfers>'
    # (case-insensitief). GEEN substring-match: anders belandt bv. WOK_Bet_Transaction.csv in de
    # WOK_Bet-bucket ("WOK_Bet" ⊂ "WOK_Bet_Transaction"), wat de organisatie-relationele tabellen kruisbesmet.
    lt_ci = {t.lower(): t for t in logical_tables}
    buckets: Dict[str, List[Path]] = {}
    for f in files:
        s_norm = f.stem.replace(" ", "_")
        if s_norm.startswith("daily_"):
            continue
        parts = s_norm.rsplit("_", 1)
        tbl = parts[0] if (len(parts) == 2 and parts[1].isdigit()) else s_norm
        canon = lt_ci.get(tbl.lower())
        if canon:
            buckets.setdefault(canon, []).append(f)
    missing_tables = set(lt_ci) - {table.lower() for table in buckets}
    wok_game_session_missing = "wok_game_session" in missing_tables
    wok_bet_missing = "wok_bet" in missing_tables
    wok_complaint_missing = "wok_complaint" in missing_tables
    if verbose:
        print("\n🧺 Buckets formed for requested tables (paths only):")
        for t in logical_tables:
            paths = buckets.get(t, [])
            print(f"  - {t}: {len(paths)} file(s)")
            if not paths:
                print("    ⚠️ No matching files for this table. Check naming (WOK_*), underscores/spaces, or subfolders.")
                if t.lower() in ("wok_game_session", "wok_bet", "wok_complaint"):
                    print("      (Hint: This table is optional; if absent, features depending on it will be skipped.)")
            else:
                for p in paths:
                    try:
                        rel = p.relative_to(base_dir)
                    except Exception:
                        rel = p
                    print(f"    → {rel}")

    if not buckets:
        raise FileNotFoundError(f"No files matched requested tables: {logical_tables}")

    # Return PATHS, and if some files are missing.
    return buckets, wok_game_session_missing, wok_bet_missing, wok_complaint_missing

def iter_csv_chunks(
    paths: List[Path],
    usecols: Optional[List[str]] = None,
    chunksize: int = 200_000,
    verbose: bool = True,
):
    """
    Yield pandas DataFrame chunks across a list of CSV files.

    Parameters
    ----------
    paths : list of Path
        CSV files to read in order.
    usecols : list of str or None
        Subset of columns to load (saves memory). Pass the subset that your
        feature actually needs.
    chunksize : int
        Number of rows per chunk to yield.
    verbose : bool
        Print basic progress info.

    Yields
    ------
    pd.DataFrame
        A chunk (<= chunksize rows) with the requested columns.
    """
    for p in paths:
        if p.stat().st_size == 0:
            if verbose:
                print(f"  ↪ skipping empty file: {p.name}")
            continue
        # organisatie-export = ;-gescheiden + komma-decimalen + NULL; oudere data = komma-gescheiden.
        sep, decimal = _sniff_csv_format(p)
        # Case-tolerante usecols: organisatie-data is snake_case, features vragen vaak CamelCase
        # (Player_Profile_ID ↔ player_profile_id). Resolve case-insensitief + hernoem terug.
        actual_usecols, rename_map = usecols, None
        if usecols is not None:
            try:
                header_cols = list(pd.read_csv(p, sep=sep, nrows=0).columns)
            except Exception:
                header_cols = []
            ci = {c.lower(): c for c in header_cols}
            resolved, rmap = [], {}
            for req in usecols:
                act = ci.get(str(req).lower())
                if act is not None:
                    resolved.append(act)
                    if act != req:
                        rmap[act] = req
            if resolved:
                actual_usecols = resolved
                rename_map = rmap or None
        if verbose:
            print(f"  ↪ reading in chunks: {p.name} (chunksize={chunksize}, sep={sep!r})")
        for chunk in pd.read_csv(
            p,
            usecols=actual_usecols,
            sep=sep,
            decimal=decimal,
            na_values=["NULL"],
            keep_default_na=True,
            low_memory=False,
            memory_map=True,
            chunksize=chunksize,
        ):
            if rename_map:
                chunk = chunk.rename(columns=rename_map)
            yield chunk


def _sniff_csv_format(path) -> tuple:
    """(sep, decimal) uit de header-regel: organisatie = (';', ','); ouder = (',', '.')."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
    except Exception:
        return ",", "."
    return (";", ",") if header.count(";") > header.count(",") else (",", ".")

# sanity check for streaming read (uncomment to run)

# from pathlib import Path
# from path_finding import load_tables_local, iter_csv_chunks

# INPUT_DIR = Path("test_files")

# tables = ["WOK_Player_Account_Transaction"]  # or include "WOK_Game_Session" if present
# buckets = load_tables_local(INPUT_DIR, tables, glob_pattern="*.csv", recursive=True, verbose=True)

# pat_paths = buckets["WOK_Player_Account_Transaction"]
# seen = 0
# for chunk in iter_csv_chunks(pat_paths, usecols=["Player_Profile_ID","Transaction_Amount"], chunksize=100_000):
#     print("chunk rows:", len(chunk), "columns:", list(chunk.columns))
#     seen += len(chunk)
#     if seen > 200_000:
#         break

# print("✅ streaming read sanity OK (rows seen ≈)", seen)
