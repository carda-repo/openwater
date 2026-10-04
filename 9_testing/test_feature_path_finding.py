"""Optional-table presence must not depend on diagnostic output."""

import importlib.util
from itertools import product
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "3_features" / "path_finding.py"
SPEC = importlib.util.spec_from_file_location("feature_path_finding", MODULE_PATH)
FEATURE_PATH_FINDING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FEATURE_PATH_FINDING)
load_tables_local = FEATURE_PATH_FINDING.load_tables_local

OPTIONAL_TABLES = ("WOK_Game_Session", "WOK_Bet", "WOK_Complaint")
ACCOUNT_TABLE = "WOK_Player_Account_Transaction"


@pytest.fixture
def cleaned_dir(tmp_path):
    # A cleaned input directory is also discoverable under pytest's test_* paths.
    directory = tmp_path / "cleaned_20260101_000000"
    directory.mkdir()
    (directory / f"{ACCOUNT_TABLE}.csv").write_text("pk_id\n", encoding="utf-8")
    return directory


@pytest.mark.parametrize("available", list(product((False, True), repeat=3)))
def test_optional_table_presence_is_identical_in_quiet_and_verbose_modes(
    cleaned_dir, capsys, available
):
    expected_buckets = {ACCOUNT_TABLE: [cleaned_dir / f"{ACCOUNT_TABLE}.csv"]}
    for table, present in zip(OPTIONAL_TABLES, available):
        if present:
            path = cleaned_dir / f"{table}.csv"
            path.write_text("pk_id\n", encoding="utf-8")
            expected_buckets[table] = [path]
    requested = [ACCOUNT_TABLE, *OPTIONAL_TABLES]
    expected = (expected_buckets, *(not present for present in available))

    assert load_tables_local(cleaned_dir, requested, verbose=False) == expected
    assert capsys.readouterr().out == ""
    assert load_tables_local(cleaned_dir, requested, verbose=True) == expected
    assert capsys.readouterr().out != ""


def test_unrequested_optional_tables_are_not_reported_missing(cleaned_dir):
    for verbose in (False, True):
        buckets, *missing = load_tables_local(
            cleaned_dir, [ACCOUNT_TABLE], verbose=verbose
        )
        assert list(buckets) == [ACCOUNT_TABLE]
        assert missing == [False, False, False]


def test_optional_table_presence_uses_case_insensitive_names(cleaned_dir):
    bet_path = cleaned_dir / "wok_bet_1.csv"
    bet_path.write_text("pk_id\n", encoding="utf-8")
    requested = [ACCOUNT_TABLE, *(table.lower() for table in OPTIONAL_TABLES)]

    for verbose in (False, True):
        buckets, *missing = load_tables_local(cleaned_dir, requested, verbose=verbose)
        assert buckets["wok_bet"] == [bet_path]
        assert missing == [True, False, True]


@pytest.mark.parametrize(
    "child_table", ["WOK_Bet_Transaction", "WOK_Game_Session_Transaction"]
)
def test_transaction_file_does_not_count_as_optional_parent_table(
    cleaned_dir, child_table
):
    (cleaned_dir / f"{child_table}.csv").write_text("pk_id\n", encoding="utf-8")
    for verbose in (False, True):
        buckets, *missing = load_tables_local(
            cleaned_dir, [ACCOUNT_TABLE, *OPTIONAL_TABLES], verbose=verbose
        )
        assert list(buckets) == [ACCOUNT_TABLE]
        assert missing == [True, True, True]


@pytest.mark.parametrize("verbose", [False, True])
def test_no_matching_requested_tables_still_raises(cleaned_dir, verbose):
    with pytest.raises(FileNotFoundError, match="No files matched requested tables"):
        load_tables_local(cleaned_dir, list(OPTIONAL_TABLES), verbose=verbose)
