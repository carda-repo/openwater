"""F16 uses financial activation history and the last transaction in the period."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3_features"))
from feature_engineering_spanish import f16_account_age
from parse_pipeline import build_features


FEATURE = "f16_account_age"
TABLE = "WOK_Player_Account_Transaction"


def _transaction(player, timestamp, kind="STAKE", status="SUCCESSFUL"):
    return {"Player_Profile_ID": player, "Transaction_Datetime": timestamp,
            "Transaction_Type": kind, "Transaction_Status": status}


def _write(path, rows, sep=",", snake_case=False):
    frame = pd.DataFrame(rows)
    if snake_case:
        frame.columns = frame.columns.str.lower()
    frame.to_csv(path, sep=sep, index=False)
    return path


def _values(result):
    assert list(result.columns) == ["Player_Profile_ID", FEATURE]
    assert result["Player_Profile_ID"].is_unique
    return result.set_index("Player_Profile_ID")[FEATURE].to_dict()


@pytest.mark.parametrize("chunksize", [1, 2, 100])
@pytest.mark.parametrize("sep,snake_case", [(",", False), (";", True)])
def test_activation_before_period_and_latest_financial_transaction_across_files(
    tmp_path, chunksize, sep, snake_case
):
    first = _write(tmp_path / "first.csv", [
        _transaction("P1", "2026-03-20", "WITHDRAWAL"),
        _transaction("P1", "2026-02-10", "DEPOSIT"),
        _transaction("P1", "2026-04-01"),
    ], sep, snake_case)
    second = _write(tmp_path / "second.csv", [
        _transaction("P1", "2026-01-01", "DEPOSIT", "UNSUCCESSFUL"),
        _transaction("P1", "2026-01-05", "OTHER"),
        _transaction("P1", "2026-02-20"),
        _transaction("P1", "2026-03-30", "WINNING"),
        _transaction("P1", "2026-03-31", "DEPOSIT", "UNSUCCESSFUL"),
        _transaction("bonus-only", "2026-03-15", "BONUS"),
        _transaction("history-only", "2026-02-01", "DEPOSIT"),
    ], sep, snake_case)
    profile = _write(tmp_path / "profiles.csv", [{
        "Player_Profile_ID": "P1", "Player_Profile_Registration_Datetime": "2026-01-01",
    }])
    result = f16_account_age(
        {TABLE: [first, second], "WOK_Player_Profile": [profile]},
        x_tijdspad=["01032026", "31032026"], chunksize=chunksize,
    )
    assert _values(result) == {"P1": 38}


def test_period_includes_start_and_entire_end_date_and_excludes_following_day(tmp_path):
    path = _write(tmp_path / "transactions.csv", [
        _transaction("start", "2026-01-01", "DEPOSIT"),
        _transaction("start", "2026-03-01T00:00:00Z"),
        _transaction("end", "2026-01-01", "DEPOSIT"),
        _transaction("end", "2026-03-31T23:59:59.999Z", "WITHDRAWAL"),
        _transaction("end", "2026-04-01T00:00:00Z"),
        _transaction("past", "2026-02-28T23:59:59Z"),
        _transaction("future", "2026-04-01T00:00:00Z"),
    ])
    result = f16_account_age({TABLE: [path]}, x_tijdspad=["01032026", "31032026"], chunksize=1)
    assert _values(result) == {"start": 59, "end": 89}


def test_calendar_days_have_minimum_one_and_no_ten_year_cap(tmp_path):
    path = _write(tmp_path / "transactions.csv", [
        _transaction("same-day", "2026-01-01T23:30:00Z", "DEPOSIT"),
        _transaction("same-day", "2026-01-01T23:59:00Z"),
        _transaction("calendar-days", "2026-01-01T23:59:00Z", "DEPOSIT"),
        _transaction("calendar-days", "2026-01-03T00:01:00Z"),
        _transaction("old-account", "2014-01-01", "DEPOSIT"),
        _transaction("old-account", "2026-01-01"),
    ])
    result = f16_account_age({TABLE: [path]}, x_tijdspad=["01012026", "31012026"], chunksize=2)
    assert _values(result) == {"same-day": 1, "calendar-days": 2, "old-account": 4383}


def test_invalid_rows_are_ignored_and_timestamps_use_utc(tmp_path):
    path = _write(tmp_path / "transactions.csv", [
        _transaction("P1", "2026-01-01T23:30:00-02:00", "DEPOSIT"),
        _transaction("P1", "2026-01-03T00:00:00Z"),
        _transaction("P1", "invalid-date"),
        _transaction("P1", None),
        _transaction(None, "2026-01-05"),
        _transaction(" ", "2026-01-05"),
    ])
    result = f16_account_age({TABLE: [path]}, x_tijdspad=["01012026", "31012026"], chunksize=1)
    assert _values(result) == {"P1": 1}


def test_without_period_uses_last_financial_transaction_in_available_history(tmp_path):
    path = _write(tmp_path / "transactions.csv", [
        _transaction("P1", "2026-01-01", "DEPOSIT"),
        _transaction("P1", "2026-01-10", "WITHDRAWAL"),
        _transaction("P1", "2026-02-01", "WINNING"),
    ])
    assert _values(f16_account_age({TABLE: [path]}, chunksize=1)) == {"P1": 9}


def test_missing_transaction_history_does_not_fall_back_to_registration(tmp_path):
    profile = _write(tmp_path / "profiles.csv", [{
        "Player_Profile_ID": "P1", "Player_Profile_Registration_Datetime": "2026-01-01",
    }])
    result = f16_account_age({"WOK_Player_Profile": [profile]}, x_tijdspad=["01032026", "31032026"])
    assert _values(result) == {}


def test_feature_pipeline_discovers_transaction_table_without_player_profiles(tmp_path):
    cleaned = tmp_path / "cleaned_20260101_000000"
    cleaned.mkdir()
    _write(cleaned / f"{TABLE}.csv", [
        _transaction("P1", "2026-02-10", "DEPOSIT"),
        _transaction("P1", "2026-03-20", "WITHDRAWAL"),
    ], sep=";", snake_case=True)
    output = tmp_path / "features.csv"
    result = build_features(
        cleaned_dir=cleaned, scenario="Flexible_spanish_plus", feature=FEATURE,
        x_tijdspad="01032026:31032026", chunksize=1, features_out=output, verbose=False,
    )
    assert _values(result) == {"P1": 38}
    assert _values(pd.read_csv(output)) == {"P1": 38}
