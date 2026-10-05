"""Snapshot-based opening balances and their shared use by F26/F27/F28."""

from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3_features"))
import balance_reconstruction as br
import feature_engineering_spanish as fes
import parse_pipeline as pipeline


PROFILE = "WOK_Player_Profile"
TX = "WOK_Player_Account_Transaction"
WINDOW = ["01012026", "01022026"]
FEATURES = [
    (fes.f26_balance_drop_frequency, "f26_balance_drop_frequency", 1.0),
    (fes.f27_deposits_after_balance_below_2_per_day, "f27_deposits_after_below2_per_day", 1.0),
    (fes.f28_median_seconds_below2_to_deposit, "f28_median_seconds_below2_to_deposit", 3600.0),
]


def _profile(player, timestamp, balance):
    return {"Player_Profile_ID": player, "Extraction_Date": timestamp,
            "Player_Profile_EOD_Balance": balance}


def _tx(player, timestamp, amount, type_="STAKE", status="SUCCESSFUL"):
    return {"Player_Profile_ID": player, "Transaction_Datetime": timestamp,
            "Transaction_Amount": amount, "Transaction_Type": type_, "Transaction_Status": status}


def _write(path, rows, sep=","):
    df = pd.DataFrame(rows)
    if sep == ";":
        df.columns = df.columns.str.lower()
    df.to_csv(path, sep=sep, decimal="," if sep == ";" else ".", index=False)
    return path


@pytest.mark.parametrize("chunksize", [1, 2, 100])
@pytest.mark.parametrize("sep", [",", ";"])
def test_later_chunk_snapshot_reconstructs_full_first_day(tmp_path, chunksize, sep):
    # Profiles are deliberately out of chronological order across files/chunks.
    first = _write(tmp_path / "profiles_1.csv", [
        _profile("P1", "2026-01-20T00:00:00Z", 700),
        _profile("P3", "2026-01-01T00:00:00Z", 20),
    ], sep)
    last = _write(tmp_path / "profiles_2.csv", [
        _profile("P1", "2026-01-02T00:00:00Z", 51),
        _profile("P1", "2026-01-03T00:00:00Z", 200),
    ], sep)
    tx = _write(tmp_path / "tx.csv", [
        _tx("P1", "2025-12-31T12:00:00Z", -500),  # outside bridge
        _tx("P1", "2026-01-01T12:00:00Z", 99),   # positive stake must become -99
        _tx("P1", "2026-01-01T13:00:00Z", 50, "DEPOSIT"),
        _tx("P1", "2026-01-01T14:00:00Z", 500, "DEPOSIT", "FAILED"),
        _tx("P1", "2026-01-02T00:00:00Z", -7),   # at anchor: not in bridge
    ], sep)
    tables = {PROFILE: [first, last], TX: [tx]}
    with patch.object(br, "iter_csv_chunks", wraps=br.iter_csv_chunks) as reads:
        result = br.reconstruct_start_balances(tables, x_tijdspad=WINDOW, chunksize=chunksize)
    assert result == {"P1": 100.0, "P3": 20.0}
    assert len(reads.call_args_list) == 2  # one profile scan, one transaction scan


def test_latest_older_snapshot_is_advanced_to_start(tmp_path):
    profiles = _write(tmp_path / "profiles.csv", [
        _profile("P1", "2025-12-31T00:00:00Z", 100),
        _profile("P1", "2025-12-30T00:00:00Z", 500),
        _profile("P1", "2026-01-02T00:00:00Z", 999),
    ])
    tx = _write(tmp_path / "tx.csv", [
        _tx("P1", "2025-12-30T12:00:00Z", -10),
        _tx("P1", "2025-12-31T00:00:00Z", 30, "DEPOSIT"),
        _tx("P1", "2025-12-31T12:00:00Z", -20),
        _tx("P1", "2026-01-01T00:00:00Z", -5),
    ])
    assert br.reconstruct_start_balances({PROFILE: [profiles], TX: [tx]}, x_tijdspad=WINDOW, chunksize=1) == {"P1": 110.0}


def test_snapshot_exactly_at_start_including_zero_needs_no_extra_tx_scan(tmp_path):
    profiles = _write(tmp_path / "profiles.csv", [_profile("P1", "2026-01-01T00:00:00Z", 0)])
    with patch.object(br, "iter_csv_chunks", wraps=br.iter_csv_chunks) as reads:
        assert br.reconstruct_start_balances({PROFILE: [profiles]}, x_tijdspad=WINDOW) == {"P1": 0.0}
    assert reads.call_count == 1


def test_no_anchor_missing_values_and_target_period_snapshots_stay_unknown(tmp_path):
    profiles = _write(tmp_path / "profiles.csv", [
        _profile("P1", "2026-02-01T00:00:00Z", 100),
        _profile("P2", "2026-01-01T00:00:00Z", None),
        _profile("P3", "invalid", 100),
        _profile("P4", "2026-01-01T00:00:00Z", float("inf")),
    ])
    assert br.reconstruct_start_balances({PROFILE: [profiles]}, x_tijdspad=WINDOW) == {}
    assert br.reconstruct_start_balances({}, x_tijdspad=WINDOW) == {}


@pytest.mark.parametrize("bad_amount,bad_timestamp", [("bad", "2026-01-01T12:00:00Z"), (float("inf"), "2026-01-01T12:00:00Z"), (10, "invalid")])
def test_unusable_bridging_transaction_does_not_invent_opening_balance(tmp_path, bad_amount, bad_timestamp):
    profiles = _write(tmp_path / "profiles.csv", [_profile("P1", "2026-01-02T00:00:00Z", 50)])
    tx = _write(tmp_path / "tx.csv", [_tx("P1", bad_timestamp, bad_amount)])
    assert br.reconstruct_start_balances({PROFILE: [profiles], TX: [tx]}, x_tijdspad=WINDOW) == {}


def test_without_window_reconstructs_opening_of_available_history(tmp_path):
    profiles = _write(tmp_path / "profiles.csv", [_profile("P1", "2026-01-02T00:00:00Z", 51)])
    tx = _write(tmp_path / "tx.csv", [
        _tx("P1", "2026-01-01T12:00:00Z", 99),
        _tx("P1", "2026-01-01T13:00:00Z", 50, "DEPOSIT"),
    ])
    assert br.reconstruct_start_balances({PROFILE: [profiles], TX: [tx]}, chunksize=1) == {"P1": 100.0}


@pytest.mark.parametrize("fn,column,expected", FEATURES)
def test_known_zero_opening_balance_is_used_instead_of_being_treated_as_missing(tmp_path, fn, column, expected):
    profiles = _write(tmp_path / "profiles.csv", [_profile("P1", "2026-01-01T00:00:00Z", 0)])
    tx = _write(tmp_path / "tx.csv", [
        _tx("P1", "2026-01-01T10:00:00Z", 100, "DEPOSIT"),
        _tx("P1", "2026-01-01T12:00:00Z", 99),
        _tx("P1", "2026-01-01T13:00:00Z", 50, "DEPOSIT"),
    ])
    result = fn({PROFILE: [profiles], TX: [tx]}, x_tijdspad=WINDOW, chunksize=1)
    assert result.set_index("Player_Profile_ID").loc["P1", column] == expected


def test_snapshot_offsets_are_compared_in_utc_and_unsorted_transactions_can_be_summed(tmp_path):
    profiles = _write(tmp_path / "profiles.csv", [_profile("P1", "2026-01-02T01:00:00+01:00", 51)])
    tx = _write(tmp_path / "tx.csv", [
        _tx("P1", "2026-01-01T14:00:00+01:00", 50, "DEPOSIT"),
        _tx("P1", "2026-01-01T12:00:00Z", 99),
    ])
    assert br.reconstruct_start_balances({PROFILE: [profiles], TX: [tx]}, x_tijdspad=WINDOW) == {"P1": 100.0}


def _feature_data(tmp_path, snapshot_timestamp="2026-01-02T00:00:00Z", balance=51):
    profiles = _write(tmp_path / f"{PROFILE}.csv", [_profile("P1", snapshot_timestamp, balance)])
    tx = _write(tmp_path / f"{TX}.csv", [
        _tx("P1", "2026-01-01T12:00:00Z", 99),
        _tx("P2", "2026-01-01T12:00:00Z", 4),
        _tx("P1", "2026-01-01T13:00:00Z", 50, "DEPOSIT"),
        _tx("P2", "2026-01-01T13:00:00Z", 10, "DEPOSIT"),
    ])
    return {PROFILE: [profiles], TX: [tx]}


@pytest.mark.parametrize("chunksize", [1, 100])
@pytest.mark.parametrize("fn,column,expected", FEATURES)
@pytest.mark.parametrize("timestamp,balance", [("2026-01-02T00:00:00Z", 51), ("2026-01-01T00:00:00Z", 100)])
def test_features_include_first_day_drop_even_when_eod_recovered_and_keep_unknown_players(
    tmp_path, chunksize, fn, column, expected, timestamp, balance
):
    tables = _feature_data(tmp_path, timestamp, balance)
    result = fn(tables, x_tijdspad=WINDOW, chunksize=chunksize).set_index("Player_Profile_ID")[column]
    assert result["P1"] == expected
    assert pd.isna(result["P2"])


def test_pipeline_shares_reconstruction_once_and_does_not_cache_between_runs(tmp_path, monkeypatch):
    cleaned = tmp_path / "cleaned_20260101_000000"
    cleaned.mkdir()
    _feature_data(cleaned)
    keys = [fn.__name__ for fn, _, _ in FEATURES]
    monkeypatch.setitem(pipeline.SCENARIOS, "balance_test", {"features": keys})
    with patch.object(pipeline, "reconstruct_start_balances", wraps=br.reconstruct_start_balances) as reconstruct:
        first = pipeline.run_scenario("balance_test", cleaned, x_tijdspad=":".join(WINDOW), chunksize=1)
        assert reconstruct.call_count == 1
        for _, column, expected in FEATURES:
            assert first.set_index("Player_Profile_ID").loc["P1", column] == expected
            assert pd.isna(first.set_index("Player_Profile_ID").loc["P2", column])
        second = pipeline.run_scenario("balance_test", cleaned, x_tijdspad=":".join(WINDOW), ignore_EOD_Balance=True)
        assert reconstruct.call_count == 2
        assert second[[column for _, column, _ in FEATURES]].isna().all().all()
    output = tmp_path / "features.csv"
    pipeline.build_features(cleaned, "balance_test", x_tijdspad=":".join(WINDOW), features_out=output)
    saved = pd.read_csv(output).set_index("Player_Profile_ID")
    assert saved.loc["P2", [column for _, column, _ in FEATURES]].isna().all()


def test_merge_preserves_unknown_balance_features_without_changing_other_features():
    left = pd.DataFrame({"Player_Profile_ID": ["P1"], "f26_balance_drop_frequency": [float("nan")], "other_feature": [float("nan")]})
    right = pd.DataFrame({"Player_Profile_ID": ["P1", "P2"], "another_feature": [1.0, 2.0]})
    result = pipeline._safe_merge(left, right)
    assert result["f26_balance_drop_frequency"].isna().all()
    assert result["other_feature"].eq(0).all()
