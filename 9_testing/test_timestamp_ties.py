"""Equal timestamps cannot establish event order, including across CSV chunks."""
from itertools import permutations
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3_features"))
import feature_engineering_spanish as fes
import balance_reconstruction as br
import parse_pipeline as pipeline
from test_balance_reconstruction import FEATURES, WINDOW, PROFILE, TX, _tx, _profile, _write
from test_f51_bet_resolution import _bet, _tables, _values


@pytest.mark.parametrize("chunksize", [1, 2, 100])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_tied_stake_and_deposit_do_not_create_transient_drop(tmp_path, chunksize, reverse, split):
    rows = [_tx("P1", "2026-01-10T12:00:00Z", -9),
            _tx("P1", "2026-01-10T13:00:00+01:00", 10, "DEPOSIT")]
    if reverse:
        rows.reverse()
    paths = [_write(tmp_path / "tx.csv", rows)] if not split else [
        _write(tmp_path / f"tx{i}.csv", [row]) for i, row in enumerate(rows)]
    for fn, column, _ in FEATURES:
        result = fn({TX: paths}, x_tijdspad=WINDOW, chunksize=chunksize,
                    start_balances={"P1": 10}).set_index("Player_Profile_ID")[column]
        if fn == fes.f26_balance_drop_frequency:
            assert result["P1"] == 0
        else:
            assert pd.isna(result["P1"])


@pytest.mark.parametrize("chunksize", [1, 2, 100])
@pytest.mark.parametrize("reverse", [False, True])
def test_net_drop_then_later_multiple_deposits(tmp_path, chunksize, reverse):
    # At 12:00: net -9. At 13:00: two deposits; no identifiable order within group.
    groups = [[_tx("P1", "2026-01-10T12:00:00Z", -14),
               _tx("P1", "2026-01-10T12:00:00Z", 5, "DEPOSIT")],
              [_tx("P1", "2026-01-10T13:00:00Z", 3, "DEPOSIT"),
               _tx("P1", "2026-01-10T13:00:00Z", 5, "DEPOSIT")]]
    if reverse:
        groups = [group[::-1] for group in groups]
    paths = [_write(tmp_path / f"tx{i}.csv", [row]) for i, row in enumerate(sum(groups, []))]
    for (fn, column, _), expected in zip(FEATURES, [1, 2, 3600]):
        result = fn({TX: paths}, x_tijdspad=WINDOW, chunksize=chunksize,
                    start_balances={"P1": 10})
        assert result.iloc[0][column] == expected


@pytest.mark.parametrize("chunksize", [1, 100])
def test_unsorted_balance_history_is_unknown(tmp_path, chunksize):
    tx = _write(tmp_path / "tx.csv", [
        _tx("P1", "2026-01-10T13:00:00Z", 10, "DEPOSIT"),
        _tx("P1", "2026-01-10T12:00:00Z", -9)])
    for fn, column, _ in FEATURES:
        assert pd.isna(fn({TX: [tx]}, x_tijdspad=WINDOW, chunksize=chunksize,
                         start_balances={"P1": 10}).iloc[0][column])


def profile(status, modified, extracted):
    return {"Player_Profile_ID": "P1", "Player_Profile_Status": status,
            "Player_Profile_Modified": modified, "Extraction_Date": extracted}


@pytest.mark.parametrize("chunksize", [1, 2, 100])
def test_f25_latest_extraction_wins_regardless_of_order(tmp_path, chunksize):
    rows = [profile("SELF_EXCLUDED_TEMP", "2026-01-09", "2026-01-09"),
            profile("ACTIVE", "2026-01-10", "2026-01-10"),
            profile("SELF_EXCLUDED_TEMP", "2026-01-10", "2026-01-11")]
    for order in permutations(rows):
        paths = [_write(tmp_path / f"profile{i}.csv", [row]) for i, row in enumerate(order)]
        result = fes.f25_voluntary_suspensions({PROFILE: paths}, x_tijdspad=WINDOW,
                                             chunksize=chunksize)
        assert result.iloc[0]["f25_voluntary_suspensions"] == 1


@pytest.mark.parametrize("chunksize", [1, 100])
@pytest.mark.parametrize("reverse", [False, True])
def test_f25_conflict_unknown_and_newer_extraction_can_resolve(tmp_path, chunksize, reverse):
    rows = [profile("ACTIVE", "2026-01-10", "2026-01-11"),
            profile("SELF_EXCLUDED_TEMP", "2026-01-10", "2026-01-11")]
    if reverse:
        rows.reverse()
    def value(rows):
        path = _write(tmp_path / "profiles.csv", rows)
        return fes.f25_voluntary_suspensions({PROFILE: [path]}, x_tijdspad=WINDOW,
            chunksize=chunksize).iloc[0]["f25_voluntary_suspensions"]
    assert pd.isna(value(rows))
    assert value(rows + [profile("SELF_EXCLUDED_TEMP", "2026-01-10", "2026-01-12")]) == 1
    assert pd.isna(value(rows + [profile("ACTIVE", "2026-01-10", "2026-02-10")]))


def test_f25_zero_and_unknown_survive_pipeline_merge(tmp_path):
    path = _write(tmp_path / "profiles.csv", [profile("ACTIVE", "2026-01-10", "2026-01-11")])
    known = fes.f25_voluntary_suspensions({PROFILE: [path]})
    assert known.iloc[0]["f25_voluntary_suspensions"] == 0
    unknown = pd.DataFrame({"Player_Profile_ID": ["P2"], "f25_voluntary_suspensions": [float("nan")]})
    features = pd.concat([known, unknown], ignore_index=True)
    merged = pipeline._safe_merge(pd.DataFrame({"Player_Profile_ID": ["P1", "P2"]}),
                                  features).set_index("Player_Profile_ID")
    assert merged.loc["P1", "f25_voluntary_suspensions"] == 0
    assert pd.isna(merged.loc["P2", "f25_voluntary_suspensions"])


@pytest.mark.parametrize("chunksize", [1, 2, 100])
@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_balance_anchor_unknown(tmp_path, chunksize, reverse):
    rows = [_profile("P1", "2026-01-01", 10), _profile("P1", "2026-01-01", 20)]
    if reverse:
        rows.reverse()
    paths = [_write(tmp_path / f"profile{i}.csv", [row]) for i, row in enumerate(rows)]
    assert br.reconstruct_start_balances({PROFILE: paths}, x_tijdspad=WINDOW,
                                         chunksize=chunksize) == {}
    assert br.reconstruct_start_balances({PROFILE: [_write(tmp_path / "both.csv", rows)]},
                                         x_tijdspad=WINDOW, chunksize=chunksize) == {}


@pytest.mark.parametrize("chunksize", [1, 2, 100])
@pytest.mark.parametrize("reverse", [False, True])
def test_f51_conflicting_latest_status_unknown_newer_status_resolves(tmp_path, chunksize, reverse):
    rows = [_bet("A", "first"), _bet("B", "first", status="BET_PLACED")]
    if reverse:
        rows.reverse()
    next_bet = _bet("C", "next", "21:10", "21:10", "BET_PLACED")
    assert pd.isna(_values(_tables(tmp_path, bets=rows + [next_bet]), chunksize)["P1"])
    # Newer status confirms settlement; original settlement proxy remains 21:00.
    tables = _tables(tmp_path, bets=rows + [_bet("D", "first", extracted="21:05"), next_bet])
    assert _values(tables, chunksize)["P1"] == 600
