"""Refund attribution and exclusion rules for F44, F45, F48 and F51."""

from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3_features"))
import feature_engineering_spanish as fes
import parse_pipeline as pipeline
from stake_time_shares import net_stake_time_shares

TX = "WOK_Player_Account_Transaction"
MORNING = "f44_morning_stakes_percentage"
EVENING = "f45_evening_stakes_percentage"
CASHOUT = "f48_percentage_bets_with_cashout"
REBETS = "f51_median_seconds_loss_to_next_bet"
WINDOW = ["01012026", "01022026"]


def _tx(txid, hour, amount, kind="STAKE", player="P1", status="SUCCESSFUL", day="2026-01-10"):
    return {"Player_Profile_ID": player, "Transaction_ID": txid,
            "Transaction_Datetime": f"{day}T{hour}:00:00Z", "Transaction_Amount": amount,
            "Transaction_Type": kind, "Transaction_Status": status}


def _write(root, table, rows, sep=",", snake=False):
    df = pd.DataFrame(rows)
    if snake:
        df.columns = df.columns.str.lower()
    path = root / f"{table}.csv"
    df.to_csv(path, sep=sep, index=False)
    return [path]


def _bet(pk, identity, status="BET_SETTLED"):
    return {"pk_id": pk, "Bet_ID": identity, "Bet_Status": status,
            "Bet_Start_Datetime": "2026-01-10T09:00:00Z"}


def _bet_ref(pk, txid, player="P1"):
    return {"wok_bet_pk_id": pk, "player_profile_id": player, "transactions_id": txid}


def _session_ref(pk, txid, player="P1"):
    return {"wok_game_session_pk_id": pk, "player_profile_id": player, "transaction_id": txid}


def _values(df, column):
    return df.set_index("Player_Profile_ID")[column]


@pytest.mark.parametrize("chunksize", [1, 2, 100])
@pytest.mark.parametrize("sep,snake", [(",", False), (";", True)])
def test_full_bet_refund_reduces_original_hour_not_refund_hour(tmp_path, chunksize, sep, snake):
    tables = {
        TX: _write(tmp_path, TX, [_tx("morning", "09", -100), _tx("evening", "18", -100),
                                _tx("refund", "20", 100, "VOID_BET")], sep, snake),
        "WOK_Bet": _write(tmp_path, "WOK_Bet", [_bet("old", "B1"), _bet("new", "B1")], sep, snake),
        "WOK_Bet_Transaction": _write(tmp_path, "WOK_Bet_Transaction", [
            _bet_ref("old", "morning"), _bet_ref("new", "refund"),
            _bet_ref("new", "refund"),  # Repeated reference must not multiply refunds.
        ], sep, snake),
    }
    for fn, column, expected in [(fes.f44_morning_stakes_percentage, MORNING, 0),
                                  (fes.f45_evening_stakes_percentage, EVENING, 1)]:
        assert _values(fn(tables, x_tijdspad=WINDOW, chunksize=chunksize), column)["P1"] == expected


@pytest.mark.parametrize("chunksize", [1, 100])
def test_partial_game_refund_and_failed_transactions(tmp_path, chunksize):
    tables = {
        TX: _write(tmp_path, TX, [_tx("morning", "09", -60), _tx("evening", "18", 120),
                                _tx("refund", "06", -30, "VOID_STAKE"),
                                _tx("failed-refund", "06", 90, "VOID_STAKE", status="FAILED"),
                                _tx("failed-stake", "09", -1000, status="FAILED")]),
        "WOK_Game_Session_Transaction": _write(tmp_path, "WOK_Game_Session_Transaction", [
            _session_ref("S1", "evening"), _session_ref("S1", "refund")]),
    }
    result = net_stake_time_shares(tables, x_tijdspad=WINDOW, chunksize=chunksize)
    assert _values(result, MORNING)["P1"] == pytest.approx(.4)
    assert _values(result, EVENING)["P1"] == pytest.approx(.6)


def test_partial_session_refund_is_proportional_across_original_stake_hours(tmp_path):
    tables = {
        TX: _write(tmp_path, TX, [_tx("morning", "09", -100), _tx("evening", "18", -100),
                                _tx("night", "01", -100), _tx("refund", "20", 100, "VOID_STAKE")]),
        "WOK_Game_Session_Transaction": _write(tmp_path, "WOK_Game_Session_Transaction", [
            _session_ref("S1", txid) for txid in ["morning", "evening", "refund"]]),
    }
    result = net_stake_time_shares(tables, chunksize=1)
    assert _values(result, MORNING)["P1"] == .25
    assert _values(result, EVENING)["P1"] == .25


def test_refund_of_stake_before_window_does_not_reduce_unrelated_period_stake(tmp_path):
    tables = {
        TX: _write(tmp_path, TX, [_tx("old", "09", -100, day="2025-12-31"),
                                _tx("evening", "18", -100), _tx("refund", "10", 100, "VOID_BET"),
                                _tx("future", "19", 100, "VOID_BET", day="2026-02-01")]),
        "WOK_Bet_Transaction": _write(tmp_path, "WOK_Bet_Transaction", [
            _bet_ref("B1", "old"), _bet_ref("B1", "refund"),
            _bet_ref("B2", "evening"), _bet_ref("B2", "future")]),
    }
    result = net_stake_time_shares(tables, x_tijdspad=WINDOW, chunksize=1)
    assert _values(result, MORNING)["P1"] == 0
    assert _values(result, EVENING)["P1"] == 1


@pytest.mark.parametrize("refund,refs", [
    (100, [_bet_ref("B1", "stake"), _bet_ref("B1", "refund")]),  # All stakes refunded.
    (50, []),  # Cannot attribute refund.
    (150, [_bet_ref("B1", "stake"), _bet_ref("B1", "refund")]),  # Excessive refund.
    (50, [_bet_ref("B1", "stake"), _bet_ref("B1", "refund"), _bet_ref("B2", "refund")]),
])
def test_undefined_or_unmatched_refunds_leave_shares_unknown(tmp_path, refund, refs):
    tables = {TX: _write(tmp_path, TX, [_tx("stake", "09", -100), _tx("refund", "10", refund, "VOID_BET")])}
    if refs:
        tables["WOK_Bet_Transaction"] = _write(tmp_path, "WOK_Bet_Transaction", refs)
    result = net_stake_time_shares(tables, chunksize=1)
    assert pd.isna(_values(result, MORNING)["P1"])
    assert pd.isna(_values(result, EVENING)["P1"])


def test_same_transaction_id_at_other_player_does_not_refund_their_stake(tmp_path):
    tables = {
        TX: _write(tmp_path, TX, [_tx("stake", "09", -100), _tx("refund", "20", 100, "VOID_BET"),
                                _tx("stake", "18", -100, player="P2")]),
        "WOK_Bet_Transaction": _write(tmp_path, "WOK_Bet_Transaction", [
            _bet_ref("B1", "stake"), _bet_ref("B1", "refund"), _bet_ref("B2", "stake", "P2")]),
    }
    result = net_stake_time_shares(tables, chunksize=1)
    assert pd.isna(_values(result, MORNING)["P1"])
    assert _values(result, EVENING)["P2"] == 1


def test_numeric_parent_id_with_nullable_reference_column_still_links_refund(tmp_path):
    tables = {
        TX: _write(tmp_path, TX, [_tx("stake", "09", -100), _tx("evening", "18", -100),
                                _tx("refund", "20", 100, "VOID_BET")]),
        "WOK_Bet": _write(tmp_path, "WOK_Bet", [_bet(1, "B1")]),
        "WOK_Bet_Transaction": _write(tmp_path, "WOK_Bet_Transaction", [
            _bet_ref(1, "stake"), _bet_ref(1, "refund"), _bet_ref(None, "ignored")]),
    }
    result = net_stake_time_shares(tables, chunksize=100)
    assert _values(result, MORNING)["P1"] == 0
    assert _values(result, EVENING)["P1"] == 1


@pytest.mark.parametrize("chunksize", [1, 2, 100])
def test_cashout_share_excludes_voids_from_both_counts_and_keeps_partial_game_refunds(tmp_path, chunksize):
    tables = {
        TX: _write(tmp_path, TX, [_tx("void", "20", 100, "VOID_BET"),
                                _tx("cash-void", "19", 10, "CASH_OUT"),
                                _tx("cash", "19", 10, "CASH_OUT"),
                                _tx("partial", "20", 5, "VOID_STAKE"),
                                _tx("failed", "20", 100, "VOID_BET", status="FAILED")]),
        "WOK_Bet": _write(tmp_path, "WOK_Bet", [
            _bet("B1", "void-bet"), _bet("B2", "cancelled", "BET_CANCELLED"),
            _bet("B3", "cash-bet"), _bet("B4", "ordinary-bet")]),
        "WOK_Bet_Transaction": _write(tmp_path, "WOK_Bet_Transaction", [
            _bet_ref("B1", "void"), _bet_ref("B1", "cash-void"),
            _bet_ref("B2", "stake"), _bet_ref("B3", "cash"),
            _bet_ref("B4", "partial"), _bet_ref("B4", "failed")]),
    }
    result = fes.f48_percentage_bets_with_cashout(tables, x_tijdspad=WINDOW, chunksize=chunksize)
    assert _values(result, CASHOUT)["P1"] == .5  # One cash-out in two eligible bets.


def test_cashout_identity_includes_player_and_void_status_survives_update_versions(tmp_path):
    tables = {
        TX: _write(tmp_path, TX, [_tx("same-id", "19", 10, "CASH_OUT"),
                                _tx("same-id", "19", 100, "VOID_BET", player="P2")]),
        "WOK_Bet": _write(tmp_path, "WOK_Bet", [
            _bet("B1", "one"), _bet("old", "two"), _bet("new", "two", "BET_CANCELLED")]),
        "WOK_Bet_Transaction": _write(tmp_path, "WOK_Bet_Transaction", [
            _bet_ref("B1", "same-id"), _bet_ref("old", "same-id", "P2"),
            _bet_ref("new", "stake", "P2")]),
    }
    result = fes.f48_percentage_bets_with_cashout(tables, chunksize=1)
    assert _values(result, CASHOUT)["P1"] == 1
    assert pd.isna(_values(result, CASHOUT)["P2"])


@pytest.mark.parametrize("status", ["BET_CANCELLED", " bet_cancelled "])
def test_cancelled_update_without_new_player_reference_still_excludes_bet(tmp_path, status):
    tables = {
        "WOK_Bet": _write(tmp_path, "WOK_Bet", [_bet("old", "B1", "BET_PLACED"),
                                                   _bet("new", "B1", status)]),
        "WOK_Bet_Transaction": _write(tmp_path, "WOK_Bet_Transaction", [_bet_ref("old", "stake")]),
    }
    result = fes.f48_percentage_bets_with_cashout(tables, chunksize=1)
    assert pd.isna(_values(result, CASHOUT)["P1"])


@pytest.mark.parametrize("status", ["CANCELLED", "CANCELED", "VOID", "VOIDED",
                                    "BET_CANCELED", "BET_VOID", "BET_VOIDED"])
def test_undocumented_status_does_not_establish_cancellation(tmp_path, status):
    tables = {
        TX: _write(tmp_path, TX, [_tx("cash", "10", 10, "CASH_OUT")]),
        "WOK_Bet": _write(tmp_path, "WOK_Bet", [_bet("B1", "cashout-bet", status)]),
        "WOK_Bet_Transaction": _write(tmp_path, "WOK_Bet_Transaction", [_bet_ref("B1", "cash")]),
    }
    result = fes.f48_percentage_bets_with_cashout(tables, chunksize=1)
    assert _values(result, CASHOUT)["P1"] == 1


@pytest.mark.parametrize("kind,status,expected", [
    ("VOID_BET", "SUCCESSFUL", 7200), ("VOID_STAKE", "SUCCESSFUL", 7200),
    ("WINNING", "SUCCESSFUL", None), ("CASH_OUT", "SUCCESSFUL", None),
    ("WINNING", "FAILED", 7200), ("CASH_OUT", "FAILED", 7200),
])
def test_f51_void_interval_counts_but_successful_wins_and_cashouts_exclude_it(tmp_path, kind, status, expected):
    tables = {TX: _write(tmp_path, TX, [_tx("next", "12", -100),
                                     _tx("refund", "10", 100, kind, status=status),
                                     _tx("first", "09", -100),
                                     _tx("failed-stake", "11", -100, status="FAILED")])}
    first = {**_bet("B1", "first", "BET_CANCELLED" if kind == "VOID_BET" else "BET_SETTLED"),
             "Extraction_Date": "2026-01-10T10:00:00Z"}
    next_bet = {**_bet("B2", "next", "BET_PLACED"), "Bet_Start_Datetime": "2026-01-10T12:00:00Z",
                "Extraction_Date": "2026-01-10T12:00:00Z"}
    tables["WOK_Bet"] = _write(tmp_path, "WOK_Bet", [first, next_bet])
    tables["WOK_Bet_Transaction"] = _write(tmp_path, "WOK_Bet_Transaction", [
        _bet_ref("B1", "first"), _bet_ref("B1", "refund"), _bet_ref("B2", "next")])
    result = fes.f51_median_seconds_loss_to_next_bet(tables, x_tijdspad=WINDOW, chunksize=1)
    value = _values(result, REBETS)["P1"]
    assert pd.isna(value) if expected is None else value == expected


def test_pipeline_discovers_refund_links_shares_one_scan_and_preserves_unknowns(tmp_path, monkeypatch):
    tmp_path = tmp_path / "cleaned_20260101_000000"
    tmp_path.mkdir()
    _write(tmp_path, TX, [_tx("morning", "09", -100), _tx("evening", "18", -100),
                        _tx("refund", "20", 100, "VOID_BET"),
                        _tx("stake", "09", -100, player="P2"),
                        _tx("unmatched", "20", 100, "VOID_BET", player="P2")])
    _write(tmp_path, "WOK_Bet_Transaction", [_bet_ref("B1", "morning"), _bet_ref("B1", "refund")])
    monkeypatch.setitem(pipeline.SCENARIOS, "void_test", {"features": [MORNING, EVENING]})
    with patch.object(pipeline, "net_stake_time_shares", wraps=net_stake_time_shares) as reconstruct:
        for run in range(2):
            result = pipeline.run_scenario("void_test", tmp_path, x_tijdspad=":".join(WINDOW), chunksize=1)
            assert reconstruct.call_count == run + 1
            assert _values(result, MORNING)["P1"] == 0
            assert _values(result, EVENING)["P1"] == 1
            assert pd.isna(_values(result, MORNING)["P2"])
            assert pd.isna(_values(result, EVENING)["P2"])


@pytest.mark.parametrize("column", [MORNING, EVENING, CASHOUT, REBETS])
def test_pipeline_preserves_undefined_void_feature_instead_of_replacing_with_zero(column):
    result = pipeline._safe_merge(None, pd.DataFrame({"Player_Profile_ID": ["P1"], column: [float("nan")]}))
    assert pd.isna(_values(result, column)["P1"])
