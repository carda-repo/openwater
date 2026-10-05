"""F51 follows a particular resolved bet, not consecutive account transactions."""

from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3_features"))
import bet_loss_intervals as bli
import feature_engineering_spanish as fes
import parse_pipeline as pipeline

FEATURE = "f51_median_seconds_loss_to_next_bet"
WINDOW = ["01012026", "01022026"]
BET, REF, TX = "WOK_Bet", "WOK_Bet_Transaction", "WOK_Player_Account_Transaction"


def _time(hm, date="2026-01-10"):
    return f"{date}T{hm}:00Z"


def _bet(pk, identity, placed="19:00", extracted="21:00", status="BET_SETTLED", date="2026-01-10", operator=None):
    return {"pk_id": pk, "Bet_ID": identity, "Bet_Start_Datetime": _time(placed, date),
            "Bet_Status": status, "Extraction_Date": _time(extracted, date) if extracted else None,
            "Operator_ID": operator}


def _tx(txid, hm="19:00", kind="STAKE", amount=-10, player="P1", status="SUCCESSFUL", date="2026-01-10", operator=None):
    return {"Player_Profile_ID": player, "Transaction_ID": txid, "Transaction_Datetime": _time(hm, date),
            "Transaction_Type": kind, "Transaction_Amount": amount, "Transaction_Status": status,
            "Operator_ID": operator}


def _ref(pk, txid, player="P1"):
    return {"wok_bet_pk_id": pk, "player_profile_id": player, "transactions_id": txid}


def _write(root, name, rows, sep=",", snake=False):
    df = pd.DataFrame(rows)
    if snake:
        df.columns = df.columns.str.lower()
    path = root / f"{name}.csv"
    df.to_csv(path, sep=sep, decimal="," if sep == ";" else ".", index=False)
    return path


def _tables(root, bets=None, refs=None, transactions=None):
    bets = bets if bets is not None else [
        _bet("A", "first"), _bet("C", "next", "21:10", "21:10", "BET_PLACED")]
    refs = refs if refs is not None else [_ref("A", "stake-A"), _ref("C", "stake-C")]
    transactions = transactions if transactions is not None else [_tx("stake-A"), _tx("stake-C", "21:10")]
    return {BET: [_write(root, BET, bets)], REF: [_write(root, REF, refs)], TX: [_write(root, TX, transactions)]}


def _values(tables, chunksize=1, window=WINDOW):
    df = fes.f51_median_seconds_loss_to_next_bet(tables, x_tijdspad=window, chunksize=chunksize)
    assert list(df.columns) == ["Player_Profile_ID", FEATURE]
    assert df["Player_Profile_ID"].is_unique
    return df.set_index("Player_Profile_ID")[FEATURE]


@pytest.mark.parametrize("chunksize", [1, 2, 100])
@pytest.mark.parametrize("sep,snake", [(",", False), (";", True)])
def test_overlapping_bets_use_first_settlement_across_update_files_and_own_prizes(tmp_path, chunksize, sep, snake):
    # Latest version is deliberately read first; repeated A is one logical bet.
    recent = _write(tmp_path, "WOK_Bet_2", [_bet("a-later", "A", extracted="23:00"),
                                            _bet("c", "C", "21:10", "21:10", "BET_PLACED")], sep, snake)
    history = _write(tmp_path, "WOK_Bet_1", [
        _bet("a-placed", "A", extracted="19:00", status="BET_PLACED"),
        _bet("a-settled", "A"), _bet("b", "B", "19:05", "20:50")], sep, snake)
    refs = _write(tmp_path, REF, [_ref("a-placed", "stake-A"), _ref("a-settled", "zero-A"),
                                 _ref("a-later", "zero-A"), _ref("b", "stake-B"), _ref("b", "win-B"),
                                 _ref("c", "stake-C")], sep, snake)
    transactions = _write(tmp_path, TX, [_tx("stake-C", "21:10"),
                                       _tx("win-B", "20:50", "WINNING", 1.25),
                                       _tx("zero-A", "20:59", "WINNING", 0),
                                       _tx("casino-stake", "21:05"), _tx("stake-A"),
                                       _tx("stake-B", "19:05")], sep, snake)
    # B before A's result does not establish a loss; B's prize does not mask A's loss.
    assert _values({BET: [recent, history], REF: [refs], TX: [transactions]}, chunksize)["P1"] == 600


@pytest.mark.parametrize("status", ["BET_PLACED", "BET_UPDATED", "BET_CANCELLED", "BET_WON", "OTHER"])
def test_open_updated_or_cancelled_without_refund_is_not_a_loss(tmp_path, status):
    tables = _tables(tmp_path, bets=[_bet("A", "first", status=status),
                                    _bet("C", "next", "21:10", "21:10", "BET_PLACED")])
    assert pd.isna(_values(tables)["P1"])


def test_explicit_lost_status_in_relational_export_is_supported(tmp_path):
    tables = _tables(tmp_path, bets=[_bet("A", "first", status="BET_LOST"),
                                    _bet("C", "next", "21:10", "21:10", "BET_PLACED")])
    assert _values(tables)["P1"] == 600


def test_explicit_won_status_with_missing_prize_does_not_become_a_void_loss(tmp_path):
    tables = _tables(tmp_path, bets=[_bet("A", "first", status="BET_WON"),
                                    _bet("C", "next", "21:10", "21:10", "BET_PLACED")],
                     refs=[_ref("A", "stake-A"), _ref("A", "refund"), _ref("C", "stake-C")],
                     transactions=[_tx("stake-A"), _tx("refund", "20:55", "VOID_BET", 10), _tx("stake-C", "21:10")])
    assert pd.isna(_values(tables)["P1"])


@pytest.mark.parametrize("failed_tx", ["stake-A", "stake-C"])
def test_unsuccessful_stake_does_not_establish_loss_or_next_placement(tmp_path, failed_tx):
    transactions = [_tx("stake-A"), _tx("stake-C", "21:10")]
    for transaction in transactions:
        if transaction["Transaction_ID"] == failed_tx:
            transaction["Transaction_Status"] = "UNSUCCESSFUL"
    assert pd.isna(_values(_tables(tmp_path, transactions=transactions))["P1"])


def test_no_settlement_time_is_not_replaced_with_stake_time(tmp_path):
    tables = _tables(tmp_path, bets=[_bet("A", "first", extracted=None),
                                    _bet("C", "next", "21:10", "21:10", "BET_PLACED")])
    assert pd.isna(_values(tables)["P1"])


def test_median_of_multiple_losses_uses_next_bet_after_each_closure(tmp_path):
    tables = _tables(tmp_path, bets=[_bet("A", "first", "09:00", "10:00"),
                                    _bet("B", "second", "10:10", "11:00"),
                                    _bet("C", "next", "11:30", "11:30", "BET_PLACED")],
                     refs=[_ref(pk, pk) for pk in "ABC"],
                     transactions=[_tx("A", "09:00"), _tx("B", "10:10"), _tx("C", "11:30")])
    assert _values(tables)["P1"] == 1200  # Median of 600 and 1800 seconds.


@pytest.mark.parametrize("kind,amount", [("WINNING", .01), ("CASH_OUT", 1)])
def test_own_prize_or_cashout_excludes_bet_even_with_later_void(tmp_path, kind, amount):
    tables = _tables(tmp_path, refs=[_ref("A", "stake-A"), _ref("A", "result"), _ref("A", "refund"),
                                    _ref("C", "stake-C")],
                     transactions=[_tx("stake-A"), _tx("result", "20:00", kind, amount),
                                   _tx("refund", "21:00", "VOID_BET", 10), _tx("stake-C", "21:10")])
    assert pd.isna(_values(tables)["P1"])


def test_void_counts_from_refund_time_without_settled_status(tmp_path):
    tables = _tables(tmp_path, bets=[_bet("A", "first", extracted="21:00", status="BET_CANCELLED"),
                                    _bet("C", "next", "21:10", "21:10", "BET_PLACED")],
                     refs=[_ref("A", "stake-A"), _ref("A", "refund"), _ref("C", "stake-C")],
                     transactions=[_tx("stake-A"), _tx("refund", "20:55", "VOID_BET", 10), _tx("stake-C", "21:10")])
    assert _values(tables)["P1"] == 900


def test_pre_window_stake_can_lose_within_window_and_earlier_prize_is_not_ignored(tmp_path):
    bets = [_bet("A", "first", date="2025-12-31", extracted="19:00", status="BET_PLACED"),
            _bet("a-settled", "first"), _bet("C", "next", "21:10", "21:10", "BET_PLACED")]
    refs = [_ref("A", "stake-A"), _ref("a-settled", "prize"), _ref("C", "stake-C")]
    transactions = [_tx("stake-A", date="2025-12-31"), _tx("stake-C", "21:10")]
    tables = _tables(tmp_path, bets, refs, transactions)
    assert _values(tables)["P1"] == 600
    tables[TX] = [_write(tmp_path, TX, transactions + [_tx("prize", "20:00", "WINNING", 1, date="2025-12-31")])]
    assert pd.isna(_values(tables)["P1"])


def test_future_settlement_does_not_turn_an_open_bet_into_a_loss(tmp_path):
    tables = _tables(tmp_path, bets=[_bet("A", "first", status="BET_PLACED", extracted="19:00"),
                                    _bet("a-future", "first", date="2026-02-01", extracted="21:00"),
                                    _bet("C", "next", "21:10", "21:10", "BET_PLACED")])
    assert pd.isna(_values(tables)["P1"])


@pytest.mark.parametrize("kind", ["WINNING", "VOID_BET", "CASH_OUT"])
def test_transactions_at_exclusive_end_do_not_change_prior_outcome(tmp_path, kind):
    tables = _tables(tmp_path, refs=[_ref("A", "stake-A"), _ref("A", "future"), _ref("C", "stake-C")],
                     transactions=[_tx("stake-A"), _tx("future", "00:00", kind, 10, date="2026-02-01"),
                                   _tx("stake-C", "21:10")])
    assert _values(tables)["P1"] == 600


@pytest.mark.parametrize("next_time,next_date", [("21:00", "2026-01-10"), ("00:00", "2026-02-01")])
def test_no_strictly_later_placement_within_window_produces_nan(tmp_path, next_time, next_date):
    tables = _tables(tmp_path, bets=[_bet("A", "first"),
                                    _bet("C", "next", next_time, next_time, "BET_PLACED", next_date)],
                     transactions=[_tx("stake-A"), _tx("stake-C", next_time, date=next_date)])
    assert pd.isna(_values(tables)["P1"])


def test_operator_and_player_transaction_identities_are_preserved(tmp_path):
    bets, refs, transactions = [], [], []
    for operator, closing, next_time in [("O1", "21:00", "21:10"), ("O2", "20:00", "20:20")]:
        bets += [_bet(f"{operator}-A", "same-bet-id", extracted=closing, operator=operator),
                 _bet(f"{operator}-C", "same-next-id", next_time, next_time, "BET_PLACED", operator=operator)]
        refs += [_ref(f"{operator}-A", "same-stake-id"), _ref(f"{operator}-C", "same-next-stake")]
        transactions += [_tx("same-stake-id", operator=operator), _tx("same-next-stake", next_time, operator=operator)]
    tables = _tables(tmp_path, bets, refs, transactions)
    assert _values(tables)["P1"] == 900
    # Without an operator the overlapping identities cannot safely be assigned.
    for transaction in transactions:
        transaction["Operator_ID"] = None
    tables[TX] = [_write(tmp_path, TX, transactions)]
    assert pd.isna(_values(tables)["P1"])


@pytest.mark.parametrize("invalid", [None, "invalid", -1])
def test_unknown_prize_amount_does_not_prove_no_prize(tmp_path, invalid):
    tables = _tables(tmp_path, refs=[_ref("A", "stake-A"), _ref("A", "prize"), _ref("C", "stake-C")],
                     transactions=[_tx("stake-A"), _tx("prize", "21:00", "WINNING", invalid), _tx("stake-C", "21:10")])
    assert pd.isna(_values(tables)["P1"])


def test_missing_parent_preserves_player_as_unknown(tmp_path):
    tables = _tables(tmp_path, refs=[_ref("missing", "stake-A")])
    assert pd.isna(_values(tables)["P1"])


def test_pipeline_declares_joins_scans_each_table_once_and_saves_unknowns(tmp_path, monkeypatch):
    cleaned = tmp_path / "cleaned_20260101_000000"
    cleaned.mkdir()
    _tables(cleaned, bets=[_bet("A", "first"), _bet("C", "next", "21:10", "21:10", "BET_PLACED"),
                          _bet("P2", "open", extracted="19:00", status="BET_PLACED")],
            refs=[_ref("A", "stake-A"), _ref("C", "stake-C"), _ref("P2", "stake-P2", "P2")],
            transactions=[_tx("stake-A"), _tx("stake-C", "21:10"), _tx("stake-P2", player="P2")])
    monkeypatch.setitem(pipeline.SCENARIOS, "bet_resolution_test", {"features": [FEATURE]})
    output = tmp_path / "features.csv"
    with patch.object(bli, "iter_csv_chunks", wraps=bli.iter_csv_chunks) as reader:
        pipeline.build_features(cleaned, "bet_resolution_test", features_out=output,
                                x_tijdspad=":".join(WINDOW), chunksize=1)
        assert reader.call_count == 3
    result = pd.read_csv(output).set_index("Player_Profile_ID")[FEATURE]
    assert result["P1"] == 600
    assert pd.isna(result["P2"])
    log = (cleaned / "logs" / f"{FEATURE}.log").read_text()
    assert "settlement-proxy intervals" in log
