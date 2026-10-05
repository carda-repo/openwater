"""Feature inputs use the KSA transaction-type and game-type-limit fields."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3_features"))
from feature_engineering_basis import var4a_tot_var4f
from reading_difficult_json import iter_limit_values


TX_TABLE = "WOK_Player_Account_Transaction"
TYPES = ["DEPOSIT", "WITHDRAWAL", "STAKE", "WINNING", "OTHER", "BONUS"]
COLUMNS = [f"var4{letter}_totaal_aantal_{kind.lower()}_transacties"
           for letter, kind in zip("abcdef", TYPES)]


def _transaction(player, kind, status="SUCCESSFUL", timestamp="2026-01-10T10:00:00Z"):
    return {"Player_Profile_ID": player, "Transaction_Datetime": timestamp,
            "Transaction_Type": kind, "Transaction_Status": status}


@pytest.mark.parametrize("chunksize", [1, 100])
@pytest.mark.parametrize("sep,snake", [(",", False), (";", True)])
@pytest.mark.parametrize("use_window", [True, False])
def test_successful_bonus_type_counts_with_other_types_and_period_filters(
        tmp_path, chunksize, sep, snake, use_window):
    rows = [_transaction("all-types", kind) for kind in TYPES]
    rows += [_transaction("all-types", "DEPOSIT")]
    rows += [_transaction("all-types", kind, "UNSUCCESSFUL") for kind in TYPES]
    rows += [_transaction("bonus-only", "BONUS"), _transaction("bonus-only", "BONUS")]
    rows += [_transaction("bonus-only", "BONUS", timestamp="2025-12-31T23:59:59Z"),
             _transaction("bonus-only", "BONUS", timestamp="2026-02-01T00:00:00Z")]
    # BONUS is never a valid transaction status, even when its type is BONUS.
    rows += [_transaction("invalid-status", "BONUS", "BONUS"),
             _transaction("invalid-status", "STAKE", "BONUS"),
             _transaction("failed-only", "BONUS", "UNSUCCESSFUL")]
    frame = pd.DataFrame(rows)
    if snake:
        frame.columns = frame.columns.str.lower()
    path = tmp_path / f"{TX_TABLE}.csv"
    frame.to_csv(path, sep=sep, index=False)

    result = var4a_tot_var4f(
        {TX_TABLE: [path]}, chunksize=chunksize,
        x_tijdspad=["01012026", "01022026"] if use_window else None,
    ).set_index("Player_Profile_ID")

    assert set(result.index) == {"all-types", "bonus-only"}
    assert result.loc["all-types", COLUMNS].tolist() == [2, 1, 1, 1, 1, 1]
    assert result.loc["bonus-only", COLUMNS].tolist() == [0, 0, 0, 0, 0, 2 if use_window else 4]


@pytest.mark.parametrize("representation", ["dict", "list", "json-dict", "json-list"])
def test_game_type_limit_reads_free_text_and_preserves_request_time_and_window(representation):
    record = {"Game_Type_Request_Datetime": "2026-01-10T08:30:00Z",
              "Game_Type_Start_Datetime": "2026-01-11T12:00:00Z",
              "Game_Type_Type": "  Roulette en live casino  ",
              "Game_Type": "wrong-field-value",
              "Game_Type_Time_Window": "WEEK"}
    payload = [record] if representation.endswith("list") else record
    if representation.startswith("json"):
        payload = json.dumps(payload)

    assert list(iter_limit_values(payload, type_="game_type")) == [
        (datetime(2026, 1, 10, 8, 30, tzinfo=timezone.utc), "Roulette en live casino", "WEEK")
    ]


def test_game_type_limit_start_time_fallback_and_no_old_field_fallback():
    records = [{"Game_Type_Request_Datetime": "", "Game_Type_Start_Datetime": "2026-01-11T12:00:00Z",
                "Game_Type_Type": "  Nieuw speltype voor toekomstige ontwikkelingen  "},
               {"Game_Type": "SLOT", "Game_Type_Time_Window": "DAY"}]
    assert list(iter_limit_values(records, type_="game_type")) == [
        (datetime(2026, 1, 11, 12, tzinfo=timezone.utc),
         "Nieuw speltype voor toekomstige ontwikkelingen", None),
        (None, None, "DAY"),
    ]
