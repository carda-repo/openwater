"""F46 only interprets documented completed KSA Bet_Status values as resolved."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3_features"))
from feature_engineering_spanish import f46_median_seconds_bet_placed_to_resolved


FEATURE = "f46_median_seconds_bet_placed_to_resolved"


@pytest.mark.parametrize("chunksize", [1, 100])
@pytest.mark.parametrize("sep,snake", [(",", False), (";", True)])
def test_only_documented_completed_statuses_contribute_to_median(tmp_path, chunksize, sep, snake):
    statuses = ["BET_SETTLED", "BET_CANCELLED", " bet_settled ",
                "BET_PLACED", "BET_UPDATED", "OTHER",
                "SETTLED", "RESOLVED", "CLOSED", "CANCELLED", "VOID", "WON", "LOST",
                "BET_RESOLVED", "BET_CLOSED", "BET_VOID", "BET_WON", "BET_LOST", "UNSETTLED"]
    bets = pd.DataFrame([
        {"pk_id": i, "Bet_Start_Datetime": "2026-01-10T10:00:00Z", "Bet_Status": status,
         "Extraction_Date": f"2026-01-10T{11 + min(i, 10):02d}:00:00Z"}
        for i, status in enumerate(statuses)])
    refs = pd.DataFrame([{"wok_bet_pk_id": i, "player_profile_id": "P1"} for i in range(len(statuses))])
    paths = {}
    for table, frame in [("WOK_Bet", bets), ("WOK_Bet_Transaction", refs)]:
        if snake:
            frame.columns = frame.columns.str.lower()
        path = tmp_path / f"{table}.csv"
        frame.to_csv(path, index=False, sep=sep)
        paths[table] = [path]
    result = f46_median_seconds_bet_placed_to_resolved(
        paths, x_tijdspad=["01012026", "01022026"], chunksize=chunksize)
    # Valid completed reports give 1h, 2h and 3h. Unsupported statuses add no durations.
    assert result.set_index("Player_Profile_ID")[FEATURE].to_dict() == {"P1": 7200}


def test_noncompleted_bets_remain_unknown(tmp_path):
    path = tmp_path / "WOK_Bet.csv"
    pd.DataFrame([{"pk_id": "B1", "Bet_Start_Datetime": "2026-01-10T10:00:00Z",
                   "Bet_Status": "OTHER", "Extraction_Date": "2026-01-10T12:00:00Z"}]).to_csv(path, index=False)
    ref = tmp_path / "WOK_Bet_Transaction.csv"
    pd.DataFrame([{"wok_bet_pk_id": "B1", "player_profile_id": "P1"}]).to_csv(ref, index=False)
    result = f46_median_seconds_bet_placed_to_resolved(
        {"WOK_Bet": [path], "WOK_Bet_Transaction": [ref]}, x_tijdspad=["01012026", "01022026"])
    assert pd.isna(result.set_index("Player_Profile_ID").loc["P1", FEATURE])
