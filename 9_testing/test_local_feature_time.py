"""Dutch play-time labels; UTC chronology, periods and DST durations remain intact."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3_features"))
import feature_engineering_spanish as fes
import feature_engineering_basis as feb
import parse_pipeline as pipeline
from local_time import local_time, local_day_labels

TX = "WOK_Player_Account_Transaction"
MORNING = "f44_morning_stakes_percentage"
EVENING = "f45_evening_stakes_percentage"
WINDOW = ["01012025", "01012026"]


def _tx(txid, timestamp, amount=-10, player="P1", kind="STAKE"):
    return {"Player_Profile_ID": player, "Transaction_ID": txid, "Transaction_Datetime": timestamp,
            "Transaction_Type": kind, "Transaction_Status": "SUCCESSFUL", "Transaction_Amount": amount}


def _write(root, table, rows, sep=",", snake=False):
    frame = pd.DataFrame(rows)
    if snake:
        frame.columns = frame.columns.str.lower()
    path = root / f"{table}.csv"
    frame.to_csv(path, sep=sep, decimal="," if sep == ";" else ".", index=False)
    return [path]


def _values(result, column):
    return result.set_index("Player_Profile_ID")[column]


@pytest.mark.parametrize("naive", [True, False])
def test_timezone_conversion_handles_both_dst_transitions_without_ambiguous_day_labels(naive):
    utc = pd.to_datetime(pd.Series(["2025-03-30T00:30:00Z", "2025-03-30T01:30:00Z",
                                    "2025-10-26T00:30:00Z", "2025-10-26T01:30:00Z"]), utc=True)
    if naive:
        utc = utc.dt.tz_localize(None)
    converted = local_time(utc)
    assert converted.dt.hour.tolist() == [1, 3, 2, 2]
    assert converted.dt.strftime("%z").tolist() == ["+0100", "+0200", "+0200", "+0100"]
    assert local_day_labels(utc).astype(str).tolist() == ["2025-03-30"] * 2 + ["2025-10-26"] * 2
    assert (converted.iloc[1] - converted.iloc[0]).total_seconds() == 3600
    assert (converted.iloc[3] - converted.iloc[2]).total_seconds() == 3600


@pytest.mark.parametrize("chunksize", [1, 100])
@pytest.mark.parametrize("date,hours", [("2025-01-02", ["07:30", "15:30", "23:30"]),
                                       ("2025-06-02", ["06:30", "14:30", "22:30"])])
@pytest.mark.parametrize("sep,snake", [(",", False), (";", True)])
def test_stake_shares_use_dutch_morning_evening_and_midnight(tmp_path, chunksize, date, hours, sep, snake):
    rows = [_tx(str(i), f"{date}T{hour}:00Z", amount) for i, (hour, amount) in enumerate(zip(hours, [-30, -20, -10]))]
    tables = {TX: _write(tmp_path, TX, rows, sep, snake)}
    assert _values(fes.f44_morning_stakes_percentage(tables, chunksize=chunksize, x_tijdspad=WINDOW), MORNING)["P1"] == .5
    assert _values(fes.f45_evening_stakes_percentage(tables, chunksize=chunksize, x_tijdspad=WINDOW), EVENING)["P1"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("scope", ["bet", "session"])
@pytest.mark.parametrize("date,hours", [("2025-01-02", ["07:30", "15:30", "23:30"]),
                                       ("2025-06-02", ["06:30", "14:30", "22:30"])])
def test_interaction_shares_use_local_clock_for_bets_and_sessions(tmp_path, scope, date, hours):
    rows = [_tx(str(i), f"{date}T{hour}:00Z", player=player)
            for i, (hour, player) in enumerate(zip(hours, ["morning", "evening", "night"]))]
    tables = {TX: _write(tmp_path, TX, rows)}
    if scope == "bet":
        tables["WOK_Bet"] = _write(tmp_path, "WOK_Bet", [
            {"pk_id": row["Transaction_ID"], "Bet_Start_Datetime": row["Transaction_Datetime"]} for row in rows])
        tables["WOK_Bet_Transaction"] = _write(tmp_path, "WOK_Bet_Transaction", [
            {"wok_bet_pk_id": row["Transaction_ID"], "transactions_id": row["Transaction_ID"],
             "player_profile_id": row["Player_Profile_ID"]} for row in rows])
    else:
        tables["WOK_Game_Session"] = _write(tmp_path, "WOK_Game_Session", [
            {"pk_id": row["Transaction_ID"], "Game_Session_Start_Datetime": row["Transaction_Datetime"]} for row in rows])
        tables["WOK_Game_Session_Transaction"] = _write(tmp_path, "WOK_Game_Session_Transaction", [
            {"wok_game_session_pk_id": row["Transaction_ID"], "transaction_id": row["Transaction_ID"],
             "player_profile_id": row["Player_Profile_ID"]} for row in rows])
    morning = fes.f42_morning_interaction_percentage(tables, chunksize=1, x_tijdspad=WINDOW)
    evening = fes.f43_evening_interaction_percentage(tables, chunksize=1, x_tijdspad=WINDOW)
    assert _values(morning, "f42_morning_interaction_percentage").to_dict() == {"morning": 1, "evening": 0, "night": 0}
    assert _values(evening, "f43_evening_interaction_percentage").to_dict() == {"morning": 0, "evening": 1, "night": 0}


def test_dominant_clock_hours_combine_same_local_hour_in_winter_and_summer(tmp_path):
    rows = [_tx("winter", "2025-01-02T07:30:00Z"), _tx("summer", "2025-06-02T06:30:00Z")]
    tables = {
        TX: _write(tmp_path, TX, rows),
        "WOK_Bet": _write(tmp_path, "WOK_Bet", [
            {"pk_id": row["Transaction_ID"], "Bet_Start_Datetime": row["Transaction_Datetime"]} for row in rows]),
        "WOK_Bet_Transaction": _write(tmp_path, "WOK_Bet_Transaction", [
            {"wok_bet_pk_id": row["Transaction_ID"], "transactions_id": row["Transaction_ID"], "player_profile_id": "P1"}
            for row in rows]),
    }
    result = fes.f41_heavy_play_hours_count(tables, chunksize=1, x_tijdspad=WINDOW)
    assert _values(result, "f41_heavy_play_hours_count")["P1"] == 1


@pytest.mark.parametrize("chunksize", [1, 100])
def test_local_midnight_changes_active_days_and_daily_amounts(tmp_path, chunksize):
    tables = {TX: _write(tmp_path, TX, [_tx("first", "2025-06-01T22:30:00Z", 10),
                                      _tx("second", "2025-06-02T20:00:00Z", 20)])}
    cases = [(fes.f1_active_days, "f1_active_days", 1),
             (feb.var3a_totaal_aantal_actieve_dagen, "var3a_totaal_aantal_actieve_dagen", 1),
             (feb.var1b_variantie_ingezet_bedrag_per_dag, "var1b_variantie_ingezet_bedrag_per_dag", 0)]
    for fn, column, expected in cases:
        assert _values(fn(tables, chunksize=chunksize, x_tijdspad=WINDOW), column)["P1"] == expected
    maximum = feb.var20a_tot_var20b(tables, chunksize=chunksize, x_tijdspad=WINDOW)
    assert _values(maximum, "var20a_max_transacties_per_dag")["P1"] == 2
    assert _values(maximum, "var20b_max_inzet_per_dag")["P1"] == 30


def test_local_dates_change_streak_and_span_without_changing_timestamps(tmp_path):
    tables = {TX: _write(tmp_path, TX, [_tx("first", "2025-06-01T21:30:00Z"),
                                      _tx("second", "2025-06-02T22:30:00Z")])}
    for fn, column, expected in [(fes.f14_active_period_span, "f14_active_period_span", 3),
                                (fes.f57_longest_daily_streak, "f57_longest_daily_streak", 1),
                                (fes.f58_longest_streak_ratio, "f58_longest_streak_ratio", .5)]:
        assert _values(fn(tables, chunksize=1, x_tijdspad=WINDOW), column)["P1"] == expected


@pytest.mark.parametrize("chunksize", [1, 100])
def test_post_median_active_days_use_local_days_across_chunks(tmp_path, chunksize):
    tables = {TX: _write(tmp_path, TX, [_tx("first", "2025-06-01T22:30:00Z"),
                                      _tx("second", "2025-06-02T20:00:00Z"),
                                      _tx("third", "2025-06-03T10:00:00Z"),
                                      _tx("fourth", "2025-06-04T10:00:00Z")])}
    # First half: one Dutch day (June 2); second half: June 3 and June 4.
    result = fes.f54_post_median_active_days_percentage(tables, chunksize=chunksize, x_tijdspad=WINDOW)
    assert _values(result, "f54_post_median_active_days_percentage")["P1"] == pytest.approx(2 / 3)


def test_local_iso_weeks_and_weekend_night_activity(tmp_path):
    tables = {TX: _write(tmp_path, TX, [_tx("sun", "2025-06-08T22:30:00Z"),
                                      _tx("mon", "2025-06-09T12:00:00Z"), _tx("tue", "2025-06-10T12:00:00Z")])}
    week = feb.var3b_variantie_aantal_actieve_dagen_per_week(tables, chunksize=1, x_tijdspad=WINDOW)
    assert _values(week, "var3b_variantie_aantal_actieve_dagen_per_week")["P1"] == 0
    tables[TX] = _write(tmp_path, TX, [_tx("fri", "2025-06-06T22:30:00Z", player="late-friday"),
                                     _tx("sat", "2025-06-07T04:30:00Z", player="morning"),
                                     _tx("eve", "2025-06-06T20:30:00Z", player="evening")])
    flags = feb.var19a_tot_var19b(tables, chunksize=1, x_tijdspad=WINDOW)
    assert _values(flags, "var19a_totaal_gokken_weekend").to_dict() == {"late-friday": 1, "morning": 1, "evening": 0}
    assert _values(flags, "var19b_totaal_gokken_nacht").to_dict() == {"late-friday": 1, "morning": 0, "evening": 1}


@pytest.mark.parametrize("date", ["2025-03-30", "2025-10-26"])
def test_elapsed_daily_gap_and_f51_wait_are_utc_across_dst(tmp_path, date):
    rows = [_tx("first", f"{date}T00:30:00Z"), _tx("next", f"{date}T01:30:00Z")]
    tables = {TX: _write(tmp_path, TX, rows)}
    gap = fes.f59_median_daily_time_off(tables, chunksize=1, x_tijdspad=WINDOW)
    assert _values(gap, "f59_median_daily_time_off")["P1"] == 1
    tables["WOK_Bet"] = _write(tmp_path, "WOK_Bet", [
        {"pk_id": "A", "Bet_ID": "A", "Bet_Start_Datetime": rows[0]["Transaction_Datetime"],
         "Extraction_Date": rows[0]["Transaction_Datetime"], "Bet_Status": "BET_SETTLED"},
        {"pk_id": "B", "Bet_ID": "B", "Bet_Start_Datetime": rows[1]["Transaction_Datetime"],
         "Extraction_Date": rows[1]["Transaction_Datetime"], "Bet_Status": "BET_PLACED"}])
    tables["WOK_Bet_Transaction"] = _write(tmp_path, "WOK_Bet_Transaction", [
        {"wok_bet_pk_id": "A", "player_profile_id": "P1", "transactions_id": "first"},
        {"wok_bet_pk_id": "B", "player_profile_id": "P1", "transactions_id": "next"}])
    wait = fes.f51_median_seconds_loss_to_next_bet(tables, chunksize=1, x_tijdspad=WINDOW)
    assert _values(wait, "f51_median_seconds_loss_to_next_bet")["P1"] == 3600


def test_window_filter_keeps_utc_boundaries_and_does_not_reinterpret_local_midnight(tmp_path):
    tables = {TX: _write(tmp_path, TX, [_tx("outside", "2025-05-31T22:30:00Z", player="outside"),
                                      _tx("inside", "2025-06-01T06:30:00Z", player="inside")])}
    result = fes.f44_morning_stakes_percentage(tables, chunksize=1, x_tijdspad=["01062025", "01072025"])
    assert _values(result, MORNING).to_dict() == {"inside": 1}


def test_pipeline_discovers_no_extra_columns_and_saves_local_shares(tmp_path, monkeypatch):
    cleaned = tmp_path / "cleaned_20250101_000000"
    cleaned.mkdir()
    _write(cleaned, TX, [_tx("morning", "2025-06-01T06:30:00Z", -30),
                        _tx("evening", "2025-06-01T14:30:00Z", -20)], sep=";", snake=True)
    monkeypatch.setitem(pipeline.SCENARIOS, "local_time_test", {"features": [MORNING, EVENING]})
    output = tmp_path / "features.csv"
    pipeline.build_features(cleaned, "local_time_test", x_tijdspad=":".join(WINDOW),
                            chunksize=1, features_out=output)
    saved = pd.read_csv(output)
    assert _values(saved, MORNING)["P1"] == .6
    assert _values(saved, EVENING)["P1"] == .4
