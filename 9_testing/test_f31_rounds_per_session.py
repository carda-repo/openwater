"""F31 measures session rounds, independently of transaction aggregation."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3_features"))
from feature_engineering_spanish import f31_median_rounds_per_session
from parse_pipeline import build_features


FEATURE = "f31_median_rounds_per_session"
SESSION = "WOK_Game_Session"
REFERENCE = "WOK_Game_Session_Transaction"


def _session(pk, rounds, timestamp="2026-01-10T12:00:00Z"):
    return {"pk_id": pk, "Game_Session_Start_Datetime": timestamp,
            "Game_Session_Rounds": rounds}


def _reference(pk, player, transaction):
    return {"wok_game_session_pk_id": pk, "player_profile_id": player,
            "transaction_id": transaction}


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
def test_rounds_median_is_independent_of_transaction_counts_across_files(
    tmp_path, chunksize, sep, snake_case
):
    first = _write(tmp_path / "sessions_1.csv", [_session("S1", 100)], sep, snake_case)
    second = _write(tmp_path / "sessions_2.csv", [
        _session("S2", 10), _session("S3", 20),
    ], sep, snake_case)
    references = _write(tmp_path / "references.csv", [
        _reference("S1", "P1", "stake-total"),
        _reference("S1", "P1", "winning-total"),
        _reference("S1", "P2", "other-player"),
        _reference("S2", "P1", "one-transaction"),
        *[_reference("S3", "P1", f"transaction-{i}") for i in range(5)],
        _reference("S3", None, "missing-player"),
        _reference("S3", " ", "blank-player"),
    ], sep, snake_case)
    result = f31_median_rounds_per_session(
        {SESSION: [first, second], REFERENCE: [references]},
        x_tijdspad=["01012026", "31012026"], chunksize=chunksize,
    )
    # P1 has 100, 10 and 20 rounds: median 20, rather than transaction median 2.
    assert _values(result) == {"P1": 20.0, "P2": 100.0}


def test_even_number_of_sessions_uses_mathematical_median(tmp_path):
    sessions = _write(tmp_path / "sessions.csv", [_session("S1", 10), _session("S2", 31)])
    refs = _write(tmp_path / "refs.csv", [
        _reference("S1", "P1", "T1"), _reference("S2", "P1", "T2"),
    ])
    result = f31_median_rounds_per_session({SESSION: [sessions], REFERENCE: [refs]}, chunksize=1)
    assert _values(result) == {"P1": 20.5}


def test_invalid_round_counts_are_not_replaced_with_transaction_counts(tmp_path):
    invalid = [None, "invalid", -5, 0, 1.5, float("inf")]
    sessions = _write(tmp_path / "sessions.csv", [
        _session("valid", 20),
        *[_session(f"invalid-{i}", value) for i, value in enumerate(invalid)],
        _session("unknown", None),
    ])
    refs = _write(tmp_path / "refs.csv", [
        _reference("valid", "P1", "valid-transaction"),
        *[_reference(f"invalid-{i}", "P1", f"T{i}") for i in range(len(invalid))],
        _reference("unknown", "P2", "unknown-transaction"),
    ])
    result = f31_median_rounds_per_session({SESSION: [sessions], REFERENCE: [refs]}, chunksize=1)
    values = _values(result)
    assert values["P1"] == 20
    assert pd.isna(values["P2"])


def test_session_time_filter_is_applied_before_rounds_are_aggregated(tmp_path):
    sessions = _write(tmp_path / "sessions.csv", [
        _session("before", 100, "2025-12-31T12:00:00Z"),
        _session("inside", 30, "2026-01-15T12:00:00Z"),
        _session("after", 200, "2026-02-01T12:00:00Z"),
    ])
    refs = _write(tmp_path / "refs.csv", [
        _reference(pk, "P1", pk) for pk in ("before", "inside", "after")
    ])
    result = f31_median_rounds_per_session(
        {SESSION: [sessions], REFERENCE: [refs]}, x_tijdspad=["01012026", "31012026"],
    )
    assert _values(result) == {"P1": 30.0}


def test_missing_rounds_column_is_reported_instead_of_counting_transactions(tmp_path):
    session = _session("S1", 100)
    del session["Game_Session_Rounds"]
    sessions = _write(tmp_path / "sessions.csv", [session])
    refs = _write(tmp_path / "refs.csv", [_reference("S1", "P1", "T1")])
    with pytest.raises(ValueError, match="Game_Session_Rounds"):
        f31_median_rounds_per_session({SESSION: [sessions], REFERENCE: [refs]})


@pytest.mark.parametrize("missing_table", [SESSION, REFERENCE])
def test_missing_sessions_or_player_links_returns_empty_result(tmp_path, missing_table):
    sessions = _write(tmp_path / "sessions.csv", [_session("S1", 100)])
    refs = _write(tmp_path / "refs.csv", [_reference("S1", "P1", "T1")])
    tables = {SESSION: [sessions], REFERENCE: [refs]}
    del tables[missing_table]
    assert _values(f31_median_rounds_per_session(tables)) == {}


def test_feature_pipeline_reads_rounds_column_and_writes_correct_value(tmp_path):
    cleaned = tmp_path / "cleaned_20260101_000000"
    cleaned.mkdir()
    _write(cleaned / f"{SESSION}.csv", [_session("S1", 100)], sep=";", snake_case=True)
    _write(cleaned / f"{REFERENCE}.csv", [
        _reference("S1", "P1", "stake-total"), _reference("S1", "P1", "winning-total"),
    ], sep=";", snake_case=True)
    output = tmp_path / "features.csv"
    result = build_features(
        cleaned_dir=cleaned, scenario="Flexible_spanish_plus", feature=FEATURE,
        x_tijdspad="01012026:31012026", features_out=output, chunksize=1, verbose=False,
    )
    assert _values(result) == {"P1": 100.0}
    assert _values(pd.read_csv(output)) == {"P1": 100.0}
