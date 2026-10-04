"""Bet-parts: newest part per bet, without merging parts of different bets."""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "1_cleaning"))
from clean_pipeline import clean_directory
from transaction_dedup import bet_parts_table_name, deduplicate_bet_parts


def _write(path, rows, sep=","):
    pd.DataFrame(rows).to_csv(path, sep=sep, index=False)


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _part(pk, date, bet="001", part="0001", **extra):
    return {"pk_id": pk, "created_at": date, "wok_bet_pk_id": bet, "part_id": part,
            "part_odds": "2.00", **extra}


@pytest.mark.parametrize("chunksize", [1, 2, 100])
def test_latest_part_across_chunks_and_files_keeps_other_bets_and_parts(tmp_path, chunksize):
    first = tmp_path / "vendor_WOK_Bet_Parts_2.csv"
    second = tmp_path / "vendor_WOK_Bet_Parts_10.csv"
    _write(first, [
        _part("old", "2026-01-01"),
        _part("same-part-other-bet", "2026-01-01", bet="002"),
        _part("other-part", "2026-01-01", part="0002"),
    ])
    _write(second, [
        _part("new", "2026-01-03", part_odds="3.70"),
        _part("middle", "2026-01-02", part_odds="1.50"),
    ])
    [report] = deduplicate_bet_parts(tmp_path, chunksize=chunksize)
    result = _rows(first) + _rows(second)
    assert {r["pk_id"] for r in result} == {"new", "same-part-other-bet", "other-part"}
    assert next(r for r in result if r["pk_id"] == "new") == {
        "pk_id": "new", "created_at": "2026-01-03", "wok_bet_pk_id": "001",
        "part_id": "0001", "part_odds": "3.70"}
    assert report["duplicates_removed"] == 2
    before = [first.read_bytes(), second.read_bytes()]
    [again] = deduplicate_bet_parts(tmp_path, chunksize=chunksize)
    assert again["duplicates_removed"] == 0
    assert before == [first.read_bytes(), second.read_bytes()]


def test_tied_and_missing_timestamps_and_incomplete_keys(tmp_path):
    path = tmp_path / "WOK_Bet_Parts.csv"
    _write(path, [
        _part("old-undated", "invalid"),
        _part("new-undated", ""),
        _part("old-tie", "2026-01-01", part="0002"),
        _part("new-tie", "2026-01-01", part="0002"),
        _part("no-bet-1", "2026-01-01", bet=""),
        _part("no-bet-2", "2026-01-02", bet="NULL"),
        _part("no-part-1", "2026-01-01", part=""),
        _part("no-part-2", "2026-01-02", part="NULL"),
    ])
    [report] = deduplicate_bet_parts(tmp_path, chunksize=1)
    assert {r["pk_id"] for r in _rows(path)} == {
        "new-undated", "new-tie", "no-bet-1", "no-bet-2", "no-part-1", "no-part-2"}
    assert report["missing_ids"] == 4
    assert report["missing_timestamps"] == 2


@pytest.mark.parametrize("sep", [",", ";"])
def test_pipeline_removes_duplicate_parts_before_label_and_feature_calculation(tmp_path, monkeypatch, sep):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw / "WOK_Bet.csv", [{
        "pk_id": "bet-1", "operator_id": "a", "bet_id": "B1",
        "bet_start_datetime": "2026-01-01T10:00:00Z", "bet_status": "BET_PLACED",
        "extraction_date": "2026-01-02T00:00:00Z",
    }], sep=";")
    _write(raw / "WOK_Bet_Transaction.csv", [{
        "pk_id": "ref-1", "wok_bet_pk_id": "bet-1", "player_profile_id": "P1",
        "transactions_id": "T1", "created_at": "2026-01-01T10:00:00Z",
    }], sep=";")
    _write(raw / "WOK_Bet_Parts.csv", [
        _part("part-new", "2026-01-03", bet="bet-1", part_odds="3.70"),
        _part("part-old", "2026-01-01", bet="bet-1", part_odds="1.50"),
    ], sep=sep)
    original_parts = (raw / "WOK_Bet_Parts.csv").read_bytes()
    label_calls = []

    def label(directory, **kwargs):
        assert [r["pk_id"] for r in _rows(Path(directory) / "WOK_Bet_Parts.csv")] == ["part-new"]
        label_calls.append(directory)

    monkeypatch.setattr("clean_pipeline.label_outliers", label)
    cleaned = clean_directory(raw, tmp_path / "output", chunksize=1)
    assert label_calls == [str(cleaned)]
    assert (raw / "WOK_Bet_Parts.csv").read_bytes() == original_parts
    assert "WOK_Bet_Parts" in (raw / "logs" / "transaction_dedup.log").read_text()
    monkeypatch.syspath_prepend(str(ROOT / "3_features"))
    fes = importlib.import_module("feature_engineering_spanish")
    tables = {name: [cleaned / f"{name}.csv"]
              for name in ["WOK_Bet", "WOK_Bet_Transaction", "WOK_Bet_Parts"]}
    single = fes.f50_single_bet_percentage(tables, x_tijdspad=["01012026", "01022026"], chunksize=1)
    assert single["f50_single_bet_percentage"].tolist() == [1.0]
    odds = fes.f61_bet_odds_variability(tables, x_tijdspad=["01012026", "01022026"], chunksize=1)
    assert len(odds) == 1
    assert odds["f61_bet_odds_variability"].isna().all()


@pytest.mark.parametrize("name,expected", [
    ("WOK_Bet_Parts.csv", "WOK_Bet_Parts"),
    ("vendor_WOK_BET_PARTS_12.csv", "WOK_Bet_Parts"),
    ("WOK_Bet_Part_[2].csv", "WOK_Bet_Parts"),
    ("WOK_Bet_Transaction.csv", None),
])
def test_part_table_names(name, expected):
    assert bet_parts_table_name(name) == expected
