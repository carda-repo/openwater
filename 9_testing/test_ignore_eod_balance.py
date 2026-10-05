#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test voor de `ignore_EOD_Balance`-flag.

Op een dataset zónder `Player_Profile_EOD_Balance` lezen f26/f27/f28 die kolom hardcoded
en crasht `Flexible_spanish_plus` (pandas: "Usecols do not match columns"). Met de flag
krijgen die features `WOK_Player_Profile` niet aangereikt → de saldofeatures blijven
onbekend en de pijplijn loopt door.

Draaien:
    pytest _organisatie_code/9_testing/test_ignore_eod_balance.py -q
    python _organisatie_code/9_testing/test_ignore_eod_balance.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
organisatie = HERE.parent
for sub in ("3_features", "1_cleaning", "2_sorting"):
    p = str(organisatie / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
if str(organisatie) not in sys.path:
    sys.path.insert(0, str(organisatie))

from _relational_fixture import stage, X_PAD, Y_PAD   # noqa: E402
from clean_pipeline import clean_directory       # noqa: E402
from parse_pipeline import build_features        # noqa: E402

EOD_FEATURES = ("f26_balance_drop_frequency", "f27_deposits_after_below2_per_day",
                "f28_median_seconds_below2_to_deposit")


def _clean_without_eod():
    """Stage voorbeeld_fake, strip player_profile_eod_balance, clean → geef (tmp, cleaned_dir)."""
    tmp = Path(tempfile.mkdtemp(prefix="organisatie_noeod_"))
    op = stage(tmp / "raw", operators=("Operator_a",)) / "Operator_a"
    # relationele data is snake_case + ;-gescheiden; strip de eod-balance-kolom.
    prof = op / "WOK_Player_Profile.csv"
    df = pd.read_csv(prof, sep=";")
    df.drop(columns=["player_profile_eod_balance"], errors="ignore").to_csv(prof, sep=";", index=False)
    cleaned = clean_directory(input_dir=op, clean_out_dir=tmp / "clean",
                              chunksize=5000, do_label=True, verbose=False)
    return tmp, cleaned


def test_without_flag_crashes_on_missing_eod():
    """Zonder de flag crasht plus op een dataset zonder Player_Profile_EOD_Balance."""
    tmp, cleaned = _clean_without_eod()
    try:
        with pytest.raises(Exception) as exc:
            build_features(cleaned_dir=cleaned, scenario="Flexible_spanish_plus",
                           x_tijdspad=X_PAD, y_tijdspad=Y_PAD, chunksize=5000,
                           verbose=False, ignore_EOD_Balance=False)
        assert "Player_Profile_EOD_Balance" in str(exc.value)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_with_flag_runs_and_keeps_eod_features():
    """Met de flag loopt plus door en blijven f26/f27/f28 als onbekende kolommen aanwezig."""
    tmp, cleaned = _clean_without_eod()
    try:
        df = build_features(cleaned_dir=cleaned, scenario="Flexible_spanish_plus",
                            x_tijdspad=X_PAD, y_tijdspad=Y_PAD, chunksize=5000,
                            verbose=False, ignore_EOD_Balance=True)
        assert len(df) >= 1
        for feat in EOD_FEATURES:
            assert feat in df.columns, f"{feat} ontbreekt — fallback zou de kolom moeten leveren"
            assert df[feat].isna().all(), f"{feat}: een ontbrekend saldo mag geen 0 opleveren"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for t in (test_without_flag_crashes_on_missing_eod, test_with_flag_runs_and_keeps_eod_features):
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {t.__name__}: {e}")
            raise
