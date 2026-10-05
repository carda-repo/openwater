#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test voor de één-call orchestrator `run_full_pipeline` (run_pipeline.py).

Draait de hele rit (1 t/m 9) op zelf-gegenereerde data via de publieke entrypoint, en checkt
dat de verwachte artefacten ontstaan.

Draaien:
    pytest _organisatie_code/9_testing/test_run_pipeline.py -q
    python _organisatie_code/9_testing/test_run_pipeline.py
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
organisatie = HERE.parent
if str(organisatie) not in sys.path:
    sys.path.insert(0, str(organisatie))

from _relational_fixture import stage, VALID_PREFIXES_3, TEST_PREFIXES_3  # noqa: E402
from run_pipeline import run_full_pipeline      # noqa: E402


def test_run_full_pipeline_optuna_end_to_end():
    tmp = Path(tempfile.mkdtemp(prefix="organisatie_main_"))
    try:
        data_dir = stage(tmp / "raw", operators=("Operator_a",))

        # Drie x-vensters (1m/2m/3m lookback) → p0/p1/p2 features op de relationele voorbeeld_fake.
        art = run_full_pipeline(
            data_dir=data_dir,
            out_dir=tmp / "out",
            scenario="Flexible_spanish_plus",
            # 3 lookback-horizonten (p0/p1/p2), apart test-venster → echte holdout (valid-y=mei, test-y=juni).
            validation_period_prefixes=VALID_PREFIXES_3,
            test_period_prefixes=TEST_PREFIXES_3,
            search="optuna",
            optuna_time_budget=4,
            optuna_cv_folds=1,
            optuna_validate_best=True,   # default is nu False (origineel opt-in) → cv5-stap expliciet aanzetten
            optuna_cv5_top_n=2,
            optuna_no_multivariate=True,
            chunksize=5000,
        )

        # Tussen-artefacten
        assert art["dataset_path"].joinpath("valid_sampled.pkl").exists()
        meta_ds = json.loads(art["dataset_path"].joinpath("meta.json").read_text(encoding="utf-8"))
        assert meta_ds["n_pos_valid"] > 0
        assert meta_ds["Niels_Identity_Confounding_switch"] is True
        assert meta_ds["identity_scope"] == "operator_player"
        with art["dataset_path"].joinpath("valid_sampled.pkl").open("rb") as handle:
            development = pickle.load(handle)
        with art["dataset_path"].joinpath("test_full.pkl").open("rb") as handle:
            test = pickle.load(handle)
        assert set(development["groups"]).isdisjoint(test["groups"])
        assert development["groups"].index.equals(development["X"].index)
        assert all(json.loads(group)[0] == "Operator_a" for group in development["groups"])

        # Drie periodes → de feature-set bevat p0_/p1_/p2_ kolommen.
        fc = meta_ds["feature_cols"]
        for px in ("p0_", "p1_", "p2_"):
            assert any(c.startswith(px) for c in fc), f"geen {px}-features (3 periodes verwacht)"

        # Model + rapportage (optuna-output)
        model_dir = art["model_out_dir"]
        assert model_dir.joinpath("optuna_meta.json").exists()
        model_meta = json.loads(model_dir.joinpath("optuna_meta.json").read_text(encoding="utf-8"))
        assert model_meta["Niels_Identity_Confounding_switch"] is True
        assert model_dir.joinpath("optuna_results.csv").exists()
        # validate_best → cv5 + test op (aparte) periode
        assert model_dir.joinpath("optuna_cv5_results.csv").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    try:
        test_run_full_pipeline_optuna_end_to_end()
        print("PASS  test_run_full_pipeline_optuna_end_to_end")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  test_run_full_pipeline_optuna_end_to_end: {e}")
        raise
