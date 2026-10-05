"""Person boundaries must survive modelling, cache loading and nested validation."""

import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "7_modelling"))
import hpsearch_runner as runner
import optuna_runner as opt_runner


class BoundaryClassifier(ClassifierMixin, BaseEstimator):
    seen = []

    def fit(self, X, y, eval_set=None, verbose=False):
        self.people_ = set(X["person_pattern"])
        self.classes_ = np.array([0, 1])
        type(self).seen.append(self.people_)
        if eval_set:
            assert self.people_.isdisjoint(eval_set[0][0]["person_pattern"])
        return self

    def predict_proba(self, X):
        assert self.people_.isdisjoint(X["person_pattern"])
        return np.tile([0.5, 0.5], (len(X), 1))


def repeated_people(n_people=30):
    people = np.repeat(np.arange(n_people), 4)
    index = pd.Index(np.arange(len(people)) * 3)
    return (pd.DataFrame({"person_pattern": people}, index=index),
            pd.Series(people % 2, index=index), pd.Series(people.astype(str), index=index))


@pytest.mark.parametrize("n_folds", [1, 5])
def test_cv_never_reuses_person_in_validation(n_folds):
    X, y, groups = repeated_people()
    BoundaryClassifier.seen.clear()
    scores = runner.cv_score(Pipeline([("model", BoundaryClassifier())]), X, y,
                             n_folds=n_folds, groups=groups)
    assert len(BoundaryClassifier.seen) == n_folds
    assert np.isfinite(scores[0])


def test_groups_none_retains_original_row_cv():
    X, y, _ = repeated_people()
    from sklearn.tree import DecisionTreeClassifier
    pipe = Pipeline([("model", DecisionTreeClassifier(random_state=7))])
    a = runner.cv_score(pipe, X, y, n_folds=3)
    b = runner.cv_score(pipe, X, y, n_folds=3, groups=None)
    assert a[:4] == b[:4]


def test_grouped_cv_rejects_too_few_people_and_misaligned_rows():
    X, y, groups = repeated_people(3)
    pipe = Pipeline([("model", BoundaryClassifier())])
    with pytest.raises(ValueError, match="5 distinct persons"):
        runner.cv_score(pipe, X, y, n_folds=5, groups=groups)
    with pytest.raises(ValueError, match="not aligned"):
        runner.cv_score(pipe, X, y, groups=groups.reset_index(drop=True))


def test_nested_early_stopping_uses_held_out_people():
    X, y, groups = repeated_people()
    X_test, y_test = X.iloc[-8:], y.iloc[-8:]
    runner.fit_and_eval(Pipeline([("model", BoundaryClassifier())]), X.iloc[:-8], y.iloc[:-8],
                        X_test, y_test, model_name="xgboost", groups=groups.iloc[:-8],
                        early_stopping_cfg={"enabled": True, "validation_fraction": 0.25})


@pytest.mark.parametrize("model", [MLPClassifier(early_stopping=True), HistGradientBoostingClassifier()])
def test_hidden_row_holdouts_disabled_in_grouped_fit(model):
    X, y, groups = repeated_people()
    pipe = Pipeline([("model", model)])
    with patch.object(pipe, "fit"):
        runner.fit_group_safe(pipe, X, y, groups)
    assert model.early_stopping is False


def test_calibration_validation_is_grouped_and_smote_rejected():
    X, y, groups = repeated_people()
    model = runner.build_estimator("linear_svc_cal", 1, 42)
    pipe = Pipeline([("model", model)])
    with patch.object(pipe, "fit"):
        runner.fit_group_safe(pipe, X, y, groups)
    assert len(model.cv) == 3
    for tr, va in model.cv:
        assert set(groups.iloc[tr]).isdisjoint(groups.iloc[va])
    with pytest.raises(ValueError, match="SMOTE"):
        runner.fit_group_safe(Pipeline([("smote", object()), ("model", model)]), X, y, groups)


def test_merged_periods_preserve_exact_operator_and_raw_person_ids(tmp_path):
    op = tmp_path / "operator-A"
    op.mkdir()
    target = "y_self_exclusion_20250101_20250131"
    for period in ["first", "second"]:
        pd.DataFrame({"Player_Profile_ID": ["001", "1"], "global_person": ["p01", "p02"],
                      "score": [0.2, 0.5], target: [0, 1]}).to_csv(op / f"{period}.csv", index=False)
    df, meta = runner.build_merged_df("operator-A", tmp_path, ["first", "second"],
                                     "Player_Profile_ID", explicit_target_col=target,
                                     column_prefixes=["p0", "p1"], identity_col="global_person")
    assert df["Player_Profile_ID"].tolist() == ["001", "1"]
    assert df["__niels_operator_id"].tolist() == ["operator-A", "operator-A"]
    assert df["__niels_person_id"].tolist() == ["p01", "p02"]
    X, _, features = runner.make_Xy(df, "Player_Profile_ID", target, identity_col="global_person")
    assert features == ["p0_score", "p1_score"]
    assert len(X) == 2


def prebuilt_fixture():
    X, y, groups = repeated_people()
    meta = {"Niels_Identity_Confounding_switch": True, "identity_column": "Player_Profile_ID",
            "identity_scope": "operator_player", "identity_split_random_state": 23,
            "identity_holdout_applied": True, "identity_dataset_id": "unit-dataset"}
    return {"X": X, "y": y, "groups": groups, "identity_metadata": meta.copy()}, meta


def test_prebuilt_alignment_and_manifest_required():
    payload, meta = prebuilt_fixture()
    assert runner.validate_prebuilt_identity(payload, meta, {}).equals(payload["groups"])
    with pytest.raises(ValueError, match="rebuild"):
        runner.validate_prebuilt_identity({"X": payload["X"], "y": payload["y"]}, {}, {})
    payload["identity_metadata"]["identity_dataset_id"] = "stale-paired-file"
    with pytest.raises(ValueError, match="manifest"):
        runner.validate_prebuilt_identity(payload, meta, {})


def test_disabled_old_cache_is_accepted_and_protected_cache_rejected():
    payload, meta = prebuilt_fixture()
    cfg = {"Niels_Identity_Confounding_switch": False}
    assert runner.validate_prebuilt_identity({"X": payload["X"], "y": payload["y"]}, {}, cfg) is None
    with pytest.raises(ValueError, match="differs"):
        runner.validate_prebuilt_identity(payload, meta, cfg)


def test_optuna_objective_passes_grouped_validation(tmp_path):
    import optuna
    X, y, groups = repeated_people()
    trial = optuna.trial.FixedTrial({"model": "decision_tree", "scaler_decision_tree": "none",
                                    "imbalance_decision_tree": "none", "decision_tree_max_depth": 2,
                                    "decision_tree_msl": 1})
    objective = opt_runner.make_objective(X, y, 1, 1, 42, [], tmp_path,
                                         active_models=["decision_tree"], groups=groups)
    with patch.object(opt_runner, "cv_score", return_value=(0.5, 0, 0.5, 0, 0)) as cv:
        assert objective(trial) == 0.5
    assert cv.call_args.kwargs["groups"] is groups


def test_optuna_revalidation_reuses_operator_person_folds(tmp_path):
    X, y, groups = repeated_people()
    rows = pd.DataFrame([{ "trial": 0, "model": "decision_tree", "cv_mean_auprc": 0.5}])
    args = SimpleNamespace(validate_best=True, cv5_only=False, cv5_top_n=1, random_state=42,
                           id_col="Player_Profile_ID")
    fold_data = [(X, y, X, y, 1, groups, groups)]
    with patch.object(opt_runner, "_pipeline_from_row", return_value=(None, "decision_tree", "median", "none", "none", {})), \
         patch.object(opt_runner, "operator_fold_score", return_value=(0.5, 0, 0.5, 0, 0)) as score, \
         patch.object(opt_runner, "cv_score") as cv:
        opt_runner._run_cv5_and_test(rows, args, 1, X, y, tmp_path, {}, False, None,
                                     list(X.columns), lambda _: None, groups=groups, fold_data=fold_data)
    assert score.call_args.args[1] is fold_data
    cv.assert_not_called()


def _raw_optuna_config(tmp_path, *, independent=False):
    target = "y_self_exclusion_20250101_20250131"
    for operator in ["alpha", "beta"]:
        folder = tmp_path / "data" / operator
        folder.mkdir(parents=True)
        pd.DataFrame({"Player_Profile_ID": [f"{i:03}" for i in range(30)],
                      "score": np.arange(30), "ACTIVE_FLAG": True,
                      target: np.arange(30) % 2}).to_csv(folder / "first_Flexible_spanish_plus.csv", index=False)
    cfg = {"all_mode": True, "data_dir": str(tmp_path / "data"),
           "all_operators": ["alpha"] if independent else ["alpha", "beta"],
           "validation_period_prefixes": ["first"], "test_period_prefixes": ["first"],
           "exclude_models": [m for m in opt_runner.ALL_MODELS if m != "decision_tree"]}
    if independent:
        cfg["fold_holdout_operators"] = ["beta"]
    else:
        cfg["operator_folds"] = [{"train": ["alpha"], "holdout": ["beta"]}]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.mark.parametrize("independent", [False, True])
def test_raw_optuna_keeps_global_test_people_out_of_operator_search(tmp_path, independent):
    import optuna
    cfg_path = _raw_optuna_config(tmp_path, independent=independent)
    trial = optuna.trial.FixedTrial({"model": "decision_tree", "scaler_decision_tree": "none",
                                    "imbalance_decision_tree": "none", "decision_tree_max_depth": 2,
                                    "decision_tree_msl": 1})
    study = SimpleNamespace(trials=[], optimize=lambda objective, **kwargs: objective(trial))
    with patch.object(optuna, "create_study", return_value=study), \
         patch.object(opt_runner, "cv_score", return_value=(0.5, 0, 0.5, 0, 0)), \
         patch.object(opt_runner, "operator_fold_score", return_value=(0.5, 0, 0.5, 0, 0)), \
         patch.object(opt_runner, "_run_cv5_and_test") as revalidate:
        opt_runner.main(["--config", str(cfg_path), "--out-dir", str(tmp_path / "out"),
                         "--time-budget", "1"])
    kwargs = revalidate.call_args.kwargs
    test_groups = set(kwargs["test_data"][2])
    assert set(kwargs["groups"]).isdisjoint(test_groups)
    if independent:
        assert {json.loads(g)[0] for g in kwargs["groups"]} == {"alpha"}
        assert {json.loads(g)[0] for g in test_groups} == {"beta"}
        assert len(kwargs["groups"]) == len(test_groups) == 30
    else:
        assert test_groups
        for fold in kwargs["fold_data"]:
            assert set(fold[5]).isdisjoint(test_groups)
            assert set(fold[6]).isdisjoint(test_groups)
            assert set(fold[5]).isdisjoint(fold[6])


def test_grid_valid_only_cache_needs_no_csv_or_test_period(tmp_path):
    payload, meta = prebuilt_fixture()
    meta["identity_holdout_applied"] = False
    payload["identity_metadata"] = meta.copy()
    meta["feature_cols"] = list(payload["X"].columns)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "valid_sampled.pkl").write_bytes(pickle.dumps(payload))
    (dataset / "meta.json").write_text(json.dumps(meta))
    cfg = {"data_dir": str(tmp_path / "missing_csv"), "dataset_path": str(dataset),
           "validation_period_prefixes": ["first"], "test_period_prefixes": [],
           "run_variants": [{"name": "base", "preprocessing": {"imputer": "median", "scaler": "none"}}],
           "models": {"decision_tree": {"grids": {"mini": [{"max_depth": 2}]}}}}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    with patch.object(runner, "build_merged_df", side_effect=AssertionError("Cache must bypass CSVs")):
        assert runner.main(["--config", str(cfg_path), "--out-dir", str(tmp_path / "out"),
                            "--operator", "alpha", "--model", "decision_tree", "--run-variant", "base",
                            "--grid-name", "mini", "--valid", "--no-test", "--cv-folds", "3"]) == 0
    assert (tmp_path / "out" / "results.csv").exists()
