#!/usr/bin/env python3
"""
optuna_runner.py — CASH (Combined Algorithm Selection + Hyperparameter) search
on the combined ALL-operators dataset, using Optuna TPE.

Runs as a single job (not array), sequentially within a time budget.
Model type is part of the search space, so Optuna learns which models
are worth more budget for this dataset.

Usage:
    python optuna_runner.py \
        --config /path/hpsearch_config_effective.yaml \
        --out-dir /path/output \
        --time-budget 12600 \
        --cv-folds 1 \
        --n-startup 20
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from hpsearch_runner import (
    build_all_operators_merged_df,
    make_Xy,
    build_estimator,
    build_preprocess_steps,
    build_pipeline,
    apply_imbalance_to_params,
    cv_score,
    focal_binary_objective,
    get_feature_names_after_pipeline,
    try_get_feature_importances,
    _score_pipe,
    ID_COL_DEFAULT,
    RANDOM_STATE_DEFAULT,
    DTYPE_MAPPING,
    identity_enabled,
    identity_groups,
    person_holdout,
    fit_group_safe,
    validate_prebuilt_identity,
)

# Maps YAML/sklearn param names → Optuna internal key names (per model prefix)
# Used to translate YAML explicit-trial dicts into enqueue_trial dicts.
# Only params that appear in suggest_model_params need to be listed.
_YAML_TO_OPTUNA: Dict[str, Dict[str, str]] = {
    "xgboost": {
        "n_estimators": "xgboost_n_est", "learning_rate": "xgboost_lr",
        "max_depth": "xgboost_max_depth", "min_child_weight": "xgboost_mcw",
        "gamma": "xgboost_gamma", "subsample": "xgboost_subsample",
        "colsample_bytree": "xgboost_colsample", "reg_lambda": "xgboost_lambda",
    },
    "xgboost_focal": {
        "n_estimators": "xgboost_focal_n_est", "learning_rate": "xgboost_focal_lr",
        "max_depth": "xgboost_focal_max_depth", "min_child_weight": "xgboost_focal_mcw",
        "focal_alpha": "xgboost_focal_alpha", "focal_gamma": "xgboost_focal_gamma",
        "subsample": "xgboost_focal_subsample", "colsample_bytree": "xgboost_focal_colsample",
    },
    "hist_gb": {
        "learning_rate": "hist_gb_lr", "max_leaf_nodes": "hist_gb_leaves",
        "min_samples_leaf": "hist_gb_msl", "l2_regularization": "hist_gb_l2",
        "max_depth": "hist_gb_max_depth",
    },
    "lightgbm": {
        "n_estimators": "lightgbm_n_est", "learning_rate": "lightgbm_lr",
        "num_leaves": "lightgbm_leaves", "min_child_samples": "lightgbm_mcs",
        "subsample": "lightgbm_subsample", "colsample_bytree": "lightgbm_colsample",
    },
    "random_forest": {
        "n_estimators": "random_forest_n_est", "max_depth": "random_forest_max_depth",
        "max_features": "random_forest_max_feat",
    },
    "extra_trees": {
        "n_estimators": "extra_trees_n_est", "max_features": "extra_trees_max_feat",
    },
    "decision_tree": {
        "max_depth": "decision_tree_max_depth", "min_samples_leaf": "decision_tree_msl",
    },
    "adaboost": {
        "n_estimators": "adaboost_n_est", "learning_rate": "adaboost_lr",
    },
    "sgd_logloss": {"alpha": "sgd_logloss_alpha"},
}

# Params in YAML that Optuna doesn't tune (structural/fixed) — skip when seeding
_YAML_SKIP_PARAMS = {
    "reg_alpha", "tree_method", "grow_policy", "max_leaves",
    "bootstrap", "max_samples", "tol", "class_weight",
    "scale_pos_weight", "eval_metric",
}


def _parse_yaml_for_optuna(cfg: dict):
    """
    Parse the YAML config and return:
      seed_trials  : list of (model_name, optuna_param_dict) to enqueue_trial
      n_est_options: {model_name: sorted list of n_estimators values}
    """
    seed_trials = []
    n_est_options: Dict[str, list] = {}

    for mname, mcfg in (cfg.get("models") or {}).items():
        if mname not in ALL_MODELS:
            continue  # model not supported by Optuna (e.g. bagging_tree)
        param_map = _YAML_TO_OPTUNA.get(mname, {})
        n_est_vals: set = set()

        for gname, gspec in ((mcfg or {}).get("grids") or {}).items():
            # Collect n_estimators from cartesian grids
            if isinstance(gspec, dict):
                for src in (gspec.get("params") or {}, gspec.get("fixed") or {}):
                    v = src.get("n_estimators")
                    if isinstance(v, list):
                        n_est_vals.update(v)
                    elif v is not None:
                        n_est_vals.add(v)
                continue  # cartesian grids are not seeded (too many combinations)

            # Explicit trial lists → seed each trial
            if not isinstance(gspec, list):
                continue
            for trial_params in gspec:
                if not isinstance(trial_params, dict):
                    continue
                if trial_params.get("grow_policy") == "lossguide":
                    continue  # Optuna doesn't support lossguide; max_depth=0 breaks suggest_int
                if "n_estimators" in trial_params:
                    n_est_vals.add(trial_params["n_estimators"])

                optuna_params: Dict[str, Any] = {"model": mname}
                # Set preprocessing/imbalance to neutral defaults for seeded trials
                if mname in NAN_NATIVE:
                    optuna_params[f"imputer_{mname}"] = "none"
                    optuna_params[f"scaler_{mname}"]  = "none"
                else:
                    optuna_params[f"scaler_{mname}"] = "none"
                optuna_params[f"imbalance_{mname}"] = "none"

                for yaml_key, val in trial_params.items():
                    if yaml_key in _YAML_SKIP_PARAMS:
                        continue
                    opt_key = param_map.get(yaml_key)
                    if opt_key is not None and val is not None:
                        optuna_params[opt_key] = val

                # Skip if no model hyperparams translated (e.g. mlp has different parameterisation)
                model_hp_keys = {k for k in optuna_params
                                 if k != "model" and not k.startswith(
                                     ("imputer_", "scaler_", "imbalance_"))}
                if not model_hp_keys:
                    continue

                seed_trials.append((mname, optuna_params))

        if n_est_vals:
            n_est_options[mname] = sorted(n_est_vals)

    return seed_trials, n_est_options


_META_COLS = {
    "trial", "model", "imputer", "scaler", "imbalance",
    "cv_mean_auprc", "cv_std_auprc", "cv_mean_auc", "cv_std_auc",
    "cv1_auprc",
    "cv5_mean_auprc", "cv5_std_auprc", "cv5_mean_auc", "cv5_std_auc",
    "val_auprc", "test_auprc", "test_auc", "test_rows", "test_pos_rate",
    "fit_seconds", "error", "focal_alpha", "focal_gamma",
}

NAN_NATIVE    = {"xgboost", "xgboost_focal", "hist_gb", "lightgbm"}
SCALE_INVAR   = {"xgboost", "xgboost_focal", "hist_gb", "lightgbm",
                 "random_forest", "extra_trees", "bagging_tree", "decision_tree", "adaboost"}

ALL_MODELS = [
    "xgboost", "hist_gb", "lightgbm", "xgboost_focal",
    "random_forest", "extra_trees", "decision_tree", "adaboost",
    "sgd_logloss", "mlp",
]


def suggest_model_params(trial: optuna.Trial, model_name: str,
                         n_est_options: Dict[str, list] = None) -> Dict[str, Any]:
    p = {}
    m = model_name
    _n = n_est_options or {}

    def _n_est(default):
        """Union of YAML-derived values and hardcoded defaults, sorted."""
        return sorted(set(_n.get(m, [])) | set(default))

    if model_name == "xgboost":
        p["n_estimators"]     = trial.suggest_categorical(f"{m}_n_est",    _n_est([400, 800, 1600]))
        p["learning_rate"]    = trial.suggest_float(      f"{m}_lr",        0.01, 0.1,  log=True)
        p["max_depth"]        = trial.suggest_int(        f"{m}_max_depth", 2,    8)
        p["min_child_weight"] = trial.suggest_int(        f"{m}_mcw",       1,    20)
        p["gamma"]            = trial.suggest_float(      f"{m}_gamma",     0.0,  1.0)
        p["subsample"]        = trial.suggest_float(      f"{m}_subsample", 0.6,  1.0)
        p["colsample_bytree"] = trial.suggest_float(      f"{m}_colsample", 0.6,  1.0)
        p["reg_lambda"]       = trial.suggest_float(      f"{m}_lambda",    0.1,  5.0,  log=True)

    elif model_name == "xgboost_focal":
        p["n_estimators"]     = trial.suggest_categorical(f"{m}_n_est",    _n_est([400, 800]))
        p["learning_rate"]    = trial.suggest_float(      f"{m}_lr",        0.01, 0.1,  log=True)
        p["max_depth"]        = trial.suggest_int(        f"{m}_max_depth", 2,    8)
        p["min_child_weight"] = trial.suggest_int(        f"{m}_mcw",       1,    20)
        p["focal_alpha"]      = trial.suggest_float(      f"{m}_alpha",     0.005, 0.5, log=True)
        p["focal_gamma"]      = trial.suggest_float(      f"{m}_gamma",     0.5,  5.0)
        p["subsample"]        = trial.suggest_float(      f"{m}_subsample", 0.6,  1.0)
        p["colsample_bytree"] = trial.suggest_float(      f"{m}_colsample", 0.6,  1.0)

    elif model_name == "hist_gb":
        p["learning_rate"]     = trial.suggest_float(      f"{m}_lr",        0.01, 0.2,  log=True)
        p["max_leaf_nodes"]    = trial.suggest_int(        f"{m}_leaves",    10,   127)
        p["min_samples_leaf"]  = trial.suggest_int(        f"{m}_msl",       5,    100)
        p["l2_regularization"] = trial.suggest_float(      f"{m}_l2",        0.0,  2.0)
        p["max_depth"]         = trial.suggest_categorical(f"{m}_max_depth", [None, 3, 6, 10])

    elif model_name == "lightgbm":
        p["n_estimators"]      = trial.suggest_categorical(f"{m}_n_est",    _n_est([400, 800]))
        p["learning_rate"]     = trial.suggest_float(      f"{m}_lr",        0.01, 0.2,  log=True)
        p["num_leaves"]        = trial.suggest_int(        f"{m}_leaves",    10,   100)
        p["min_child_samples"] = trial.suggest_int(        f"{m}_mcs",       5,    100)
        p["subsample"]         = trial.suggest_float(      f"{m}_subsample", 0.6,  1.0)
        p["colsample_bytree"]  = trial.suggest_float(      f"{m}_colsample", 0.6,  1.0)

    elif model_name == "random_forest":
        p["n_estimators"] = trial.suggest_categorical(f"{m}_n_est",    _n_est([300, 600]))
        p["max_depth"]    = trial.suggest_categorical(f"{m}_max_depth",[None, 10, 20])
        _mf = trial.suggest_categorical(f"{m}_max_feat", ["sqrt", "0.3", "0.5"])
        p["max_features"] = float(_mf) if _mf not in ("sqrt", "log2") else _mf
        p["class_weight"] = "balanced_subsample"

    elif model_name == "extra_trees":
        p["n_estimators"] = trial.suggest_categorical(f"{m}_n_est",   _n_est([300, 600]))
        _mf = trial.suggest_categorical(f"{m}_max_feat",["sqrt", "0.3", "0.5"])
        p["max_features"] = float(_mf) if _mf not in ("sqrt", "log2") else _mf
        p["class_weight"] = "balanced_subsample"

    elif model_name == "decision_tree":
        p["max_depth"]        = trial.suggest_int(f"{m}_max_depth", 2, 12)
        p["min_samples_leaf"] = trial.suggest_int(        f"{m}_msl",      1, 50)

    elif model_name == "adaboost":
        p["n_estimators"]  = trial.suggest_categorical(f"{m}_n_est", _n_est([100, 300, 500]))
        p["learning_rate"] = trial.suggest_float(      f"{m}_lr",    0.01, 1.0, log=True)

    elif model_name == "sgd_logloss":
        p["alpha"]        = trial.suggest_float(f"{m}_alpha", 1e-6, 1e-2, log=True)
        p["max_iter"]     = 3000
        p["class_weight"] = "balanced"

    elif model_name == "mlp":
        n_layers   = trial.suggest_int(        f"{m}_n_layers",   1, 3)
        layer_size = trial.suggest_categorical(f"{m}_layer_size", [32, 64, 128])
        p["hidden_layer_sizes"] = tuple([layer_size] * n_layers)
        p["alpha"]              = trial.suggest_float(f"{m}_alpha", 1e-5, 1e-2, log=True)
        p["learning_rate_init"] = trial.suggest_float(f"{m}_lr",    1e-4, 1e-2, log=True)
        p["max_iter"]           = 60

    return p


def _pipeline_from_row(row, spw: float, random_state: int):
    """Reconstruct a fitted-ready pipeline from a row of optuna_results.csv."""
    model_name    = str(row["model"])
    imputer       = str(row.get("imputer",   "median") or "median")
    scaler        = str(row.get("scaler",    "standard") or "standard")
    imbalance_str = str(row.get("imbalance", "none") or "none")
    _fa = row.get("focal_alpha")
    _fg = row.get("focal_gamma")
    focal_alpha = None if (isinstance(_fa, float) and pd.isna(_fa)) else _fa
    focal_gamma = None if (isinstance(_fg, float) and pd.isna(_fg)) else _fg

    raw_params = {k: v for k, v in row.items()
                  if k not in _META_COLS and not (isinstance(v, float) and pd.isna(v))}
    params = {k: (int(v) if isinstance(v, float) and v == int(v) else v)
              for k, v in raw_params.items()}

    imbalance_cfg = {"strategy": imbalance_str, "class_weight": "balanced",
                     "sampling_strategy": 0.2, "k_neighbors": 5}
    params = apply_imbalance_to_params(params, model_name, imbalance_cfg, spw)
    rv = {"preprocessing": {"imputer": imputer, "scaler": scaler}, "imbalance": imbalance_cfg}

    est = build_estimator(model_name, spw=spw, random_state=random_state)
    est.set_params(**params)
    if model_name == "xgboost_focal" and focal_alpha is not None:
        est.set_params(objective=focal_binary_objective(focal_alpha, focal_gamma))
    pipe = build_pipeline(build_preprocess_steps(rv), rv["imbalance"], model_name, est)
    return pipe, model_name, imputer, scaler, imbalance_str, raw_params


def _write_second_half_report(feat_counts: dict, second_half_rows: list, out_dir: Path, log=print):
    """Write optuna_second_half_report.txt with feature importance + HP analysis for second-half trials."""
    import datetime
    n_trials = len(second_half_rows)
    lines = [
        "#" * 80,
        "# OPTUNA SECOND-HALF REPORT",
        f"# Trials in second half : {n_trials}",
        f"# Timestamp             : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "#" * 80,
        "",
    ]

    # ── Section 1: Feature importance ─────────────────────────────────────────
    lines += [
        "=" * 80,
        "SECTION 1: FEATURE IMPORTANCE  (second-half trials, incremental — no extra fit)",
        "=" * 80,
        f"  (Feature appears in top-12 importances for N of {n_trials} second-half trials)",
        f"  {'Feature':<60} {'#Trials':>8}  Bar",
        "  " + "-" * 80,
    ]
    if feat_counts:
        ranked = sorted(feat_counts.items(), key=lambda x: -x[1])
        max_cnt = ranked[0][1]
        bar_width = 40
        for feat, cnt in ranked:
            filled = round(cnt / max_cnt * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            lines.append(f"  {feat:<60} {cnt:>8}  {bar}")
    else:
        lines.append("  [WARN] No feature importances collected (model may not support them).")
    lines.append("")

    # ── Section 2: HP analysis ─────────────────────────────────────────────────
    lines += [
        "=" * 80,
        "SECTION 2: HYPERPARAMETER ANALYSIS  (second-half trials only)",
        "=" * 80,
        "  Per param-value: avg cv_AUPRC, std, n_trials, #top10, position bar.",
        "  *** hints fire when ≥60 % of trials cluster at one end of a 3+ value range.",
        "",
    ]

    RANK_LOW  = 0.2
    RANK_HIGH = 0.8
    df = pd.DataFrame(second_half_rows)

    for model_name in sorted(df["model"].dropna().unique()):
        mdf = df[df["model"] == model_name].copy()
        if len(mdf) < 2:
            continue

        hp_cols = sorted(
            c for c in mdf.columns
            if c not in _META_COLS and mdf[c].notna().any()
        )
        if not hp_cols:
            continue

        lines.append(f"  ── Model: {model_name}  ({len(mdf)} trials) ──")

        for param in hp_cols:
            col_vals = mdf[param].dropna()
            unique_vals = col_vals.unique()
            if len(unique_vals) < 2:
                continue

            try:
                sorted_vals = sorted(unique_vals, key=lambda v: float(v))
            except (TypeError, ValueError):
                sorted_vals = sorted(unique_vals, key=str)

            n_unique  = len(sorted_vals)
            val_to_nr = {v: i / (n_unique - 1) for i, v in enumerate(sorted_vals)}
            grp       = mdf.groupby(param)["cv_mean_auprc"].agg(["mean", "std", "count"])
            grp_dict  = grp.to_dict(orient="index")  # {val: {"mean": ..., "std": ..., "count": ...}}
            top10_vals = mdf.head(min(10, len(mdf)))[param].value_counts().to_dict()
            n_total   = int(col_vals.shape[0])
            best_val  = grp["mean"].idxmax()
            best_nrank = val_to_nr.get(best_val, 0.5)
            n_low  = int((col_vals == sorted_vals[0]).sum())
            n_high = int((col_vals == sorted_vals[-1]).sum())

            vals_repr = str([str(v) for v in sorted_vals])
            if len(vals_repr) > 72:
                vals_repr = f"[{sorted_vals[0]} … {sorted_vals[-1]}]  ({n_unique} values)"

            lines.append(f"\n  Param: {param}  (range: {vals_repr})")
            lines.append(f"  best_value={best_val}  nrank={best_nrank:.2f}  low={n_low}  high={n_high}  n_trials={n_total}")
            lines.append(f"  {'Value':<22} {'avg_PRAUC':>10} {'std':>8} {'n_trials':>9} {'#top10':>7}  Position")
            lines.append(f"  {'-'*72}")

            for val in sorted_vals:
                r = grp_dict.get(val)
                if r is None:
                    continue
                nrank_v = val_to_nr[val]
                wins    = top10_vals.get(val, 0)
                pos_idx = round(nrank_v * 9)
                pos_bar = "·" * pos_idx + "█" + "·" * (9 - pos_idx)
                std_v   = r["std"] if not pd.isna(r["std"]) else 0.0
                lines.append(
                    f"  {str(val):<22} {r['mean']:>10.4f} {std_v:>8.4f} "
                    f"{int(r['count']):>9} {wins:>7}  [{pos_bar}]  nrank={nrank_v:.2f}"
                )

            if n_unique >= 3:
                if best_nrank <= RANK_LOW and n_low / n_total >= 0.6:
                    lines.append(f"  *** << CONSIDER DECREASING RANGE  ({n_low}/{n_total} trials at min={sorted_vals[0]})")
                elif best_nrank >= RANK_HIGH and n_high / n_total >= 0.6:
                    lines.append(f"  *** >> CONSIDER INCREASING RANGE  ({n_high}/{n_total} trials at max={sorted_vals[-1]})")

        lines.append("")

    lines += ["#" * 80, "# END OF SECOND-HALF REPORT", "#" * 80]
    report_path = out_dir / "optuna_second_half_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[SECOND HALF] Report written → {report_path}")


def operator_fold_score(pipe, fold_data):
    """Reuse the same operator/person boundaries in search and revalidation."""
    from sklearn.base import clone
    from hpsearch_runner import safe_auc, safe_auprc
    scores_pr, scores_auc = [], []
    for fold in fold_data:
        X_tr, y_tr, X_ho, y_ho, _spw_fold = fold[:5]
        groups_tr, groups_ho = fold[5:7] if len(fold) >= 7 else (None, None)
        if groups_tr is not None and set(groups_tr).intersection(groups_ho):
            raise ValueError("Operator fold shares persons between training and holdout")
        fitted = clone(pipe)
        fit_group_safe(fitted, X_tr, y_tr, groups_tr)
        scores = _score_pipe(fitted, X_ho)
        scores_pr.append(safe_auprc(y_ho, scores))
        scores_auc.append(safe_auc(y_ho, scores))
    if not scores_pr:
        raise ValueError("No operator validation folds are available")
    return (float(np.nanmean(scores_pr)), float(np.nanstd(scores_pr)),
            float(np.nanmean(scores_auc)), float(np.nanstd(scores_auc)), 0.0)


def make_objective(X_train, y_train, spw: float, cv_folds: int, random_state: int, rows: list, out_dir: Path, log_fn=print, n_est_options: Dict[str, list] = None, time_budget: float = 0, feature_cols: list = None, active_models: list = None, fold_data: list = None, groups=None):
    t_search_start = time.time()
    second_half: Dict[str, Any] = {"active": False}  # True once elapsed >= 50% of budget
    feat_counts: Dict[str, int] = defaultdict(int)   # feature importance counts (second half only)
    second_half_rows: list = []                       # trial rows from the second half

    def objective(trial: optuna.Trial) -> float:
        model_name = trial.suggest_categorical("model", active_models or ALL_MODELS)

        imputer = (
            trial.suggest_categorical(f"imputer_{model_name}", ["none", "median"])
            if model_name in NAN_NATIVE else "median"
        )
        scaler = (
            trial.suggest_categorical(f"scaler_{model_name}", ["none", "standard"])
            if model_name in SCALE_INVAR else "standard"
        )
        # SMOTE requires no NaN; NAN_NATIVE models can skip imputation, so exclude SMOTE for them
        imbalance_choices = (
            ["none", "class_weight"]
            if model_name in NAN_NATIVE
            else ["none", "class_weight", "smote"]
        )
        imbalance_str = trial.suggest_categorical(f"imbalance_{model_name}", imbalance_choices)

        rv = {
            "preprocessing": {"imputer": imputer, "scaler": scaler},
            "imbalance": {"strategy": imbalance_str, "class_weight": "balanced",
                          "sampling_strategy": 0.2, "k_neighbors": 5},
        }

        params = suggest_model_params(trial, model_name, n_est_options)
        focal_alpha = params.pop("focal_alpha", None)
        focal_gamma = params.pop("focal_gamma", None)

        params = apply_imbalance_to_params(params, model_name, rv["imbalance"], spw)

        try:
            est = build_estimator(model_name, spw=spw, random_state=random_state)
            est.set_params(**params)
            if model_name == "xgboost_focal" and focal_alpha is not None:
                est.set_params(objective=focal_binary_objective(focal_alpha, focal_gamma))
        except Exception as e:
            raise optuna.exceptions.TrialPruned() from e

        pipe = build_pipeline(build_preprocess_steps(rv), rv["imbalance"], model_name, est)

        t0 = time.time()

        if fold_data is not None:
            # Operator cross-validation: fit on each fold's train, evaluate on holdout operators
            if groups is not None:
                try:
                    mean_pr, std_pr, mean_auc, std_auc, _ = operator_fold_score(pipe, fold_data)
                except Exception as e:
                    raise optuna.exceptions.TrialPruned(str(e)) from e
            else:
                # Preserve the original disabled-switch objective exactly.
                import sklearn.base
                from sklearn.metrics import average_precision_score as _aps
                fold_praucs = []
                for X_tr, y_tr, X_ho, y_ho, _spw_fold in fold_data:
                    try:
                        pipe_fold = sklearn.base.clone(pipe)
                        pipe_fold.fit(X_tr, y_tr)
                        proba = pipe_fold.predict_proba(X_ho)[:, 1]
                        fold_praucs.append(float(_aps(y_ho, proba)))
                    except Exception:
                        fold_praucs.append(0.0)
                mean_pr = float(np.mean(fold_praucs)) if fold_praucs else 0.0
                std_pr  = float(np.std(fold_praucs))  if len(fold_praucs) > 1 else 0.0
                mean_auc, std_auc = float("nan"), float("nan")
            _fitted_out = None
        else:
            _fitted_out: list = [] if feature_cols is not None else None
            try:
                mean_pr, std_pr, mean_auc, std_auc, _ = cv_score(
                    pipe, X_train, y_train,
                    n_folds=cv_folds, random_state=random_state,
                    model_name=model_name,
                    fitted_out=_fitted_out,
                    groups=groups,
                )
            except Exception as e:
                rows.append({
                    "trial": trial.number, "model": model_name,
                    "imputer": imputer, "scaler": scaler, "imbalance": imbalance_str,
                    "cv_mean_auprc": float("nan"), "error": str(e)[:200],
                })
                pd.DataFrame(rows).to_csv(out_dir / "optuna_results.csv", index=False)
                raise optuna.exceptions.TrialPruned() from e

        elapsed_total = time.time() - t_search_start
        rows.append({
            "trial": trial.number, "model": model_name,
            "imputer": imputer, "scaler": scaler, "imbalance": imbalance_str,
            "cv_mean_auprc": mean_pr, "cv_std_auprc": std_pr,
            "cv_mean_auc": mean_auc, "cv_std_auc": std_auc,
            "fit_seconds": round(time.time() - t0, 1),
            **params,
            **({"focal_alpha": focal_alpha, "focal_gamma": focal_gamma} if focal_alpha else {}),
        })
        pd.DataFrame(rows).to_csv(out_dir / "optuna_results.csv", index=False)
        best_so_far = max((r["cv_mean_auprc"] for r in rows if "cv_mean_auprc" in r and r["cv_mean_auprc"] == r["cv_mean_auprc"]), default=float("nan"))
        log_fn(f"[TRIAL {trial.number:>4}] {model_name:<16} AUPRC={mean_pr:.4f}"
               f"  best={best_so_far:.4f}  t={time.time()-t0:.0f}s  elapsed={elapsed_total/60:.1f}m")

        # Activate second half once we pass 50% of time budget
        if not second_half["active"] and time_budget > 0 and elapsed_total >= time_budget * 0.5:
            second_half["active"] = True
            log_fn(f"[SECOND HALF] Starting at elapsed={elapsed_total/60:.1f}m — collecting feature importances & HP stats")

        # Collect feature importances and HP rows for second-half trials only (no extra fit)
        if second_half["active"]:
            second_half_rows.append(rows[-1])
            if _fitted_out and feature_cols is not None:
                try:
                    _fp = _fitted_out[0]
                    _feat_names = get_feature_names_after_pipeline(_fp, feature_cols)
                    _imp_df = try_get_feature_importances(_fp.named_steps.get("model"), _feat_names)
                    if _imp_df is not None and len(_imp_df) > 0:
                        for feat in _imp_df.head(12)["feature"]:
                            feat_counts[feat] += 1
                except Exception:
                    pass

        return mean_pr

    objective.feat_counts = feat_counts
    objective.second_half_rows = second_half_rows

    return objective




# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers (threshold, confusion, significance)
# ─────────────────────────────────────────────────────────────────────────────

def _youden_threshold(y_true, y_score):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx]), float(tpr[idx]), float(1.0 - fpr[idx])


def _f1_threshold(y_true, y_score):
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    with np.errstate(invalid="ignore"):
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-9)
    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(precision[idx]), float(recall[idx]), float(f1[idx])


def _confusion_metrics(y_true, y_pred):
    import math
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv  = tp / max(tp + fp, 1)
    npv  = tn / max(tn + fn, 1)
    f1   = 2 * ppv * sens / max(ppv + sens, 1e-9)
    denom = math.sqrt(max(0.0, float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)))
    mcc  = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return dict(tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
                sens=sens, spec=spec, ppv=ppv, npv=npv, f1=f1, mcc=mcc)


def _normality_ok(a, b, alpha: float = 0.05) -> bool:
    """
    Returns True if both groups appear normally distributed (Shapiro-Wilk for
    n<50, D'Agostino K² otherwise).  True → t-test preferred; False → Mann-Whitney.
    """
    try:
        from scipy.stats import shapiro, normaltest
        def _ok(x):
            if len(x) < 8:
                return True          # too small to test, assume ok
            if len(x) < 50:
                _, p = shapiro(x[:5000])   # shapiro capped at 5000
            else:
                _, p = normaltest(x)
            return p >= alpha
        return _ok(a) and _ok(b)
    except Exception:
        return True   # scipy missing — default to t-test label


def _feature_significance_table(X: pd.DataFrame, y: pd.Series, top_n: int = 50):
    """
    For continuous features: Welch t-test + Mann-Whitney U, plus normality check
    to recommend which to use.  For binary features: chi2.
    Returns list of dicts sorted by the recommended p-value (ascending).
    """
    try:
        from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
    except ImportError:
        return []

    pos_mask = y == 1
    neg_mask = y == 0
    results = []
    for col in X.columns:
        a = X.loc[pos_mask, col].dropna().values
        b = X.loc[neg_mask, col].dropna().values
        if len(a) < 5 or len(b) < 5:
            continue
        try:
            col_vals = X[col].dropna()
            n_unique = col_vals.nunique()
            is_binary = n_unique <= 2 and set(col_vals.unique()).issubset({0, 1, 0.0, 1.0})
            if is_binary:
                ct = pd.crosstab(y, X[col].round().astype(int))
                if ct.shape != (2, 2):
                    continue
                chi2_stat, chi2_p, _, _ = chi2_contingency(ct.values)
                results.append({
                    "feature":      col,
                    "type":         "binary",
                    "mean_pos":     float(np.nanmean(a)),
                    "mean_neg":     float(np.nanmean(b)),
                    "mean_diff":    float(np.nanmean(a) - np.nanmean(b)),
                    "median_pos":   float(np.nanmedian(a)),
                    "median_neg":   float(np.nanmedian(b)),
                    "median_diff":  float(np.nanmedian(a) - np.nanmedian(b)),
                    "t_stat":       float("nan"), "t_pval": float("nan"),
                    "mw_stat":      float("nan"), "mw_pval": float("nan"),
                    "chi2_stat":    float(chi2_stat), "chi2_pval": float(chi2_p),
                    "preferred":    "chi2",
                    "p_value":      float(chi2_p),
                })
            else:
                t_stat,  t_p  = ttest_ind(a, b, equal_var=False)
                mw_stat, mw_p = mannwhitneyu(a, b, alternative="two-sided")
                normal = _normality_ok(a, b)
                preferred = "t-test" if normal else "Mann-Whitney"
                p_value = float(t_p) if normal else float(mw_p)
                results.append({
                    "feature":      col,
                    "type":         "continuous",
                    "mean_pos":     float(np.nanmean(a)),
                    "mean_neg":     float(np.nanmean(b)),
                    "mean_diff":    float(np.nanmean(a) - np.nanmean(b)),
                    "median_pos":   float(np.nanmedian(a)),
                    "median_neg":   float(np.nanmedian(b)),
                    "median_diff":  float(np.nanmedian(a) - np.nanmedian(b)),
                    "t_stat":       float(t_stat),  "t_pval":  float(t_p),
                    "mw_stat":      float(mw_stat), "mw_pval": float(mw_p),
                    "chi2_stat":    float("nan"),   "chi2_pval": float("nan"),
                    "preferred":    preferred,
                    "p_value":      p_value,
                })
        except Exception:
            pass
    results.sort(key=lambda r: r["p_value"])
    return results[:top_n]


def _write_top_model_statistics(df_cv5, X_train, y_train, X_test, y_test,
                                 feature_cols, spw, random_state, out_dir, log, groups=None):
    """
    For each top-N cv5 trial: fit once on X_train, determine thresholds on
    X_train predictions, evaluate confusion matrix on X_test (3 settings),
    compute feature importances and significance tests.
    Writes top_model_statistics.txt.
    """
    from sklearn.metrics import average_precision_score as _aps, roc_auc_score as _roc

    lines = []
    W = lines.append
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    n_train = len(y_train); pos_train = int(y_train.sum())
    n_test  = len(y_test);  pos_test  = int(y_test.sum())

    W("#" * 80)
    W("# TOP MODEL STATISTICS")
    W(f"# Generated:        {now}")
    W(f"# Train (valid) set: {n_train:,} rows | {pos_train:,} pos ({pos_train/max(n_train,1):.4f})")
    W(f"# Test  set:         {n_test:,} rows  | {pos_test:,}  pos ({pos_test/max(n_test,1):.4f})")
    W(f"# Features:          {len(feature_cols)}")
    W("#" * 80)

    # ── Feature significance table (computed on test set — full unsampled distribution) ──
    W("\n" + "=" * 80)
    W("FEATURE SIGNIFICANCE  (test set — target group comparison)")
    W("  Continuous: Welch t-test + Mann-Whitney U; normality tested per group")
    W("  Binary:     Chi-squared test")
    W("  'Use' column = recommended test based on normality (Shapiro / D'Agostino K²)")
    W("  Significance: * p<0.05   ** p<0.01   *** p<0.001")
    W("=" * 80)
    X_feat = X_test[feature_cols] if set(feature_cols) <= set(X_test.columns) \
        else X_test.reindex(columns=feature_cols)
    sig_rows = _feature_significance_table(X_feat, y_test)
    if sig_rows:
        # Header
        W(f"\n  {'#':>3}  {'Feature':<46}  {'Use':<14}"
          f"  {'Mean(1)':>9}  {'Mean(0)':>9}  {'ΔMean':>9}"
          f"  {'Median(1)':>9}  {'Median(0)':>9}  {'ΔMedian':>9}"
          f"  {'t-stat':>9}  {'t-p':>11}"
          f"  {'U-stat':>9}  {'MW-p':>11}"
          f"  {'p(use)':>11}  Sig")
        W(f"  {'-'*175}")
        for i, r in enumerate(sig_rows, 1):
            p    = r["p_value"]
            sig  = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "   "
            pref = r["preferred"]
            t_s  = f"{r['t_stat']:>9.3f}"  if not (isinstance(r['t_stat'],  float) and np.isnan(r['t_stat']))  else f"{'—':>9}"
            t_p  = f"{r['t_pval']:>11.3e}" if not (isinstance(r['t_pval'],  float) and np.isnan(r['t_pval']))  else f"{'—':>11}"
            mw_s = f"{r['mw_stat']:>9.1f}" if not (isinstance(r['mw_stat'], float) and np.isnan(r['mw_stat'])) else f"{'—':>9}"
            mw_p = f"{r['mw_pval']:>11.3e}"if not (isinstance(r['mw_pval'], float) and np.isnan(r['mw_pval']))else f"{'—':>11}"
            if r["type"] == "binary":
                t_s = mw_s = t_p = mw_p = f"{'—':>9}"
                mw_p = f"{'—':>11}"
                pref = f"chi2 (p={r['chi2_pval']:.2e})"
            W(f"  {i:>3}  {r['feature']:<46}  {pref:<14}"
              f"  {r['mean_pos']:>9.4f}  {r['mean_neg']:>9.4f}  {r['mean_diff']:>+9.4f}"
              f"  {r['median_pos']:>9.4f}  {r['median_neg']:>9.4f}  {r['median_diff']:>+9.4f}"
              f"  {t_s}  {t_p}"
              f"  {mw_s}  {mw_p}"
              f"  {p:>11.3e}  {sig}")
    else:
        W("  (scipy not available — significance tests skipped)")

    # ── Per-model sections ────────────────────────────────────────────────────
    # Support both cv5 results (cv5_mean_auprc) and fallback from cv1 (cv_mean_auprc)
    if "cv5_mean_auprc" in df_cv5.columns:
        valid_cv5 = df_cv5.dropna(subset=["cv5_mean_auprc"])
    else:
        # fallback: cv5 wasn't run, work from cv1 rows
        df_cv5 = df_cv5.copy()
        df_cv5["cv5_mean_auprc"] = df_cv5.get("cv_mean_auprc", float("nan"))
        df_cv5["cv5_std_auprc"]  = df_cv5.get("cv_std_auprc",  float("nan"))
        df_cv5["cv5_mean_auc"]   = df_cv5.get("cv_mean_auc",   float("nan"))
        df_cv5["cv1_auprc"]      = df_cv5.get("cv_mean_auprc", float("nan"))
        valid_cv5 = df_cv5.dropna(subset=["cv5_mean_auprc"])
    for rank, (_, row) in enumerate(valid_cv5.iterrows(), 1):
        trial      = int(row["trial"])
        model_name = str(row["model"])
        cv1_auprc  = float(row.get("cv1_auprc", float("nan")))
        cv5_auprc  = float(row.get("cv5_mean_auprc", float("nan")))
        cv5_std    = float(row.get("cv5_std_auprc", float("nan")))
        cv5_auc    = float(row.get("cv5_mean_auc", float("nan")))

        W("\n" + "=" * 80)
        W(f"MODEL #{rank}  trial={trial}  {model_name.upper()}")
        W("=" * 80)
        W(f"  cv=1  AUPRC: {cv1_auprc:.4f}")
        W(f"  cv=5  AUPRC: {cv5_auprc:.4f} ± {cv5_std:.4f}   AUC: {cv5_auc:.4f}")

        try:
            pipe, _, imputer, scaler, imbalance_str, raw_params = \
                _pipeline_from_row(row, spw, random_state)

            W(f"\n  {'Imputer':<26}: {imputer}")
            W(f"  {'Scaler':<26}: {scaler}")
            W(f"  {'Imbalance':<26}: {imbalance_str}")
            for k, v in sorted(raw_params.items()):
                W(f"  {k:<26}: {v}")

            # Align feature columns between train and test
            X_tr = X_train[feature_cols] if set(feature_cols) <= set(X_train.columns) \
                else X_train.reindex(columns=feature_cols)
            X_te = X_test[feature_cols]  if set(feature_cols) <= set(X_test.columns) \
                else X_test.reindex(columns=feature_cols)

            # Single fit on training (validation) data
            fit_group_safe(pipe, X_tr, y_train, groups)
            y_prob_train = _score_pipe(pipe, X_tr)
            y_prob_test  = _score_pipe(pipe, X_te)

            train_auprc = float(_aps(y_train, y_prob_train))
            train_auc   = float(_roc(y_train, y_prob_train))
            test_auprc  = float(_aps(y_test,  y_prob_test))
            test_auc    = float(_roc(y_test,  y_prob_test))

            W(f"\n  {'Train (valid) AUPRC':<26}: {train_auprc:.4f}   AUC: {train_auc:.4f}")
            W(f"  {'Test AUPRC':<26}: {test_auprc:.4f}   AUC: {test_auc:.4f}")

            # Thresholds determined on training (validation) set — no leakage
            thr_y, tpr_y, tnr_y     = _youden_threshold(y_train.values, y_prob_train)
            thr_f, prec_f, rec_f, f1_f = _f1_threshold(y_train.values, y_prob_train)
            thresholds = [
                ("Youden's J", thr_y,  f"TPR(val)={tpr_y:.4f}  TNR(val)={tnr_y:.4f}"),
                ("F1-optimum", thr_f,  f"PPV(val)={prec_f:.4f}  Recall(val)={rec_f:.4f}  F1(val)={f1_f:.4f}"),
                ("Fixed 0.5",  0.5,   "no optimisation"),
            ]

            W("\n  ── Confusion matrices (test set) ──────────────────────────────────────────")
            W(f"\n  {'Method':<14}  {'Thr':>6}  {'TP':>7}  {'FP':>7}  {'TN':>7}  {'FN':>7}  "
              f"{'Sens':>7}  {'Spec':>7}  {'PPV':>7}  {'NPV':>7}  {'F1':>7}  {'MCC':>7}")
            W(f"  {'-'*105}")
            for label, thr, notes in thresholds:
                y_pred = (y_prob_test >= thr).astype(int)
                m = _confusion_metrics(y_test.values, y_pred)
                W(f"  {label:<14}  {thr:>6.4f}  {m['tp']:>7}  {m['fp']:>7}  "
                  f"{m['tn']:>7}  {m['fn']:>7}  "
                  f"{m['sens']:>7.4f}  {m['spec']:>7.4f}  {m['ppv']:>7.4f}  "
                  f"{m['npv']:>7.4f}  {m['f1']:>7.4f}  {m['mcc']:>7.4f}")
                W(f"                  ({notes})")

            # Feature importances
            W("\n  ── Feature importances (top 20) ───────────────────────────────────────────")
            try:
                final_est = pipe.named_steps.get("model")
                surviving = get_feature_names_after_pipeline(pipe, feature_cols)
                X_te_tr = X_te.copy()
                for sname, step in pipe.steps[:-1]:
                    if sname == "smote" or not hasattr(step, "transform"):
                        continue
                    X_te_tr = step.transform(X_te_tr)
                fi = try_get_feature_importances(
                    final_est, surviving,
                    X_test=X_te_tr, y_test=y_test,
                    random_state=random_state,
                )
                if fi is not None and not fi.empty:
                    top20   = fi.head(20)
                    imp_col = "importance_mean" if "importance_mean" in fi.columns else "importance"
                    std_col = "importance_std"  if "importance_std"  in fi.columns else None
                    imp_type = "permutation" if std_col else "native (gain/weight)"
                    W(f"  Type: {imp_type}")
                    W(f"  {'#':>3}  {'Feature':<52}  {'Importance':>12}{'  ±Std' if std_col else ''}")
                    W(f"  {'-'*75}")
                    for i, (_, frow) in enumerate(top20.iterrows(), 1):
                        std_s = f"  ±{frow[std_col]:.6f}" if std_col else ""
                        W(f"  {i:>3}  {str(frow['feature']):<52}  {frow[imp_col]:>12.6f}{std_s}")
                else:
                    W("  (not available)")
            except Exception as fe:
                W(f"  [Feature importances failed: {fe}]")

        except Exception as exc:
            W(f"\n  [ERROR evaluating model: {exc}]")

    W("\n" + "#" * 80)
    W("# END")
    W("#" * 80)

    out_path = out_dir / "top_model_statistics.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[OUT] {out_path}")


def _run_cv5_and_test(df_out, args, spw, X_train, y_train, out_dir, meta,
                      use_prebuilt, dataset_path, feature_cols, log, groups=None, cfg=None,
                      fold_data=None, test_data=None):
    """Run cv=5 re-evaluation and/or test evaluation. Shared by normal and --cv5-only paths."""
    best_cv1 = df_out.dropna(subset=["cv_mean_auprc"]).iloc[0] if not df_out.empty else None
    if best_cv1 is not None:
        log(f"[BEST cv=1] model={best_cv1['model']}  AUPRC={best_cv1['cv_mean_auprc']:.4f}")
    best_final = best_cv1

    # ── cv=5 re-evaluation of top N trials ───────────────────────────────────
    df_cv5 = None
    if (args.validate_best or args.cv5_only) and best_cv1 is not None:
        top_rows = df_out.dropna(subset=["cv_mean_auprc"]).head(args.cv5_top_n)
        log(f"[CV5] Re-evaluating top {len(top_rows)} trials with cv=5 ...")
        cv5_rows = []
        for _, row in top_rows.iterrows():
            t0 = time.time()
            try:
                pipe, model_name, imputer, scaler, imbalance_str, raw_params = \
                    _pipeline_from_row(row, spw, args.random_state)
                if fold_data is not None and groups is not None:
                    mean_pr, std_pr, mean_auc, std_auc, _ = operator_fold_score(pipe, fold_data)
                else:
                    mean_pr, std_pr, mean_auc, std_auc, _ = cv_score(
                        pipe, X_train, y_train,
                        n_folds=5, random_state=args.random_state,
                        model_name=model_name, groups=groups,
                    )
                cv5_rows.append({
                    "trial": int(row["trial"]), "model": model_name,
                    "imputer": imputer, "scaler": scaler, "imbalance": imbalance_str,
                    "cv1_auprc": float(row["cv_mean_auprc"]),
                    "cv5_mean_auprc": mean_pr, "cv5_std_auprc": std_pr,
                    "cv5_mean_auc": mean_auc, "cv5_std_auc": std_auc,
                    "fit_seconds": round(time.time() - t0, 1),
                    **raw_params,
                })
                log(f"[CV5] trial={int(row['trial'])}  {model_name}"
                    f"  cv5_AUPRC={mean_pr:.4f} ± {std_pr:.4f}")
            except Exception as exc:
                log(f"[CV5] trial={int(row['trial'])} FAILED: {exc}")
                cv5_rows.append({
                    "trial": int(row["trial"]), "model": str(row["model"]),
                    "cv1_auprc": float(row["cv_mean_auprc"]),
                    "cv5_mean_auprc": float("nan"), "error": str(exc)[:200],
                })

        df_cv5 = pd.DataFrame(cv5_rows).sort_values(
            "cv5_mean_auprc", ascending=False, na_position="last")
        df_cv5.to_csv(out_dir / "optuna_cv5_results.csv", index=False)
        log(f"[OUT] {out_dir / 'optuna_cv5_results.csv'}")

        best_cv5 = df_cv5.dropna(subset=["cv5_mean_auprc"]).iloc[0] \
            if not df_cv5.empty else None
        if best_cv5 is not None:
            log(f"[BEST cv=5] model={best_cv5['model']}"
                f"  cv5_AUPRC={best_cv5['cv5_mean_auprc']:.4f}")
            meta["best_cv5_model"] = str(best_cv5["model"])
            meta["best_cv5_auprc"] = float(best_cv5["cv5_mean_auprc"])
            (out_dir / "optuna_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            best_final = best_cv5

    # ── Test evaluation on best trial + top_model_statistics.txt ────────────
    test_pkl_path = None
    if use_prebuilt and dataset_path is not None:
        _tp = dataset_path / "test_full.pkl"
        if _tp.exists():
            test_pkl_path = _tp

    if best_final is None:
        pass
    elif test_pkl_path is None and test_data is None:
        log("[TEST] No test_full.pkl found — skipping test eval and statistics.")
    else:
        import pickle as _pickle
        from sklearn.metrics import average_precision_score, roc_auc_score

        if test_data is not None:
            X_test, y_test, groups_test = test_data
        else:
            with open(test_pkl_path, "rb") as _f:
                _dt = _pickle.load(_f)
            X_test, y_test = _dt["X"], _dt["y"]
            _meta_ds = json.loads((dataset_path / "meta.json").read_text(encoding="utf-8"))
            groups_test = validate_prebuilt_identity(_dt, _meta_ds, cfg or {}, args.id_col)
        if groups is not None:
            if groups_test is None or set(groups).intersection(groups_test):
                raise ValueError("Training and test share persons or lack test groups; rebuild the dataset")
        # Apply same feature filter as training data
        if feature_cols is not None:
            X_test = X_test[[c for c in feature_cols if c in X_test.columns]]

        # Best-trial summary CSV (optuna_best_test_result.csv)
        log(f"[TEST] Evaluating best trial on test set ({test_pkl_path}) ...")
        try:
            pipe, model_name, imputer, scaler, imbalance_str, raw_params = \
                _pipeline_from_row(best_final, spw, args.random_state)
            log("[TEST] Fitting best pipeline on training data ...")
            fit_group_safe(pipe, X_train, y_train, groups)
            y_prob     = pipe.predict_proba(X_test)[:, 1]
            test_auprc = float(average_precision_score(y_test, y_prob))
            test_auc   = float(roc_auc_score(y_test, y_prob))
            test_pos   = int(y_test.sum())
            test_neg   = int((y_test == 0).sum())
            test_rate  = test_pos / max(test_pos + test_neg, 1)
            log(f"[TEST] AUPRC={test_auprc:.4f}  AUC={test_auc:.4f}  "
                f"rows={test_pos+test_neg}  pos_rate={test_rate:.4f}")
            val_auprc = float(
                best_final.get("cv5_mean_auprc",
                               best_final.get("cv_mean_auprc", float("nan"))))
            pd.DataFrame([{
                "model": model_name, "imputer": imputer, "scaler": scaler,
                "imbalance": imbalance_str, "val_auprc": val_auprc,
                "test_auprc": test_auprc, "test_auc": test_auc,
                "test_rows": test_pos + test_neg,
                "test_pos_rate": round(test_rate, 6),
                **raw_params,
            }]).to_csv(out_dir / "optuna_best_test_result.csv", index=False)
            meta["best_test_auprc"] = test_auprc
            meta["best_test_auc"]   = test_auc
            (out_dir / "optuna_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            log(f"[OUT] {out_dir / 'optuna_best_test_result.csv'}")
        except Exception as exc:
            log(f"[TEST] ⚠  Failed to evaluate best trial on test set: {exc}")

        # Detailed statistics for all top cv5 models
        _df_for_stats = df_cv5 if df_cv5 is not None else pd.DataFrame([best_final])
        try:
            _write_top_model_statistics(
                _df_for_stats, X_train, y_train, X_test, y_test,
                feature_cols, spw, args.random_state, out_dir, log, groups=groups,
            )
        except Exception as exc:
            log(f"[STATS] ⚠  top_model_statistics.txt failed: {exc}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       required=True)
    ap.add_argument("--out-dir",      required=True)
    ap.add_argument("--time-budget",  type=int, default=12600,
                    help="Seconds to run (default: 3.5h)")
    ap.add_argument("--cv-folds",     type=int, default=1)
    ap.add_argument("--n-startup",    type=int, default=20,
                    help="Random trials before TPE kicks in")
    ap.add_argument("--round",        action="store_true",
                    help="Apply DTYPE_MAPPING after loading (reduces memory).")
    ap.add_argument("--validate-best", action="store_true",
                    help="After the search, re-evaluate top N trials with cv=5.")
    ap.add_argument("--cv5-only",     action="store_true",
                    help="Skip search; load optuna_results.csv and run cv=5 + test only.")
    ap.add_argument("--cv5-top-n",    type=int, default=5,
                    help="Number of top trials to re-evaluate with cv=5 (default: 5).")
    ap.add_argument("--id-col",       default=ID_COL_DEFAULT)
    ap.add_argument("--random-state", type=int, default=RANDOM_STATE_DEFAULT)
    ap.add_argument("--no-multivariate", action="store_true",
                    help="Disable multivariate TPE (reduces lock-in on fast models)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg): print(msg, flush=True)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    use_identity = identity_enabled(cfg)
    identity_col = (cfg.get("Niels_identity_column") or args.id_col) if use_identity else None
    groups_train = None
    test_data = None
    reserved_test_groups = set()
    data_dir = Path(cfg.get("data_dir", ""))
    target_col = cfg.get("target_col", "") or ""
    validation_period_prefixes = cfg.get("validation_period_prefixes") or []
    test_period_prefixes = cfg.get("test_period_prefixes") or []

    if not cfg.get("all_mode"):
        raise SystemExit("optuna_runner requires all_mode=True in config (use --all in submit_hpsearch.sh)")

    generic_col_prefixes = [f"p{i}" for i in range(len(test_period_prefixes))]

    dataset_path_str = cfg.get("dataset_path", "")
    dataset_path = Path(dataset_path_str) if dataset_path_str else None
    use_prebuilt = (
        dataset_path is not None
        and (dataset_path / "valid_sampled.pkl").exists()
    )

    t_load = time.time()
    if use_prebuilt:
        import pickle as _pickle
        log(f"[DATA] Loading pre-built dataset from {dataset_path}")
        with open(dataset_path / "valid_sampled.pkl", "rb") as _f:
            _dv = _pickle.load(_f)
        X_train, y_train = _dv["X"], _dv["y"]
        import json as _json
        _meta_ds = _json.loads((dataset_path / "meta.json").read_text(encoding="utf-8"))
        groups_train = validate_prebuilt_identity(_dv, _meta_ds, cfg, args.id_col)
        feature_cols = _meta_ds["feature_cols"]
        if use_identity and (dataset_path / "test_full.pkl").exists():
            with open(dataset_path / "test_full.pkl", "rb") as _f:
                _dt = _pickle.load(_f)
            groups_test = validate_prebuilt_identity(_dt, _meta_ds, cfg, args.id_col)
            if not _meta_ds.get("identity_holdout_applied") or set(groups_train).intersection(groups_test):
                raise ValueError("Pre-built dataset has no valid person test holdout; rebuild the dataset")
            reserved_test_groups = set(groups_test)
    else:
        log("[DATA] Building combined ALL-operators dataset ...")
        df_valid, df_test, merge_meta_valid, merge_meta_test = build_all_operators_merged_df(
            cfg=cfg, data_dir=data_dir,
            validation_period_prefixes=validation_period_prefixes,
            test_period_prefixes=test_period_prefixes,
            id_col=args.id_col, explicit_target_col=target_col,
            column_prefixes=generic_col_prefixes,
        )
        if use_identity and cfg.get("fold_holdout_operators"):
            # Independent operator folds evaluate the configured held-out
            # providers in the same period, matching dataset preparation.
            cfg_holdout = {**cfg, "all_operators": cfg["fold_holdout_operators"]}
            df_test, _, merge_meta_test, _ = build_all_operators_merged_df(
                cfg=cfg_holdout, data_dir=data_dir,
                validation_period_prefixes=validation_period_prefixes,
                test_period_prefixes=[], id_col=args.id_col,
                explicit_target_col=target_col, column_prefixes=generic_col_prefixes,
            )
        tgt = merge_meta_valid["target_col"]
        df_valid, df_test, groups_train, groups_test, identity_meta = person_holdout(
            df_valid, df_test, cfg, args.id_col, random_state=23,
        )
        X_train, y_train, feature_cols = make_Xy(df_valid, id_col=args.id_col, target_col=tgt, identity_col=identity_col)
        if use_identity and not df_test.empty:
            X_test, y_test, _ = make_Xy(df_test, id_col=args.id_col, target_col=merge_meta_test["target_col"], identity_col=identity_col)
            test_data = (X_test, y_test, groups_test)
            reserved_test_groups = set(groups_test)

    # ── --only-these-vars column filter ──────────────────────────────────────
    exclude_models = cfg.get("exclude_models") or []
    active_models = [m for m in ALL_MODELS if m not in exclude_models]
    if exclude_models:
        log(f"[FILTER] exclude_models: {exclude_models} → active: {active_models}")

    only_these = cfg.get("only_these_vars") or []
    if only_these:
        import re as _re
        n_before = len(feature_cols)
        keep = [c for c in feature_cols
                if any(_re.sub(r'^p\d+_', '', c) == v or c == v for v in only_these)]
        if not keep:
            raise SystemExit(
                f"[FILTER] --only-these-vars: none of {only_these} matched any feature column. "
                f"Sample columns: {feature_cols[:8]}"
            )
        X_train = X_train[keep]
        feature_cols = keep
        log(f"[FILTER] only_these_vars: kept {len(keep)} of {n_before} feature columns")

    if args.round and DTYPE_MAPPING:
        apply = {c: t for c, t in DTYPE_MAPPING.items() if c in X_train.columns}
        if apply:
            X_train = X_train.astype(apply)
            log(f"[DATA] --round: {len(apply)} columns dtype-converted")

    # ── operator_folds mode: preload per-fold data for averaged CV objective ─
    operator_folds_cfg = cfg.get("operator_folds")
    fold_data = None  # list of (X_tr, y_tr, X_ho, y_ho, spw_fold)
    if operator_folds_cfg and not use_prebuilt:
        log(f"[OPERATOR_FOLDS] Preloading data for {len(operator_folds_cfg)} folds ...")
        fold_data = []
        for fi, fold_info in enumerate(operator_folds_cfg):
            train_ops = fold_info["train"]
            holdout_ops = fold_info["holdout"]
            log(f"[OPERATOR_FOLDS] fold={fi}  train={train_ops}  holdout={holdout_ops}")
            cfg_train   = {**cfg, "all_operators": train_ops}
            cfg_holdout = {**cfg, "all_operators": holdout_ops}
            df_tr, _, meta_tr, _ = build_all_operators_merged_df(
                cfg=cfg_train, data_dir=data_dir,
                validation_period_prefixes=validation_period_prefixes,
                test_period_prefixes=validation_period_prefixes,
                id_col=args.id_col, explicit_target_col=target_col,
                column_prefixes=generic_col_prefixes,
            )
            df_ho, _, meta_ho, _ = build_all_operators_merged_df(
                cfg=cfg_holdout, data_dir=data_dir,
                validation_period_prefixes=validation_period_prefixes,
                test_period_prefixes=validation_period_prefixes,
                id_col=args.id_col, explicit_target_col=target_col,
                column_prefixes=generic_col_prefixes,
            )
            tgt = meta_tr["target_col"]
            if use_identity and reserved_test_groups:
                # Search folds may rebuild raw CSVs, but globally held-out test
                # persons must remain unavailable during hyperparameter selection.
                df_tr = df_tr.loc[~identity_groups(df_tr, args.id_col, cfg).isin(reserved_test_groups)].copy()
                df_ho = df_ho.loc[~identity_groups(df_ho, args.id_col, cfg).isin(reserved_test_groups)].copy()
            df_tr, df_ho, groups_tr, groups_ho, _ = person_holdout(
                df_tr, df_ho, cfg, args.id_col, random_state=23,
            )
            X_tr, y_tr, _ = make_Xy(df_tr, id_col=args.id_col, target_col=tgt, identity_col=identity_col)
            X_ho, y_ho, _ = make_Xy(df_ho, id_col=args.id_col, target_col=tgt, identity_col=identity_col)
            # Align to same feature columns as global X_train (respects only_these_vars filter)
            X_tr = X_tr[[c for c in feature_cols if c in X_tr.columns]]
            X_ho = X_ho[[c for c in feature_cols if c in X_ho.columns]]
            pos_tr = int(y_tr.sum())
            neg_tr = int((y_tr == 0).sum())
            spw_fold = float(neg_tr / max(pos_tr, 1))
            log(f"[OPERATOR_FOLDS] fold={fi}  train={len(X_tr)} rows pos={pos_tr} spw={spw_fold:.1f}"
                f"  holdout={len(X_ho)} rows pos={int(y_ho.sum())}")
            fold = (X_tr, y_tr, X_ho, y_ho, spw_fold)
            fold_data.append(fold + (groups_tr, groups_ho) if use_identity else fold)

    positives = int(y_train.sum())
    negatives = int((y_train == 0).sum())
    total     = positives + negatives
    pos_rate  = positives / max(total, 1)
    spw       = float(negatives / max(positives, 1))

    log(f"[DATA] rows={total}  positives={positives} ({pos_rate:.4f})  "
        f"features={len(feature_cols)}  load_s={time.time()-t_load:.1f}")

    # ── cv5-only mode: skip search, load existing results ────────────────────
    if args.cv5_only:
        csv_path = out_dir / "optuna_results.csv"
        if not csv_path.exists():
            raise SystemExit(f"[CV5-ONLY] {csv_path} not found — run search first.")
        log(f"[CV5-ONLY] Loading {csv_path}")
        df_out = pd.read_csv(csv_path).sort_values(
            "cv_mean_auprc", ascending=False, na_position="last")
        meta_path = out_dir / "optuna_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        best_cv1 = df_out.dropna(subset=["cv_mean_auprc"]).iloc[0] if not df_out.empty else None
        best_final = best_cv1
        _run_cv5_and_test(
            df_out, args, spw, X_train, y_train, out_dir, meta,
            use_prebuilt, dataset_path, feature_cols, log, groups=groups_train,
            cfg=cfg, fold_data=fold_data, test_data=test_data)
        return

    seed_trials, n_est_options = _parse_yaml_for_optuna(cfg)

    rows: list = []
    _multivariate = not args.no_multivariate
    log(f"[OPTUNA] cv_folds={args.cv_folds}  time_budget={args.time_budget}s  "
        f"n_startup={args.n_startup}  multivariate={_multivariate}")
    sampler = optuna.samplers.TPESampler(
        multivariate=_multivariate,
        n_startup_trials=args.n_startup,
        seed=args.random_state,
        warn_independent_sampling=False,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)

    active_set = set(active_models)
    enqueued_seeds = [(m, p) for m, p in seed_trials if m in active_set]
    skipped_seeds  = len(seed_trials) - len(enqueued_seeds)
    for _, params in enqueued_seeds:
        study.enqueue_trial(params)
    if seed_trials:
        log(f"[OPTUNA] Enqueued {len(enqueued_seeds)} seed trials from YAML "
            f"({', '.join(sorted({m for m, _ in enqueued_seeds}))})"
            + (f"  [skipped {skipped_seeds} for excluded models]" if skipped_seeds else ""))
    if n_est_options:
        log(f"[OPTUNA] n_estimators choices from YAML: "
            + "  ".join(f"{m}={v}" for m, v in sorted(n_est_options.items())))

    log("[OPTUNA] Starting search ...")
    t_start = time.time()
    _objective = make_objective(
        X_train, y_train, spw, args.cv_folds, args.random_state, rows, out_dir, log, n_est_options,
        time_budget=args.time_budget, feature_cols=feature_cols, active_models=active_models,
        fold_data=fold_data,
        groups=groups_train,
    )
    study.optimize(_objective, timeout=args.time_budget, catch=(Exception,))
    elapsed = time.time() - t_start
    completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)

    log(f"[OPTUNA] {completed} completed trials in {elapsed:.0f}s")

    # Write second-half report (feature importance + HP analysis)
    feat_counts      = getattr(_objective, "feat_counts", {})
    second_half_rows = getattr(_objective, "second_half_rows", [])
    if second_half_rows:
        _write_second_half_report(feat_counts, second_half_rows, out_dir, log)

    if not rows:
        log("[WARN] No results.")
        return

    df_out = pd.DataFrame(rows).sort_values("cv_mean_auprc", ascending=False, na_position="last")
    df_out.to_csv(out_dir / "optuna_results.csv", index=False)
    log(f"[OUT] {out_dir / 'optuna_results.csv'}")

    best_cv1 = df_out.dropna(subset=["cv_mean_auprc"]).iloc[0] if not df_out.empty else None
    if best_cv1 is not None:
        log(f"[BEST cv=1] model={best_cv1['model']}  AUPRC={best_cv1['cv_mean_auprc']:.4f}")

    meta = {
        "completed_trials": completed,
        "elapsed_seconds": round(elapsed, 1),
        "cv_folds": args.cv_folds,
        "n_startup": args.n_startup,
        "n_rows": total, "n_features": len(feature_cols), "pos_rate": round(pos_rate, 6),
        "best_model": str(best_cv1["model"]) if best_cv1 is not None else None,
        "best_val_auprc": float(best_cv1["cv_mean_auprc"]) if best_cv1 is not None else None,
        "Niels_Identity_Confounding_switch": use_identity,
        "identity_scope": "operator_player" if use_identity else None,
    }
    (out_dir / "optuna_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _run_cv5_and_test(
        df_out, args, spw, X_train, y_train, out_dir, meta,
        use_prebuilt, dataset_path, feature_cols, log, groups=groups_train,
        cfg=cfg, fold_data=fold_data, test_data=test_data)


if __name__ == "__main__":
    main()
