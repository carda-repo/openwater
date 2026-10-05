#!/usr/bin/env python3
"""
prepare_hpo_dataset.py — pre-build the combined all-operators dataset for HPO.

Runs via sbatch (not on login node). Reads the effective config, builds the
combined dataset with optional negative undersampling, and saves to pickle.

Usage:
    python prepare_hpo_dataset.py --config /path/to/hpsearch_config_effective.yaml
"""

import os
import sys
import json
import pickle
import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Import helpers from hpsearch_runner.py (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hpsearch_runner import (
    build_all_operators_merged_df,
    make_Xy,
    extract_targets,
    log,
)

from identity_splitting import (
    identity_enabled, person_holdout,
)

ID_COL = "Player_Profile_ID"


def undersample_negatives(X: pd.DataFrame, y: pd.Series, ratio: int, random_state: int = 42):
    """Keep all positives + ratio× as many negatives (1:ratio in output)."""
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index
    n_keep = min(int(len(pos_idx) * ratio), len(neg_idx))
    neg_keep = neg_idx.to_series().sample(n=n_keep, random_state=random_state).index
    keep = pos_idx.union(neg_keep)
    return X.loc[keep].copy(), y.loc[keep].copy()


# NB (organisatie-versie): de body van `main()` is uitgelicht naar `prepare_dataset(cfg)`, zodat de
# pure-Python runner het in-process kan aanroepen (met een cfg-dict) zonder yaml/subprocess.
# `main()` blijft een dunne CLI-wrapper die de yaml-config inleest en `prepare_dataset` aanroept.
def prepare_dataset(cfg: dict):
    dataset_path = Path(cfg.get("dataset_path", ""))
    if not dataset_path:
        raise SystemExit("config missing: dataset_path")

    sampling_ratio = int(cfg.get("sampling_ratio", 0))
    data_dir = Path(cfg.get("data_dir", "")).expanduser()
    validation_period_prefixes = cfg.get("validation_period_prefixes", []) or []
    test_period_prefixes = cfg.get("test_period_prefixes", []) or []
    target_col = cfg.get("target_col", "") or ""
    generic_col_prefixes = [f"p{i}" for i in range(len(validation_period_prefixes))]

    dataset_path.mkdir(parents=True, exist_ok=True)
    log(f"[PREPARE] Building combined dataset → {dataset_path}")
    log(f"[PREPARE] sampling_ratio={sampling_ratio} (0=no sampling)")

    if not cfg.get("all_mode"):
        raise SystemExit("prepare_hpo_dataset.py requires all_mode=True in config")

    df_valid, df_test, meta_valid, meta_test = build_all_operators_merged_df(
        cfg=cfg,
        data_dir=data_dir,
        validation_period_prefixes=validation_period_prefixes,
        test_period_prefixes=test_period_prefixes,
        id_col=ID_COL,
        explicit_target_col=target_col,
        column_prefixes=generic_col_prefixes,
    )

    # Determine target columns
    def get_target(df):
        cols = extract_targets(df)
        if len(cols) != 1:
            raise SystemExit(f"Expected 1 target column, found: {cols}")
        return cols[0]

    tgt_valid = get_target(df_valid)

    filter_active = int(os.environ.get("FILTER_ACTIVE", "1")) == 1
    log(f"[PREPARE] FILTER_ACTIVE={filter_active} (env={os.environ.get('FILTER_ACTIVE', '<not set, default=1>')})")
    _af_cols = [c for c in df_valid.columns if "ACTIVE_FLAG" in c]
    log(f"[PREPARE] ACTIVE_FLAG columns in df_valid: {_af_cols if _af_cols else 'none'}")
    if filter_active and "ACTIVE_FLAG" in df_valid.columns:
        n_before = len(df_valid)
        df_valid = df_valid[df_valid["ACTIVE_FLAG"] != False]  # noqa: E712
        log(f"[PREPARE] ACTIVE_FLAG filter: {n_before - len(df_valid)} inactieve rijen verwijderd ({len(df_valid):,} over)")
    elif filter_active and "ACTIVE_FLAG" not in df_valid.columns:
        log("[PREPARE] ⚠  ACTIVE_FLAG niet gevonden in data — geen filtering toegepast")

    # Build and filter the entire test candidate set before assigning persons.
    # This also covers the operator-holdout path; sampling applies only afterwards.
    holdout_operators = cfg.get("fold_holdout_operators") or []
    if holdout_operators:
        log(f"[PREPARE] Building holdout dataset for operators: {holdout_operators}")
        cfg_holdout = {**cfg, "all_operators": holdout_operators}
        df_test, _, meta_test, _ = build_all_operators_merged_df(
            cfg=cfg_holdout,
            data_dir=data_dir,
            validation_period_prefixes=validation_period_prefixes,
            test_period_prefixes=[],
            id_col=ID_COL,
            explicit_target_col=target_col,
            column_prefixes=generic_col_prefixes,
        )
    if filter_active and "ACTIVE_FLAG" in df_test.columns:
        n_before = len(df_test)
        df_test = df_test[df_test["ACTIVE_FLAG"] != False]  # noqa: E712
        log(f"[PREPARE] ACTIVE_FLAG filter (test): {n_before - len(df_test)} rijen verwijderd")

    enabled = identity_enabled(cfg)
    identity_col = (cfg.get("Niels_identity_column") or ID_COL) if enabled else None
    df_valid, df_test, groups_train, groups_test, identity_meta = person_holdout(
        df_valid, df_test, cfg, id_col=ID_COL,
    )
    if enabled:
        # Every prepared payload carries provenance, so a stale test pickle cannot
        # accidentally be paired with a newly prepared development dataset.
        identity_meta["identity_dataset_id"] = uuid.uuid4().hex
        log(f"[PREPARE] Person separation: {identity_meta['n_persons_development']:,} development, "
            f"{identity_meta['n_persons_test']:,} test persons")

    X_train, y_train, feature_cols = make_Xy(df_valid, ID_COL, tgt_valid, identity_col=identity_col)
    log(f"[PREPARE] Valid: {len(X_train):,} rows, {len(feature_cols)} features, "
        f"{int(y_train.sum()):,} positives ({y_train.mean()*100:.2f}%)")

    if sampling_ratio > 0:
        X_train, y_train = undersample_negatives(X_train, y_train, sampling_ratio)
        if enabled:
            groups_train = groups_train.loc[X_train.index].copy()
        log(f"[PREPARE] After undersampling 1:{sampling_ratio}: "
            f"{len(X_train):,} rows ({int(y_train.sum()):,} pos + {int((y_train==0).sum()):,} neg)")
    if enabled and X_train.empty:
        raise ValueError("Identity-split development dataset is empty after undersampling")

    log("[PREPARE] Saving pickles...")
    train_payload = {"X": X_train, "y": y_train}
    if enabled:
        train_payload.update({"groups": groups_train, "identity_metadata": identity_meta})
    with open(dataset_path / "valid_sampled.pkl", "wb") as f:
        pickle.dump(train_payload, f, protocol=4)

    meta = {
        "n_valid": int(len(X_train)),
        "n_pos_valid": int(y_train.sum()),
        "n_neg_valid": int((y_train == 0).sum()),
        "sampling_ratio": sampling_ratio,
        "n_features": len(feature_cols),
        "feature_cols": list(feature_cols),
        "target_valid": tgt_valid,
        **identity_meta,
    }
    if not df_test.empty:
        tgt_test = get_target(df_test)
        X_test, y_test, _ = make_Xy(df_test, ID_COL, tgt_test, identity_col=identity_col)
        log(f"[PREPARE] Test: {len(X_test):,} rows, {int(y_test.sum()):,} positives")
        test_payload = {"X": X_test, "y": y_test}
        if enabled:
            test_payload.update({"groups": groups_test, "identity_metadata": identity_meta})
        with open(dataset_path / "test_full.pkl", "wb") as f:
            pickle.dump(test_payload, f, protocol=4)
        meta["n_test"] = int(len(X_test))
        meta["n_pos_test"] = int(y_test.sum())
        meta["target_test"] = tgt_test
        if holdout_operators:
            meta["holdout_operators"] = holdout_operators
    else:
        if enabled:
            (dataset_path / "test_full.pkl").unlink(missing_ok=True)
        log("[PREPARE] No test data — skipping test_full.pkl")
    (dataset_path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"[PREPARE] Done: {dataset_path}")
    return dataset_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    prepare_dataset(cfg)


if __name__ == "__main__":
    main()
