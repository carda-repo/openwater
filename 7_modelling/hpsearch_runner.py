#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Iterable

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Skipping features without any observed values")

# Route all Python warnings to stdout so .err is reserved for real errors
# (SLURM's slurmstepd still writes OOM/step errors to .err directly, bypassing this)
def _warn_to_stdout(message, category, filename, lineno, file=None, line=None):
    print(f"{filename}:{lineno}: {category.__name__}: {message}", flush=True)

warnings.showwarning = _warn_to_stdout

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

_merge_module_dir = Path(__file__).resolve().parents[1] / "6_merge_sample"
if str(_merge_module_dir) not in sys.path:
    sys.path.insert(0, str(_merge_module_dir))
from identity_splitting import identity_enabled, identity_groups, person_holdout

# Optional packages
try:
    import yaml
except Exception as e:
    raise SystemExit("PyYAML missing. Install in venv: pip install pyyaml") from e

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from lightgbm import LGBMClassifier

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False


ID_COL_DEFAULT = "Player_Profile_ID"
RANDOM_STATE_DEFAULT = 42
TEST_SIZE_DEFAULT = 0.2
DTYPE_MAPPING = {
    '70_procent_BINGO': 'int16',
    '70_procent_CASINO': 'int16',
    '70_procent_OTHER': 'int16',
    '70_procent_SLOTS': 'int16',
    '70_procent_VIRTUAL_SPORTS': 'int16',
    '70_procent_weddenschappen': 'int16',
    'f0_net_winloss': 'float16',
    'f10_canceled_withdrawals_per_day': 'float16',
    'f11_withdrawals_per_day': 'float16',
    'f12_deposits_per_day': 'float16',
    'f13_canceled_deposits_per_day': 'float16',
    'f14_active_period_span': 'int16',
    'f15_active_day_fraction': 'float16',
    'f16_account_age': 'int16',
    'f17_number_of_bet_countries': 'int16',
    'f18_number_of_bet_sports': 'int16',
    'f19_max_bet_parts': 'int16',
    'f1_active_days': 'float16',
    'f20_dutch_domestic_bets_pct': 'float16',
    'f22_limit_increases': 'int16',
    'f23_limit_decreases': 'int16',
    'f24_payment_method_variety': 'int16',
    'f25_voluntary_suspensions': 'int16',
    'f26_balance_drop_frequency': 'float16',
    'f27_deposits_after_below2_per_day': 'float16',
    'f28_median_seconds_below2_to_deposit': 'int32',
    'f29_sessions_per_day': 'float16',
    'f2_net_loss_per_day': 'float16',
    'f30_avg_interactions_per_session': 'float16',
    'f31_median_rounds_per_session': 'float16',
    'f32_game_types_count': 'int16',
    'f39_dominant_segment_share': 'float16',
    'f3_total_wagered': 'float16',
    'f40_products_per_active_day': 'float16',
    'f41_heavy_play_hours_count': 'float16',
    'f42_morning_interaction_percentage': 'float16',
    'f43_evening_interaction_percentage': 'float16',
    'f44_morning_stakes_percentage': 'float16',
    'f45_evening_stakes_percentage': 'float16',
    'f46_median_seconds_bet_placed_to_resolved': 'int16',
    'f47_median_seconds_session_start_to_period_end': 'int32',
    'f48_percentage_bets_with_cashout': 'float16',
    'f49_percentage_live_bets': 'float16',
    'f4_average_wager_per_day': 'float16',
    'f50_single_bet_percentage': 'float16',
    'f51_median_seconds_loss_to_next_bet': 'float16',
    'f52_big_win_wager_increase_count': 'int16',
    'f53_abs_gradient_wagered_around_median_date': 'float16',
    'f54_post_median_active_days_percentage': 'float16',
    'f55_stake_variance_difference': 'int32',
    'f56_stake_cv_difference': 'float16',
    'f57_longest_daily_streak': 'int16',
    'f58_longest_streak_ratio': 'float16',
    'f59_median_daily_time_off': 'float16',
    'f60_deposit_amount_variability': 'float16',
    'f61_bet_odds_variability': 'float16',
    'f6_age': 'int16',
    'f7_rtp_deviation': 'float16',
    'f8_interactions_per_day': 'float16',
    'f9_big_wins_per_day': 'float16',
    'y_self_exclusion_20250601_20250630': 'int16',
}

def log(msg: str):
    print(msg, flush=True)


def safe_auc(y_true, y_score) -> float:
    if pd.Series(y_true).nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true, y_score) -> float:
    if pd.Series(y_true).nunique() < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def extract_targets(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if isinstance(c, str) and c.startswith("y_self_exclusion_")]


def prefix_feature_cols(df: pd.DataFrame, prefix: str, id_col: str, target_cols: List[str]) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        if c == id_col:
            continue
        if c in target_cols:
            continue
        if c == "ACTIVE_FLAG":
            continue
        if c in {"__niels_person_id", "__niels_operator_id"}:
            continue
        rename[c] = f"{prefix}_{c}"
    return df.rename(columns=rename)


def ensure_single_target_after_merge(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Your _x/_y collapse rule.
    """
    target_candidates = [c for c in df.columns if isinstance(c, str) and c.startswith("y_self_exclusion_")]

    if (
        len(target_candidates) == 2
        and any(c.endswith("_x") for c in target_candidates)
        and any(c.endswith("_y") for c in target_candidates)
    ):
        yx = [c for c in target_candidates if c.endswith("_x")][0]
        yy = [c for c in target_candidates if c.endswith("_y")][0]
        base = yx[:-2]
        df = df.drop(columns=[yy]).rename(columns={yx: base})
        target_candidates = [base]

    if len(target_candidates) != 1:
        raise ValueError(f"bad_target_candidates: {target_candidates}")

    return df, target_candidates[0]


def find_latest_file(op_dir: Path, prefix: str) -> Path:
    candidates = sorted(op_dir.glob(f"{prefix}*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No files matching {op_dir}/{prefix}*.csv")
    return candidates[0]


def _ddmmyyyy_to_yyyymmdd(s: str) -> str:
    """'01082024' → '20240801'"""
    return s[4:8] + s[2:4] + s[0:2]


def _expected_target_col_from_prefix(prefix: str) -> str:
    """Given prefix 'DDMMYYYY_DDMMYYYY_DDMMYYYY_DDMMYYYY_pass_...',
    derive 'y_self_exclusion_YYYYMMDD_YYYYMMDD' from parts[2] and parts[3].
    Returns '' if the prefix doesn't match this format."""
    parts = prefix.split("_")
    if len(parts) < 5:
        return ""
    try:
        y_start = _ddmmyyyy_to_yyyymmdd(parts[2])
        y_end   = _ddmmyyyy_to_yyyymmdd(parts[3])
        return f"y_self_exclusion_{y_start}_{y_end}"
    except Exception:
        return ""


def build_merged_df(
    operator: str,
    data_dir: Path,
    period_prefixes: List[str],
    id_col: str,
    target_source_prefix: str = "",
    explicit_target_col: str = "",
    column_prefixes: Optional[List[str]] = None,
    identity_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Merge CSVs found by period_prefixes into one DataFrame.

    column_prefixes: if given, use these for column naming instead of the
                     period_prefixes themselves. Must be same length.
                     This allows validation and test DataFrames to have
                     matching column names (e.g. ["p0", "p1", "p2"]).
    """
    op_dir = data_dir / operator
    if not op_dir.is_dir():
        raise FileNotFoundError(f"Operator dir missing: {op_dir}")

    if column_prefixes and len(column_prefixes) != len(period_prefixes):
        raise ValueError(f"column_prefixes length ({len(column_prefixes)}) != period_prefixes length ({len(period_prefixes)})")

    if not target_source_prefix:
        target_source_prefix = period_prefixes[-1]

    col_prefixes = column_prefixes or period_prefixes

    files_by_prefix: Dict[str, Path] = {}
    dfs_by_prefix: Dict[str, pd.DataFrame] = {}

    for pref in period_prefixes:
        try:
            p = find_latest_file(op_dir, pref)
        except FileNotFoundError:
            # Fall back to x-only file: {x_start}_{x_end}_{pass}_*_x_merged.csv
            # Period prefixes may have 5 parts (x_x_y_y_pass) with no scenario embedded.
            pparts = pref.split("_")
            if len(pparts) >= 5:
                if len(pparts) >= 6:
                    x_only_pref = f"{pparts[0]}_{pparts[1]}_{pparts[4]}_{'_'.join(pparts[5:])}_x"
                else:
                    # No scenario in prefix — base prefix, find_latest_file will glob
                    x_only_pref = f"{pparts[0]}_{pparts[1]}_{pparts[4]}_"
                p = find_latest_file(op_dir, x_only_pref)  # raises FileNotFoundError if not found
                log(f"[INFO] '{pref}' not found; using x-only fallback: {x_only_pref}")
            else:
                raise
        files_by_prefix[pref] = p
        df = pd.read_csv(p, dtype={id_col: str, identity_col: str} if identity_col else None)
        if id_col not in df.columns:
            raise ValueError(f"Missing id_col '{id_col}' in {p}")
        if identity_col and identity_col == id_col and df[id_col].isna().any():
            raise ValueError(f"Missing person identities in '{id_col}' in {p}")
        df[id_col] = df[id_col].astype(str)
        if identity_col:
            df["__niels_operator_id"] = str(operator)
        if identity_col and identity_col != id_col:
            if identity_col not in df.columns:
                raise ValueError(f"Missing person identity column '{identity_col}' in {p}")
            if df[identity_col].isna().any():
                raise ValueError(f"Missing person identities in '{identity_col}' in {p}")
            # Carry the person key without turning it into a period feature.
            df["__niels_person_id"] = df[identity_col].astype(str)
            if df.groupby(id_col)["__niels_person_id"].nunique().gt(1).any():
                raise ValueError(f"Account IDs map to conflicting persons in {p}")
            df = df.drop(columns=[identity_col])
        before = len(df)
        df = df.drop_duplicates(subset=[id_col], keep="first")
        if len(df) < before:
            log(f"[WARN] dedup {pref}: removed {before - len(df)} duplicate {id_col} rows ({before} → {len(df)})")
        dfs_by_prefix[pref] = df

    # decide which target to keep
    expected_tgt = _expected_target_col_from_prefix(target_source_prefix)

    if explicit_target_col:
        target_keep = explicit_target_col
    else:
        if target_source_prefix not in dfs_by_prefix:
            raise ValueError(f"target_source_prefix '{target_source_prefix}' not in {list(dfs_by_prefix)}")
        src_tcols = extract_targets(dfs_by_prefix[target_source_prefix])
        if expected_tgt and expected_tgt in src_tcols:
            # Happy path: expected Y column present in source file
            target_keep = expected_tgt
        elif expected_tgt and src_tcols:
            # Wrong-period Y present — drop it, fall back to Y-target file
            log(f"[WARN] source prefix '{target_source_prefix}' has Y columns {src_tcols} "
                f"but expected '{expected_tgt}'; will try Y-target file.")
            target_keep = None
        elif not src_tcols:
            # No Y at all in source file — will try Y-target file
            target_keep = None
        else:
            # Prefix format not parseable — old behavior (exactly 1 target required)
            if len(src_tcols) != 1:
                raise ValueError(f"source_prefix '{target_source_prefix}' has {len(src_tcols)} targets: {src_tcols}")
            target_keep = src_tcols[0]

    merged = None
    for i, pref in enumerate(period_prefixes):
        df = dfs_by_prefix[pref]
        cp = col_prefixes[i]
        tcols = extract_targets(df)

        keep_target_here = False
        if target_keep is not None:
            if explicit_target_col:
                keep_target_here = (target_keep in df.columns)
            else:
                keep_target_here = (pref == target_source_prefix)

        drop_targets = [c for c in tcols if (not keep_target_here) or (c != target_keep)]
        df2 = df.drop(columns=drop_targets, errors="ignore")

        tcols2 = extract_targets(df2)
        df2 = prefix_feature_cols(df2, cp, id_col=id_col, target_cols=tcols2)

        if merged is None:
            merged = df2
        else:
            if "__niels_operator_id" in df2.columns:
                df2 = df2.drop(columns=["__niels_operator_id"])
            if "__niels_person_id" in df2.columns:
                mapping = merged[[id_col, "__niels_person_id"]].merge(
                    df2[[id_col, "__niels_person_id"]], on=id_col,
                    suffixes=("_previous", "_current"), how="inner",
                )
                if not mapping["__niels_person_id_previous"].equals(mapping["__niels_person_id_current"]):
                    raise ValueError("Person identities differ across feature periods")
                df2 = df2.drop(columns=["__niels_person_id"])
            merged = pd.merge(
                merged,
                df2,
                on=id_col,
                how="inner",
                suffixes=("_x", "_y"),
            )

    if merged is None or merged.empty:
        raise ValueError("merge_empty")

    # Consolidate ACTIVE_FLAG: inner merge can produce ACTIVE_FLAG_x / ACTIVE_FLAG_y
    _af_variants = [c for c in merged.columns if c == "ACTIVE_FLAG" or c.startswith("ACTIVE_FLAG_")]
    if len(_af_variants) > 1:
        merged["ACTIVE_FLAG"] = merged[_af_variants[0]]
        merged = merged.drop(columns=[c for c in _af_variants if c != "ACTIVE_FLAG"])

    # If ACTIVE_FLAG still absent: look for per-operator lookup file
    if "ACTIVE_FLAG" not in merged.columns:
        _tsparts = target_source_prefix.split("_")
        if len(_tsparts) >= 5:
            _ys = _tsparts[2]  # y-start date DDMMYYYY
            _lookup_path = op_dir / f"LAST_STATUS_LOOKUP_BEFORE_{_ys}.csv"
            if _lookup_path.exists():
                _lookup = pd.read_csv(_lookup_path, usecols=["Player_Profile_ID", "last_status"])
                _lookup["Player_Profile_ID"] = _lookup["Player_Profile_ID"].astype(str)
                _lookup["ACTIVE_FLAG"] = _lookup["last_status"].str.upper().eq("ACTIVE")
                merged = merged.merge(_lookup[["Player_Profile_ID", "ACTIVE_FLAG"]], on=id_col, how="left")
                merged["ACTIVE_FLAG"] = merged["ACTIVE_FLAG"].fillna(False)
                log(f"[INFO] ACTIVE_FLAG from lookup {_lookup_path.name} ({merged['ACTIVE_FLAG'].sum():,} active / {len(merged):,} total)")
            else:
                log(f"[WARN] No ACTIVE_FLAG in merged data and no lookup file found: {_lookup_path}")

    # Drop duplicate column names (keeps first occurrence).
    # pandas can silently produce duplicate cols when a CSV has them; LightGBM fatal-errors on them.
    dup_cols = merged.columns[merged.columns.duplicated()].tolist()
    if dup_cols:
        log(f"[WARN] Dropping {len(dup_cols)} duplicate column(s) after merge: {dup_cols[:10]}")
        merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]

    if target_keep is None:
        # No Y in any period file — look for Y-target or y-only fallback file
        _tsparts = target_source_prefix.split("_")
        period_5 = "_".join(_tsparts[:5])  # x_x_y_y_pass
        ytarget_pref = f"{period_5}_Y-target"
        # y-only file produced by merge_features.py: {y_start}_{y_end}_{pass}_y
        y_only_pref = f"{_tsparts[2]}_{_tsparts[3]}_{_tsparts[4]}_y" if len(_tsparts) >= 5 else ""

        yf = None
        for _try_pref in ([ytarget_pref] + ([y_only_pref] if y_only_pref else [])):
            try:
                yf = find_latest_file(op_dir, _try_pref)
                break
            except FileNotFoundError:
                continue
        if yf is None:
            tried = f"'{ytarget_pref}*.csv'" + (f" and '{y_only_pref}*.csv'" if y_only_pref else "")
            raise ValueError(
                f"No Y target in merged files and no Y-target/y-only file found for prefix "
                f"'{target_source_prefix}' (tried {tried} in {op_dir})"
            )
        y_df = pd.read_csv(yf)
        y_df[id_col] = y_df[id_col].astype(str)
        y_tgts = extract_targets(y_df)
        if not y_tgts:
            raise ValueError(f"Y-target file '{yf}' contains no y_self_exclusion_* columns")
        if expected_tgt and expected_tgt not in y_tgts:
            raise ValueError(
                f"Expected target '{expected_tgt}' not found in Y-target file '{yf}'; found {y_tgts}"
            )
        target_col = expected_tgt or y_tgts[0]
        merged = merged.merge(y_df[[id_col, target_col]], on=id_col, how="left")
    else:
        # Normal path: resolve any _x/_y suffixes introduced by the period merge
        if explicit_target_col and explicit_target_col not in merged.columns:
            merged, resolved = ensure_single_target_after_merge(merged)
            if resolved != explicit_target_col:
                raise ValueError(f"Explicit target_col='{explicit_target_col}' not found; resolved to '{resolved}' instead")
            target_col = resolved
        else:
            merged, target_col = ensure_single_target_after_merge(merged)

    meta = {
        "operator": operator,
        "data_dir": str(data_dir),
        "id_col": id_col,
        "identity_column": identity_col or id_col,
        "period_prefixes": period_prefixes,
        "column_prefixes": col_prefixes,
        "files": {k: str(v) for k, v in files_by_prefix.items()},
        "target_source_prefix": target_source_prefix,
        "explicit_target_col": explicit_target_col or "",
        "target_col": target_col,
        "n_rows": int(len(merged)),
        "n_cols": int(merged.shape[1]),
    }
    return merged, meta


def _load_all_scenario_stats(
    op_dir: Path,
    period_prefixes: List[str],
    all_scenario_name: str,
    id_col: str,
    column_prefixes: Optional[List[str]],
    identity_col: Optional[str] = None,
) -> Dict[str, float]:
    """
    Load ALL scenario CSV(s) for one operator (1 row each — pre-aggregated operator stats).
    Returns a flat dict of {prefixed_col: value} to broadcast to all player rows.
    """
    col_prefixes = column_prefixes or period_prefixes
    stats: Dict[str, float] = {}
    for i, pref in enumerate(period_prefixes):
        p = find_latest_file(op_dir, f"{pref}_{all_scenario_name}")
        df = pd.read_csv(p)
        if len(df) != 1:
            log(f"[WARN] ALL scenario CSV {p} has {len(df)} rows, expected 1. Using first row.")
        row = df.iloc[0]
        cp = col_prefixes[i]
        for col in df.columns:
            if col in {id_col, identity_col, "__niels_person_id", "__niels_operator_id"}:
                continue
            val = row[col]
            stats[f"{cp}_{col}"] = float(val) if pd.notna(val) else float("nan")
    return stats


def build_all_operators_merged_df(
    cfg: Dict,
    data_dir: Path,
    validation_period_prefixes: List[str],
    test_period_prefixes: List[str],
    id_col: str,
    explicit_target_col: str,
    column_prefixes: Optional[List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """
    Build combined (all-operators) validation and test DataFrames.
    For each operator:
      - loads base scenario CSVs (full player features + target)
      - loads ALL scenario 1-row CSVs (pre-aggregated operator stats)
      - broadcasts those stats as constant columns to all player rows
      - adds 26-column OHE
    Concatenates all operators.
    """
    all_operators: List[str] = cfg.get("all_operators") or []
    base_scenario: str = cfg.get("base_scenario", "Flexible_spanish_plus")
    all_scenario_name: str = cfg.get("all_scenario_name", "ALL")

    if not all_operators:
        raise ValueError("all_mode=True but config missing all_operators")

    base_valid_prefixes = [f"{p}_{base_scenario}" for p in validation_period_prefixes]
    base_test_prefixes = [f"{p}_{base_scenario}" for p in test_period_prefixes]

    dfs_valid: List[pd.DataFrame] = []
    dfs_test: List[pd.DataFrame] = []
    last_meta_valid: Dict = {}
    last_meta_test: Dict = {}

    for op in all_operators:
        op_dir = data_dir / op
        if not op_dir.is_dir():
            log(f"[ALL] Skipping operator {op}: directory not found")
            continue

        try:
            df_valid_base, meta_valid = build_merged_df(
                operator=op, data_dir=data_dir,
                period_prefixes=base_valid_prefixes,
                id_col=id_col, target_source_prefix="",
                explicit_target_col=explicit_target_col,
                column_prefixes=column_prefixes,
                identity_col=(cfg.get("Niels_identity_column") or id_col) if identity_enabled(cfg) else None,
            )
            if base_test_prefixes:
                df_test_base, meta_test = build_merged_df(
                    operator=op, data_dir=data_dir,
                    period_prefixes=base_test_prefixes,
                    id_col=id_col, target_source_prefix="",
                    explicit_target_col=explicit_target_col,
                    column_prefixes=column_prefixes,
                    identity_col=(cfg.get("Niels_identity_column") or id_col) if identity_enabled(cfg) else None,
                )
            else:
                df_test_base, meta_test = pd.DataFrame(), meta_valid
        except Exception as e:
            if identity_enabled(cfg):
                raise
            log(f"[ALL] Skipping operator {op}: base scenario load failed: {e}")
            continue

        # Load pre-aggregated ALL stats (1 row per period) and broadcast to all player rows
        # Leakage guard: validation stats applied to test set (not test stats)
        try:
            op_stats = _load_all_scenario_stats(
                op_dir, validation_period_prefixes, all_scenario_name, id_col, column_prefixes,
                identity_col=(cfg.get("Niels_identity_column") or id_col) if identity_enabled(cfg) else None,
            )
            for col, val in op_stats.items():
                df_valid_base[col] = val
                if not df_test_base.empty:
                    df_test_base[col] = val
            log(f"[ALL] operator={op}: {len(op_stats)} op-stat columns added")
        except Exception as e:
            log(f"[ALL] operator={op}: ALL scenario not available, skipping op stats: {e}")

        # 26-column OHE (a-z) — uitschakelbaar via cfg["add_operator_ohe"]=False (organisatie-uitbreiding;
        # default True = origineel gedrag).
        if cfg.get("add_operator_ohe", True):
            for letter in "abcdefghijklmnopqrstuvwxyz":
                df_valid_base[f"ohe_{letter}"] = int(op == letter)
                if not df_test_base.empty:
                    df_test_base[f"ohe_{letter}"] = int(op == letter)

        dfs_valid.append(df_valid_base)
        if not df_test_base.empty:
            dfs_test.append(df_test_base)
        last_meta_valid = meta_valid
        last_meta_test = meta_test

    if not dfs_valid:
        raise ValueError("build_all_operators_merged_df: no operators produced data")

    combined_valid = pd.concat(dfs_valid, ignore_index=True)
    combined_test = pd.concat(dfs_test, ignore_index=True) if dfs_test else pd.DataFrame()
    log(f"[ALL] Combined: valid={len(combined_valid)} rows ({combined_valid.shape[1]} cols), test={len(combined_test)} rows, operators={len(dfs_valid)}")

    meta_valid_combined = {**last_meta_valid, "all_operators": all_operators, "n_rows": int(len(combined_valid))}
    meta_test_combined = {**last_meta_test, "n_rows": int(len(combined_test))}
    return combined_valid, combined_test, meta_valid_combined, meta_test_combined


def make_Xy(df: pd.DataFrame, id_col: str, target_col: str, identity_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    seen = set()
    feature_cols = []
    for c in df.columns:
        if (c not in {id_col, target_col, identity_col, "__niels_person_id", "__niels_operator_id"}
                and not (isinstance(c, str) and "__niels_" in c)
                and not (identity_col and isinstance(c, str) and c.endswith("_" + identity_col))
                and not (isinstance(c, str) and c.startswith("y_self_exclusion_"))
                and c != "ACTIVE_FLAG"
                and pd.api.types.is_numeric_dtype(df[c])
                and c not in seen):
            seen.add(c)
            feature_cols.append(c)
    if not feature_cols:
        raise ValueError("no_numeric_features")

    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    return X, y, feature_cols


def build_estimator(model_name: str, spw: float, random_state: int):
    if model_name == "decision_tree":
        return DecisionTreeClassifier(random_state=random_state)

    if model_name == "random_forest":
        return RandomForestClassifier(random_state=random_state, n_jobs=-1)

    if model_name == "extra_trees":
        return ExtraTreesClassifier(random_state=random_state, n_jobs=-1)

    if model_name == "bagging_tree":
        base = DecisionTreeClassifier(random_state=random_state)
        return BaggingClassifier(estimator=base, random_state=random_state, n_jobs=-1)

    if model_name == "hist_gb":
        return HistGradientBoostingClassifier(random_state=random_state)

    if model_name == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost not available")
        return XGBClassifier(random_state=random_state, n_jobs=-1, eval_metric="logloss")

    if model_name == "xgboost_focal":
        if not HAS_XGB:
            raise RuntimeError("xgboost not available")
        # objective (focal loss closure) is applied in the trial loop after focal_alpha/focal_gamma are known
        return XGBClassifier(random_state=random_state, n_jobs=-1, eval_metric="logloss")

    if model_name == "adaboost":
        return AdaBoostClassifier(random_state=random_state)

    if model_name == "lightgbm":
        return LGBMClassifier(random_state=random_state, n_jobs=-1, verbosity=-1)

    if model_name == "sgd_logloss":
        return SGDClassifier(loss="log_loss", penalty="l2", random_state=random_state)

    if model_name == "mlp":
        return MLPClassifier(activation="relu", early_stopping=True, random_state=random_state)

    if model_name == "linear_svc_cal":
        base = LinearSVC(class_weight="balanced", random_state=random_state, max_iter=20000)
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)

    raise ValueError(f"Unknown model: {model_name}")


def build_preprocess_steps(run_variant: Dict[str, Any]) -> List[Tuple[str, Any]]:
    prep = (run_variant or {}).get("preprocessing", {}) or {}
    imputer = prep.get("imputer", "median")
    scaler = prep.get("scaler", "standard")

    steps: List[Tuple[str, Any]] = []

    if imputer and imputer != "none":
        if imputer not in ("median", "mean"):
            raise ValueError(f"Unknown imputer: {imputer}")
        steps.append(("imputer", SimpleImputer(strategy=imputer)))

    if scaler and scaler != "none":
        if scaler != "standard":
            raise ValueError(f"Unknown scaler: {scaler}")
        steps.append(("var_thresh", VarianceThreshold(threshold=0.0)))
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))

    return steps


def focal_binary_objective(alpha: float = 0.25, gamma: float = 2.0):
    """Binary focal loss for XGBoost (custom objective, replaces binary:logistic).

    Handles class imbalance via `alpha` (weight for the positive class).
    `gamma` controls focusing strength: 0 = standard cross-entropy, 2 = original paper.

    Returns a closure compatible with XGBoost's `obj` parameter.
    Note: predict_proba() still applies sigmoid correctly on XGBoost >= 1.6.
    """
    def _obj(preds, dtrain):
        labels = dtrain.get_label() if hasattr(dtrain, "get_label") else dtrain
        p = 1.0 / (1.0 + np.exp(-preds))   # sigmoid of raw logits
        eps = 1e-7
        at  = np.where(labels == 1, alpha, 1.0 - alpha)   # per-sample class weight
        pt  = np.where(labels == 1, p,     1.0 - p)       # prob of the true class
        # gradient: dFL/dx  (negative for positives, positive for negatives)
        sign = np.where(labels == 1, -1.0, 1.0)
        g = sign * at * pt ** gamma * (1.0 - pt) * (gamma * np.log(pt + eps) + 1.0)
        # hessian: diagonal approximation using p*(1-p) (same as binary cross-entropy)
        h = np.maximum(p * (1.0 - p), eps)
        return g, h
    return _obj


MAX_SCALE_POS_WEIGHT = 50.0


def apply_imbalance_to_params(
    params: Dict[str, Any],
    model_name: str,
    imbalance: Dict[str, Any],
    spw: float,
) -> Dict[str, Any]:
    out = dict(params or {})
    strategy = (imbalance or {}).get("strategy", "none")

    if model_name in ("xgboost", "lightgbm"):
        if "scale_pos_weight" not in out:
            effective_spw = min(float(spw), MAX_SCALE_POS_WEIGHT)
            out["scale_pos_weight"] = effective_spw
        return out

    if model_name == "xgboost_focal":
        # focal_alpha already handles class imbalance; do not inject scale_pos_weight
        return out

    if strategy == "class_weight":
        cw = (imbalance or {}).get("class_weight", "balanced")
        if model_name in ("decision_tree", "random_forest", "extra_trees", "sgd_logloss", "mlp"):
            out.setdefault("class_weight", cw)

    return out


def build_pipeline(
    preprocess_steps: List[Tuple[str, Any]],
    imbalance: Dict[str, Any],
    model_name: str,
    estimator,
) -> Any:
    strategy = (imbalance or {}).get("strategy", "none")

    steps = []
    if preprocess_steps:
        for name, obj in preprocess_steps:
            steps.append((name, obj))

    if strategy == "smote":
        if not HAS_IMBLEARN:
            raise RuntimeError("SMOTE requested but imbalanced-learn not available (pip install imbalanced-learn)")
        sampling_strategy = (imbalance or {}).get("sampling_strategy", 0.2)
        k_neighbors = (imbalance or {}).get("k_neighbors", 5)
        steps.append(("smote", SMOTE(sampling_strategy=sampling_strategy, k_neighbors=int(k_neighbors), random_state=42)))
        steps.append(("model", estimator))
        return ImbPipeline(steps=steps)

    steps.append(("model", estimator))
    return Pipeline(steps=steps)


EARLY_STOPPING_MODELS = {"xgboost", "xgboost_focal", "hist_gb", "lightgbm"}


def apply_early_stopping(est, model_name: str, early_stopping_cfg: Dict[str, Any]):
    """Apply early stopping params to estimator based on model type."""
    if not early_stopping_cfg or not early_stopping_cfg.get("enabled"):
        return
    patience = early_stopping_cfg.get("patience", 50)
    fraction = early_stopping_cfg.get("validation_fraction", 0.15)

    if model_name in ("xgboost", "xgboost_focal"):
        est.set_params(early_stopping_rounds=patience)
    elif model_name == "lightgbm":
        est.set_params(early_stopping_round=patience)
    elif model_name == "hist_gb":
        est.set_params(
            early_stopping=True,
            n_iter_no_change=patience,
            validation_fraction=fraction,
        )


def _fit_pipe_with_eval_set(pipe, X_train, y_train, X_eval, y_eval, model_name: str):
    """Fit pipeline manually so eval_set goes through same preprocessing as training data."""
    steps = list(pipe.steps)
    model_step_name, model = steps[-1]

    # Fit and transform through preprocessing steps
    X_tr = X_train
    X_ev = X_eval
    for name, step in steps[:-1]:
        if name == "smote":
            # SMOTE only on training data
            X_tr, y_train = step.fit_resample(X_tr, y_train)
        else:
            step.fit(X_tr, y_train)
            X_tr = step.transform(X_tr)
            X_ev = step.transform(X_ev)

    # Fit model with eval_set
    if model_name in ("xgboost", "xgboost_focal"):
        model.fit(X_tr, y_train, eval_set=[(X_ev, y_eval)], verbose=False)
    elif model_name == "lightgbm":
        model.fit(X_tr, y_train, eval_set=[(X_ev, y_eval)])
    else:
        model.fit(X_tr, y_train)


def _score_pipe(pipe, X_test):
    """Get prediction scores from a fitted pipeline."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X_test)[:, 1]
    elif hasattr(pipe, "decision_function"):
        return pipe.decision_function(X_test)
    else:
        return pipe.predict(X_test)


def aligned_person_groups(groups, X, y=None):
    """Validate group alignment before any positional model split."""
    if groups is None:
        return None
    if len(groups) != len(X) or (y is not None and len(y) != len(X)):
        raise ValueError("Person groups must have one entry for every model row")
    if isinstance(groups, pd.Series):
        if not groups.index.equals(X.index):
            raise ValueError("Person groups are not aligned with model row indexes")
        result = groups.copy()
    else:
        result = pd.Series(groups, index=X.index)
    if y is not None and isinstance(y, pd.Series) and not y.index.equals(X.index):
        raise ValueError("Target indexes are not aligned with model rows")
    normalized = result.astype("string").str.strip()
    if (normalized.isna() | normalized.str.lower().isin({"", "nan", "none", "null", "<na>"})).any():
        raise ValueError("Every model row requires a non-empty person group")
    return normalized.astype(str)


def person_cv_indices(X, y, groups, n_folds=5, random_state=42, test_size=0.2):
    groups = aligned_person_groups(groups, X, y)
    required = 2 if n_folds == 1 else n_folds
    if groups.nunique() < required:
        raise ValueError(f"Person validation requires at least {required} distinct persons; got {groups.nunique()}")
    if n_folds == 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    else:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, val_idx in splitter.split(X, y, groups):
        if set(groups.iloc[train_idx]).intersection(groups.iloc[val_idx]):
            raise ValueError("A person occurs in both training and validation")
        yield train_idx, val_idx


def fit_group_safe(pipe, X, y, groups=None):
    """Prevent estimators from creating a hidden row-wise validation split."""
    groups = aligned_person_groups(groups, X, y)
    model = pipe.steps[-1][1]
    if groups is not None:
        if isinstance(model, CalibratedClassifierCV):
            if any(name == "smote" for name, _ in pipe.steps[:-1]):
                raise ValueError("Person-group calibration cannot be combined with SMOTE; use none or class_weight")
            # Calibration is itself a validation operation and must share the
            # same person boundary as the outer model evaluation.
            model.set_params(cv=list(person_cv_indices(X, y, groups, n_folds=3)))
        elif isinstance(model, (MLPClassifier, HistGradientBoostingClassifier, SGDClassifier)):
            # These estimators cannot accept groups for their internal holdout.
            # Train without that holdout rather than reintroducing identity overlap.
            model.set_params(early_stopping=False)
    pipe.fit(X, y)


def validate_prebuilt_identity(payload, metadata, cfg, id_col=ID_COL_DEFAULT):
    """Reject cached datasets built using another identity policy."""
    enabled = identity_enabled(cfg)
    stored = metadata.get("Niels_Identity_Confounding_switch", False)
    if not isinstance(stored, bool) or stored != enabled:
        raise ValueError("Pre-built dataset identity switch differs from this run; rebuild the dataset")
    if not enabled:
        return None
    identity_col = cfg.get("Niels_identity_column") or id_col
    if (metadata.get("identity_column") != identity_col
            or metadata.get("identity_scope") != "operator_player"
            or metadata.get("identity_split_random_state") != int(cfg.get("random_state", 23))):
        raise ValueError("Pre-built dataset person key or split seed differs from this run; rebuild the dataset")
    if "groups" not in payload:
        raise ValueError("Pre-built dataset has no person groups; rebuild the dataset")
    payload_meta = payload.get("identity_metadata", {})
    manifest_keys = ("Niels_Identity_Confounding_switch", "identity_column",
                     "identity_scope", "identity_split_random_state", "identity_dataset_id", "identity_holdout_applied")
    if not metadata.get("identity_dataset_id") or any(payload_meta.get(k) != metadata.get(k) for k in manifest_keys):
        raise ValueError("Pre-built dataset identity metadata differs from its manifest; rebuild the dataset")
    X, y = payload["X"], payload["y"]
    forbidden = {id_col, identity_col, "__niels_person_id", "__niels_operator_id"}
    if any(c in forbidden or (isinstance(c, str) and ("__niels_" in c or c.endswith("_" + identity_col))) for c in X.columns):
        raise ValueError("Pre-built dataset includes person identifiers as model features; rebuild the dataset")
    return aligned_person_groups(payload["groups"], X, y)


def fit_and_eval(
    pipe, X_train, y_train, X_test, y_test,
    model_name: str = "", early_stopping_cfg: Optional[Dict[str, Any]] = None,
    groups=None,
) -> Tuple[float, float, float]:
    t0 = time.time()

    needs_eval_set = (
        early_stopping_cfg and early_stopping_cfg.get("enabled")
        and model_name in ("xgboost", "xgboost_focal", "lightgbm")
    )

    if needs_eval_set:
        # Split 15% from training data for early stopping eval_set
        fraction = early_stopping_cfg.get("validation_fraction", 0.15)
        if groups is not None:
            train_idx, eval_idx = next(person_cv_indices(X_train, y_train, groups, n_folds=1, test_size=fraction))
            X_tr, X_ev = X_train.iloc[train_idx], X_train.iloc[eval_idx]
            y_tr, y_ev = y_train.iloc[train_idx], y_train.iloc[eval_idx]
        else:
            X_tr, X_ev, y_tr, y_ev = train_test_split(
                X_train, y_train, test_size=fraction, random_state=42,
                stratify=y_train if y_train.nunique() >= 2 else None,
            )
        _fit_pipe_with_eval_set(pipe, X_tr, y_tr, X_ev, y_ev, model_name)
    else:
        fit_group_safe(pipe, X_train, y_train, groups)

    fit_s = time.time() - t0

    # Align test columns to training columns — handles mismatches in both directions:
    # - features in train but not test: filled with NaN (imputer substitutes median)
    # - features in test but not train: dropped (never seen at fit time → sklearn error)
    missing_cols = set(X_train.columns) - set(X_test.columns)
    extra_cols = set(X_test.columns) - set(X_train.columns)
    if missing_cols or extra_cols:
        if missing_cols:
            log(f"[WARN] {len(missing_cols)} feature(s) missing from test set, filling with NaN: {sorted(missing_cols)}")
        if extra_cols:
            log(f"[WARN] {len(extra_cols)} feature(s) in test set not seen at fit time, dropping: {sorted(extra_cols)}")
        X_test = X_test.reindex(columns=X_train.columns)

    y_score = _score_pipe(pipe, X_test)
    return safe_auc(y_test, y_score), safe_auprc(y_test, y_score), fit_s


def cv_score(
    pipe, X, y, n_folds=5, random_state=42,
    model_name: str = "", early_stopping_cfg: Optional[Dict[str, Any]] = None,
    fitted_out: list = None,
    groups=None,
) -> Tuple[float, float, float, float, float]:
    """Return (mean_auprc, std_auprc, mean_auc, std_auc, total_fit_seconds).

    Supplied groups use StratifiedGroupKFold, or an 80/20 GroupShuffleSplit for
    n_folds=1. Without groups, retain the original stratified row-wise splits.
    fitted_out receives the last fitted fold pipeline without additional fitting.
    """
    needs_eval_set = (
        early_stopping_cfg and early_stopping_cfg.get("enabled")
        and model_name in ("xgboost", "xgboost_focal", "lightgbm")
    )

    if groups is not None:
        groups = aligned_person_groups(groups, X, y)
        aucs, auprcs = [], []
        t0 = time.time()
        last_fold_pipe = None
        for train_idx, val_idx in person_cv_indices(X, y, groups, n_folds, random_state):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            fold_pipe = clone(pipe)
            if needs_eval_set:
                _fit_pipe_with_eval_set(fold_pipe, X_tr, y_tr, X_val, y_val, model_name)
            else:
                fit_group_safe(fold_pipe, X_tr, y_tr, groups.iloc[train_idx])
            last_fold_pipe = fold_pipe
            y_score = _score_pipe(fold_pipe, X_val)
            aucs.append(safe_auc(y_val, y_score))
            auprcs.append(safe_auprc(y_val, y_score))
        if fitted_out is not None and last_fold_pipe is not None:
            fitted_out.append(last_fold_pipe)
        return (float(np.nanmean(auprcs)), float(np.nanstd(auprcs)),
                float(np.nanmean(aucs)), float(np.nanstd(aucs)), time.time() - t0)

    if n_folds == 1:
        stratify = y if y.nunique() >= 2 else None
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=stratify
        )
        fold_pipe = clone(pipe)
        t0 = time.time()
        if needs_eval_set:
            _fit_pipe_with_eval_set(fold_pipe, X_tr, y_tr, X_val, y_val, model_name)
        else:
            fold_pipe.fit(X_tr, y_tr)
        if fitted_out is not None:
            fitted_out.append(fold_pipe)
        y_score = _score_pipe(fold_pipe, X_val)
        fit_s = time.time() - t0
        auc = safe_auc(y_val, y_score)
        pr  = safe_auprc(y_val, y_score)
        return float(pr), 0.0, float(auc), 0.0, fit_s

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    aucs, auprcs = [], []
    t0 = time.time()
    last_fold_pipe = None
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_pipe = clone(pipe)

        if needs_eval_set:
            _fit_pipe_with_eval_set(fold_pipe, X_tr, y_tr, X_val, y_val, model_name)
        else:
            fold_pipe.fit(X_tr, y_tr)

        last_fold_pipe = fold_pipe
        y_score = _score_pipe(fold_pipe, X_val)
        aucs.append(safe_auc(y_val, y_score))
        auprcs.append(safe_auprc(y_val, y_score))

    if fitted_out is not None and last_fold_pipe is not None:
        fitted_out.append(last_fold_pipe)
    fit_s = time.time() - t0
    return float(np.nanmean(auprcs)), float(np.nanstd(auprcs)), float(np.nanmean(aucs)), float(np.nanstd(aucs)), fit_s


def get_feature_names_after_pipeline(pipe, feature_cols):
    names = list(feature_cols)
    steps = list(pipe.steps)[:-1]  # skip the final model step
    for _, step in steps:
        if hasattr(step, "get_support"):
            mask = step.get_support()
            names = [n for n, keep in zip(names, mask) if keep]
    return names


def try_get_feature_importances(
    final_estimator,
    feature_names: List[str],
    X_test=None,
    y_test=None,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Optional[pd.DataFrame]:
    est = final_estimator

    # BaggingClassifier: no feature_importances_ on the ensemble itself;
    # average across the fitted sub-estimators in estimators_.
    if hasattr(est, "estimators_") and not hasattr(est, "feature_importances_"):
        imps = [
            np.asarray(e.feature_importances_, dtype=float)
            for e in est.estimators_
            if hasattr(e, "feature_importances_")
        ]
        if not imps:
            return None
        imp = np.mean(imps, axis=0)
        if imp.size == len(feature_names):
            return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)
        return None

    # CalibratedClassifierCV: unwrap to the inner estimator.
    # Guard with "not hasattr(est, 'estimators_')" so AdaBoost/RandomForest are
    # NOT unwrapped — they expose feature_importances_ directly on themselves.
    if not hasattr(est, "feature_importances_") and not hasattr(est, "coef_"):
        if hasattr(est, "estimator") and not hasattr(est, "estimators_"):
            est = est.estimator

    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
        if imp.size == len(feature_names):
            return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)

    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float).reshape(-1)
        if coef.size == len(feature_names):
            return pd.DataFrame({"feature": feature_names, "importance": np.abs(coef)}).sort_values("importance", ascending=False)

    # Fallback: permutation importance (for models like hist_gb with no native importances)
    if X_test is not None and y_test is not None:
        try:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(
                est,
                X_test,
                y_test,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=-1,
                scoring="average_precision",
            )
            imp = result.importances_mean
            if imp.size == len(feature_names) and np.isfinite(imp).any():
                return pd.DataFrame({
                    "feature": feature_names,
                    "importance": imp,
                    "importance_std": result.importances_std,
                }).sort_values("importance", ascending=False)
        except Exception:
            pass

    return None


# ---------------------------
# NEW: grid expansion helpers
# ---------------------------
def _coerce_yaml_value(v: Any) -> Any:
    """
    For YAML lists that should be tuples (mlp hidden_layer_sizes etc.).
    Keep this conservative: only fix well-known cases later, but support nested lists in cartesian too.
    """
    return v


def _cartesian_products(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    # params: {"a":[1,2], "b":[3,4]} => 4 dicts
    keys = list(params.keys())
    values_lists: List[List[Any]] = []
    for k in keys:
        vals = params[k]
        if isinstance(vals, list):
            values_lists.append([_coerce_yaml_value(x) for x in vals])
        else:
            # allow singletons as shorthand
            values_lists.append([_coerce_yaml_value(vals)])

    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*values_lists):
        d = {k: v for k, v in zip(keys, combo)}
        out.append(d)
    return out


def expand_grid_spec(grid_spec: Any) -> List[Dict[str, Any]]:
    """
    Accepts:
      - list[dict]  (current behavior)
      - dict with:
          mode: cartesian
          params: {param: [values...], ...}
          fixed: {param: value, ...}   (optional)
        OR:
          mode: list
          trials: [ {..}, {..} ]
    Returns list[dict] of trial params.
    """
    if isinstance(grid_spec, list):
        # legacy explicit trials
        if not all(isinstance(x, dict) for x in grid_spec):
            raise ValueError("grid spec is a list but contains non-dict entries")
        return [dict(x) for x in grid_spec]

    if isinstance(grid_spec, dict):
        mode = (grid_spec.get("mode") or "list").lower()

        if mode == "list":
            trials = grid_spec.get("trials")
            if trials is None:
                # allow dict-as-single-trial convenience, but be strict:
                # if user wrote a dict without 'mode'/'trials'/'params', treat as single trial
                if any(k in grid_spec for k in ("params", "fixed")):
                    raise ValueError("grid mode=list expects 'trials', but found params/fixed keys")
                return [dict(grid_spec)]
            if not isinstance(trials, list) or not all(isinstance(x, dict) for x in trials):
                raise ValueError("grid mode=list requires trials: list[dict]")
            return [dict(x) for x in trials]

        if mode == "cartesian":
            params = grid_spec.get("params") or {}
            fixed = grid_spec.get("fixed") or {}
            if not isinstance(params, dict) or not params:
                raise ValueError("grid mode=cartesian requires non-empty params: dict")
            if not isinstance(fixed, dict):
                raise ValueError("grid mode=cartesian fixed must be dict (or omitted)")

            combos = _cartesian_products(params)
            out = []
            for c in combos:
                merged = dict(fixed)
                merged.update(c)
                out.append(merged)
            return out

        raise ValueError(f"Unknown grid mode: {mode}")

    raise ValueError(f"Unsupported grid spec type: {type(grid_spec)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--taskmap", required=False, default="")
    ap.add_argument("--operator", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-variant", required=True)
    ap.add_argument("--grid-name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--trial-idx", type=int, default=-1,
                    help="Run only this trial index (0-based). -1 = run all within time budget.")
    ap.add_argument("--round", action="store_true",
                    help="Apply DTYPE_MAPPING to X_train/X_test after loading (reduces memory).")
    ap.add_argument("--cv-folds", type=int, default=5,
                    help="Number of CV folds. Use 1 for a single 80/20 holdout (faster, less memory).")
    ap.add_argument("--valid", dest="run_valid", action="store_true", default=None,
                    help="Run validation CV. If neither --valid nor --test given, both run (backward compat).")
    ap.add_argument("--no-valid", dest="run_valid", action="store_false")
    ap.add_argument("--test",  dest="run_test",  action="store_true", default=None,
                    help="Run test evaluation (refit best on full valid set, evaluate on test).")
    ap.add_argument("--no-test",  dest="run_test",  action="store_false")

    ap.add_argument("--id-col", default=ID_COL_DEFAULT)
    ap.add_argument("--random-state", type=int, default=RANDOM_STATE_DEFAULT)
    ap.add_argument("--test-size", type=float, default=TEST_SIZE_DEFAULT)

    args = ap.parse_args(argv)

    filter_active = int(os.environ.get("FILTER_ACTIVE", "1")) == 1

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    use_identity = identity_enabled(cfg)
    identity_col = (cfg.get("Niels_identity_column") or args.id_col) if use_identity else None
    groups_train = groups_test = None
    valid_only_cache = (use_identity and args.run_test is False and bool(cfg.get("dataset_path"))
                        and (Path(cfg["dataset_path"]) / "valid_sampled.pkl").exists())

    data_dir = Path(cfg.get("data_dir", "")).expanduser()
    if not data_dir:
        raise SystemExit("config missing: data_dir")

    test_period_prefixes = cfg.get("test_period_prefixes", []) or []
    if not test_period_prefixes and not valid_only_cache:
        raise SystemExit("config missing: test_period_prefixes")

    validation_period_prefixes = cfg.get("validation_period_prefixes", []) or []
    has_validation = bool(validation_period_prefixes)
    if has_validation and not valid_only_cache and len(validation_period_prefixes) != len(test_period_prefixes):
        raise SystemExit(
            f"validation_period_prefixes ({len(validation_period_prefixes)}) must have "
            f"same length as test_period_prefixes ({len(test_period_prefixes)})"
        )

    # Generic column prefixes so validation and test DataFrames have matching column names
    generic_col_prefixes = [f"p{i}" for i in range(len(test_period_prefixes))]

    target_source_prefix = cfg.get("target_source_prefix", "") or ""
    target_col = cfg.get("target_col", "") or ""

    time_budget = int(cfg.get("time_budget_seconds", 60))

    run_variants = cfg.get("run_variants", []) or []
    rv = None
    for x in run_variants:
        if (x or {}).get("name") == args.run_variant:
            rv = x
            break
    if rv is None:
        raise SystemExit(f"run_variant '{args.run_variant}' not found in config")

    only_models = rv.get("only_models")
    if only_models and args.model not in only_models:
        log(f"[SKIP] run_variant '{args.run_variant}' only applies to {only_models}, not '{args.model}'")
        return 0

    models = cfg.get("models", {}) or {}
    if args.model not in models:
        raise SystemExit(f"model '{args.model}' not found in config models")

    grids = (models[args.model] or {}).get("grids", {}) or {}
    if args.grid_name not in grids:
        raise SystemExit(f"grid_name '{args.grid_name}' not found for model '{args.model}'")

    grid_spec = grids[args.grid_name]

    # NEW: expand grid spec (legacy list OR cartesian dict)
    try:
        param_list = expand_grid_spec(grid_spec)
    except Exception as e:
        raise SystemExit(f"grid '{args.grid_name}' for model '{args.model}' could not be expanded: {e}") from e

    if not isinstance(param_list, list) or len(param_list) == 0:
        raise SystemExit(f"grid '{args.grid_name}' for model '{args.model}' is empty after expansion")

    # When --trial-idx is given: select only that one trial and use a per-trial subdirectory
    trial_idx = args.trial_idx
    if trial_idx >= 0:
        if trial_idx >= len(param_list):
            raise SystemExit(f"trial_idx={trial_idx} out of range (grid has {len(param_list)} trials)")
        param_list = [param_list[trial_idx]]
        out_dir = Path(args.out_dir).resolve() / f"t{trial_idx}"
        time_budget = 10 * 24 * 3600  # effectively no time limit for single-trial mode
    else:
        out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=== RUN ===")
    log(f"operator={args.operator}")
    log(f"model={args.model}")
    log(f"run_variant={args.run_variant}")
    log(f"grid_name={args.grid_name}")
    log(f"trial_idx={trial_idx if trial_idx >= 0 else 'all'}")
    log(f"out_dir={out_dir}")
    log(f"time_budget_seconds={time_budget}")
    log(f"config={args.config}")
    log(f"expanded_trials={len(param_list)}")
    log("===========")

    all_mode = bool(cfg.get("all_mode"))

    # --dataset_path: load pre-built pickle if available (fast path for sampling runs)
    dataset_path_str = cfg.get("dataset_path", "")
    dataset_path = Path(dataset_path_str) if dataset_path_str else None
    use_prebuilt = (
        dataset_path is not None
        and (dataset_path / "valid_sampled.pkl").exists()
        and ((dataset_path / "test_full.pkl").exists()
             or (use_identity and args.run_test is False))
    )

    # --valid / --test flags: if neither given, default to both (backward compat)
    explicit_valid = args.run_valid is True
    explicit_test  = args.run_test  is True
    neither_given  = (args.run_valid is None and args.run_test is None)
    run_valid = explicit_valid or neither_given
    run_test  = explicit_test  or neither_given
    log(f"[INFO] run_valid={run_valid} run_test={run_test}")

    try:
        if use_prebuilt:
            # Cached data contains its own aligned identities and merge metadata.
            # It must remain loadable when original CSVs are no longer present.
            pass
        elif has_validation:
            # --- Temporal validation/test split ---
            if all_mode:
                log("[INFO] ALL mode: building combined all-operators validation and test DataFrames")
                df_valid, df_test, merge_meta_valid, merge_meta_test = build_all_operators_merged_df(
                    cfg=cfg,
                    data_dir=data_dir,
                    validation_period_prefixes=validation_period_prefixes,
                    test_period_prefixes=test_period_prefixes,
                    id_col=args.id_col,
                    explicit_target_col=target_col,
                    column_prefixes=generic_col_prefixes,
                )
            else:
                log("[INFO] Validation mode: building separate validation (train) and test DataFrames")
                df_valid, merge_meta_valid = build_merged_df(
                    operator=args.operator,
                    data_dir=data_dir,
                    period_prefixes=validation_period_prefixes,
                    id_col=args.id_col,
                    target_source_prefix=target_source_prefix,
                    explicit_target_col=target_col,
                    column_prefixes=generic_col_prefixes,
                    identity_col=identity_col,
                )
                df_test, merge_meta_test = build_merged_df(
                    operator=args.operator,
                    data_dir=data_dir,
                    period_prefixes=test_period_prefixes,
                    id_col=args.id_col,
                    target_source_prefix=target_source_prefix,
                    explicit_target_col=target_col,
                    column_prefixes=generic_col_prefixes,
                    identity_col=identity_col,
                )
            merge_meta = {
                "validation": merge_meta_valid,
                "test": merge_meta_test,
                "mode": "all_operators_temporal" if all_mode else "temporal_validation_test",
            }
        else:
            # --- Legacy: single set, random train/test split ---
            log("[INFO] Legacy mode: single period set with random train/test split")
            df, merge_meta_single = build_merged_df(
                operator=args.operator,
                data_dir=data_dir,
                period_prefixes=test_period_prefixes,
                id_col=args.id_col,
                target_source_prefix=target_source_prefix,
                explicit_target_col=target_col,
                identity_col=identity_col,
            )
            merge_meta = merge_meta_single
    except Exception as e:
        if use_identity:
            raise
        log(f"[SKIP] merge failed: {e}")
        (out_dir / "meta_failed.json").write_text(json.dumps({
            "error": str(e),
            "operator": args.operator,
            "model": args.model,
            "run_variant": args.run_variant,
            "grid_name": args.grid_name,
        }, indent=2), encoding="utf-8")
        return 0

    if use_prebuilt:
        import pickle as _pickle
        log(f"[INFO] Loading pre-built dataset from {dataset_path}")
        with open(dataset_path / "valid_sampled.pkl", "rb") as _f:
            _dv = _pickle.load(_f)
        if (dataset_path / "test_full.pkl").exists():
            with open(dataset_path / "test_full.pkl", "rb") as _f:
                _dt = _pickle.load(_f)
        else:
            _dt = None
        X_train, y_train = _dv["X"], _dv["y"]
        X_test, y_test = (_dt["X"], _dt["y"]) if _dt is not None else (X_train.iloc[:0], y_train.iloc[:0])
        _meta_ds = json.loads((dataset_path / "meta.json").read_text(encoding="utf-8"))
        groups_train = validate_prebuilt_identity(_dv, _meta_ds, cfg, args.id_col)
        groups_test = validate_prebuilt_identity(_dt, _meta_ds, cfg, args.id_col) if _dt is not None else None
        if use_identity and _dt is not None:
            if not _meta_ds.get("identity_holdout_applied"):
                raise ValueError("Pre-built dataset has no person test holdout; rebuild the dataset")
            if set(groups_train).intersection(groups_test):
                raise ValueError("Pre-built training and test share persons; rebuild the dataset")
        feature_cols = _meta_ds["feature_cols"]
        positives = int(y_train.sum())
        total     = int(len(y_train))
        negatives = total - positives
        pos_rate  = positives / max(total, 1)
        spw       = float(negatives / max(positives, 1))
        merge_meta = {"mode": "prebuilt_dataset", "dataset_path": str(dataset_path)}
        merge_meta["identity_split"] = {k: v for k, v in _meta_ds.items() if k.startswith("identity_") or k == "Niels_Identity_Confounding_switch"}
        log(f"[INFO] pre-built valid: {total:,} rows, {positives:,} pos ({pos_rate:.4f}), {len(feature_cols)} features")
        log(f"[INFO] pre-built test:  {len(X_test):,} rows, {int(y_test.sum()):,} pos")
        has_validation = True

    elif has_validation:
        tgt_valid = merge_meta_valid["target_col"]
        tgt_test = merge_meta_test["target_col"]
        log(f"[INFO] validation: rows={merge_meta_valid['n_rows']} cols={merge_meta_valid['n_cols']} target={tgt_valid}")
        log(f"[INFO] test:       rows={merge_meta_test['n_rows']} cols={merge_meta_test['n_cols']} target={tgt_test}")

        if filter_active and "ACTIVE_FLAG" in df_valid.columns:
            n_before = len(df_valid)
            df_valid = df_valid[df_valid["ACTIVE_FLAG"] != False]  # noqa: E712
            log(f"[ACTIVE] valid: {n_before - len(df_valid)} inactieve rijen gefilterd ({len(df_valid)} over)")
        if filter_active and "ACTIVE_FLAG" in df_test.columns:
            n_before = len(df_test)
            df_test = df_test[df_test["ACTIVE_FLAG"] != False]  # noqa: E712
            log(f"[ACTIVE] test:  {n_before - len(df_test)} inactieve rijen gefilterd ({len(df_test)} over)")

        df_valid, df_test, groups_train, groups_test, identity_meta = person_holdout(
            df_valid, df_test, cfg, args.id_col, test_size=args.test_size, random_state=23,
        )
        merge_meta["identity_split"] = identity_meta
        X_train, y_train, feature_cols = make_Xy(df_valid, id_col=args.id_col, target_col=tgt_valid, identity_col=identity_col)
        X_test, y_test, _ = make_Xy(df_test, id_col=args.id_col, target_col=tgt_test, identity_col=identity_col)

        if args.round and DTYPE_MAPPING:
            apply = {c: t for c, t in DTYPE_MAPPING.items() if c in X_train.columns}
            if apply:
                X_train = X_train.astype(apply)
                X_test  = X_test.astype({c: t for c, t in apply.items() if c in X_test.columns})
                log(f"[INFO] --round: {len(apply)} columns dtype-converted")

        positives = int(y_train.sum())
        total = int(len(y_train))
        negatives = total - positives
        pos_rate = positives / max(total, 1)
        spw = float(negatives / max(positives, 1))

        log(f"[INFO] train(valid): total={total} positives={positives} negatives={negatives} pos_rate={pos_rate:.6f} spw={spw:.3f}")
        log(f"[INFO] test:         total={len(y_test)} positives={int(y_test.sum())}")
        log(f"[INFO] n_features={len(feature_cols)}")
    else:
        tgt = merge_meta["target_col"]
        log(f"[INFO] merged rows={merge_meta['n_rows']} cols={merge_meta['n_cols']} target={tgt}")

        if filter_active and "ACTIVE_FLAG" in df.columns:
            n_before = len(df)
            df = df[df["ACTIVE_FLAG"] != False]  # noqa: E712
            log(f"[ACTIVE] {n_before - len(df)} inactieve rijen gefilterd ({len(df)} over)")

        groups = identity_groups(df, args.id_col, cfg) if use_identity else None
        X, y, feature_cols = make_Xy(df, id_col=args.id_col, target_col=tgt, identity_col=identity_col)
        positives = int(y.sum())
        total = int(len(y))
        negatives = total - positives
        pos_rate = positives / max(total, 1)
        spw = float(negatives / max(positives, 1))

        log(f"[INFO] total={total} positives={positives} negatives={negatives} pos_rate={pos_rate:.6f} spw={spw:.3f}")
        log(f"[INFO] n_features={len(feature_cols)}")

        stratify = y if (y.nunique() == 2 and positives >= 2 and negatives >= 2) else None
        if use_identity:
            train_idx, test_idx = next(person_cv_indices(X, y, groups, n_folds=1,
                                                        random_state=int(cfg.get("random_state", 23)), test_size=args.test_size))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=args.random_state, stratify=stratify
            )

    preprocess_steps = build_preprocess_steps(rv)
    imbalance = (rv or {}).get("imbalance", {}) or {}
    early_stopping_cfg = (rv or {}).get("early_stopping", {}) or {}

    if early_stopping_cfg.get("enabled") and args.model in EARLY_STOPPING_MODELS:
        log(f"[INFO] Early stopping enabled: patience={early_stopping_cfg.get('patience', 50)} model={args.model}")

    if args.model in ("xgboost", "lightgbm") and spw > MAX_SCALE_POS_WEIGHT:
        log(f"[INFO] {args.model} scale_pos_weight capped: {spw:.1f} -> {MAX_SCALE_POS_WEIGHT:.1f}")
    if args.model == "xgboost_focal":
        log(f"[INFO] xgboost_focal: focal loss objective active; scale_pos_weight not injected (data spw={spw:.1f})")

    # Manual search within time budget
    cv_folds = args.cv_folds
    t0 = time.time()
    trial_rows = []
    best = None  # (key, ..., params, pipe)

    if has_validation:
        cv_kind = "person-grouped" if use_identity else "stratified"
        log(f"[INFO] Using {cv_folds}-fold {cv_kind} CV on validation set for HP selection")

    for i, raw_params in enumerate(param_list, start=1):
        if (time.time() - t0) >= time_budget:
            log(f"[TIME] budget hit after {time.time()-t0:.1f}s, stopping trials.")
            break

        params = apply_imbalance_to_params(raw_params, args.model, imbalance, spw)

        # keep your MLP tuple fix
        if args.model == "mlp" and "hidden_layer_sizes" in params and isinstance(params["hidden_layer_sizes"], list):
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])

        est = build_estimator(args.model, spw=spw, random_state=args.random_state)
        apply_early_stopping(est, args.model, early_stopping_cfg)

        # For xgboost_focal: extract focal_alpha/focal_gamma, set objective closure.
        # These are not valid XGBClassifier params so must be removed before set_params.
        # They remain in `params` so they are logged in params_json / results.csv.
        set_params = dict(params)
        if args.model == "xgboost_focal":
            fa = float(set_params.pop("focal_alpha", 0.25))
            fg = float(set_params.pop("focal_gamma", 2.0))
            est.set_params(objective=focal_binary_objective(fa, fg))

        try:
            est.set_params(**set_params)
        except Exception as e:
            log(f"[SKIP] bad params trial={i} params={params} err={e}")
            continue

        pipe = build_pipeline(preprocess_steps, imbalance, args.model, est)

        log(f"[TRY {i}/{len(param_list)}] params={params}")

        if has_validation and run_valid:
            # k-fold CV (or 1-fold holdout) on validation set
            try:
                mean_pr, std_pr, mean_auc, std_auc, fit_s = cv_score(
                    pipe, X_train, y_train, n_folds=cv_folds, random_state=args.random_state,
                    model_name=args.model, early_stopping_cfg=early_stopping_cfg,
                    groups=groups_train,
                )
            except Exception as e:
                log(f"[FAIL] trial={i} CV failed: {e}")
                continue

            elapsed = time.time() - t0
            row = {
                "trial": i,
                "cv_mean_auprc": mean_pr,
                "cv_std_auprc": std_pr,
                "cv_mean_auc": mean_auc,
                "cv_std_auc": std_auc,
                "fit_seconds": fit_s,
                "elapsed_seconds": elapsed,
                "params_json": json.dumps(params, sort_keys=True),
            }
            trial_rows.append(row)
            log(f"[RES] CV_AUC={mean_auc:.6f}+/-{std_auc:.4f} CV_AUPRC={mean_pr:.6f}+/-{std_pr:.4f} fit_s={fit_s:.2f} elapsed={elapsed:.1f}s")

            key = (np.nan_to_num(mean_pr, nan=-1.0), np.nan_to_num(mean_auc, nan=-1.0))
            if best is None or key > best[0]:
                best = (key, mean_auc, mean_pr, std_auc, std_pr, params, pipe)
        else:
            # Legacy: single holdout eval
            try:
                auc, pr, fit_s = fit_and_eval(
                    pipe, X_train, y_train, X_test, y_test,
                    model_name=args.model, early_stopping_cfg=early_stopping_cfg,
                    groups=groups_train,
                )
            except Exception as e:
                log(f"[FAIL] trial={i} fit/eval failed: {e}")
                continue

            elapsed = time.time() - t0
            row = {
                "trial": i,
                "roc_auc": auc,
                "auprc": pr,
                "fit_seconds": fit_s,
                "elapsed_seconds": elapsed,
                "params_json": json.dumps(params, sort_keys=True),
            }
            trial_rows.append(row)
            log(f"[RES] AUC={auc:.6f} AUPRC={pr:.6f} fit_s={fit_s:.2f} elapsed={elapsed:.1f}s")

            key = (np.nan_to_num(pr, nan=-1.0), np.nan_to_num(auc, nan=-1.0))
            if best is None or key > best[0]:
                best = (key, auc, pr, params, pipe)

    meta_out = {
        "operator": args.operator,
        "model": args.model,
        "run_variant": args.run_variant,
        "grid_name": args.grid_name,
        "time_budget_seconds": time_budget,
        "config_path": args.config,
        "merge_meta": merge_meta,
        "n_rows": total,
        "n_features": len(feature_cols),
        "positives": positives,
        "negatives": negatives,
        "pos_rate": pos_rate,
        "spw": spw,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "preprocessing": (rv or {}).get("preprocessing", {}),
        "imbalance": imbalance,
        "wall_seconds": time.time() - t0,
        "expanded_trials": len(param_list),
        "cv_folds": cv_folds if has_validation else None,
        "mode": "temporal_cv" if has_validation else "legacy_random",
        "early_stopping": early_stopping_cfg if early_stopping_cfg.get("enabled") else None,
        "Niels_Identity_Confounding_switch": use_identity,
        "identity_column": identity_col or args.id_col,
        "identity_scope": "operator_player" if use_identity else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    if not trial_rows:
        log("[WARN] no successful trials; wrote meta only")
        return 0

    pd.DataFrame(trial_rows).to_csv(out_dir / "results.csv", index=False)

    if best is None:
        log("[WARN] best is None (unexpected); wrote results")
        return 0

    if has_validation and run_test:
        # Retrain best config on full validation set, evaluate on test
        _, best_cv_auc, best_cv_pr, best_cv_std_auc, best_cv_std_pr, best_params, best_pipe = best
        log(f"[BEST CV] AUPRC={best_cv_pr:.6f}+/-{best_cv_std_pr:.4f} AUC={best_cv_auc:.6f}+/-{best_cv_std_auc:.4f} params={best_params}")

        log("[INFO] Retraining best config on full validation set...")
        test_auc, test_pr, refit_s = fit_and_eval(
            best_pipe, X_train, y_train, X_test, y_test,
            model_name=args.model, early_stopping_cfg=early_stopping_cfg,
            groups=groups_train,
        )
        log(f"[TEST] AUC={test_auc:.6f} AUPRC={test_pr:.6f} refit_s={refit_s:.2f}")

        (out_dir / "best.json").write_text(json.dumps({
            "cv_mean_auprc": best_cv_pr,
            "cv_std_auprc": best_cv_std_pr,
            "cv_mean_auc": best_cv_auc,
            "cv_std_auc": best_cv_std_auc,
            "test_roc_auc": test_auc,
            "test_auprc": test_pr,
            "best_params": best_params,
        }, indent=2), encoding="utf-8")
    elif has_validation and not run_test:
        # CV only — save CV scores without test evaluation
        _, best_cv_auc, best_cv_pr, best_cv_std_auc, best_cv_std_pr, best_params, best_pipe = best
        log(f"[BEST CV] AUPRC={best_cv_pr:.6f}+/-{best_cv_std_pr:.4f} AUC={best_cv_auc:.6f}+/-{best_cv_std_auc:.4f} params={best_params}")
        (out_dir / "best.json").write_text(json.dumps({
            "cv_mean_auprc": best_cv_pr,
            "cv_std_auprc": best_cv_std_pr,
            "cv_mean_auc": best_cv_auc,
            "cv_std_auc": best_cv_std_auc,
            "best_params": best_params,
        }, indent=2), encoding="utf-8")
    else:
        # Legacy: single holdout, no validation/test split
        _, best_auc, best_pr, best_params, best_pipe = best
        (out_dir / "best.json").write_text(json.dumps({
            "best_roc_auc": best_auc,
            "best_auprc": best_pr,
            "best_params": best_params,
        }, indent=2), encoding="utf-8")
        log(f"[BEST] AUC={best_auc:.6f} AUPRC={best_pr:.6f} params={best_params}")

    try:
        final_est = best_pipe.named_steps["model"] if hasattr(best_pipe, "named_steps") else None
        if final_est is not None:
            surviving_features = get_feature_names_after_pipeline(best_pipe, feature_cols)
            # Pre-transform X_test through preprocessing steps so shapes match the estimator
            X_test_tr = X_test
            if hasattr(best_pipe, "steps"):
                for step_name, step in best_pipe.steps[:-1]:
                    if step_name == "smote" or not hasattr(step, "transform"):
                        continue
                    X_test_tr = step.transform(X_test_tr)
            fi = try_get_feature_importances(
                final_est, surviving_features,
                X_test=X_test_tr, y_test=y_test,
                random_state=args.random_state,
            )
            if fi is not None and not fi.empty:
                fi.to_csv(out_dir / "feature_importances.csv", index=False)
                if "importance_std" in fi.columns:
                    log(f"[INFO] feature importances saved (permutation, n={len(fi)} features)")
                else:
                    log(f"[INFO] feature importances saved (native, n={len(fi)} features)")
    except Exception as e:
        log(f"[WARN] feature importances failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
