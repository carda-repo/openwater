"""Person groups shared by dataset preparation and model evaluation.

The key combines the data-folder operator identifier and Player_Profile_ID (or
the configured identity column). Groups are split before sampling and are never
model features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import GroupShuffleSplit


IDENTITY_SWITCH = "Niels_Identity_Confounding_switch"
INTERNAL_PERSON_ID = "__niels_person_id"
INTERNAL_OPERATOR_ID = "__niels_operator_id"


def identity_enabled(cfg: dict | None = None) -> bool:
    """Read the default-on switch without treating the string 'false' as true."""
    value = (cfg or {}).get(IDENTITY_SWITCH, True)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
        return value.strip().lower() in {"true", "1"}
    raise ValueError(f"{IDENTITY_SWITCH} must be True or False; got {value!r}")


def identity_groups(
    df: pd.DataFrame, id_col: str = "Player_Profile_ID", cfg: dict | None = None
) -> pd.Series:
    """Return aligned, collision-safe (operator, player) identity groups."""
    identity_col = (cfg or {}).get("Niels_identity_column") or id_col
    source = INTERNAL_PERSON_ID if INTERNAL_PERSON_ID in df.columns else identity_col
    if source not in df.columns:
        raise ValueError(
            f"Identity splitting requires person column {identity_col!r}. "
            "Provide the player identifier or rebuild the dataset with this column."
        )
    values = df[source]
    if not isinstance(values, pd.Series):
        raise ValueError(f"Person column {source!r} occurs more than once")
    normalized = values.astype("string").str.strip()
    # CSV readers upstream may already have converted a missing value to 'nan'.
    missing = normalized.isna() | normalized.str.lower().isin({"", "nan", "none", "null", "<na>"})
    if missing.any():
        raise ValueError(
            f"Identity splitting requires a non-empty person identifier for every row; "
            f"{int(missing.sum())} missing values in {identity_col!r}."
        )
    if INTERNAL_OPERATOR_ID not in df.columns:
        raise ValueError(
            "Identity splitting requires operator metadata '__niels_operator_id'. "
            "Rebuild the dataset with the operator folder identifier preserved."
        )
    operators = df[INTERNAL_OPERATOR_ID]
    if not isinstance(operators, pd.Series):
        raise ValueError(f"Operator column {INTERNAL_OPERATOR_ID!r} occurs more than once")
    normalized_operators = operators.astype("string").str.strip()
    missing_operators = normalized_operators.isna() | normalized_operators.str.lower().isin(
        {"", "nan", "none", "null", "<na>"}
    )
    if missing_operators.any():
        raise ValueError(
            "Identity splitting requires a non-empty operator identifier for every row; "
            f"{int(missing_operators.sum())} missing operator values."
        )
    # Preserve the original operator-folder spelling used by operator folds.
    keys = [json.dumps([str(operator), str(person)], separators=(",", ":"))
            for operator, person in zip(operators, normalized)]
    return pd.Series(keys, index=df.index, name=INTERNAL_PERSON_ID)


def person_holdout(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cfg: dict,
    id_col: str = "Player_Profile_ID",
    test_size: float = 0.2,
    random_state: int = 23,
):
    """Keep development and test candidates from disjoint person partitions.

    The union of person IDs is allocated without using labels. Rows keep their
    original role: test rows never become earlier training observations. Without
    a test dataset all development rows remain, with groups for validation CV.
    """
    enabled = identity_enabled(cfg)
    identity_col = cfg.get("Niels_identity_column") or id_col
    seed = int(cfg.get("random_state", random_state))
    metadata = {
        IDENTITY_SWITCH: enabled,
        "identity_column": identity_col,
        "identity_split_random_state": seed,
        "identity_scope": "operator_player",
    }
    if not enabled:
        return df_train, df_test, None, None, metadata
    groups_train = identity_groups(df_train, id_col, cfg)
    groups_test = (
        identity_groups(df_test, id_col, cfg)
        if not df_test.empty
        else pd.Series(index=df_test.index, dtype=object, name=INTERNAL_PERSON_ID)
    )
    if df_train.empty:
        raise ValueError("Identity splitting has no development rows after filtering")
    metadata.update({
        "identity_test_size": float(test_size),
        "n_persons_development": int(groups_train.nunique()),
        "n_persons_test": int(groups_test.nunique()),
        "identity_holdout_applied": False,
    })
    if df_test.empty:
        return df_train, df_test, groups_train, groups_test, metadata

    if set(groups_train).isdisjoint(groups_test):
        # Operator holdouts or previously separated identity sets already satisfy
        # the requirement; keep all candidates rather than reducing them again.
        metadata.update({
            "identity_holdout_applied": True,
            "identity_split_method": "already_disjoint",
            "n_rows_removed_development": 0,
            "n_rows_removed_test": 0,
        })
        return df_train, df_test, groups_train, groups_test, metadata

    people = np.array(sorted(set(groups_train).union(groups_test)), dtype=object)
    if len(people) < 2:
        raise ValueError("Identity splitting requires at least two distinct persons for development and test")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    development_idx, test_idx = next(splitter.split(people, groups=people))
    development_people, test_people = set(people[development_idx]), set(people[test_idx])
    keep_train, keep_test = groups_train.isin(development_people), groups_test.isin(test_people)
    if not keep_train.any() or not keep_test.any():
        raise ValueError(
            "The person split leaves an empty development or test dataset. "
            "Use more persons or adjust the split seed; identity separation cannot be skipped."
        )
    train = df_train.loc[keep_train].copy()
    test = df_test.loc[keep_test].copy()
    groups_train = groups_train.loc[keep_train].copy()
    groups_test = groups_test.loc[keep_test].copy()
    metadata.update({
        "identity_holdout_applied": True,
        "identity_split_method": "group_shuffle",
        "n_persons_development": int(groups_train.nunique()),
        "n_persons_test": int(groups_test.nunique()),
        "n_rows_removed_development": int((~keep_train).sum()),
        "n_rows_removed_test": int((~keep_test).sum()),
    })
    return train, test, groups_train, groups_test, metadata
