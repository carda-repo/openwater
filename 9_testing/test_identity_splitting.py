"""Regression coverage for the default-on operator/player identity split."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

MS_DIR = Path(__file__).resolve().parents[1] / "6_merge_sample"
sys.path.insert(0, str(MS_DIR))

from identity_splitting import (  # noqa: E402
    INTERNAL_OPERATOR_ID, INTERNAL_PERSON_ID, identity_enabled,
    identity_groups, person_holdout,
)
import prepare_hpo_dataset as prepare  # noqa: E402
from merge_sample_pipeline import build_config  # noqa: E402

SWITCH = "Niels_Identity_Confounding_switch"
TARGET = "y_self_exclusion_20250701_20250731"
TEST_TARGET = "y_self_exclusion_20250801_20250831"


def _frame(operator="Operator_a", n=100, later=False):
    return pd.DataFrame({
        "Player_Profile_ID": [f"player-{i}" for i in range(n)],
        INTERNAL_OPERATOR_ID: operator,
        "feature": [i + (1000 if later else 0) for i in range(n)],
        "ACTIVE_FLAG": True,
        TEST_TARGET if later else TARGET: [int(i % 5 == 0) for i in range(n)],
    })


@pytest.mark.parametrize("value,expected", [(True, True), (False, False), ("true", True),
                                                ("false", False), (1, True), (0, False)])
def test_switch_parses_explicit_boolean(value, expected):
    assert identity_enabled({SWITCH: value}) is expected
    assert identity_enabled({}) is True


@pytest.mark.parametrize("value", [None, "yes", "disabled", 2, [], {}])
def test_switch_rejects_ambiguous_values(value):
    with pytest.raises(ValueError, match=SWITCH):
        identity_enabled({SWITCH: value})


def test_composite_groups_keep_operator_folder_and_avoid_delimiter_collisions():
    df = pd.DataFrame({
        "Player_Profile_ID": ["c", "b_c", "c", "c"],
        INTERNAL_OPERATOR_ID: ["a_b", "a", "Operator_a", "Operator_b"],
    }, index=[8, 3, 4, 9])
    groups = identity_groups(df)
    assert groups.index.equals(df.index)
    assert groups.nunique() == 4
    assert json.loads(groups.loc[4]) == ["Operator_a", "c"]


@pytest.mark.parametrize("missing", [None, "", "   ", "nan"])
def test_missing_player_or_operator_identity_fails(missing):
    df = _frame(n=2)
    df.loc[0, "Player_Profile_ID"] = missing
    with pytest.raises(ValueError, match="person identifier"):
        identity_groups(df)
    df = _frame(n=2)
    df.loc[0, INTERNAL_OPERATOR_ID] = missing
    with pytest.raises(ValueError, match="operator identifier"):
        identity_groups(df)


def test_operator_and_configured_identity_are_required():
    with pytest.raises(ValueError, match="operator metadata"):
        identity_groups(_frame().drop(columns=INTERNAL_OPERATOR_ID))
    with pytest.raises(ValueError, match="custom_person"):
        identity_groups(_frame(), cfg={"Niels_identity_column": "custom_person"})


def test_custom_person_column_can_be_preserved_in_reserved_metadata():
    df = _frame(n=2)
    df[INTERNAL_PERSON_ID] = ["person-A", "person-A"]
    groups = identity_groups(df, cfg={"Niels_identity_column": "custom_person"})
    assert groups.nunique() == 1
    assert json.loads(groups.iloc[0]) == ["Operator_a", "person-A"]


def test_holdout_excludes_all_repeated_pairs_without_moving_temporal_rows():
    train = pd.concat([_frame(), _frame()], ignore_index=True)
    test = pd.concat([_frame(later=True), _frame(later=True)], ignore_index=True)
    development, heldout, development_groups, test_groups, meta = person_holdout(train, test, {})
    assert set(development_groups).isdisjoint(test_groups)
    assert len(development) == 160
    assert len(heldout) == 40
    assert development["feature"].max() < 1000
    assert heldout["feature"].min() >= 1000
    assert development_groups.index.equals(development.index)
    assert test_groups.index.equals(heldout.index)
    assert set(development_groups).union(test_groups) == set(identity_groups(train))
    assert meta["identity_scope"] == "operator_player"
    assert meta["identity_split_random_state"] == 23


def test_holdout_is_reproducible_and_independent_of_row_order_and_test_labels():
    train, test = _frame(), _frame(later=True)
    first = person_holdout(train, test, {})
    test[TEST_TARGET] = 1 - test[TEST_TARGET]
    second = person_holdout(train.sample(frac=1, random_state=5), test.iloc[::-1], {})
    assert set(first[2]) == set(second[2])
    assert set(first[3]) == set(second[3])


def test_same_player_id_at_different_operators_is_already_disjoint():
    train, test = _frame("Operator_a", n=2), _frame("Operator_b", n=2, later=True)
    result = person_holdout(train, test, {})
    pd.testing.assert_frame_equal(result[0], train)
    pd.testing.assert_frame_equal(result[1], test)
    assert set(result[2]).isdisjoint(result[3])
    assert result[4]["identity_split_method"] == "already_disjoint"


def test_no_test_keeps_development_rows_and_groups_for_cv():
    train = _frame()
    result = person_holdout(train, pd.DataFrame(), {})
    pd.testing.assert_frame_equal(result[0], train)
    assert result[2].index.equals(train.index)
    assert result[3].empty
    assert result[4]["identity_holdout_applied"] is False


def test_single_overlapping_pair_cannot_fall_back_to_unsafe_split():
    with pytest.raises(ValueError, match="at least two"):
        person_holdout(_frame(n=1), _frame(n=1, later=True), {})


def test_disabled_split_needs_no_identity_metadata_and_preserves_all_rows():
    train = _frame().drop(columns=["Player_Profile_ID", INTERNAL_OPERATOR_ID])
    test = _frame(later=True).drop(columns=["Player_Profile_ID", INTERNAL_OPERATOR_ID])
    result = person_holdout(train, test, {SWITCH: False})
    assert result[0] is train and result[1] is test
    assert result[2] is None and result[3] is None


def _payload(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _prepare_config(tmp_path, **kwargs):
    return build_config(dataset_path=tmp_path / "dataset", data_dir=tmp_path,
                        validation_period_prefixes=["valid"],
                        test_period_prefixes=["test"], all_operators=["Operator_a"], **kwargs)


def test_preparation_samples_only_development_and_writes_aligned_group_provenance(tmp_path, monkeypatch):
    train, test = _frame(), _frame(later=True)
    monkeypatch.setattr(prepare, "build_all_operators_merged_df", lambda **kwargs: (train, test, {}, {}))
    cfg = _prepare_config(tmp_path, sampling_ratio=1)
    out = prepare.prepare_dataset(cfg)
    development, heldout = _payload(out / "valid_sampled.pkl"), _payload(out / "test_full.pkl")
    assert development["groups"].index.equals(development["X"].index)
    assert development["groups"].index.equals(development["y"].index)
    assert set(development["groups"]).isdisjoint(heldout["groups"])
    assert int((development["y"] == 0).sum()) == int(development["y"].sum())
    assert len(heldout["X"]) == 20  # Test remains unsampled.
    assert list(development["X"].columns) == ["feature"]
    meta = json.loads((out / "meta.json").read_text())
    assert meta[SWITCH] is True
    assert meta["identity_scope"] == "operator_player"
    assert meta["identity_dataset_id"] == development["identity_metadata"]["identity_dataset_id"]
    assert development["identity_metadata"] == heldout["identity_metadata"]


def test_preparation_loads_operator_holdout_before_split_and_sampling(tmp_path, monkeypatch):
    calls = []
    def build(**kwargs):
        calls.append(kwargs["cfg"]["all_operators"])
        if calls[-1] == ["Operator_b"]:
            return _frame("Operator_b"), pd.DataFrame(), {}, {}
        return _frame(), _frame(later=True), {}, {}
    monkeypatch.setattr(prepare, "build_all_operators_merged_df", build)
    out = prepare.prepare_dataset(_prepare_config(tmp_path, sampling_ratio=1,
                                                 fold_holdout_operators=["Operator_b"]))
    development, heldout = _payload(out / "valid_sampled.pkl"), _payload(out / "test_full.pkl")
    assert calls == [["Operator_a"], ["Operator_b"]]
    assert len(heldout["X"]) == 100
    assert set(development["groups"]).isdisjoint(heldout["groups"])


def test_disabled_preparation_preserves_legacy_X_y_exactly(tmp_path, monkeypatch):
    train, test = _frame(), _frame(later=True)
    train = train.drop(columns=INTERNAL_OPERATOR_ID)
    test = test.drop(columns=INTERNAL_OPERATOR_ID)
    monkeypatch.setattr(prepare, "build_all_operators_merged_df", lambda **kwargs: (train, test, {}, {}))
    out = prepare.prepare_dataset(_prepare_config(tmp_path, Niels_Identity_Confounding_switch=False))
    development, heldout = _payload(out / "valid_sampled.pkl"), _payload(out / "test_full.pkl")
    assert set(development) == {"X", "y"}
    assert set(heldout) == {"X", "y"}
    pd.testing.assert_frame_equal(development["X"], train[["feature"]])
    pd.testing.assert_series_equal(development["y"], train[TARGET])
    pd.testing.assert_frame_equal(heldout["X"], test[["feature"]])
    pd.testing.assert_series_equal(heldout["y"], test[TEST_TARGET])


def test_preparation_excludes_numeric_custom_identity_and_prefixed_copies(tmp_path, monkeypatch):
    train, test = _frame(), _frame(later=True)
    for frame in (train, test):
        frame["person_id"] = range(len(frame))
        frame["p0_person_id"] = range(len(frame))
        frame[INTERNAL_PERSON_ID] = frame["person_id"].astype(str)
    monkeypatch.setattr(prepare, "build_all_operators_merged_df", lambda **kwargs: (train, test, {}, {}))
    out = prepare.prepare_dataset(_prepare_config(tmp_path, Niels_identity_column="person_id"))
    assert list(_payload(out / "valid_sampled.pkl")["X"].columns) == ["feature"]


def test_preparation_removes_stale_test_when_enabled_and_no_test_candidates(tmp_path, monkeypatch):
    cfg = _prepare_config(tmp_path)
    out = Path(cfg["dataset_path"])
    out.mkdir()
    (out / "test_full.pkl").write_bytes(b"stale")
    monkeypatch.setattr(prepare, "build_all_operators_merged_df",
                        lambda **kwargs: (_frame(), pd.DataFrame(), {}, {}))
    prepare.prepare_dataset(cfg)
    assert not (out / "test_full.pkl").exists()
    assert len(_payload(out / "valid_sampled.pkl")["groups"]) == 100
