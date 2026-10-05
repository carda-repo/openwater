#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_sample_pipeline.py  (organisatie-versie)
======================================

Pure-Python orkestratie voor stap 6 (**Samenvoegen & samplen van alle operators**) uit de
organisatie-README (origineel: stap 13). Vervangt `run_prepare_hpo_dataset.sbatch`.

Dit is de **faithful** variant: hij voert exact `prepare_hpo_dataset.prepare_dataset(cfg)` uit,
die via `hpsearch_runner.build_all_operators_merged_df` de per-operator merged feature-CSV's
samenvoegt (incl. de 1-rij ALL-stats uit stap 3b, de 26-koloms a..z one-hot, en — indien
aanwezig — het ACTIVE_FLAG uit de lookup van stap 4), de target (`y_self_exclusion_*`) bepaalt,
en optioneel **negatieve undersampling** (1:ratio) toepast. Output: `valid_sampled.pkl`,
optioneel `test_full.pkl`, en `meta.json` in `dataset_path`.

Op Snellius leverde `submit_hpsearch.sh --all` een effectieve config-yaml die deze stap
afvuurde. Op de organisatie bouw je diezelfde config met `build_config(...)` (of geef je een eigen
yaml mee) en draai je `run_merge_sample.py` — geen sbatch.

> De data moet in de door het origineel verwachte layout staan: `data_dir/<operator>/` met
> per period-prefix een `{prefix}_{base_scenario}*.csv` (merged features + target), optioneel
> `{prefix}_{all_scenario_name}*.csv` (ALL-stats uit stap 3b) en
> `LAST_STATUS_LOOKUP_BEFORE_{ys}.csv` (stap 4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

# Faithful kern uit de hoofdrepo (gekopieerd in deze map).
from prepare_hpo_dataset import prepare_dataset  # noqa: F401  (re-export)


def build_config(
    *,
    dataset_path: str | Path,
    data_dir: str | Path,
    validation_period_prefixes: List[str],
    test_period_prefixes: Optional[List[str]] = None,
    all_operators: List[str],
    sampling_ratio: int = 0,
    target_col: str = "",
    base_scenario: str = "Flexible_spanish_plus",
    all_scenario_name: str = "ALL",
    fold_holdout_operators: Optional[List[str]] = None,
    Niels_Identity_Confounding_switch: bool = True,
    Niels_identity_column: Optional[str] = None,
    random_state: int = 23,
) -> Dict:
    """
    Stel de config-dict samen die `prepare_dataset` verwacht (identiek aan de effectieve
    yaml die `submit_hpsearch.sh --all` op Snellius genereerde).

    sampling_ratio : 0 = geen sampling; N = behoud alle positives + N× zoveel negatives.
    target_col     : leeg = automatisch afleiden uit de period-prefix (y_self_exclusion_*).
    Niels_Identity_Confounding_switch : standaard True; scheidt personen vóór sampling.
    Niels_identity_column : gedeelde persoonscode, standaard Player_Profile_ID.
    """
    cfg: Dict = {
        "all_mode": True,
        "dataset_path": str(dataset_path),
        "data_dir": str(data_dir),
        "validation_period_prefixes": list(validation_period_prefixes),
        "test_period_prefixes": list(test_period_prefixes or []),
        "all_operators": list(all_operators),
        "sampling_ratio": int(sampling_ratio),
        "target_col": target_col or "",
        "base_scenario": base_scenario,
        "all_scenario_name": all_scenario_name,
        "Niels_Identity_Confounding_switch": Niels_Identity_Confounding_switch,
        "random_state": int(random_state),
    }
    if Niels_identity_column:
        cfg["Niels_identity_column"] = Niels_identity_column
    if fold_holdout_operators:
        cfg["fold_holdout_operators"] = list(fold_holdout_operators)
    return cfg


def merge_and_sample(cfg: Dict) -> Path:
    """Voer de faithful prepare-stap uit voor een (samengestelde) config-dict."""
    return prepare_dataset(cfg)
