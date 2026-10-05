#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modelling_pipeline.py  (organisatie-versie)
===================================

Pure-Python orkestratie voor de modelling-stap (organisatie-stappen 7 & 8; origineel 14 & 15).
Vervangt de bash-orchestratie van `submit_hpsearch.sh` (config genereren + taken aftrappen)
door pure Python — beide zoekstrategieën:

* **Optuna** (single-process, tijd-gebudgetteerd) → `optuna_runner.main(...)`
* **Gridsearch** (per-taak worker, hier in een gewone Python-loop i.p.v. SLURM-array)
  → `hpsearch_runner.main(...)` per (operator × model × run_variant × grid).

Beide draaien op een *effectieve config* (zoals `submit_hpsearch.sh` die schreef): de
basis-config (bijv. `hpsearch_config.yaml`) met `all_mode: True` en — voor het snelle pad —
een `dataset_path` dat naar de prebuilt pickle uit stap 6 wijst.

De feitelijke training/zoeklogica zit ongewijzigd in `hpsearch_runner.py` / `optuna_runner.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
HPSEARCH_RUNNER = SCRIPT_DIR / "hpsearch_runner.py"
OPTUNA_RUNNER = SCRIPT_DIR / "optuna_runner.py"


def _run_worker(script: Path, args: List[str]) -> None:
    """
    Draai een worker-script (hpsearch_runner.py / optuna_runner.py) als subprocess.

    Bewust een subprocess i.p.v. een in-process import: de twee runners bestaan ook als
    kopie in andere stap-mappen (zelfde modulenaam), dus in-process importeren zou tot
    naamconflicten leiden. Een subprocess geeft elk worker een schone, eigen sys.path —
    en het spiegelt het origineel, dat de workers ook als losse processen draaide.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        [sys.executable, str(script)] + args,
        cwd=str(SCRIPT_DIR), env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"worker faalde (exit {proc.returncode}): {script.name} {' '.join(args)}")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def load_config(path: str | Path) -> Dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def make_effective_config(
    base_config: str | Path | Dict,
    *,
    dataset_path: Optional[str | Path] = None,
    data_dir: Optional[str | Path] = None,
    operators: Optional[List[str]] = None,
    exclude_models: Optional[List[str]] = None,
    overrides: Optional[Dict] = None,
) -> Dict:
    """
    Bouw de effectieve config (zoals submit_hpsearch.sh schreef): basis-config + `all_mode=True`,
    optioneel een prebuilt `dataset_path`, `data_dir`, operator-selectie en model-uitsluiting.
    """
    cfg = dict(load_config(base_config) if not isinstance(base_config, dict) else base_config)
    cfg["all_mode"] = True
    if dataset_path is not None:
        cfg["dataset_path"] = str(dataset_path)
    if data_dir is not None:
        cfg["data_dir"] = str(data_dir)
    if operators is not None:
        cfg["operators"] = list(operators)
        cfg["all_operators"] = list(operators)
    elif "all_operators" not in cfg:
        cfg["all_operators"] = cfg.get("operators", [])
    if exclude_models is not None:
        cfg["exclude_models"] = list(exclude_models)
    if overrides:
        cfg.update(overrides)
    cfg.setdefault("Niels_Identity_Confounding_switch", True)
    return cfg


def write_effective_config(cfg: Dict, out_dir: str | Path) -> Path:
    """Schrijf de effectieve config naar out_dir/hpsearch_config_effective.yaml."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "hpsearch_config_effective.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


# -----------------------------------------------------------------------------
# Grid-taak enumeratie (zoals submit_hpsearch.sh: operators × models × run_variants × grids)
# -----------------------------------------------------------------------------
def enumerate_grid_tasks(
    cfg: Dict,
    *,
    models: Optional[List[str]] = None,
    grids: Optional[List[str]] = None,
    operators: Optional[List[str]] = None,
    run_variants: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Geef de lijst grid-taken (operator, model, run_variant, grid_name). Een run_variant geldt
    alleen voor een model als die geen `only_models` heeft, of als het model erin staat — exact
    de filtering die de worker (`hpsearch_runner.main`) ook toepast.
    """
    ops = operators or cfg.get("operators") or cfg.get("all_operators") or []
    rvs = cfg.get("run_variants") or []
    all_models = cfg.get("models") or {}

    model_names = models or list(all_models.keys())
    rv_names = run_variants

    tasks: List[Dict] = []
    for op in ops:
        for model in model_names:
            mspec = all_models.get(model) or {}
            grid_names = grids or list((mspec.get("grids") or {}).keys())
            for rv in rvs:
                rv_name = (rv or {}).get("name")
                if rv_names and rv_name not in rv_names:
                    continue
                only = (rv or {}).get("only_models")
                if only and model not in only:
                    continue
                for grid_name in grid_names:
                    tasks.append({
                        "operator": op, "model": model,
                        "run_variant": rv_name, "grid_name": grid_name,
                    })
    return tasks


# -----------------------------------------------------------------------------
# Gridsearch (per-taak worker in een loop; vervangt de SLURM-array)
# -----------------------------------------------------------------------------
def run_grid_search(
    cfg: Dict,
    out_dir: str | Path,
    *,
    models: Optional[List[str]] = None,
    grids: Optional[List[str]] = None,
    operators: Optional[List[str]] = None,
    run_variants: Optional[List[str]] = None,
    cv_folds: int = 5,
    run_valid: bool = True,
    run_test: bool = True,
    do_round: bool = False,
    random_state: int = 23,
) -> List[Path]:
    """
    Draai alle grid-taken sequentieel via `hpsearch_runner.main(...)`. Output per taak in
    out_dir/<operator>__<model>__<run_variant>__<grid_name>/. Geeft de lijst out-mappen terug.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = write_effective_config(cfg, out_dir)

    tasks = enumerate_grid_tasks(cfg, models=models, grids=grids,
                                 operators=operators, run_variants=run_variants)
    print(f"[grid] {len(tasks)} taken te draaien")

    produced: List[Path] = []
    for i, t in enumerate(tasks, start=1):
        # Geneste layout operator/model/run_variant/grid — zoals de originele model-outputs,
        # zodat de rapportage-scripts (stap 10/11) de results.csv/best.json/meta.json vinden.
        task_dir = out_dir / t["operator"] / t["model"] / t["run_variant"] / t["grid_name"]
        argv = [
            "--config", str(cfg_path),
            "--operator", t["operator"],
            "--model", t["model"],
            "--run-variant", t["run_variant"],
            "--grid-name", t["grid_name"],
            "--out-dir", str(task_dir),
            "--cv-folds", str(cv_folds),
            "--random-state", str(random_state),
        ]
        argv += ["--valid"] if run_valid else ["--no-valid"]
        argv += ["--test"] if run_test else ["--no-test"]
        if do_round:
            argv.append("--round")
        print(f"[grid] ({i}/{len(tasks)}) {task_dir.name}")
        _run_worker(HPSEARCH_RUNNER, argv)
        produced.append(task_dir)
    return produced


# -----------------------------------------------------------------------------
# Optuna (single-process search)
# -----------------------------------------------------------------------------
def run_optuna_search(
    cfg: Dict,
    out_dir: str | Path,
    *,
    time_budget: int = 12600,
    cv_folds: int = 1,
    n_startup: int = 20,
    validate_best: bool = False,
    cv5_top_n: int = 5,
    do_round: bool = False,
    no_multivariate: bool = False,
    random_state: int = 23,
) -> Path:
    """Draai de Optuna-search via `optuna_runner.main(...)` op de effectieve config."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = write_effective_config(cfg, out_dir)

    argv = [
        "--config", str(cfg_path),
        "--out-dir", str(out_dir),
        "--time-budget", str(time_budget),
        "--cv-folds", str(cv_folds),
        "--n-startup", str(n_startup),
        "--cv5-top-n", str(cv5_top_n),
        "--random-state", str(random_state),
    ]
    if validate_best:
        argv.append("--validate-best")
    if do_round:
        argv.append("--round")
    if no_multivariate:
        argv.append("--no-multivariate")

    _run_worker(OPTUNA_RUNNER, argv)
    return out_dir
