#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_grid.py  (organisatie-versie)
=========================

Pure-Python *runner* voor de modelling-stap via **gridsearch** (organisatie-stap 7 / origineel 14).
Vervangt `submit_hpsearch.sh` → de SLURM-array van `hpsearch_runner.py`-taken door een gewone
sequentiële Python-loop.

Voor elke (operator × model × run_variant × grid) wordt `hpsearch_runner.main(...)` aangeroepen,
die het hele grid voor die combinatie doorrekent (CV op de validatieset) en per taak
`results.csv`, `best.json` en `meta.json` wegschrijft. Draait op de prebuilt dataset uit stap 6.

Gebruik
-------
    python run_grid.py \
        --config /pad/naar/hpsearch_config.yaml \
        --dataset-path /pad/naar/hpo_dataset \
        --out-dir /pad/naar/grid_run \
        --models decision_tree --cv-folds 5 --valid --no-test

    # subset: bepaalde grids / operators
    python run_grid.py --config <cfg> --dataset-path <ds> --out-dir <out> \
        --models xgboost --grids sweep_cart --operators x
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from modelling_pipeline import make_effective_config, run_grid_search  # noqa: E402


def _split(s):
    return [x.strip() for x in s.replace("|", ",").split(",") if x.strip()] if s else None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Pure-Python gridsearch modelling runner (organisatie).")
    p.add_argument("--config", required=True, type=Path, help="Basis-config yaml (bijv. hpsearch_config.yaml).")
    p.add_argument("--out-dir", required=True, type=Path, help="Outputmap voor de grid-resultaten.")
    p.add_argument("--dataset-path", type=Path, default=None, help="Prebuilt dataset (stap 6) voor het snelle pad.")
    p.add_argument("--data-dir", type=Path, default=None, help="Override data_dir (als geen dataset-path).")
    p.add_argument("--operators", default=None, help="Operator-selectie ',' of '|' gescheiden.")
    p.add_argument("--models", default=None, help="Model-subset ',' of '|' gescheiden (default: alle uit config).")
    p.add_argument("--grids", default=None, help="Grid-subset ',' of '|' gescheiden (default: alle van het model).")
    p.add_argument("--run-variants", default=None, help="Run-variant-subset ',' of '|' gescheiden.")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--valid", dest="run_valid", action="store_true", default=True)
    p.add_argument("--no-valid", dest="run_valid", action="store_false")
    p.add_argument("--test", dest="run_test", action="store_true", default=True)
    p.add_argument("--no-test", dest="run_test", action="store_false")
    p.add_argument("--round", dest="do_round", action="store_true", default=False)
    p.add_argument("--random-state", type=int, default=None)
    p.add_argument("--Niels_Identity_Confounding_switch", "--niels-identity-confounding-switch",
                   dest="Niels_Identity_Confounding_switch", action=argparse.BooleanOptionalAction,
                   default=None, help="Scheid personen; default aan, of volg de YAML-instelling.")
    p.add_argument("--niels-identity-column", dest="Niels_identity_column", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = make_effective_config(
        args.config,
        dataset_path=args.dataset_path,
        data_dir=args.data_dir,
        operators=_split(args.operators),
    )
    if args.Niels_Identity_Confounding_switch is not None:
        cfg["Niels_Identity_Confounding_switch"] = args.Niels_Identity_Confounding_switch
    if args.Niels_identity_column is not None:
        cfg["Niels_identity_column"] = args.Niels_identity_column
    if args.random_state is not None:
        cfg["random_state"] = args.random_state
    model_seed = 23 if args.random_state is None else args.random_state
    print(f"[run_grid] config       : {args.config}")
    print(f"[run_grid] dataset-path : {args.dataset_path or '(geen — bouwt uit data_dir)'}")
    print(f"[run_grid] out-dir      : {args.out_dir}")

    t0 = time.perf_counter()
    produced = run_grid_search(
        cfg, args.out_dir,
        models=_split(args.models), grids=_split(args.grids),
        operators=_split(args.operators), run_variants=_split(args.run_variants),
        cv_folds=args.cv_folds, run_valid=args.run_valid, run_test=args.run_test,
        do_round=args.do_round, random_state=model_seed,
    )
    print(f"\n[run_grid] ✅ Klaar in {time.perf_counter()-t0:,.2f}s — {len(produced)} taken")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
