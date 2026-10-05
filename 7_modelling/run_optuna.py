#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_optuna.py  (organisatie-versie)
===========================

Pure-Python *runner* voor de modelling-stap via **Optuna** (organisatie-stap 8 / origineel 15, het
optuna-pad). Vervangt `submit_hpsearch.sh --optuna ... --all` → `optuna_runner.py`.

Eén proces, tijd-gebudgetteerde Optuna-search over alle (niet-uitgesloten) modellen; met
`--validate-best` worden de beste trials op cv=5 herijkt en — als er een `test_full.pkl` is —
op de testperiode geëvalueerd. Draait op de prebuilt dataset uit stap 6 (snelle pad).

Gebruik
-------
    python run_optuna.py \
        --config /pad/naar/hpsearch_config.yaml \
        --dataset-path /pad/naar/hpo_dataset \
        --out-dir /pad/naar/optuna_run \
        --time-budget 14400 --validate-best
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from modelling_pipeline import make_effective_config, run_optuna_search  # noqa: E402


def _split(s):
    return [x.strip() for x in s.replace("|", ",").split(",") if x.strip()] if s else None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Pure-Python Optuna modelling runner (organisatie).")
    p.add_argument("--config", required=True, type=Path, help="Basis-config yaml (bijv. hpsearch_config.yaml).")
    p.add_argument("--out-dir", required=True, type=Path, help="Outputmap voor de optuna-resultaten.")
    p.add_argument("--dataset-path", type=Path, default=None, help="Prebuilt dataset (stap 6) voor het snelle pad.")
    p.add_argument("--data-dir", type=Path, default=None, help="Override data_dir (als geen dataset-path).")
    p.add_argument("--operators", default=None, help="Operator-selectie ',' of '|' gescheiden.")
    p.add_argument("--exclude-models", default=None, help="Modellen uitsluiten ',' of '|' gescheiden.")
    p.add_argument("--time-budget", type=int, default=12600, help="Tijdsbudget in seconden.")
    p.add_argument("--cv-folds", type=int, default=1)
    p.add_argument("--n-startup", type=int, default=20)
    p.add_argument("--validate-best", action="store_true", default=False)
    p.add_argument("--cv5-top-n", type=int, default=5)
    p.add_argument("--round", dest="do_round", action="store_true", default=False)
    p.add_argument("--no-multivariate", action="store_true", default=False)
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
        exclude_models=_split(args.exclude_models),
    )
    if args.Niels_Identity_Confounding_switch is not None:
        cfg["Niels_Identity_Confounding_switch"] = args.Niels_Identity_Confounding_switch
    if args.Niels_identity_column is not None:
        cfg["Niels_identity_column"] = args.Niels_identity_column
    if args.random_state is not None:
        cfg["random_state"] = args.random_state
    model_seed = 23 if args.random_state is None else args.random_state
    print(f"[run_optuna] config       : {args.config}")
    print(f"[run_optuna] dataset-path : {args.dataset_path or '(geen — bouwt uit data_dir)'}")
    print(f"[run_optuna] out-dir      : {args.out_dir}")
    print(f"[run_optuna] time-budget  : {args.time_budget}s  validate_best={args.validate_best}")

    t0 = time.perf_counter()
    out = run_optuna_search(
        cfg, args.out_dir,
        time_budget=args.time_budget, cv_folds=args.cv_folds, n_startup=args.n_startup,
        validate_best=args.validate_best, cv5_top_n=args.cv5_top_n,
        do_round=args.do_round, no_multivariate=args.no_multivariate,
        random_state=model_seed,
    )
    print(f"\n[run_optuna] ✅ Klaar in {time.perf_counter()-t0:,.2f}s — output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
