#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_merge_sample.py  (organisatie-versie)
=================================

Pure-Python *runner* voor stap 6 (**Samenvoegen & samplen van alle operators**) uit de
organisatie-README (origineel: stap 13). Vervangt:

    run_prepare_hpo_dataset.sbatch  →  prepare_hpo_dataset.py

Voegt de per-operator merged feature-CSV's samen tot één gecombineerde dataset (incl. de
ALL-stats uit stap 3b, 26-koloms a..z OHE en ACTIVE_FLAG uit stap 4), bepaalt de target en
past optioneel negatieve undersampling (1:ratio) toe. Output: `valid_sampled.pkl`,
optioneel `test_full.pkl` en `meta.json` in `--dataset-path`.

Gebruik
-------
    # Met losse vlaggen (config wordt automatisch samengesteld):
    python run_merge_sample.py \
        --data-dir /pad/naar/feature_files \
        --operators x \
        --validation-period-prefixes 01062025_30062025_01072025_31072025_valid \
        --sampling-ratio 20 \
        --dataset-path /pad/naar/hpo_dataset

    # Of: geef rechtstreeks een effectieve config-yaml mee (puur faithful):
    python run_merge_sample.py --config /pad/naar/hpsearch_config_effective.yaml

Met `--sampling-ratio 0` (default) wordt niet gesampled. `FILTER_ACTIVE=0` in de omgeving
schakelt de ACTIVE_FLAG-filtering uit (default aan).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Zorg dat de modules naast dit script importeerbaar zijn, ongeacht vanwaar je het start.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from merge_sample_pipeline import build_config, merge_and_sample  # noqa: E402


def _split(s: str | None) -> list[str]:
    if not s:
        return []
    return [p.strip() for p in s.replace("|", ",").split(",") if p.strip()]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure-Python merge+sample runner (organisatie): bouw de gecombineerde HPO-dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Effectieve config-yaml. Identiteitsswitch, spelerskolom en split-seed kunnen expliciet worden overschreven.",
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Map met per-operator submappen (data_dir/<op>/).")
    parser.add_argument("--operators", default=None,
                        help="Operators, ',' of '|' gescheiden (bijv. 'x' of 'a,b,c').")
    parser.add_argument("--validation-period-prefixes", default=None,
                        help="Validation period-prefixes, ',' of '|' gescheiden.")
    parser.add_argument("--test-period-prefixes", default=None,
                        help="Test period-prefixes (optioneel).")
    parser.add_argument("--dataset-path", type=Path, default=None,
                        help="Outputmap voor valid_sampled.pkl / test_full.pkl / meta.json.")
    parser.add_argument("--sampling-ratio", type=int, default=0,
                        help="0=geen sampling; N=alle positives + N× zoveel negatives.")
    parser.add_argument("--target-col", default="",
                        help="Expliciete target-kolom (leeg = automatisch afleiden).")
    parser.add_argument("--base-scenario", default="Flexible_spanish_plus",
                        help="Base-scenario naam in de bestandsnamen (default: Flexible_spanish_plus).")
    parser.add_argument("--all-scenario-name", default="ALL",
                        help="Naam van de ALL-stats bestanden (default: ALL).")
    parser.add_argument("--Niels_Identity_Confounding_switch", "--niels-identity-confounding-switch",
                        dest="Niels_Identity_Confounding_switch", action=argparse.BooleanOptionalAction,
                        default=None, help="Scheid identiteiten; default aan, of volg de YAML-instelling.")
    parser.add_argument("--niels-identity-column", dest="Niels_identity_column", default=None,
                        help="Spelerskolom in featurebestanden; default Player_Profile_ID. Aanbieder blijft deel van de sleutel.")
    parser.add_argument("--random-state", type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.config is not None:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        print(f"[run_merge_sample] config (yaml): {args.config}")
    else:
        missing = [n for n, v in [
            ("--data-dir", args.data_dir),
            ("--operators", args.operators),
            ("--validation-period-prefixes", args.validation_period_prefixes),
            ("--dataset-path", args.dataset_path),
        ] if not v]
        if missing:
            sys.exit(f"❌ Ontbrekende vlaggen (of gebruik --config): {', '.join(missing)}")

        cfg = build_config(
            dataset_path=args.dataset_path,
            data_dir=args.data_dir,
            validation_period_prefixes=_split(args.validation_period_prefixes),
            test_period_prefixes=_split(args.test_period_prefixes),
            all_operators=_split(args.operators),
            sampling_ratio=args.sampling_ratio,
            target_col=args.target_col,
            base_scenario=args.base_scenario,
            all_scenario_name=args.all_scenario_name,
            Niels_Identity_Confounding_switch=(True if args.Niels_Identity_Confounding_switch is None
                                               else args.Niels_Identity_Confounding_switch),
            Niels_identity_column=args.Niels_identity_column,
            random_state=23 if args.random_state is None else args.random_state,
        )
        print(f"[run_merge_sample] data-dir       : {args.data_dir}")
        print(f"[run_merge_sample] operators      : {cfg['all_operators']}")
        print(f"[run_merge_sample] valid-prefixes : {cfg['validation_period_prefixes']}")
        print(f"[run_merge_sample] sampling-ratio : {cfg['sampling_ratio']}")
        print(f"[run_merge_sample] dataset-path   : {args.dataset_path}")

    if args.Niels_Identity_Confounding_switch is not None:
        cfg["Niels_Identity_Confounding_switch"] = args.Niels_Identity_Confounding_switch
    if args.Niels_identity_column is not None:
        cfg["Niels_identity_column"] = args.Niels_identity_column
    if args.random_state is not None:
        cfg["random_state"] = args.random_state

    t0 = time.perf_counter()
    out = merge_and_sample(cfg)
    dt = time.perf_counter() - t0

    print(f"\n[run_merge_sample] ✅ Klaar in {dt:,.2f}s")
    print(f"[run_merge_sample] Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
