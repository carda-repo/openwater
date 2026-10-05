#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py  —  ÉÉN entrypoint voor de hele organisatie-pijplijn (stappen 1 t/m 9)
==============================================================================

Dit is de "main" van `_organisatie_code`: één functie `run_full_pipeline(...)` die de volledige rit
in één keer op jouw bestanden draait, met **alle** belangrijke parameters expliciet (en met
defaults), zodat duidelijk is wat er in elke stap gebeurt.

Waarom geen echte Python-package?
---------------------------------
De stapmappen heten `1_cleaning`, `2_sorting`, ... — die namen zijn geen geldige
Python-modulenamen, dus ze zijn niet als package te importeren. Daarom zet deze main de
stapmappen op `sys.path` en importeert de pipeline-functies daaruit. Functioneel werkt dit als
de package-entrypoint, zonder de mappen te hoeven hernoemen.

Verwachte input-layout (zoals uit de organisatie-systemen / `_maak_fake`):
    <data_dir>/Operator_1 .. Operator_N/   met de WOK_*-CSV's (opgesplitst in _1/_2 mag).

Wat er per stap gebeurt (zie de banners in de output):
    1  cleaning   — schoonmaken volgens schema + automatische outlier-labelling (stap 1+ labelling)
    2  sorting    — chronologisch sorteren per tabel (tabellen zonder datumkolom: ongesorteerd door)
    3  features   — feature-engineering per scenario over de x-periode + target over de y-periode
    3b operator-stats — ALL-aggregaat (mean/std/min/max van kernfeatures) per operator
    4  active-filter   — last-status lookup → ACTIVE_FLAG per cutoff
    6  merge & sample  — combineer operators tot één HPO-dataset (+ optionele undersampling)
    7/8 modelling       — Optuna (default) of gridsearch
    10/11 rapportage    — bij gridsearch: dekkingsgraad + modelrapport; bij Optuna: de optuna-output

Overgeslagen omdat ze al onderdeel van een andere stap zijn (zie README):
    - outlier-labelling: zit in stap 1 (cleaning).
    - descriptive (stap 5): losse zij-analyse, niet nodig voor model + rapport.
    - logging (stap 9): dwarsliggend; wrap deze main desgewenst met `_logging/run_with_log.py`.

Gebruik (Python):
    from run_pipeline import run_full_pipeline
    art = run_full_pipeline(data_dir="/pad/Operator_mappen", out_dir="/pad/output",
                            x_tijdspad="01062025:30062025", y_tijdspad="01072025:31072025")

Gebruik (CLI):
    python run_pipeline.py --data-dir /pad --out-dir /pad/out \
        --x-tijdspad 01062025:30062025 --y-tijdspad 01072025:31072025
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# --- Zet de stapmappen op sys.path (3_features eerst i.v.m. gedeelde path_finding) ---
_organisatie = Path(__file__).resolve().parent
for _sub in ("3_features", "1_cleaning", "2_sorting", "3b_operator_specific_features",
             "4_active_filter", "5_descriptive", "6_merge_sample", "7_modelling", "8_rapportages"):
    _p = str(_organisatie / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from clean_pipeline import clean_directory                          # noqa: E402  (stap 1)
from sort_pipeline import sort_directory                            # noqa: E402  (stap 2)
from parse_pipeline import build_features                           # noqa: E402  (stap 3)
from operator_features_pipeline import build_all_stats_for_file     # noqa: E402  (stap 3b)
from active_filter_pipeline import build_status_lookups             # noqa: E402  (stap 4)
from merge_sample_pipeline import build_config, merge_and_sample    # noqa: E402  (stap 6)
from modelling_pipeline import run_optuna_search, run_grid_search   # noqa: E402  (stap 7/8)


def _banner(nr: str, titel: str) -> None:
    print("\n" + "=" * 78)
    print(f"  STAP {nr}: {titel}")
    print("=" * 78, flush=True)


def _period_prefix(x_tijdspad: str, y_tijdspad: str, pass_label: str) -> str:
    """Bouw een period-prefix 'xstart_xend_ystart_yend_pass' (DDMMYYYY) uit de tijdspaden."""
    xs, xe = x_tijdspad.split(":")
    ys, ye = y_tijdspad.split(":")
    return f"{xs}_{xe}_{ys}_{ye}_{pass_label}"


def _parse_prefix(prefix: str) -> tuple[str, str]:
    """Inverse van _period_prefix: 'xs_xe_ys_ye_label' → (x_tijdspad 'xs:xe', y_tijdspad 'ys:ye').

    Elk period-prefix draagt z'n eigen x- én y-venster (zoals in hpsearch_config.yaml).
    Het label (valid/test) wordt genegeerd.
    """
    parts = prefix.split("_")
    if len(parts) < 4:
        raise ValueError(
            f"ongeldig period-prefix '{prefix}': verwacht 'xstart_xend_ystart_yend_label' (DDMMYYYY)"
        )
    xs, xe, ys, ye = parts[0], parts[1], parts[2], parts[3]
    return f"{xs}:{xe}", f"{ys}:{ye}"


def _make_operator_folds(operators: List[str], n_folds: int, seed: int) -> List[Dict[str, list]]:
    """Splits de operators in `n_folds` folds (zoals submit_hpsearch.sh): husselen met `seed`,
    elk fold = {train: operators NIET in fold, holdout: operators in fold}. 1-op-1 uit het origineel."""
    import math
    import random as _random
    ops = sorted(operators)
    rng = _random.Random(seed)
    shuffled = ops[:]
    rng.shuffle(shuffled)
    fold_size = math.ceil(len(shuffled) / n_folds)
    folds: List[Dict[str, list]] = []
    for i in range(n_folds):
        holdout = shuffled[i * fold_size:(i + 1) * fold_size]
        train = [op for op in ops if op not in set(holdout)]
        folds.append({"train": train, "holdout": holdout})
    return folds


def run_full_pipeline(
    # ── Invoer/uitvoer ───────────────────────────────────────────────────────
    data_dir: str | Path,
    out_dir: str | Path,
    operators: Optional[List[str]] = None,        # default: alle Operator_*-mappen in data_dir
    label: Optional[str] = None,                   # cosmetisch run-label (origineel: --label)
    # ── Periodes (x = features, y = target) ──────────────────────────────────
    # PRODUCTIE (origineel-getrouw, lek-vrij): volledig gespecificeerde period-prefixes
    # 'xstart_xend_ystart_yend_label' (DDMMYYYY). validation en test zijn ONAFHANKELIJKE lijsten,
    # per index gepaard (p0/p1/p2 = lookback-horizonten). Elk prefix draagt z'n eigen x- én y-venster.
    validation_period_prefixes: Optional[List[str]] = None,
    test_period_prefixes: Optional[List[str]] = None,
    # GEMAK (smoke-test): één valid-venster + optioneel een APART test-venster.
    x_tijdspad: str = "01062025:30062025",
    y_tijdspad: str = "01072025:31072025",
    x_test_tijdspad: Optional[str] = None,
    y_test_tijdspad: Optional[str] = None,
    # ── Stap 1 cleaning ──────────────────────────────────────────────────────
    chunksize: int = 400_000,
    do_label: bool = True,                         # automatische outlier-labelling in cleaning
    clean_glob: str = "*.csv",
    only_first_chunk: bool = False,                # snelle test op alleen de eerste chunk
    # ── Stap 3 features ──────────────────────────────────────────────────────
    scenario: str = "Flexible_spanish_plus",       # base-scenario (moet de kernfeatures opleveren)
    single_feature: Optional[str] = None,          # alleen deze ene feature (leeg = alle)
    # ── Stap 3b operator-features (totaalalgoritme) ──────────────────────────
    do_operator_stats: bool = True,                # ALL-aggregaat (mean/std/min/max kernfeatures); zet False om uit te zetten
    add_ohe: bool = True,                           # 26-koloms a–z one-hot per operator (in merge); zet False om uit te zetten
    ignore_EOD_Balance: bool = False,              # zonder Player_Profile_EOD_Balance blijven f26/f27/f28 onbekend
    # ── Stap 4 active-filter ─────────────────────────────────────────────────
    do_active_filter: bool = True,
    also_inactives: bool = False,                  # origineel --also-inactives: niet filteren op actieve spelers
    # ── Stap 6 merge & sample ────────────────────────────────────────────────
    sampling_ratio: int = 0,                       # 0 = geen sampling; N = 1 positive : N negatives
    target_col: str = "",                          # leeg = automatisch afleiden uit prefix
    Niels_Identity_Confounding_switch: bool = True, # personen scheiden tussen train/valid/test
    Niels_identity_column: Optional[str] = None,  # spelerskolom in feature-CSV; default Player_Profile_ID
    exclude_models: Optional[List[str]] = None,
    # ── Stap 7/8 modelling ───────────────────────────────────────────────────
    search: str = "optuna",                        # 'optuna' of 'grid'
    random_state: int = 23,
    do_round: bool = False,                        # dtype-afronding voor snelheid
    # Optuna-specifiek
    optuna_time_budget: int = 900,                 # seconden (default 15 min); origineel --optuna-time
    optuna_cv_folds: int = 1,
    optuna_n_startup: int = 20,
    optuna_validate_best: bool = False,            # origineel: opt-in (--validate_best); top-N cv=5 + test
    optuna_cv5_top_n: int = 5,
    optuna_no_multivariate: bool = False,
    only_these_vars: Optional[str] = None,         # origineel --only-these-vars: featurekolommen filteren
    # ── Operator-folds (operator-niveau CV; origineel --operator-folds/--independent-searches) ──
    operator_folds: int = 0,                       # 0 = uit; N = N folds (leave-operators-out)
    independent_searches: bool = False,            # N losse searches i.p.v. één averaged CV
    operator_fold_seed: int = 0,
    skip_all_check: bool = False,                  # origineel --skip-all-check
    # Grid-specifiek
    grid_models: Optional[List[str]] = None,       # None = alle modellen uit de config
    grid_grids: Optional[List[str]] = None,
    grid_cv_folds: int = 5,
    grid_config: Optional[str | Path] = None,      # basis hpsearch_config.yaml (vereist voor grid)
    # ── Stap 10/11 rapportage (alleen relevant bij grid) ─────────────────────
    do_report: bool = True,
    # ── Overig ───────────────────────────────────────────────────────────────
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Draai de volledige organisatie-pijplijn (1 t/m 9) op `data_dir` en schrijf alles onder `out_dir`.

    Retourneert een dict met de paden van alle tussen- en eindartefacten:
        {cleaned, sorted, feature_files_dir, dataset_path, model_out_dir, report}

    Alle parameters hierboven hebben expliciete defaults; pas aan naar jouw situatie.
    De `search='grid'`-modus vereist een basis-config (`grid_config`, bv. hpsearch_config.yaml)
    met `models`/`grids`/`run_variants`; `search='optuna'` heeft die niet nodig.
    """
    t_start = time.perf_counter()
    data_dir = Path(data_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if operators is None:
        operators = sorted(p.name for p in data_dir.glob("Operator_*") if p.is_dir())
    if not operators:
        raise FileNotFoundError(f"Geen Operator_*-mappen gevonden in {data_dir}")

    if also_inactives:
        do_active_filter = False   # origineel: --also-inactives → niet filteren op actieve spelers

    # Period-prefixes bepalen. Voorkeur (origineel-getrouw): expliciete, volledig gespecificeerde
    # valid- + test-lijsten → echte holdout. Anders: gemak-modus met één valid-venster + (evt.) apart test.
    if validation_period_prefixes or test_period_prefixes:
        if not validation_period_prefixes:
            raise ValueError("geef validation_period_prefixes (test_period_prefixes mag bij operator-folds weg)")
        valid_prefixes = list(validation_period_prefixes)
        if test_period_prefixes:
            if len(validation_period_prefixes) != len(test_period_prefixes):
                raise ValueError(
                    f"validation_period_prefixes ({len(validation_period_prefixes)}) en "
                    f"test_period_prefixes ({len(test_period_prefixes)}) moeten even lang zijn (p0/p1/p2)."
                )
            test_prefixes = list(test_period_prefixes)
        elif operator_folds:
            # Operator-folds: een aparte test-holdout is niet nodig (de engine bouwt per fold uit de
            # validation-prefixes). We hergebruiken de validation-prefixes als 'test' — enkel om de
            # p0/p1/p2 kolom-prefix-lengte kloppend te houden; er worden GEEN test-features gebouwd.
            test_prefixes = list(valid_prefixes)
        else:
            raise ValueError("geef ZOWEL validation_period_prefixes ALS test_period_prefixes "
                             "(of gebruik operator_folds, dan is test niet nodig)")
        window_specs = []
        for vpfx, tpfx in zip(valid_prefixes, test_prefixes):
            xv, yv = _parse_prefix(vpfx)
            xt, yt = _parse_prefix(tpfx)
            window_specs.append((xv, yv, xt, yt))
    else:
        x_test = x_test_tijdspad or x_tijdspad
        y_test = y_test_tijdspad or y_tijdspad
        window_specs = [(x_tijdspad, y_tijdspad, x_test, y_test)]
        valid_prefixes = [_period_prefix(x_tijdspad, y_tijdspad, "valid")]
        test_prefixes = [_period_prefix(x_test, y_test, "test")]
        if (x_test, y_test) == (x_tijdspad, y_tijdspad):
            if Niels_Identity_Confounding_switch:
                print("⚠️  GEEN temporele holdout: test-periode == valid-periode. "
                      "De identiteiten worden wel gescheiden; gebruik een latere testperiode "
                      "om toekomstige prestaties te beoordelen.")
            else:
                print("⚠️  GEEN holdout: test-periode == valid-periode (test spiegelt valid). "
                      "Alleen voor smoke-tests. Gebruik validation_period_prefixes + "
                      "test_period_prefixes (of x_test_tijdspad/y_test_tijdspad) voor een echte holdout.")

    # active-filter heeft een LAST_STATUS_LOOKUP per (distinct) y-start nodig
    y_starts = sorted({p.split("_")[2] for p in (valid_prefixes + test_prefixes)})

    print("#" * 78)
    print("#  organisatie FULL PIPELINE")
    print(f"#  data_dir={data_dir}")
    print(f"#  operators={operators}")
    print(f"#  scenario={scenario}  search={search}" + (f"  label={label}" if label else ""))
    print(f"#  periodes ({len(window_specs)}):")
    print(f"#    valid: {valid_prefixes}")
    print(f"#    test : {test_prefixes}")
    if operator_folds:
        print(f"#  operator-folds: {operator_folds} ({'independent' if independent_searches else 'averaged'})")
    print("#" * 78, flush=True)

    feat_root = out_dir / "feature_files"          # data_dir-layout voor stap 6
    cleaned_dirs: Dict[str, Path] = {}
    sorted_dirs: Dict[str, Path] = {}

    # ── STAPPEN 1–4 per operator ─────────────────────────────────────────────
    for op in operators:
        op_raw = data_dir / op
        if not op_raw.is_dir():
            print(f"[SKIP] {op}: map niet gevonden ({op_raw})")
            continue

        _banner("1", f"CLEANING + outlier-labelling — operator {op}")
        cleaned = clean_directory(
            input_dir=op_raw, clean_out_dir=out_dir / "clean" / op,
            glob_pattern=clean_glob, chunksize=chunksize,
            only_first_chunk=only_first_chunk, verbose=verbose, do_label=do_label,
        )
        cleaned_dirs[op] = cleaned

        _banner("2", f"SORTING op datum — operator {op}")
        sorted_dir = sort_directory(cleaned_dir=cleaned, chunksize=chunksize)
        sorted_dirs[op] = sorted_dir

        op_feat_dir = feat_root / op
        op_feat_dir.mkdir(parents=True, exist_ok=True)
        # Per periode-venster: valid-features (x_v, y_v), test-features (x_t, y_t) en ALL-stats.
        for i, ((xv, yv, xt, yt), vpfx, tpfx) in enumerate(zip(window_specs, valid_prefixes, test_prefixes)):
            _banner("3", f"FEATURES p{i} (scenario={scenario}, x={xv}) — operator {op}")
            feat_csv = op_feat_dir / f"{vpfx}_{scenario}.csv"
            build_features(
                cleaned_dir=sorted_dir, scenario=scenario, features_out=feat_csv,
                x_tijdspad=xv, y_tijdspad=yv, feature=single_feature,
                chunksize=chunksize, verbose=verbose, ignore_EOD_Balance=ignore_EOD_Balance,
            )
            # Test-features (overgeslagen bij operator-folds: de engine bouwt per fold uit validation).
            if not operator_folds:
                test_csv = op_feat_dir / f"{tpfx}_{scenario}.csv"
                if (xt, yt) != (xv, yv):
                    build_features(
                        cleaned_dir=sorted_dir, scenario=scenario, features_out=test_csv,
                        x_tijdspad=xt, y_tijdspad=yt, feature=single_feature,
                        chunksize=chunksize, verbose=verbose, ignore_EOD_Balance=ignore_EOD_Balance,
                    )
                else:
                    shutil.copy(feat_csv, test_csv)

            if do_operator_stats:
                _banner("3b", f"OPERATOR-STATS p{i} (ALL-aggregaat) — operator {op}")
                build_all_stats_for_file(feat_csv, out_path=op_feat_dir / f"{vpfx}_ALL_merged.csv")

        if do_active_filter:
            _banner("4", f"ACTIVE-FILTER (last-status lookups: {y_starts}) — operator {op}")
            build_status_lookups(sorted_dir, y_starts, out_dir=op_feat_dir, overwrite=True)

    # Operator-folds (leave-operators-out): bouw de folds. Bij folds slaan we de globale merge
    # over zodat de optuna-engine per fold VERS uit de features bouwt (niet-prebuilt; vereist door
    # de operator_folds-tak in optuna_runner).
    folds = _make_operator_folds(operators, operator_folds, operator_fold_seed) if operator_folds else None

    dataset_path = out_dir / "hpo_dataset"
    if folds is None:
        # ── STAP 6: merge & sample ───────────────────────────────────────────
        _banner("6", "SAMENVOEGEN & SAMPLEN (HPO-dataset bouwen)")
        cfg6 = build_config(
            dataset_path=dataset_path, data_dir=feat_root,
            validation_period_prefixes=valid_prefixes, test_period_prefixes=test_prefixes,
            all_operators=operators, sampling_ratio=sampling_ratio,
            target_col=target_col, base_scenario=scenario,
            Niels_Identity_Confounding_switch=Niels_Identity_Confounding_switch,
            Niels_identity_column=Niels_identity_column, random_state=random_state,
        )
        cfg6["add_operator_ohe"] = add_ohe   # OHE-kolommen wel/niet toevoegen in de merge
        if exclude_models:
            cfg6["exclude_models"] = list(exclude_models)
        merge_and_sample(cfg6)
    else:
        _banner("6", f"OPERATOR-FOLDS: globale merge overgeslagen — optuna bouwt per fold vers "
                      f"({operator_folds} folds, {'independent' if independent_searches else 'averaged'})")
        dataset_path.mkdir(parents=True, exist_ok=True)  # leeg → use_prebuilt=False

    # ── STAP 7/8: modelling ──────────────────────────────────────────────────
    model_out = out_dir / f"model_{search}"
    model_cfg = {
        "data_dir": str(feat_root), "dataset_path": str(dataset_path), "all_mode": True,
        "validation_period_prefixes": valid_prefixes, "test_period_prefixes": test_prefixes,
        "operators": operators, "all_operators": operators, "target_col": target_col,
        "base_scenario": scenario, "add_operator_ohe": add_ohe,
        "Niels_Identity_Confounding_switch": Niels_Identity_Confounding_switch,
        "Niels_identity_column": Niels_identity_column,
        "random_state": random_state,
    }
    if exclude_models:
        model_cfg["exclude_models"] = list(exclude_models)
    if skip_all_check:
        model_cfg["skip_all_check"] = True
    if only_these_vars:
        model_cfg["only_these_vars"] = only_these_vars

    _opt = dict(time_budget=optuna_time_budget, cv_folds=optuna_cv_folds, n_startup=optuna_n_startup,
                validate_best=optuna_validate_best, cv5_top_n=optuna_cv5_top_n,
                do_round=do_round, no_multivariate=optuna_no_multivariate, random_state=random_state)

    report_path: Optional[Path] = None
    if search == "optuna":
        if folds is not None and independent_searches:
            _banner("7/8", f"MODELLING via OPTUNA — {operator_folds} INDEPENDENTE searches (per fold)")
            for fi, fold in enumerate(folds):
                fold_out = model_out / f"fold{fi}"
                cfg_i = {**model_cfg, "dataset_path": str(fold_out),
                         "all_operators": fold["train"], "fold_idx": fi,
                         "fold_holdout_operators": fold["holdout"]}
                print(f"\n── fold {fi}: train={fold['train']}  holdout={fold['holdout']}")
                run_optuna_search(cfg_i, fold_out, **_opt)
        else:
            if folds is not None:
                model_cfg["operator_folds"] = folds   # averaged CV over de folds
            _banner("7/8", "MODELLING via OPTUNA (search + cv5 + test op aparte periode)")
            run_optuna_search(model_cfg, model_out, **_opt)
        # Bij Optuna ís de optuna-output de rapportage.
        report_path = model_out
        print(f"\n[REPORT] Optuna-rapportage in: {model_out} "
              f"(optuna_meta.json, optuna_cv5_results.csv, optuna_best_test_result.csv, "
              f"optuna_second_half_report.txt, top_model_statistics.txt)")
    elif search == "grid":
        if grid_config is None:
            raise ValueError("search='grid' vereist een basis-config via grid_config (bv. hpsearch_config.yaml)")
        _banner("7", "MODELLING via GRIDSEARCH")
        import yaml
        from modelling_pipeline import make_effective_config
        base_cfg = yaml.safe_load(Path(grid_config).read_text(encoding="utf-8"))
        base_cfg.update(model_cfg)
        eff = make_effective_config(base_cfg, dataset_path=dataset_path)
        run_grid_search(
            eff, model_out, models=grid_models, grids=grid_grids,
            operators=operators, cv_folds=grid_cv_folds,
            run_valid=True, run_test=True, do_round=do_round, random_state=random_state,
        )
        if do_report:
            _banner("10/11", "RAPPORTAGE (dekkingsgraad + modelrapport)")
            from run_report import main as report_main
            report_main(["--run-dir", str(model_out)])
            report_path = model_out / f"{model_out.name}_report.txt"
    else:
        raise ValueError(f"Onbekende search='{search}' (kies 'optuna' of 'grid')")

    dt = time.perf_counter() - t_start
    print("\n" + "#" * 78)
    print(f"#  ✅ KLAAR in {dt:,.1f}s")
    print(f"#  cleaned   : {out_dir / 'clean'}")
    print(f"#  features  : {feat_root}")
    print(f"#  dataset   : {dataset_path}")
    print(f"#  model     : {model_out}")
    print(f"#  rapport   : {report_path}")
    print("#" * 78, flush=True)

    return {
        "cleaned": cleaned_dirs, "sorted": sorted_dirs,
        "feature_files_dir": feat_root, "dataset_path": dataset_path,
        "model_out_dir": model_out, "report": report_path,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _split(s):
    return [x.strip() for x in s.replace("|", ",").split(",") if x.strip()] if s else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Draai de volledige organisatie-pijplijn (1 t/m 9) in één keer. "
                    "Vlaggen volgen waar mogelijk submit_hpsearch.sh.")
    # I/O — specifiek voor de pure-Python pijplijn (features worden hier zelf gebouwd)
    p.add_argument("--data-dir", required=True, help="Map met Operator_*-submappen (ruwe WOK-CSV's).")
    p.add_argument("--out-dir", required=True, help="Outputmap voor alle tussen- en eindresultaten.")
    p.add_argument("--operators", default=None, help="Subset operators ',' of '|' gescheiden (default: alle).")
    p.add_argument("--label", default=None, help="Cosmetisch run-label.")
    p.add_argument("--base-scenario", dest="scenario", default="Flexible_spanish_plus")
    # Periodes — origineel-getrouw: volledig gespecificeerde prefixes (valid + test onafhankelijk)
    p.add_argument("--validation-period-prefixes", default=None,
                   help="Volledige valid-prefixes 'xs_xe_ys_ye_valid' (DDMMYYYY), ',' of '|' gescheiden (p0/p1/p2).")
    p.add_argument("--test-period-prefixes", default=None,
                   help="Bijbehorende test-prefixes 'xs_xe_ys_ye_test', per index gepaard → echte holdout.")
    # Gemak-modus (single period)
    p.add_argument("--x-tijdspad", default="01062025:30062025", help="Gemak-modus: één valid x-venster.")
    p.add_argument("--y-tijdspad", default="01072025:31072025")
    p.add_argument("--x-test-tijdspad", default=None, help="Apart test-x-venster (anders = valid → geen holdout).")
    p.add_argument("--y-test-tijdspad", default=None)
    # Modelling — namen zoals submit_hpsearch.sh
    p.add_argument("--optuna", dest="search", action="store_const", const="optuna", default="optuna",
                   help="Optuna-modus (default).")
    p.add_argument("--grid", dest="search", action="store_const", const="grid",
                   help="Gridsearch-modus (vereist --config).")
    p.add_argument("--optuna-time", dest="optuna_time_budget", type=int, default=900,
                   help="Optuna tijdsbudget in seconden.")
    p.add_argument("--cv-folds", dest="optuna_cv_folds", type=int, default=1)
    p.add_argument("--n-startup", dest="optuna_n_startup", type=int, default=20)
    p.add_argument("--cv5-top-n", dest="optuna_cv5_top_n", type=int, default=5)
    p.add_argument("--validate_best", dest="optuna_validate_best", action="store_true", default=False,
                   help="Top-N op cv=5 + test op aparte periode (origineel: opt-in).")
    p.add_argument("--no-multivariate", dest="optuna_no_multivariate", action="store_true", default=False)
    p.add_argument("--all", dest="all_mode", action="store_true", default=False,
                   help="No-op: deze pijplijn draait altijd op de gecombineerde all-operators-dataset.")
    p.add_argument("--round", dest="do_round", action="store_true", default=False)
    p.add_argument("--sampling", dest="sampling_ratio", type=int, default=0,
                   help="1 positive : N negatives (0 = geen sampling).")
    p.add_argument("--exclude-models", default=None)
    p.add_argument("--only-these-vars", dest="only_these_vars", default=None,
                   help="Filter featurekolommen (| of , gescheiden).")
    p.add_argument("--random-state", type=int, default=23)
    p.add_argument("--Niels_Identity_Confounding_switch", "--niels-identity-confounding-switch",
                   dest="Niels_Identity_Confounding_switch", action=argparse.BooleanOptionalAction,
                   default=True, help="Scheid personen tussen training, validatie en test (default aan).")
    p.add_argument("--niels-identity-column", dest="Niels_identity_column", default=None,
                   help="Spelerskolom in featurebestanden; default Player_Profile_ID. Aanbieder blijft deel van de sleutel.")
    p.add_argument("--config", dest="grid_config", default=None,
                   help="hpsearch_config.yaml (vereist bij --grid).")
    # Operator-folds (operator-niveau CV)
    p.add_argument("--operator-folds", dest="operator_folds", type=int, default=0,
                   help="N operator-folds (leave-operators-out).")
    p.add_argument("--independent-searches", dest="independent_searches", action="store_true", default=False,
                   help="N losse searches i.p.v. één averaged CV.")
    p.add_argument("--operator-fold-seed", dest="operator_fold_seed", type=int, default=0)
    p.add_argument("--skip-all-check", dest="skip_all_check", action="store_true", default=False)
    p.add_argument("--also-inactives", dest="also_inactives", action="store_true", default=False,
                   help="Niet filteren op actieve spelers (active-filter uit).")
    # Feature-opties specifiek voor deze pure-Python pijplijn
    p.add_argument("--no-operator-stats", dest="do_operator_stats", action="store_false", default=True,
                   help="Zet de ALL-aggregaat operator-stats (stap 3b) uit.")
    p.add_argument("--no-ohe", dest="add_ohe", action="store_false", default=True,
                   help="Zet de 26-koloms a–z one-hot per operator uit.")
    p.add_argument("--ignore-eod-balance", dest="ignore_EOD_Balance", action="store_true", default=False,
                   help="Dataset zonder Player_Profile_EOD_Balance: f26/f27/f28 blijven onbekend i.p.v. crashen.")
    p.add_argument("--chunksize", type=int, default=400_000)
    p.add_argument("--verbose", action="store_true", default=False)
    args = p.parse_args(argv)

    run_full_pipeline(
        data_dir=args.data_dir, out_dir=args.out_dir, operators=_split(args.operators),
        label=args.label,
        validation_period_prefixes=_split(args.validation_period_prefixes),
        test_period_prefixes=_split(args.test_period_prefixes),
        x_tijdspad=args.x_tijdspad, y_tijdspad=args.y_tijdspad,
        x_test_tijdspad=args.x_test_tijdspad, y_test_tijdspad=args.y_test_tijdspad,
        do_operator_stats=args.do_operator_stats, add_ohe=args.add_ohe,
        ignore_EOD_Balance=args.ignore_EOD_Balance, also_inactives=args.also_inactives,
        scenario=args.scenario, chunksize=args.chunksize, sampling_ratio=args.sampling_ratio,
        search=args.search, optuna_time_budget=args.optuna_time_budget,
        optuna_cv_folds=args.optuna_cv_folds, optuna_n_startup=args.optuna_n_startup,
        optuna_validate_best=args.optuna_validate_best, optuna_cv5_top_n=args.optuna_cv5_top_n,
        optuna_no_multivariate=args.optuna_no_multivariate, do_round=args.do_round,
        only_these_vars=args.only_these_vars,
        operator_folds=args.operator_folds, independent_searches=args.independent_searches,
        operator_fold_seed=args.operator_fold_seed, skip_all_check=args.skip_all_check,
        exclude_models=_split(args.exclude_models), grid_config=args.grid_config,
        random_state=args.random_state, verbose=args.verbose,
        Niels_Identity_Confounding_switch=args.Niels_Identity_Confounding_switch,
        Niels_identity_column=args.Niels_identity_column,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
