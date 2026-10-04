# Organisation version — Pure Python pipeline

> 🇬🇧 **English first.** 🇳🇱 Nederlandse versie staat onderaan ([spring erheen](#-nederlands)).

This folder (`openwater`) contains the **organisation variant** of the gambling-addiction-prevention
pipeline. Where the original ran on an HPC cluster with SLURM `sbatch` jobs, a SLURM array per operator
(a..z) and `$TMPDIR` staging with symlinks, the organisation just
works with **local CSVs** and **pure Python**. You kick off each step with a `.py` runner — no
`sbatch`, no `.sh`, no environment variables needed.

That's why the structure and numbering differ:

- The steps start here at **1** (instead of 4 in the original).
- The original steps **4. CLEANING** and **5. Outlier detection via labelling** are merged into
  **step 1 (Cleaning & outlier labelling)**, because the cleaning already takes the labelling along
  automatically and the standalone labelling step reuses the same code.
- All following steps shift along (original 6 → organisation 2, 7 → 3, and so on).

> **Porting approach.** Code is never modified outside ``. Per step:
> (a) figure out which code is needed (dependencies), (b) copy it **unchanged** to
> `<step>/`, (c) adapt it there into a pure-Python runner, (d) write tests in
> `9_testing/` against the dummy data in `0_dummy_data/voorbeeld_fake/`.

## Folder structure

```
openwater/
├── 1_cleaning/                     # step 1: cleaning & outlier labelling (pure Python)
├── 2_sorting/                      # step 2: sorting on date (pure Python)
├── 3_features/                     # step 3: build features per scenario (pure Python)
├── 3b_operator_specific_features/  # step 3b: ALL-aggregate features for the total algorithm
├── 4_active_filter/                # step 4: filter on 'Active' (last-status lookup per cutoff)
├── 5_descriptive/                  # step 5: descriptive tasks (report + plots)
├── 6_merge_sample/                 # step 6: merge & sample (build HPO dataset)
├── 7_modelling/                    # steps 7 & 8: train models (grid + optuna)
├── 8_rapportages/                  # steps 10 & 11: coverage + model report
├── _logging/                       # step 9: logging helper (run output → one log file)
├── 9_testing/                      # tests against 0_dummy_data/voorbeeld_fake/
├── 0_dummy_data/                   # test data, generators + data-delivery spec
│   ├── voorbeeld_fake/             # bundled relational test set (100 players, multi-period, signal)
│   ├── voorbeeld_noise/            # same, but label-independent features (leakage canary)
│   ├── voorbeeld_11/               # minimal masked structure sample (real export shape)
│   ├── _maak_fake.py               # generator for the relational test data (→ voorbeeld_fake/)
│   ├── _maak_fake_big_data.py      # streaming generator for large (GB-scale) datasets
│   ├── _maak_dummy_data.py         # alternative relational dummy generator
│   ├── _inventariseer_databehoefte.py  # derive the data-delivery spec from FEATURES_REGISTRY
│   └── databehoefte.txt            # generated data-delivery spec (which WOK files + columns)
├── run_pipeline.py                 # ⭐ single entrypoint: the whole run 1 through 9 at once
├── pyproject.toml                  # Poetry project + dependencies
├── README.md                       # this file (technical doc)
└── README_organisatie.md           # cookbook usage guide (EN + NL)
```

## Requirements

Python 3.11+ with `pandas`, `numpy`, `python-dateutil`, `scikit-learn`, `xgboost`, `lightgbm`,
`optuna`, `scipy`, `pyyaml`, `matplotlib` (the modelling/reporting steps). Optionally
`imbalanced-learn` for SMOTE variants.

## ⭐ Everything at once: `run_pipeline.py`

Don't want step-by-step, but the **whole pipeline (1 through 9)** in one call on your own files? Use
the orchestrator. It expects a folder with `Operator_*` subfolders (as from the organisation systems
/ `0_dummy_data/_maak_fake`) and runs: cleaning(+labelling) → sorting → features → 3b → active-filter →
merge&sample → modelling (Optuna or grid) → reporting.

The CLI flags follow the original `submit_hpsearch.sh` where possible. **Production (faithful to the
original, real holdout):** give fully specified period prefixes `xstart_xend_ystart_yend_label`
(DDMMYYYY); `validation` and `test` are independent lists, paired by index (p0/p1/p2):

```bash
cd openwater
python run_pipeline.py --data-dir /path/with/Operator_folders --out-dir /path/output \
    --base-scenario Flexible_spanish_plus \
    --validation-period-prefixes "01052025_31052025_01062025_30062025_valid,01122024_31052025_01062025_30062025_valid,01062024_31052025_01062025_30062025_valid" \
    --test-period-prefixes       "01052025_30062025_01072025_31072025_test,01122024_30062025_01072025_31072025_test,01062024_30062025_01072025_31072025_test" \
    --optuna --optuna-time 14400 --validate_best
```

**Operator-level CV** (leave-operators-out, like `--operator-folds`): then `--test-period-prefixes`
is not needed (folding replaces the holdout); `--independent-searches` does N separate searches
instead of one averaged CV:

```bash
python run_pipeline.py --data-dir ... --out-dir ... --base-scenario Flexible_spanish_plus \
    --validation-period-prefixes "..._valid,..._valid,..._valid" \
    --operator-folds 5 --independent-searches \
    --optuna --optuna-time 14400 --validate_best --no-multivariate --random-state 23
```

If your dataset lacks `Player_Profile_EOD_Balance` (like the minimal organisation set), add
`--ignore-eod-balance` (f26/f27/f28 then fall back instead of crashing).

Or from Python (all parameters have **explicit defaults** + explanation in the docstring):

```python
from run_pipeline import run_full_pipeline
artefacten = run_full_pipeline(
    data_dir="/path/with/Operator_folders", out_dir="/path/output",
    scenario="Flexible_spanish_plus",
    # PRODUCTION: independent valid/test prefixes (real temporal holdout):
    validation_period_prefixes=["01052025_31052025_01062025_30062025_valid", "..."],
    test_period_prefixes=["01052025_30062025_01072025_31072025_test", "..."],
    # CONVENIENCE (smoke test): single window + optional separate test window
    #   x_tijdspad="01062025:30062025", y_tijdspad="01072025:31072025",
    #   x_test_tijdspad="01072025:31072025", y_test_tijdspad="01082025:31082025",
    search="optuna", sampling_ratio=20, optuna_time_budget=14400, optuna_validate_best=True,
    # ignore_EOD_Balance=True,   # if your data has no Player_Profile_EOD_Balance
    # operator_folds=5, independent_searches=True,   # operator-level CV (test not needed then)
)
# → {cleaned, sorted, feature_files_dir, dataset_path, model_out_dir, report}
```

> **Real holdout (important).** `validation` and `test` are **independent** windows: the test period
> should be a different, later window (e.g. test-y one month after valid-y) → a real temporal
> holdout. The convenience mode without a separate test window mirrors test=valid (**no holdout**,
> with a warning) and is only for smoke tests. With `--operator-folds`, test is not needed (folding
> provides the holdout). With N validation prefixes each feature is computed N× (`p0_*`/`p1_*`/…) →
> the production config uses 3 lookback horizons (1m/6m/12m).

> **Flag alignment & nuances.** The flag names match `submit_hpsearch.sh`. Three differences follow
> from the absence of SLURM/YAML: `--validate_best` is **opt-in** (off by default), `--all` is a
> no-op (this pipeline always runs all-operators), and `--config` is optional for optuna (you pass
> test prefixes via the CLI instead of from the YAML).

[run_pipeline.py](run_pipeline.py) shows a banner per step describing what happens, and has all
important parameters explicit (periods, scenario, sampling, optuna time budget, validate-best,
exclude-models, grid vs optuna, etc.). The standalone per-step runners (below) remain available for
anyone who wants to run a single step separately.

> **NB (not a real Python package):** the step folders are named `1_cleaning`, `2_sorting`, … — not
> valid module names, so not importable as a package. `run_pipeline.py` solves this by putting the
> step folders on `sys.path`; it thus acts as the package entrypoint without renaming the folders.

### Generating test data

The bundled `0_dummy_data/voorbeeld_fake/` is produced by `0_dummy_data/_maak_fake.py` (relational format, 100 players,
multi-period by default). Regenerate or customise:

```bash
python 0_dummy_data/_maak_fake.py                                    # → voorbeeld_fake/ (100 players, multi-period)
python 0_dummy_data/_maak_fake.py --out my_set --players 250 --seed 7
python 0_dummy_data/_maak_fake.py --no-signal --out 0_dummy_data/voorbeeld_noise  # leakage canary (label-independent features)
```

- **multi-period (default)** — activity Jan–Apr 2026 and exclusions spread across **May and June**
  (two target months), so a **real temporal holdout** is possible: validation y=May, test y=June. A
  holdout/leakage check on a single target month (test=valid) is meaningless, so multi-period is the
  default. `--single-period` puts all exclusions in one month (no holdout).
- **`--no-signal`** — features are generated **independent** of the label (noise). On such data an
  honest model should reach **holdout test ≈0.5 AUC** → **leakage canary**: if it still scores high,
  the target leaks. (Default = exclusions gamble more heavily → learnable.)
- For very large sets use `0_dummy_data/_maak_fake_big_data.py` (streaming): `--target-gb` or `--players`.

> **Important (leakage check):** always assess a leakage canary on a **real holdout** (multi-period:
> validation and test target in different months). The one-shot without a separate test window
> mirrors `test = valid` → then even pure noise shows a high "test" score (memorisation, not a
> leak). Use the CV metric or a real holdout window.

## Running individual steps

```bash
cd 1_cleaning
python run_cleaning.py            # see step 1 below
```

---

# Steps

## 1. Cleaning & outlier labelling
*(original: step 4 CLEANING + step 5 Outlier detection via labelling)*

Status: **ported** → [1_cleaning/](1_cleaning/)

Cleaning reads each raw WOK CSV streaming in chunks, normalises column names and values per the
schema (`check_format.py`), applies optional operator-specific prevalidation filters, and writes the
clean CSVs to a new `cleaned_<timestamp>/` folder. Transaction deduplication then keeps the newest
row per player/transaction in `WOK_Player_Account_Transaction`, `WOK_Bet_Transaction` and
`WOK_Game_Session_Transaction`, across chunks and CSV split files. Operators are kept separate
using `operator_id`, or the linked parent table for bet/session transactions; without that
information, the input directory defines the operator scope. Account rows are ranked by
`extraction_date`, `_write_timestamp`, then `created_at`; bet/session rows by `created_at`,
`_write_timestamp`, then `extraction_date`. Missing timestamps fall back to the next available
field; ties or entirely missing timestamps keep the last row read (files in natural filename
order, then row order). IDs remain text; rows without a player or transaction ID remain separate.
The SQLite index uses disk storage, and counts are written to `logs/transaction_dedup.log` beside
the raw input. Afterwards the **outlier labelling**
runs automatically, adding the columns `outlier_Registration_Date` and
`outlier_Player_Profile_Modified` to the `WOK_Player_Profile` file.

The labelling can also be run **separately** (to save memory/time, or to repeat on an
already-cleaned folder) — that is the pure-Python equivalent of the original `--mode label_only`.

### Files

| File | Role |
|---|---|
| [run_cleaning.py](1_cleaning/run_cleaning.py) | Runner: clean a folder + automatic labelling. Replaces `run_ALL_clean_only.sbatch` → `clean_runner.py`. |
| [run_label_only.py](1_cleaning/run_label_only.py) | Runner: **only** labelling on an existing `cleaned_<stamp>/` folder. Replaces `run_ALL_label_only.sbatch` → `label_runner.py`. |
| [clean_pipeline.py](1_cleaning/clean_pipeline.py) | Orchestration (`clean_directory`, `label_only_directory`, `newest_cleaned_dir`). Stripped-down `clean_and_parse.py` — clean+label only, no feature parse. |
| [transaction_dedup.py](1_cleaning/transaction_dedup.py) | Keep the latest transaction per operator/player/ID across CSV parts, before labelling. |
| `cleaner.py`, `check_format.py`, `operator_filters.py`, `outlier_labeling.py`, `reading_difficult_json.py`, `path_finding.py` | Cleaning, validation, filters, labelling and readers based on the main repo. |

### Usage — cleaning

```bash
cd 1_cleaning

# Clean the default dummy data (../0_dummy_data/voorbeeld_fake) → cleaned_<stamp>/ next to it:
python run_cleaning.py

# Custom input/output folder:
python run_cleaning.py --input-dir /path/to/raw_csvs --clean-out-dir /path/to/output

# Quick test on only the first chunk (small TEST_cleaned_<stamp>/ folder):
python run_cleaning.py --onlyfirstchunk --chunksize 2000

# Without the automatic outlier labelling:
python run_cleaning.py --no-label

# With operator-specific prevalidation filter (original: OPERATOR_PREFIX):
python run_cleaning.py --operator-prefix b
```

### Usage — labelling only

```bash
cd 1_cleaning

# Label a concrete cleaned_<stamp>/ folder directly:
python run_label_only.py --cleaned-dir /path/to/cleaned_20260101_120000

# Or: automatically pick the newest cleaned_<stamp>/ in a parent folder:
python run_label_only.py --parent-dir /path/to/output

# Only registration or modified labels (default: both):
python run_label_only.py --cleaned-dir <dir> --mode registration
```

### Tests

```bash
cd openwater
python -m pytest 9_testing/ -q
# or without pytest:
python 9_testing/test_cleaning.py
python 9_testing/test_label_only.py
```

- [9_testing/test_cleaning.py](9_testing/test_cleaning.py) — schema mapping, all files cleaned,
  headers normalised, label columns added, `--onlyfirstchunk`.
- [9_testing/test_label_only.py](9_testing/test_label_only.py) — standalone labelling on a cleaned
  folder, auto-pick of the newest folder, and `mode=registration`.

---

## 2. Sorting on date
*(original: step 6)*

Status: **ported** → [2_sorting/](2_sorting/)

Because many features are time-based, the cleaned files are sorted on time variables. Per table it
sorts on the relevant date column (see `TABLE_SORT_COL` in `bucket_sort.py`); for the two tables
where the time is in JSON it uses `Extraction_Date` resp. `Player_Profile_Modified`. The result is a
new folder `<cleaned_dir>_sorted_<stamp>/` next to the input.

Sorting is **memory-efficient** via a two-pass bucket sort (`bucket_sort.py`): pass 1 splits rows
into month buckets (`2023_07 .. 2025_07`), pass 2 sorts each (small) bucket. Output per table:
`<table>_<n>.csv` per month bucket + `<table>_NaT.csv` for rows without a valid date. An audit checks
that no rows are lost (`rows_in == rows_out`).

### Files

| File | Role |
|---|---|
| [run_sorting.py](2_sorting/run_sorting.py) | Runner: sort a whole cleaned folder. Replaces `run_ALL_sorting.sbatch`. |
| [sort_pipeline.py](2_sorting/sort_pipeline.py) | Orchestration (`sort_directory`, `newest_cleaned_dir`): loops over all tables. |
| [bucket_sort.py](2_sorting/bucket_sort.py) | Copied bucket sort; `main()` body extracted into reusable `sort_table(...)`. |

### Usage

```bash
cd 2_sorting

# Sort a concrete cleaned_<stamp>/ folder (output of step 1):
python run_sorting.py --cleaned-dir /path/to/cleaned_20260101_120000

# Or: automatically pick the newest cleaned_<stamp>/ in a parent folder:
python run_sorting.py --parent-dir /path/to/output

# Only a subset of tables:
python run_sorting.py --cleaned-dir <dir> --tables WOK_Player_Profile,WOK_Bet

# Custom output folder / chunk size:
python run_sorting.py --cleaned-dir <dir> --out-dir /path/out --chunksize 200000
```

### Tests

[9_testing/test_sorting.py](9_testing/test_sorting.py) — row preservation per table (no data
loss), ascending sort order within buckets, table subset, and auto-pick of the newest cleaned folder.

> **Skipped:** *Descriptive statistics on RAW cleaned data* (original step 7) is skipped for now on
> the organisation. As a result the numbering shifts by one from here: original step 8 = organisation
> step 3, and so on.

## 3. Build features per scenario
*(original: step 8 — "Build features for an operator")*

Status: **ported** → [3_features/](3_features/)

Builds features for a chosen **scenario** based on a cleaned (and preferably sorted) folder. Per
feature, `FEATURES_REGISTRY` determines which tables/columns are needed; each feature function
streams through the CSVs itself and produces a table per `Player_Profile_ID`, which are merged on
`Player_Profile_ID`. Optionally with an X period (feature window) and Y period (target window). The
output is one features CSV (one row per player).

On an HPC cluster this ran per operator via `run_parse_only.sbatch` → `parse_runner.py` →
`clean_and_parse.py --mode parse`. On the organisation there is one dataset (= one operator), so the
per-operator SLURM array drops out: you point to one folder and choose a scenario.

> The original step 9 ("Features for a scenario for **all** operators", `submit_parse.sh`) is the
> same process on the organisation — there is simply only one operator/dataset, so `run_features.py`
> fully covers that case.

### Files

| File | Role |
|---|---|
| [run_features.py](3_features/run_features.py) | Runner: build features for a scenario on a (sorted) cleaned folder. Replaces `run_parse_only.sbatch` → `parse_runner.py`. |
| [parse_pipeline.py](3_features/parse_pipeline.py) | `SCENARIOS`, `_safe_merge`, `run_scenario` (**verbatim** from `clean_and_parse.py`) + `build_features` (parse branch of `main()`) + `newest_features_input_dir`. |
| `feature_engineering.py`, `feature_engineering_spanish.py`, `feature_engineering_basis.py`, `mapping_helpers.py`, `path_finding.py`, `reading_difficult_json.py` | Unchanged copied dependencies from the main repo. |

### Usage

```bash
cd 3_features

# Features for a scenario on a concrete (sorted) cleaned folder:
python run_features.py --cleaned-dir /path/to/cleaned_..._sorted_... \
  --scenario Flexible_spanish --x-tijdspad 01062025:30062025 --y-tijdspad 01072025:31072025

# Or: automatically pick the newest suitable folder (sorted > cleaned) in a parent:
python run_features.py --parent-dir /path/to/output --scenario Y-target

# Only a single feature from a scenario:
python run_features.py --cleaned-dir <dir> --scenario Flexible_spanish --feature f0_net_winloss
```

> The available scenarios are in `SCENARIOS` in [parse_pipeline.py](3_features/parse_pipeline.py).
> Which features get computed depends on which tables/columns are in your data.

### Tests

[9_testing/test_features.py](9_testing/test_features.py) — building features on working scenarios
(`Y-target`, `Scenario_1`), single-feature override, dedup on `Player_Profile_ID`, and the input
folder picker. Equivalence with the original (`--mode parse`) is proven in
[9_testing/test_equivalence.py](9_testing/test_equivalence.py).

## 3b. Extra features for the total algorithm
*(original: step 10 — "build operator features")*

Status: **ported** → [3b_operator_specific_features/](3b_operator_specific_features/)

When you don't train a separate model per operator but one **total algorithm** across all operators,
it helps to add a few summarising ("operator-specific") features per operator: `mean`/`std`/`min`/
`max` of a fixed set of core features (`ALL_FEATURES` = `f0_net_winloss`, `f3_total_wagered`,
`f25_voluntary_suspensions`, `f12_deposits_per_day`, `f11_withdrawals_per_day`). The result is one
**1-row aggregate CSV** (5 features × 4 stats = 20 columns) that supports the tree model.

On an HPC cluster this ran per operator (a..z) and per period over `data_dir/<op>/`
(`run_ALL_build_operator_features.sbatch`). On the organisation there is one dataset, so you simply
aggregate over one merged features CSV (the output of step 3).

### Files

| File | Role |
|---|---|
| [run_operator_features.py](3b_operator_specific_features/run_operator_features.py) | Runner: aggregate over one features CSV. Replaces `run_ALL_build_operator_features.sbatch`. |
| [operator_features_pipeline.py](3b_operator_specific_features/operator_features_pipeline.py) | `build_all_stats_for_file(...)`: reads one CSV, writes the aggregate row. |
| [build_all_stats.py](3b_operator_specific_features/build_all_stats.py) | Copied; stats computation extracted into reusable `compute_all_stats_row(...)`. |

### Usage

```bash
cd 3b_operator_specific_features

# Aggregate over one features CSV (output of step 3):
python run_operator_features.py --features-csv /path/to/features_..._all.csv

# Or: automatically pick the newest features_*.csv in a folder:
python run_operator_features.py --parent-dir /path/to/output
```

> Note: the core features (f0/f3/f25/f12/f11) come from a large scenario such as
> `Flexible_spanish_plus`. So make sure that in step 3 you run a scenario that produces those columns
> before doing this aggregation.

### Tests

[9_testing/test_operator_features.py](9_testing/test_operator_features.py) — aggregation values,
1-row output with `ALL_SUFFIX` name, partially-present features, clean `ValueError` when there are no
features. Equivalence with the original in
[9_testing/test_equivalence.py](9_testing/test_equivalence.py).

## 4. Filtering on 'Active'
*(original: step 11 — "build status lookup")*

Status: **ported** → [4_active_filter/](4_active_filter/)

Builds per **cutoff date** a `LAST_STATUS_LOOKUP_BEFORE_{DDMMYYYY}.csv` with the last known
`Player_Profile_Status` before that date per player (columns: `Player_Profile_ID`, `last_status`).
With that you can later filter on players who were `ACTIVE` at a reference moment.

The input is a **sorted** folder (output of step 2), because the status history is in the
`WOK_Player_Profile_*.csv` chunk files. On an HPC cluster this ran per operator over
`cleaned_root/<op>/cleaned_*_sorted_*` → `features_root/<op>/`
(`run_build_status_lookup.sbatch`); on the organisation you point to one sorted folder.

### Files

| File | Role |
|---|---|
| [run_active_filter.py](4_active_filter/run_active_filter.py) | Runner: build last-status lookups per cutoff. Replaces `run_build_status_lookup.sbatch`. |
| [active_filter_pipeline.py](4_active_filter/active_filter_pipeline.py) | `build_status_lookups(...)`, `newest_sorted_dir(...)`: orchestration over one sorted folder. |
| [build_status_lookup.py](4_active_filter/build_status_lookup.py) | Copied; core computation extracted into reusable `last_status_before_cutoff(...)`. |

### Usage

```bash
cd 4_active_filter

# On a concrete sorted folder, one or more cutoffs (DDMMYYYY, '|' or ',' separated):
python run_active_filter.py --sorted-dir /path/to/cleaned_..._sorted_... --cutoffs 01072026|01082026

# Or: automatically pick the newest cleaned_*_sorted_* in a parent folder:
python run_active_filter.py --parent-dir /path/to/output --cutoffs 01072026

# Custom output folder / overwrite:
python run_active_filter.py --sorted-dir <dir> --cutoffs 01072026 --out-dir /path/out --overwrite
```

### Tests

[9_testing/test_active_filter.py](9_testing/test_active_filter.py) — schema + last_status per
player (clean→sort→lookup on the dummy data), empty lookup at an early cutoff, multiple cutoffs,
skip/overwrite, and the input folder picker. Equivalence with the original in
[9_testing/test_equivalence.py](9_testing/test_equivalence.py).

## 5. Running descriptive tasks
*(original: step 12 — "descriptive stats")*

Status: **ported** → [5_descriptive/](5_descriptive/)

Turns a feature CSV (output of step 3) into a **descriptive report**: a text file with target
distribution, `describe()` per numeric column, value counts per object column and a correlation
matrix, plus **plots** (target distribution, correlation heatmap, histograms). Everything lands in a
dated subfolder `{output-dir}/{stamp}/`. Heatmap/histograms are made over the feature columns
starting with `x` (convention from the original).

On an HPC cluster this ran per operator (a..z) over `data_dir/<letter>.csv`
(`run_ALL_description.sbatch`); on the organisation you point to one features CSV.

### Files

| File | Role |
|---|---|
| [run_descriptive.py](5_descriptive/run_descriptive.py) | Runner: report + plots for a features CSV. Replaces `run_ALL_description.sbatch`. |
| [descriptive_pipeline.py](5_descriptive/descriptive_pipeline.py) | `build_descriptive_report(...)`, `newest_features_csv(...)`. |
| [descriptive_stats.py](5_descriptive/descriptive_stats.py) | Copied; `main()` body extracted into reusable `generate_descriptive_report(...)`. |

### Usage

```bash
cd 5_descriptive

# On a concrete features CSV with target:
python run_descriptive.py --features-csv /path/to/features_..._all.csv \
  --target y_self_exclusion_20250701_20250731

# Or: automatically pick the newest features_*.csv in a folder:
python run_descriptive.py --parent-dir /path/to/output --target <column>
```

> Requires `matplotlib` (for the plots). The heatmap/histograms only appear if there are columns
> with prefix `x` in the features CSV.

### Tests

[9_testing/test_descriptive.py](9_testing/test_descriptive.py) — report + plots for x-features,
report without target, running on real pipeline output, and the features picker. Equivalence
(byte-identical text report) with the original in
[9_testing/test_equivalence.py](9_testing/test_equivalence.py).

## 6. Merge & sample all operators
*(original: step 13 — "prepare hpo dataset")*

Status: **ported** → [6_merge_sample/](6_merge_sample/) — **faithful** variant

In the original this happens automatically inside model training (`submit_hpsearch.sh --all` →
`run_prepare_hpo_dataset.sbatch`). This organisation step makes it **runnable standalone**.

It merges the per-operator merged feature CSVs into one combined dataset: per period prefix the
base-scenario features + target, enriched with the 1-row **ALL stats from step 3b** (as constant
columns), the **26-column a..z one-hot**, and — if present — the **ACTIVE_FLAG from step 4**
(`LAST_STATUS_LOOKUP_BEFORE_*`). Then the target (`y_self_exclusion_*`) is determined, optionally
**negative undersampling (1:ratio)** is applied, and it is written as `valid_sampled.pkl`,
optionally `test_full.pkl`, and `meta.json`.

> **Faithful = expects the HPC-cluster layout.** The data must be in `data_dir/<operator>/` with per
> period prefix a `{prefix}_{base_scenario}*.csv`, optionally `{prefix}_{all_scenario_name}*.csv`
> (step 3b) and `LAST_STATUS_LOOKUP_BEFORE_{ys}.csv` (step 4). On the organisation there is usually
> one operator. `FILTER_ACTIVE=0` in the environment disables the ACTIVE_FLAG filtering.

### Files

| File | Role |
|---|---|
| [run_merge_sample.py](6_merge_sample/run_merge_sample.py) | Runner: builds the config (from flags or `--config` yaml) and runs the prepare step. Replaces `run_prepare_hpo_dataset.sbatch`. |
| [merge_sample_pipeline.py](6_merge_sample/merge_sample_pipeline.py) | `build_config(...)` + `merge_and_sample(...)` (wrapper around `prepare_dataset`). |
| [prepare_hpo_dataset.py](6_merge_sample/prepare_hpo_dataset.py) | Copied; `main()` body extracted into `prepare_dataset(cfg)`. |
| [hpsearch_runner.py](6_merge_sample/hpsearch_runner.py) | Copied (unchanged); provides `build_all_operators_merged_df`, `make_Xy`, `extract_targets`. |

### Usage

```bash
cd 6_merge_sample

# With individual flags (config is assembled):
python run_merge_sample.py \
  --data-dir /path/to/feature_files --operators x \
  --validation-period-prefixes 01062025_30062025_01072025_31072025_valid \
  --sampling-ratio 20 --dataset-path /path/to/hpo_dataset

# Or: directly an effective config yaml (purely faithful):
python run_merge_sample.py --config /path/to/hpsearch_config_effective.yaml
```

### Tests

[9_testing/test_merge_sample.py](9_testing/test_merge_sample.py) — no-sampling vs undersampling
(1:ratio), the 26-column OHE, test_full.pkl with test prefixes, and `build_config`. Equivalence
(meta.json identical + equal X/y) with the original in
[9_testing/test_equivalence.py](9_testing/test_equivalence.py).

## 7 & 8. Train models (grid + optuna)
*(original: step 14 "per operator" + step 15 "for ALL operators")*

Status: **ported** → [7_modelling/](7_modelling/) — **both** search strategies

In the original, `submit_hpsearch.sh` (39KB bash) orchestrated generating an *effective config* and
firing off a SLURM array of training tasks. On the organisation, pure Python replaces that
orchestration — there are two runners:

- **Gridsearch** ([run_grid.py](7_modelling/run_grid.py)) — for each
  (operator × model × run_variant × grid), `hpsearch_runner.py` is called (CV on the validation set),
  in a plain sequential loop instead of the SLURM array. Output per task: `results.csv`, `best.json`,
  `meta.json`.
- **Optuna** ([run_optuna.py](7_modelling/run_optuna.py)) — one process, time-budgeted Optuna search
  over all models; with `--validate-best` the best trials are recalibrated on cv=5 and (if there is a
  `test_full.pkl`) tested on the test period. Output: `optuna_results.csv`, `optuna_cv5_results.csv`,
  `optuna_meta.json`.

Both run on the **prebuilt dataset from step 6** (`dataset_path` → fast path) and on a base config
(e.g. `hpsearch_config.yaml`) to which `all_mode: True` is added.

### Files

| File | Role |
|---|---|
| [run_grid.py](7_modelling/run_grid.py) | Runner: gridsearch (per-task loop). Replaces the SLURM array of `submit_hpsearch.sh`. |
| [run_optuna.py](7_modelling/run_optuna.py) | Runner: Optuna single-process search. Replaces `submit_hpsearch.sh --optuna`. |
| [modelling_pipeline.py](7_modelling/modelling_pipeline.py) | `make_effective_config`, `enumerate_grid_tasks`, `run_grid_search`, `run_optuna_search` (run the workers as subprocess). |
| [hpsearch_runner.py](7_modelling/hpsearch_runner.py), [optuna_runner.py](7_modelling/optuna_runner.py) | Copied; only `main(argv=None)` added so they are also callable as a function. |

### Usage

```bash
cd 7_modelling

# Gridsearch on one model:
python run_grid.py --config /path/hpsearch_config.yaml --dataset-path /path/hpo_dataset \
  --out-dir /path/grid_run --models decision_tree --cv-folds 5 --no-test

# Optuna search (time-budgeted, with validate-best):
python run_optuna.py --config /path/hpsearch_config.yaml --dataset-path /path/hpo_dataset \
  --out-dir /path/optuna_run --time-budget 14400 --validate-best
```

> The workers are run as a **subprocess**: this mirrors the original (separate processes per task)
> and prevents module-name conflicts with the copies in other step folders.

### Tests

[9_testing/test_modelling.py](9_testing/test_modelling.py) — task enumeration (only_models), grid
writes `results.csv`/`best.json`/`meta.json` (2 trials), deterministic CV metrics, and the optuna
runner runs + writes output. Equivalence of the **deterministic CV metrics** (grid) with the
original in [9_testing/test_equivalence.py](9_testing/test_equivalence.py). *(The Optuna output is
time-dependent and therefore not byte-comparable.)*

## 9. Logging
*(original: step 16 "Read logs" — generalised)*

Status: **ported** → [_logging/](_logging/)

Original step 16 (`get_log_info.sh`) read **SLURM log files** to see which array tasks had failed —
that exact tool is SLURM-specific and n/a on the organisation. But the underlying logging itself is
ported: in the main repo, `clean_runner.py`, `label_runner.py` and `parse_runner.py` **all did the
same** logging — running the work as a subprocess and tee'ing the full output live to one timestamped
log file, with exit code / wall time / peak memory afterwards. On an HPC cluster the SLURM `.out` caught
that; locally `print()` is gone afterwards. This step brings that logging together so that an
organisation run too yields one re-readable log file.

The log lines, the tee loop and the footer are taken **unchanged** from those runners; only the phase
(clean/label_only/parse/...) has become a parameter.

### Files

| File | Role |
|---|---|
| [run_with_log.py](_logging/run_with_log.py) | Generic wrapper: run any command and log the output to one log file. Pure-Python equivalent of `clean_runner.py` / `label_runner.py` / `parse_runner.py`. |
| [run_logger.py](_logging/run_logger.py) | Consolidated logging core (verbatim): `log`, `detect_env`, `make_logfile`, `run_and_log`. |

### Usage

```bash
cd _logging

# Wrap any organisation runner; output → logs/<scenario>_<phase>_<stamp>.log
python run_with_log.py --logdir logs --scenario Scenario_1 --phase clean -- \
  python ../1_cleaning/run_cleaning.py --input-dir ../0_dummy_data/voorbeeld_fake --clean-out-dir out

# Wrap the optuna runner:
python run_with_log.py --logdir logs --phase optuna -- \
  python ../7_modelling/run_optuna.py --config cfg.yaml --dataset-path ds --out-dir run
```

Everything after `--` is the command that gets executed; its exit code is passed through. Locally the
log file is called `*_latest.log` (overwriting), on an HPC cluster `*_<timestamp>.log` — exactly as in the
original runners.

> **Note (verbatim behaviour):** the peak-memory line divides `ru_maxrss` by 1024 (correct in KB→MB
> on Linux/HPC-cluster). On macOS `ru_maxrss` is in bytes, so there the value looks too large. This is
> taken unchanged from the original.

### Tests

[9_testing/test_logging.py](9_testing/test_logging.py) — output to log file (incl. merged stderr),
exit-code pass-through, the footer to stdout, the file-name convention, and the wrapper end-to-end.

### Alongside this step: built-in logs of steps 1 & 3

Independently of this, some steps already write their **own** per-file log files, inherited from the
original: step 1 (cleaning) to `<input>/logs/*_clean.log`, and step 3 (features) to `<cleaned>/logs/`
per feature. The remaining steps print to stdout — wrap those with `run_with_log.py` if you want a
log file of them.

## 10 & 11. Reports (coverage + model report)
*(original: step 17 "HP coverage" + step 18 "Make report")*

Status: **ported** → [8_rapportages/](8_rapportages/) — **combined** in one runner

Both reports run on a grid-run folder (output of step 7) and are merged into
[run_report.py](8_rapportages/run_report.py):

1. **Coverage** (`hpsearch_coverage_report.py`) — per (operator, model, run_variant, grid): how many
   trials ran versus the grid (coverage %), whether the time budget was hit, and the best CV/test
   AUPRC. Table on stdout.
2. **Model report** (`create_model_report.py`) — an extensive text report `<label>_report.txt`
   (completeness, best model per operator, tuning analysis, feature importance, etc.).

On an HPC cluster these were two separate sbatches (`hpsearch_coverage_report.sbatch` +
`create_model_report.sbatch`) that read a `run_label` under a fixed outputs root. On the organisation
you simply point to the grid-run folder; the hardcoded HPC-cluster paths have become parameters.

> **Requires test results.** The model report expects a `test_auprc` per task, so run step 7 **with**
> test (`--test`, and make sure step 6 built a `test_full.pkl` via `--test-period-prefixes`). The
> grid output must have the nested layout `operator/model/run_variant/grid/` — which `run_grid.py`
> now writes.

### Files

| File | Role |
|---|---|
| [run_report.py](8_rapportages/run_report.py) | Combined runner: coverage + model report. Replaces both sbatches. |
| [hpsearch_coverage_report.py](8_rapportages/hpsearch_coverage_report.py) | Copied; now accepts a direct run folder (instead of only a run_label under the fixed root). |
| [create_model_report.py](8_rapportages/create_model_report.py) | Copied; `generate_report` got optional `run_folder`/`out_path` parameters. |

### Usage

```bash
cd 8_rapportages

# Coverage + model report for a grid run:
python run_report.py --run-dir /path/to/grid_run

# Only the coverage table:
python run_report.py --run-dir /path/to/grid_run --coverage-only

# Custom report path:
python run_report.py --run-dir /path/to/grid_run --out /path/report.txt
```

### Tests

[9_testing/test_report.py](9_testing/test_report.py) — nested grid layout, coverage table, model
report with head/tail, the combined runner, and `--coverage-only`.

## 12. OLD: Train model on all operators
*(original: step 19)*

Status: **deprecated** — replaced by step 8.

## 13. Deleting data
*(original: step 20)*

Status: **n/a on the organisation** — local management of the cleaned folders; no HPC-cluster scratch to
clean up.

<br>

---
---

# 🇳🇱 Nederlands

# organisatie-versie — Pure Python pijplijn

Deze map (`openwater`) bevat de **organisatie-variant** van de pijplijn voor preventie van gokverslaving.
Waar het origineel op een HPC-cluster draaide met SLURM-`sbatch` jobs, een SLURM-array per operator
(a..z) en `$TMPDIR`-staging met symlinks, werkt de organisatie
gewoon met **lokale CSV's** en **pure Python**. Je trapt elke stap af met een `.py`-runner —
geen `sbatch`, geen `.sh`, geen omgevingsvariabelen nodig.

Daardoor verschilt de structuur en de nummering:

- De stappen beginnen hier bij **1** (i.p.v. 4 in het origineel).
- De originele stappen **4. CLEANING** en **5. Outlier detection via labelling** zijn
  samengevoegd tot **stap 1 (Cleaning & outlier labelling)**, omdat de schoonmaak de
  labelling al automatisch meeneemt en de losse labelling-stap dezelfde code hergebruikt.
- Alle volgende stappen schuiven mee op (origineel 6 → organisatie 2, 7 → 3, enzovoort).

> **Werkwijze bij overzetten.** Code wordt nooit buiten `` aangepast. Per stap:
> (a) uitvinden welke code nodig is (dependencies), (b) die **ongewijzigd** kopiëren naar
> `<stap>/`, (c) daar aanpassen naar een pure-Python runner, (d) tests schrijven
> in `9_testing/` tegen de dummy-data in `0_dummy_data/voorbeeld_fake/`.

## Mapstructuur

```
openwater/
├── 1_cleaning/                     # stap 1: cleaning & outlier labelling (pure Python)
├── 2_sorting/                      # stap 2: sorting op date (pure Python)
├── 3_features/                     # stap 3: features bouwen per scenario (pure Python)
├── 3b_operator_specific_features/  # stap 3b: ALL-aggregaat features voor totaalalgoritme
├── 4_active_filter/                # stap 4: filteren op 'Active' (last-status lookup per cutoff)
├── 5_descriptive/                  # stap 5: descriptive tasks (rapport + plots)
├── 6_merge_sample/                 # stap 6: samenvoegen & samplen (HPO-dataset bouwen)
├── 7_modelling/                    # stappen 7 & 8: modellen trainen (grid + optuna)
├── 8_rapportages/                  # stappen 10 & 11: dekkingsgraad + modelrapport
├── _logging/                       # stap 9: logging-hulp (run-output → één logbestand)
├── 9_testing/                      # tests tegen 0_dummy_data/voorbeeld_fake/
├── 0_dummy_data/                   # test-data, generators + data-aanlever-spec
│   ├── voorbeeld_fake/             # meegeleverde relationele testset (100 spelers, multi-period, signaal)
│   ├── voorbeeld_noise/            # idem, maar label-onafhankelijke features (lekkage-canary)
│   ├── voorbeeld_11/               # minimaal gemaskeerd structuur-sample (echte export-vorm)
│   ├── _maak_fake.py               # generator voor de relationele test-data (→ voorbeeld_fake/)
│   ├── _maak_fake_big_data.py      # streamende generator voor grote (GB-schaal) datasets
│   ├── _maak_dummy_data.py         # alternatieve relationele dummy-generator
│   ├── _inventariseer_databehoefte.py  # leidt de data-aanlever-spec af uit FEATURES_REGISTRY
│   └── databehoefte.txt            # gegenereerde data-aanlever-spec (welke WOK-bestanden + kolommen)
├── run_pipeline.py                 # ⭐ één entrypoint: hele rit 1 t/m 9 in één keer
├── pyproject.toml                  # Poetry-project + dependencies
├── README.md                       # dit bestand (technische doc)
└── README_organisatie.md           # kook-het-zelf-handleiding (EN + NL)
```

## Vereisten

Python 3.11+ met `pandas`, `numpy`, `python-dateutil`, `scikit-learn`, `xgboost`, `lightgbm`,
`optuna`, `scipy`, `pyyaml`, `matplotlib` (de modelling-/rapportage-stappen). Optioneel
`imbalanced-learn` voor SMOTE-varianten.

## ⭐ Alles in één keer: `run_pipeline.py`

Wil je niet stap-voor-stap, maar de **hele pijplijn (1 t/m 9)** in één aanroep op je eigen
bestanden? Gebruik de orchestrator. Hij verwacht een map met `Operator_*`-submappen (zoals uit
de organisatie-systemen / `0_dummy_data/_maak_fake`) en draait: cleaning(+labelling) → sorting → features →
3b → active-filter → merge&sample → modelling (Optuna of grid) → rapportage.

De CLI-vlaggen volgen waar mogelijk de originele `submit_hpsearch.sh`. **Productie (origineel-getrouw,
echte holdout):** geef volledig gespecificeerde period-prefixes `xstart_xend_ystart_yend_label`
(DDMMYYYY); `validation` en `test` zijn onafhankelijke lijsten, per index gepaard (p0/p1/p2):

```bash
cd openwater
python run_pipeline.py --data-dir /pad/met/Operator_mappen --out-dir /pad/output \
    --base-scenario Flexible_spanish_plus \
    --validation-period-prefixes "01052025_31052025_01062025_30062025_valid,01122024_31052025_01062025_30062025_valid,01062024_31052025_01062025_30062025_valid" \
    --test-period-prefixes       "01052025_30062025_01072025_31072025_test,01122024_30062025_01072025_31072025_test,01062024_30062025_01072025_31072025_test" \
    --optuna --optuna-time 14400 --validate_best
```

**Operator-niveau CV** (leave-operators-out, zoals `--operator-folds`): dan is `--test-period-prefixes`
níet nodig (folding vervangt de holdout); `--independent-searches` doet N losse searches i.p.v. één
averaged CV:

```bash
python run_pipeline.py --data-dir ... --out-dir ... --base-scenario Flexible_spanish_plus \
    --validation-period-prefixes "..._valid,..._valid,..._valid" \
    --operator-folds 5 --independent-searches \
    --optuna --optuna-time 14400 --validate_best --no-multivariate --random-state 23
```

Mist je dataset `Player_Profile_EOD_Balance` (zoals de minimale organisatie-set), voeg dan
`--ignore-eod-balance` toe (f26/f27/f28 vallen dan terug i.p.v. te crashen).

Of vanuit Python (alle parameters hebben **expliciete defaults** + uitleg in de docstring):

```python
from run_pipeline import run_full_pipeline
artefacten = run_full_pipeline(
    data_dir="/pad/met/Operator_mappen", out_dir="/pad/output",
    scenario="Flexible_spanish_plus",
    # PRODUCTIE: onafhankelijke valid/test-prefixes (echte temporele holdout):
    validation_period_prefixes=["01052025_31052025_01062025_30062025_valid", "..."],
    test_period_prefixes=["01052025_30062025_01072025_31072025_test", "..."],
    # GEMAK (smoke-test): één venster + evt. apart test-venster
    #   x_tijdspad="01062025:30062025", y_tijdspad="01072025:31072025",
    #   x_test_tijdspad="01072025:31072025", y_test_tijdspad="01082025:31082025",
    search="optuna", sampling_ratio=20, optuna_time_budget=14400, optuna_validate_best=True,
    # ignore_EOD_Balance=True,   # als je data geen Player_Profile_EOD_Balance heeft
    # operator_folds=5, independent_searches=True,   # operator-niveau CV (test dan niet nodig)
)
# → {cleaned, sorted, feature_files_dir, dataset_path, model_out_dir, report}
```

> **Echte holdout (belangrijk).** `validation` en `test` zijn **onafhankelijke** vensters: de
> test-periode hoort een ander, later venster te zijn (bv. test-y één maand ná valid-y) → een
> echte temporele holdout. De gemak-modus zonder apart test-venster spiegelt test=valid (**geen
> holdout**, met waarschuwing) en is alleen voor smoke-tests. Met `--operator-folds` is test niet
> nodig (folding levert de holdout). Bij N validation-prefixes wordt elke feature N× berekend
> (`p0_*`/`p1_*`/…) → de productie-config gebruikt 3 lookback-horizonten (1m/6m/12m).

> **Vlag-uitlijning & nuances.** De vlagnamen komen overeen met `submit_hpsearch.sh`. Drie
> verschillen volgen uit het ontbreken van SLURM/YAML: `--validate_best` is **opt-in** (default uit),
> `--all` is een no-op (deze pijplijn draait altijd all-operators), en `--config` is optioneel voor
> optuna (test-prefixes geef je via CLI i.p.v. uit de YAML).

[run_pipeline.py](run_pipeline.py) toont per stap een banner met wat er gebeurt, en heeft álle
belangrijke parameters expliciet (periodes, scenario, sampling, optuna-tijdsbudget, validate-best,
exclude-models, grid vs optuna, enz.). De losse runners per stap (hieronder) blijven beschikbaar
voor wie één stap apart wil draaien.

> **NB (geen echte Python-package):** de stapmappen heten `1_cleaning`, `2_sorting`, … — geen
> geldige modulenamen, dus niet als package te importeren. `run_pipeline.py` lost dit op door de
> stapmappen op `sys.path` te zetten; het fungeert zo als de package-entrypoint zonder de mappen
> te hernoemen.

### Test-data genereren

De meegeleverde `0_dummy_data/voorbeeld_fake/` komt uit `0_dummy_data/_maak_fake.py` (relationeel formaat, 100 spelers,
multi-period standaard). Opnieuw genereren of aanpassen:

```bash
python 0_dummy_data/_maak_fake.py                                    # → voorbeeld_fake/ (100 spelers, multi-period)
python 0_dummy_data/_maak_fake.py --out mijn_set --players 250 --seed 7
python 0_dummy_data/_maak_fake.py --no-signal --out 0_dummy_data/voorbeeld_noise  # lekkage-canary (label-onafhankelijke features)
```

- **multi-period (standaard)** — activiteit jan–apr 2026 en uitsluiters verdeeld over **mei én juni**
  (twee target-maanden), zodat een **echte temporele holdout** kan: validatie y=mei, test y=juni. Een
  holdout/lekkage-check op één target-maand (test=valid) is zinloos, dus multi-period is de default.
  `--single-period` zet alle uitsluiters in één maand (geen holdout).
- **`--no-signal`** — features worden **onafhankelijk** van het label gegenereerd (noise). Op zulke
  data hoort een eerlijk model op de **holdout-test ≈0.5 AUC** te halen → **lekkage-canary**: scoort
  het tóch hoog, dan lekt het target. (Default = uitsluiters gokken zwaarder → leerbaar.)
- Voor heel grote sets: `0_dummy_data/_maak_fake_big_data.py` (streamend): `--target-gb` of `--players`.

> **Belangrijk (lekkage-check):** beoordeel een lekkage-canary **altijd** op een **echt holdout**
> (multi-period: validatie- én test-target in verschillende maanden). De one-shot zónder apart
> test-venster spiegelt `test = valid` → dan toont zelfs pure noise een hoge "test"-score
> (memorisatie, geen lek). Gebruik de CV-metriek of een echt holdout-venster.

## Losse stappen draaien

```bash
cd 1_cleaning
python run_cleaning.py            # zie stap 1 hieronder
```

---

# Stappen

## 1. Cleaning & outlier labelling
*(origineel: stap 4 CLEANING + stap 5 Outlier detection via labelling)*

Status: **overgezet** → [1_cleaning/](1_cleaning/)

Het schoonmaken leest elke ruwe WOK-CSV streamend in chunks, normaliseert kolomnamen en
waarden volgens het schema (`check_format.py`), past optionele operator-specifieke
prevalidatie-filters toe, en schrijft de schone CSV's naar een nieuwe `cleaned_<timestamp>/`
map. Daarna bewaart transactie-deduplicatie de nieuwste rij per speler/transactie in
`WOK_Player_Account_Transaction`, `WOK_Bet_Transaction` en `WOK_Game_Session_Transaction`, ook over
chunks en CSV-deelbestanden heen. Operators blijven gescheiden via `operator_id`, of via de
gekoppelde parenttabel bij bet-/sessietransacties; zonder die informatie bepaalt de inputmap de
operator-scope. Accounttransacties worden geordend op `extraction_date`, `_write_timestamp`, dan
`created_at`; bet-/sessietransacties op `created_at`, `_write_timestamp`, dan `extraction_date`.
Ontbreekt een tijdstempel, dan gebruiken we het volgende beschikbare veld. Bij gelijke of geheel
ontbrekende tijdstempels blijft de laatst gelezen rij staan (natuurlijke bestandsnaamvolgorde,
daarna rijvolgorde). ID's blijven tekst; rijen zonder speler- of transactie-ID blijven afzonderlijk
behouden. De SQLite-index gebruikt schijfopslag; de aantallen staan in `logs/transaction_dedup.log`
bij de ruwe input. Vervolgens draait automatisch de **outlier-labelling**, die op het
`WOK_Player_Profile`-bestand de kolommen `outlier_Registration_Date` en
`outlier_Player_Profile_Modified` toevoegt.

De labelling kan ook **apart** worden gedraaid (om geheugen/tijd te besparen, of om te
herhalen op een al-geschoonde map) — dat is het pure-Python equivalent van het originele
`--mode label_only`.

### Bestanden

| Bestand | Rol |
|---|---|
| [run_cleaning.py](1_cleaning/run_cleaning.py) | Runner: schoon een map + automatische labelling. Vervangt `run_ALL_clean_only.sbatch` → `clean_runner.py`. |
| [run_label_only.py](1_cleaning/run_label_only.py) | Runner: **alleen** labelling op een bestaande `cleaned_<stamp>/` map. Vervangt `run_ALL_label_only.sbatch` → `label_runner.py`. |
| [clean_pipeline.py](1_cleaning/clean_pipeline.py) | Orkestratie (`clean_directory`, `label_only_directory`, `newest_cleaned_dir`). Uitgeklede `clean_and_parse.py` — alleen clean+label, geen feature-parse. |
| [transaction_dedup.py](1_cleaning/transaction_dedup.py) | Nieuwste transactie per operator/speler/ID bewaren over CSV-delen, vóór labelling. |
| `cleaner.py`, `check_format.py`, `operator_filters.py`, `outlier_labeling.py`, `reading_difficult_json.py`, `path_finding.py` | Cleaning, validatie, filters, labelling en readers gebaseerd op de hoofdrepo. |

### Gebruik — cleaning

```bash
cd 1_cleaning

# Schoon de standaard dummy-data (../0_dummy_data/voorbeeld_fake) → cleaned_<stamp>/ ernaast:
python run_cleaning.py

# Eigen input/output map:
python run_cleaning.py --input-dir /pad/naar/raw_csvs --clean-out-dir /pad/naar/output

# Snelle test op alleen de eerste chunk (kleine TEST_cleaned_<stamp>/ map):
python run_cleaning.py --onlyfirstchunk --chunksize 2000

# Zonder de automatische outlier-labelling:
python run_cleaning.py --no-label

# Met operator-specifieke prevalidatie-filter (origineel: OPERATOR_PREFIX):
python run_cleaning.py --operator-prefix b
```

### Gebruik — alleen labelling

```bash
cd 1_cleaning

# Direct een concrete cleaned_<stamp>/ map labellen:
python run_label_only.py --cleaned-dir /pad/naar/cleaned_20260101_120000

# Of: kies automatisch de nieuwste cleaned_<stamp>/ in een parent-map:
python run_label_only.py --parent-dir /pad/naar/output

# Alleen registratie- of modified-labels (default: both):
python run_label_only.py --cleaned-dir <dir> --mode registration
```

### Tests

```bash
cd openwater
python -m pytest 9_testing/ -q
# of zonder pytest:
python 9_testing/test_cleaning.py
python 9_testing/test_label_only.py
```

- [9_testing/test_cleaning.py](9_testing/test_cleaning.py) — schema-mapping, alle
  bestanden geschoond, headers genormaliseerd, labelkolommen toegevoegd, `--onlyfirstchunk`.
- [9_testing/test_label_only.py](9_testing/test_label_only.py) — losse labelling op een
  cleaned-map, auto-pick van de nieuwste map, en `mode=registration`.

---

## 2. Sorting op date
*(origineel: stap 6)*

Status: **overgezet** → [2_sorting/](2_sorting/)

Omdat veel features op tijd zijn gebaseerd, worden de geschoonde bestanden gesorteerd op
tijdsvariabelen. Per tabel wordt op de bijbehorende datumkolom gesorteerd (zie
`TABLE_SORT_COL` in `bucket_sort.py`); voor de twee tabellen waar de tijd in JSON zit
wordt de `Extraction_Date` resp. `Player_Profile_Modified` gebruikt. Het resultaat is een
nieuwe map `<cleaned_dir>_sorted_<stamp>/` naast de input.

Er wordt **memory-efficiënt** gesorteerd via een twee-passes bucket-sort (`bucket_sort.py`):
pass 1 verdeelt rijen in maand-buckets (`2023_07 .. 2025_07`), pass 2 sorteert elke (kleine)
bucket. Output per tabel: `<tabel>_<n>.csv` per maand-bucket + `<tabel>_NaT.csv` voor rijen
zonder geldige datum. Een audit controleert dat geen rijen verloren gaan
(`rows_in == rows_out`).

### Bestanden

| Bestand | Rol |
|---|---|
| [run_sorting.py](2_sorting/run_sorting.py) | Runner: sorteer een hele geschoonde map. Vervangt `run_ALL_sorting.sbatch`. |
| [sort_pipeline.py](2_sorting/sort_pipeline.py) | Orkestratie (`sort_directory`, `newest_cleaned_dir`): loopt over alle tabellen. |
| [bucket_sort.py](2_sorting/bucket_sort.py) | Gekopieerde bucket-sort; `main()`-body uitgelicht naar herbruikbare `sort_table(...)`. |

### Gebruik

```bash
cd 2_sorting

# Sorteer een concrete cleaned_<stamp>/ map (output van stap 1):
python run_sorting.py --cleaned-dir /pad/naar/cleaned_20260101_120000

# Of: kies automatisch de nieuwste cleaned_<stamp>/ in een parent-map:
python run_sorting.py --parent-dir /pad/naar/output

# Slechts een subset tabellen:
python run_sorting.py --cleaned-dir <dir> --tables WOK_Player_Profile,WOK_Bet

# Eigen output-map / chunkgrootte:
python run_sorting.py --cleaned-dir <dir> --out-dir /pad/out --chunksize 200000
```

### Tests

[9_testing/test_sorting.py](9_testing/test_sorting.py) — rij-behoud per tabel (geen
dataverlies), oplopende sorteervolgorde binnen buckets, tabel-subset, en auto-pick van de
nieuwste cleaned-map.

> **Overgeslagen:** *Descriptive statistics op RAW cleaned data* (origineel stap 7) wordt
> voorlopig overgeslagen op de organisatie. Daardoor schuift de nummering vanaf hier één op:
> origineel stap 8 = organisatie-stap 3, enzovoort.

## 3. Features bouwen per scenario
*(origineel: stap 8 — "Features voor een operator maken")*

Status: **overgezet** → [3_features/](3_features/)

Bouwt features voor een gekozen **scenario** op basis van een geschoonde (en bij voorkeur
gesorteerde) map. Per feature bepaalt `FEATURES_REGISTRY` welke tabellen/kolommen nodig zijn;
elke feature-functie streamt zelf door de CSV's en levert een tabel per `Player_Profile_ID`,
die op `Player_Profile_ID` worden gemerged. Optioneel met een X-tijdspad (features-periode)
en Y-tijdspad (target-periode). Output is één features-CSV (één rij per speler).

Op een HPC-cluster liep dit per operator via `run_parse_only.sbatch` → `parse_runner.py` →
`clean_and_parse.py --mode parse`. Op de organisatie is er één dataset (= één operator), dus de
per-operator SLURM-array vervalt: je wijst één map aan en kiest een scenario.

> De originele stap 9 ("Features voor een scenario voor **alle** operators", `submit_parse.sh`)
> is op de organisatie hetzelfde proces — er is simpelweg maar één operator/dataset, dus `run_features.py`
> dekt dat geval volledig af.

### Bestanden

| Bestand | Rol |
|---|---|
| [run_features.py](3_features/run_features.py) | Runner: bouw features voor een scenario op een (gesorteerde) cleaned-map. Vervangt `run_parse_only.sbatch` → `parse_runner.py`. |
| [parse_pipeline.py](3_features/parse_pipeline.py) | `SCENARIOS`, `_safe_merge`, `run_scenario` (**verbatim** uit `clean_and_parse.py`) + `build_features` (parse-branch van `main()`) + `newest_features_input_dir`. |
| `feature_engineering.py`, `feature_engineering_spanish.py`, `feature_engineering_basis.py`, `mapping_helpers.py`, `path_finding.py`, `reading_difficult_json.py` | Ongewijzigd gekopieerde dependencies uit de hoofdrepo. |

### Gebruik

```bash
cd 3_features

# Features voor een scenario op een concrete (gesorteerde) cleaned-map:
python run_features.py --cleaned-dir /pad/naar/cleaned_..._sorted_... \
  --scenario Flexible_spanish --x-tijdspad 01062025:30062025 --y-tijdspad 01072025:31072025

# Of: kies automatisch de nieuwste geschikte map (sorted > cleaned) in een parent:
python run_features.py --parent-dir /pad/naar/output --scenario Y-target

# Slechts één enkele feature uit een scenario:
python run_features.py --cleaned-dir <dir> --scenario Flexible_spanish --feature f0_net_winloss
```

> De beschikbare scenario's staan in `SCENARIOS` in [parse_pipeline.py](3_features/parse_pipeline.py).
> Welke features doorrekenen hangt af van welke tabellen/kolommen in je data zitten.

### Tests

[9_testing/test_features.py](9_testing/test_features.py) — features bouwen op werkende
scenario's (`Y-target`, `Scenario_1`), single-feature override, dedup op `Player_Profile_ID`,
en de input-map picker. De equivalentie met het origineel (`--mode parse`) wordt bewezen in
[9_testing/test_equivalence.py](9_testing/test_equivalence.py).

## 3b. Extra features voor het totaalalgoritme
*(origineel: stap 10 — "build operator features")*

Status: **overgezet** → [3b_operator_specific_features/](3b_operator_specific_features/)

Wanneer je niet per operator een apart model traint maar één **totaalalgoritme** over alle
operators, helpt het om per operator een paar samenvattende ("operator-specifieke") features
toe te voegen: `mean`/`std`/`min`/`max` van een vaste set kernfeatures (`ALL_FEATURES` =
`f0_net_winloss`, `f3_total_wagered`, `f25_voluntary_suspensions`, `f12_deposits_per_day`,
`f11_withdrawals_per_day`). Het resultaat is één **1-rij aggregaat-CSV** (5 features × 4 stats
= 20 kolommen) die het tree-model ondersteunt.

Op een HPC-cluster liep dit per operator (a..z) en per periode over `data_dir/<op>/`
(`run_ALL_build_operator_features.sbatch`). Op de organisatie is er één dataset, dus je aggregeert
simpelweg over één merged features-CSV (de output van stap 3).

### Bestanden

| Bestand | Rol |
|---|---|
| [run_operator_features.py](3b_operator_specific_features/run_operator_features.py) | Runner: aggregeer over één features-CSV. Vervangt `run_ALL_build_operator_features.sbatch`. |
| [operator_features_pipeline.py](3b_operator_specific_features/operator_features_pipeline.py) | `build_all_stats_for_file(...)`: leest één CSV, schrijft de aggregaat-rij. |
| [build_all_stats.py](3b_operator_specific_features/build_all_stats.py) | Gekopieerd; stats-berekening uitgelicht naar herbruikbare `compute_all_stats_row(...)`. |

### Gebruik

```bash
cd 3b_operator_specific_features

# Aggregeer over één features-CSV (output van stap 3):
python run_operator_features.py --features-csv /pad/naar/features_..._all.csv

# Of: kies automatisch de nieuwste features_*.csv in een map:
python run_operator_features.py --parent-dir /pad/naar/output
```

> Let op: de kernfeatures (f0/f3/f25/f12/f11) ontstaan uit een groot scenario zoals
> `Flexible_spanish_plus`. Zorg dus dat je in stap 3 een scenario draait dat die kolommen
> oplevert vóór je deze aggregatie doet.

### Tests

[9_testing/test_operator_features.py](9_testing/test_operator_features.py) — aggregatie-
waarden, 1-rij output met `ALL_SUFFIX`-naam, deel-aanwezige features, nette `ValueError` bij
geen features. Equivalentie met het origineel in
[9_testing/test_equivalence.py](9_testing/test_equivalence.py).

## 4. Filteren op 'Active'
*(origineel: stap 11 — "build status lookup")*

Status: **overgezet** → [4_active_filter/](4_active_filter/)

Maakt per **cutoff-datum** een `LAST_STATUS_LOOKUP_BEFORE_{DDMMYYYY}.csv` met de laatst
bekende `Player_Profile_Status` vóór die datum per speler (kolommen: `Player_Profile_ID`,
`last_status`). Daarmee kun je later filteren op spelers die op een peilmoment `ACTIVE` waren.

Input is een **gesorteerde** map (output van stap 2), want de status-historie zit in de
`WOK_Player_Profile_*.csv` chunk-bestanden. Op een HPC-cluster liep dit per operator over
`cleaned_root/<op>/cleaned_*_sorted_*` → `features_root/<op>/`
(`run_build_status_lookup.sbatch`); op de organisatie wijs je één gesorteerde map aan.

### Bestanden

| Bestand | Rol |
|---|---|
| [run_active_filter.py](4_active_filter/run_active_filter.py) | Runner: bouw last-status lookups per cutoff. Vervangt `run_build_status_lookup.sbatch`. |
| [active_filter_pipeline.py](4_active_filter/active_filter_pipeline.py) | `build_status_lookups(...)`, `newest_sorted_dir(...)`: orkestratie over één gesorteerde map. |
| [build_status_lookup.py](4_active_filter/build_status_lookup.py) | Gekopieerd; kernberekening uitgelicht naar herbruikbare `last_status_before_cutoff(...)`. |

### Gebruik

```bash
cd 4_active_filter

# Op een concrete gesorteerde map, één of meer cutoffs (DDMMYYYY, '|' of ',' gescheiden):
python run_active_filter.py --sorted-dir /pad/naar/cleaned_..._sorted_... --cutoffs 01072026|01082026

# Of: kies automatisch de nieuwste cleaned_*_sorted_* in een parent-map:
python run_active_filter.py --parent-dir /pad/naar/output --cutoffs 01072026

# Eigen output-map / overschrijven:
python run_active_filter.py --sorted-dir <dir> --cutoffs 01072026 --out-dir /pad/out --overwrite
```

### Tests

[9_testing/test_active_filter.py](9_testing/test_active_filter.py) — schema + last_status
per speler (clean→sort→lookup op de dummy-data), lege lookup bij vroege cutoff, meerdere
cutoffs, skip/overwrite, en de input-map picker. Equivalentie met het origineel in
[9_testing/test_equivalence.py](9_testing/test_equivalence.py).

## 5. Descriptive tasks uitvoeren
*(origineel: stap 12 — "descriptive stats")*

Status: **overgezet** → [5_descriptive/](5_descriptive/)

Maakt van een feature-CSV (output van stap 3) een **beschrijvend rapport**: een tekstbestand
met target-verdeling, `describe()` per numerieke kolom, value-counts per object-kolom en een
correlatiematrix, plus **plots** (target-verdeling, correlatie-heatmap, histogrammen). Alles
landt in een dated submap `{output-dir}/{stamp}/`. Heatmap/histogrammen worden gemaakt over de
feature-kolommen die met `x` beginnen (conventie uit het origineel).

Op een HPC-cluster liep dit per operator (a..z) over `data_dir/<letter>.csv`
(`run_ALL_description.sbatch`); op de organisatie wijs je één features-CSV aan.

### Bestanden

| Bestand | Rol |
|---|---|
| [run_descriptive.py](5_descriptive/run_descriptive.py) | Runner: rapport + plots voor een features-CSV. Vervangt `run_ALL_description.sbatch`. |
| [descriptive_pipeline.py](5_descriptive/descriptive_pipeline.py) | `build_descriptive_report(...)`, `newest_features_csv(...)`. |
| [descriptive_stats.py](5_descriptive/descriptive_stats.py) | Gekopieerd; `main()`-body uitgelicht naar herbruikbare `generate_descriptive_report(...)`. |

### Gebruik

```bash
cd 5_descriptive

# Op een concrete features-CSV met target:
python run_descriptive.py --features-csv /pad/naar/features_..._all.csv \
  --target y_self_exclusion_20250701_20250731

# Of: kies automatisch de nieuwste features_*.csv in een map:
python run_descriptive.py --parent-dir /pad/naar/output --target <kolom>
```

> Vereist `matplotlib` (voor de plots). De heatmap/histogrammen verschijnen alleen als er
> kolommen met prefix `x` in de features-CSV zitten.

### Tests

[9_testing/test_descriptive.py](9_testing/test_descriptive.py) — rapport + plots voor
x-features, rapport zonder target, draaien op echte pipeline-output, en de features-picker.
Equivalentie (byte-identiek tekstrapport) met het origineel in
[9_testing/test_equivalence.py](9_testing/test_equivalence.py).

## 6. Samenvoegen & samplen van alle operators
*(origineel: stap 13 — "prepare hpo dataset")*

Status: **overgezet** → [6_merge_sample/](6_merge_sample/) — **faithful** variant

In het origineel gebeurt dit automatisch binnen de modeltraining (`submit_hpsearch.sh --all`
→ `run_prepare_hpo_dataset.sbatch`). Deze organisatie-stap maakt het **los runbaar**.

Het voegt de per-operator merged feature-CSV's samen tot één gecombineerde dataset:
per period-prefix de base-scenario features + target, verrijkt met de 1-rij **ALL-stats uit
stap 3b** (als constante kolommen), de **26-koloms a..z one-hot**, en — indien aanwezig — het
**ACTIVE_FLAG uit stap 4** (`LAST_STATUS_LOOKUP_BEFORE_*`). Daarna wordt de target
(`y_self_exclusion_*`) bepaald, optioneel **negatieve undersampling (1:ratio)** toegepast, en
weggeschreven als `valid_sampled.pkl`, optioneel `test_full.pkl`, en `meta.json`.

> **Faithful = verwacht de HPC-cluster-layout.** De data moet in `data_dir/<operator>/` staan met
> per period-prefix een `{prefix}_{base_scenario}*.csv`, optioneel `{prefix}_{all_scenario_name}*.csv`
> (stap 3b) en `LAST_STATUS_LOOKUP_BEFORE_{ys}.csv` (stap 4). Op de organisatie is er doorgaans één
> operator. `FILTER_ACTIVE=0` in de omgeving schakelt de ACTIVE_FLAG-filtering uit.

### Bestanden

| Bestand | Rol |
|---|---|
| [run_merge_sample.py](6_merge_sample/run_merge_sample.py) | Runner: bouwt de config (uit vlaggen of `--config` yaml) en draait de prepare-stap. Vervangt `run_prepare_hpo_dataset.sbatch`. |
| [merge_sample_pipeline.py](6_merge_sample/merge_sample_pipeline.py) | `build_config(...)` + `merge_and_sample(...)` (wrapper om `prepare_dataset`). |
| [prepare_hpo_dataset.py](6_merge_sample/prepare_hpo_dataset.py) | Gekopieerd; `main()`-body uitgelicht naar `prepare_dataset(cfg)`. |
| [hpsearch_runner.py](6_merge_sample/hpsearch_runner.py) | Gekopieerd (ongewijzigd); levert `build_all_operators_merged_df`, `make_Xy`, `extract_targets`. |

### Gebruik

```bash
cd 6_merge_sample

# Met losse vlaggen (config wordt samengesteld):
python run_merge_sample.py \
  --data-dir /pad/naar/feature_files --operators x \
  --validation-period-prefixes 01062025_30062025_01072025_31072025_valid \
  --sampling-ratio 20 --dataset-path /pad/naar/hpo_dataset

# Of: rechtstreeks een effectieve config-yaml (puur faithful):
python run_merge_sample.py --config /pad/naar/hpsearch_config_effective.yaml
```

### Tests

[9_testing/test_merge_sample.py](9_testing/test_merge_sample.py) — geen-sampling vs
undersampling (1:ratio), de 26-koloms OHE, test_full.pkl bij test-prefixes, en `build_config`.
Equivalentie (meta.json identiek + gelijke X/y) met het origineel in
[9_testing/test_equivalence.py](9_testing/test_equivalence.py).

## 7 & 8. Modellen trainen (grid + optuna)
*(origineel: stap 14 "per operator" + stap 15 "voor ALLE operators")*

Status: **overgezet** → [7_modelling/](7_modelling/) — **beide** zoekstrategieën

In het origineel orkestreerde `submit_hpsearch.sh` (39KB bash) het genereren van een
*effectieve config* en het afvuren van een SLURM-array van trainingstaken. Op de organisatie vervangt
pure Python die orchestratie — er zijn twee runners:

- **Gridsearch** ([run_grid.py](7_modelling/run_grid.py)) — voor elke
  (operator × model × run_variant × grid) wordt `hpsearch_runner.py` aangeroepen
  (CV op de validatieset), in een gewone sequentiële loop i.p.v. de SLURM-array. Output per
  taak: `results.csv`, `best.json`, `meta.json`.
- **Optuna** ([run_optuna.py](7_modelling/run_optuna.py)) — één proces, tijd-gebudgetteerde
  Optuna-search over alle modellen; met `--validate-best` worden de beste trials op cv=5
  herijkt en (als er een `test_full.pkl` is) op de testperiode getest. Output:
  `optuna_results.csv`, `optuna_cv5_results.csv`, `optuna_meta.json`.

Beide draaien op de **prebuilt dataset uit stap 6** (`dataset_path` → snelle pad) en op een
basis-config (bijv. `hpsearch_config.yaml`) waar `all_mode: True` aan wordt toegevoegd.

### Bestanden

| Bestand | Rol |
|---|---|
| [run_grid.py](7_modelling/run_grid.py) | Runner: gridsearch (per-taak loop). Vervangt de SLURM-array van `submit_hpsearch.sh`. |
| [run_optuna.py](7_modelling/run_optuna.py) | Runner: Optuna single-process search. Vervangt `submit_hpsearch.sh --optuna`. |
| [modelling_pipeline.py](7_modelling/modelling_pipeline.py) | `make_effective_config`, `enumerate_grid_tasks`, `run_grid_search`, `run_optuna_search` (draaien de workers als subprocess). |
| [hpsearch_runner.py](7_modelling/hpsearch_runner.py), [optuna_runner.py](7_modelling/optuna_runner.py) | Gekopieerd; alleen `main(argv=None)` toegevoegd zodat ze ook als functie aanroepbaar zijn. |

### Gebruik

```bash
cd 7_modelling

# Gridsearch op één model:
python run_grid.py --config /pad/hpsearch_config.yaml --dataset-path /pad/hpo_dataset \
  --out-dir /pad/grid_run --models decision_tree --cv-folds 5 --no-test

# Optuna-search (tijd-gebudgetteerd, met validate-best):
python run_optuna.py --config /pad/hpsearch_config.yaml --dataset-path /pad/hpo_dataset \
  --out-dir /pad/optuna_run --time-budget 14400 --validate-best
```

> De workers worden als **subprocess** gedraaid: dit spiegelt het origineel (losse processen
> per taak) en voorkomt modulenaam-conflicten met de kopieën in andere stap-mappen.

### Tests

[9_testing/test_modelling.py](9_testing/test_modelling.py) — taak-enumeratie (only_models),
grid schrijft `results.csv`/`best.json`/`meta.json` (2 trials), deterministische CV-metrieken,
en de optuna-runner draait + schrijft output. Equivalentie van de **deterministische
CV-metrieken** (grid) met het origineel in
[9_testing/test_equivalence.py](9_testing/test_equivalence.py). *(De Optuna-output is
tijd-afhankelijk en daarom niet byte-vergelijkbaar.)*

## 9. Logging
*(origineel: stap 16 "Logs lezen" — gegeneraliseerd)*

Status: **overgezet** → [_logging/](_logging/)

Origineel stap 16 (`get_log_info.sh`) las **SLURM-logbestanden** uit om te zien welke
array-taken gefaald waren — dat exacte hulpmiddel is SLURM-specifiek en n.v.t. op de organisatie. Maar
de onderliggende logging zelf is wél overgezet: in de hoofdrepo deden `clean_runner.py`,
`label_runner.py` en `parse_runner.py` **allemaal dezelfde** logging — het werk als subprocess
draaien en de volledige output live naar één getimestampt logbestand tee'en, met daarna
exitcode / wall time / piekgeheugen. Op een HPC-cluster ving de SLURM `.out` dat op; lokaal is `print()`
na afloop weg. Deze stap brengt die logging samen zodat ook een organisatie-run één terugleesbaar
logbestand oplevert.

De logregels, de tee-loop en de footer zijn **ongewijzigd** overgenomen uit die runners; alleen
de fase (clean/label_only/parse/...) is een parameter geworden.

### Bestanden

| Bestand | Rol |
|---|---|
| [run_with_log.py](_logging/run_with_log.py) | Generieke wrapper: draai elk commando en log de output naar één logbestand. Pure-Python equivalent van `clean_runner.py` / `label_runner.py` / `parse_runner.py`. |
| [run_logger.py](_logging/run_logger.py) | Geconsolideerde logging-kern (verbatim): `log`, `detect_env`, `make_logfile`, `run_and_log`. |

### Gebruik

```bash
cd _logging

# Wrap een willekeurige organisatie-runner; output → logs/<scenario>_<phase>_<stamp>.log
python run_with_log.py --logdir logs --scenario Scenario_1 --phase clean -- \
  python ../1_cleaning/run_cleaning.py --input-dir ../0_dummy_data/voorbeeld_fake --clean-out-dir out

# Wrap de optuna-runner:
python run_with_log.py --logdir logs --phase optuna -- \
  python ../7_modelling/run_optuna.py --config cfg.yaml --dataset-path ds --out-dir run
```

Alles na `--` is het commando dat wordt uitgevoerd; de exitcode ervan wordt doorgegeven.
Lokaal heet het logbestand `*_latest.log` (overschrijvend), op een HPC-cluster `*_<timestamp>.log` —
exact zoals in de originele runners.

> **Let op (verbatim gedrag):** de piekgeheugen-regel deelt `ru_maxrss` door 1024 (correct in
> KB→MB op Linux/HPC-cluster). Op macOS is `ru_maxrss` in bytes, dus daar oogt de waarde te groot.
> Dit is ongewijzigd uit het origineel overgenomen.

### Tests

[9_testing/test_logging.py](9_testing/test_logging.py) — output naar logbestand (incl.
samengevoegde stderr), exitcode-doorgifte, de footer naar stdout, bestandsnaam-conventie, en de
wrapper end-to-end.

### Naast deze stap: ingebouwde logs van stap 1 & 3

Onafhankelijk hiervan schrijven enkele stappen al **eigen** per-bestand logbestanden, geërfd
van het origineel: stap 1 (cleaning) naar `<input>/logs/*_clean.log`, en stap 3 (features) naar
`<cleaned>/logs/` per feature. De overige stappen printen naar stdout — wrap die met
`run_with_log.py` als je er een logbestand van wilt.

## 10 & 11. Rapportages (dekkingsgraad + modelrapport)
*(origineel: stap 17 "HP-dekkingsgraad" + stap 18 "Rapportage maken")*

Status: **overgezet** → [8_rapportages/](8_rapportages/) — **gecombineerd** in één runner

Beide rapportages draaien op een grid-run-map (output van stap 7) en zijn samengevoegd in
[run_report.py](8_rapportages/run_report.py):

1. **Dekkingsgraad** (`hpsearch_coverage_report.py`) — per (operator, model, run_variant, grid):
   hoeveel trials gedraaid t.o.v. het grid (coverage %), of het tijdsbudget geraakt is, en de
   beste CV-/test-AUPRC. Tabel op stdout.
2. **Modelrapport** (`create_model_report.py`) — een uitgebreid tekstrapport `<label>_report.txt`
   (compleetheid, beste model per operator, tuning-analyse, feature-importance, enz.).

Op een HPC-cluster waren dit twee losse sbatches (`hpsearch_coverage_report.sbatch` +
`create_model_report.sbatch`) die een `run_label` onder een vaste outputs-root lazen. Op de organisatie
wijs je gewoon de grid-run-map aan; de hardcoded HPC-cluster-paden zijn parameters geworden.

> **Vereist test-resultaten.** Het modelrapport verwacht een `test_auprc` per taak, dus draai
> stap 7 mét test (`--test`, en zorg dat stap 6 een `test_full.pkl` bouwde via
> `--test-period-prefixes`). De grid-output moet de geneste layout
> `operator/model/run_variant/grid/` hebben — die schrijft `run_grid.py` nu weg.

### Bestanden

| Bestand | Rol |
|---|---|
| [run_report.py](8_rapportages/run_report.py) | Gecombineerde runner: dekkingsgraad + modelrapport. Vervangt beide sbatches. |
| [hpsearch_coverage_report.py](8_rapportages/hpsearch_coverage_report.py) | Gekopieerd; accepteert nu een directe run-map (i.p.v. alleen een run_label onder de vaste root). |
| [create_model_report.py](8_rapportages/create_model_report.py) | Gekopieerd; `generate_report` kreeg optionele `run_folder`/`out_path` parameters. |

### Gebruik

```bash
cd 8_rapportages

# Dekkingsgraad + modelrapport voor een grid-run:
python run_report.py --run-dir /pad/naar/grid_run

# Alleen de dekkingsgraad-tabel:
python run_report.py --run-dir /pad/naar/grid_run --coverage-only

# Eigen rapport-pad:
python run_report.py --run-dir /pad/naar/grid_run --out /pad/rapport.txt
```

### Tests

[9_testing/test_report.py](9_testing/test_report.py) — geneste grid-layout, dekkingsgraad-tabel,
modelrapport met kop/staart, de gecombineerde runner, en `--coverage-only`.

## 12. OUD: Model trainen op alle operators
*(origineel: stap 19)*

Status: **verouderd** — vervangen door stap 8.

## 13. Data verwijderen
*(origineel: stap 20)*

Status: **n.v.t. op organisatie** — lokaal beheer van de cleaned-mappen; geen HPC-cluster-scratch om op
te ruimen.
