# Openwater — 9-module pipeline · usage guide (Windows / cmd)

> 🇬🇧 **English first.** 🇳🇱 Nederlandse versie staat onderaan ([spring erheen](#-nederlands)).

This folder (`openwater`) contains the complete **pure-Python software for gambling-addiction
prevention** in 9 steps: **cleaning → sorting → features → operator-stats → active-filter →
merge/sample → modelling → reporting** (+ logging). You run everything with **Poetry** — no manual
fiddling with pip or `PYTHONPATH`.

> For the architecture / per-step explanation: see `README.md` (the technical doc). This
> `README_organisatie.md` is the **cookbook** guide.

---

## 📋 Data delivery spec (which WOK files + columns?)

The full list of **WOK files + columns** the pipeline needs is in
**[`databehoefte.txt`](0_dummy_data/databehoefte.txt)** — 13 files, snake_case, `;`-separated, comma decimals,
`NULL` for missing.

That list is **generated automatically from the code (`FEATURES_REGISTRY`)**, so it never goes
stale. Regenerate it after a feature change with:

```cmd
poetry run python 0_dummy_data\_inventariseer_databehoefte.py --txt 0_dummy_data\databehoefte.txt
```

---

## 0. Prerequisites (one-time)

1. **Python 3.11, 3.12 or 3.13** — install via [python.org](https://www.python.org/downloads/),
   tick **"Add python.exe to PATH"**.
2. **Poetry** — in a **cmd** window:
   ```cmd
   py -m pip install --user poetry
   poetry --version
   ```
   (`poetry` not working? Close cmd and reopen, or use `py -m poetry ...`.)

---

## 1. Get it + install

1. Copy the whole **`openwater`** folder to the Windows machine (e.g. `C:\openwater`)
   — via zip, shared drive or `git`.
2. Open **cmd** in that folder and install the environment:
   ```cmd
   cd C:\openwater
   poetry install
   ```
   Poetry creates a virtual environment with all dependencies (pandas, scikit-learn, xgboost,
   lightgbm, optuna, imbalanced-learn, matplotlib, …). Then you run commands with
   **`poetry run python ...`**.

Quick check that it works:
```cmd
poetry run python -c "import pandas, sklearn, xgboost, optuna; print('OK')"
```

---

## 2. Try it on the **bundled test data**

The folder already contains a ready-made test set: **`0_dummy_data\voorbeeld_fake\`** (100 players,
**multi-period**: activity Jan–Apr 2026, exclusions split across **two target months** May + June).
Those two target months let you run a **real temporal holdout** (validation y=May, test y=June).
The "all-in-one" orchestrator expects the data in `Operator_*` subfolders, so we set those up first:

```cmd
mkdir 0_dummy_data\data\Operator_a
copy 0_dummy_data\voorbeeld_fake\*.csv 0_dummy_data\data\Operator_a\

poetry run python run_pipeline.py ^
  --data-dir 0_dummy_data\data --out-dir 0_dummy_data\out ^
  --base-scenario Flexible_spanish_plus ^
  --x-tijdspad 01012026:30042026 --y-tijdspad 01052026:31052026 ^
  --x-test-tijdspad 01012026:30042026 --y-test-tijdspad 01062026:30062026 ^
  --optuna --optuna-time 60
```

Here **validation** y=May and **test** y=June — a genuinely later holdout window. (In **cmd**, `^`
is the line-continuation char; in **PowerShell** you use `` ` ``.)

You'll see a banner per step; the final result is under `0_dummy_data\out\` (`clean\`, `feature_files\`,
`hpo_dataset\`, `model_optuna\`). A successful run ends with `✅ KLAAR`.

> Leave out `--x-test-tijdspad`/`--y-test-tijdspad` and `test = valid` is mirrored and the pipeline
> warns with **"GEEN holdout"** — only for a quick smoke test, not for a leakage/model assessment
> (see §6).

### Running all steps separately (to inspect / iterate)
Each step runs on its own with a plain `cmd` runner. Steps **1 through 5** chain via
`--parent-dir 0_dummy_data\out\clean` (the standalone `run_features.py` writes `0_dummy_data\out\clean\features_*.csv`, and
3b/4/5 pick those up automatically):
```cmd
poetry run python 1_cleaning\run_cleaning.py   --input-dir 0_dummy_data\voorbeeld_fake --clean-out-dir 0_dummy_data\out\clean
poetry run python 2_sorting\run_sorting.py     --parent-dir 0_dummy_data\out\clean
poetry run python 3_features\run_features.py   --parent-dir 0_dummy_data\out\clean ^
  --scenario Flexible_spanish_plus --x-tijdspad 01012026:30042026 --y-tijdspad 01052026:31052026
poetry run python 3b_operator_specific_features\run_operator_features.py --parent-dir 0_dummy_data\out\clean
poetry run python 4_active_filter\run_active_filter.py --parent-dir 0_dummy_data\out\clean --cutoffs "01052026|01062026"
poetry run python 5_descriptive\run_descriptive.py --parent-dir 0_dummy_data\out\clean ^
  --target y_self_exclusion_20260501_20260531
```

Steps **6 (merge & sample)**, **7/8 (modelling)** and **9 (logging)** work on the combined HPO
dataset. The one-shot (above) builds it as `0_dummy_data\out\hpo_dataset`; on that you iterate the model without
cleaning/sorting/feature-building again:
```cmd
poetry run python 7_modelling\run_optuna.py --config 0_dummy_data\out\model_optuna\hpsearch_config_effective.yaml ^
  --dataset-path 0_dummy_data\out\hpo_dataset --out-dir 0_dummy_data\out\model_rerun --time-budget 120 --validate-best
poetry run python 8_rapportages\run_report.py --run-dir 0_dummy_data\out\model_grid
poetry run python _logging\run_with_log.py --logdir 0_dummy_data\out\logs --phase optuna -- ^
  python 7_modelling\run_optuna.py --config 0_dummy_data\out\model_optuna\hpsearch_config_effective.yaml ^
    --dataset-path 0_dummy_data\out\hpo_dataset --out-dir 0_dummy_data\out\model_rerun --time-budget 120
```

> 3b and 5 pick only `features_*` files via `--parent-dir`; otherwise point `--features-csv <path>`.
> The standalone merge (step 6, `run_merge_sample.py`) is config-coupled and error-prone — the
> one-shot already builds the `hpo_dataset`; see `README.md` (§6) for the full merge flags.

---

## 3. **Large** test data (generate it yourself) and running on it

First make a large dataset — **streaming**, so even files of several GB don't blow up memory. Pick a
target size or a number of players:

```cmd
:: up to ~2 GB (sum across all 13 tables)
poetry run python 0_dummy_data\_maak_fake_big_data.py --target-gb 2 --out 0_dummy_data\fake_big

:: or a fixed number of players
poetry run python 0_dummy_data\_maak_fake_big_data.py --players 1000000 --operators 26 --out 0_dummy_data\fake_big
```

The large generator is also **multi-period** (May + June targets). Then through the pipeline (same
holdout windows; raise the Optuna budget if needed):
```cmd
mkdir 0_dummy_data\data_big\Operator_a
copy 0_dummy_data\fake_big\*.csv 0_dummy_data\data_big\Operator_a\

poetry run python run_pipeline.py ^
  --data-dir 0_dummy_data\data_big --out-dir 0_dummy_data\out_big ^
  --base-scenario Flexible_spanish_plus ^
  --x-tijdspad 01012026:30042026 --y-tijdspad 01052026:31052026 ^
  --x-test-tijdspad 01012026:30042026 --y-test-tijdspad 01062026:30062026 ^
  --optuna --optuna-time 900 --chunksize 200000
```

> **Tip for very large data:** lower `--chunksize` (e.g. `100000`) if memory is tight; the
> cleaning/sorting/features read streaming in chunks, so memory stays manageable.

---

## 4. Running on **external (real organisation) data**

The pipeline expects the **relational organisation export format**: one folder with **one CSV per
table** (`WOK_Bet.csv`, `WOK_Bet_Transaction.csv`, `WOK_Player_Profile.csv`, …), `;`-separated,
comma decimals, `NULL` for missing. The monthly bucketing is **data-driven**: whatever date range
your data has, it makes the right monthly buckets automatically.

**Steps:**
1. Put your export folder ready as an operator subfolder (one operator):
   ```cmd
   mkdir data_extern\Operator_a
   copy C:\path\to\your_export\*.csv data_extern\Operator_a\
   ```
   Multiple operators you want to keep separate? Make `Operator_a`, `Operator_b`, … each with their
   own CSVs.
2. Choose the **windows** that fit your data — features (`--x-tijdspad`) and target/exclusion
   (`--y-tijdspad`), format `DDMMYYYY:DDMMYYYY`. Use a **real holdout**: a test target window after
   the validation target window (`--x-test-tijdspad`/`--y-test-tijdspad`). Example:
   ```cmd
   poetry run python run_pipeline.py ^
     --data-dir data_extern --out-dir out_extern ^
     --base-scenario Flexible_spanish_plus ^
     --x-tijdspad 01012026:30042026 --y-tijdspad 01052026:31052026 ^
     --x-test-tijdspad 01012026:31052026 --y-test-tijdspad 01062026:30062026 ^
     --optuna --optuna-time 900
   ```
   (Validation target = May, test target = June → a later, unseen holdout window.)

**Two flags for deviating data:**
- `--ignore-eod-balance` — if your `WOK_Player_Profile` has no `player_profile_eod_balance` column
  (features f26/f27/f28 then remain missing instead of assuming an opening balance of zero).
- `--also-inactives` — if the active filter filters everyone out (no status record before the cutoff
  in your data); then you skip the active filter.

The full list of flags (real holdout with `--validation-period-prefixes` / `--test-period-prefixes`,
`--operator-folds`, gridsearch instead of optuna, sampling, etc.) is in `README.md` and via:
```cmd
poetry run python run_pipeline.py --help
```

---

## 5. Good to know

- **Output folders** are created under `--out-dir`; nothing is deleted. Give each run its own
  `--out-dir` or clean up yourself.
- **Paths with spaces**: wrap in `"..."`. The code itself is path-independent.
- **Steps 5 (descriptive) and 9 (logging)** aren't in the one-shot; run them separately with
  `5_descriptive\run_descriptive.py` resp. `_logging\run_with_log.py` (see `README.md`).
- **Data-driven monthly bucketing**: the sorting picks buckets based on the actual date range of
  your data — no fixed window, no data loss.
- **Dutch play times**: clock hours, weekdays and transaction activity dates use
  `Europe/Amsterdam`, including summer/winter time. This applies to F41–F45, active-day
  counts, calendar-day spans, streaks and daily/weekly aggregates, including the basis
  features for night/weekend activity. Conversion runs per chunk without extra file scans.
  Source timestamps, analysis-window boundaries, balance snapshots and elapsed durations
  remain UTC; profile `Extraction_Date` reporting-day counts retain their UTC definition.
- **F16 financial activation**: uses the first successful stake, deposit or withdrawal in the
  available history and the last such transaction in the feature period. Supply transaction
  history from before that period to recover the actual activation date. The period includes
  the entire UTC end date; Dutch calendar-day differences have a minimum of 1 and no upper limit.
- **F26–F28 opening balances**: use the latest snapshot at/before the feature-window start,
  or reconstruct from the earliest later snapshot within that window. Older snapshots are
  advanced using the intervening transactions. The reconstruction is shared across the three
  features and needs at most one extra chunked transaction scan, with state per player.
  It requires complete, deduplicated movements between the snapshot and the start; the
  snapshot timestamp is `Extraction_Date`, and transactions at that timestamp follow it.
  Without a usable snapshot the features stay missing, including in the output CSV.
- **Equal timestamps (F25/F26–F28/F51)**: F26–F28 apply all successful movements of a
  player at the same UTC timestamp together. Drops compare the balance before and after
  that moment; deposits at that moment do not count as subsequent deposits. F27 counts
  each deposit at a later eligible moment, F28 measures until that moment once. Transactions
  must be chronological per player across files; backward timestamps make that player's
  balance features unknown. Processing retains one unfinished group per player, rather than
  full transaction histories. F25 chooses the latest in-window extraction per modification
  time and retains one record per distinct modification. Conflicting statuses at that
  extraction leave F25 unknown. Conflicting balances at the selected snapshot leave the
  opening balance unknown. Conflicting final bet statuses at the latest extraction leave
  ordinary F51 losses unknown; a newer status can resolve that conflict. VOID_BET refund
  evidence remains independent. These rules also apply across chunk and file boundaries.
- **Identity-separated evaluation**: `Niels_Identity_Confounding_switch=True` is the default.
  The grouping key is `(operator, Player_Profile_ID)`, using the same operator identifier
  as operator-level folds (the operator folder name). All periods for that combination
  remain together. The same player ID at two operators represents two different groups.
  For overlapping development/test identities, 20% of the available identities are reserved
  for the test set before sampling; their earlier rows are excluded from development, and
  only their test-period rows are evaluated. Already disjoint holdouts, including operator
  holdouts, keep their existing boundaries. Cross-validation keeps all rows of an identity
  together. With no test-period data, only grouped cross-validation is applied.
  The split is reproducible using `random_state` and is recorded in dataset metadata.
  Set `Niels_Identity_Confounding_switch=False` in Python/YAML, or pass
  `--no-Niels_Identity_Confounding_switch`, to use the previous splitting behaviour.
  Rebuild prepared datasets when changing the setting; older datasets without identity
  metadata cannot be used with the switch on. Group identifiers are not model inputs.
  This tests generalisation to held-out operator/player combinations; use a later test
  period as well when evaluating future performance. It does not link one person across
  different operators. `Niels_identity_column` / `--niels-identity-column` can select a
  different player identity column in the feature CSVs; the operator remains part of the key.
- **Void rules for F44/F45/F48/F51**: F44/F45 use stakes net of successful
  `VOID_BET`/`VOID_STAKE` refunds, allocated proportionally to the linked bet/session's
  original stakes and placement hours. Refunds after the exclusive window end are ignored.
  Their chunked calculation is shared per window; reference maps and aggregate sums remain
  in memory. Missing refund links or zero net stakes produce missing shares. F48 excludes
  cancelled bets and bets with successful `VOID_BET` from both counts. F51 combines bet
  updates by operator/Bet_ID and measures from settlement to the next bet with a successful
  stake. Only settled bets without their own positive winnings/cash-outs qualify as losses;
  the first settled `Extraction_Date` approximates resolution time. `VOID_BET` also qualifies
  by project choice, using the refund timestamp. Both closure and next placement must be in
  the feature window; earlier transaction history is used for outcomes. The linked history
  must be complete to establish that no prize was paid. F51 scans each of
  its three tables once and retains reference indexes and per-bet state, not event rows.
  Historical settled reports are needed: a later final export alone cannot recover earlier
  resolution times. Undefined results stay missing in the output CSV.

---

## 6. Leakage check (verify nothing leaks)

The folder contains a **noise dataset** `0_dummy_data\voorbeeld_noise\` — same structure as `0_dummy_data\voorbeeld_fake`, but
the features are generated **independent of the label** (`--no-signal`). An honest model should
score on the **holdout test ≈ chance level** (AUC ≈ 0.5, AUPRC ≈ the base rate). If it still scores
high → the target leaks into the features.

> **Crucial:** always assess this on a **real holdout** (test target window after the validation
> target, as below). Without a separate test window, `test = valid` is mirrored and even pure noise
> shows a high "test" score — that's **memorisation, not a leak**. That's why the bundled data is
> multi-period.

```cmd
:: noise through the pipeline, with a REAL holdout (validation=May, test=June)
mkdir 0_dummy_data\data_noise\Operator_a
copy 0_dummy_data\voorbeeld_noise\*.csv 0_dummy_data\data_noise\Operator_a\

poetry run python run_pipeline.py ^
  --data-dir 0_dummy_data\data_noise --out-dir 0_dummy_data\out_noise ^
  --base-scenario Flexible_spanish_plus ^
  --x-tijdspad 01012026:30042026 --y-tijdspad 01052026:31052026 ^
  --x-test-tijdspad 01012026:30042026 --y-test-tijdspad 01062026:30062026 ^
  --optuna --optuna-time 60
```

Look in `0_dummy_data\out_noise\model_optuna\` (incl. `optuna_best_test_result.csv`, `top_model_statistics.txt`):
- **noise** → test-AUC ≈ 0.5, test-AUPRC ≈ base rate → **no leakage** ✅
- compare with the same run on `0_dummy_data\voorbeeld_fake` (signal) → that should score **clearly higher**.

A big difference between the two = healthy. Does noise also score high → investigate the leak.

> Need a large noise set? `poetry run python 0_dummy_data\_maak_fake_big_data.py --no-signal --target-gb 2 --out 0_dummy_data\noise_big`.

<br>

---
---

# 🇳🇱 Nederlands

# Openwater — 9-module pijplijn · gebruiksinstructie (Windows / cmd)

Deze map (`openwater`) bevat de volledige **pure-Python software voor preventie van gokverslaving**
in 9 stappen: **cleaning → sorting → features → operator-stats → active-filter → merge/sample →
modelling → rapportage** (+ logging). Je draait alles met **Poetry** — geen handmatig gedoe met
pip of `PYTHONPATH`.

> Voor de architectuur/uitleg per stap: zie `README.md` (de technische doc). Deze `README_organisatie.md`
> is de **kook-het-zelf**-handleiding.

---

## 📋 Data-aanlever-spec (welke WOK-bestanden + kolommen?)

De volledige lijst **WOK-bestanden + kolommen** die de pijplijn nodig heeft staat in
**[`databehoefte.txt`](0_dummy_data/databehoefte.txt)** — 13 bestanden, snake_case, `;`-gescheiden,
komma-decimalen, `NULL` voor missing.

Die lijst wordt **automatisch uit de code (`FEATURES_REGISTRY`) gegenereerd**, zodat 'ie nooit
veroudert. Regenereer 'm na een feature-wijziging met:

```cmd
poetry run python 0_dummy_data\_inventariseer_databehoefte.py --txt 0_dummy_data\databehoefte.txt
```

---

## 0. Vereisten (eenmalig)

1. **Python 3.11, 3.12 of 3.13** — installeer via [python.org](https://www.python.org/downloads/),
   vink **"Add python.exe to PATH"** aan.
2. **Poetry** — in een **cmd**-venster:
   ```cmd
   py -m pip install --user poetry
   poetry --version
   ```
   (Werkt `poetry` niet? Sluit cmd en open opnieuw, of gebruik `py -m poetry ...`.)

---

## 1. Binnenhalen + installeren

1. Kopieer de hele map **`openwater`** naar de Windows-machine (bv. `C:\openwater`)
   — via zip, gedeelde schijf of `git`.
2. Open **cmd** in die map en installeer de omgeving:
   ```cmd
   cd C:\openwater
   poetry install
   ```
   Poetry maakt een virtuele omgeving met álle dependencies (pandas, scikit-learn, xgboost,
   lightgbm, optuna, imbalanced-learn, matplotlib, …). Daarna draai je commando's met
   **`poetry run python ...`**.

Snelle controle dat het werkt:
```cmd
poetry run python -c "import pandas, sklearn, xgboost, optuna; print('OK')"
```

---

## 2. Uitproberen op de **meegeleverde testdata**

In de map zit al een kant-en-klare testset: **`0_dummy_data\voorbeeld_fake\`** (100 spelers, **multi-period**:
activiteit jan–apr 2026, uitsluiters verdeeld over **twee target-maanden** mei + juni). Door die
twee target-maanden kun je een **echte temporele holdout** draaien (validatie y=mei, test y=juni).
De "alles-in-één" orchestrator verwacht de data in `Operator_*`-submappen, dus die zetten we eerst
klaar:

```cmd
mkdir 0_dummy_data\data\Operator_a
copy 0_dummy_data\voorbeeld_fake\*.csv 0_dummy_data\data\Operator_a\

poetry run python run_pipeline.py ^
  --data-dir 0_dummy_data\data --out-dir 0_dummy_data\out ^
  --base-scenario Flexible_spanish_plus ^
  --x-tijdspad 01012026:30042026 --y-tijdspad 01052026:31052026 ^
  --x-test-tijdspad 01012026:30042026 --y-test-tijdspad 01062026:30062026 ^
  --optuna --optuna-time 60
```

Hier is **validatie** y=mei en **test** y=juni — een echt later holdout-venster. (In **cmd** is `^`
het regelvervolg-teken; in **PowerShell** gebruik je `` ` ``.)

Je ziet per stap een banner; het eindresultaat staat onder `0_dummy_data\out\` (`clean\`, `feature_files\`,
`hpo_dataset\`, `model_optuna\`). Een geslaagde run eindigt met `✅ KLAAR`.

> Laat je `--x-test-tijdspad`/`--y-test-tijdspad` weg, dan spiegelt `test = valid` en waarschuwt
> de pijplijn met **"GEEN holdout"** — alleen voor een snelle rooktest, niet voor een
> lekkage-/modelbeoordeling (zie §6).

### Alle stappen los draaien (om te inspecteren / itereren)
Elke stap draait apart op een gewone `cmd`-runner. Stappen **1 t/m 5** haken op elkaar in via
`--parent-dir 0_dummy_data\out\clean` (de losse `run_features.py` schrijft `0_dummy_data\out\clean\features_*.csv`, en 3b/4/5
pikken die automatisch op):
```cmd
poetry run python 1_cleaning\run_cleaning.py   --input-dir 0_dummy_data\voorbeeld_fake --clean-out-dir 0_dummy_data\out\clean
poetry run python 2_sorting\run_sorting.py     --parent-dir 0_dummy_data\out\clean
poetry run python 3_features\run_features.py   --parent-dir 0_dummy_data\out\clean ^
  --scenario Flexible_spanish_plus --x-tijdspad 01012026:30042026 --y-tijdspad 01052026:31052026
poetry run python 3b_operator_specific_features\run_operator_features.py --parent-dir 0_dummy_data\out\clean
poetry run python 4_active_filter\run_active_filter.py --parent-dir 0_dummy_data\out\clean --cutoffs "01052026|01062026"
poetry run python 5_descriptive\run_descriptive.py --parent-dir 0_dummy_data\out\clean ^
  --target y_self_exclusion_20260501_20260531
```

Stappen **6 (merge & sample)**, **7/8 (modelling)** en **9 (logging)** werken op de gecombineerde
HPO-dataset. Die bouwt de one-shot (hierboven) als `0_dummy_data\out\hpo_dataset`; daarop itereer je het model
zónder opnieuw te clean/sort/feature'en:
```cmd
poetry run python 7_modelling\run_optuna.py --config 0_dummy_data\out\model_optuna\hpsearch_config_effective.yaml ^
  --dataset-path 0_dummy_data\out\hpo_dataset --out-dir 0_dummy_data\out\model_rerun --time-budget 120 --validate-best
poetry run python 8_rapportages\run_report.py --run-dir 0_dummy_data\out\model_grid
poetry run python _logging\run_with_log.py --logdir 0_dummy_data\out\logs --phase optuna -- ^
  python 7_modelling\run_optuna.py --config 0_dummy_data\out\model_optuna\hpsearch_config_effective.yaml ^
    --dataset-path 0_dummy_data\out\hpo_dataset --out-dir 0_dummy_data\out\model_rerun --time-budget 120
```

> 3b en 5 kiezen via `--parent-dir` alléén `features_*`-bestanden; wijs anders `--features-csv <pad>`
> aan. De losse merge (stap 6, `run_merge_sample.py`) is config-gekoppeld en foutgevoelig — de
> one-shot bouwt de `hpo_dataset` al; zie `README.md` (§6) voor de volledige merge-vlaggen.

---

## 3. **Grote** testdata (zelf genereren) en daarop draaien

Maak eerst een grote dataset — **streamend**, dus ook bestanden van enkele GB lopen niet uit het
geheugen. Kies een doelgrootte óf een aantal spelers:

```cmd
:: tot ~2 GB (som over alle 13 tabellen)
poetry run python 0_dummy_data\_maak_fake_big_data.py --target-gb 2 --out 0_dummy_data\fake_big

:: of een vast aantal spelers
poetry run python 0_dummy_data\_maak_fake_big_data.py --players 1000000 --operators 26 --out 0_dummy_data\fake_big
```

Ook de grote generator is **multi-period** (mei + juni targets). Daarna door de pijplijn (zelfde
holdout-vensters; verhoog evt. het Optuna-budget):
```cmd
mkdir 0_dummy_data\data_big\Operator_a
copy 0_dummy_data\fake_big\*.csv 0_dummy_data\data_big\Operator_a\

poetry run python run_pipeline.py ^
  --data-dir 0_dummy_data\data_big --out-dir 0_dummy_data\out_big ^
  --base-scenario Flexible_spanish_plus ^
  --x-tijdspad 01012026:30042026 --y-tijdspad 01052026:31052026 ^
  --x-test-tijdspad 01012026:30042026 --y-test-tijdspad 01062026:30062026 ^
  --optuna --optuna-time 900 --chunksize 200000
```

> **Tip bij heel grote data:** verlaag `--chunksize` (bv. `100000`) als het geheugen krap is; de
> cleaning/sorting/features lezen streamend in chunks, dus geheugen blijft beheersbaar.

---

## 4. Draaien op **externe (echte organisatie) data**

De pijplijn verwacht het **relationele organisatie-exportformaat**: één map met **één CSV per tabel**
(`WOK_Bet.csv`, `WOK_Bet_Transaction.csv`, `WOK_Player_Profile.csv`, …), `;`-gescheiden, decimalen
met komma, `NULL` voor missing. De maand-bucketing is **datagedreven**: welk datumbereik je data
ook heeft, hij maakt automatisch de juiste maand-buckets.

**Stappen:**
1. Zet je export-map klaar als operator-submap (één operator):
   ```cmd
   mkdir data_extern\Operator_a
   copy C:\pad\naar\jouw_export\*.csv data_extern\Operator_a\
   ```
   Heb je meerdere operators die je apart wilt houden? Maak dan `Operator_a`, `Operator_b`, … elk
   met hun eigen CSV's.
2. Kies de **vensters** die bij jóuw data passen — features (`--x-tijdspad`) en doel/uitsluiting
   (`--y-tijdspad`), formaat `DDMMYYYY:DDMMYYYY`. Gebruik een **echt holdout**: een test-doelvenster
   ná het validatie-doelvenster (`--x-test-tijdspad`/`--y-test-tijdspad`). Voorbeeld:
   ```cmd
   poetry run python run_pipeline.py ^
     --data-dir data_extern --out-dir out_extern ^
     --base-scenario Flexible_spanish_plus ^
     --x-tijdspad 01012026:30042026 --y-tijdspad 01052026:31052026 ^
     --x-test-tijdspad 01012026:31052026 --y-test-tijdspad 01062026:30062026 ^
     --optuna --optuna-time 900
   ```
   (Validatie-doel = mei, test-doel = juni → een later, ongezien holdout-venster.)

**Twee vlaggen voor afwijkende data:**
- `--ignore-eod-balance` — als je `WOK_Player_Profile` géén `player_profile_eod_balance`-kolom
  heeft (features f26/f27/f28 blijven dan onbekend; er wordt geen beginsaldo van nul aangenomen).
- `--also-inactives` — als de active-filter iedereen wegfiltert (geen statusrecord vóór de
  cutoff in je data); dan sla je het actief-filter over.

De volledige lijst vlaggen (echte holdout met `--validation-period-prefixes` /
`--test-period-prefixes`, `--operator-folds`, gridsearch i.p.v. optuna, sampling, enz.) staat in
`README.md` en via:
```cmd
poetry run python run_pipeline.py --help
```

---

## 5. Goed om te weten

- **Outputmappen** worden onder `--out-dir` aangemaakt; niets wordt verwijderd. Geef per run een
  eigen `--out-dir` of ruim zelf op.
- **Paden met spaties**: zet tussen `"..."`. De code zelf is padonafhankelijk.
- **Stappen 5 (descriptive) en 9 (logging)** zitten niet in de one-shot; draai die los met
  `5_descriptive\run_descriptive.py` resp. `_logging\run_with_log.py` (zie `README.md`).
- **Datagedreven maand-bucketing**: de sortering kiest de buckets op basis van het werkelijke
  datumbereik van je data — geen vast venster, geen dataverlies.
- **Nederlandse speeltijden**: uren, weekdagen en transactiedagen gebruiken
  `Europe/Amsterdam`, inclusief zomer- en wintertijd. Dit geldt voor F41–F45, actieve dagen,
  kalenderdagverschillen, speelreeksen en dag-/weekaggregaties, waaronder de basisfeatures
  voor nacht- en weekendactiviteit. De omzetting gebeurt per chunk zonder extra bestandsdoorlopen.
  Brontijdstippen, analysevenstergrenzen, saldomomenten en verstreken tijd blijven UTC;
  rapportagedagen op basis van profiel-`Extraction_Date` behouden hun UTC-definitie.
- **F16 financiële activatie**: gebruikt de eerste succesvolle inzet, storting of opname in de
  beschikbare historie en de laatste daarvan binnen het featurevenster. Lever ook eerdere
  transactiehistorie aan om de echte activatiedatum te kunnen bepalen. De volledige UTC-einddatum
  telt mee; Nederlandse kalenderdagverschillen hebben een minimum van 1 en geen bovengrens.
- **Beginsaldi voor F26–F28**: gebruiken het laatste saldomoment op/vóór de start van het
  featurevenster, of rekenen terug vanaf het eerste latere saldomoment binnen dat venster.
  Bij oudere saldi worden ook de tussenliggende transacties verwerkt. De reconstructie wordt
  gedeeld door de drie features en kost hooguit één extra chunkgewijze transactiedoorloop,
  met gegevens per speler in het geheugen. Alle mutaties tussen het saldomoment en de start
  moeten volledig en gededupliceerd zijn. Het saldotijdstip is `Extraction_Date`; transacties
  exact op dat tijdstip volgen op het saldo. Zonder bruikbaar saldo blijven de features
  onbekend, ook in het uiteindelijke CSV-bestand.
- **Gelijke tijdstippen (F25/F26–F28/F51)**: F26–F28 verwerken alle succesvolle mutaties
  van een speler op hetzelfde UTC-tijdstip samen. Een daling wordt bepaald tussen het saldo
  vóór en ná dat moment; stortingen op dat moment gelden niet als latere stortingen. F27
  telt iedere storting op een later geschikt moment, F28 meet één interval tot dat moment.
  Transacties moeten per speler chronologisch staan, ook over bestanden heen. Een tijdstip
  dat terugloopt maakt diens saldofeatures onbekend. Alleen de onafgeronde groep per speler
  blijft in het geheugen, zonder volledige transactiehistorie. F25 kiest per wijzigingstijd
  de nieuwste extractie binnen het venster en bewaart één record per unieke wijzigingstijd.
  Tegenstrijdige statussen op die extractietijd maken F25 onbekend. Tegenstrijdige saldi op
  het gekozen saldomoment maken het beginsaldo onbekend. Tegenstrijdige laatste betstatussen
  op dezelfde extractietijd maken gewone F51-verliezen onbekend; een nieuwere status kan
  dat oplossen. Een VOID_BET-terugbetaling blijft onafhankelijk bewijs. Deze regels gelden
  ook over chunk- en bestandsgrenzen.
- **Evaluatie met gescheiden identiteiten**: `Niels_Identity_Confounding_switch=True` is de
  standaard. De groepssleutel is `(aanbieder, Player_Profile_ID)`, met dezelfde aanbiederidentifier
  als de operator-folds (de naam van de aanbiedermap). Alle periodes voor die combinatie blijven
  bijeen. Dezelfde speler-ID bij twee aanbieders vormt twee afzonderlijke groepen.
  Bij overlappende ontwikkel-/testidentiteiten wordt vóór het samplen 20% voor de testset
  gereserveerd. Hun eerdere rijen worden uit de ontwikkelset gehouden; alleen hun rijen uit
  de testperiode worden geëvalueerd. Reeds gescheiden holdouts, waaronder operator-holdouts,
  behouden hun bestaande verdeling. Cross-validation houdt alle rijen van dezelfde identiteit
  bijeen. Zonder testperiode wordt alleen cross-validation op identiteitsgroepen toegepast.
  `random_state` maakt de verdeling reproduceerbaar; de datasetmetadata leggen de instelling
  vast. Zet in Python/YAML `Niels_Identity_Confounding_switch=False`, of gebruik
  `--no-Niels_Identity_Confounding_switch`, voor de eerdere manier van splitsen. Bouw de dataset
  opnieuw bij een andere instelling; oude datasets zonder identiteitsmetadata kunnen niet met
  de switch aan worden gebruikt. Groepsidentifiers zijn geen modelkenmerken. Deze verdeling
  toetst generalisatie naar achtergehouden aanbieder/speler-combinaties; gebruik daarnaast een
  latere testperiode om toekomstige prestaties te meten. De verdeling koppelt dezelfde persoon
  bij verschillende aanbieders niet. Met `Niels_identity_column` / `--niels-identity-column`
  kun je een andere spelerskolom in de feature-CSV's kiezen; de aanbieder blijft deel van de sleutel.
- **Voidregels voor F44/F45/F48/F51**: F44/F45 verrekenen succesvolle
  `VOID_BET`/`VOID_STAKE` naar verhouding met de oorspronkelijke inzetten van de gekoppelde
  weddenschap/spelsessie en hun inzettijdstippen. Terugbetalingen na de exclusieve einddatum
  tellen niet mee. De chunkgewijze berekening wordt per venster gedeeld; koppelingsindexen en
  sommen blijven in het geheugen. Ontbrekende terugbetalingskoppelingen of netto-inzet nul
  geven een onbekend aandeel. F48 sluit geannuleerde bets en bets met succesvolle `VOID_BET`
  uit van teller en noemer. F51 combineert betupdates op aanbieder/Bet_ID en meet vanaf
  afwikkeling tot de volgende bet met een succesvolle inzet. Alleen afgewikkelde bets zonder
  eigen positieve prijzen/cash-outs tellen als verlies; de eerste afgewikkelde extractie
  benadert het afwikkelmoment. `VOID_BET` telt volgens onze keuze ook mee, vanaf de
  terugbetaling. Afwikkeling en volgende plaatsing moeten binnen het featurevenster vallen;
  eerdere transacties bepalen mede de uitkomst. De gekoppelde historie moet volledig zijn
  om vast te stellen dat geen prijs is betaald. F51 leest elk van zijn drie tabellen één
  keer en bewaart koppelingsindexen en gegevens per bet, zonder transactiegebeurtenissen
  te bufferen. Historische afgewikkelde rapportages zijn nodig: alleen een latere eindexport
  levert de eerdere afwikkeltijden niet. Niet-berekenbare uitkomsten blijven onbekend in het
  uiteindelijke CSV-bestand.

---

## 6. Lekkage-check (controleer dat er niets lekt)

De map bevat een **noise-dataset** `0_dummy_data\voorbeeld_noise\` — dezelfde structuur als `0_dummy_data\voorbeeld_fake`, maar
de features zijn **onafhankelijk van het label** gegenereerd (`--no-signal`). Een eerlijk model
hoort hierop op de **holdout-test ≈kansniveau** te scoren (AUC ≈ 0.5, AUPRC ≈ de base-rate). Scoort
het tóch hoog → het target lekt in de features.

> **Cruciaal:** beoordeel dit **altijd op een echt holdout** (test-doelvenster ná validatie-doel,
> zoals hieronder). Zónder apart test-venster spiegelt `test = valid` en toont zelfs pure noise een
> hoge "test"-score — dat is **memorisatie, geen lek**. Daarom is de meegeleverde data multi-period.

```cmd
:: noise door de pijplijn, met ECHT holdout (validatie=mei, test=juni)
mkdir 0_dummy_data\data_noise\Operator_a
copy 0_dummy_data\voorbeeld_noise\*.csv 0_dummy_data\data_noise\Operator_a\

poetry run python run_pipeline.py ^
  --data-dir 0_dummy_data\data_noise --out-dir 0_dummy_data\out_noise ^
  --base-scenario Flexible_spanish_plus ^
  --x-tijdspad 01012026:30042026 --y-tijdspad 01052026:31052026 ^
  --x-test-tijdspad 01012026:30042026 --y-test-tijdspad 01062026:30062026 ^
  --optuna --optuna-time 60
```

Kijk in `0_dummy_data\out_noise\model_optuna\` (o.a. `optuna_best_test_result.csv`, `top_model_statistics.txt`):
- **noise** → test-AUC ≈ 0.5, test-AUPRC ≈ base-rate → **geen lekkage** ✅
- vergelijk met dezelfde run op `0_dummy_data\voorbeeld_fake` (signaal) → die hoort **duidelijk hoger** te scoren.

Een groot verschil tussen beide = gezond. Scoort noise óók hoog → onderzoek de lekkage.

> Grote noise-set nodig? `poetry run python 0_dummy_data\_maak_fake_big_data.py --no-signal --target-gb 2 --out 0_dummy_data\noise_big`.
