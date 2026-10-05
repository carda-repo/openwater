#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_pipeline.py  (organisatie-versie)
===============================

Pure-Python feature-engineering pijplijn voor stap 3 (**Features bouwen**) uit de
organisatie-README (origineel: stap 8, "Features voor een operator maken").

Dit is de *parse*-helft van `clean_and_parse.py` uit de hoofdrepo. De `SCENARIOS`-dict,
`_safe_merge` en `run_scenario` zijn **verbatim** overgenomen (zodat het resultaat
byte-identiek is aan `clean_and_parse.py --mode parse`); de clean-helft is hier weggelaten.

Input is een geschoonde (en bij voorkeur gesorteerde) map met WOK-CSV's — de output van
stap 1 (en eventueel stap 2). Output is één features-CSV met één rij per `Player_Profile_ID`.
F26–F28 delen een saldoreconstructie per venster; ontbrekende saldofeatures blijven NaN.

Op Snellius liep dit per operator via `run_parse_only.sbatch` → `parse_runner.py`; op de organisatie
wijs je gewoon één cleaned-map aan en draai je `run_features.py`.
"""

from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import inspect
import re

# Lokale modules (liggen naast dit bestand in _organisatie_code/3_features/)
from path_finding import load_tables_local
from balance_reconstruction import reconstruct_start_balances
from stake_time_shares import net_stake_time_shares
from feature_engineering import FEATURES_REGISTRY
from feature_engineering_spanish import FEATURES_REGISTRY as FEATURES_REGISTRY_SPANISH
from feature_engineering_basis import FEATURES_REGISTRY as FEATURES_REGISTRY_BASIS

# Voeg Spaanse en basis features toe aan hoofdregistry (identiek aan clean_and_parse.py)
FEATURES_REGISTRY.update(FEATURES_REGISTRY_SPANISH)
FEATURES_REGISTRY.update(FEATURES_REGISTRY_BASIS)

# -----------------------------------------------------------------------------
# Scenario-definities  (VERBATIM uit clean_and_parse.py)
# -----------------------------------------------------------------------------
SCENARIOS: Dict[str, Dict] = {
    "Scenario_1": {"features": ["avg_nr_of_game_transactions"]},
    "Scenario_2": {"features": ["transaction_amount_sum", "transaction_amount_sum_in_games"]},
    "Scenario_3": {"features": ["transaction_amount_sum", "transaction_amount_sum_in_games", "time_eas"]},
    "Scenario_4": {"features": ["nr_of_bank_accounts", "nr_of_risk_classes", "latest_limits"]},
    "Scenario_5": {"features": ["avg_nr_of_game_transactions", "avg_nr_of_bet_parts_per_bet", "latest_limits"]},
    "Scenario_6": {"features": ["nr_of_bank_accounts", "nr_of_risk_classes", "avg_nr_of_game_transactions",
                                "avg_nr_of_bet_parts_per_bet", "latest_limits", "nr_of_complaints_and_variance_in_nr_responses_per_complaint"]},
    "Scenario_for_local_test": {"features": ["nr_of_bank_accounts", "nr_of_risk_classes", "avg_nr_of_game_transactions",
                                "latest_limits", "nr_of_complaints_and_variance_in_nr_responses_per_complaint", "has_person_ever_requested_an_exclusion"]},
    "Scenario_for_b": {"features": ["nr_of_bank_accounts", "nr_of_risk_classes",
                                "latest_limits", "nr_of_complaints_and_variance_in_nr_responses_per_complaint", "transactions", "has_person_ever_requested_an_exclusion"]},
    "Only_transactions_and_target": {"features": ["transactions", "has_person_ever_requested_an_exclusion"]},
    "Only_monthly_target": {"features": ["maak_exclusion_maand_variabelen"]},
    "Two_months_target_two_months_features": {"features": ["april_mei_2025_features", "maak_self_exclusion_juni_juli_2025"]},
    "Only_flexible_y_target": {"features": ["maak_self_exclusion_flexible_y"]},
    "Flexible_test": {"features": ["maak_flexible_x_features", "maak_self_exclusion_flexible_y"]},
    "Flexible_spanish_test": {"features": ["f0_net_winloss", "f26_balance_drop_frequency",
        # Target
        "maak_self_exclusion_flexible_y"
    ]},
    "Flexible_spanish": {"features": [
        "f0_net_winloss", "f1_active_days", "f3_total_wagered",
        "f2_net_loss_per_day", "f4_average_wager_per_day",
        "f8_interactions_per_day", "f11_withdrawals_per_day", "f12_deposits_per_day",
        "f14_active_period_span", "f15_active_day_fraction",
        # Target
        "maak_self_exclusion_flexible_y"
    ]},
    "Flexible_spanish_extra": {"features": [
        "f0_net_winloss", "f1_active_days", "f3_total_wagered",
        "f2_net_loss_per_day", "f4_average_wager_per_day",
        "f8_interactions_per_day", "f11_withdrawals_per_day", "f12_deposits_per_day",
        "f14_active_period_span", "f15_active_day_fraction", "f25_voluntary_suspensions",
        # Target
        "maak_self_exclusion_flexible_y"
    ]},
    "Flexible_spanish_plus": {"features": [
        # Original Spanish features (10)
        "f0_net_winloss", "f1_active_days", "f3_total_wagered",
        "f2_net_loss_per_day", "f4_average_wager_per_day",
        "f8_interactions_per_day", "f11_withdrawals_per_day", "f12_deposits_per_day",
        "f14_active_period_span", "f15_active_day_fraction",
        # New features batch 1 (8): Demographics, RTP, Deposits/Withdrawals
        "f6_age", "f7_rtp_deviation", "f9_big_wins_per_day",
        "f10_canceled_withdrawals_per_day", "f13_canceled_deposits_per_day",
        "f24_payment_method_variety", "f25_voluntary_suspensions", "f60_deposit_amount_variability",
        # New features batch 2 (6): Account age, Limits, Sessions, Betting patterns
        "f16_account_age", "f22_limit_increases", "f23_limit_decreases",
        "f50_single_bet_percentage", "f61_bet_odds_variability",
        # New features batch 3 (6): Game diversity, Time patterns, Behavioral trends
        "f31_median_rounds_per_session", "f32_game_types_count", "f40_products_per_active_day",
        "f41_heavy_play_hours_count", "f51_median_seconds_loss_to_next_bet", "f52_big_win_wager_increase_count",
        "f53_abs_gradient_wagered_around_median_date",
        # New features batch 4 (6): Balance management, Time-of-day patterns
        "f26_balance_drop_frequency", "f27_deposits_after_balance_below_2_per_day", "f28_median_seconds_below2_to_deposit",
        "f42_morning_interaction_percentage", "f43_evening_interaction_percentage", "f44_morning_stakes_percentage",
        # New features batch 5: JSON-based features from Bet_Parts
        "f17_number_of_bet_countries", "f18_number_of_bet_sports",
        "f19_PROXY_FOR_nr_unique_competitions_by_max_bet_parts",
        "f20_dutch_domestic_bets_percentage",
        # New features batch 6 (7): Game segments & Temporal patterns
        "f39_dominant_segment_share", "f45_evening_stakes_percentage",
        "f54_post_median_active_days_percentage", "f55_stake_variance_difference", "f56_stake_cv_difference",
        "f57_longest_daily_streak", "f58_longest_streak_ratio", "f59_median_daily_time_off",
        # New features batch 7: Session-based features
        "f29_sessions_other_predrawn_per_day", "f30_avg_interactions_per_session",
        "f47_median_seconds_session_start_to_period_end",
        # New features batch 8: Bet timing & live/cashout features
        "f46_median_seconds_bet_placed_to_resolved", "f48_percentage_bets_with_cashout",
        "f49_percentage_live_bets",
        # New features batch 9: 70% segment flags (CDB6)
        "f33_f34_f35_f36_f37_f38_segments_cdb6",
        # Target
        "maak_self_exclusion_flexible_y"
    ]},
    "test_nan": {
        "features": [
            "f1_active_days",
            "f8_interactions_per_day",
            "f29_sessions_other_predrawn_per_day",
            "f40_products_per_active_day",
            "f41_heavy_play_hours_count",
            "f51_median_seconds_loss_to_next_bet",
        ]
    },
    "ALL_test": {
        "features": [
            "f0_net_winloss",
            "f3_total_wagered",
            "f25_voluntary_suspensions",
            "f12_deposits_per_day",
            "f11_withdrawals_per_day",
        ]
    },
    "alleen_totale_inzet_and_tijd": {
        "features": [
            "var1a_totaal_ingezet_bedrag",
            "var14d_totale_sessieduur_seconden",
            "maak_self_exclusion_flexible_y"
        ]
    },
    "basic_time_based_variables": {
        "features": [
            "var1a_totaal_ingezet_bedrag",
            "var1b_variantie_ingezet_bedrag_per_dag",
            "var1c_totaal_gewonnen_bedrag",
            "var2a_totaal_aantal_transacties",
            "var2b_variantie_aantal_transacties_per_dag",
            "var3a_totaal_aantal_actieve_dagen",
            "var3b_variantie_aantal_actieve_dagen_per_week",
            "var3c_percentage_actieve_dagen_in_periode",
            "var4a_totaal_aantal_deposit_transacties",
            "var4b_totaal_aantal_withdrawal_transacties",
            "var4c_totaal_aantal_stake_transacties",
            "var4d_totaal_aantal_winning_transacties",
            "var4e_totaal_aantal_other_transacties",
            "var4f_totaal_aantal_bonus_transacties",
            "var5a_totaal_aantal_transaction_instruments",
            "var5b_totaal_aantal_CREDIT_CARD_instrument",
            "var5c_totaal_aantal_ELECTRONIC_MONEY_instrument",
            "var5d_totaal_aantal_BANK_TRANSFER_instrument",
            "var5e_totaal_aantal_OTHER_instrument",
            "var6a_totaal_aantal_dagen_met_5_tranacties_of_meer",
            "var6b_totaal_aantal_dagen_met_25_tranacties_of_meer",
            "var6c_totaal_aantal_dagen_met_100_tranacties_of_meer",
            "var7a_totaal_aantal_limits_in_periode",
            "var7b_totaal_aantal_limit_deposits_in_periode",
            "var7c_totaal_aantal_limit_login_in_periode",
            "var7d_totaal_aantal_limit_balance_in_periode",
            "var8a_geboortedatum",
            "var9a_aantal_bankrekeningen",
            "var10a_status_active",
            "var10b_status_trial",
            "var10c_status_suspended",
            "var10d_status_suspended_death",
            "var10e_status_blocked",
            "var10f_status_self_excluded_temp",
            "var10g_status_self_excluded_indef",
            "var10h_status_other",
            "var10i_gemiddeld_saldo",
            "var10j_rg_class",
            "var10k_totaal_aantal_onsuccesvolle_transacties",
            "var10l_ratio_succesvolle_transacties",
            "var11_totaal_aantal_complaints_in_periode",
            "var12_totaal_aantal_interventions",
            "var13_aantal_game_types",
            "var14a_aantal_game_sessions",
            "var14b_totaal_rondes_game_sessions",
            "var14c_gemiddelde_sessieduur_seconden",
            "var14d_totale_sessieduur_seconden",
            "var15a_totaal_aantal_bets",
            "var15b_totaal_aantal_bets_meerdere_delen",
            "var16_totaal_aantal_responses",
            "var17_self_excl_temp_count",
            "var18_aantal_gok_dagen",
            "var19a_totaal_gokken_weekend",
            "var19b_totaal_gokken_nacht",
            "var20a_max_transacties_per_dag",
            "var20b_max_inzet_per_dag",
            # Target
            "maak_self_exclusion_flexible_y"
        ]
    },
    "Y-target": {"features": ["maak_self_exclusion_flexible_y"]},
}


# -----------------------------------------------------------------------------
# Input-map picker (prefereert gesorteerde mappen, zoals run_parse_only.sbatch)
# -----------------------------------------------------------------------------
_SORTED_RE = re.compile(r"^(?:TEST_)?cleaned_\d{8}_\d{6}_sorted_", re.IGNORECASE)
_CLEANED_RE = re.compile(r"^(?:TEST_)?cleaned_\d{8}_\d{6}$", re.IGNORECASE)


def newest_features_input_dir(parent_dir: str | Path) -> Optional[Path]:
    """
    Kies de meest geschikte input-map voor feature-bouw onder `parent_dir`.

    Net als `run_parse_only.sbatch` heeft een gesorteerde map (`cleaned_*_sorted_*`,
    output van stap 2) voorrang; valt terug op een gewone `cleaned_<stamp>/` (stap 1).
    Retourneert de nieuwste match, of None.
    """
    parent = Path(parent_dir)
    if not parent.is_dir():
        return None
    sorted_dirs = sorted(p for p in parent.iterdir() if p.is_dir() and _SORTED_RE.match(p.name))
    if sorted_dirs:
        return sorted_dirs[-1].resolve()
    cleaned_dirs = sorted(p for p in parent.iterdir() if p.is_dir() and _CLEANED_RE.match(p.name))
    if cleaned_dirs:
        return cleaned_dirs[-1].resolve()
    return None


# -----------------------------------------------------------------------------
# _safe_merge (preserves undefined balance and corrected void features)
# -----------------------------------------------------------------------------
_UNKNOWN_FEATURE_COLUMNS = {
    "f26_balance_drop_frequency",
    "f27_deposits_after_below2_per_day",
    "f28_median_seconds_below2_to_deposit",
    "f44_morning_stakes_percentage",
    "f45_evening_stakes_percentage",
    "f48_percentage_bets_with_cashout",
    "f51_median_seconds_loss_to_next_bet",
}


def _safe_merge(left: Optional[pd.DataFrame], right: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if left is None:
        out = None if right is None else right.copy()
    elif right is None:
        out = left.copy()
    else:
        L = left.copy()
        R = right.copy()

        if "Player_Profile_ID" not in L.columns:
            raise ValueError(
                f"❌ MERGE ERROR: Left DataFrame mist 'Player_Profile_ID' kolom!\n"
                f"   Left kolommen: {list(L.columns)}\n"
                f"   Left shape: {L.shape}\n"
                f"   Dit kan betekenen dat een feature functie een lege DataFrame heeft geretourneerd.\n"
                f"   Check de logs voor de feature die net is afgelopen."
            )
        if "Player_Profile_ID" not in R.columns:
            raise ValueError(
                f"❌ MERGE ERROR: Right DataFrame mist 'Player_Profile_ID' kolom!\n"
                f"   Right kolommen: {list(R.columns)}\n"
                f"   Right shape: {R.shape}\n"
                f"   Dit kan betekenen dat een feature functie een lege DataFrame heeft geretourneerd.\n"
                f"   Check de logs voor de feature die net is afgelopen."
            )

        L["Player_Profile_ID"] = L["Player_Profile_ID"].astype(str)
        R["Player_Profile_ID"] = R["Player_Profile_ID"].astype(str)

        try:
            out = L.merge(R, on="Player_Profile_ID", how="outer")
        except Exception as e:
            raise ValueError(
                f"❌ MERGE ERROR tijdens merge operatie:\n"
                f"   Error: {e}\n"
                f"   Left shape: {L.shape}, kolommen: {list(L.columns)}\n"
                f"   Right shape: {R.shape}, kolommen: {list(R.columns)}\n"
            ) from e

    if out is None or out.empty:
        return out
    for col in out.columns:
        if (
            col != "Player_Profile_ID"
            and col not in _UNKNOWN_FEATURE_COLUMNS
            and pd.api.types.is_numeric_dtype(out[col])
        ):
            out[col] = out[col].fillna(0)
    return out


# -----------------------------------------------------------------------------
# Feature runner  (VERBATIM uit clean_and_parse.py)
# -----------------------------------------------------------------------------
def run_scenario(
    scenario_name: str,
    base_dir: Path,
    glob_pattern: str = "*.csv",
    verbose: bool = False,
    chunksize: int = 200_000,
    x_tijdspad: Optional[str] = None,
    y_tijdspad: Optional[str] = None,
    feature_override: Optional[str] = None,
    ignore_EOD_Balance: bool = False,
) -> pd.DataFrame:
    """
    Voert de feature engineering uit voor het opgegeven scenario.
    Verbatim overgenomen uit clean_and_parse.py zodat het resultaat identiek is.

    ignore_EOD_Balance: als de dataset geen `Player_Profile_EOD_Balance` heeft, krijgen
    de features die die kolom lezen (f26/f27/f28) `WOK_Player_Profile` niet aangereikt →
    de saldoafhankelijke features blijven dan onbekend i.p.v. te crashen.
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Onbekend scenario ‘{scenario_name}’. Kies uit {list(SCENARIOS)}")

    feature_keys = SCENARIOS[scenario_name]["features"]

    if feature_override:
        if feature_override not in feature_keys:
            raise ValueError(
                f"Feature ‘{feature_override}’ zit niet in scenario ‘{scenario_name}’. "
                f"Kies uit: {feature_keys}"
            )
        feature_keys = [feature_override]

    # 1) Verzamel vereiste tabellen/kolommen
    required_tables: List[str] = []
    usecols_by_table: Dict[str, List[str]] = {}
    for key in feature_keys:
        print(f"🔍 Analyseren feature '{key}' voor benodigde tabellen/kolommen...")
        spec = FEATURES_REGISTRY[key]
        for t in spec["tables"]:
            if t not in required_tables:
                required_tables.append(t)
        for t, cols in spec["usecols"].items():
            keep = usecols_by_table.setdefault(t, [])
            for c in cols:
                if c not in keep:
                    keep.append(c)

    print("🔩 Benodigde tabellen:", required_tables)
    print("🔩 Kolommen per tabel:", usecols_by_table)

    # 2) Haal paden op (geen DataFrames; streamende functies lezen zelf)
    try:
        tables, game_session_missing, bet_missing, comlaints_missing = load_tables_local(
            base_dir,
            logical_tables=required_tables,
            usecols_by_table=usecols_by_table,
            glob_pattern=glob_pattern,
            recursive=True,
            verbose=verbose,
        )
    except FileNotFoundError as e:
        _optional = {"WOK_Bet", "WOK_Game_Session", "WOK_Complaint"}
        if set(required_tables).issubset(_optional):
            print(f"⚠️ Alle benodigde tabellen zijn optioneel en ontbreken — features worden overgeslagen.")
            tables = {}
            game_session_missing = "WOK_Game_Session" in required_tables
            bet_missing = "WOK_Bet" in required_tables
            comlaints_missing = "WOK_Complaint" in required_tables
        else:
            raise RuntimeError(f"load_tables_local mislukt: {e}")
    except Exception as e:
        raise RuntimeError(f"load_tables_local mislukt: {e}")

    # 3) Features berekenen en samenvoegen
    features_df: Optional[pd.DataFrame] = None
    logs_dir = Path(base_dir) / "logs"
    logs_dir.mkdir(exist_ok=True)
    # Run-scoped cache: share one reconstruction across F26/F27/F28 for each window.
    opening_balance_cache = {}
    stake_share_cache = {}

    for key in feature_keys:
        print(f"\n🔨 Verwerken feature: {key}")
        huidige_tijd = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"   🕒 Tijdstip: {huidige_tijd}")
        spec = FEATURES_REGISTRY[key]
        if bet_missing and "WOK_Bet" in spec["tables"] and "WOK_Bet" not in spec.get("optional_tables", []):
            print(f"⏭️ Feature '{key}' wordt overgeslagen (WOK_Bet ontbreekt).")
            continue
        if game_session_missing and "WOK_Game_Session" in spec["tables"] and "WOK_Game_Session" not in spec.get("optional_tables", []):
            print(f"⏭️ Feature '{key}' wordt overgeslagen (WOK_Game_Session ontbreekt).")
            continue
        if comlaints_missing and "WOK_Complaint" in spec["tables"]:
            print(f"⏭️ Feature '{key}' wordt overgeslagen (WOK_Complaint ontbreekt).")
            continue
        stream_fn = spec.get("stream_fn")
        if not stream_fn:
            print(f"⏭️ Feature '{key}' wordt overgeslagen (geen streamingfunctie).")
            continue

        log_path = logs_dir / spec.get("log_name", f"feature_{key}.log")
        if verbose:
            print(f"\n⚙️ Start feature '{key}' via {stream_fn.__name__}")
            print(f"   → logbestand: {log_path.name}")

        kwargs = dict(spec.get("kwargs", {}))

        sig = inspect.signature(stream_fn)

        if x_tijdspad and "x_tijdspad" in sig.parameters:
            kwargs["x_tijdspad"] = x_tijdspad.split(":")
            print(f"   🔧 Override x_tijdspad vanuit CLI: {kwargs['x_tijdspad']}")
        if y_tijdspad and "y_tijdspad" in sig.parameters:
            kwargs["y_tijdspad"] = y_tijdspad.split(":")
            print(f"   🔧 Override y_tijdspad vanuit CLI: {kwargs['y_tijdspad']}")

        # ignore_EOD_Balance: features die Player_Profile_EOD_Balance lezen (f26/f27/f28)
        # krijgen WOK_Player_Profile niet aangereikt → onbekende saldofeatures
        # i.p.v. een crash op een dataset zonder die kolom.
        feat_tables = tables
        if ignore_EOD_Balance and "Player_Profile_EOD_Balance" in spec.get("usecols", {}).get("WOK_Player_Profile", []):
            feat_tables = {k: v for k, v in tables.items() if k != "WOK_Player_Profile"}
            print(f"   ⏭️  ignore_EOD_Balance: WOK_Player_Profile niet aangereikt aan '{key}' (saldo onbekend).")

        if "start_balances" in sig.parameters:
            window = kwargs.get("x_tijdspad")
            cache_key = (tuple(window) if window else None, ignore_EOD_Balance)
            if cache_key not in opening_balance_cache:
                opening_balance_cache[cache_key] = reconstruct_start_balances(
                    feat_tables, x_tijdspad=window, chunksize=chunksize, verbose=verbose,
                )
            kwargs["start_balances"] = opening_balance_cache[cache_key]

        if "stake_shares" in sig.parameters:
            window = kwargs.get("x_tijdspad")
            cache_key = tuple(window) if window else None
            if cache_key not in stake_share_cache:
                stake_share_cache[cache_key] = net_stake_time_shares(
                    feat_tables, x_tijdspad=window, chunksize=chunksize, verbose=verbose,
                )
            kwargs["stake_shares"] = stake_share_cache[cache_key]

        f = stream_fn(
            tables=feat_tables,
            chunksize=chunksize,
            log_path=log_path,
            verbose=verbose,
            **kwargs,
        )
        huidige_tijd = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"   🕒 Feature '{key}' voltooid om {huidige_tijd}")

        if feature_override and key in f.columns:
            f = f[["Player_Profile_ID", key]].copy()

        print(f"   → resultaatvorm: {f.shape}, kolommen: {list(f.columns)}")

        features_df = _safe_merge(features_df, f)

    if features_df is None:
        features_df = pd.DataFrame(columns=["Player_Profile_ID"])

    if verbose:
        print("\n✅ Scenario afgerond. Eindvorm features:", features_df.shape)
    return features_df


# -----------------------------------------------------------------------------
# Hoofd-API: bouw features voor één geschoonde map
# (equivalent van de parse-branch van clean_and_parse.main())
# -----------------------------------------------------------------------------
def build_features(
    cleaned_dir: str | Path,
    scenario: str,
    *,
    features_out: Optional[str | Path] = None,
    x_tijdspad: Optional[str] = None,
    y_tijdspad: Optional[str] = None,
    feature: Optional[str] = None,
    chunksize: int = 400_000,
    verbose: bool = False,
    ignore_EOD_Balance: bool = False,
) -> pd.DataFrame:
    """
    Bouw de features voor `scenario` op basis van de geschoonde CSV's in `cleaned_dir`.

    ignore_EOD_Balance: zet aan als je dataset geen `Player_Profile_EOD_Balance` heeft —
    f26/f27/f28 blijven dan onbekend i.p.v. te crashen (zie run_scenario).

    Spiegelt exact de parse-branch van `clean_and_parse.py --mode parse`: feature-
    berekening via `run_scenario`, dedup op `Player_Profile_ID`, optionele ALL-aggregatie,
    en wegschrijven naar `features_out` (indien opgegeven).
    """
    cleaned_src = Path(cleaned_dir).resolve()
    if not cleaned_src.exists():
        raise FileNotFoundError(f"CLEANED map niet gevonden: {cleaned_src}")

    if verbose:
        print(f"\n📥 Features inlezen uit: {cleaned_src}")
        print("    Glob voor features-inlees: *.csv")

    features = run_scenario(
        scenario,
        base_dir=cleaned_src,
        glob_pattern="*.csv",
        verbose=verbose,
        chunksize=chunksize,
        x_tijdspad=x_tijdspad,
        y_tijdspad=y_tijdspad,
        feature_override=feature,
        ignore_EOD_Balance=ignore_EOD_Balance,
    )

    # Dedup op Player_Profile_ID
    id_col = "Player_Profile_ID"
    if id_col in features.columns:
        before = len(features)
        features = features.drop_duplicates(subset=[id_col], keep="first")
        dupes = before - len(features)
        if dupes > 0:
            print(f"⚠️  {dupes} duplicate {id_col} rijen verwijderd ({before} → {len(features)})")

    # For the ALL scenario: aggregate per-player rows to 1-row operator summary
    if scenario == "ALL" and id_col in features.columns:
        numeric_cols = [c for c in features.columns if c != id_col and features[c].dtype.kind in "iufcb"]
        agg = {}
        for col in numeric_cols:
            vals = features[col].dropna()
            agg[f"mean_{col}"] = float(vals.mean()) if len(vals) > 0 else float("nan")
            agg[f"std_{col}"]  = float(vals.std())  if len(vals) > 0 else float("nan")
            agg[f"min_{col}"]  = float(vals.min())  if len(vals) > 0 else float("nan")
            agg[f"max_{col}"]  = float(vals.max())  if len(vals) > 0 else float("nan")
        features = pd.DataFrame([agg])

    if features_out:
        out_path = Path(features_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(out_path, index=False)
        if verbose:
            print(f"\n💾 Features weggeschreven naar: {out_path}")

    print("\n✅ Klaar. Eindvorm features:", features.shape)
    return features
