"""
Feature Engineering Basis (streaming)
======================================

Doel
----
Dit bestand definieert de *basis tijdgebaseerde features* die we uit de WOK-tabellen
berekenen voor het scenario `basic_time_based_variables`. Elke feature is een
Python-functie die op een uniforme manier aangeroepen wordt door de runner
(`clean_and_parse.py`).

Uniforme signature
------------------
Elke feature-functie volgt dezelfde call-signature:

    stream_fn(
        tables: Dict[str, List[Path]],
        *,
        chunksize: int,
        log_path: Path | None,
        verbose: bool
    ) -> pd.DataFrame

- `tables`: dictionary met logische tabelnamen als keys en een lijst van CSV-paths
- De functie streamt zelf de bestanden via `iter_csv_chunks`
- Return: DataFrame met per speler één of meer featurekolommen, altijd inclusief:
    ["Player_Profile_ID", "<feature_naam(en)>"]

FEATURES_REGISTRY
-----------------
Onderin dit bestand staat `FEATURES_REGISTRY`: de "bron van waarheid".
Daar staat per feature:
- `stream_fn`: de Python-functie
- `tables`: welke tabellen nodig zijn
- `usecols`: welke kolommen minimaal geladen moeten worden
- `log_name`: naam van het logbestand
- `kwargs`: extra parameters voor de functie

De runner gebruikt alléén deze registry. Hij weet niets van de inhoud van de features.

Groep-aanpak
------------
Voor efficiency zijn gerelateerde features gegroepeerd:
- Groep 1 (var1a,b,c): Transactiebedragen
- Groep 2 (var2a,b): Transactieaantallen
- Groep 3 (var3a,b,c): Actieve dagen
- Groep 4 (var4a-f): Transactietypes — anchor var4a retourneert alle 6 kolommen,
  var4b-var4f zijn no-ops (retourneren lege DataFrame) om dubbel lezen te vermijden
- Groep 5 (var5a-e): Transactie-instrumenten — zelfde patroon als groep 4
- Groep 6 (var6a-c): Dagen met veel transacties — zelfde patroon als groep 4
- Groep 7 (var7a-d): Limieten — zelfde patroon als groep 4
- var11: Klachten
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import logging
import pandas as pd
import numpy as np
from path_finding import iter_csv_chunks
from local_time import local_time, local_day_labels
from reading_difficult_json import (
    iter_limit_values,
    _safe_load_json_relaxed,
    simple_Player_Profile_Bank_Account_json_iterator,
    simple_RG_Class_Value_from_FLAG_RG_CLASS_json_iterator,
    iter_transaction_ids_from_Game_Transactions,
    iter_player_profile_ids_from_Bet_Transactions,
    get_list_of_response_ids_from_Responses_list,
)


def _setup_feature_logger(log_path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(f"feature:{name}")
    for _h in logger.handlers[:]:   # sluit oude handler écht (anders fd-lek bij hergebruik per featurenaam)
        _h.close()
        logger.removeHandler(_h)
    logger.setLevel(logging.INFO)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def parse_ddmmyyyy_to_timestamp(date_str: str) -> pd.Timestamp:
    """
    Converteer 'DDMMYYYY' string naar pd.Timestamp.

    Voorbeeld: '01012025' -> pd.Timestamp('2025-01-01')
    """
    day = date_str[:2]
    month = date_str[2:4]
    year = date_str[4:]
    return pd.Timestamp(f"{year}-{month}-{day}")


# ============================================================================
# HELPER: No-op stream functie voor sub-features van gecombineerde groepen
# ============================================================================

def _noop_stream_fn(
    tables: Dict[str, List[Path]],
    *,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Retourneert een lege DataFrame met alleen Player_Profile_ID.
    Gebruikt voor sub-features van groepen waarbij de anchor al alle kolommen berekent.
    Bij _safe_merge met een lege DF blijven alle bestaande kolommen intact.
    """
    return pd.DataFrame(columns=["Player_Profile_ID"])


# ============================================================================
# GROEP 1: Transactiebedragen (WOK_Player_Account_Transaction)
# var1a: totaal ingezet bedrag (som abs STAKE)
# var1b: variantie ingezet bedrag per dag
# var1c: totaal gewonnen bedrag (som WINNING)
# ============================================================================

def var1a_totaal_ingezet_bedrag(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var1a: Totaal ingezet bedrag

    Som van de absolute waarde van alle STAKE-transacties per speler.
    Retourneert een positief getal (totaal ingezet).

    Output:
        - var1a_totaal_ingezet_bedrag: Float >= 0
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var1a_totaal_ingezet_bedrag")
        logger.info("▶ START Var1a: Totaal ingezet bedrag")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    ingezet_per_speler: Dict[str, float] = {}

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "var1a_totaal_ingezet_bedrag"])

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Amount", "Transaction_Datetime", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()

        df = df[(df["Transaction_Type"] == "STAKE") & (df["Transaction_Status"] == "SUCCESSFUL")]
        if df.empty:
            continue

        df["amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce").abs()
        df = df[df["amount"].notna()]

        sums = df.groupby("Player_Profile_ID")["amount"].sum()
        for pid, v in sums.items():
            ingezet_per_speler[pid] = ingezet_per_speler.get(pid, 0.0) + float(v)

    records = [{"Player_Profile_ID": pid, "var1a_totaal_ingezet_bedrag": v}
               for pid, v in ingezet_per_speler.items()]
    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "var1a_totaal_ingezet_bedrag"])

    if logger:
        logger.info(f"✅ Var1a klaar: {len(result):,} spelers")
    return result


def var1b_variantie_ingezet_bedrag_per_dag(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var1b: Variantie ingezet bedrag per dag

    Bereken per speler de variantie van het dagelijks ingezet bedrag (STAKE).
    Eerst som van abs(STAKE) per dag, dan variantie over de dagelijkse sommen.

    Output:
        - var1b_variantie_ingezet_bedrag_per_dag: Float >= 0
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var1b_variantie_ingezet_bedrag_per_dag")
        logger.info("▶ START Var1b: Variantie ingezet bedrag per dag")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    # {pid: {date: som_ingezet}}
    ingezet_per_dag: Dict[str, Dict[str, float]] = {}

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "var1b_variantie_ingezet_bedrag_per_dag"])

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Amount", "Transaction_Datetime", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)

        if start_datum is not None:
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()
            ts = ts.loc[mask_periode]

        df = df[(df["Transaction_Type"] == "STAKE") & (df["Transaction_Status"] == "SUCCESSFUL")]
        if df.empty:
            continue

        df["date"] = local_day_labels(ts.loc[df.index])
        df["amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce").abs()
        df = df[df["amount"].notna() & df["date"].notna()]

        dag_sommen = df.groupby(["Player_Profile_ID", "date"])["amount"].sum()
        for (pid, date), som in dag_sommen.items():
            date_str = str(date)
            if pid not in ingezet_per_dag:
                ingezet_per_dag[pid] = {}
            ingezet_per_dag[pid][date_str] = ingezet_per_dag[pid].get(date_str, 0.0) + float(som)

    records = []
    for pid, dag_dict in ingezet_per_dag.items():
        waarden = list(dag_dict.values())
        variantie = float(np.var(waarden)) if len(waarden) > 1 else 0.0
        records.append({"Player_Profile_ID": pid, "var1b_variantie_ingezet_bedrag_per_dag": variantie})

    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "var1b_variantie_ingezet_bedrag_per_dag"])

    if logger:
        logger.info(f"✅ Var1b klaar: {len(result):,} spelers")
    return result


def var1c_totaal_gewonnen_bedrag(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var1c: Totaal gewonnen bedrag

    Som van alle WINNING-transacties per speler (positieve bedragen).

    Output:
        - var1c_totaal_gewonnen_bedrag: Float >= 0
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var1c_totaal_gewonnen_bedrag")
        logger.info("▶ START Var1c: Totaal gewonnen bedrag")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    gewonnen_per_speler: Dict[str, float] = {}

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "var1c_totaal_gewonnen_bedrag"])

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Amount", "Transaction_Datetime", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()

        df = df[(df["Transaction_Type"] == "WINNING") & (df["Transaction_Status"] == "SUCCESSFUL")]
        if df.empty:
            continue

        df["amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df = df[df["amount"].notna()]

        sums = df.groupby("Player_Profile_ID")["amount"].sum()
        for pid, v in sums.items():
            gewonnen_per_speler[pid] = gewonnen_per_speler.get(pid, 0.0) + float(v)

    records = [{"Player_Profile_ID": pid, "var1c_totaal_gewonnen_bedrag": v}
               for pid, v in gewonnen_per_speler.items()]
    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "var1c_totaal_gewonnen_bedrag"])

    if logger:
        logger.info(f"✅ Var1c klaar: {len(result):,} spelers")
    return result


# ============================================================================
# GROEP 2: Transactieaantallen (WOK_Player_Account_Transaction)
# var2a: totaal aantal transacties
# var2b: variantie aantal transacties per dag
# ============================================================================

def var2a_totaal_aantal_transacties(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var2a: Totaal aantal transacties

    Tel alle transacties per speler (Transaction_ID, geen unique check vereist).

    Output:
        - var2a_totaal_aantal_transacties: Int >= 0
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var2a_totaal_aantal_transacties")
        logger.info("▶ START Var2a: Totaal aantal transacties")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    counts_per_speler: Dict[str, int] = {}

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "var2a_totaal_aantal_transacties"])

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()

        counts = df.groupby("Player_Profile_ID").size()
        for pid, count in counts.items():
            counts_per_speler[pid] = counts_per_speler.get(pid, 0) + int(count)

    records = [{"Player_Profile_ID": pid, "var2a_totaal_aantal_transacties": v}
               for pid, v in counts_per_speler.items()]
    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "var2a_totaal_aantal_transacties"])

    if logger:
        logger.info(f"✅ Var2a klaar: {len(result):,} spelers")
    return result


def var2b_variantie_aantal_transacties_per_dag(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var2b: Variantie aantal transacties per dag

    Per speler: tel transacties per dag, bereken variantie over de dagelijkse tellingen.

    Output:
        - var2b_variantie_aantal_transacties_per_dag: Float >= 0
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var2b_variantie_aantal_transacties_per_dag")
        logger.info("▶ START Var2b: Variantie aantal transacties per dag")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    # {pid: {date_str: count}}
    counts_per_dag: Dict[str, Dict[str, int]] = {}

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "var2b_variantie_aantal_transacties_per_dag"])

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)

        if start_datum is not None:
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()
            ts = ts.loc[mask_periode]

        df["date"] = local_day_labels(ts.loc[df.index])
        df = df[df["date"].notna()]

        dag_counts = df.groupby(["Player_Profile_ID", "date"]).size()
        for (pid, date), count in dag_counts.items():
            date_str = str(date)
            if pid not in counts_per_dag:
                counts_per_dag[pid] = {}
            counts_per_dag[pid][date_str] = counts_per_dag[pid].get(date_str, 0) + int(count)

    records = []
    for pid, dag_dict in counts_per_dag.items():
        waarden = list(dag_dict.values())
        variantie = float(np.var(waarden)) if len(waarden) > 1 else 0.0
        records.append({"Player_Profile_ID": pid, "var2b_variantie_aantal_transacties_per_dag": variantie})

    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "var2b_variantie_aantal_transacties_per_dag"])

    if logger:
        logger.info(f"✅ Var2b klaar: {len(result):,} spelers")
    return result


# ============================================================================
# GROEP 3: Actieve dagen (WOK_Player_Account_Transaction)
# var3a: totaal aantal actieve dagen
# var3b: variantie aantal actieve dagen per week
# var3c: percentage actieve dagen in periode
# ============================================================================

def var3a_totaal_aantal_actieve_dagen(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var3a: Totaal aantal actieve dagen

    Tel het aantal unieke kalenderdagen waarop de speler een transactie had.

    Output:
        - var3a_totaal_aantal_actieve_dagen: Int >= 0
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var3a_totaal_aantal_actieve_dagen")
        logger.info("▶ START Var3a: Totaal aantal actieve dagen")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    # {pid: set of date strings}
    actieve_dagen: Dict[str, set] = {}

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "var3a_totaal_aantal_actieve_dagen"])

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)

        if start_datum is not None:
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()
            ts = ts.loc[mask_periode]

        df["date"] = local_day_labels(ts.loc[df.index])
        df = df[df["date"].notna()]

        for pid, grp in df.groupby("Player_Profile_ID"):
            if pid not in actieve_dagen:
                actieve_dagen[pid] = set()
            actieve_dagen[pid].update(str(d) for d in grp["date"])

    records = [{"Player_Profile_ID": pid, "var3a_totaal_aantal_actieve_dagen": len(dagen)}
               for pid, dagen in actieve_dagen.items()]
    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "var3a_totaal_aantal_actieve_dagen"])

    if logger:
        logger.info(f"✅ Var3a klaar: {len(result):,} spelers")
    return result


def var3b_variantie_aantal_actieve_dagen_per_week(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var3b: Variantie aantal actieve dagen per week

    Per speler: tel het aantal unieke actieve dagen per ISO-week, dan bereken
    de variantie van het aantal actieve dagen per week over alle weken.

    Output:
        - var3b_variantie_aantal_actieve_dagen_per_week: Float >= 0
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var3b_variantie_aantal_actieve_dagen_per_week")
        logger.info("▶ START Var3b: Variantie aantal actieve dagen per week")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    # {pid: {week_str: set of dates}}
    actieve_dagen_per_week: Dict[str, Dict[str, set]] = {}

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "var3b_variantie_aantal_actieve_dagen_per_week"])

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)

        if start_datum is not None:
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()
            ts = ts.loc[mask_periode]

        ts_clean = ts.loc[df.index]
        df["date"] = local_day_labels(ts_clean)
        df["week"] = df["date"].dt.to_period("W").astype(str)
        df = df[df["date"].notna()]

        # Vectorized: unieke datums per (speler, week)
        grp_data = df.groupby(["Player_Profile_ID", "week"])["date"].agg(set)
        for (pid, week), dates_set in grp_data.items():
            if pid not in actieve_dagen_per_week:
                actieve_dagen_per_week[pid] = {}
            if week not in actieve_dagen_per_week[pid]:
                actieve_dagen_per_week[pid][week] = set()
            actieve_dagen_per_week[pid][week].update(str(d) for d in dates_set)

    records = []
    for pid, week_dict in actieve_dagen_per_week.items():
        actieve_per_week = [len(dates) for dates in week_dict.values()]
        variantie = float(np.var(actieve_per_week)) if len(actieve_per_week) > 1 else 0.0
        records.append({"Player_Profile_ID": pid, "var3b_variantie_aantal_actieve_dagen_per_week": variantie})

    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "var3b_variantie_aantal_actieve_dagen_per_week"])

    if logger:
        logger.info(f"✅ Var3b klaar: {len(result):,} spelers")
    return result


def var3c_percentage_actieve_dagen_in_periode(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var3c: Percentage actieve dagen in periode

    Fractie = aantal actieve dagen / (max_datum - min_datum + 1).
    De periode is de span van de eigen transacties (of x_tijdspad als gegeven).

    Output:
        - var3c_percentage_actieve_dagen_in_periode: Float [0, 1]
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var3c_percentage_actieve_dagen_in_periode")
        logger.info("▶ START Var3c: Percentage actieve dagen in periode")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    # {pid: set of date strings}
    actieve_dagen: Dict[str, set] = {}

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "var3c_percentage_actieve_dagen_in_periode"])

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)

        if start_datum is not None:
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()
            ts = ts.loc[mask_periode]

        df["date"] = local_day_labels(ts.loc[df.index])
        df = df[df["date"].notna()]

        for pid, grp in df.groupby("Player_Profile_ID"):
            if pid not in actieve_dagen:
                actieve_dagen[pid] = set()
            actieve_dagen[pid].update(str(d) for d in grp["date"])

    records = []
    for pid, dagen_set in actieve_dagen.items():
        n_actief = len(dagen_set)
        if n_actief == 0:
            pct = 0.0
        elif start_datum is not None:
            # Gebruik de expliciete periode als die gegeven is
            span_dagen = (eind_datum - start_datum).days
            pct = n_actief / span_dagen if span_dagen > 0 else 1.0
        else:
            # Gebruik de span van eigen transacties
            dates = sorted(pd.Timestamp(d) for d in dagen_set)
            span_dagen = (dates[-1] - dates[0]).days + 1
            pct = n_actief / span_dagen if span_dagen > 0 else 1.0
        records.append({"Player_Profile_ID": pid, "var3c_percentage_actieve_dagen_in_periode": pct})

    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "var3c_percentage_actieve_dagen_in_periode"])

    if logger:
        logger.info(f"✅ Var3c klaar: {len(result):,} spelers")
    return result


# ============================================================================
# GROEP 4: Transactietypes (WOK_Player_Account_Transaction)
# var4a: DEPOSIT (SUCCESSFUL)
# var4b: WITHDRAWAL (SUCCESSFUL)
# var4c: STAKE (SUCCESSFUL)
# var4d: WINNING (SUCCESSFUL)
# var4e: OTHER (SUCCESSFUL)
# var4f: Transaction_Status = BONUS
# → anchor: var4a retourneert alle 6 kolommen; var4b-var4f zijn no-ops
# ============================================================================

def var4a_tot_var4f(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var4a-4f: Transactietype-tellingen (gecombineerde anchor functie)

    Berekent in één doorloop:
    - var4a: aantal DEPOSIT (SUCCESSFUL)
    - var4b: aantal WITHDRAWAL (SUCCESSFUL)
    - var4c: aantal STAKE (SUCCESSFUL)
    - var4d: aantal WINNING (SUCCESSFUL)
    - var4e: aantal OTHER (SUCCESSFUL)
    - var4f: aantal transacties met Transaction_Status = BONUS

    Output: DataFrame met alle 6 kolommen + Player_Profile_ID
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var4a_transaction_types")
        logger.info("▶ START Var4a-4f: Transactietype-tellingen")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    # {pid: {var: count}}
    counts: Dict[str, Dict[str, int]] = {}

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        cols = ["Player_Profile_ID", "var4a_totaal_aantal_deposit_transacties",
                "var4b_totaal_aantal_withdrawal_transacties", "var4c_totaal_aantal_stake_transacties",
                "var4d_totaal_aantal_winning_transacties", "var4e_totaal_aantal_other_transacties",
                "var4f_totaal_aantal_bonus_transacties"]
        return pd.DataFrame(columns=cols)

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()

        # Vectorized: tellingen per (speler, type) voor SUCCESSFUL transacties
        type_mapping = {"DEPOSIT": "4a", "WITHDRAWAL": "4b", "STAKE": "4c",
                        "WINNING": "4d", "OTHER": "4e"}
        successful = df[df["Transaction_Status"] == "SUCCESSFUL"]
        if not successful.empty:
            chunk_type_counts = successful.groupby(
                ["Player_Profile_ID", "Transaction_Type"]).size()
            for (pid, tx_type), count in chunk_type_counts.items():
                if tx_type not in type_mapping:
                    continue
                if pid not in counts:
                    counts[pid] = {"4a": 0, "4b": 0, "4c": 0, "4d": 0, "4e": 0, "4f": 0}
                counts[pid][type_mapping[tx_type]] += int(count)

        # Bonus: filter op Transaction_Status = BONUS
        bonus = df[df["Transaction_Status"] == "BONUS"]
        if not bonus.empty:
            chunk_bonus = bonus.groupby("Player_Profile_ID").size()
            for pid, count in chunk_bonus.items():
                if pid not in counts:
                    counts[pid] = {"4a": 0, "4b": 0, "4c": 0, "4d": 0, "4e": 0, "4f": 0}
                counts[pid]["4f"] += int(count)

    records = [
        {
            "Player_Profile_ID": pid,
            "var4a_totaal_aantal_deposit_transacties": c["4a"],
            "var4b_totaal_aantal_withdrawal_transacties": c["4b"],
            "var4c_totaal_aantal_stake_transacties": c["4c"],
            "var4d_totaal_aantal_winning_transacties": c["4d"],
            "var4e_totaal_aantal_other_transacties": c["4e"],
            "var4f_totaal_aantal_bonus_transacties": c["4f"],
        }
        for pid, c in counts.items()
    ]
    cols = ["Player_Profile_ID", "var4a_totaal_aantal_deposit_transacties",
            "var4b_totaal_aantal_withdrawal_transacties", "var4c_totaal_aantal_stake_transacties",
            "var4d_totaal_aantal_winning_transacties", "var4e_totaal_aantal_other_transacties",
            "var4f_totaal_aantal_bonus_transacties"]
    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=cols)

    if logger:
        logger.info(f"✅ Var4a-4f klaar: {len(result):,} spelers")
    return result


# ============================================================================
# GROEP 5: Transactie-instrumenten (WOK_Player_Account_Transaction)
# var5a: aantal unieke instrumenten per speler
# var5b: CREDIT_CARD
# var5c: ELECTRONIC_MONEY
# var5d: BANK_TRANSFER
# var5e: OTHER (instrument)
# → anchor: var5a retourneert alle 5 kolommen; var5b-var5e zijn no-ops
# ============================================================================

def var5a_tot_var5e(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var5a-5e: Transactie-instrument tellingen (gecombineerde anchor functie)

    Berekent in één doorloop:
    - var5a: aantal unieke Transaction_Deposit_Instrument waarden per speler
    - var5b: aantal CREDIT_CARD transacties
    - var5c: aantal ELECTRONIC_MONEY transacties
    - var5d: aantal BANK_TRANSFER transacties
    - var5e: aantal OTHER instrument transacties

    Output: DataFrame met alle 5 kolommen + Player_Profile_ID
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var5a_transaction_instruments")
        logger.info("▶ START Var5a-5e: Transactie-instrument tellingen")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    # {pid: {instrument: count}}
    instrument_counts: Dict[str, Dict[str, int]] = {}
    # {pid: set of unique instruments}
    unique_instruments: Dict[str, set] = {}

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        cols = ["Player_Profile_ID", "var5a_totaal_aantal_transaction_instruments",
                "var5b_totaal_aantal_CREDIT_CARD_instrument", "var5c_totaal_aantal_ELECTRONIC_MONEY_instrument",
                "var5d_totaal_aantal_BANK_TRANSFER_instrument", "var5e_totaal_aantal_OTHER_instrument"]
        return pd.DataFrame(columns=cols)

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Deposit_Instrument"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()

        # Normaliseer instrument kolom (leeg = overslaan)
        df["instr"] = df["Transaction_Deposit_Instrument"].fillna("").astype(str).str.strip().str.upper()
        df_instr = df[df["instr"] != ""][["Player_Profile_ID", "instr"]]
        if df_instr.empty:
            continue

        # Vectorized: unieke instrumenten per speler
        unique_grp = df_instr.groupby("Player_Profile_ID")["instr"].agg(set)
        for pid, instr_set in unique_grp.items():
            unique_instruments.setdefault(pid, set()).update(instr_set)
            if pid not in instrument_counts:
                instrument_counts[pid] = {"CREDIT_CARD": 0, "ELECTRONIC_MONEY": 0,
                                          "BANK_TRANSFER": 0, "OTHER": 0}

        # Vectorized: tellingen per type
        known = {"CREDIT_CARD", "ELECTRONIC_MONEY", "BANK_TRANSFER"}
        for instr_type in known:
            chunk_counts = df_instr[df_instr["instr"] == instr_type].groupby("Player_Profile_ID").size()
            for pid, count in chunk_counts.items():
                instrument_counts.setdefault(pid, {"CREDIT_CARD": 0, "ELECTRONIC_MONEY": 0,
                                                    "BANK_TRANSFER": 0, "OTHER": 0})
                instrument_counts[pid][instr_type] += int(count)
        # OTHER = alles wat niet een bekende type is
        other_chunk = df_instr[~df_instr["instr"].isin(known)].groupby("Player_Profile_ID").size()
        for pid, count in other_chunk.items():
            instrument_counts.setdefault(pid, {"CREDIT_CARD": 0, "ELECTRONIC_MONEY": 0,
                                               "BANK_TRANSFER": 0, "OTHER": 0})
            instrument_counts[pid]["OTHER"] += int(count)

    records = [
        {
            "Player_Profile_ID": pid,
            "var5a_totaal_aantal_transaction_instruments": len(unique_instruments.get(pid, set())),
            "var5b_totaal_aantal_CREDIT_CARD_instrument": c["CREDIT_CARD"],
            "var5c_totaal_aantal_ELECTRONIC_MONEY_instrument": c["ELECTRONIC_MONEY"],
            "var5d_totaal_aantal_BANK_TRANSFER_instrument": c["BANK_TRANSFER"],
            "var5e_totaal_aantal_OTHER_instrument": c["OTHER"],
        }
        for pid, c in instrument_counts.items()
    ]
    cols = ["Player_Profile_ID", "var5a_totaal_aantal_transaction_instruments",
            "var5b_totaal_aantal_CREDIT_CARD_instrument", "var5c_totaal_aantal_ELECTRONIC_MONEY_instrument",
            "var5d_totaal_aantal_BANK_TRANSFER_instrument", "var5e_totaal_aantal_OTHER_instrument"]
    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=cols)

    if logger:
        logger.info(f"✅ Var5a-5e klaar: {len(result):,} spelers")
    return result


# ============================================================================
# GROEP 6: Dagen met veel transacties (WOK_Player_Account_Transaction)
# var6a: dagen met >= 5 transacties
# var6b: dagen met >= 25 transacties
# var6c: dagen met >= 100 transacties
# → anchor: var6a retourneert alle 3 kolommen; var6b, var6c zijn no-ops
# ============================================================================

def var6a_tot_var6c(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var6a-6c: Dagen met veel transacties (gecombineerde anchor functie)

    Berekent in één doorloop:
    - var6a: aantal dagen met >= 5 transacties
    - var6b: aantal dagen met >= 25 transacties
    - var6c: aantal dagen met >= 100 transacties

    Output: DataFrame met alle 3 kolommen + Player_Profile_ID
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var6a_hoge_activiteit_dagen")
        logger.info("▶ START Var6a-6c: Dagen met veel transacties")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    # {pid: {date_str: count}}
    dag_counts: Dict[str, Dict[str, int]] = {}

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        cols = ["Player_Profile_ID",
                "var6a_totaal_aantal_dagen_met_5_tranacties_of_meer",
                "var6b_totaal_aantal_dagen_met_25_tranacties_of_meer",
                "var6c_totaal_aantal_dagen_met_100_tranacties_of_meer"]
        return pd.DataFrame(columns=cols)

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)

        if start_datum is not None:
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()
            ts = ts.loc[mask_periode]

        df["date"] = local_day_labels(ts.loc[df.index])
        df = df[df["date"].notna()]

        chunk_dag_counts = df.groupby(["Player_Profile_ID", "date"]).size()
        for (pid, date), count in chunk_dag_counts.items():
            date_str = str(date)
            if pid not in dag_counts:
                dag_counts[pid] = {}
            dag_counts[pid][date_str] = dag_counts[pid].get(date_str, 0) + int(count)

    records = []
    for pid, dag_dict in dag_counts.items():
        counts_lijst = list(dag_dict.values())
        records.append({
            "Player_Profile_ID": pid,
            "var6a_totaal_aantal_dagen_met_5_tranacties_of_meer": sum(1 for c in counts_lijst if c >= 5),
            "var6b_totaal_aantal_dagen_met_25_tranacties_of_meer": sum(1 for c in counts_lijst if c >= 25),
            "var6c_totaal_aantal_dagen_met_100_tranacties_of_meer": sum(1 for c in counts_lijst if c >= 100),
        })

    cols = ["Player_Profile_ID",
            "var6a_totaal_aantal_dagen_met_5_tranacties_of_meer",
            "var6b_totaal_aantal_dagen_met_25_tranacties_of_meer",
            "var6c_totaal_aantal_dagen_met_100_tranacties_of_meer"]
    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=cols)

    if logger:
        logger.info(f"✅ Var6a-6c klaar: {len(result):,} spelers")
    return result


# ============================================================================
# GROEP 7: Limieten (WOK_Player_Limits)
# var7a: totaal aantal limit-entries over alle types
# var7b: totaal aantal deposit limits
# var7c: totaal aantal login limits
# var7d: totaal aantal balance limits
# → anchor: var7a retourneert alle 4 kolommen; var7b-var7d zijn no-ops
# ============================================================================

def var7a_tot_var7d(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var7a-7d: Limiet-tellingen (gecombineerde anchor functie)

    Telt per speler het aantal limit-entries in WOK_Player_Limits:
    - var7a: totaal over alle limiet-types (deposit + login + balance + participation + game_type)
    - var7b: deposit limits (Limit_Deposit)
    - var7c: login limits (Limit_Login)
    - var7d: balance limits (Limit_Balance)

    Filtert optioneel op x_tijdspad via de timestamp in elk limit-record.

    Output: DataFrame met alle 4 kolommen + Player_Profile_ID
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var7a_limieten")
        logger.info("▶ START Var7a-7d: Limiet-tellingen")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]}")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    def _in_period(ts) -> bool:
        if ts is None:
            return start_datum is None  # als geen filter: altijd tellen
        if start_datum is None:
            return True
        try:
            t = pd.Timestamp(ts)
            if t.tzinfo is not None:
                t = t.tz_localize(None)
            return (t >= start_datum) and (t < eind_datum)
        except Exception:
            return False

    limits_paths = tables.get("WOK_Player_Limits")
    if not limits_paths:
        cols = ["Player_Profile_ID", "var7a_totaal_aantal_limits_in_periode",
                "var7b_totaal_aantal_limit_deposits_in_periode",
                "var7c_totaal_aantal_limit_login_in_periode",
                "var7d_totaal_aantal_limit_balance_in_periode"]
        return pd.DataFrame(columns=cols)

    # Bepaal beschikbare kolommen
    desired_cols = ["Limit_Deposit", "Limit_Login", "Limit_Balance", "Limit_Participation", "Limit_Game_Type"]
    available: set = set()
    for p in limits_paths:
        try:
            available.update(pd.read_csv(p, nrows=0).columns.tolist())
        except Exception:
            pass
    usecols = ["Player_Profile_ID"] + [c for c in desired_cols if c in available]

    # {pid: {"dep": int, "login": int, "balance": int, "other": int}}
    counts: Dict[str, Dict[str, int]] = {}

    for df in iter_csv_chunks(
        paths=limits_paths,
        usecols=usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        has_dep  = "Limit_Deposit" in df.columns
        has_log  = "Limit_Login" in df.columns
        has_bal  = "Limit_Balance" in df.columns
        has_part = "Limit_Participation" in df.columns
        has_gt   = "Limit_Game_Type" in df.columns

        col_pid = df.columns.get_loc("Player_Profile_ID")
        idx_dep  = df.columns.get_loc("Limit_Deposit") if has_dep else None
        idx_log  = df.columns.get_loc("Limit_Login") if has_log else None
        idx_bal  = df.columns.get_loc("Limit_Balance") if has_bal else None
        idx_part = df.columns.get_loc("Limit_Participation") if has_part else None
        idx_gt   = df.columns.get_loc("Limit_Game_Type") if has_gt else None

        for i in range(len(df)):
            pid = df.iat[i, col_pid]
            if pd.isna(pid):
                continue
            pid = str(pid)
            if pid not in counts:
                counts[pid] = {"dep": 0, "login": 0, "balance": 0, "other": 0}

            if has_dep and idx_dep is not None:
                for ts, val, _ in iter_limit_values(df.iat[i, idx_dep], type_="deposit"):
                    if _in_period(ts):
                        counts[pid]["dep"] += 1

            if has_log and idx_log is not None:
                for ts, val, _ in iter_limit_values(df.iat[i, idx_log], type_="login"):
                    if _in_period(ts):
                        counts[pid]["login"] += 1

            if has_bal and idx_bal is not None:
                for ts, val, _ in iter_limit_values(df.iat[i, idx_bal], type_="balance"):
                    if _in_period(ts):
                        counts[pid]["balance"] += 1

            if has_part and idx_part is not None:
                for ts, val, _ in iter_limit_values(df.iat[i, idx_part], type_="participation"):
                    if _in_period(ts):
                        counts[pid]["other"] += 1

            if has_gt and idx_gt is not None:
                for ts, val, _ in iter_limit_values(df.iat[i, idx_gt], type_="game_type"):
                    if _in_period(ts):
                        counts[pid]["other"] += 1

    records = [
        {
            "Player_Profile_ID": pid,
            "var7a_totaal_aantal_limits_in_periode": c["dep"] + c["login"] + c["balance"] + c["other"],
            "var7b_totaal_aantal_limit_deposits_in_periode": c["dep"],
            "var7c_totaal_aantal_limit_login_in_periode": c["login"],
            "var7d_totaal_aantal_limit_balance_in_periode": c["balance"],
        }
        for pid, c in counts.items()
    ]
    cols = ["Player_Profile_ID", "var7a_totaal_aantal_limits_in_periode",
            "var7b_totaal_aantal_limit_deposits_in_periode",
            "var7c_totaal_aantal_limit_login_in_periode",
            "var7d_totaal_aantal_limit_balance_in_periode"]
    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=cols)

    if logger:
        logger.info(f"✅ Var7a-7d klaar: {len(result):,} spelers")
    return result


# ============================================================================
# Var11: Klachten (WOK_Complaint)
# var11: totaal aantal klachten per speler
# ============================================================================

def var11_totaal_aantal_complaints_in_periode(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var11: Totaal aantal klachten in periode

    Tel Complaint_ID per speler (via Complaint_Player_ID).
    Filtert optioneel op Complaint_Datetime via x_tijdspad.

    Output:
        - var11_totaal_aantal_complaints_in_periode: Int >= 0
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var11_complaints")
        logger.info("▶ START Var11: Totaal aantal klachten")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    counts_per_speler: Dict[str, int] = {}

    complaint_paths = tables.get("WOK_Complaint")
    if not complaint_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "var11_totaal_aantal_complaints_in_periode"])

    # Sniff beschikbare kolommen (Complaint_Datetime is optioneel)
    available: set = set()
    for p in complaint_paths:
        try:
            available.update(pd.read_csv(p, nrows=0).columns.tolist())
        except Exception:
            pass

    usecols = ["Complaint_Player_ID", "Complaint_ID"]
    if "Complaint_Datetime" in available:
        usecols.append("Complaint_Datetime")
    has_datetime = "Complaint_Datetime" in available

    for df in iter_csv_chunks(
        paths=complaint_paths,
        usecols=usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Complaint_Player_ID"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None and has_datetime:
            ts = pd.to_datetime(df["Complaint_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()

        counts = df.groupby("Complaint_Player_ID")["Complaint_ID"].count()
        for pid, count in counts.items():
            counts_per_speler[str(pid)] = counts_per_speler.get(str(pid), 0) + int(count)

    records = [{"Player_Profile_ID": pid, "var11_totaal_aantal_complaints_in_periode": v}
               for pid, v in counts_per_speler.items()]
    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "var11_totaal_aantal_complaints_in_periode"])

    if logger:
        logger.info(f"✅ Var11 klaar: {len(result):,} spelers")
    return result


# ============================================================================
# Var8a: Geboortedatum
# ============================================================================

def var8a_geboortedatum(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var8a: Geboortedatum — meest recente Player_Profile_DOB per speler.

    Data: WOK_Player_Profile (Player_Profile_DOB)
    Output: var8a_geboortedatum (string, bijv. '5/5/1975')

    Geen tijdsfiltering: DOB is een statisch kenmerk.
    Recency: kiest het meest recente record op basis van Player_Profile_Modified
    of Extraction_Date (zelfde logica als f6_age in feature_engineering_spanish.py).
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var8a_geboortedatum")
        logger.info("▶ START Var8a: Geboortedatum")
    else:
        logger = None

    profile_paths = tables.get("WOK_Player_Profile")
    if not profile_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "var8a_geboortedatum"])

    desired_cols = ["Player_Profile_DOB", "Player_Profile_Modified", "Extraction_Date"]
    best: Dict[str, tuple] = {}  # pid -> (rank_ts, dob_str)

    for p in profile_paths:
        try:
            file_cols = set(pd.read_csv(p, nrows=0).columns.tolist())
        except Exception:
            continue
        if "Player_Profile_DOB" not in file_cols:
            continue
        usecols = ["Player_Profile_ID", "Player_Profile_DOB"] + [
            c for c in desired_cols if c in file_cols and c != "Player_Profile_DOB"
        ]
        for df in iter_csv_chunks(paths=[p], usecols=usecols, chunksize=chunksize, verbose=verbose):
            df = df[df["Player_Profile_ID"].notna() & df["Player_Profile_DOB"].notna()].copy()
            if df.empty:
                continue

            recency = None
            if "Player_Profile_Modified" in df.columns:
                recency = pd.to_datetime(df["Player_Profile_Modified"], errors="coerce", utc=True)
            if recency is None or recency.isna().all():
                if "Extraction_Date" in df.columns:
                    recency = pd.to_datetime(df["Extraction_Date"], errors="coerce", utc=True)
            if recency is None:
                recency = pd.Series([pd.NaT] * len(df), index=df.index)
            df["rank_ts"] = recency.fillna(pd.Timestamp.min.tz_localize("UTC"))

            for row in df.itertuples(index=False):
                pid = str(getattr(row, "Player_Profile_ID"))
                dob = getattr(row, "Player_Profile_DOB")
                rank_ts = getattr(row, "rank_ts")
                prev = best.get(pid)
                if prev is None or rank_ts > prev[0]:
                    best[pid] = (rank_ts, dob)

    records = [{"Player_Profile_ID": pid, "var8a_geboortedatum": v} for pid, (_, v) in best.items()]
    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "var8a_geboortedatum"])
    if logger:
        logger.info(f"✅ Var8a klaar: {len(out):,} spelers")
    return out


# ============================================================================
# Var9a: Aantal bankrekeningen
# ============================================================================

def var9a_aantal_bankrekeningen(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var9a: Aantal unieke bankrekeningen per speler.

    Data: WOK_Player_Profile (Player_Profile_Bank_Account JSON)
    Output: var9a_aantal_bankrekeningen (int >= 0)

    Geen tijdsfiltering: telt alle gekende bankrekeningen (Bank_Account_ID's).
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var9a_aantal_bankrekeningen")
        logger.info("▶ START Var9a: Aantal bankrekeningen")
    else:
        logger = None

    profile_paths = tables.get("WOK_Player_Profile")
    if not profile_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "var9a_aantal_bankrekeningen"])

    bank_ids: Dict[str, set] = {}

    for df in iter_csv_chunks(
        paths=profile_paths,
        usecols=["Player_Profile_ID", "Player_Profile_Bank_Account"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        col_id = df.columns.get_loc("Player_Profile_ID")
        col_json = df.columns.get_loc("Player_Profile_Bank_Account")

        for i in range(len(df)):
            pid = str(df.iat[i, col_id])
            json_veld = df.iat[i, col_json]
            if pid not in bank_ids:
                bank_ids[pid] = set()
            for account_id in simple_Player_Profile_Bank_Account_json_iterator(
                json_veld, needed_vars=["Bank_Account_ID"]
            ):
                bank_ids[pid].add(account_id)

    out = pd.DataFrame({
        "Player_Profile_ID": list(bank_ids.keys()),
        "var9a_aantal_bankrekeningen": [len(s) for s in bank_ids.values()],
    })
    if logger:
        logger.info(f"✅ Var9a klaar: {len(out):,} spelers")
    return out


# ============================================================================
# Groep 10a-h: Aanwezigheid status (anchor + no-ops)
# ============================================================================

_STATUS_COLS = {
    "ACTIVE":               "var10a_status_active",
    "TRIAL":                "var10b_status_trial",
    "SUSPENDED":            "var10c_status_suspended",
    "SUSPENDED_DEATH":      "var10d_status_suspended_death",
    "BLOCKED":              "var10e_status_blocked",
    "SELF_EXCLUDED_TEMP":   "var10f_status_self_excluded_temp",
    "SELF_EXCLUDED_INDEF":  "var10g_status_self_excluded_indef",
    "OTHER":                "var10h_status_other",
}

def var10a_tot_var10h(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var10a–10h: Aanwezigheid van elk spelersstatus als binaire kolom (0/1).

    Data: WOK_Player_Profile (Player_Profile_Status, Player_Profile_Modified)
    Logica:
    - Per speler: verzamel alle statussen gezien in de periode
      (filter op Player_Profile_Modified in [start, eind) als x_tijdspad gegeven).
    - Voor elke status: 1 als speler ooit die status had, anders 0.
    Output: 8 kolommen var10a … var10h
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var10a_tot_var10h")
        logger.info("▶ START Var10a-h: Aanwezigheid status")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    profile_paths = tables.get("WOK_Player_Profile")
    if not profile_paths:
        cols = ["Player_Profile_ID"] + list(_STATUS_COLS.values())
        return pd.DataFrame(columns=cols)

    statuses_per_player: Dict[str, set] = {}

    for df in iter_csv_chunks(
        paths=profile_paths,
        usecols=["Player_Profile_ID", "Player_Profile_Status", "Player_Profile_Modified"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna() & df["Player_Profile_Status"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None and "Player_Profile_Modified" in df.columns:
            ts = pd.to_datetime(df["Player_Profile_Modified"], errors="coerce", utc=True).dt.tz_localize(None)
            df["__ts"] = ts
            mask = df["__ts"].isna() | ((df["__ts"] >= start_datum) & (df["__ts"] < eind_datum))
            df = df[mask].copy()
            if df.empty:
                continue

        col_id = df.columns.get_loc("Player_Profile_ID")
        col_st = df.columns.get_loc("Player_Profile_Status")

        for i in range(len(df)):
            pid = str(df.iat[i, col_id])
            status = str(df.iat[i, col_st]).strip().upper()
            if pid not in statuses_per_player:
                statuses_per_player[pid] = set()
            statuses_per_player[pid].add(status)

    records = []
    for pid, seen in statuses_per_player.items():
        row = {"Player_Profile_ID": pid}
        for status_key, col_name in _STATUS_COLS.items():
            row[col_name] = 1 if status_key in seen else 0
        records.append(row)

    cols = ["Player_Profile_ID"] + list(_STATUS_COLS.values())
    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=cols)
    if logger:
        logger.info(f"✅ Var10a-h klaar: {len(out):,} spelers")
    return out


# ============================================================================
# Var10i: Gemiddeld saldo
# ============================================================================

def var10i_gemiddeld_saldo(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var10i: Gemiddeld EOD-saldo per speler over de periode.

    Data: WOK_Player_Profile (Player_Profile_EOD_Balance, Extraction_Date)
    Logica:
    - Filter op Extraction_Date in [start, eind) als x_tijdspad gegeven.
    - Bereken het gemiddelde van alle EOD-saldo-snapshots per speler.
    Output: var10i_gemiddeld_saldo (float)
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var10i_gemiddeld_saldo")
        logger.info("▶ START Var10i: Gemiddeld saldo")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    profile_paths = tables.get("WOK_Player_Profile")
    if not profile_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "var10i_gemiddeld_saldo"])

    balances: Dict[str, list] = {}

    for df in iter_csv_chunks(
        paths=profile_paths,
        usecols=["Player_Profile_ID", "Player_Profile_EOD_Balance", "Extraction_Date"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None and "Extraction_Date" in df.columns:
            ts = pd.to_datetime(df["Extraction_Date"], errors="coerce", utc=True).dt.tz_localize(None)
            mask = (ts >= start_datum) & (ts < eind_datum)
            df = df[mask].copy()
            if df.empty:
                continue

        bal = pd.to_numeric(df["Player_Profile_EOD_Balance"], errors="coerce")
        df["__bal"] = bal

        col_id = df.columns.get_loc("Player_Profile_ID")
        col_bal = df.columns.get_loc("__bal")

        for i in range(len(df)):
            pid = str(df.iat[i, col_id])
            b = df.iat[i, col_bal]
            if pd.isna(b):
                continue
            if pid not in balances:
                balances[pid] = []
            balances[pid].append(float(b))

    records = [
        {"Player_Profile_ID": pid, "var10i_gemiddeld_saldo": float(np.mean(vals))}
        for pid, vals in balances.items()
    ]
    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "var10i_gemiddeld_saldo"])
    if logger:
        logger.info(f"✅ Var10i klaar: {len(out):,} spelers")
    return out


# ============================================================================
# Var10j: RG-klasse one-hot (dynamische kolommen)
# ============================================================================

def var10j_rg_class(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var10j: RG-klasse als one-hot kolommen per speler.

    Data: WOK_Player_Flags (Flag_RG_Class JSON → RG_Class_Value)
    Logica:
    - Verzamel per speler alle unieke RG_Class_Value's.
    - Maak per unieke waarde een binaire kolom: var10j_rg_<waarde> (lowercase).
    Output: 1+ kolommen var10j_rg_* (operator-afhankelijk)

    Noot: kolomnamen zijn dynamisch (operator-specifiek). Bij het samenvoegen
    van operators is dit geen probleem: dezelfde klasse-namen worden hergebruikt.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var10j_rg_class")
        logger.info("▶ START Var10j: RG-klasse one-hot")
    else:
        logger = None

    flags_paths = tables.get("WOK_Player_Flags")
    if not flags_paths:
        return pd.DataFrame(columns=["Player_Profile_ID"])

    rg_per_player: Dict[str, set] = {}

    for df in iter_csv_chunks(
        paths=flags_paths,
        usecols=["Player_Profile_ID", "Flag_RG_Class"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        col_id = df.columns.get_loc("Player_Profile_ID")
        col_json = df.columns.get_loc("Flag_RG_Class")

        for i in range(len(df)):
            pid = str(df.iat[i, col_id])
            json_veld = df.iat[i, col_json]
            if pid not in rg_per_player:
                rg_per_player[pid] = set()
            for rg_val in simple_RG_Class_Value_from_FLAG_RG_CLASS_json_iterator(
                json_veld, needed_vars=["Flag_RG_Class"]
            ):
                rg_per_player[pid].add(str(rg_val).strip())

    if not rg_per_player:
        return pd.DataFrame(columns=["Player_Profile_ID"])

    all_classes = sorted({v for s in rg_per_player.values() for v in s})
    col_names = {c: f"var10j_rg_{c.lower()}" for c in all_classes}

    records = []
    for pid, seen in rg_per_player.items():
        row: Dict = {"Player_Profile_ID": pid}
        for cls, col in col_names.items():
            row[col] = 1 if cls in seen else 0
        records.append(row)

    out = pd.DataFrame.from_records(records)
    if logger:
        logger.info(f"✅ Var10j klaar: {len(out):,} spelers, klassen: {all_classes}")
    return out


# ============================================================================
# Groep 10k-l: Onsuccesvolle/succesvolle transacties (anchor + no-op)
# ============================================================================

def var10k_tot_var10l(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var10k–10l: Onsuccesvolle transacties + ratio succesvolle transacties.

    Data: WOK_Player_Account_Transaction (Transaction_Status, Transaction_Datetime)
    Logica:
    - var10k: aantal transacties waarbij Transaction_Status != 'SUCCESSFUL'
    - var10l: ratio succesvolle transacties (SUCCESSFUL / totaal)
    Output: 2 kolommen
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var10k_tot_var10l")
        logger.info("▶ START Var10k-l: Onsuccesvol/ratio transacties")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID",
                                     "var10k_totaal_aantal_onsuccesvolle_transacties",
                                     "var10l_ratio_succesvolle_transacties"])

    # {pid: {"succ": int, "total": int}}
    counts: Dict[str, Dict[str, int]] = {}

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask = (ts >= start_datum) & (ts < eind_datum)
            df = df[mask].copy()
            if df.empty:
                continue

        df["__succ"] = df["Transaction_Status"].astype(str).str.upper() == "SUCCESSFUL"

        grp = df.groupby("Player_Profile_ID")["__succ"]
        succ = grp.sum()
        total = grp.count()

        for pid in total.index:
            pid_s = str(pid)
            if pid_s not in counts:
                counts[pid_s] = {"succ": 0, "total": 0}
            counts[pid_s]["succ"]  += int(succ[pid])
            counts[pid_s]["total"] += int(total[pid])

    records = []
    for pid, c in counts.items():
        ratio = (c["succ"] / c["total"]) if c["total"] > 0 else np.nan
        records.append({
            "Player_Profile_ID": pid,
            "var10k_totaal_aantal_onsuccesvolle_transacties": c["total"] - c["succ"],
            "var10l_ratio_succesvolle_transacties": ratio,
        })

    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID",
                 "var10k_totaal_aantal_onsuccesvolle_transacties",
                 "var10l_ratio_succesvolle_transacties"])
    if logger:
        logger.info(f"✅ Var10k-l klaar: {len(out):,} spelers")
    return out


# ============================================================================
# Groep 19a-b: Gokken in weekend en 's nachts (anchor + no-op)
# ============================================================================

def var19a_tot_var19b(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var19a–19b: Absolute aantal transacties in het weekend en 's nachts.

    Data: WOK_Player_Account_Transaction (Transaction_Datetime)
    Definities (Europe/Amsterdam, inclusief zomer-/wintertijd):
      - Weekend: Transaction_Datetime valt op zaterdag (dayofweek=5) of zondag (6)
      - Nacht:   Transaction_Datetime uur in [22, 23, 0, 1, 2, 3, 4, 5] (22:00–06:00)
    Tijdsfiltering: x_tijdspad op Transaction_Datetime indien opgegeven.

    Output:
      - var19a_totaal_gokken_weekend (int >= 0)
      - var19b_totaal_gokken_nacht   (int >= 0)
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var19a_tot_var19b")
        logger.info("▶ START Var19a-b: Gokken weekend/nacht")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID",
                                     "var19a_totaal_gokken_weekend",
                                     "var19b_totaal_gokken_nacht"])

    counts: Dict[str, Dict[str, int]] = {}

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["__ts"] = ts
        df = df[df["__ts"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            df = df[(df["__ts"] >= start_datum) & (df["__ts"] < eind_datum)].copy()
            if df.empty:
                continue

        civil_time = local_time(df["__ts"])
        df["__dow"]  = civil_time.dt.dayofweek        # 0=Mon … 5=Sat, 6=Sun
        df["__hour"] = civil_time.dt.hour

        weekend_mask = df["__dow"] >= 5
        nacht_mask   = (df["__hour"] >= 22) | (df["__hour"] < 6)

        for pid, grp in df.groupby("Player_Profile_ID"):
            pid_s = str(pid)
            if pid_s not in counts:
                counts[pid_s] = {"weekend": 0, "nacht": 0}
            counts[pid_s]["weekend"] += int(weekend_mask[grp.index].sum())
            counts[pid_s]["nacht"]   += int(nacht_mask[grp.index].sum())

    records = [
        {
            "Player_Profile_ID": pid,
            "var19a_totaal_gokken_weekend": c["weekend"],
            "var19b_totaal_gokken_nacht":   c["nacht"],
        }
        for pid, c in counts.items()
    ]
    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID",
                 "var19a_totaal_gokken_weekend",
                 "var19b_totaal_gokken_nacht"])
    if logger:
        logger.info(f"✅ Var19a-b klaar: {len(out):,} spelers")
    return out


# ============================================================================
# Var20a-b: Max transacties per dag / max ingezet bedrag per dag
# ============================================================================

def var20a_tot_var20b(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var20a–20b: Maximale dagactiviteit per speler.

    Data: WOK_Player_Account_Transaction
      - var20a_max_transacties_per_dag  : hoogste aantal transacties op één dag
      - var20b_max_inzet_per_dag        : hoogste totaal Transaction_Amount op één dag

    Tijdsfiltering: x_tijdspad op Transaction_Datetime indien opgegeven.
    Datum afgeleid van Transaction_Datetime in Europe/Amsterdam. UTC-venstergrenzen blijven gelijk.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var20a_tot_var20b")
        logger.info("▶ START Var20a-b: Max transacties/inzet per dag")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID",
                                     "var20a_max_transacties_per_dag",
                                     "var20b_max_inzet_per_dag"])

    # Per speler: dict day_str → (count, amount_sum)
    daily: Dict[str, Dict[str, list]] = {}   # pid → {day: [count, amount]}

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Amount"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["__ts"] = ts
        df = df[df["__ts"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            df = df[(df["__ts"] >= start_datum) & (df["__ts"] < eind_datum)].copy()
            if df.empty:
                continue

        df["__day"] = local_day_labels(df["__ts"])
        df["Transaction_Amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce").fillna(0.0)

        for pid, grp in df.groupby("Player_Profile_ID"):
            pid_s = str(pid)
            if pid_s not in daily:
                daily[pid_s] = {}
            for day, day_grp in grp.groupby("__day"):
                day_key = str(day)
                cnt = len(day_grp)
                amt = float(day_grp["Transaction_Amount"].sum())
                if day_key not in daily[pid_s]:
                    daily[pid_s][day_key] = [cnt, amt]
                else:
                    daily[pid_s][day_key][0] += cnt
                    daily[pid_s][day_key][1] += amt

    records = []
    for pid, days in daily.items():
        counts = [v[0] for v in days.values()]
        amounts = [v[1] for v in days.values()]
        records.append({
            "Player_Profile_ID": pid,
            "var20a_max_transacties_per_dag": int(max(counts)) if counts else 0,
            "var20b_max_inzet_per_dag":       float(max(amounts)) if amounts else 0.0,
        })

    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID",
                 "var20a_max_transacties_per_dag",
                 "var20b_max_inzet_per_dag"])
    if logger:
        logger.info(f"✅ Var20a-b klaar: {len(out):,} spelers")
    return out


# ============================================================================
# Var18: Aantal gok-dagen (via WOK_Player_Profile Extraction_Date)
# ============================================================================

def var18_aantal_gok_dagen(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var18: Aantal dagen dat er gegokt is in de periode.

    Proxy: WOK_Player_Profile wordt elke dag geüpdatet als er activiteit is.
    Door per speler het aantal unieke Extraction_Date-datums te tellen krijg je
    het aantal actieve gokdagen — zonder alle transactiebestanden te hoeven lezen.

    Data: WOK_Player_Profile (Player_Profile_ID, Extraction_Date)
    Logica:
    - Herleid Extraction_Date naar een datum (zonder tijd).
    - Filter op [start, eind) als x_tijdspad gegeven.
    - Tel unieke datums per speler.
    Output: var18_aantal_gok_dagen (int >= 0)
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var18_aantal_gok_dagen")
        logger.info("▶ START Var18: Aantal gok-dagen")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]}")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    profile_paths = tables.get("WOK_Player_Profile")
    if not profile_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "var18_aantal_gok_dagen"])

    dates_per_player: Dict[str, set] = {}

    for df in iter_csv_chunks(
        paths=profile_paths,
        usecols=["Player_Profile_ID", "Extraction_Date"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        ts = pd.to_datetime(df["Extraction_Date"], errors="coerce", utc=True).dt.tz_localize(None)
        df["__date"] = ts.dt.normalize()

        if start_datum is not None:
            mask = (ts >= start_datum) & (ts < eind_datum)
            df = df[mask].copy()
            if df.empty:
                continue

        col_id   = df.columns.get_loc("Player_Profile_ID")
        col_date = df.columns.get_loc("__date")

        for i in range(len(df)):
            pid = str(df.iat[i, col_id])
            d   = df.iat[i, col_date]
            if pd.isna(d):
                continue
            if pid not in dates_per_player:
                dates_per_player[pid] = set()
            dates_per_player[pid].add(d)

    records = [{"Player_Profile_ID": pid, "var18_aantal_gok_dagen": len(s)}
               for pid, s in dates_per_player.items()]
    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "var18_aantal_gok_dagen"])
    if logger:
        logger.info(f"✅ Var18 klaar: {len(out):,} spelers")
    return out


# ============================================================================
# Var17: Self-exclusion temporary count
# ============================================================================

def var17_self_excl_temp_count(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var17: Aantal snapshots waarbij Player_Profile_Status == 'SELF_EXCLUDED_TEMP'.

    Data: WOK_Player_Profile (Player_Profile_Status, Extraction_Date)
    Logica:
    - Tel per speler het aantal rijen met status SELF_EXCLUDED_TEMP.
    - Spelers zonder die status krijgen 0.
    - Tijdsfiltering op Extraction_Date in [start, eind) als x_tijdspad gegeven.
    Output: var17_self_excl_temp_count (int >= 0)
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var17_self_excl_temp_count")
        logger.info("▶ START Var17: Self-Exclusion Temporary Count")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]}")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    profile_paths = tables.get("WOK_Player_Profile")
    if not profile_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "var17_self_excl_temp_count"])

    counts: Dict[str, int] = {}

    usecols = ["Player_Profile_ID", "Player_Profile_Status"]
    if x_tijdspad:
        usecols.append("Extraction_Date")

    for df in iter_csv_chunks(paths=profile_paths, usecols=usecols, chunksize=chunksize, verbose=verbose):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            df["_date"] = pd.to_datetime(df["Extraction_Date"], errors="coerce", utc=True).dt.tz_localize(None)
            df = df[(df["_date"] >= start_datum) & (df["_date"] < eind_datum)].copy()
            if df.empty:
                continue

        mask = df["Player_Profile_Status"] == "SELF_EXCLUDED_TEMP"
        for pid, grp in df[mask].groupby("Player_Profile_ID"):
            counts[str(pid)] = counts.get(str(pid), 0) + len(grp)

        for pid in df["Player_Profile_ID"].unique():
            if str(pid) not in counts:
                counts[str(pid)] = 0

    records = [{"Player_Profile_ID": pid, "var17_self_excl_temp_count": cnt}
               for pid, cnt in counts.items()]
    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "var17_self_excl_temp_count"])

    if logger:
        logger.info(f"✅ Var17 klaar: {len(out):,} spelers")
        if len(out) > 0:
            n_ever = (out["var17_self_excl_temp_count"] > 0).sum()
            logger.info(f"   Spelers ooit SELF_EXCLUDED_TEMP: {n_ever:,}")
    return out


# ============================================================================
# Var12: Aantal interventies
# ============================================================================

def var12_totaal_aantal_interventions(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var12: Totaal aantal interventies per speler in de periode.

    Data: WOK_Intervention (Player_Profile_ID, Intervention_ID, Intervention_Begin_Datetime)
    Output: var12_totaal_aantal_interventions (int >= 0)
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var12_interventions")
        logger.info("▶ START Var12: Aantal interventies")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    interv_paths = tables.get("WOK_Intervention")
    if not interv_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "var12_totaal_aantal_interventions"])

    counts: Dict[str, int] = {}

    for df in iter_csv_chunks(
        paths=interv_paths,
        usecols=["Player_Profile_ID", "Intervention_ID", "Intervention_Begin_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna() & df["Intervention_ID"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            ts = pd.to_datetime(df["Intervention_Begin_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask = (ts >= start_datum) & (ts < eind_datum)
            df = df[mask].copy()
            if df.empty:
                continue

        grp = df.groupby("Player_Profile_ID")["Intervention_ID"].count()
        for pid, cnt in grp.items():
            counts[str(pid)] = counts.get(str(pid), 0) + int(cnt)

    records = [{"Player_Profile_ID": pid, "var12_totaal_aantal_interventions": v}
               for pid, v in counts.items()]
    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "var12_totaal_aantal_interventions"])
    if logger:
        logger.info(f"✅ Var12 klaar: {len(out):,} spelers")
    return out


# ============================================================================
# Var13: Aantal game types
# ============================================================================

def var13_aantal_game_types(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var13: Aantal unieke game types per speler in de periode.

    Data: WOK_Game (Game_ID → Game_Type lookup) + WOK_Game_Session
          (Game_Transactions JSON → Player_Profile_ID)
    Output: var13_aantal_game_types (int >= 0)
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var13_game_types")
        logger.info("▶ START Var13: Aantal game types")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    session_paths = tables.get("WOK_Game_Session")
    if not session_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "var13_aantal_game_types"])

    # Build Game_ID -> Game_Type lookup
    game_type_by_id: Dict[str, str] = {}
    game_paths = tables.get("WOK_Game") or []
    for gdf in iter_csv_chunks(
        paths=game_paths,
        usecols=["Game_ID", "Game_Type"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        gdf = gdf[gdf["Game_ID"].notna()].copy()
        if gdf.empty:
            continue
        col_id = gdf.columns.get_loc("Game_ID")
        col_ty = gdf.columns.get_loc("Game_Type") if "Game_Type" in gdf.columns else None
        for i in range(len(gdf)):
            gid = str(gdf.iat[i, col_id])
            gty = str(gdf.iat[i, col_ty]).strip() if col_ty is not None else ""
            if gty and gty not in ("nan", ""):
                game_type_by_id[gid] = gty

    # Accumulate {pid: set(game_types)} from sessions
    game_types_per_player: Dict[str, set] = {}

    for df in iter_csv_chunks(
        paths=session_paths,
        usecols=["Game_ID", "Game_Session_Start_Datetime", "Game_Transactions"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        if start_datum is not None:
            ts = pd.to_datetime(df["Game_Session_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask = (ts >= start_datum) & (ts < eind_datum)
            df = df[mask].copy()
            if df.empty:
                continue

        col_gid = df.columns.get_loc("Game_ID")
        col_tx  = df.columns.get_loc("Game_Transactions")

        for i in range(len(df)):
            gid = str(df.iat[i, col_gid]) if not pd.isna(df.iat[i, col_gid]) else ""
            game_type = game_type_by_id.get(gid, gid)  # fallback to Game_ID
            json_veld = df.iat[i, col_tx]
            try:
                for pid, _tx_id in iter_transaction_ids_from_Game_Transactions(json_veld):
                    if pid not in game_types_per_player:
                        game_types_per_player[pid] = set()
                    if game_type:
                        game_types_per_player[pid].add(game_type)
            except Exception:
                continue

    records = [{"Player_Profile_ID": pid, "var13_aantal_game_types": len(s)}
               for pid, s in game_types_per_player.items()]
    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "var13_aantal_game_types"])
    if logger:
        logger.info(f"✅ Var13 klaar: {len(out):,} spelers")
    return out


# ============================================================================
# Groep 14a-c: Game sessions (anchor + no-ops)
# ============================================================================

def var14a_tot_var14d(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var14a–14d: Statistieken over game sessions per speler.

    Data: WOK_Game_Session
      - Game_Transactions JSON → Player_Profile_ID
      - Game_Session_Rounds → totaal rondes
      - Game_Session_Start/End_Datetime → gemiddelde en totale sessieduur (seconden)

    Output:
      - var14a_aantal_game_sessions (int >= 0)
      - var14b_totaal_rondes_game_sessions (int >= 0)
      - var14c_gemiddelde_sessieduur_seconden (float, NaN als geen sessies)
      - var14d_totale_sessieduur_seconden (float >= 0, 0 als geen geldige sessies)
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var14a_tot_var14d")
        logger.info("▶ START Var14a-d: Game session statistieken")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    session_paths = tables.get("WOK_Game_Session")
    if not session_paths:
        return pd.DataFrame(columns=["Player_Profile_ID",
                                     "var14a_aantal_game_sessions",
                                     "var14b_totaal_rondes_game_sessions",
                                     "var14c_gemiddelde_sessieduur_seconden",
                                     "var14d_totale_sessieduur_seconden"])

    # {pid: {"sessions": int, "rounds": int, "durations": [float]}}
    acc: Dict[str, Dict] = {}

    for df in iter_csv_chunks(
        paths=session_paths,
        usecols=["Game_Session_Start_Datetime", "Game_Session_End_Datetime",
                 "Game_Session_Rounds", "Game_Transactions"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        ts_start = pd.to_datetime(df["Game_Session_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["__ts_start"] = ts_start

        if start_datum is not None:
            mask = (ts_start >= start_datum) & (ts_start < eind_datum)
            df = df[mask].copy()
            if df.empty:
                continue

        ts_end = pd.to_datetime(df["Game_Session_End_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["__ts_end"] = ts_end

        rounds_col = pd.to_numeric(df["Game_Session_Rounds"], errors="coerce")
        df["__rounds"] = rounds_col

        col_tx     = df.columns.get_loc("Game_Transactions")
        col_start  = df.columns.get_loc("__ts_start")
        col_end    = df.columns.get_loc("__ts_end")
        col_rounds = df.columns.get_loc("__rounds")

        for i in range(len(df)):
            json_veld = df.iat[i, col_tx]
            ts_s = df.iat[i, col_start]
            ts_e = df.iat[i, col_end]
            rounds = df.iat[i, col_rounds]

            duration = None
            if not pd.isna(ts_s) and not pd.isna(ts_e) and ts_e >= ts_s:
                duration = (ts_e - ts_s).total_seconds()

            try:
                pids = {pid for pid, _ in iter_transaction_ids_from_Game_Transactions(json_veld)}
            except Exception:
                pids = set()

            for pid in pids:
                if pid not in acc:
                    acc[pid] = {"sessions": 0, "rounds": 0, "durations": []}
                acc[pid]["sessions"] += 1
                if not pd.isna(rounds):
                    acc[pid]["rounds"] += int(rounds)
                if duration is not None:
                    acc[pid]["durations"].append(duration)

    records = []
    for pid, c in acc.items():
        avg_dur = float(np.mean(c["durations"])) if c["durations"] else np.nan
        tot_dur = float(np.sum(c["durations"])) if c["durations"] else 0.0
        records.append({
            "Player_Profile_ID": pid,
            "var14a_aantal_game_sessions": c["sessions"],
            "var14b_totaal_rondes_game_sessions": c["rounds"],
            "var14c_gemiddelde_sessieduur_seconden": avg_dur,
            "var14d_totale_sessieduur_seconden": tot_dur,
        })

    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID",
                 "var14a_aantal_game_sessions",
                 "var14b_totaal_rondes_game_sessions",
                 "var14c_gemiddelde_sessieduur_seconden",
                 "var14d_totale_sessieduur_seconden"])
    if logger:
        logger.info(f"✅ Var14a-d klaar: {len(out):,} spelers")
    return out


# ============================================================================
# Groep 15a-b: Bets (anchor + no-op)
# ============================================================================

def var15a_tot_var15b(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var15a–15b: Bet-statistieken per speler.

    Data: WOK_Bet (Bet_Transactions JSON → Player_Profile_ID, Bet_Parts JSON)
    Output:
      - var15a_totaal_aantal_bets (int >= 0)
      - var15b_totaal_aantal_bets_meerdere_delen (int >= 0, bets met >1 Bet_Parts)
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var15a_tot_var15b")
        logger.info("▶ START Var15a-b: Bet statistieken")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    bet_paths = tables.get("WOK_Bet")
    if not bet_paths:
        return pd.DataFrame(columns=["Player_Profile_ID",
                                     "var15a_totaal_aantal_bets",
                                     "var15b_totaal_aantal_bets_meerdere_delen"])

    def _count_parts(bet_parts_cell) -> int:
        obj = _safe_load_json_relaxed(bet_parts_cell)
        if obj is None:
            return 0
        if isinstance(obj, dict):
            parts = obj.get("Part")
            if isinstance(parts, list):
                return sum(1 for x in parts if isinstance(x, dict))
            if isinstance(parts, dict):
                return 1
            return 0
        if isinstance(obj, list):
            return sum(1 for x in obj if isinstance(x, dict))
        return 0

    counts: Dict[str, Dict[str, int]] = {}

    for df in iter_csv_chunks(
        paths=bet_paths,
        usecols=["Bet_Start_Datetime", "Bet_Transactions", "Bet_Parts"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        if start_datum is not None:
            ts = pd.to_datetime(df["Bet_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask = (ts >= start_datum) & (ts < eind_datum)
            df = df[mask].copy()
            if df.empty:
                continue

        col_tx    = df.columns.get_loc("Bet_Transactions")
        col_parts = df.columns.get_loc("Bet_Parts")

        for i in range(len(df)):
            bet_tx    = df.iat[i, col_tx]
            bet_parts = df.iat[i, col_parts]
            n_parts   = _count_parts(bet_parts)

            try:
                pids = list(iter_player_profile_ids_from_Bet_Transactions(bet_tx))
            except Exception:
                continue
            if not pids:
                continue
            pid = str(pids[0])

            if pid not in counts:
                counts[pid] = {"total": 0, "multi": 0}
            counts[pid]["total"] += 1
            if n_parts > 1:
                counts[pid]["multi"] += 1

    records = [
        {
            "Player_Profile_ID": pid,
            "var15a_totaal_aantal_bets": c["total"],
            "var15b_totaal_aantal_bets_meerdere_delen": c["multi"],
        }
        for pid, c in counts.items()
    ]
    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID",
                 "var15a_totaal_aantal_bets",
                 "var15b_totaal_aantal_bets_meerdere_delen"])
    if logger:
        logger.info(f"✅ Var15a-b klaar: {len(out):,} spelers")
    return out


# ============================================================================
# Var16: Aantal responses
# ============================================================================

def var16_totaal_aantal_responses(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Var16: Totaal aantal responses op klachten per speler.

    Data: WOK_Complaint (Complaint_Player_ID, Complaint_ID, Responses JSON)
    Output: var16_totaal_aantal_responses (int >= 0)
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "var16_responses")
        logger.info("▶ START Var16: Totaal aantal responses")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    complaint_paths = tables.get("WOK_Complaint")
    if not complaint_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "var16_totaal_aantal_responses"])

    available: set = set()
    for p in complaint_paths:
        try:
            available.update(pd.read_csv(p, nrows=0).columns.tolist())
        except Exception:
            pass

    usecols = ["Complaint_Player_ID", "Complaint_ID", "Responses"]
    if "Complaint_Datetime" in available:
        usecols.append("Complaint_Datetime")
    has_datetime = "Complaint_Datetime" in available

    response_counts: Dict[str, int] = {}

    for df in iter_csv_chunks(
        paths=complaint_paths,
        usecols=usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Complaint_Player_ID"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None and has_datetime:
            ts = pd.to_datetime(df["Complaint_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask = (ts >= start_datum) & (ts < eind_datum)
            df = df[mask].copy()
            if df.empty:
                continue

        col_pid  = df.columns.get_loc("Complaint_Player_ID")
        col_resp = df.columns.get_loc("Responses")

        for i in range(len(df)):
            pid = str(df.iat[i, col_pid])
            json_veld = df.iat[i, col_resp]
            try:
                resp_ids = get_list_of_response_ids_from_Responses_list(json_veld)
                n = len(resp_ids) if resp_ids else 0
            except Exception:
                n = 0
            response_counts[pid] = response_counts.get(pid, 0) + n

    records = [{"Player_Profile_ID": pid, "var16_totaal_aantal_responses": v}
               for pid, v in response_counts.items()]
    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "var16_totaal_aantal_responses"])
    if logger:
        logger.info(f"✅ Var16 klaar: {len(out):,} spelers")
    return out


# ============================================================================
# FEATURES_REGISTRY - Alle beschikbare basis features
# ============================================================================
# Dit is de "bron van waarheid" voor alle features in dit bestand.
# De runner (clean_and_parse.py) gebruikt deze registry om te weten welke
# tabellen en kolommen nodig zijn voor elke feature.

_TX_USECOLS_FULL = {
    "WOK_Player_Account_Transaction": [
        "Player_Profile_ID",
        "Transaction_Datetime",
        "Transaction_Amount",
        "Transaction_Type",
        "Transaction_Status",
        "Transaction_Deposit_Instrument",
    ]
}
_TX_USECOLS_DT = {
    "WOK_Player_Account_Transaction": [
        "Player_Profile_ID",
        "Transaction_Datetime",
    ]
}
_TX_USECOLS_TYPE = {
    "WOK_Player_Account_Transaction": [
        "Player_Profile_ID",
        "Transaction_Datetime",
        "Transaction_Type",
        "Transaction_Status",
    ]
}
_TX_USECOLS_AMT = {
    "WOK_Player_Account_Transaction": [
        "Player_Profile_ID",
        "Transaction_Datetime",
        "Transaction_Amount",
        "Transaction_Type",
        "Transaction_Status",
    ]
}

FEATURES_REGISTRY = {
    # ---- Groep 1: Transactiebedragen ----
    "var1a_totaal_ingezet_bedrag": {
        "stream_fn": var1a_totaal_ingezet_bedrag,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_AMT,
        "log_name": "var1a_totaal_ingezet_bedrag.log",
        "kwargs": {},
    },
    "var1b_variantie_ingezet_bedrag_per_dag": {
        "stream_fn": var1b_variantie_ingezet_bedrag_per_dag,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_AMT,
        "log_name": "var1b_variantie_ingezet_bedrag_per_dag.log",
        "kwargs": {},
    },
    "var1c_totaal_gewonnen_bedrag": {
        "stream_fn": var1c_totaal_gewonnen_bedrag,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_AMT,
        "log_name": "var1c_totaal_gewonnen_bedrag.log",
        "kwargs": {},
    },
    # ---- Groep 2: Transactieaantallen ----
    "var2a_totaal_aantal_transacties": {
        "stream_fn": var2a_totaal_aantal_transacties,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_DT,
        "log_name": "var2a_totaal_aantal_transacties.log",
        "kwargs": {},
    },
    "var2b_variantie_aantal_transacties_per_dag": {
        "stream_fn": var2b_variantie_aantal_transacties_per_dag,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_DT,
        "log_name": "var2b_variantie_aantal_transacties_per_dag.log",
        "kwargs": {},
    },
    # ---- Groep 3: Actieve dagen ----
    "var3a_totaal_aantal_actieve_dagen": {
        "stream_fn": var3a_totaal_aantal_actieve_dagen,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_DT,
        "log_name": "var3a_totaal_aantal_actieve_dagen.log",
        "kwargs": {},
    },
    "var3b_variantie_aantal_actieve_dagen_per_week": {
        "stream_fn": var3b_variantie_aantal_actieve_dagen_per_week,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_DT,
        "log_name": "var3b_variantie_aantal_actieve_dagen_per_week.log",
        "kwargs": {},
    },
    "var3c_percentage_actieve_dagen_in_periode": {
        "stream_fn": var3c_percentage_actieve_dagen_in_periode,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_DT,
        "log_name": "var3c_percentage_actieve_dagen_in_periode.log",
        "kwargs": {},
    },
    # ---- Groep 4: Transactietypes (anchor + no-ops) ----
    "var4a_totaal_aantal_deposit_transacties": {
        "stream_fn": var4a_tot_var4f,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_TYPE,
        "log_name": "var4_transaction_types.log",
        "kwargs": {},
    },
    "var4b_totaal_aantal_withdrawal_transacties": {
        "stream_fn": var4a_tot_var4f,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_TYPE,
        "log_name": "var4_transaction_types.log",
        "kwargs": {},
    },
    "var4c_totaal_aantal_stake_transacties": {
        "stream_fn": var4a_tot_var4f,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_TYPE,
        "log_name": "var4_transaction_types.log",
        "kwargs": {},
    },
    "var4d_totaal_aantal_winning_transacties": {
        "stream_fn": var4a_tot_var4f,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_TYPE,
        "log_name": "var4_transaction_types.log",
        "kwargs": {},
    },
    "var4e_totaal_aantal_other_transacties": {
        "stream_fn": var4a_tot_var4f,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_TYPE,
        "log_name": "var4_transaction_types.log",
        "kwargs": {},
    },
    "var4f_totaal_aantal_bonus_transacties": {
        "stream_fn": var4a_tot_var4f,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_TYPE,
        "log_name": "var4_transaction_types.log",
        "kwargs": {},
    },
    # ---- Groep 5: Transactie-instrumenten (anchor + no-ops) ----
    "var5a_totaal_aantal_transaction_instruments": {
        "stream_fn": var5a_tot_var5e,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_FULL,
        "log_name": "var5_transaction_instruments.log",
        "kwargs": {},
    },
    "var5b_totaal_aantal_CREDIT_CARD_instrument": {
        "stream_fn": var5a_tot_var5e,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_FULL,
        "log_name": "var5_transaction_instruments.log",
        "kwargs": {},
    },
    "var5c_totaal_aantal_ELECTRONIC_MONEY_instrument": {
        "stream_fn": var5a_tot_var5e,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_FULL,
        "log_name": "var5_transaction_instruments.log",
        "kwargs": {},
    },
    "var5d_totaal_aantal_BANK_TRANSFER_instrument": {
        "stream_fn": var5a_tot_var5e,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_FULL,
        "log_name": "var5_transaction_instruments.log",
        "kwargs": {},
    },
    "var5e_totaal_aantal_OTHER_instrument": {
        "stream_fn": var5a_tot_var5e,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_FULL,
        "log_name": "var5_transaction_instruments.log",
        "kwargs": {},
    },
    # ---- Groep 6: Dagen met veel transacties (anchor + no-ops) ----
    "var6a_totaal_aantal_dagen_met_5_tranacties_of_meer": {
        "stream_fn": var6a_tot_var6c,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_DT,
        "log_name": "var6_hoge_activiteit_dagen.log",
        "kwargs": {},
    },
    "var6b_totaal_aantal_dagen_met_25_tranacties_of_meer": {
        "stream_fn": var6a_tot_var6c,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_DT,
        "log_name": "var6_hoge_activiteit_dagen.log",
        "kwargs": {},
    },
    "var6c_totaal_aantal_dagen_met_100_tranacties_of_meer": {
        "stream_fn": var6a_tot_var6c,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": _TX_USECOLS_DT,
        "log_name": "var6_hoge_activiteit_dagen.log",
        "kwargs": {},
    },
    # ---- Groep 7: Limieten (anchor + no-ops) ----
    "var7a_totaal_aantal_limits_in_periode": {
        "stream_fn": var7a_tot_var7d,
        "tables": ["WOK_Player_Limits"],
        "usecols": {
            "WOK_Player_Limits": [
                "Player_Profile_ID",
                "Limit_Deposit",
                "Limit_Login",
                "Limit_Balance",
                "Limit_Participation",
                "Limit_Game_Type",
            ]
        },
        "log_name": "var7_limieten.log",
        "kwargs": {},
    },
    "var7b_totaal_aantal_limit_deposits_in_periode": {
        "stream_fn": var7a_tot_var7d,
        "tables": ["WOK_Player_Limits"],
        "usecols": {
            "WOK_Player_Limits": [
                "Player_Profile_ID",
                "Limit_Deposit",
                "Limit_Login",
                "Limit_Balance",
                "Limit_Participation",
                "Limit_Game_Type",
            ]
        },
        "log_name": "var7_limieten.log",
        "kwargs": {},
    },
    "var7c_totaal_aantal_limit_login_in_periode": {
        "stream_fn": var7a_tot_var7d,
        "tables": ["WOK_Player_Limits"],
        "usecols": {
            "WOK_Player_Limits": [
                "Player_Profile_ID",
                "Limit_Deposit",
                "Limit_Login",
                "Limit_Balance",
                "Limit_Participation",
                "Limit_Game_Type",
            ]
        },
        "log_name": "var7_limieten.log",
        "kwargs": {},
    },
    "var7d_totaal_aantal_limit_balance_in_periode": {
        "stream_fn": var7a_tot_var7d,
        "tables": ["WOK_Player_Limits"],
        "usecols": {
            "WOK_Player_Limits": [
                "Player_Profile_ID",
                "Limit_Deposit",
                "Limit_Login",
                "Limit_Balance",
                "Limit_Participation",
                "Limit_Game_Type",
            ]
        },
        "log_name": "var7_limieten.log",
        "kwargs": {},
    },
    # ---- Var8a: Geboortedatum ----
    "var8a_geboortedatum": {
        "stream_fn": var8a_geboortedatum,
        "tables": ["WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Profile": [
                "Player_Profile_ID",
                "Player_Profile_DOB",
                "Player_Profile_Modified",
                "Extraction_Date",
            ]
        },
        "log_name": "var8a_geboortedatum.log",
        "kwargs": {},
    },
    # ---- Var9a: Bankrekeningen ----
    "var9a_aantal_bankrekeningen": {
        "stream_fn": var9a_aantal_bankrekeningen,
        "tables": ["WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Profile": [
                "Player_Profile_ID",
                "Player_Profile_Bank_Account",
            ]
        },
        "log_name": "var9a_bankrekeningen.log",
        "kwargs": {},
    },
    # ---- Groep 10a-h: Status one-hot (anchor + no-ops) ----
    "var10a_status_active": {
        "stream_fn": var10a_tot_var10h,
        "tables": ["WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Profile": [
                "Player_Profile_ID",
                "Player_Profile_Status",
                "Player_Profile_Modified",
            ]
        },
        "log_name": "var10_status.log",
        "kwargs": {},
    },
    "var10b_status_trial":           {"stream_fn": var10a_tot_var10h, "tables": ["WOK_Player_Profile"], "usecols": {"WOK_Player_Profile": ["Player_Profile_ID", "Player_Profile_Status", "Player_Profile_Modified"]}, "log_name": "var10_status.log", "kwargs": {}},
    "var10c_status_suspended":       {"stream_fn": var10a_tot_var10h, "tables": ["WOK_Player_Profile"], "usecols": {"WOK_Player_Profile": ["Player_Profile_ID", "Player_Profile_Status", "Player_Profile_Modified"]}, "log_name": "var10_status.log", "kwargs": {}},
    "var10d_status_suspended_death": {"stream_fn": var10a_tot_var10h, "tables": ["WOK_Player_Profile"], "usecols": {"WOK_Player_Profile": ["Player_Profile_ID", "Player_Profile_Status", "Player_Profile_Modified"]}, "log_name": "var10_status.log", "kwargs": {}},
    "var10e_status_blocked":         {"stream_fn": var10a_tot_var10h, "tables": ["WOK_Player_Profile"], "usecols": {"WOK_Player_Profile": ["Player_Profile_ID", "Player_Profile_Status", "Player_Profile_Modified"]}, "log_name": "var10_status.log", "kwargs": {}},
    "var10f_status_self_excluded_temp":  {"stream_fn": var10a_tot_var10h, "tables": ["WOK_Player_Profile"], "usecols": {"WOK_Player_Profile": ["Player_Profile_ID", "Player_Profile_Status", "Player_Profile_Modified"]}, "log_name": "var10_status.log", "kwargs": {}},
    "var10g_status_self_excluded_indef": {"stream_fn": var10a_tot_var10h, "tables": ["WOK_Player_Profile"], "usecols": {"WOK_Player_Profile": ["Player_Profile_ID", "Player_Profile_Status", "Player_Profile_Modified"]}, "log_name": "var10_status.log", "kwargs": {}},
    "var10h_status_other":               {"stream_fn": var10a_tot_var10h, "tables": ["WOK_Player_Profile"], "usecols": {"WOK_Player_Profile": ["Player_Profile_ID", "Player_Profile_Status", "Player_Profile_Modified"]}, "log_name": "var10_status.log", "kwargs": {}},
    # ---- Var10i: Gemiddeld saldo ----
    "var10i_gemiddeld_saldo": {
        "stream_fn": var10i_gemiddeld_saldo,
        "tables": ["WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Profile": [
                "Player_Profile_ID",
                "Player_Profile_EOD_Balance",
                "Extraction_Date",
            ]
        },
        "log_name": "var10i_saldo.log",
        "kwargs": {},
    },
    # ---- Var10j: RG-klasse one-hot ----
    "var10j_rg_class": {
        "stream_fn": var10j_rg_class,
        "tables": ["WOK_Player_Flags"],
        "usecols": {
            "WOK_Player_Flags": [
                "Player_Profile_ID",
                "Flag_RG_Class",
            ]
        },
        "log_name": "var10j_rg_class.log",
        "kwargs": {},
    },
    # ---- Groep 10k-l: Onsuccesvolle transacties (anchor + no-op) ----
    "var10k_totaal_aantal_onsuccesvolle_transacties": {
        "stream_fn": var10k_tot_var10l,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Status",
            ]
        },
        "log_name": "var10k_onsuccesvol.log",
        "kwargs": {},
    },
    "var10l_ratio_succesvolle_transacties": {
        "stream_fn": var10k_tot_var10l,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Status",
            ]
        },
        "log_name": "var10k_onsuccesvol.log",
        "kwargs": {},
    },
    # ---- Groep 19a-b: Weekend/nacht (anchor + no-op) ----
    "var19a_totaal_gokken_weekend": {
        "stream_fn": var19a_tot_var19b,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
            ]
        },
        "log_name": "var19_weekend_nacht.log",
        "kwargs": {},
    },
    "var19b_totaal_gokken_nacht": {
        "stream_fn": var19a_tot_var19b,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
            ]
        },
        "log_name": "var19_weekend_nacht.log",
        "kwargs": {},
    },
    # ---- Groep 20a-b: Max transacties / inzet per dag (anchor + no-op) ----
    "var20a_max_transacties_per_dag": {
        "stream_fn": var20a_tot_var20b,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Amount",
            ]
        },
        "log_name": "var20_max_dag.log",
        "kwargs": {},
    },
    "var20b_max_inzet_per_dag": {
        "stream_fn": var20a_tot_var20b,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Amount",
            ]
        },
        "log_name": "var20_max_dag.log",
        "kwargs": {},
    },
    # ---- Var18: Aantal gok-dagen ----
    "var18_aantal_gok_dagen": {
        "stream_fn": var18_aantal_gok_dagen,
        "tables": ["WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Profile": [
                "Player_Profile_ID",
                "Extraction_Date",
            ]
        },
        "log_name": "var18_gok_dagen.log",
        "kwargs": {},
    },
    # ---- Var17: Self-exclusion temporary count ----
    "var17_self_excl_temp_count": {
        "stream_fn": var17_self_excl_temp_count,
        "tables": ["WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Profile": [
                "Player_Profile_ID",
                "Player_Profile_Status",
                "Extraction_Date",
            ]
        },
        "log_name": "var17_self_excl_temp.log",
        "kwargs": {},
    },
    # ---- Var12: Interventies ----
    "var12_totaal_aantal_interventions": {
        "stream_fn": var12_totaal_aantal_interventions,
        "tables": ["WOK_Intervention"],
        "usecols": {
            "WOK_Intervention": [
                "Player_Profile_ID",
                "Intervention_ID",
                "Intervention_Begin_Datetime",
            ]
        },
        "log_name": "var12_interventions.log",
        "kwargs": {},
    },
    # ---- Var13: Game types ----
    "var13_aantal_game_types": {
        "stream_fn": var13_aantal_game_types,
        "tables": ["WOK_Game", "WOK_Game_Session"],
        "usecols": {
            "WOK_Game": ["Game_ID", "Game_Type"],
            "WOK_Game_Session": ["Game_ID", "Game_Session_Start_Datetime", "Game_Transactions"],
        },
        "log_name": "var13_game_types.log",
        "kwargs": {},
    },
    # ---- Groep 14a-c: Game sessions (anchor + no-ops) ----
    "var14a_aantal_game_sessions": {
        "stream_fn": var14a_tot_var14d,
        "tables": ["WOK_Game_Session"],
        "usecols": {
            "WOK_Game_Session": [
                "Game_Session_Start_Datetime",
                "Game_Session_End_Datetime",
                "Game_Session_Rounds",
                "Game_Transactions",
            ]
        },
        "log_name": "var14_game_sessions.log",
        "kwargs": {},
    },
    "var14b_totaal_rondes_game_sessions": {
        "stream_fn": var14a_tot_var14d,
        "tables": ["WOK_Game_Session"],
        "usecols": {
            "WOK_Game_Session": [
                "Game_Session_Start_Datetime",
                "Game_Session_End_Datetime",
                "Game_Session_Rounds",
                "Game_Transactions",
            ]
        },
        "log_name": "var14_game_sessions.log",
        "kwargs": {},
    },
    "var14c_gemiddelde_sessieduur_seconden": {
        "stream_fn": var14a_tot_var14d,
        "tables": ["WOK_Game_Session"],
        "usecols": {
            "WOK_Game_Session": [
                "Game_Session_Start_Datetime",
                "Game_Session_End_Datetime",
                "Game_Session_Rounds",
                "Game_Transactions",
            ]
        },
        "log_name": "var14_game_sessions.log",
        "kwargs": {},
    },
    "var14d_totale_sessieduur_seconden": {
        "stream_fn": var14a_tot_var14d,
        "tables": ["WOK_Game_Session"],
        "usecols": {
            "WOK_Game_Session": [
                "Game_Session_Start_Datetime",
                "Game_Session_End_Datetime",
                "Game_Session_Rounds",
                "Game_Transactions",
            ]
        },
        "log_name": "var14d_game_sessions.log",
        "kwargs": {},
    },
    # ---- Groep 15a-b: Bets (anchor + no-op) ----
    "var15a_totaal_aantal_bets": {
        "stream_fn": var15a_tot_var15b,
        "tables": ["WOK_Bet"],
        "usecols": {
            "WOK_Bet": [
                "Bet_Start_Datetime",
                "Bet_Transactions",
                "Bet_Parts",
            ]
        },
        "log_name": "var15_bets.log",
        "kwargs": {},
    },
    "var15b_totaal_aantal_bets_meerdere_delen": {
        "stream_fn": var15a_tot_var15b,
        "tables": ["WOK_Bet"],
        "usecols": {
            "WOK_Bet": [
                "Bet_Start_Datetime",
                "Bet_Transactions",
                "Bet_Parts",
            ]
        },
        "log_name": "var15_bets.log",
        "kwargs": {},
    },
    # ---- Var16: Responses ----
    "var16_totaal_aantal_responses": {
        "stream_fn": var16_totaal_aantal_responses,
        "tables": ["WOK_Complaint"],
        "usecols": {
            "WOK_Complaint": [
                "Complaint_Player_ID",
                "Complaint_ID",
                "Responses",
                "Complaint_Datetime",
            ]
        },
        "log_name": "var16_responses.log",
        "kwargs": {},
    },
    # ---- Var11: Klachten ----
    "var11_totaal_aantal_complaints_in_periode": {
        "stream_fn": var11_totaal_aantal_complaints_in_periode,
        "tables": ["WOK_Complaint"],
        "usecols": {
            "WOK_Complaint": [
                "Complaint_Player_ID",
                "Complaint_ID",
                "Complaint_Datetime",
            ]
        },
        "log_name": "var11_complaints.log",
        "kwargs": {},
    },
}
