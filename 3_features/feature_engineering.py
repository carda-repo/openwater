"""
Feature Engineering (streaming)
===============================

Doel
----
Dit bestand definieert alle *streaming features* die we uit de WOK-tabellen
berekenen. Elke feature is een Python-functie die op een uniforme manier
aangeroepen wordt door de runner (`clean_and_parse.py`).

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
- Return: DataFrame met per speler één featurekolom, altijd inclusief:
    ["Player_Profile_ID", "<feature_name>"]

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

Nieuwe feature toevoegen — stappenplan
--------------------------------------
1. Schrijf de stream_fn
   - Definieer een nieuwe functie met de uniforme signature.  
   - Stream de juiste tabellen uit `tables`.  
   - Bereken de feature per speler.  
   - Return een DataFrame met twee kolommen:
        ["Player_Profile_ID", "<jouw_feature_naam>"]

2. Voeg hem toe aan FEATURES_REGISTRY
   - Kies een unieke key (bijv. `"std_casino"`)  
   - Zet daar je functie, benodigde tabellen/kolommen en logbestand in.  

3.	Activeer de feature in een scenario
	- In clean_and_parse.py staat de dictionary SCENARIOS.
	- Voeg je feature-key toe aan de lijst features van een bestaand scenario, of maak een nieuw scenario aan.

De indeling is als volgt:
- eerst de test-features die gebruikt zijn om te kijken of de Json Import werkt
- daarna de target features, die beginnen met 'y_'
- daarna de overige features, die beginnen met 'x_'   
    
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from path_finding import iter_csv_chunks
from local_time import local_day_labels
from reading_difficult_json import simple_Player_Profile_Bank_Account_json_iterator, simple_RG_Class_Value_from_FLAG_RG_CLASS_json_iterator 
from mapping_helpers import build_txid_to_player_map_ram, haal_uit_bank_json_iterator
from reading_difficult_json import iter_limit_values, iter_transaction_ids_from_Game_Transactions, iter_part_ids_from_Bet_Parts, iter_player_profile_ids_from_Bet_Transactions, get_list_of_response_ids_from_Responses_list

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


######################################################################
####################### Test_features ################################
######################################################################

# ------------------------------
# Feature 1: transaction_amount_sum
# ------------------------------
def transaction_amount_sum(
    tables: Dict[str, List[Path]],
    *,
    chunksize: int = 200_000,
    log_path: Path | None = None,   # accepted for uniform signature
    verbose: bool = False,
) -> pd.DataFrame:
    from path_finding import iter_csv_chunks

    pat_paths = tables.get("WOK_Player_Account_Transaction")
    if not pat_paths:
        raise FileNotFoundError("No Player Account Transaction(WOK_Player_Account_Transaction) paths found.")
    else:
        if verbose:
            print(f"Found {len(pat_paths)} Player Account Transactionfiles.")

    logger = _setup_feature_logger(
        (Path(pat_paths[0]).parent / "feature_sum.log") if log_path is None else log_path,
        "sum"
    )
    logger.info("▶ START feature 'sum' (streaming)")
    logger.info(f"pat_files={len(pat_paths)} | chunksize={chunksize}")

    totals: Dict[str, float] = {}
    usecols = ["Player_Profile_ID", "Transaction_Amount"]

    for p in pat_paths:
        if verbose:
            logger.info(f"  ↪ transaction_amount_sum reading in chunks: {p.name} (chunksize={chunksize})")
        for chunk in iter_csv_chunks([p], usecols=usecols, chunksize=chunksize, verbose=verbose):
            chunk = chunk.copy()
            chunk["Player_Profile_ID"] = chunk["Player_Profile_ID"].astype(str)
            chunk["Transaction_Amount"] = pd.to_numeric(chunk["Transaction_Amount"], errors="coerce").fillna(0.0)
            g = chunk.groupby("Player_Profile_ID")["Transaction_Amount"].sum()
            for personID, val in g.items():
                totals[personID] = totals.get(personID, 0.0) + float(val)

    if not totals:
        logger.info("No rows → returning empty feature frame.")
        return pd.DataFrame(columns=["Player_Profile_ID", "total_transaction_amount"])

    out = pd.DataFrame({
        "Player_Profile_ID": list(totals.keys()),
        "total_transaction_amount": list(totals.values()),
    })
    out["Player_Profile_ID"] = out["Player_Profile_ID"].astype(str)
    logger.info(f"Feature 'sum' ready: players={len(out):,}")
    return out


# ------------------------------
# Feature 2: transaction_amount_sum_in_casino_games
# ------------------------------
def transaction_amount_sum_in_games(
    tables: Dict[str, List[Path]],
    *,
    chunksize: int = 200_000,
    batch_out: int = 200_000,
    max_ram_entries: int = 10_000_000,   # soft guard
    verbose: bool = False,
    log_path: Path | None = None,
) -> pd.DataFrame:
    """
    Maakt eerst een map van alle Player Account Transaction ID's naar spelers.
    Daarna streamen we Player Account Transaction en tellen per speler het bedrag op
    van alle transacties die in de mapping voorkomen (dus in een casinospel zaten).
    """

    # resolve paths
    session_paths = tables.get("WOK_Game_Session") or tables.get("game_session")
    pat_paths     = tables.get("WOK_Player_Account_Transaction") or tables.get("player_account_transaction")
    if session_paths is None or pat_paths is None:
        raise FileNotFoundError("Need both WOK_Game_Session and WOK_Player_Account_Transaction.")

    logger = _setup_feature_logger(
        (Path(pat_paths[0]).parent / "feature_game_sum.log") if log_path is None else log_path,
        "game_sum"
    )
    logger.info("▶ START feature 'game_sum' (streaming)")
    logger.info(f"session_files={len(session_paths)} | pat_files={len(pat_paths)} | chunksize={chunksize}")

    # 1) mapping Transaction ID -> Player
    logger.info("Maakt een transaction ID→Player mapping, zodat we zaken kunnen optellen over chunks")
    map_van_alle_transaction_id_naar_persoon = build_txid_to_player_map_ram(
        session_paths,
        chunksize=chunksize,
        batch_out=batch_out,
        verbose=verbose,
        logger=logger,
    )
    logger.info(f"Mapping built: Aantal Transaction ID's={len(map_van_alle_transaction_id_naar_persoon):,}")

    # 2) stream Player Account Transaction and sum per player for hits
    # Woordenboek: per speler het totale bedrag (float)
    totaal_per_speler: Dict[str, float] = {}

    # Tellers voor rapportage/controle
    totaal_rijen = 0           # totaal aantal ingelezen rijen (over alle chunks)
    gematchte_rijen = 0        # aantal rijen waarvan de Transaction_ID in de mapping zit én een speler heeft
    gematchte_som = 0.0        # som van alle bedragen die aan spelers zijn toegewezen

    # Minimale kolommen die we nodig hebben uit PAT
    te_gebruiken_kolommen = ["Transaction_ID", "Transaction_Amount"]

    for pad in pat_paths:
        if verbose:
            print(f"  ↪ casino-feature: lees in chunks uit {pad.name} (chunksize={chunksize})")

        # Stream de CSV in stukken (chunks) om geheugen te sparen
        for chunk_nr, deel in enumerate(
            iter_csv_chunks([pad], usecols=te_gebruiken_kolommen, chunksize=chunksize, verbose=verbose),
            start=1
        ):
            # Basis boekhouding
            aantal_rijen_in_deel = len(deel)
            totaal_rijen += aantal_rijen_in_deel

            # Werk op een kopie (voorkomt SettingWithCopyWarnings etc.)
            deel = deel.copy()

            # Zorg dat Transaction_ID als string wordt behandeld (robuster bij mixed types)
            deel["Transaction_ID"] = deel["Transaction_ID"].astype(str)

            # Transaction_Amount veilig naar numeriek; ongeldige waarden → 0.0
            bedragen = pd.to_numeric(deel["Transaction_Amount"], errors="coerce").fillna(0.0)

            # Voor snelle lookups
            transactie_ids = deel["Transaction_ID"].values

            # Bepaal per rij of de Transaction_ID in de sessie→speler mapping zit
            hit_masker = [tx in map_van_alle_transaction_id_naar_persoon for tx in transactie_ids]
            aantal_hits = sum(1 for h in hit_masker if h)

            # Alleen iets te doen als er matches zijn
            if aantal_hits:
                # Koppel per match het bedrag aan de bijbehorende speler
                for transactie_id, bedrag, is_hit in zip(transactie_ids, bedragen.values, hit_masker):
                    if not is_hit:
                        continue
                    speler_id = map_van_alle_transaction_id_naar_persoon.get(transactie_id)
                    if speler_id is None:
                        # Geen speler bekend voor deze transactie → overslaan
                        continue

                    # Tel bedrag op bij deze speler
                    totaal_per_speler[speler_id] = totaal_per_speler.get(speler_id, 0.0) + float(bedrag)
                    gematchte_rijen += 1
                    gematchte_som += float(bedrag)

            # Log naar bestand (altijd)
            logger.info(
                f"chunk#{chunk_nr:04d} rows={aantal_rijen_in_deel:,} "
                f"hits={aantal_hits:,} matched_rows_total={gematchte_rijen:,} "
                f"matched_sum_total={gematchte_som:,.2f}"
            )

            # Extra log naar console wanneer verbose aan staat (meer mensentaal)
            if verbose:
                print(
                    f"    • chunk {chunk_nr:04d}: {aantal_rijen_in_deel:,} rijen | "
                    f"{aantal_hits:,} hits | totaal gematchte rijen: {gematchte_rijen:,} | "
                    f"totaal som: {gematchte_som:,.2f}"
                )
        logger.info(f"DONE Player Account Transactionstreaming: total_rows={totaal_rijen:,}, matched_rows={gematchte_rijen:,}, matched_sum={gematchte_som:,.2f}")

    if not totaal_per_speler:
        logger.info("No matches → returning empty feature frame.")
        return pd.DataFrame(columns=["Player_Profile_ID", "transaction_amount_sum_in_casino_games"])

    out = pd.DataFrame({
        "Player_Profile_ID": list(totaal_per_speler.keys()),
        "transaction_amount_sum_in_casino_games": list(totaal_per_speler.values()),
    })
    out["Player_Profile_ID"] = out["Player_Profile_ID"].astype(str)
    logger.info(f"Feature 'game_sum' ready: players={len(out):,}")
    return out


# ------------------------------
# Feature 3 (placeholder): end_accel_score_stream
# ------------------------------
def end_accel_score_stream(
    tables: Dict[str, List[Path]],
    *,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Placeholder zodat Scenario_3 niet crasht.
    We leveren voor alle spelers een score 0.0 op basis van PAT-keys.
    """
    from path_finding import iter_csv_chunks

    pat_paths = tables.get("WOK_Player_Account_Transaction")
    if not pat_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "end_accel_score_mean"])

    logger = _setup_feature_logger(
        (Path(pat_paths[0]).parent / "feature_time_eas.log") if log_path is None else log_path,
        "time_eas"
    )
    logger.info("▶ START feature 'time_eas' (placeholder)")
    logger.info(f"pat_files={len(pat_paths)} | chunksize={chunksize}")

    players: set[str] = set()
    for p in pat_paths:
        for chunk in iter_csv_chunks([p], usecols=["Player_Profile_ID"], chunksize=chunksize, verbose=verbose):
            players.update(map(str, chunk["Player_Profile_ID"].dropna().astype(str).unique()))

    if not players:
        return pd.DataFrame(columns=["Player_Profile_ID", "end_accel_score_mean"])

    out = pd.DataFrame({"Player_Profile_ID": sorted(players)})
    out["end_accel_score_mean"] = 0.0
    logger.info(f"Feature 'time_eas' placeholder ready: players={len(out):,}")
    return out

# ------------------------------
# Feature 4: Aantal bank accounts
# ------------------------------    
def nr_of_bank_accounts(
    tables: Dict[str, List[Path]],
    *,
    chunksize: int = 200_000,
    batch_out: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Haal alle bank IDs uit de JSON kolom Player_Profile_Bank_Account.
    Geef een dataframe terug wat per Player_Profile_ID het aantal bank accounts bevat.
    Player_Profile_Bank_Account:
    [{"Bank_Account_ID":"13zsdfa","Bank_Account_Datetime":"2025-06-19T13:10:39Z","Bank_Account_Active":"true"},
    {"Bank_Account_ID":"21t1tb","Bank_Account_Datetime":"2025-06-20T11:05:21Z","Bank_Account_Active":"false"}] 
    """
    from mapping_helpers import maak_een_list_van_spelers

    player_profile_paths = tables.get("WOK_Player_Profile")
    if player_profile_paths is None:
        raise FileNotFoundError("Need WOK_Player_Profile.")

    logger = _setup_feature_logger(
        (Path(player_profile_paths[0]).parent / "feature_nr_of_bank_accounts.log") if log_path is None else log_path,
        "player_profile_paths"
    )
    logger.info("▶ START feature 'player_profile_paths' (streaming)")
    logger.info(f"session_files={len(player_profile_paths)} | chunksize={chunksize}")
    need_cols = ["Player_Profile_ID","Player_Profile_Bank_Account"]
    dict_bank_ids_per_speler = {}
    for df in iter_csv_chunks(paths=player_profile_paths,
        usecols=need_cols,
        chunksize= chunksize,
        verbose = verbose,
    ):
        # print('dit is de df ---------', df)
        # hier alvast de kolomindexen bepalen voor later in de rij, dat is efficiënter
        col_idx_id = df.columns.get_loc(need_cols[0])
        col_idx_json = df.columns.get_loc(need_cols[1])
        for rij in range(len(df)):
            # print('dit is de rij -------', df.iloc[rij])
            personID = df.iat[rij, col_idx_id]
            Player_Profile_Bank_Account_json = df.iat[rij, col_idx_json]
            if personID not in dict_bank_ids_per_speler:
                # print('NIET ERIN')
                dict_bank_ids_per_speler[personID] = set()
                # print('dict -----', dict_bank_ids_per_speler)
            # print('bestaande_bank_accounts -----', dict_bank_ids_per_speler[personID])
            bestaande_bank_accounts = dict_bank_ids_per_speler[personID]
            # print('bestaande_bank_accounts -----', bestaande_bank_accounts)
            for bank_account_ID in simple_Player_Profile_Bank_Account_json_iterator(json_obj = Player_Profile_Bank_Account_json, needed_vars = ["Bank_Account_ID"]):
                # print('bank_account_ID -----', bank_account_ID)
                bestaande_bank_accounts.add(bank_account_ID)
                dict_bank_ids_per_speler[personID] = bestaande_bank_accounts
                # print ('dict -----', dict_bank_ids_per_speler)
    out = pd.DataFrame({
        "Player_Profile_ID": list(dict_bank_ids_per_speler.keys()),
        "number_of_bank_accounts": (dict_bank_ids_per_speler.values()),
    })
    out["number_of_bank_accounts"] = out["number_of_bank_accounts"].apply(lambda x: len(x) if isinstance(x, set) else 0)
    return out

# ------------------------------
# Feature 5: Aantal risicocategoriëen
# ------------------------------    
def nr_of_risk_classes(
    tables: Dict[str, List[Path]],
    *,
    chunksize: int = 200_000,
    batch_out: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Haal alle de JSON kolom Flag_RG_Class uit de WOK_Player_Flags en tel de unieke waardes per Player_Profile_ID.
    Geef een dataframe terug wat per Player_Profile_ID het aantal risicocategorieen bevat.
    Voorbeeld:
    "{""RG_Class_Value"": ""NO_RISK_ASSIGNED"", ""RG_Class_Datetime"": ""2017-02-28T19:17:28Z""}"
    """
    player_flags_paths = tables.get("WOK_Player_Flags")
    if player_flags_paths is None:
        raise FileNotFoundError("Need WOK_Player_Flags.")

    logger = _setup_feature_logger(
        (Path(player_flags_paths[0]).parent / "feature_nr_of_risk_classes.log") if log_path is None else log_path,
        "player_flags_paths"
    )
    logger.info("▶ START feature 'player_flags_paths' (streaming)")
    logger.info(f"session_files={len(player_flags_paths)} | chunksize={chunksize}")
    need_cols = ["Player_Profile_ID","Flag_RG_Class"]
    dict_flags_ids_per_speler = {}
    for df in iter_csv_chunks(paths=player_flags_paths,
        usecols=need_cols,
        chunksize= chunksize,
        verbose = verbose,
    ):
        # print('dit is de df ---------', df)
        # hier alvast de kolomindexen bepalen voor later in de rij, dat is efficiënter
        col_idx_id = df.columns.get_loc(need_cols[0])
        col_idx_json = df.columns.get_loc(need_cols[1])
        for rij in range(len(df)):
            # print('dit is de rij -------', df.iloc[rij])
            personID = df.iat[rij, col_idx_id]
            Flag_RG_Class_json = df.iat[rij, col_idx_json]
            if personID not in dict_flags_ids_per_speler:
                # print('NIET ERIN')
                dict_flags_ids_per_speler[personID] = set()
                # print('dict -----', dict_bank_ids_per_speler)
            # print('bestaande_bank_accounts -----', dict_bank_ids_per_speler[personID])
            bestaande_flags = dict_flags_ids_per_speler[personID]
            # print('bestaande_bank_accounts -----', bestaande_bank_accounts)
            for RG_Class_Value in simple_RG_Class_Value_from_FLAG_RG_CLASS_json_iterator(json_obj = Flag_RG_Class_json, needed_vars = ["Flag_RG_Class"]):
                if RG_Class_Value is None:
                    logger.info(f"Geen RG_Class_Value voor speler {personID} in rij {rij}")
                    continue
                bestaande_flags.add(RG_Class_Value)
                dict_flags_ids_per_speler[personID] = bestaande_flags
    out = pd.DataFrame({
        "Player_Profile_ID": list(dict_flags_ids_per_speler.keys()),
        "number_of_risk_classes": (dict_flags_ids_per_speler.values()),
    })
    out["number_of_risk_classes"] = out["number_of_risk_classes"].apply(lambda x: len(x) if isinstance(x, set) else 0)
    return out

# ------------------------------
# Features 6, 7, 8, 9, 10: Limit index: laatste deposit amount limit+ time window, laatste login duration limit + time window, laatste balance limit
# ------------------------------  

def latest_limits(
    tables: Dict[str, List[Path]],
    *,
    chunksize: int = 200_000,
    batch_out: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Haal drie waarden JSON kolommen (Deposit_Amount uit Limit_Deposit, Login_Time_Window uit Limit_Login en Balance_Amount uit Limit_Balance) uit de WOK_Player_Limits en pak voor elke Player_Profile_ID de laatste waarde.
    Geef een dataframe terug wat per Player_Profile_ID deze drie waarden bevat.
    Voorbeelden van input:
    "[{""Deposit_Request_Datetime"": ""1-1-1T1:1:1Z"", ""Deposit_Start_Datetime"": ""1-1-1T1:1:1Z"", ""Deposit_Amount"": 1, ""Deposit_Time_Window"": ""Week""}]"
    "[{""Login_Request_Datetime"": ""1-1-1T1:1:1Z"", ""Login_Start_Datetime"": ""1-1-1T1:1:1Z"", ""Login_Duration"": 1.1, ""Login_Time_Window"": ""Week""}]"
    "[{""Balance_Request_Datetime"": ""1-1-1T1:1:1Z"", ""Balance_Start_Datetime"": ""1-1-1T1:1:1Z"", ""Balance_Amount"": 1}]"
    """
    player_limits_paths = tables.get("WOK_Player_Limits")
    if player_limits_paths is None:
        raise FileNotFoundError("Need WOK_Player_Limits.")

    logger = _setup_feature_logger(
        (Path(player_limits_paths[0]).parent / "feature_latest_limits.log") if log_path is None else log_path,
        "latest_limits"
    )
    logger.info("▶ START feature 'latest_limits' (streaming)")
    logger.info(f"limit_files={len(player_limits_paths)} | chunksize={chunksize}")

    need_cols = ["Player_Profile_ID", "Limit_Deposit", "Limit_Login", "Limit_Balance"]

    DEPOSIT_INDEX, LOGIN_INDEX, BALANCE_INDEX = 0, 1, 2
    limit_dict_per_speler: Dict[str, list] = {}
    # hier alvast de kolomindexen bepalen voor later in de rij, dat is efficiënter
    for df in iter_csv_chunks(paths=player_limits_paths, usecols=need_cols, chunksize=chunksize, verbose=verbose):
        col_idx_id  = df.columns.get_loc("Player_Profile_ID")
        col_idx_dep = df.columns.get_loc("Limit_Deposit")
        col_idx_log = df.columns.get_loc("Limit_Login")
        col_idx_bal = df.columns.get_loc("Limit_Balance")

        for rij in range(len(df)):
            personID = df.iat[rij, col_idx_id]
            if personID not in limit_dict_per_speler:
                limit_dict_per_speler[personID] = [None, None, None]

            for timestamp, amount, window in iter_limit_values(df.iat[rij, col_idx_dep], type_="deposit"):
                prev = limit_dict_per_speler[personID][DEPOSIT_INDEX]
                if timestamp and (prev is None or (prev[0] and timestamp > prev[0])):
                    limit_dict_per_speler[personID][DEPOSIT_INDEX] = (timestamp, amount, window)

            for timestamp, duration, window in iter_limit_values(df.iat[rij, col_idx_log], type_="login"):
                prev = limit_dict_per_speler[personID][LOGIN_INDEX]
                if timestamp and (prev is None or (prev[0] and timestamp > prev[0])):
                    limit_dict_per_speler[personID][LOGIN_INDEX] = (timestamp, duration, window)

            for timestamp, amount, _ in iter_limit_values(df.iat[rij, col_idx_bal], type_="balance"):
                prev = limit_dict_per_speler[personID][BALANCE_INDEX]
                if timestamp and (prev is None or (prev[0] and timestamp > prev[0])):
                    limit_dict_per_speler[personID][BALANCE_INDEX] = (timestamp, amount, None)

    rows = []
    for personID, triple in limit_dict_per_speler.items():
        dep, logn, bal = triple
        rows.append({
            "Player_Profile_ID": personID,
            "latest_deposit_amount_limit": dep[1] if dep else None,
            "latest_deposit_time_window_limit": dep[2] if dep else None,
            "latest_login_duration_limit": logn[1] if logn else None,
            "latest_login_time_window_limit": logn[2] if logn else None,
            "latest_balance_amount_limit": bal[1] if bal else None,
        })

    out = pd.DataFrame(rows)
    logger.info(f"Feature 'latest_limits' klaar: players={len(out):,}")
    return out

# ------------------------------
# Features 11: Avg number of transactions within a Game Session
# ------------------------------

def avg_nr_of_game_transactions(
    tables: Dict[str, List[Path]],
    *,
    usecols: Optional[List[str]] = None,
    chunksize: int = 200_000,
    batch_out: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Elke game session heeft meerdere game sessions 
    die elk weer telkens één transaction ID en één 
    player profile ID hebben.

    Momenteel is het format van de kolom Game_Transactions: 
    "{""Game_Transaction"":[{""Player_Profile_ID"":""12345"",""Transaction_ID"":""tx67890""},
    {""Player_Profile_ID"":""12345"",""Transaction_ID"":""tx67891""}]}"
    Dit is dus iets anders dan de XML-structuur, want daar heb je meerdere Game_Transaction blokken.
    Maar het werkt want wij hebben nu meerdere Player_Profile_ID/Transaction_ID paren in een list.
    - We tellen voor elke sessie het aantal transacties die bij de speler van die rij horen
      (op basis van Player_Profile_ID in de rij), en middelen dat over alle sessies
      van die speler.

    Output
    ------
    DataFrame met kolommen:
    - Player_Profile_ID
    - avg_number_of_game_transactions  (float)
    """
    game_session_paths = tables.get("WOK_Game_Session")
    if game_session_paths is None:
        raise FileNotFoundError("Need WOK_Game_Session.")

    logger = _setup_feature_logger(
        (Path(game_session_paths[0]).parent / "feature_avg_nr_of_game_transactions.log") if log_path is None else log_path,
        "avg_nr_of_game_transactions",
    )
    logger.info("▶ START feature 'avg_nr_of_game_transactions' (streaming)")
    logger.info(f"session_files={len(game_session_paths)} | chunksize={chunksize}")

    # Accumulators per speler:
    # alle_transacties_per_speler[speler] = alle transacties over al zijn/haar sessies
    # aantal_sessies_per_speler[speler] = aantal sessies van die speler
    alle_transacties_per_speler: Dict[str, int] = {}
    aantal_sessies_per_speler: Dict[str, int] = {}

    # Stream de CSV's in chunks om geheugen te sparen
    for df in iter_csv_chunks(
        paths=game_session_paths,
        usecols=usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        # Kolomindexen vooraf bepalen is sneller in de rij-lus
        kol_idx_json   = df.columns.get_loc("Game_Transactions")

        # Rij-voor-rij verwerken (snel met .iat)
        for rij_index in range(len(df)):
            json_veld = df.iat[rij_index, kol_idx_json]

            # Tel transacties in deze sessie voor deze specifieke speler
            # (Filter in de parser op Player_Profile_ID van de rij.)
            aantal_in_deze_sessie = 0
            for speler_id, _transaction_id in iter_transaction_ids_from_Game_Transactions(
                json_veld
            ):
                if verbose: 
                    print('speler id -----', speler_id)
                aantal_in_deze_sessie += 1

                # Initialiseerslag als speler nog niet gezien
                if speler_id not in alle_transacties_per_speler:
                    print('NIET ERIN')
                    alle_transacties_per_speler[speler_id] = 0
                    aantal_sessies_per_speler[speler_id] = 0

                # Accumuleren voor gemiddelde
                alle_transacties_per_speler[speler_id] += aantal_in_deze_sessie
                if verbose:
                    print('alle_transacties_per_speler -----', alle_transacties_per_speler)
                aantal_sessies_per_speler[speler_id] += 1
                if verbose:
                    print('aantal_sessies_per_speler -----', aantal_sessies_per_speler)

    # Bouw het resultaat-DataFrame
    spelers = []
    gemiddelden = []
    for speler_id, som_transacties in alle_transacties_per_speler.items():
        sessies = aantal_sessies_per_speler.get(speler_id, 0)
        gemiddelde = (som_transacties / sessies) if sessies else 0.0
        spelers.append(speler_id)
        gemiddelden.append(gemiddelde)

    resultaat = pd.DataFrame({
        "Player_Profile_ID": spelers,
        "avg_number_of_game_transactions": gemiddelden,
    })

    logger.info(f"Feature 'avg_nr_of_game_transactions' klaar: players={len(resultaat):,}")
    return resultaat

# ------------------------------
# Features 12: gemiddeld aantal bet parts in een bet
# ------------------------------

def avg_nr_of_bet_parts_per_bet(
    tables: Dict[str, List[Path]],
    *,
    usecols: Optional[List[str]] = None,
    chunksize: int = 200_000,
    batch_out: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Elke WOK_Bet heeft meerdere Bet_Parts met een eigen ID

    Momenteel is het format van de kolom Bet_Parts: 
    "{""Part"":[{""Part_ID"":""first"",""Part_Event"":""bla"",""Part_Odds"":""11.00"",""Part_Sport"":""FOOTBALL"",""Part_Live"":""false"",""Part_Bank"":""false"",""Part_Match_Datetime"":""2025-07-11T15:30:00Z"",""Part_Prognosis_Result_Type"":""MATCH ODDS"",""Part_Prognosis_Value"":""|Draw|"",""Part_Stake"":""10.00""},
    {""Part_ID"":""second"",""Part_Event"":""bla2"",""Part_Odds"":""14.00"",""Part_Sport"":""FOOTBALL"",""Part_Live"":""false"",""Part_Bank"":""false"",""Part_Match_Datetime"":""2025-07-11T14:00:00Z"",""Part_Prognosis_Result_Type"":""MATCH ODDS"",""Part_Prognosis_Value"":""|Draw|"",""Part_Stake"":""10.00""}]}"
    Overigens lijkt het er in de documentatie op dat er binnen Part ook meerdere Part_ID's kunnen zitten,
    maar uit de XSD blijkt dat Part_ID enkelvoudig is. We gaan er dus vanuit dat elk Part maar één Part_ID heeft.

    De ID halen we uit een andere kolom, namelijk Bet_Transactions, die de volgende structuur heeft:
    "{""Bet_Transaction"":[{""Player_Profile_ID"":""291463a9-5ab0-f2d8-9c44-6c9d499aa0a0"",""Transaction_ID"":""20d231e6-38d1-4d0d-80c7-21186526e5e0""}]}"

    - We halen eerst van elke rij de Player_Profile_ID op uit Bet_Transactions. 
    - In het uitzonderlijke geval dat dit er meer dan 1 zijn (een bet die met meerdere mensen is afgesloten) geven we een waarschuwing en pakken we de eerste.

    Output
    ------
    DataFrame met kolommen:
    - Player_Profile_ID
    - avg_number_of_bet_parts (float)
    - avg_stake_per_bet (float)
    """
    bet_paths = tables.get("WOK_Bet")
    if bet_paths is None:
        raise FileNotFoundError("Need WOK_Bet.")

    logger = _setup_feature_logger(
        (Path(bet_paths[0]).parent / "feature_avg_nr_of_bet_parts.log") if log_path is None else log_path,
        "avg_nr_of_bet_parts",
    )
    logger.info("▶ START feature 'avg_nr_of_bet_parts' (streaming)")
    logger.info(f"bet_files={len(bet_paths)} | chunksize={chunksize}")

    # Accumulators per speler:
    # totaal_parts_per_speler[speler] = totaal aantal parts over al zijn/haar bets
    # aantal_bets_per_speler[speler]   = aantal bets van die speler
    totaal_parts_per_speler: Dict[str, int] = {}
    aantal_bets_per_speler: Dict[str, int] = {}

    # Stream de CSV's in chunks om geheugen te sparen
    for df in iter_csv_chunks(
        paths=bet_paths,
        usecols=usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        # Kolomindexen vooraf bepalen is sneller in de rij-lus
        kol_idx_tx    = df.columns.get_loc("Bet_Transactions")
        kol_idx_parts = df.columns.get_loc("Bet_Parts")

        # Rij-voor-rij verwerken (snel met .iat)
        for rij_index in range(len(df)):
            json_veld_tx    = df.iat[rij_index, kol_idx_tx]
            json_veld_parts = df.iat[rij_index, kol_idx_parts]

            # Haal Player_Profile_ID(s) uit Bet_Transactions
            speler_ids = list(iter_player_profile_ids_from_Bet_Transactions(json_veld_tx))
            if not speler_ids:
                # Geen speler-ID gevonden; sla deze rij over
                continue
            if len(speler_ids) > 1:
                logger.warning("Bet_Transactions bevat meerdere Player_Profile_IDs; eerste genomen.")
            speler_id = speler_ids[0]

            # Tel parts in deze bet
            aantal_parts_in_deze_bet = 0
            for _part_id, _part_obj in iter_part_ids_from_Bet_Parts(json_veld_parts):
                aantal_parts_in_deze_bet += 1

            # Initialiseerslag als speler nog niet gezien
            if speler_id not in totaal_parts_per_speler:
                totaal_parts_per_speler[speler_id] = 0
                aantal_bets_per_speler[speler_id] = 0

            # Accumuleren voor gemiddelde
            totaal_parts_per_speler[speler_id] += aantal_parts_in_deze_bet
            aantal_bets_per_speler[speler_id] += 1

    # Bouw het resultaat-DataFrame
    spelers = []
    gemiddelden = []
    for speler_id, som_parts in totaal_parts_per_speler.items():
        bets = aantal_bets_per_speler.get(speler_id, 0)
        gemiddelde = (som_parts / bets) if bets else 0.0
        spelers.append(speler_id)
        gemiddelden.append(gemiddelde)

    resultaat = pd.DataFrame({
        "Player_Profile_ID": spelers,
        "avg_number_of_bet_parts": gemiddelden,
    })

    logger.info(f"Feature 'avg_nr_of_bet_parts' klaar: players={len(resultaat):,}")
    return resultaat

# ------------------------------
# Features 13, 14: aantal klachten per persoon, gemiddeld aantal responses per klacht
# ------------------------------

def nr_of_complaints_and_variance_in_nr_responses_per_complaint(
    tables: Dict[str, List[Path]],
    *,
    usecols: List[str] = ["Complaint_ID", "Complaint_Player_ID", "Responses"],
    chunksize: int = 200_000,
    batch_out: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Elke WOK_Complaint heeft een optionele 'Complaint_Player_ID' kolom en een 'Responses' kolom.
    In de oorspronkelijke json voorbeeldstructuur stond een foutje, namelijk:

    "{""Response"":{
    ""Response_ID"":""start1234"",
    ""Response_Type"":""Complaint Closing"",
    ""Response_Description"":""Klantgegevens aangepast"",
    ""Response_Datetime"":""2025-07-14T10:38:06Z""
    }}"

    Maar dat is mogelijk problematisch, want er kunnen meerdere responses per complaint zijn. 
    Daarom passen we dit aan in het cleaning process.
    
    De Responses data ziet er als volgt uit:

    "[{""Response"":[{
    ""Response_ID"":""start1234"",
    ""Response_Type"":""Complaint Closing"",
    ""Response_Description"":""Klantgegevens aangepast"",
    ""Response_Datetime"":""2025-07-14T10:38:06Z""
    }]}]"

    We tellen per speler het aantal klachten waarbij de Complaint_Player_ID overeenkomt met de speler.
    Ook bepalen we de variance in het aantal responses per klacht voor die speler.
    """
    complaint_paths = tables.get("WOK_Complaint")
    if complaint_paths is None:
        raise FileNotFoundError("Need WOK_Complaint.")

    logger = _setup_feature_logger(
        (Path(complaint_paths[0]).parent / "feature_number_of_complaints.log") if log_path is None else log_path,
        "number_of_complaints",
    )
    logger.info("▶ START feature 'number_of_complaints' (streaming)")
    logger.info(f"complaint_files={len(complaint_paths)} | chunksize={chunksize}")

    # Accumulator per speler:
    # klachten_per_speler[speler] = list van klacht IDs
    klachten_en_aantal_responses_per_speler = {}

    # Stream de CSV's in chunks om geheugen te sparen
    for df in iter_csv_chunks(
        paths=complaint_paths,
        usecols=usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        # Kolomindexen vooraf bepalen is sneller in de rij-lus
        kol_idx_player   = df.columns.get_loc("Complaint_Player_ID")
        kol_idx_responses = df.columns.get_loc("Responses")
        kol_idx_complaint_id = df.columns.get_loc("Complaint_ID")

        # Rij-voor-rij verwerken (snel met .iat)
        for rij_index in range(len(df)):
            if verbose:
                print(f'Rij-Index: {rij_index}  / len df: {len(df)}')
                print('--- Testing for row:', df.loc[[rij_index]])
            speler_id = df.iat[rij_index, kol_idx_player]
            json_veld = df.iat[rij_index, kol_idx_responses]
            complaint_id = df.iat[rij_index, kol_idx_complaint_id]

            new_list_of_response_ids = get_list_of_response_ids_from_Responses_list(json_veld)

            # Nu dus geen iterator nodig want iedere complaint telt exact één keer voor de speler.

            # for _response_id in iter_responses_from_Complaints(json_veld):
            #     # We gebruiken de helper om dezelfde JSON-verwachting te hanteren als elders.
            #     # Geen verdere actie nodig voor de telling.
            #     break
            if new_list_of_response_ids is None:
                # Geen response_id gevonden; sla deze rij over
                if verbose:
                    print(f"Geen response_ids voor klacht {complaint_id} van speler {speler_id}")
            elif speler_id is None:
                # Geen speler_id gevonden; sla deze rij over
                if verbose:
                    print(f"Geen speler_id voor klacht {complaint_id} met responses {new_list_of_response_ids}")
            else:
                pass
            
            # print()
            # print("oude klachten en aantal responses") 
            # print(klachten_en_aantal_responses_per_speler)
            # print()
            # 1. Als de speler nog niet is gezien: voeg toe aan dictionary, de key is de complaint ID
            if speler_id not in klachten_en_aantal_responses_per_speler:
                # Eerste klacht voor speler: zorg dat we nooit None opslaan, maar altijd een lijst
                if new_list_of_response_ids is not None:
                    initial_list = new_list_of_response_ids
                else:
                    initial_list = []
                klachten_en_aantal_responses_per_speler[speler_id] = [{complaint_id: initial_list}]
            
            # 2. Als de speler de klacht nog niet heeft als key: voeg klacht & lege lijst toe aan lijst, want krijgt ook klachten zonder resonse
            elif not any(complaint_id in dic for dic in klachten_en_aantal_responses_per_speler[speler_id]):
                # print(f"Nieuwe klacht {complaint_id} voor speler {speler_id}")
                # als het een onbekende klacht is met response dan toevoegen
                if new_list_of_response_ids is not None:
                    # print(f"Nieuwe klacht {complaint_id} voor speler {speler_id} met response {response_id}")
                    klachten_en_aantal_responses_per_speler[speler_id].append({complaint_id: new_list_of_response_ids})
                # als het een onbekende klacht is zonder response dan toevoegen met lege lijst
                else:
                    # print(f"Nieuwe klacht {complaint_id} voor speler {speler_id} zonder response")
                    klachten_en_aantal_responses_per_speler[speler_id].append({complaint_id: []})                
                klachten_en_aantal_responses_per_speler[speler_id] = klachten_en_aantal_responses_per_speler[speler_id]

            # 3. Als de speler de klacht al heeft maar de response_id nog niet: voeg toe aan lijst
            else:
                # print(f"Bestaande klacht {complaint_id} voor speler {speler_id}, response {response_id}")
                # print('klachten_en_aantal_responses_per_speler[speler_id] -------', klachten_en_aantal_responses_per_speler[speler_id])
                for complaint_id_dict in klachten_en_aantal_responses_per_speler[speler_id]:
                    # print('complaint_id_dict --_-_--', complaint_id_dict)
                    if complaint_id in complaint_id_dict and new_list_of_response_ids is not None:
                        # pak de bestaande lijst, als die er is, zo niet, maak een lege lijst
                        existing = complaint_id_dict.get(complaint_id) or []
                        merged = list(set(existing) | set(new_list_of_response_ids))
                        complaint_id_dict[complaint_id] = merged

            # (4). Als de response_id al bestaat voor deze klacht, dan telt die al mee, dus niets doen
            # print()
            # print("NIEUWE klachten en aantal responses") 
            # print(klachten_en_aantal_responses_per_speler)
            # print()
    # Bouw het resultaat-DataFrame
    spelers = []
    aantal_klachten = []
    variance_per_speler = []
    for speler_id, klachten_id_dic in klachten_en_aantal_responses_per_speler.items():
        spelers.append(speler_id)
        aantal_klachten.append(len(klachten_id_dic))
        aantallen_response_ids = []
        for klacht_dic in klachten_id_dic:
            for complaint_id, response_id_list in klacht_dic.items():
                if response_id_list is None:
                    aantallen_response_ids.append(0)
                else:
                    aantallen_response_ids.append(len(response_id_list))
                if verbose:
                    print(f'aantallen_response_ids {speler_id}: ', aantallen_response_ids)
        if len(aantallen_response_ids) > 1:
            gemiddelde = sum(aantallen_response_ids) / len(aantallen_response_ids)
            variance = sum((x - gemiddelde) ** 2 for x in aantallen_response_ids) / (len(aantallen_response_ids) - 1)
            variance_per_speler.append(variance)
            if verbose:
                print(f"Speler {speler_id} heeft variance in aantal responses per complaint: {variance}")
        else:
            variance_per_speler.append(0.0)
            if verbose:
                print(f"Speler {speler_id} heeft geen variance in aantal responses per complaint (slechts 1 klacht)")

    resultaat = pd.DataFrame({
        "Player_Profile_ID": spelers,
        "number_of_complaints": aantal_klachten,
        "variance_in_number_of_responses_per_complaint": variance_per_speler,
    })

    logger.info(f"Feature 'number_of_complaints' klaar: players={len(resultaat):,}")
    return resultaat


######################################################################
####################### Target features ##############################
######################################################################


# ------------------------------
# Helper functies voor flexibele tijdsparameters
# ------------------------------

def parse_ddmmyyyy_to_timestamp(date_str: str) -> pd.Timestamp:
    """
    Converteer 'DDMMYYYY' string naar pd.Timestamp.

    Parameters
    ----------
    date_str : str
        Datum in formaat 'DDMMYYYY', bijvoorbeeld '01012024' voor 1 januari 2024.

    Returns
    -------
    pd.Timestamp
        Pandas Timestamp object.

    Examples
    --------
    >>> parse_ddmmyyyy_to_timestamp("01012024")
    Timestamp('2024-01-01 00:00:00')
    >>> parse_ddmmyyyy_to_timestamp("31122025")
    Timestamp('2025-12-31 00:00:00')
    """
    day = date_str[:2]
    month = date_str[2:4]
    year = date_str[4:]
    return pd.Timestamp(f"{year}-{month}-{day}")


def generate_y_column_name(y_tijdspad: List[str]) -> str:
    """
    Genereer dynamische Y-target kolomnaam op basis van tijdspad.

    Parameters
    ----------
    y_tijdspad : List[str]
        Lijst met [start_datum, eind_datum] in formaat 'DDMMYYYY'.

    Returns
    -------
    str
        Kolomnaam zoals 'y_self_exclusion_20241219_20241221'.

    Examples
    --------
    >>> generate_y_column_name(["01012024", "19012024"])
    'y_self_exclusion_20240101_20240119'
    >>> generate_y_column_name(["19122024", "21122024"])
    'y_self_exclusion_20241219_20241221'
    """
    start = y_tijdspad[0]  # DDMMYYYY
    end = y_tijdspad[1]    # DDMMYYYY
    # Convert to YYYYMMDD format
    start_yyyymmdd = start[4:] + start[2:4] + start[:2]
    end_yyyymmdd = end[4:] + end[2:4] + end[:2]
    return f"y_self_exclusion_{start_yyyymmdd}_{end_yyyymmdd}"


def generate_x_prefix(x_tijdspad: List[str]) -> str:
    """
    Genereer dynamische X-feature prefix op basis van tijdspad.

    Parameters
    ----------
    x_tijdspad : List[str]
        Lijst met [start_datum, eind_datum] in formaat 'DDMMYYYY'.

    Returns
    -------
    str
        Prefix zoals 'x_20240101_20240119_' (eindigt met underscore).

    Examples
    --------
    >>> generate_x_prefix(["01012024", "31012024"])
    'x_20240101_20240131_'
    >>> generate_x_prefix(["01042025", "01062025"])
    'x_20250401_20250601_'
    """
    start = x_tijdspad[0]  # DDMMYYYY
    end = x_tijdspad[1]    # DDMMYYYY
    # Convert to YYYYMMDD format
    start_yyyymmdd = start[4:] + start[2:4] + start[:2]
    end_yyyymmdd = end[4:] + end[2:4] + end[:2]
    return f"x_{start_yyyymmdd}_{end_yyyymmdd}_"


# ------------------------------
# Target-feature A: heeft de persoon zich ooit uitgeschreven (gedurende de tijdsperiode van de data)?
# ------------------------------

def has_person_ever_requested_an_exclusion(
    tables: Dict[str, List[Path]],
    *,
    usecols: List[str] = ["Player_Profile_ID", "Player_Profile_Status", "Extraction_Date"],
    chunksize: int = 200_000,
    batch_out: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Player_Profile_Status kan de volgende waardes hebben:
        • ACTIVE: the player’s identity has been verified and his/her status is normal;
        • TRIAL: the verification of the player’s identity is still ongoing, a player can only
        log in to complete the registration. Transactions and gambling are not allowed
        until the status ACTIVE is assigned;10
        • SUSPENDED: the player has been suspended by the operator due to not accessing
        his account for a certain amount of time;
        • SUSPENDED_DEATH: the player has been suspended by the operator because
        he died;
        • BLOCKED: the player’s account has been blocked because the operator suspects
        fraudulent behaviour and/or the player does not meet the operator’s terms and
        conditions;
        • SELF_EXCLUDED_TEMP: the player has requested to be excluded from playing
        for a set period;
        • SELF_EXCLUDED_INDEF: the player has requested to be excluded from playing
        for an indefinite period;
        • OTHER: the player has a status that does not fit any of the other categories.

    We zullen per speler bepalen of hij/zij ooit de status SELF_EXCLUDED_TEMP of SELF_EXCLUDED_INDEF heeft gehad.
    Dit is een binaire target-feature: 
    1 = heeft zich ooit uitgeschreven,
    0 = heeft zich nooit uitgeschreven.

    We moeten echter ook een False categorie hebben, voor spelers die niet Active zijn.
    Zowel suspected death als vagere categorieen kunnen namelijk onder zowel self excluded als active horen, die moeten we verwijderen als dit de laatste stand van zaken is.
    """
    profile_paths = tables.get("WOK_Player_Profile")
    if not profile_paths:
        msg = "⚠️ WOK_Player_Profile niet beschikbaar – has_person_ever_requested_an_exclusion wordt leeg teruggegeven"
        print(msg)
        return pd.DataFrame(columns=["Player_Profile_ID", "y_has_person_ever_requested_an_exclusion"])

    logger = _setup_feature_logger(
        (Path(profile_paths[0]).parent / "feature_has_person_ever_requested_an_exclusion.log") if log_path is None else log_path,
        "has_person_ever_requested_an_exclusion",
    )
    logger.info("▶ START feature 'has_person_ever_requested_an_exclusion' (streaming)")
    logger.info(f"exclusion files={len(profile_paths)} | chunksize={chunksize}")

    # binaire status per speler:
    statussen_lijst = {}
    # aantal records per speler in WOK_Player_Profile
    aantal_rijen_per_speler: Dict[str, int] = {}

    # Stream de CSV's in chunks om geheugen te sparen
    for df in iter_csv_chunks(
        paths=profile_paths,
        usecols=usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        # Kolomindexen vooraf bepalen is sneller in de rij-lus
        kol_idx_player = df.columns.get_loc("Player_Profile_ID")
        kol_idx_player_profile_status = df.columns.get_loc("Player_Profile_Status")
        kol_idx_extr_date = df.columns.get_loc("Extraction_Date")

        # Rij-voor-rij verwerken (snel met .iat)
        for rij_index in range(len(df)):
            if verbose:
                print(f'Rij-Index: {rij_index}  / len df: {len(df)}')
                print('--- Testing for row:', df.loc[[rij_index]])
            speler_id = df.iat[rij_index, kol_idx_player]
            status = df.iat[rij_index, kol_idx_player_profile_status]
            extr_date = df.iat[rij_index, kol_idx_extr_date]
            aantal_rijen_per_speler[speler_id] = aantal_rijen_per_speler.get(speler_id, 0) + 1

            # eerst checken of er ooit is uitgeschreven.
            if speler_id not in statussen_lijst:
                if status in {"SELF_EXCLUDED_TEMP", "SELF_EXCLUDED_INDEF"}:
                    statussen_lijst[speler_id] = [1, status, extr_date]
                else:
                    statussen_lijst[speler_id] = [0, status, extr_date]
            else:
                last_date = statussen_lijst[speler_id][2]
                if extr_date > last_date:
                    if status in {"SELF_EXCLUDED_TEMP", "SELF_EXCLUDED_INDEF"} or statussen_lijst[speler_id][1] in {"SELF_EXCLUDED_TEMP", "SELF_EXCLUDED_INDEF"}:
                        statussen_lijst[speler_id] = [1, status, extr_date]
                    else:
                        statussen_lijst[speler_id] = [0, status, extr_date]

    # Bouw het resultaat-DataFrame
    spelers_lijst = []
    ooit_uitgeschreven_lijst = []
    laatste_status_lijst = []
    datum_lijst = []
    aantal_rijen_lijst = []

    for speler_id, lijst_van_statussen in statussen_lijst.items():
        spelers_lijst.append(speler_id)
        ooit_uitgeschreven_lijst.append(lijst_van_statussen[0])
        laatste_status_lijst.append(lijst_van_statussen[1])
        datum_lijst.append(lijst_van_statussen[2])
        aantal_rijen_lijst.append(aantal_rijen_per_speler.get(speler_id, 0))
    resultaat = pd.DataFrame({
        "Player_Profile_ID": spelers_lijst,
        "y_1b_has_ever_requested_exclusion": ooit_uitgeschreven_lijst,
        "y_1c_last_status": laatste_status_lijst,
        "y_1d_date_profile_player_status": datum_lijst,
        "y_1x_check_aantal_rijen_in_WOK_Player_Profile": aantal_rijen_lijst
    })


    resultaat['y_1e_certain_which_category'] = resultaat['y_1c_last_status'].apply(
        lambda x: 1 if x in {"SELF_EXCLUDED_TEMP", "SELF_EXCLUDED_INDEF", "ACTIVE"} else 0
    )

    resultaat['y_1_final_has_ever_requested_exclusion'] = resultaat["y_1b_has_ever_requested_exclusion"].where(
        resultaat["y_1e_certain_which_category"] == 1,  # zet op NaN als uncertain
        np.nan
    )

    logger.info(f"Feature 'y_1_final_has_ever_requested_exclusion' klaar: players={len(resultaat):,}")

    return resultaat

# ------------------------------
# Target-feature B: heeft de persoon zich ooit uitgeschreven in Juni of Juli?
# ------------------------------


def maak_self_exclusion_juni_juli_2025(
    tables: Dict[str, List[Path]],
    *,
    usecols: List[str] = ["Player_Profile_ID", "Player_Profile_Status", "Player_Profile_Modified"],
    chunksize: int = 200_000,
    verbose: bool = False,
    log_path: Path | None = None,
) -> pd.DataFrame:
    """
    Binaire feature:
    - y_self_exclusion_juni_juli_2025 = 1
      als speler in juni of juli 2025 minstens één keer
      SELF_EXCLUDED_TEMP of SELF_EXCLUDED_INDEF heeft gehad.
    """

    profile_paths = tables.get("WOK_Player_Profile")
    if not profile_paths:
        msg = "⚠️ WOK_Player_Profile niet beschikbaar – maak_self_exclusion_juni_juli_2025 wordt leeg teruggegeven"
        print(msg)
        return pd.DataFrame(columns=["Player_Profile_ID", "y_self_exclusion_juni_juli_2025"])

    # Grenzen
    start_datum = pd.Timestamp("2025-06-01")
    eind_datum = pd.Timestamp("2025-08-01")  # exclusief

    self_exclusion_statussen = {"SELF_EXCLUDED_TEMP", "SELF_EXCLUDED_INDEF"}

    # Accumulator: speler_id -> 0/1
    self_excluded_spelers: set[Any] = set()
    alle_spelers: set[Any] = set()

    for df in iter_csv_chunks(
        paths=profile_paths,
        usecols=usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        ts = pd.to_datetime(
            df["Player_Profile_Modified"],
            errors="coerce",
            utc=True
        ).dt.tz_localize(None)
        mask_periode = (ts >= start_datum) & (ts < eind_datum)
        if not mask_periode.any():
            continue

        sub = df.loc[mask_periode, ["Player_Profile_ID", "Player_Profile_Status"]]
        alle_spelers.update(sub["Player_Profile_ID"].dropna().unique())

        mask_self_excl = sub["Player_Profile_Status"].astype(str).isin(self_exclusion_statussen)
        if mask_self_excl.any():
            self_excluded_spelers.update(
                sub.loc[mask_self_excl, "Player_Profile_ID"].dropna().unique()
            )

    # Bouw output
    records = []
    for speler_id in alle_spelers:
        records.append(
            {
                "Player_Profile_ID": speler_id,
                "y_self_exclusion_juni_juli_2025": int(speler_id in self_excluded_spelers),
            }
        )

    return pd.DataFrame.from_records(records)


# ------------------------------
# Target-feature B2: heeft de persoon zich ooit uitgeschreven (flexibele periode)?
# ------------------------------

def maak_self_exclusion_flexible_y(
    tables: Dict[str, List[Path]],
    *,
    y_tijdspad: List[str] | None = None,
    usecols: List[str] = ["Player_Profile_ID", "Player_Profile_Status", "Player_Profile_Modified"],
    chunksize: int = 200_000,
    verbose: bool = False,
    log_path: Path | None = None,
) -> pd.DataFrame:
    """
    Binaire feature: speler heeft SELF_EXCLUDED status in gegeven periode.

    Dit is een flexibele versie van `maak_self_exclusion_juni_juli_2025` waarbij
    de tijdsperiode als parameter kan worden meegegeven.

    Parameters
    ----------
    tables : Dict[str, List[Path]]
        Dictionary met tabel namen en hun bestandspaden.
    y_tijdspad : List[str], optional
        Lijst met [start_datum, eind_datum] in formaat 'DDMMYYYY'.
        Bijvoorbeeld: ['01062025', '01082025'] voor juni-juli 2025.
        Default: ['01062025', '01082025'] (juni-juli 2025).
    usecols : List[str]
        Kolommen die ingelezen moeten worden.
    chunksize : int
        Aantal rijen per chunk.
    verbose : bool
        Uitgebreide logging.
    log_path : Path, optional
        Pad naar logbestand.

    Returns
    -------
    pd.DataFrame
        DataFrame met kolommen:
        - Player_Profile_ID
        - y_self_exclusion_{start}_{eind} : 1 als speler in periode SELF_EXCLUDED status had, anders 0

    Examples
    --------
    >>> tables = {"WOK_Player_Profile": [Path("...")]}
    >>> result = maak_self_exclusion_flexible_y(
    ...     tables,
    ...     y_tijdspad=["19122024", "21122024"],  # 19-21 december 2024
    ... )
    >>> result.columns
    Index(['Player_Profile_ID', 'y_self_exclusion_20241219_20241221'], dtype='object')
    """
    # Default: juni-juli 2025 (backward compatible)
    if y_tijdspad is None:
        y_tijdspad = ["01062025", "01082025"]

    # Converteer string datums naar Timestamps
    start_datum = parse_ddmmyyyy_to_timestamp(y_tijdspad[0])
    eind_datum = parse_ddmmyyyy_to_timestamp(y_tijdspad[1])

    # Genereer dynamische kolomnaam
    column_name = generate_y_column_name(y_tijdspad)

    # Setup logger indien nodig
    if log_path:
        logger = _setup_feature_logger(log_path, "flexible_y_target")
        logger.info(f"▶ START flexible Y-target: {column_name}")
        logger.info(f"  Periode: {start_datum} tot {eind_datum}")
    else:
        logger = None

    profile_paths = tables.get("WOK_Player_Profile")
    if not profile_paths:
        msg = "⚠️ WOK_Player_Profile niet beschikbaar – self_exclusion target wordt leeg teruggegeven"
        if logger:
            logger.warning(msg)
        print(msg)
        return pd.DataFrame(columns=["Player_Profile_ID", column_name])

    self_exclusion_statussen = {"SELF_EXCLUDED_TEMP", "SELF_EXCLUDED_INDEF"}

    # Accumulators:
    # - ALL players in entire dataset (for universe)
    # - Players with data in Y period
    # - Players with self-exclusion in Y period
    alle_spelers_in_dataset: set[Any] = set()
    spelers_met_data_in_y_periode: set[Any] = set()
    self_excluded_spelers: set[Any] = set()

    # Pass 1: Collect ALL players from entire dataset
    if logger:
        logger.info("  📊 Pass 1: Collecting ALL players from dataset...")
    for df in iter_csv_chunks(
        paths=profile_paths,
        usecols=usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        alle_spelers_in_dataset.update(df["Player_Profile_ID"].dropna().unique())

    if logger:
        logger.info(f"  → Found {len(alle_spelers_in_dataset):,} unique players in dataset")

    # Pass 2: Identify players with data in Y period and their exclusion status
    if logger:
        logger.info(f"  📊 Pass 2: Checking exclusions in Y period ({start_datum} to {eind_datum})...")
    for df in iter_csv_chunks(
        paths=profile_paths,
        usecols=usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        ts = pd.to_datetime(
            df["Player_Profile_Modified"],
            errors="coerce",
            utc=True
        ).dt.tz_localize(None)
        mask_periode = (ts >= start_datum) & (ts < eind_datum)
        if not mask_periode.any():
            continue

        sub = df.loc[mask_periode, ["Player_Profile_ID", "Player_Profile_Status"]]
        spelers_met_data_in_y_periode.update(sub["Player_Profile_ID"].dropna().unique())

        mask_self_excl = sub["Player_Profile_Status"].astype(str).isin(self_exclusion_statussen)
        if mask_self_excl.any():
            self_excluded_spelers.update(
                sub.loc[mask_self_excl, "Player_Profile_ID"].dropna().unique()
            )

    # Bouw output: ALL players from dataset
    records = []
    for speler_id in alle_spelers_in_dataset:
        if speler_id in spelers_met_data_in_y_periode:
            # Player has data in Y period
            y_value = 1 if speler_id in self_excluded_spelers else 0
        else:
            # Player has NO data in Y period → NaN
            y_value = np.nan

        records.append(
            {
                "Player_Profile_ID": speler_id,
                column_name: y_value,
            }
        )

    # Error handling: check if DataFrame would be empty
    if not records:
        error_msg = (
            f"❌ ERROR: Y-target functie '{column_name}' zou lege DataFrame produceren!\n"
            f"   Periode: {start_datum} tot {eind_datum}\n"
            f"   Geen spelers gevonden in dataset.\n"
            f"   Check of de input bestanden correct zijn en data bevatten."
        )
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)

    result = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ Feature '{column_name}' klaar: {len(records):,} spelers")
        logger.info(f"   - Met data in Y periode: {len(spelers_met_data_in_y_periode):,}")
        logger.info(f"   - Waarvan SELF_EXCLUDED: {len(self_excluded_spelers):,}")
        logger.info(f"   - Zonder data in Y periode (NaN): {len(alle_spelers_in_dataset - spelers_met_data_in_y_periode):,}")

    return result


# ------------------------------
# Target-feature C: heeft de persoon zich ooit uitgeschreven (per maand)?
# ------------------------------

def maak_exclusion_maand_variabelen(
    tables: Dict[str, List[Path]],
    *,
    usecols: List[str] = ["Player_Profile_ID", "Player_Profile_Status", "Player_Profile_Modified"],
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Player_Profile_Status kan de volgende waardes hebben:
        • ACTIVE: the player’s identity has been verified and his/her status is normal;
        • TRIAL: the verification of the player’s identity is still ongoing, a player can only
        log in to complete the registration. Transactions and gambling are not allowed
        until the status ACTIVE is assigned;10
        • SUSPENDED: the player has been suspended by the operator due to not accessing
        his account for a certain amount of time;
        • SUSPENDED_DEATH: the player has been suspended by the operator because
        he died;
        • BLOCKED: the player’s account has been blocked because the operator suspects
        fraudulent behaviour and/or the player does not meet the operator’s terms and
        conditions;
        • SELF_EXCLUDED_TEMP: the player has requested to be excluded from playing
        for a set period;
        • SELF_EXCLUDED_INDEF: the player has requested to be excluded from playing
        for an indefinite period;
        • OTHER: the player has a status that does not fit any of the other categories.

    We zullen per speler bepalen of hij/zij in de 24 maanden de status SELF_EXCLUDED_TEMP of SELF_EXCLUDED_INDEF heeft gehad.
    - maand_{m}_zelfuitsluiting_waargenomen (0/1)
    - maand_{m}_start_zelfuitsluiting (0/1)
    - maand_{m}_einde_zelfuitsluiting (0/1)
    - maand_{m}_status_onzeker (0/1)
    """
    profile_paths = tables.get("WOK_Player_Profile")
    if not profile_paths:
        msg = "⚠️ WOK_Player_Profile niet beschikbaar – maak_exclusion_maand_variabelen wordt leeg teruggegeven"
        print(msg)
        return pd.DataFrame(columns=["Player_Profile_ID"])

    logger = _setup_feature_logger(
        (Path(profile_paths[0]).parent / "feature_maak_exclusion_maand_variabelen.log") if log_path is None else log_path,
        "maak_exclusion_maand_variabelen",
    )
    logger.info("▶ START feature 'maak_exclusion_maand_variabelen' (streaming)")
    logger.info(f"exclusion files={len(profile_paths)} | chunksize={chunksize}")

    # ============================
    # Config: 24 maanden venster
    # ============================
    start_maand = pd.Period("2023-08", freq="M")
    aantal_maanden = 24
    
    # Self exclusion 
    self_exclusion_statussen = {'SELF_EXCLUDED_TEMP', 'SELF_EXCLUDED_INDEF'}
    # Onzekere status
    # TRIAL nemen we hier NIET op als "onzeker":
    # TRIAL = registratie/identiteitsverificatie loopt nog; gokken/transacties niet toegestaan.
    # We behandelen dit dus als pre-activatie, niet als ambigu/censoring-status.
    onzekere_statussen = {'SUSPENDED', 'SUSPENDED_DEATH', 'BLOCKED', 'OTHER'}

    # ============================
    # Accumulators per speler:
    # - bytearray(24) met 0/1
    # ============================
    maanden_zelfuitsluiting_per_speler: Dict[Any, bytearray] = {}
    maanden_onzeker_per_speler: Dict[Any, bytearray] = {}
    alle_spelers_in_venster: set[Any] = set()

    for df in iter_csv_chunks(
        paths=profile_paths,
        usecols=usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        maand = pd.to_datetime(df["Player_Profile_Modified"], errors="coerce").dt.to_period("M")

        # maand-index (0..23) t.o.v. start_maand
        maand_index = (maand - start_maand).apply(lambda x: x.n if pd.notna(x) else np.nan)
        maand_index = maand_index.astype("Int64")
        # Houd alleen rijen binnen jouw 24-maanden venster
        df_met_geldige_data = maand_index.notna() & (maand_index >= 0) & (maand_index < aantal_maanden)
        
        # Neem alle spelers mee die in dit 24-maanden venster voorkomen (voor negatives)
        alle_spelers_in_venster.update(df.loc[df_met_geldige_data, "Player_Profile_ID"].dropna().unique())
        
        # speler_id_kolom = df.loc[df_met_geldige_data, "Player_Profile_ID"]
        # status_kolom = df.loc[df_met_geldige_data, "Player_Profile_Status"].astype(str)
        # maand_index_kolom = maand_index.loc[df_met_geldige_data].astype(int)

        # Subset die we nodig hebben
        sub = df.loc[df_met_geldige_data, ["Player_Profile_ID", "Player_Profile_Status"]].copy()
        sub["maand_index"] = maand_index.loc[df_met_geldige_data].astype(int).to_numpy()
        status_string = sub["Player_Profile_Status"].astype(str)

        # =========================================================
        # 1) Zelfuitsluiting waargenomen (per speler, per maand)
        #
        # We willen: voor elke speler een vector van 24 posities (0/1),
        # waarbij positie m = 1 betekent: in maand m is minstens één keer
        # een self-exclusion status gezien.
        #
        # Omdat er meerdere records in dezelfde maand kunnen zijn,
        # reduceren we eerst naar unieke combinaties (speler, maand_index).
        # =========================================================

        is_zelfuitsluiting = status_string.isin(self_exclusion_statussen)
        if is_zelfuitsluiting.any():
            unieke_speler_maand_combinaties = (
                sub.loc[is_zelfuitsluiting, ["Player_Profile_ID", "maand_index"]]
                   .drop_duplicates()
            )
            # print(unieke_speler_maand_combinaties)

            # Heb ik deze speler al een maand-overzicht gegeven (lukt get)?
            for speler_profiel_id, maand_index in unieke_speler_maand_combinaties.itertuples(index=False, name=None):
                maanden_vector = maanden_zelfuitsluiting_per_speler.get(speler_profiel_id)
                # print('maanden_vector -----', maanden_vector)
                
                if maanden_vector is None:
                    maanden_vector = bytearray(aantal_maanden)  # default allemaal 0
                    maanden_zelfuitsluiting_per_speler[speler_profiel_id] = maanden_vector

                maanden_vector[int(maand_index)] = 1
            
        # =========================================================
        # 2) Markeer maanden met onzekere status (0/1)
        # =========================================================
        is_onzeker = status_string.isin(onzekere_statussen)
        if is_onzeker.any():
            pairs_onz = sub.loc[is_onzeker, ["Player_Profile_ID", "maand_index"]].drop_duplicates()
            for speler_profiel_id, maand_index in pairs_onz.itertuples(index=False, name=None):
                maanden_vector = maanden_onzeker_per_speler.get(speler_profiel_id)
                if maanden_vector is None:
                    maanden_vector = bytearray(aantal_maanden)
                    maanden_onzeker_per_speler[speler_profiel_id] = maanden_vector
                maanden_vector[int(maand_index)] = 1
    
    # =========================================================
    # Bouw output tabel
    # =========================================================

    # Alle spelers die in het 24-maanden venster voorkwamen (incl. "pure negatives")
    alle_spelers = alle_spelers_in_venster

    # Maak lijst met echte maandlabels (YYYY_MM)
    maanden = [
        (start_maand + i).to_timestamp().strftime("%Y_%m")
        for i in range(aantal_maanden)
    ]

    records: List[dict] = []

    for speler_profiel_id in alle_spelers:
        # Haal vectors op (default = alles 0)
        waargenomen = maanden_zelfuitsluiting_per_speler.get(
            speler_profiel_id, bytearray(aantal_maanden)
        )
        onzeker = maanden_onzeker_per_speler.get(
            speler_profiel_id, bytearray(aantal_maanden)
        )

        # -----------------------------------------
        # Start / einde afleiden uit waargenomen
        # -----------------------------------------
        start = bytearray(aantal_maanden)
        einde = bytearray(aantal_maanden)

        for i in range(aantal_maanden):
            if waargenomen[i] == 1:
                # start = 0 → 1
                if i == 0 or waargenomen[i - 1] == 0:
                    start[i] = 1
                # einde = 1 → 0
                if i == aantal_maanden - 1 or waargenomen[i + 1] == 0:
                    einde[i] = 1

        # -----------------------------------------
        # Bouw rij voor deze speler
        # -----------------------------------------
        row = {
            "Player_Profile_ID": speler_profiel_id,
        }

        for i, maand_label in enumerate(maanden):
            row[f"y_2a_maand_{maand_label}_zelfuitsluiting_waargenomen"] = int(waargenomen[i])
            row[f"y_2b_maand_{maand_label}_start_zelfuitsluiting"] = int(start[i])
            row[f"y_2c_maand_{maand_label}_einde_zelfuitsluiting"] = int(einde[i])
            row[f"y_2d_maand_{maand_label}_status_onzeker"] = int(onzeker[i])

        records.append(row)

    resultaat = pd.DataFrame.from_records(records)

    logger.info(
        "Feature 'maak_exclusion_maand_variabelen' klaar: "
        f"players={len(resultaat):,} | maanden={aantal_maanden} | start_maand={start_maand}"
    )
    return resultaat


#################################################################
####################### X-features ##############################
#################################################################

# gebaseerd op features uit eerder onderzoek:
    # stortingsdichtheid (aantal stortingen per dag per tijdseenheid)
    # leeftijd
    # gemiddelde inzetgrootte
    # dagen sinds laatste inzet



# ------------------------------
# X-feature 0: april + mei 2025 eenvoudige aggregaties per speler
# ------------------------------

def april_mei_2025_features(
    tables: Dict[str, List[Path]],
    *,
    # Transactions
    tx_usecols: List[str] = [
        "Player_Profile_ID",
        "Transaction_Amount",
        "Transaction_Datetime",
        "Transaction_Type",
        "Transaction_Status",
    ],
    # Profile (for temp self-exclusion via Modified)
    profile_usecols: List[str] = [
        "Player_Profile_ID",
        "Player_Profile_Status",
        "Player_Profile_Modified",
    ],
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    April + Mei 2025 (2025-04-01 t/m 2025-05-31) eenvoudige aggregaties per speler.

    Output kolommen:
    - Player_Profile_ID
    - x_am_total_deposits
    - x_am_total_active_days
    - x_am_active_days_per_deposit_ratio
    - x_am_nr_temp_self_exclusions
    - x_am_deposit_amount_variance
    - x_am_withdrawal_vs_deposit_net   (abs(withdrawals) - abs(deposits))

    Let op:
    - Active day = dag met >=1 succesvolle transactie (ongeacht type)
    - Successful = Transaction_Status == "successful" (case-insensitive)
    - Temp self-exclusion: Player_Profile_Status == SELF_EXCLUDED_TEMP op Player_Profile_Modified
    """

    # -----------------------
    # Config: periode
    # -----------------------
    start_datum = pd.Timestamp("2025-04-01")
    eind_datum = pd.Timestamp("2025-06-01")  # exclusief

    # -----------------------
    # Input checks
    # -----------------------
    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if tx_paths is None:
        raise FileNotFoundError("Need WOK_Player_Account_Transaction.")

    profile_paths = tables.get("WOK_Player_Profile")
    if profile_paths is None:
        raise FileNotFoundError("Need WOK_Player_Profile (for temp self-exclusions).")
    
    print("PROFILE PATHS:", [p.name for p in profile_paths])
    print("TX PATHS:", [p.name for p in tx_paths])
    print("PROFILE[0] header:", pd.read_csv(profile_paths[0], nrows=0).columns.tolist())
    # -----------------------
    # Accumulators (transactions)
    # -----------------------
    # deposits count
    dep_count: Dict[Any, int] = {}
    # sum abs deposits
    dep_abs_sum: Dict[Any, float] = {}
    # sum abs withdrawals
    wd_abs_sum: Dict[Any, float] = {}
    # variance deposits: Welford online
    dep_n: Dict[Any, int] = {}
    dep_mean: Dict[Any, float] = {}
    dep_M2: Dict[Any, float] = {}
    # active days: set per speler (simpel; alleen 2 maanden)
    active_days: Dict[Any, set[pd.Timestamp]] = {}
    # also keep seen players in tx window
    players_tx_window: set[Any] = set()

    # -----------------------
    # 1) STREAM transactions (april-mei 2025)
    # -----------------------
    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=tx_usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        # normaliseer timestamp (incl. UTC-aware cases)
        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        mask_periode = (ts >= start_datum) & (ts < eind_datum)
        if not mask_periode.any():
            continue

        sub = df.loc[mask_periode, ["Player_Profile_ID", "Transaction_Amount", "Transaction_Type", "Transaction_Status"]].copy()
        sub_ts = ts.loc[mask_periode]

        # success mask
        success = sub["Transaction_Status"].astype(str).str.lower().eq("successful")
        sub = sub.loc[success]
        sub_ts = sub_ts.loc[success]
        if sub.empty:
            continue

        # active days (floor to day)
        days = local_day_labels(sub_ts)
        ids = sub["Player_Profile_ID"]

        # register players
        players_tx_window.update(ids.dropna().unique())

        # update active days sets
        # (2 maanden -> set per player is ok)
        for speler_id, dag in zip(ids.to_numpy(), days.to_numpy()):
            if pd.isna(speler_id) or pd.isna(dag):
                continue
            s = active_days.get(speler_id)
            if s is None:
                active_days[speler_id] = {pd.Timestamp(dag)}
            else:
                s.add(pd.Timestamp(dag))

        # deposits / withdrawals
        tx_type = sub["Transaction_Type"].astype(str)
        amt = pd.to_numeric(sub["Transaction_Amount"], errors="coerce")

        is_dep = tx_type.eq("DEPOSIT") & amt.notna()
        is_wd = tx_type.eq("WITHDRAWAL") & amt.notna()

        # deposits: count + abs sum + variance online
        if is_dep.any():
            dep_ids = ids.loc[is_dep].to_numpy()
            dep_vals = amt.loc[is_dep].to_numpy(dtype=float)

            for speler_id, v in zip(dep_ids, dep_vals):
                if pd.isna(speler_id) or np.isnan(v):
                    continue
                v_abs = float(abs(v))

                dep_count[speler_id] = dep_count.get(speler_id, 0) + 1
                dep_abs_sum[speler_id] = dep_abs_sum.get(speler_id, 0.0) + v_abs

                # Welford
                n = dep_n.get(speler_id, 0) + 1
                mean = dep_mean.get(speler_id, 0.0)
                M2 = dep_M2.get(speler_id, 0.0)

                delta = v_abs - mean
                mean += delta / n
                delta2 = v_abs - mean
                M2 += delta * delta2

                dep_n[speler_id] = n
                dep_mean[speler_id] = mean
                dep_M2[speler_id] = M2

        # withdrawals: abs sum
        if is_wd.any():
            wd_ids = ids.loc[is_wd].to_numpy()
            wd_vals = amt.loc[is_wd].to_numpy(dtype=float)
            for speler_id, v in zip(wd_ids, wd_vals):
                if pd.isna(speler_id) or np.isnan(v):
                    continue
                wd_abs_sum[speler_id] = wd_abs_sum.get(speler_id, 0.0) + float(abs(v))

    # -----------------------
    # 2) STREAM profile (temp self-exclusions on Modified) april-mei 2025
    # -----------------------
    temp_excl_count: Dict[Any, int] = {}
    players_profile_window: set[Any] = set()

    for df in iter_csv_chunks(
        paths=profile_paths,
        usecols=profile_usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        mod = pd.to_datetime(df["Player_Profile_Modified"], errors="coerce", utc=True).dt.tz_localize(None)
        mask_periode = (mod >= start_datum) & (mod < eind_datum)
        if not mask_periode.any():
            continue

        sub = df.loc[mask_periode, ["Player_Profile_ID", "Player_Profile_Status"]]
        status = sub["Player_Profile_Status"].astype(str)

        players_profile_window.update(sub["Player_Profile_ID"].dropna().unique())

        mask_temp = status.eq("SELF_EXCLUDED_TEMP")
        if mask_temp.any():
            temp_ids = sub.loc[mask_temp, "Player_Profile_ID"].dropna().to_numpy()
            for speler_id in temp_ids:
                temp_excl_count[speler_id] = temp_excl_count.get(speler_id, 0) + 1

    # -----------------------
    # 3) Build output (union of players seen in either stream)
    # -----------------------
    all_players = set(players_tx_window) | set(players_profile_window)

    records: List[dict] = []
    for speler_id in all_players:
        td = int(len(active_days.get(speler_id, set())))
        dep = int(dep_count.get(speler_id, 0))

        ratio = (td / dep) if dep > 0 else 0.0

        # variance of deposit amounts (population variance or sample variance?)
        # Simpel: sample variance als n>=2, anders 0
        n = dep_n.get(speler_id, 0)
        if n >= 2:
            var = float(dep_M2[speler_id] / (n - 1))
        else:
            var = 0.0

        dep_sum = float(dep_abs_sum.get(speler_id, 0.0))
        wd_sum = float(wd_abs_sum.get(speler_id, 0.0))
        net = wd_sum - dep_sum  # abs(withdrawals) - abs(deposits)

        records.append(
            {
                "Player_Profile_ID": speler_id,
                "x_am_total_deposits": dep,
                "x_am_total_active_days": td,
                "x_am_active_days_per_deposit_ratio": ratio,
                "x_am_nr_temp_self_exclusions": int(temp_excl_count.get(speler_id, 0)),
                "x_am_deposit_amount_variance": var,
                "x_am_withdrawal_vs_deposit_net": net,
            }
        )

    return pd.DataFrame.from_records(records)


# ------------------------------
# X-feature 1: Flexibele periode-gebaseerde aggregaties per speler
# ------------------------------

def maak_flexible_x_features(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    # Transactions
    tx_usecols: List[str] = [
        "Player_Profile_ID",
        "Transaction_Amount",
        "Transaction_Datetime",
        "Transaction_Type",
        "Transaction_Status",
    ],
    # Profile (for temp self-exclusion via Modified)
    profile_usecols: List[str] = [
        "Player_Profile_ID",
        "Player_Profile_Status",
        "Player_Profile_Modified",
    ],
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Flexibele periode-gebaseerde aggregaties per speler (zoals april_mei_2025_features maar met tijdsparameters).

    Dit is een flexibele versie van `april_mei_2025_features` waarbij de tijdsperiode als parameter
    kan worden meegegeven.

    Parameters
    ----------
    tables : Dict[str, List[Path]]
        Dictionary met tabel namen en hun bestandspaden.
    x_tijdspad : List[str], optional
        Lijst met [start_datum, eind_datum] in formaat 'DDMMYYYY'.
        Bijvoorbeeld: ['01042025', '01062025'] voor april-mei 2025.
        Default: ['01042025', '01062025'] (april-mei 2025).
    tx_usecols : List[str]
        Kolommen uit Transaction tabel.
    profile_usecols : List[str]
        Kolommen uit Profile tabel.
    chunksize : int
        Aantal rijen per chunk.
    log_path : Path, optional
        Pad naar logbestand.
    verbose : bool
        Uitgebreide logging.

    Returns
    -------
    pd.DataFrame
        DataFrame met kolommen:
        - Player_Profile_ID
        - x_{start}_{end}_total_deposits : aantal deposits
        - x_{start}_{end}_total_active_days : aantal unieke dagen met transacties
        - x_{start}_{end}_active_days_per_deposit_ratio : actieve dagen / deposits
        - x_{start}_{end}_nr_temp_self_exclusions : aantal temp self-exclusions
        - x_{start}_{end}_deposit_amount_variance : variantie van deposit bedragen
        - x_{start}_{end}_withdrawal_vs_deposit_net : net (withdrawals - deposits)

    Examples
    --------
    >>> tables = {
    ...     "WOK_Player_Account_Transaction": [Path("...")],
    ...     "WOK_Player_Profile": [Path("...")]
    ... }
    >>> result = maak_flexible_x_features(
    ...     tables,
    ...     x_tijdspad=["01012024", "31012024"],  # Januari 2024
    ... )
    >>> result.columns
    Index(['Player_Profile_ID', 'x_20240101_20240131_total_deposits', ...], dtype='object')
    """
    # Default: april-mei 2025 (backward compatible)
    if x_tijdspad is None:
        x_tijdspad = ["01042025", "01062025"]

    # Converteer string datums naar Timestamps
    start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
    eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])

    # Genereer dynamische prefix
    prefix = generate_x_prefix(x_tijdspad)

    # Setup logger indien nodig
    if log_path:
        logger = _setup_feature_logger(log_path, "flexible_x_features")
        logger.info(f"▶ START flexible X-features: prefix={prefix}")
        logger.info(f"  Periode: {start_datum} tot {eind_datum}")
    else:
        logger = None

    # -----------------------
    # Input checks
    # -----------------------
    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if tx_paths is None:
        raise FileNotFoundError("Need WOK_Player_Account_Transaction.")

    profile_paths = tables.get("WOK_Player_Profile")
    if profile_paths is None:
        raise FileNotFoundError("Need WOK_Player_Profile (for temp self-exclusions).")

    # -----------------------
    # Accumulators
    # -----------------------
    # ALL players in entire dataset (for universe)
    alle_spelers_in_dataset: set[Any] = set()

    # Players with data in X period
    spelers_met_data_in_x_periode: set[Any] = set()

    # Transaction data (only for X period)
    dep_count: Dict[Any, int] = {}
    dep_abs_sum: Dict[Any, float] = {}
    wd_abs_sum: Dict[Any, float] = {}
    # Welford online variance
    dep_n: Dict[Any, int] = {}
    dep_mean: Dict[Any, float] = {}
    dep_M2: Dict[Any, float] = {}
    # Active days
    active_days: Dict[Any, set[pd.Timestamp]] = {}
    # Temp self-exclusions
    temp_excl_count: Dict[Any, int] = {}

    # -----------------------
    # Pass 1: Collect ALL players from entire dataset
    # -----------------------
    if logger:
        logger.info("  📊 Pass 1: Collecting ALL players from dataset...")

    # From transactions
    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=tx_usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        alle_spelers_in_dataset.update(df["Player_Profile_ID"].dropna().unique())

    # From profile
    for df in iter_csv_chunks(
        paths=profile_paths,
        usecols=profile_usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        alle_spelers_in_dataset.update(df["Player_Profile_ID"].dropna().unique())

    if logger:
        logger.info(f"  → Found {len(alle_spelers_in_dataset):,} unique players in dataset")

    # -----------------------
    # Pass 2: Collect data for X period only
    # -----------------------
    if logger:
        logger.info(f"  📊 Pass 2: Processing X period ({start_datum} to {eind_datum})...")

    # 2a) STREAM transactions in X period
    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=tx_usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        mask_periode = (ts >= start_datum) & (ts < eind_datum)
        if not mask_periode.any():
            continue

        sub = df.loc[mask_periode, ["Player_Profile_ID", "Transaction_Amount", "Transaction_Type", "Transaction_Status"]].copy()
        sub_ts = ts.loc[mask_periode]

        # Success mask
        success = sub["Transaction_Status"].astype(str).str.lower().eq("successful")
        sub = sub.loc[success]
        sub_ts = sub_ts.loc[success]
        if sub.empty:
            continue

        # Active days (floor to day)
        days = local_day_labels(sub_ts)
        ids = sub["Player_Profile_ID"]

        # Register players with data in X period
        spelers_met_data_in_x_periode.update(ids.dropna().unique())

        # Update active days sets
        for speler_id, dag in zip(ids.to_numpy(), days.to_numpy()):
            if pd.isna(speler_id) or pd.isna(dag):
                continue
            s = active_days.get(speler_id)
            if s is None:
                active_days[speler_id] = {pd.Timestamp(dag)}
            else:
                s.add(pd.Timestamp(dag))

        # Deposits / withdrawals
        tx_type = sub["Transaction_Type"].astype(str)
        amt = pd.to_numeric(sub["Transaction_Amount"], errors="coerce")

        is_dep = tx_type.eq("DEPOSIT") & amt.notna()
        is_wd = tx_type.eq("WITHDRAWAL") & amt.notna()

        # Deposits: count + abs sum + variance (Welford)
        if is_dep.any():
            dep_ids = ids.loc[is_dep].to_numpy()
            dep_vals = amt.loc[is_dep].to_numpy(dtype=float)

            for speler_id, v in zip(dep_ids, dep_vals):
                if pd.isna(speler_id) or np.isnan(v):
                    continue
                v_abs = float(abs(v))

                dep_count[speler_id] = dep_count.get(speler_id, 0) + 1
                dep_abs_sum[speler_id] = dep_abs_sum.get(speler_id, 0.0) + v_abs

                # Welford algorithm
                n = dep_n.get(speler_id, 0) + 1
                mean = dep_mean.get(speler_id, 0.0)
                M2 = dep_M2.get(speler_id, 0.0)

                delta = v_abs - mean
                mean += delta / n
                delta2 = v_abs - mean
                M2 += delta * delta2

                dep_n[speler_id] = n
                dep_mean[speler_id] = mean
                dep_M2[speler_id] = M2

        # Withdrawals: abs sum
        if is_wd.any():
            wd_ids = ids.loc[is_wd].to_numpy()
            wd_vals = amt.loc[is_wd].to_numpy(dtype=float)
            for speler_id, v in zip(wd_ids, wd_vals):
                if pd.isna(speler_id) or np.isnan(v):
                    continue
                wd_abs_sum[speler_id] = wd_abs_sum.get(speler_id, 0.0) + float(abs(v))

    # 2b) STREAM profile in X period (temp self-exclusions)
    for df in iter_csv_chunks(
        paths=profile_paths,
        usecols=profile_usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        mod = pd.to_datetime(df["Player_Profile_Modified"], errors="coerce", utc=True).dt.tz_localize(None)
        mask_periode = (mod >= start_datum) & (mod < eind_datum)
        if not mask_periode.any():
            continue

        sub = df.loc[mask_periode, ["Player_Profile_ID", "Player_Profile_Status"]]
        status = sub["Player_Profile_Status"].astype(str)

        spelers_met_data_in_x_periode.update(sub["Player_Profile_ID"].dropna().unique())

        mask_temp = status.eq("SELF_EXCLUDED_TEMP")
        if mask_temp.any():
            temp_ids = sub.loc[mask_temp, "Player_Profile_ID"].dropna().to_numpy()
            for speler_id in temp_ids:
                temp_excl_count[speler_id] = temp_excl_count.get(speler_id, 0) + 1

    # -----------------------
    # 3) Build output: ALL players from dataset
    # -----------------------
    records: List[dict] = []
    for speler_id in alle_spelers_in_dataset:
        if speler_id in spelers_met_data_in_x_periode:
            # Player has data in X period → compute features
            td = int(len(active_days.get(speler_id, set())))
            dep = int(dep_count.get(speler_id, 0))

            ratio = (td / dep) if dep > 0 else 0.0

            # Variance of deposit amounts (sample variance if n>=2)
            n = dep_n.get(speler_id, 0)
            if n >= 2:
                var = float(dep_M2[speler_id] / (n - 1))
            else:
                var = 0.0

            dep_sum = float(dep_abs_sum.get(speler_id, 0.0))
            wd_sum = float(wd_abs_sum.get(speler_id, 0.0))
            net = wd_sum - dep_sum  # abs(withdrawals) - abs(deposits)

            records.append(
                {
                    "Player_Profile_ID": speler_id,
                    f"{prefix}total_deposits": dep,
                    f"{prefix}total_active_days": td,
                    f"{prefix}active_days_per_deposit_ratio": ratio,
                    f"{prefix}nr_temp_self_exclusions": int(temp_excl_count.get(speler_id, 0)),
                    f"{prefix}deposit_amount_variance": var,
                    f"{prefix}withdrawal_vs_deposit_net": net,
                }
            )
        else:
            # Player has NO data in X period → all features NaN
            records.append(
                {
                    "Player_Profile_ID": speler_id,
                    f"{prefix}total_deposits": np.nan,
                    f"{prefix}total_active_days": np.nan,
                    f"{prefix}active_days_per_deposit_ratio": np.nan,
                    f"{prefix}nr_temp_self_exclusions": np.nan,
                    f"{prefix}deposit_amount_variance": np.nan,
                    f"{prefix}withdrawal_vs_deposit_net": np.nan,
                }
            )

    # Error handling: check if DataFrame would be empty
    if not records:
        error_msg = (
            f"❌ ERROR: X-features functie '{prefix}' zou lege DataFrame produceren!\n"
            f"   Periode: {start_datum} tot {eind_datum}\n"
            f"   Geen spelers gevonden in dataset.\n"
            f"   Check of de input bestanden correct zijn en data bevatten."
        )
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)

    result = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ Flexible X-features '{prefix}' klaar: {len(records):,} spelers")
        logger.info(f"   - Met data in X periode: {len(spelers_met_data_in_x_periode):,}")
        logger.info(f"   - Zonder data in X periode (NaN): {len(alle_spelers_in_dataset - spelers_met_data_in_x_periode):,}")
        logger.info(f"   Periode: {start_datum} → {eind_datum}")

    return result



# ------------------------------
# X-feature 1: Gemiddeld aantal stortingen per dag voor de afgelopen 30, 60, 300 en 600 dagen
# ------------------------------


def transactions(
    tables: Dict[str, List[Path]],
    *,
    usecols: List[str] = ["Player_Profile_ID", "Transaction_ID", "Transaction_Amount", "Transaction_Datetime", "Transaction_Deposit_Instrument", "Transaction_Type","Transaction_Status"],
    chunksize: int = 200_000,
    batch_out: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Het totaal aantal stortingen als absoluut getal per speler berekenen.
    Totaal actieve dagen dat er transacties zijn geweest heeft gedaan in de betreffende periode berekenen.
    Voor elke speler het gemiddelde aantal stortingen per dag berekenen over de afgelopen 30, 60, 300 en 600 dagen van die speler.
    Voor elke tijdsperiode ook het gemiddelde bedrag van de stortingen per dag berekenen (dit kan gewoon met pandas denk ik).
    
    We definiëren een storting als een transactie met Transaction_Type 'DEPOSIT' en Transaction_Status 'COMPLETED'.
    Output
    ------
    DataFrame met kolommen:
    - Player_Profile_ID
    - x_1_a_total_nr_of_deposit_transactions (int)
    - x_1_b_total_nr_of_active_days (int)
    - x_1_c_30_avg_nr_of_deposit_transactions_per_day (float)
    - x_1_c_60_avg_nr_of_deposit_transactions_per_day (float)
    - x_1_c_300_avg_nr_of_deposit_transactions_per_day (float)
    - x_1_c_600_avg_nr_of_deposit_transactions_per_day (float)
    - x_1_d_30_avg_amount_of_deposit_transactions_per_day (float)
    - x_1_d_60_avg_amount_of_deposit_transactions_per_day (float)
    - x_1_d_300_avg_amount_of_deposit_transactions_per_day (float)
    - x_1_d_600_avg_amount_of_deposit_transactions_per_day (float)

    We gaan heel ambitieus zijn en een dict maken als {speler: [{actieve_dag: [stortingen op die dag]}, ...]}
    Zodat we per speler precies weten op welke dagen er hoeveel stortingen zijn gedaan.

    """
    transaction_paths = tables.get("WOK_Player_Account_Transaction")
    if transaction_paths is None:
        raise FileNotFoundError("Need WOK_Player_Account_Transaction.")
    else:
        print('transaction_paths:', transaction_paths) if verbose else None
    logger = _setup_feature_logger(
        (Path(transaction_paths[0]).parent / "feature_x_1_a_avg_nr_of_deposit_transactions_per_day.log") if log_path is None else log_path,
        "x_1_a_avg_nr_of_deposit_transactions_per_day",
    )
    logger.info("▶ START feature 'x_1_a_avg_nr_of_deposit_transactions_per_day' (streaming)")
    logger.info(f"transaction_files={len(transaction_paths)} | chunksize={chunksize}")

    # Accumulator per speler:
    # deposit_counts_per_speler[speler] = aantal stortingen per speler
    deposit_counts_per_speler_per_day = {}

    # Stream de CSV's in chunks om geheugen te sparen
    for df in iter_csv_chunks(
        paths=transaction_paths,
        usecols=usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        current_time = datetime.now()
        current_time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f'chunk {df.columns} at current_time {current_time_string}')
        
        # --- 1) Parse naar dagen, dus verwijder tijdsgedeelte ---
        date_column = local_day_labels(pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True, format="mixed"))

        # --- 2) Success mask (jouw definitie: successful/SUCCESSFUL = True) ---
        success = df["Transaction_Status"].astype(str).str.lower().eq("successful")

        # --- 3) Active days: per (player,date) het aantal succesvolle transacties (maakt straks zoveel 0's) ---
        active_counts = (
            pd.DataFrame({"Player_Profile_ID": df["Player_Profile_ID"], "date": date_column})
            .loc[success]
            .dropna(subset=["Player_Profile_ID", "date"])
            
            # doe .groupby().size() om rijen te tellen. Dit maakt een series met een multi-index van (Player_Profile_ID, date).
            .groupby(["Player_Profile_ID", "date"], sort=False)
            .size()
            # maak er weer een dataframe van
            .reset_index(name="number_of_successful_transactions")
        )
        if verbose:
            print('active_counts -----')
            print(active_counts.head(10))

        # --- 4) Deposits: per (player,date) count + lijst met bedragen ---
        dep = pd.DataFrame(
            {
                "Player_Profile_ID": df["Player_Profile_ID"],
                "date": date_column,
                "amt": df["Transaction_Amount"],
                "type": df["Transaction_Type"],
            }
        )

        deposit_mask = success & dep["type"].eq("DEPOSIT")

        # pak alleen de deposits, gegroepeerd per (player,date), met lijst van bedragen
        dep_list = (
            dep.loc[deposit_mask]
            .dropna(subset=["Player_Profile_ID", "date"])
            .groupby(["Player_Profile_ID", "date"], sort=False)["amt"]
            .apply(list)                      # <- lijst met echte bedragen
            .reset_index(name="deposit_amounts")
        )

        # --- 5) Integratie: update jouw bestaande dict met (a) 0's voor activiteit, (b) bedragen voor deposits ---

        # a) voor elke succesvolle tx op die dag -> voeg 0's toe (zoals je oude code deed)
        for speler_id, dag, n in active_counts.itertuples(index=False, name=None):
            speler_dict = deposit_counts_per_speler_per_day.setdefault(speler_id, {})
            lst = speler_dict.get(dag)
            if lst is None:
                speler_dict[dag] = [0] * int(n)
            else:
                lst.extend([0] * int(n))

        # b) voor deposits op die dag -> voeg echte bedragen toe (en vervang eenzelfde aantal 0's als je wilt)
        for speler_id, dag, bedragen in dep_list.itertuples(index=False, name=None):
            speler_dict = deposit_counts_per_speler_per_day.setdefault(speler_id, {})
            lst = speler_dict.get(dag)
            if lst is None:
                speler_dict[dag] = list(bedragen)
            else:
                lst.extend(bedragen)
        
        time_elapsed = datetime.now() - current_time
        elapsed_seconds = time_elapsed.total_seconds()
        print(f"Total {chunksize} rows in {elapsed_seconds} seconds. ",
              f"So {chunksize/elapsed_seconds} rows per second")

                
    # Bouw het resultaat-DataFrame
    print('building result dataframe')
    spelers = []
    totaal_aantal_stortingen = []
    totaal_aantal_actieve_dagen = []
    avg_30_days = []
    avg_60_days = []
    avg_300_days = []
    avg_600_days = []
    avg_amount_30_days = []
    avg_amount_60_days = []
    avg_amount_300_days = []
    avg_amount_600_days = []

    for speler_id, dagen_dict in deposit_counts_per_speler_per_day.items():
        spelers.append(speler_id)
        totaal_stortingen = sum(len([amt for amt in bedragen if amt > 0]) for bedragen in dagen_dict.values())
        totaal_aantal_stortingen.append(totaal_stortingen)
        totaal_aantal_actieve_dagen.append(len(dagen_dict))

        # Bereken gemiddelden voor 30, 60, 300, 600 dagen
        # Voor elke gewenste periode berekenen we:
        # - hoeveel stortingen er in dat tijdsvenster zijn gedaan
        # - het totaalbedrag aan stortingen in dat tijdsvenster
        # - daaruit het gemiddelde per kalenderdag (niet per actieve dag)
        for periode in [30, 60, 300, 600]:
            print(f'Calculating averages for periode: {periode} days') if verbose else None

            # Het venster loopt terug vanaf de laatste stortingsdag van deze speler
            eind_datum = max(dagen_dict.keys())
            start_datum = eind_datum - pd.Timedelta(days=periode)

            # Selecteer alleen de dagen met stortingen binnen het tijdsvenster
            relevante_dagen = [
                dag for dag in dagen_dict.keys()
                if start_datum <= dag <= eind_datum
            ]

            # Totaal aantal stortingen (count van bedragen > 0 binnen dit venster)
            totaal_stortingen_periode = sum(
                sum(1 for amt in dagen_dict[dag] if amt > 0)
                for dag in relevante_dagen
            )

            # Totaal bedrag aan stortingen binnen dit venster
            totaal_bedrag_periode = sum(
                sum(dagen_dict[dag]) 
                for dag in relevante_dagen
            )

            # Gemiddeld per kalenderdag in dit venster
            avg_stortingen = totaal_stortingen_periode / periode
            avg_bedrag     = totaal_bedrag_periode / periode

            # Waarden opslaan in de juiste variabelen
            if periode == 30:
                avg_30_days.append(avg_stortingen)
                avg_amount_30_days.append(avg_bedrag)
            elif periode == 60:
                avg_60_days.append(avg_stortingen)
                avg_amount_60_days.append(avg_bedrag)
            elif periode == 300:
                avg_300_days.append(avg_stortingen)
                avg_amount_300_days.append(avg_bedrag)
            elif periode == 600:
                avg_600_days.append(avg_stortingen)
                avg_amount_600_days.append(avg_bedrag)

    resultaat = pd.DataFrame({
        "Player_Profile_ID": spelers,
        "x_1_a_total_nr_of_deposit_transactions": totaal_aantal_stortingen,
        "x_1_b_total_nr_of_active_days": totaal_aantal_actieve_dagen,
        "x_1_c_30_avg_nr_of_deposit_transactions_per_day": avg_30_days,
        "x_1_c_60_avg_nr_of_deposit_transactions_per_day": avg_60_days,
        "x_1_c_300_avg_nr_of_deposit_transactions_per_day": avg_300_days,
        "x_1_c_600_avg_nr_of_deposit_transactions_per_day": avg_600_days,
        "x_1_d_30_avg_amount_of_deposit_transactions_per_day": avg_amount_30_days,
        "x_1_d_60_avg_amount_of_deposit_transactions_per_day": avg_amount_60_days,
        "x_1_d_300_avg_amount_of_deposit_transactions_per_day": avg_amount_300_days,
        "x_1_d_600_avg_amount_of_deposit_transactions_per_day": avg_amount_600_days,
    }) 
    return resultaat

# ------------------------------
# Centrale registry
# ------------------------------
FEATURES_REGISTRY = {
    "transaction_amount_sum": {
        "stream_fn": transaction_amount_sum,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": ["Player_Profile_ID", "Transaction_Amount"],
        },
        "log_name": "feature_sum.log",
        "kwargs": {},
    },
    "transaction_amount_sum_in_games": {
        "stream_fn": transaction_amount_sum_in_games,
        "tables": ["WOK_Player_Account_Transaction", "WOK_Game_Session"],
        "usecols": {
            "WOK_Player_Account_Transaction": ["Transaction_ID", "Transaction_Amount"],
            "WOK_Game_Session": ["Game_Session_ID", "Game_Transactions"],
        },
        "log_name": "feature_game_sum.log",
        "kwargs": {},
    },
    "time_eas": {
        "stream_fn": end_accel_score_stream,  # echte implementatie volgt
        "tables": ["WOK_Player_Account_Transaction", "WOK_Game_Session"],
        "usecols": {
            "WOK_Player_Account_Transaction": ["Player_Profile_ID", "Transaction_Datetime"],
            "WOK_Game_Session": ["Game_Session_ID", "Game_Transactions"],
        },
        "log_name": "feature_time_eas.log",
        "kwargs": {},
    },
    "nr_of_bank_accounts": {
        "stream_fn": nr_of_bank_accounts,
        "tables": ["WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Profile": ["Player_Profile_ID","Player_Profile_Bank_Account"],
        },
        "log_name": "nr_of_bank_accounts.log",
        "kwargs": {},
    },
    "nr_of_risk_classes": {
        "stream_fn": nr_of_risk_classes,
        "tables": ["WOK_Player_Flags"],
        "usecols": {
            "WOK_Player_Flags": ["Player_Profile_ID","Flag_RG_Class"],
        },
        "log_name": "nr_of_Flag_RG_Class.log",
        "kwargs": {},
    },
    "latest_limits": {
        "stream_fn": latest_limits,
        "tables": ["WOK_Player_Limits"],
        "usecols": {
            "WOK_Player_Limits": ["Player_Profile_ID","Limit_Deposit","Limit_Login","Limit_Balance"],
        },
        "log_name": "Player_Limits.log",
        "kwargs": {},
    },
    "avg_nr_of_game_transactions": {
        "stream_fn": avg_nr_of_game_transactions,
        "tables": ["WOK_Game_Session"],
        "usecols": {
            "WOK_Game_Session": ["Game_Transactions"],
        },
        "log_name": "Game_Session.log",
        "kwargs": {},
    },
    "avg_nr_of_bet_parts_per_bet": {
        "stream_fn": avg_nr_of_bet_parts_per_bet,
        "tables": ["WOK_Bet"],
        "usecols": {
            "WOK_Bet": ["Bet_Parts","Bet_Transactions"],
        },
        "log_name": "Bet.log",
        "kwargs": {},
    },
    "nr_of_complaints_and_variance_in_nr_responses_per_complaint": {
        "stream_fn": nr_of_complaints_and_variance_in_nr_responses_per_complaint,
        "tables": ["WOK_Complaint"],
        "usecols": {
            "WOK_Complaint": ["Complaint_Player_ID","Complaint_ID","Responses"],
        },
        "log_name": "Bet.log",
        "kwargs": {},
    },
    "has_person_ever_requested_an_exclusion": {
        "stream_fn": has_person_ever_requested_an_exclusion,
        "tables": ["WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Profile": ["Player_Profile_ID","Player_Profile_Status", "Exraction_Date"],
        },
        "log_name": "has_person_ever_requested_an_exclusion.log",
        "kwargs": {},
    },
        "maak_self_exclusion_juni_juli_2025": {
        "stream_fn": maak_self_exclusion_juni_juli_2025,
        "tables": ["WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Profile": ["Player_Profile_ID","Player_Profile_Status", "Player_Profile_Modified"],
        },
        "log_name": "maak_self_exclusion_juni_juli_2025.log",
        "kwargs": {},
    },
    "maak_exclusion_maand_variabelen": {
        "stream_fn": maak_exclusion_maand_variabelen,
        "tables": ["WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Profile": ["Player_Profile_ID","Player_Profile_Status", "Exraction_Date"],
        },
        "log_name": "maak_exclusion_maand_variabelen.log",
        "kwargs": {},
    },
    "april_mei_2025_features": {
        "stream_fn": april_mei_2025_features,
        "tables": ["WOK_Player_Account_Transaction", "WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
        "Player_Profile_ID",
        "Transaction_Amount",
        "Transaction_Datetime",
        "Transaction_Type",
        "Transaction_Status",
    ],
            "WOK_Player_Profile": [
        "Player_Profile_ID",
        "Player_Profile_Status",
        "Player_Profile_Modified",
    ],
        },
        "log_name": "april_mei_2025_features.log",
        "kwargs": {},
    },
    "transactions": {
        "stream_fn": transactions,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": ["Player_Profile_ID", "Transaction_ID", "Transaction_Amount", "Transaction_Deposit_Instrument", "Transaction_Type","Transaction_Status","Transaction_Datetime"],
        },
        "log_name": "transactions.log",
        "kwargs": {},
    },
    "maak_self_exclusion_flexible_y": {
        "stream_fn": maak_self_exclusion_flexible_y,
        "tables": ["WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Profile": ["Player_Profile_ID", "Player_Profile_Status", "Player_Profile_Modified"],
        },
        "log_name": "maak_self_exclusion_flexible_y.log",
        "kwargs": {
            "y_tijdspad": ["01062025", "01082025"],  # Default: juni-juli 2025
        },
    },
    "maak_flexible_x_features": {
        "stream_fn": maak_flexible_x_features,
        "tables": ["WOK_Player_Account_Transaction", "WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Amount",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
            "WOK_Player_Profile": [
                "Player_Profile_ID",
                "Player_Profile_Status",
                "Player_Profile_Modified",
            ],
        },
        "log_name": "maak_flexible_x_features.log",
        "kwargs": {
            "x_tijdspad": ["01042025", "01062025"],  # Default: april-mei 2025
        },
    },

}