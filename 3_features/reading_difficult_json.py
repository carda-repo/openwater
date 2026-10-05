"""
reading_difficult_json.py — STREAMING ONLY

Doel
----
Streamend (in chunks) JSON-kolommen uit WOK-datasets normaliseren zonder
grote DataFrames in het geheugen te houden.

Belangrijk
----------
- Deze module levert *streaming* normalizers (generators) die per batch
  een kleine DataFrame yielden.
- De oude niet-streamende helper is verwijderd.

Uitvoer normalize_game_transactions_stream:
  batches met kolommen ["Transaction_ID", "Player_Profile_ID", "Game_Session_ID"]
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Tuple
import pandas as pd

# ----------------------------
# Kleine, lokale JSON helpers
# ----------------------------

def _clean_json_string(s: Optional[str]) -> Optional[str]:
    """Herstel CSV-geëxporteerde JSON (verdubbelde quotes) en omringende quotes."""
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not t:
        return None
    if t.startswith('"') and t.endswith('"'):
        t = t[1:-1]
    # CSV met dubbele quotes → enkel maken
    t = t.replace('""', '"')
    return t


def _safe_load_json_relaxed(val) -> Optional[dict | list]:
    """Stream-vriendelijke loader: returnt object of None (geen exception per rij)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (dict, list)):
        return val
    if not isinstance(val, str):
        return None
    cleaned = _clean_json_string(val)
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def _iter_tx_entries(obj) -> Iterator[Tuple[str, Optional[str]]]:
    """
    Yield (Transaction_ID, Player_Profile_ID) uit verschillende JSON-vormen:
      - {"Game_Transactions": {"Transaction": [ {...}, ... ]}}
      - {"Game_Transaction": [ {...}, ... ]}
      - {"Transactions": [ {...} ]}
      - directe lijst [ {...}, ... ]
      - of enkele dict met Transaction_ID
    """
    if obj is None:
        return

    def _get_ids(rec: dict) -> Tuple[Optional[str], Optional[str]]:
        txid = rec.get("Transaction_ID") or rec.get("TransactionId") or rec.get("TxId")
        pid = rec.get("Player_Profile_ID") or rec.get("PlayerId") or rec.get("Player_ID")
        return (str(txid) if txid is not None else None,
                str(pid) if pid is not None else None)

    if isinstance(obj, dict):
        # geneste vorm: {"Game_Transactions":{"Transaction":[...]}} of {"Game_Transactions":{"Transactions":[...]}}
        if "Game_Transactions" in obj and isinstance(obj["Game_Transactions"], dict):
            inner = obj["Game_Transactions"]
            for subk in ("Transaction", "Transactions"):
                if subk in inner and isinstance(inner[subk], list):
                    for rec in inner[subk]:
                        if isinstance(rec, dict):
                            txid, pid = _get_ids(rec)
                            if txid:
                                yield (txid, pid)
                    return

        # vlakke lijst direct in dict
        for k in ("Game_Transaction", "Transactions", "Transaction"):
            if k in obj and isinstance(obj[k], list):
                for rec in obj[k]:
                    if isinstance(rec, dict):
                        txid, pid = _get_ids(rec)
                        if txid:
                            yield (txid, pid)
                return

        # fallback: enkelvoudige dict met Transaction_ID
        txid, pid = _get_ids(obj)
        if txid:
            yield (txid, pid)
        return

    if isinstance(obj, list):
        for rec in obj:
            if isinstance(rec, dict):
                txid, pid = _get_ids(rec)
                if txid:
                    yield (txid, pid)


def Player_Profile_Bank_Account_json_iterator(obj) -> Iterator[Tuple[str, Optional[str]]]:
    """
    Yield (Transaction_ID, Player_Profile_ID) uit verschillende JSON-vormen:
      - {"Game_Transactions": {"Transaction": [ {...}, ... ]}}
      - {"Game_Transaction": [ {...}, ... ]}
      - {"Transactions": [ {...} ]}
      - directe lijst [ {...}, ... ]
      - of enkele dict met Transaction_ID
    """
    if obj is None:
        return

    def _get_ids(rec: dict) -> Tuple[Optional[str], Optional[str]]:
        txid = rec.get("Transaction_ID") or rec.get("TransactionId") or rec.get("TxId")
        pid = rec.get("Player_Profile_ID") or rec.get("PlayerId") or rec.get("Player_ID")
        return (str(txid) if txid is not None else None,
                str(pid) if pid is not None else None)

    if isinstance(obj, dict):
        # geneste vorm: {"Game_Transactions":{"Transaction":[...]}} of {"Game_Transactions":{"Transactions":[...]}}
        if "Game_Transactions" in obj and isinstance(obj["Game_Transactions"], dict):
            inner = obj["Game_Transactions"]
            for subk in ("Transaction", "Transactions"):
                if subk in inner and isinstance(inner[subk], list):
                    for rec in inner[subk]:
                        if isinstance(rec, dict):
                            txid, pid = _get_ids(rec)
                            if txid:
                                yield (txid, pid)
                    return

        # vlakke lijst direct in dict
        for k in ("Game_Transaction", "Transactions", "Transaction"):
            if k in obj and isinstance(obj[k], list):
                for rec in obj[k]:
                    if isinstance(rec, dict):
                        txid, pid = _get_ids(rec)
                        if txid:
                            yield (txid, pid)
                return

        # fallback: enkelvoudige dict met Transaction_ID
        txid, pid = _get_ids(obj)
        if txid:
            yield (txid, pid)
        return

    if isinstance(obj, list):
        for rec in obj:
            if isinstance(rec, dict):
                txid, pid = _get_ids(rec)
                if txid:
                    yield (txid, pid)

# ---------------------------------------------------
# STREAMING normalizer: Game_Transactions uit Session
# ---------------------------------------------------

#         # Normaliseer sessie-ID naar string (robust tegen gemixte types)
#         if "Game_Session_ID" in deel_df.columns:
#             deel_df["Game_Session_ID"] = deel_df["Game_Session_ID"].astype(str)

#         speltransacties_kolom = "Game_Transactions"
#         sessie_id_kolom = "Game_Session_ID"

#         # Doorloop de rijen *snel* met .iat (index-based cell access)
#         # Opmerking: .iat[rij_index, kolom_index] is sneller dan .loc[…, …]
#         for rij_index in range(len(deel_df)):
#             # Haal de ruwe JSON-achtige payload uit Game_Transactions
#             if speltransacties_kolom in deel_df.columns:
#                 kolom_index_gt = deel_df.columns.get_loc(speltransacties_kolom)
#                 celwaarde_gt = deel_df.iat[rij_index, kolom_index_gt]
#             else:
#                 celwaarde_gt = None

#             json_object = _safe_load_json_relaxed(celwaarde_gt)
#             if json_object is None:
#                 # JSON was leeg/ongeldig → overslaan maar wel tellen
#                 aantal_foute_json_rijen += 1
#                 continue

#             # Haal de sessie-ID op (indien aanwezig)
#             if sessie_id_kolom in deel_df.columns:
#                 kolom_index_sid = deel_df.columns.get_loc(sessie_id_kolom)
#                 sessie_id = deel_df.iat[rij_index, kolom_index_sid]
#             else:
#                 sessie_id = None

#             # Loop over alle (Transaction_ID, Player_Profile_ID) entries in het JSON-object
#             # _iter_tx_entries(json_object) moet tuples (transactie_id, speler_id) opleveren
#             for transactie_id, speler_profiel_id in _iter_tx_entries(json_object):
#                 uitgaande_rijen.append((
#                     transactie_id,                                      # Transaction_ID
#                     speler_profiel_id,                                   # Player_Profile_ID (mag None)
#                     str(sessie_id) if sessie_id is not None else None    # Game_Session_ID als string
#                 ))
#                 totaal_aantal_transacties += 1

#                 # Zodra we genoeg rijen hebben verzameld, geef een batch terug
#                 if len(uitgaande_rijen) >= batch_grootte:
#                     yield pd.DataFrame(
#                         uitgaande_rijen,
#                         columns=["Transaction_ID", "Player_Profile_ID", "Game_Session_ID"]
#                     )
#                     uitgaande_rijen = []

#     # Eventueel resterende rijen als laatste batch teruggeven
#     if uitgaande_rijen:
#         yield pd.DataFrame(
#             uitgaande_rijen,
#             columns=["Transaction_ID", "Player_Profile_ID", "Game_Session_ID"]
#         )

#     # Samenvatting
#     if luidruchtig:
#         print(
#             "normaliseer_speltransacties_streamend: "
#             f"totaal_transacties={totaal_aantal_transacties:,}, "
#             f"rijen_met_ongeldige_json={aantal_foute_json_rijen:,}"
#         )

def normalize_game_transactions_iterator(
    session_paths: List[Path],
    *,
    chunksize: int = 200_000,
    batch_out: int = 200_000,
    verbose: bool = True,
) -> Iterator[pd.DataFrame]:
    """
    Streamend normaliseren van WOK_Game_Session naar *batches* met kolommen:
        ["Transaction_ID", "Player_Profile_ID", "Game_Session_ID"]

    Wat dit doet (in gewone taal):
    - We lezen de kolommen `Game_Transactions` (JSON-achtige tekst) en `Game_Session_ID` uit de
      WOK_Game_Session CSV-bestanden, in *stukken* (chunks) om geheugen te sparen.
    - Uit het veld `Game_Transactions` halen we per sessie de *individuele transacties* los,
      en koppelen die aan de speler en de sessie.
    - We geven de resultaten niet in één keer terug, maar *in delen* (batches) via `yield`.
      Dat maakt dit een **iterator/generator**: je krijgt telkens een DataFrame zodra er
      `batch_grootte` rijen verzameld zijn (of het restant aan het einde).

    Begrippen:
    - **Iterator/generator**: een functie die niet alles in één keer teruggeeft, maar
      stapje-voor-stapje via `yield`. Dat is efficiënt bij grote datasets.
    - **.iat**: ultralicht-gewicht manier om één cel op te halen uit een DataFrame via
      *rij-index* en *kolom-index* (snel en zuinig, zonder extra checks).
    - **TxID / Transaction_ID**: het *unieke* ID van een transactie (denk: bonnummer).
      We noemen het hier steeds `Transaction_ID` omdat dat de kolomnaam is die downstream
      wordt verwacht.

    Parameters:
    - sessie_paden: lijst met paden naar WOK_Game_Session CSV’s.
    - grootte_chunks: hoeveel rijen per keer we inlezen (streamend).
    - batch_grootte: hoeveel genormaliseerde transactieregels we verzamelen voordat we
      een DataFrame `yield`-en.
    - luidruchtig: of we extra voortgangs-prints doen.

    Returns:
    - Een **iterator** over pandas DataFrames met telkens maximaal `batch_grootte` rijen,
      met kolommen: ["Transaction_ID", "Player_Profile_ID", "Game_Session_ID"].
    """
    from path_finding import iter_csv_chunks  # lazy import

    need_cols = ["Game_Transactions", "Game_Session_ID"]  # let op: géén Player_Profile_ID kolom vereist
    out_rows: List[Tuple[str, Optional[str], Optional[str]]] = []
    total_tx = 0
    bad_json_rows = 0

    for chunk in iter_csv_chunks(session_paths, usecols=need_cols, chunksize=chunksize, verbose=verbose):
        # lichte normalisatie
        if "Game_Session_ID" in chunk.columns:
            chunk["Game_Session_ID"] = chunk["Game_Session_ID"].astype(str)

        gt_col = "Game_Transactions"
        sid_col = "Game_Session_ID"

#         # UITLEG WAT HIER ONDER GEBEURT: zoiets als dit
#         # Opmerking: .iat[rij_index, kolom_index] is sneller dan .loc[…, …]
#         for rij_index in range(len(deel_df)):
#             # Haal de ruwe JSON-achtige payload uit Game_Transactions
#             if speltransacties_kolom in deel_df.columns:
#                 kolom_index_gt = deel_df.columns.get_loc(speltransacties_kolom)
#                 celwaarde_gt = deel_df.iat[rij_index, kolom_index_gt]
#             else:
#                 celwaarde_gt = None
        for i in range(len(chunk)):
            cell = chunk.iat[i, chunk.columns.get_loc(gt_col)] if gt_col in chunk.columns else None
            obj = _safe_load_json_relaxed(cell)
            if obj is None:
                bad_json_rows += 1
                continue

            sid = chunk.iat[i, chunk.columns.get_loc(sid_col)] if sid_col in chunk.columns else None

            for txid, pid in _iter_tx_entries(obj):
                out_rows.append((txid, pid, str(sid) if sid is not None else None))
                total_tx += 1
                if len(out_rows) >= batch_out:
                    yield pd.DataFrame(out_rows, columns=["Transaction_ID", "Player_Profile_ID", "Game_Session_ID"])
                    out_rows = []

    if out_rows:
        yield pd.DataFrame(out_rows, columns=["Transaction_ID", "Player_Profile_ID", "Game_Session_ID"])

    if verbose:
        print(f"normalize_game_transactions_stream: total_tx={total_tx:,}, bad_json_rows={bad_json_rows:,}")

# ---------------------------------------------------
# STREAMING normalizer: Player_account
# ---------------------------------------------------

def simple_Player_Profile_Bank_Account_json_iterator(json_obj, needed_vars):
    """
    Geeft Bank_Account_ID’s terug uit de JSON-kolom Player_Profile_Bank_Accounts.
    Voorbeeld:
    [{"Bank_Account_ID":"1a","Bank_Account_Datetime":"2025-06-19T13:10:39Z","Bank_Account_Active":"true"}]
    """
    # Als de waarde None, 'null', 'NULL' of leeg is gewoon overslaan
    json_obj = _safe_load_json_relaxed(json_obj)

    # guard: skip if None
    # print('json_obj_2:', json_obj)
    if json_obj is None:
        return
    # guard: skip if not a list/dict or empty
    if not isinstance(json_obj, (list, dict)) or not json_obj:
        return
    
    # als het een list is, door de lijst loopen
    if isinstance(json_obj, list):
        for rec in json_obj:
            if isinstance(rec, dict):
                bank_account_ID = rec.get("Bank_Account_ID")
                if bank_account_ID:
                    yield str(bank_account_ID)
    # zo niet, dan bevat het direct een dict
    else:
        bank_account_ID = json_obj.get("Bank_Account_ID")
        if bank_account_ID:
            yield str(bank_account_ID)
    return

def simple_RG_Class_Value_from_FLAG_RG_CLASS_json_iterator(json_obj, needed_vars):
    """
    Eigenlijk hoeft dit geen iterator te zijn, want er is maar 1 waarde per record.

    Geeft Bank_Account_ID’s terug uit de JSON-kolom Player_Profile_Bank_Accounts.
    Voorbeeld:
    "{""RG_Class_Value"": ""NO_RISK_ASSIGNED"", ""RG_Class_Datetime"": ""2017-02-28T19:17:28Z""}"
    """
    # Als de waarde None, 'null', 'NULL' of leeg is gewoon overslaan
    # print('json_obj_1:', json_obj)
    if json_obj in (None, "null", "NULL", ""):
        return
    json_obj = _safe_load_json_relaxed(json_obj)
    # print('json_obj_2:', json_obj)
    if isinstance(json_obj, dict):
        RG_Class_Value = json_obj.get("RG_Class_Value")
        if RG_Class_Value:
            yield str(RG_Class_Value)
    return

from dateutil import parser as dateparser

def iter_limit_values(json_obj, *, type_: str):
    """
    Generieke iterator voor limit-JSON kolommen zoals:
      - Limit_Deposit
      - Limit_Login
      - Limit_Balance
      - Limit_Participation
      - Limit_Game_Type

    Yield: (ts: datetime|None, value: float|str|None, time_window: str|None)

    type_ ∈ {"deposit", "login", "balance", "participation", "game_type"}
    """
    # Parse de JSON veilig en normaliseer naar lijst
    json_obj = _safe_load_json_relaxed(json_obj)
    if json_obj is None:
        return
    if isinstance(json_obj, dict):
        json_obj = [json_obj]
    if not isinstance(json_obj, list):
        return

    # Type-specifieke keys
    if type_ == "deposit":
        ts_keys = ["Deposit_Request_Datetime", "Deposit_Start_Datetime"]
        val_key = "Deposit_Amount"
        win_key = "Deposit_Time_Window"
    elif type_ == "login":
        ts_keys = ["Login_Request_Datetime", "Login_Start_Datetime"]
        val_key = "Login_Duration"
        win_key = "Login_Time_Window"
    elif type_ == "balance":
        ts_keys = ["Balance_Request_Datetime", "Balance_Start_Datetime"]
        val_key = "Balance_Amount"
        win_key = None
    elif type_ == "participation":
        ts_keys = ["Participation_Request_Datetime", "Participation_Start_Datetime"]
        val_key = "Participation_Amount"
        win_key = "Participation_Time_Window"
    elif type_ == "game_type":
        # Limit_Game_Type is een lijst/dict met o.a. request/start datetime en een game-type limit.
        # Time window kan ontbreken; laat None als het er niet is.
        ts_keys = ["Game_Type_Request_Datetime", "Game_Type_Start_Datetime"]
        val_key = "Game_Type_Type"  # stringMedium: vrije tekst, geen enum
        win_key = "Game_Type_Time_Window"  # indien aanwezig
    else:
        raise ValueError(f"Onbekend limit-type: {type_}")

    def _parse_ts(ts_raw):
        if ts_raw is None:
            return None
        try:
            s = str(ts_raw).strip()
            if not s or s.lower() in {"nan", "none", "null"}:
                return None
            return dateparser.parse(s)
        except Exception:
            return None

    def _parse_val(v):
        if v is None:
            return None
        # game_type is meestal geen numeriek; geef raw (str) terug
        if type_ == "game_type":
            if isinstance(v, str):
                s = v.strip()
                return None if (not s or s.lower() in {"nan", "none", "null"}) else s
            return v

        # overige types: probeer naar float te coerceden
        try:
            if isinstance(v, str):
                s = v.strip()
                if not s or s.lower() in {"nan", "none", "null"}:
                    return None
                s = s.replace(",", ".")
                return float(s)
            return float(v)
        except Exception:
            return v

    for rec in json_obj:
        if not isinstance(rec, dict):
            continue

        # Neem de eerste beschikbare timestamp-key
        ts_raw = None
        for k in ts_keys:
            vv = rec.get(k)
            if vv not in (None, "", "null"):
                ts_raw = vv
                break
        ts = _parse_ts(ts_raw)

        val = _parse_val(rec.get(val_key))

        win = None
        if win_key:
            w = rec.get(win_key)
            if w is not None:
                win = str(w).strip() if isinstance(w, str) else str(w)

        yield (ts, val, win)


def iter_transaction_ids_from_Game_Transactions(
    json_obj: Any,
    verbose = False,
) -> Iterable[str]:
    """
    Iterator over alle Transaction_ID's in de kolom 'Game_Transactions'.
    - Verwacht input uit CSV als string met JSON of reeds geparste dict/list.
    - Structuur zoals nu gebruikt in CSV:
        {
          "Game_Transaction": [
            {"Player_Profile_ID": "12345", "Transaction_ID": "tx1"},
            {"Player_Profile_ID": "12345", "Transaction_ID": "tx2"},
            ...
          ]
        }

    Filtert optioneel op een specifieke Player_Profile_ID (exacte stringvergelijking).
    Bij ongeldige/lege JSON levert de iterator niets op.

    Parameters
    ----------
    json_obj : Any
        De rauwe waarde uit de CSV-kolom 'Game_Transactions' (string/dict/list/None).
    player_profile_id : Optional[str]
        Indien gezet: alleen transacties voor deze speler worden teruggegeven.

    Yields
    ------
    str
        Elke gevonden Transaction_ID (één voor één).
    """
    if json_obj is None or pd.isna(json_obj) or str(json_obj).strip() == "":
        print('json_obj is empty/NaN, nothing to parse') if verbose else None
        return

    try:
        # Indien de input een string is: parse naar Python object
        if isinstance(json_obj, str):
            json_obj = json.loads(json_obj)
    except Exception:
        # Niet te parsen → niets teruggeven
        return

    # De CSV-structuur plaatst items onder key "Game_Transaction"
    if isinstance(json_obj, dict):
        items = json_obj.get("Game_Transaction", [])
    elif isinstance(json_obj, list):
        # Mocht iemand direct een lijst hebben gestort
        items = json_obj
    else:
        items = []
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        transaction_ID = item.get("Transaction_ID")
        speler_id = item.get("Player_Profile_ID")

        if isinstance(transaction_ID, str) and isinstance(speler_id, str):
            yield speler_id, transaction_ID


def iter_part_ids_from_Bet_Parts(json_veld: str) -> Iterator[Tuple[str, dict]]:
    """
    Parse de JSON-string uit de kolom 'Bet_Parts' en yield elk (Part_ID, Part_obj)-paar.

    Voorbeeld input:
    "{""Part"":[{""Part_ID"":""first"",""Part_Event"":""..."",...}, {""Part_ID"":""second"",...}]}"

    Output:
    Yields tuples:
        ("first", {...}),
        ("second", {...}),
        etc.
    """
    if json_veld is None or pd.isna(json_veld) or str(json_veld).strip() == "":
        print('json_veld is empty/NaN, nothing to parse')
        return

    try:
        data = json.loads(json_veld)
    except Exception:
        return

    parts = data.get("Part")
    if isinstance(parts, list):
        for part in parts:
            part_id = part.get("Part_ID")
            yield (part_id, part)

def iter_player_profile_ids_from_Bet_Transactions(json_veld: str, verbose = False) -> Iterator[str]:
    """
    Parse de JSON-string uit de kolom 'Bet_Transactions' en yield elke Player_Profile_ID.

    Voorbeeld input:
    "{""Bet_Transaction"":[{""Player_Profile_ID"":""2914..."",""Transaction_ID"":""20d2...""}]}"
    """
    if json_veld is None or pd.isna(json_veld) or str(json_veld).strip() == "":
        return

    try:
        data = json.loads(json_veld)
    except Exception:
        return

    items = data.get("Bet_Transaction")
    if isinstance(items, list):
        for obj in items:
            pid = obj.get("Player_Profile_ID")
            if pid is not None:
                yield pid
    elif isinstance(items, dict):
        pid = items.get("Player_Profile_ID")
        if pid is not None:
            yield pid


def iter_transaction_ids_from_Bet_Transactions(json_veld: str, verbose = False) -> Iterator[Tuple[str, str]]:
    """
    Parse de JSON-string uit de kolom 'Bet_Transactions' en yield (Player_Profile_ID, Transaction_ID) tuples.

    Voorbeeld input:
    "{""Bet_Transaction"":[{""Player_Profile_ID"":""2914..."",""Transaction_ID"":""20d2...""}]}"
    """
    if json_veld is None or pd.isna(json_veld) or str(json_veld).strip() == "":
        print('json_veld is empty/NaN, nothing to parse') if verbose else None
        return

    try:
        data = json.loads(json_veld)
    except Exception:
        return

    items = data.get("Bet_Transaction")
    if isinstance(items, list):
        for obj in items:
            pid = obj.get("Player_Profile_ID")
            tid = obj.get("Transaction_ID")
            if pid is not None and tid is not None:
                yield pid, tid
    elif isinstance(items, dict):
        pid = items.get("Player_Profile_ID")
        tid = items.get("Transaction_ID")
        if pid is not None and tid is not None:
            yield pid, tid

def iter_part_live_flags_from_Bet_Parts(json_obj, verbose: bool = False):
    """
    Iterate over *parts* in WOK_Bet.Bet_Parts and yield the Part_Live flag (bool) per part.

    Expected shapes (after _safe_load_json_relaxed):
      - {"Part": [ {...}, {...} ]}
      - {"Part": {...}}  (single part dict)
      - direct list of part dicts: [ {...}, {...} ]
      - single part dict: {...}

    Yields
    ------
    bool
        Part_Live per part (True/False) when present & parseable.
        Missing/unparseable values are skipped (yield nothing).
    """
    obj = _safe_load_json_relaxed(json_obj)
    if obj is None:
        return

    # Normalize to list of part dicts
    parts = None
    if isinstance(obj, dict):
        parts = obj.get("Part") if "Part" in obj else obj
    else:
        parts = obj

    if isinstance(parts, dict):
        parts = [parts]
    elif not isinstance(parts, list):
        return

    for part in parts:
        if not isinstance(part, dict):
            continue

        v = part.get("Part_Live")
        if v is None:
            continue

        if isinstance(v, bool):
            yield v
            continue

        if isinstance(v, (int, float)):
            yield bool(int(v))
            continue

        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "t", "1", "yes", "y"}:
                yield True
                continue
            if s in {"false", "f", "0", "no", "n"}:
                yield False
                continue

        if verbose:
            print(f"[iter_part_live_flags_from_Bet_Parts] Unparseable Part_Live={v!r}")


def get_list_of_response_ids_from_Responses_list(json_obj: Any, verbose = False) -> Optional[List]:
    """
    Haal een list van Response_IDs uit de JSON-kolom 'Responses' van WOK_Complaint.
    - Verwacht input uit CSV als string met JSON of reeds geparste dict/list.
    - Structuur zoals nu gebruikt in CSV:
        [{
          "Response": [{
            "Response_ID": "start1234",
            "Response_Type": "Complaint Closing",
            "Response_Description": "Klantgegevens aangepast",
            "Response_Datetime": "2025-07-14T10:38:06Z"
          }]
        }]
    Bij ongeldige of lege JSON levert de functie None op.
    Parameters
    ----------
    list_of_json : Any
        De rauwe waarde uit de CSV-kolom 'Responses' (string/dict/list/None).
    Returns
    -------
    Optional[str]
        De gevonden Response_ID of None als niet gevonden.
    """
    print('json_obj:', json_obj) if verbose else None
    if json_obj is None or pd.isna(json_obj) or json_obj == '''[{"Response": null}]''' or json_obj == '''[{""Response"": null}]''' or json_obj == '''[{'Response': null}]''':
        print('json_obj is None, mogelijk is er wel een complaint dus doorgaan') if verbose else None
        return None
    try:
        # Indien de input een string is: parse naar Python object
        if isinstance(json_obj, str):
            print('het is een json string') if verbose else None
            list_of_json = json.loads(json_obj)
            print(f'list_of_json: {type(list_of_json)}') if verbose else None
        else:
            print('het is geen json string') if verbose else None
            list_of_json = json_obj
    except Exception:
        # Niet te parsen → niets teruggeven
        return None
    response_id_list = []
    for json_obj in list_of_json:
        print(f"list: {list_of_json}") if verbose else None
        if verbose:
           print()
           print(f"json object: {json_obj}")
           print()
        if json_obj is None:
            return None

        # De CSV-structuur plaatst item onder key "Response"
        if isinstance(json_obj, dict):
            list_of_responses = json_obj.get("Response")
        else:
            1/0
        for response in list_of_responses:
            response_id = response.get("Response_ID")
            response_id_list.append(response_id)
    if verbose:
        print(f"response_id_list: {response_id_list}")
    return response_id_list

# Dit is dus eigenlijk niet nodig want iedere complaint telt exact één keer voor de speler.

# def iter_responses_from_complaints(json_obj: Any) -> Iterable[str]:
#     """
#     Iterator over alle Response_ID's in de kolom 'Responses' van WOK_Complaint.
#     - Verwacht input uit CSV als string met JSON of reeds geparste dict/list.
#     - Structuur zoals nu gebruikt in CSV:
#         {
#           "Response": [
#             {
#               "Response_ID": "start1234",
#               "Response_Type": "Complaint Closing",
#               "Response_Description": "Klantgegevens aangepast",
#               "Response_Datetime": "2025-07-14T10:38:06Z"
#             },
#             ...
#           ]
#         }

#     Bij ongeldige of lege JSON levert de iterator niets op.

#     Parameters
#     ----------
#     json_obj : Any
#         De rauwe waarde uit de CSV-kolom 'Responses' (string/dict/list/None).

#     Yields
#     ------
#     str
#         Elke gevonden Response_ID (één voor één).
#     """
#     if json_obj is None:
#         return

#     try:
#         # Indien de input een string is: parse naar Python object
#         if isinstance(json_obj, str):
#             json_obj = json.loads(json_obj)
#     except Exception:
#         # Niet te parsen → niets teruggeven
#         return

#     # De CSV-structuur plaatst items onder key "Response"
#     if isinstance(json_obj, dict):
#         items = json_obj.get("Response", [])
#     elif isinstance(json_obj, list):
#         # Mocht iemand direct een lijst hebben gestort
#         items = json_obj
#     else:
#         items = []

#     if not isinstance(items, list):
#         return

#     for item in items:
#         if not isinstance(item, dict):
#             continue
#         response_id = item.get("Response_ID")

#         if isinstance(response_id, str):
#             yield response_id