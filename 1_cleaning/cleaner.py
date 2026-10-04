# cleaner.py
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
import re
import json
from pprint import pformat
from operator_filters import operator_specific_filter  # NEW: pre-validation operator filter
import io
import csv  # alleen voor duidelijkheid/consistentie met quoting-terminologie
import os
# Hergebruik schema + normalizers (schema-key resolver gebruikt path_finding met fallback)
from check_format import organisatie_SCHEMA_BY_FILE, clean_row_for_file, resolve_schema_key
from transaction_dedup import dedup_table_name

from reading_difficult_json import iter_transaction_ids_from_Game_Transactions

MISSING_TOKENS = {"""""","", " ", "null", "NULL", "NaN", "N/A"}

# Alias-map voor kolomnamen per schema-bestand; alleen naam-variaties corrigeren.
# NB: puur hernoemen vóór validatie; geen type/waarde mutaties.
# COL_ALIASES = {
#     "WOK_Player_Limits.csv": {
#         "Limit_deposit": "Limit_Deposit",
#         "Limit_balance": "Limit_Balance",
#         "Limit_participation": "Limit_Participation",
#         "Limit_loss": "Limit_Loss",
#     },
#     "WOK_Bet.csv": {
#         "Bet_parts": "Bet_Parts",
#     },
# }

COL_ALIASES = {
    "WOK_Bet.csv": {
        "record_id": "Record_ID",
        "extraction_date": "Extraction_Date",
        "operator_id": "Operator_ID",
        "data_safe_id": "Data_Safe_ID",
        "replaced_record_id": "Replaced_Record_ID",
        "bet_id": "Bet_ID",
        "bet_start_datetime": "Bet_Start_Datetime",
        "bet_cancellation_reason": "Bet_Cancellation_Reason",
        "bet_type": "Bet_Type",
        "bet_xy": "Bet_XY",
        "bet_commission": "Bet_Commission",
        "bet_status": "Bet_Status",
        "bet_parts": "Bet_Parts",
        "bet_total_stake": "Bet_Total_Stake",
        "bet_transactions": "Bet_Transactions",
        "_event_date": "_event_date",
        "_filter_label": "_filter_label",
        "_write_timestamp": "_write_timestamp",
        "_partitiontime": "_partitiontime",
    },

    "WOK_Complaint.csv": {
        "record_id": "Record_ID",
        "extraction_date": "Extraction_Date",
        "operator_id": "Operator_ID",
        "data_safe_id": "Data_Safe_ID",
        "complaint_id": "Complaint_ID",
        "complaint_type": "Complaint_Type",
        "complaint_datetime": "Complaint_Datetime",
        "complaint_player_id": "Complaint_Player_ID",
        "responses": "Responses",
        "response": "Responses",
    },

    "WOK_Game_Session.csv": {
        "record_id": "Record_ID",
        "extraction_date": "Extraction_Date",
        "operator_id": "Operator_ID",
        "data_safe_id": "Data_Safe_ID",
        "replaced_record_id": "Replaced_Record_ID",
        "game_id": "Game_ID",
        "game_session_id": "Game_Session_ID",
        "game_session_start_datetime": "Game_Session_Start_Datetime",
        "game_session_end_datetime": "Game_Session_End_Datetime",
        "game_session_commission": "Game_Session_Commission",
        "game_session_rounds": "Game_Session_Rounds",
        "game_session_rounds_won": "Game_Session_Rounds_Won",
        "game_transactions": "Game_Transactions",
        "_event_date": "_event_date",
        "_write_timestamp": "_write_timestamp",
        "_partitiontime": "_partitiontime",
    },

    "WOK_Game.csv": {
        "record_id": "Record_ID",
        "extraction_date": "Extraction_Date",
        "operator_id": "Operator_ID",
        "data_safe_id": "Data_Safe_ID",
        "game_id": "Game_ID",
        "game_type": "Game_Type",
        "game_commercial_name": "Game_Commercial_Name",
        "game_datetime_introduction": "Game_Datetime_Introduction",
        "game_datetime_active": "Game_Datetime_Active",
        "game_datetime_inactive": "Game_Datetime_Inactive",
    },

    "WOK_Intervention.csv": {
        "record_id": "Record_ID",
        "extraction_date": "Extraction_Date",
        "operator_id": "Operator_ID",
        "data_safe_id": "Data_Safe_ID",
        "replaced_record_id": "Replaced_Record_ID",
        "player_profile_id": "Player_Profile_ID",
        "intervention_id": "Intervention_ID",
        "intervention_begin_datetime": "Intervention_Begin_Datetime",
        "intervention_end_datetime": "Intervention_End_Datetime",
        "intervention_type": "Intervention_Type",
        "intervention_cause": "Intervention_Cause",
        "intervention_owner": "Intervention_Owner",
    },

    "WOK_Net_Deposit_Threshold.csv": {
        "record_id": "Record_ID",
        "extraction_date": "Extraction_Date",
        "operator_id": "Operator_ID",
        "data_safe_id": "Data_Safe_ID",
        "replaced_record_id": "Replaced_Record_ID",
        "player_profile_id": "Player_Profile_ID",
        "net_deposit_threshold_value": "Net_Deposit_Threshold_Value",
        "net_deposit_threshold_datetime": "Net_Deposit_Threshold_Datetime",
    },

    "WOK_Operator.csv": {
        "record_id": "Record_ID",
        "extraction_date": "Extraction_Date",
        "operator_id": "Operator_ID",
        "data_safe_id": "Data_Safe_ID",
        "concerned_date": "Concerned_Date",
        "replaced_record_id": "Replaced_Record_ID",
        "totals": "Totals",
        "_event_date": "_event_date",
        "_write_timestamp": "_write_timestamp",
        "_partitiontime": "_partitiontime",
        "_filter_label": "_filter_label",
    },

    "WOK_Player_Account_Transaction.csv": {
        "record_id": "Record_ID",
        "extraction_date": "Extraction_Date",
        "operator_id": "Operator_ID",
        "data_safe_id": "Data_Safe_ID",
        "replaced_record_id": "Replaced_Record_ID",
        "player_profile_id": "Player_Profile_ID",
        "transaction_id": "Transaction_ID",
        "transaction_datetime": "Transaction_Datetime",
        "transaction_amount": "Transaction_Amount",
        "transaction_deposit_instrument": "Transaction_Deposit_Instrument",
        "transaction_type": "Transaction_Type",
        "transaction_status": "Transaction_Status",
    },

    "WOK_Player_Flags.csv": {
        "record_id": "Record_ID",
        "extraction_date": "Extraction_Date",
        "operator_id": "Operator_ID",
        "data_safe_id": "Data_Safe_ID",
        "player_profile_id": "Player_Profile_ID",
        "flag_rg_class": "Flag_RG_Class",
    },

    "WOK_Player_Limits.csv": {
        "record_id": "Record_ID",
        "extraction_date": "Extraction_Date",
        "operator_id": "Operator_ID",
        "data_safe_id": "Data_Safe_ID",
        "replaced_record_id": "Replaced_Record_ID",
        "player_profile_id": "Player_Profile_ID",
        "limit_deposit": "Limit_Deposit",
        "limit_participation": "Limit_Participation",
        "limit_login": "Limit_Login",
        "limit_game_type": "Limit_Game_Type",
        "limit_balance": "Limit_Balance",
    },

    "WOK_Player_Profile.csv": {
        "record_id": "Record_ID",
        "extraction_date": "Extraction_Date",
        "operator_id": "Operator_ID",
        "data_safe_id": "Data_Safe_ID",
        "player_profile_id": "Player_Profile_ID",
        "player_profile_registration_datetime": "Player_Profile_Registration_Datetime",
        "player_profile_dob": "Player_Profile_DOB",
        "player_profile_modified": "Player_Profile_Modified",
        "player_profile_status": "Player_Profile_Status",
        "player_profile_eod_balance": "Player_Profile_EOD_Balance",
        "player_profile_bank_account": "Player_Profile_Bank_Account",
        "_event_date": "_event_date",
        "_filter_label": "_filter_label",
        "_write_timestamp": "_write_timestamp",
        "_partitiontime": "_partitiontime",
        "_content_hash": "_content_hash",
        "_sub_player_profile_id": "_sub_player_profile_id",
    },
}


# Matches bijv.: "... [row 123] <FIELD>: ontbreekt (optioneel)"
OPTIONAL_MISSING_RE = re.compile(r"\[row\s+\d+\]\s+(?P<field>.+?):\s+ontbreekt\s+\(optioneel\)$")

PIPE_QUOTED_HEADER_RE = re.compile(r'^\|[^|]*\|(?:,\|[^|]*\|)*\s*$')

def detect_csv_wrapping(path: str | Path, *, max_lines: int = 2) -> dict:
    """
    Detecteer of de CSV 'pipe-quoted' is: |value|,|value2|,...
    Retourneert dict met keys: is_pipe_quoted (bool), sep (str), quotechar (str|None), reason (str)
    """
    p = Path(path)
    is_pipe = False
    reason = "default"
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            # kijk naar header (en evt. eerste datarij) voor robuuste detectie
            lines = [next(f) for _ in range(max_lines)]
    except StopIteration:
        lines = []
    except Exception:
        lines = []

    header = lines[0].rstrip("\r\n") if lines else ""
    data = lines[1].rstrip("\r\n") if len(lines) > 1 else ""

    if PIPE_QUOTED_HEADER_RE.match(header):
        # extra zekerheid: als er ook in 1e datarij duidelijke |...|,|...| patronen zitten
        if data and data.count("|") >= 2 and "," in data:
            is_pipe = True
            reason = "header+data pattern"
        else:
            is_pipe = True
            reason = "header pattern"

    return {
        "is_pipe_quoted": is_pipe,
        "sep": ",",
        "quotechar": "|" if is_pipe else None,
        "reason": reason,
    }


def _log(logger, msg: str) -> None:
    if logger:
        logger.info(msg)
    else:
        # gaat naar de slurm, ook mooi
        print(msg, flush=True)

def make_warn_logger(raw_filename: str, logger, seen_optional_missing: set[str], inc_warn):
    """
    Geeft een warn(msg) terug die alle WARNs logt, behalve herhaalde
    'ontbreekt (optioneel)' voor hetzelfde veld in hetzelfde bestand.
    inc_warn() wordt alleen aangeroepen als er daadwerkelijk gelogd is.
    """
    def _warn(msg: str):
        m = OPTIONAL_MISSING_RE.search(msg)
        if m:
            field = m.group("field")
            key = f"{raw_filename}::{field}::optional-missing"
            if key in seen_optional_missing:
                return
            seen_optional_missing.add(key)
        # tellen + loggen
        inc_warn()
        if logger:
            logger.warning(f"WARN {raw_filename} {msg}")
        else:
            print(f"WARN {raw_filename} {msg}", flush=True)
    return _warn

def _is_organisatie_semicolon_format(input_csv: str) -> bool:
    """True als het bestand de organisatie-export is (;-gescheiden) i.p.v. de oudere komma-data."""
    try:
        with open(input_csv, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
    except Exception:
        return False
    return header.count(";") > header.count(",")


def _clean_organisatie_passthrough(input_csv: str, cleaned_dir: str, *, chunksize: int = 200_000,
                           logger=None, check_only: bool = False,
                           only_first_chunk: bool = False) -> dict:
    """Minimale cleaning voor het organisatie-relationele formaat.

    De organisatie-export is al schoon/relationeel/getypeerd, dus we slaan de oude CamelCase- en
    JSON-schema-normalisatie over: lees ;-CSV (komma-decimalen → float, `NULL` → NaN), strip
    whitespace op tekstkolommen, en schrijf als standaard komma-CSV (snake_case behouden) zodat
    de format-bewuste reader het downstream gewoon oppikt.
    """
    raw_filename = Path(input_csv).name
    out_path = Path(cleaned_dir) / raw_filename
    if check_only:
        return {"file": raw_filename, "rows": 0, "mode": "organisatie_check_only"}
    Path(cleaned_dir).mkdir(parents=True, exist_ok=True)
    rows_total, first = 0, True
    sep = ";" if _is_organisatie_semicolon_format(input_csv) else ","
    header = pd.read_csv(input_csv, sep=sep, nrows=0)
    # Identifiers are text, including purely numeric IDs with leading zeroes.
    id_types = {c: str for c in header.columns if c.strip().lower().endswith("_id")}
    for chunk in pd.read_csv(input_csv, sep=sep, decimal="," if sep == ";" else ".",
                             dtype=id_types, na_values=["", "NULL", "null", "NaN", "nan", "N/A", "n/a", "None", "none"],
                             keep_default_na=False, low_memory=False, chunksize=chunksize):
        for c in chunk.select_dtypes(include=["object", "string"]).columns:
            chunk[c] = chunk[c].map(lambda x: x.strip() if isinstance(x, str) else x)
        chunk.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
        first, rows_total = False, rows_total + len(chunk)
        if only_first_chunk:
            break
    if logger:
        logger.info(f"[CLEAN-organisatie] {raw_filename} → {out_path}  ({rows_total:,} rijen, pass-through)")
    else:
        print(f"[CLEAN-organisatie] {raw_filename} → {out_path.name}  ({rows_total:,} rijen, pass-through)", flush=True)
    return {"file": raw_filename, "rows": rows_total, "output": str(out_path), "mode": "organisatie_passthrough"}


def clean_csv_streaming(input_csv: str | Path,
                        cleaned_dir: str | Path,
                        *,
                        chunksize: int = 200_000,
                        logger=None,
                        check_only: bool = False,
                        sample_rows: int = 10,
                        only_first_chunk: bool = False,
                        verbose = False,
                        local_test_output=False) -> dict:
    """
    Stream-clean a (large) CSV file in chunks.

    Modes
    -----
    - check_only=True:
        * Per chunk: controleer de EERSTE `sample_rows` rijen (ruwe strings) tegen organisatie_SCHEMA_BY_FILE.
        * Geen mutaties en er wordt NIET geschreven.
        * Logging:
            - Optional velden: géén waarschuwing als ze ontbreken of misformat zijn (dit komt uit check_format).
            - Niet-optional: waarschuwing bij ontbreken/misformat (conversie wordt alleen gemeld, niet toegepast).

    - check_only=False (default):
        * Per chunk: normaliseer ALLE rijen via `clean_row_for_file(..., mode="clean")`.
            - Optional: bij misformat/missing → waarschuwing; converteren waar mogelijk.
            - Niet-optional: bij misformat/missing → ERROR (raise), ook als converteren zou lukken (strikt).
        * Daarna generieke basis-cleaning (missings → None, trimmen) en schrijven.

    Notities
    --------
    - We lezen chunks als ruwe strings (dtype=str, keep_default_na=False, na_filter=False) zodat pandas niet
      alvast gaat parsen; de schema-normalisatie bepaalt het gewenste format.
    - Als er geen schema is voor dit bestand (bestandsnaam-key), slaan we schema-checks/normalisatie over.
    """
    t0 = time.perf_counter()
    input_csv = str(input_csv)
    cleaned_dir = str(cleaned_dir)
    raw_filename = Path(input_csv).name

    # ► organisatie-export (relationeel, ;-gescheiden): route langs de oude schema-machine heen.
    if _is_organisatie_semicolon_format(input_csv) or (
        dedup_table_name(raw_filename)
        and not resolve_schema_key(raw_filename, organisatie_SCHEMA_BY_FILE.keys())[0]
    ):
        return _clean_organisatie_passthrough(input_csv, cleaned_dir, chunksize=chunksize,
                                      logger=logger, check_only=check_only,
                                      only_first_chunk=only_first_chunk)

    # ► Bepaal schema-key via path_finding (met fallback) en haal schema op
    schema_key, chunk_suffix = resolve_schema_key(raw_filename, organisatie_SCHEMA_BY_FILE.keys())
    
    # remove .csv from schema_key
    name = schema_key[0:-4]
    output_csv = cleaned_dir + '/' + name + chunk_suffix + '.csv'
    schema_for_file = organisatie_SCHEMA_BY_FILE.get(schema_key, {})

    if schema_for_file:
        _log(logger, f"[INFO] schema gebruikt: {schema_key} (bestand: {raw_filename})")
    else:
        _log(logger, f"[INFO] geen schema gevonden voor {raw_filename} (probeerde key: {schema_key}); schema-check/normalisatie wordt overgeslagen.")

    # ► Verwachte basis-kolommen uit schema (flat + json-basiskolommen)
    expected_base_cols: set[str] = set()
    if schema_for_file:
        for key in schema_for_file.keys():
            base = key.split(".", 1)[0]
            base = base.split("[]", 1)[0]
            expected_base_cols.add(base)

    if not check_only:
        Path(cleaned_dir).mkdir(parents=True, exist_ok=True)

    rows_total = 0
    warns_total = 0
    first_write = True
    bytes_in = Path(input_csv).stat().st_size if Path(input_csv).exists() else 0.0
    mode_label = "CHECK-ONLY" if check_only else "CLEAN"
    _log(logger, f"[{mode_label}] {raw_filename} ({bytes_in/1_048_576:,.2f} MB), chunksize={chunksize:,}, sample_rows={sample_rows}")

    warned_missing_once = False
    seen_optional_missing: set[str] = set()

    def _inc_warn():
        nonlocal warns_total
        warns_total += 1

    warn_fn = make_warn_logger(raw_filename, logger, seen_optional_missing, _inc_warn)
    
    # éénmalige debug-dump bij 1e error
    first_error_state = {"dumped": False, "skipped": 0}

    # how many rows the operator filter dropped across chunks
    prefilter_dropped_total = 0

    # --- auto-detectie van pipe-quoted CSV ---
    sniff = detect_csv_wrapping(input_csv)
    if sniff["is_pipe_quoted"]:
        _log(logger, f'[INFO] detected pipe-quoted CSV in {raw_filename} '
                     f'(reason={sniff["reason"]}); parsing with sep="," and quotechar="|".')
    else:
        _log(logger, f'[INFO] detected regular CSV in {raw_filename}; parsing with default sep=",".')

    # Bouw de read_csv kwargs dynamisch, zodat we streaming behouden
    read_kwargs = dict(
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
        na_filter=False
    )
    if sniff["is_pipe_quoted"]:
        # sep="," is default, maar expliciet maakt intentie duidelijk
        read_kwargs.update(sep=",", quotechar=sniff["quotechar"])
        # engine='python' kan bij exotische quoting helpen; meestal niet nodig, maar kan je desgewenst aanzetten:
        # read_kwargs.update(engine="python")

    this_is_first_chunk = True


# ######################
#     print('#################### start gekke test #######################')
#     test_chunksizes = [250, 500, 1000, 2500, 5000, 10_000, 25_000, 50_000, 100_000, 200_000, 400_000, 800_000, 1_600_000, 3_200_000]
#     for test_chunksize in test_chunksizes:
#         print(f'Testing chunksize: {test_chunksize}')
#         read_kwargs['chunksize'] = test_chunksize
#         test_t_chunk = time.perf_counter()
#         ############### gekke test loop


    # loop over chunks
    for i, df in enumerate(pd.read_csv(input_csv, **read_kwargs), start=1):
        t_chunk = time.perf_counter()
        
        # Operator-specific pre-validation filter (to keep things moving)
        df, filt_report = operator_specific_filter(
            df,
            raw_filename=raw_filename,
            chunk_index=i,
            rows_processed_so_far=rows_total,
            logger=logger
        )
        if filt_report.get("applied"):
            # Log once per chunk if it *would* or *did* drop rows
            wd = filt_report.get("would_drop", 0)
            dr = filt_report.get("dropped", 0)
            frac = filt_report.get("fraction", 0.0)
            mode = filt_report.get("mode", "apply")
            if wd > 0 or dr > 0:
                msg = (f"[FILTER] {raw_filename} chunk {i}: "
                    f"operator={filt_report.get('operator')} rule={filt_report.get('rule')} "
                    f"mode={mode} dropped={dr} (would_drop={wd}, {frac:.1%}) "
                    f"examples_global={filt_report.get('examples')}")
                if logger:
                    logger.warning(msg)
                else:
                    print(msg, flush=True)
            prefilter_dropped_total += dr

        if schema_for_file:
            # alle hoofdletters vertalen naar normale letters
            df.rename(columns=lambda c: c.strip().lower(), inplace=True)
            # Kolom-aliassen hernoemen volgens COL_ALIASES
            alias_map = COL_ALIASES.get(schema_key, {})
            if alias_map:
                # Alleen hernoemen wat daadwerkelijk aanwezig is (veilig/idempotent)
                present = set(df.columns)
                to_apply = {src: dst for src, dst in alias_map.items() if src in present and src != dst}
                if to_apply:
                    df.rename(columns=to_apply, inplace=True)

        # Eénmalige check op ontbrekende kolommen t.o.v. schema (alleen als er schema is)
        if schema_for_file and not warned_missing_once:
            present_cols = set(df.columns)
            missing_cols = sorted(col for col in expected_base_cols if col not in present_cols)
            
            # de alias map is knullig, laten we een dict maken die hoofdletterissues oplost:
            fixed = []
            for missing_column in missing_cols:
                for c in present_cols:
                    if missing_column.lower() == c.lower():
                        warn_fn(f"Kolom '{missing_column}' ontbreekt, maar gevonden met verkeerde hoofdletters als '{c}'")
                        df.rename(columns={c: missing_column}, inplace=True)
                        fixed.append(missing_column)
                        break

            # verwijder pas daarna de gefixte kolommen uit missing_cols
            missing_cols = [m for m in missing_cols if m not in fixed]

            if missing_cols:
                if logger:
                    logger.warning(f"WARN {raw_filename} ontbrekende kolommen t.o.v. schema({schema_key}): {', '.join(missing_cols)}")
                else:
                    print(f"WARN {raw_filename} ontbrekende kolommen t.o.v. schema({schema_key}): {', '.join(missing_cols)}", flush=True)
            warned_missing_once = True
        

        # Add an extra checks for JSON columns that might be wrongly parsed
        if this_is_first_chunk == True:            
            # JSON-specifieke check op correctheid
            # check of er wel meerdere Player_Profile_IDs in Game_Transactions staan 
            if schema_key[0:16] == "WOK_Game_Session":
            # 1.check of er wel meerdere Player_Profile_IDs in Game_Transactions staan
                try:
                    gemiddeld_aantal_per_rij = (
                        df["Game_Transactions"]
                        .astype(str)
                        .apply(lambda s: s.count("Transaction_ID"))
                        .mean()
                    )
                    logger.info(f"Gemiddeld aantal 'Transaction_ID'-blokken per rij: {gemiddeld_aantal_per_rij:.2f}")
                    if gemiddeld_aantal_per_rij <= 1:
                        logger.warning("⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️")
                        logger.warning(f"⚠️  Let op: Gemiddeld {gemiddeld_aantal_per_rij} 'Transaction_ID' per rij.") 
                        logger.warning("Minder dan 1 = mogelijk slechts één Transaction_ID per sessie (waarschijnlijk de laatste), contacteer operator en check evt. data")
                        logger.warning("⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️")
                except Exception as e:
                    logger.error(f"Fout bij berekening gemiddelde aantal Game_Transaction-blokken: {e}")
            
            # 2.check of er wel meerdere Part_IDs in WOK_Bet staan 
            if schema_key[0:8] == "WOK_Bet.":
            # check of er wel meerdere Player_Profile_IDs in Game_Transactions staan
                try:
                    gemiddeld_aantal_per_rij = (
                        df["Bet_Parts"]
                        .astype(str)
                        .apply(lambda s: s.count("Part_ID"))
                        .mean()
                    )
                    logger.info(f"Gemiddeld aantal 'Part_ID'-blokken per rij: {gemiddeld_aantal_per_rij:.2f}")
                    if gemiddeld_aantal_per_rij <= 1:
                        logger.warning("⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️")
                        logger.warning(f"⚠️  Let op: Gemiddeld {gemiddeld_aantal_per_rij} 'Part_ID' per rij.") 
                        logger.warning("Minder dan 1 = mogelijk slechts één Part_ID per sessie (waarschijnlijk de laatste), contacteer operator en check evt. data")
                        logger.warning("⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️")
                except Exception as e:
                    logger.error(f"Fout bij berekening gemiddelde aantal Part_ID's {e}")
        

        # === CHECK-ONLY MODUS: alleen de eerste 'sample_rows' rijen signaleren (zonder mutatie/schrijven) ===
        if check_only:
            if schema_for_file and sample_rows and sample_rows > 0:
                present_cols = set(df.columns)  # organisatie: kolommen die echt in het bestand staan
                n = min(sample_rows, len(df))
                if n > 0:
                    sample = df.head(n).to_dict(orient="records")
                    for j, row in enumerate(sample, start=1):
                        rownum = rows_total + j + 1  # header=1 → eerste data=2

                        def _err(msg: str):
                            # In check-modus: nooit raise, alleen melden
                            if logger:
                                logger.error(f"ERROR {raw_filename} {msg}")
                            else:
                                print(f"ERROR {raw_filename} {msg}", flush=True)

                        _ = clean_row_for_file(
                            row, schema_key,
                            rownum=rownum,    # ← schema_key i.p.v. raw_filename
                            emit_warn=warn_fn,
                            emit_error=_err,
                            mode="check",
                            verbose=verbose,
                            first_row=True,
                            present_columns=present_cols,
                        )
            # alleen tellen/loggen
            rows_total += len(df)
            dt = time.perf_counter() - t_chunk
            rps = (len(df) / dt) if dt > 0 else float("inf")
            _log(logger, f"[CHECK] chunk {i:,}: {len(df):,} rows checked-head({min(sample_rows, len(df))}) in {dt:,.2f}s  ({rps:,.0f} rows/s)")
            
            
            
            continue

        # === CLEAN MODUS: eerst schema-normalisatie (conversie + optional/required regels) ===
        if schema_for_file:
            print('df before cleaning rows:', df) if verbose else None
            present_cols = set(df.columns)  # organisatie: kolommen die echt in het bestand staan
            records = df.to_dict(orient="records")
            # if verbose:
            #     for record in records:
            #         print(record)
            new_records: list[dict] = []

            # helper: stille no-op
            def _silent(*a, **k):
                return

            # Standaard max 5 per-rij waarschuwingen, tenzij sample_rows expliciet is gezet
            max_warn_rows = sample_rows if sample_rows is not None else 5

            for j, row in enumerate(records, start=1):
                # print('1 ROW to clean:', row) if verbose else None
                rownum = rows_total + j + 1

                # Vanaf rij > max_warn_rows: per-rij WARNs dempen
                warn_for_row = warn_fn
                if max_warn_rows is not None and max_warn_rows >= 0 and j > max_warn_rows:
                    warn_for_row = _silent

                def _err(msg: str):
                    # Converteerbare misformat? → normaal een WARN en doorgaan
                    # Dempen we óók na max_warn_rows voor efficiency
                    if "wel geconverteerd" in msg.lower():
                        if not (max_warn_rows is not None and max_warn_rows >= 0 and j > max_warn_rows):
                            if logger:
                                logger.warning(f"WARN {raw_filename} {msg} maar wordt opgelost door cleaner")
                            else:
                                print(f"WARN {raw_filename} {msg}  maar wordt opgelost door cleaner", flush=True)
                        return
                    # Overige fouten blijven hard
                    raise ValueError(f"{raw_filename} {msg}")

                # print('ROW to clean:', row) if verbose else None

                try:
                    new_row = clean_row_for_file(
                        row,
                        schema_key,
                        rownum=rownum,      # ← schema_key i.p.v. raw_filename
                        emit_warn=warn_for_row,              # ⬅️ demper na max_warn_rows
                        emit_error=_err,
                        mode="clean",
                        verbose=verbose,
                        first_row=(j == 1),
                        present_columns=present_cols,
                    )
                    new_records.append(new_row)
                    print('!!!!!!!!! cleaned row:', new_row) if verbose else None
                except ValueError as e:
                    if not first_error_state["dumped"]:
                        # Volledige ruwe rij eenmalig loggen
                        try:
                            row_str = json.dumps(row, ensure_ascii=False)
                        except Exception:
                            row_str = pformat(row, width=120, compact=True)

                        header = f"[DEBUG-ONCE] Eerste error bij cleanen van {raw_filename} op row {rownum}"
                        msg = f"{header}\nException: {e}\nVolledige rij (raw): {row_str}"

                        if logger:
                            logger.error(msg)
                        else:
                            print(msg, flush=True)

                        first_error_state["dumped"] = True

                    # Gedrag behouden: dezelfde error doorgeven
                    raise

            df = pd.DataFrame.from_records(new_records, columns=df.columns)
        # === Daarna generieke basis-cleaning ===
        # normalize missing-like tokens → None
        # pandas ≥2.1 gebruikt DataFrame.map; .applymap is deprecated in 2.1 en VERWIJDERD in 3.0.
        _elementwise = df.map if hasattr(df, "map") else df.applymap
        df = _elementwise(lambda x: None if isinstance(x, str) and x.strip() in MISSING_TOKENS else x)

        
        # strip whitespace op string-kolommen
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)

        # === Schrijven ===
        if local_test_output:
            output_csv = '2_feature_testdata/clean_' + name + '.csv'
            print(f"LOCAL TEST OUTPUT = TRUE: appending cleaned data to {output_csv}")

            # append if file already exists, otherwise write new
            if os.path.exists(output_csv):
                print("APPENDING (via pandas concat) to:", output_csv)

                # Read old file
                old_df = pd.read_csv(output_csv)

                # Strict column check
                if set(old_df.columns) != set(df.columns):
                    print(
                        f"Column mismatch between existing CSV and new df:\n"
                        f"  Existing: {list(old_df.columns)}\n"
                        f"  New:      {list(df.columns)}"
                    )

                # Append safely (aligns by column names)
                combined = pd.concat([old_df, df], axis=0, ignore_index=True)

                # Write back (overwrite)
                combined.to_csv(output_csv, index=False)

            else:
                print("CREATING new cleaned file:", output_csv)
                df.to_csv(output_csv, index=False)

        else:
            df.to_csv(output_csv, mode="w" if first_write else "a", index=False, header=first_write)
        first_write = False

        rows_total += len(df)
        dt = time.perf_counter() - t_chunk
        rps = (len(df) / dt) if dt > 0 else float("inf")
        _log(logger, f"[CLEAN] chunk {i:,}: {len(df):,} rows in {dt:,.2f}s  ({rps:,.0f} rows/s)")
        
        # ####################### read snelheid ook testen!
        # test_dt = time.perf_counter() - test_t_chunk
        # test_rps = (len(df) / test_dt) if test_dt > 0 else float("inf")
        # print(f"[CHECK-READ-TEST] chunk {i:,}: {len(df):,} rows read in {test_dt:,.2f}s  ({test_rps:,.0f} rows/s)")
        # test_t_chunk = time.perf_counter()
        # ############### einde gekke test loop

        if only_first_chunk and this_is_first_chunk:
            _log(logger, f"[CLEAN] only_first_chunk=True, stopping after first chunk.")
            break

        this_is_first_chunk = False

        # ############### einde gekke test loop
        # if i == 4:
        #     print(f'Einde gekke test loop')
        #     break

    dt_all = time.perf_counter() - t0
    mbps = (bytes_in/1_048_576) / dt_all if dt_all > 0 else 0.0

    if prefilter_dropped_total > 0:
        _log(logger, f"[CLEAN] operator-specific pre-filter dropped {prefilter_dropped_total:,} rows in total (pre-validation).")

    if check_only:
        _log(logger, f"[CHECK] total {rows_total:,} rows scanned (head {sample_rows} per chunk) in {dt_all:,.2f}s, warnings={warns_total:,}")
        return {
            "mode": "check_only",
            "rows": rows_total,
            "seconds": dt_all,
            "warnings": warns_total
        }

    _log(logger, f"[CLEAN] wrote → {output_csv}")
    _log(logger, f"[CLEAN] total {rows_total:,} rows in {dt_all:,.2f}s  ({mbps:,.2f} MB/s)")
    return {
        "mode": "clean",
        "rows": rows_total,
        "seconds": dt_all,
        "input_mb": bytes_in/1_048_576,
        "warnings": warns_total
    }
