#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_pipeline.py  (organisatie-versie)
===============================

Pure-Python clean+label pijplijn voor stap 4 (**CLEANING**) uit de README.

Deze module is een *uitgeklede* variant van `clean_and_parse.py` uit de hoofdrepo:
alleen de schoonmaak (`--mode clean`) + de automatische outlier-labelling zijn
overgenomen. Alle feature-engineering / parse-logica (en daarmee de imports van
`path_finding`, `feature_engineering*`) is bewust weggelaten, zodat de cleaning-stap
zelfstandig draait zonder die zware dependencies.

Verschillen met het Snellius-origineel
---------------------------------------
- Geen sbatch / SLURM-array per operator (a..z), geen $TMPDIR-staging met symlinks,
  geen subprocess-wrapper (`clean_runner.py`). De organisatie werkt met lokale CSV's in een map.
- De selectie van te schonen bestanden gebeurt altijd over *alle* WOK-CSV's in de
  inputmap (equivalent aan `--clean-scope all` in het origineel), want zonder
  scenario/feature-keuze is er geen subset om op te beperken.
- Eén Python-aanroep `clean_directory(...)` doet de hele map; geen env-variabelen nodig.

De regel-voor-regel schoonmaak zit in `cleaner.py`, de transactie-deduplicatie in
`transaction_dedup.py` en de labelling in `outlier_labeling.py`.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List, Optional
import logging
import re
import sys

from cleaner import clean_csv_streaming, _is_organisatie_semicolon_format
from check_format import resolve_schema_key, organisatie_SCHEMA_BY_FILE
from outlier_labeling import label_outliers
from transaction_dedup import deduplicate_transactions, deduplicate_bet_parts, dedup_table_name

# Herkent `cleaned_YYYYmmdd_HHMMSS` (en TEST_cleaned_...) mapnamen — overgenomen uit
# clean_and_parse.py, gebruikt om de nieuwste cleaned-map automatisch te kiezen.
CLEANED_RE = re.compile(r"^(?:TEST_)?cleaned_(\d{8})_(\d{6})$", re.IGNORECASE)


# -----------------------------------------------------------------------------
# Logging helper (overgenomen uit clean_and_parse.py)
# -----------------------------------------------------------------------------
def _setup_file_logger(log_path: Path) -> logging.Logger:
    """
    Maakt een logger die naar `log_path` schrijft én naar stdout spiegelt.
    Per te schonen CSV krijgt elk bestand zo een eigen log.
    """
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    _teardown_file_logger(logger)   # sluit eventuele eerdere handlers écht (geen fd-lek)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def _teardown_file_logger(logger: logging.Logger) -> None:
    """Sluit én verwijder alle handlers. `handlers.clear()` haalt ze alleen uit de lijst maar
    sluit het open OS-logbestand niet -> met een unieke logger per CSV lekt dat file-descriptors
    (op macOS standaard 256 -> 'Too many open files' rond operator 24). Daarom expliciet sluiten."""
    for _h in logger.handlers[:]:
        try:
            _h.close()
        finally:
            logger.removeHandler(_h)


def _iter_root_csvs(root: Path, glob_pat: str):
    """Yield CSV's in de root van `root` (geen submappen), gesorteerd."""
    for p in sorted(root.glob(glob_pat)):
        if p.is_file() and p.suffix.lower() == ".csv":
            yield p.resolve()


# -----------------------------------------------------------------------------
# Hoofd-API: schoon één map met ruwe WOK-CSV's
# -----------------------------------------------------------------------------
def clean_directory(
    input_dir: str | Path,
    clean_out_dir: Optional[str | Path] = None,
    *,
    glob_pattern: str = "*.csv",
    chunksize: int = 400_000,
    only_first_chunk: bool = False,
    verbose: bool = False,
    do_label: bool = True,
) -> Path:
    """
    Schoon alle WOK-CSV's in `input_dir` en schrijf ze naar een nieuwe
    `cleaned_<timestamp>/` map. Bewaar daarna de nieuwste rij per speler/transactie
    in de drie transactietabellen en de nieuwste bet-part per weddenschap/part-ID;
    vervolgens (optioneel) outlier-labelling.

    Parameters
    ----------
    input_dir : map met ruwe WOK-CSV's (flat; geen submappen worden gescand).
    clean_out_dir : rootmap waaronder `cleaned_<stamp>/` wordt aangemaakt.
                    Default: een `cleaned_<stamp>/` map *naast* de input.
    glob_pattern : welke bestanden als input gelden (default: '*.csv').
    chunksize : chunkgrootte bij streamend inlezen.
    only_first_chunk : alleen de eerste chunk schoonmaken (snelle test). Output
                       komt dan in een `TEST_cleaned_<stamp>/` map.
    verbose : uitgebreide logging op stdout.
    do_label : voer na het schoonmaken `label_outliers` uit op de cleaned-map.

    Returns
    -------
    Path naar de aangemaakte `cleaned_<stamp>/` (of `TEST_cleaned_<stamp>/`) map.
    """
    raw_dir = Path(input_dir).resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"--input-dir bestaat niet: {raw_dir}")

    # Logs komen naast de RAW input (net als in het origineel).
    logs_dir = raw_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"📂 Basisdirectory (RAW): {raw_dir}")

    # 1) Selectie van te schonen bestanden — schema-aware, scope = 'all'.
    to_clean: List[Path] = []
    for p in _iter_root_csvs(raw_dir, glob_pattern):
        skey, chunk_suffix = resolve_schema_key(p.name, organisatie_SCHEMA_BY_FILE.keys())
        # organisatie-relationele export (;-gescheiden) heeft géén oud schema, maar moet wél mee — die
        # tabellen (WOK_Bet_Parts, WOK_Bet_Transaction, WOK_Player_Limits_*, …) zijn nodig downstream.
        if not skey and not dedup_table_name(p.name) and not _is_organisatie_semicolon_format(str(p)):
            if verbose:
                print(f"  ⏭️  Onbekend schema, overslaan: {p.name}")
            continue
        to_clean.append(p)
        if verbose:
            print(f"  ✅ {p.name}  →  {skey}{' ('+chunk_suffix+')' if chunk_suffix else ''}")

    if not to_clean:
        raise FileNotFoundError(
            f"Geen herkenbare WOK-CSV's gevonden om te schonen in {raw_dir} "
            f"(glob='{glob_pattern}')."
        )

    if verbose:
        print("\n🧾 Lijst met te schonen RAW-bestanden:")
        for p in to_clean:
            print("  -", p.name)

    # 2) Bepaal cleaned output-locatie.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "TEST_cleaned_" if only_first_chunk else "cleaned_"
    out_root = Path(clean_out_dir).resolve() if clean_out_dir else raw_dir
    cleaned_dir = out_root / f"{prefix}{stamp}"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n🗂️  Map voor opgeschoonde CSV's: {cleaned_dir}")
        print(f"📝  Logmap (per bestand): {logs_dir}")

    # 3) Schoonmaak per bestand (met per-chunk logging).
    for in_path in to_clean:
        print(f"\n🧽 Start schoonmaken: {in_path.name}")
        log_path = logs_dir / (in_path.stem + "_clean.log")
        logger = _setup_file_logger(log_path)
        try:
            clean_csv_streaming(
                input_csv=in_path,
                cleaned_dir=cleaned_dir,
                chunksize=chunksize,
                logger=logger,
                only_first_chunk=only_first_chunk,
                verbose=verbose,
                local_test_output=False,
            )
        finally:
            _teardown_file_logger(logger)   # sluit het logbestand -> geen fd-lek over 300+ CSV's

    # 4) Nieuwste transacties en bet-parts, over chunks én CSV-delen.
    logger = _setup_file_logger(logs_dir / "transaction_dedup.log")
    try:
        deduplicate_transactions(cleaned_dir, chunksize=chunksize, logger=logger)
        deduplicate_bet_parts(cleaned_dir, chunksize=chunksize, logger=logger)
    finally:
        _teardown_file_logger(logger)

    # 5) Automatische outlier-labelling (net als clean_all in het origineel).
    if do_label:
        print(f"\n🏷️  Start outlier-labelling op: {cleaned_dir}")
        label_outliers(str(cleaned_dir), verbose=verbose)
    else:
        print("\n⏭️  Labelling overgeslagen (do_label=False).")

    print(f"\n✅ Schoonmaak klaar. Cleaned map: {cleaned_dir}")
    return cleaned_dir


# -----------------------------------------------------------------------------
# Label-only API (stap 5 uit de README)
# -----------------------------------------------------------------------------
def newest_cleaned_dir(parent_dir: str | Path) -> Optional[Path]:
    """
    Geef de nieuwste `cleaned_<stamp>/` (of `TEST_cleaned_<stamp>/`) submap binnen
    `parent_dir`, of None als er geen is. Sortering op mapnaam = chronologisch dankzij
    het timestamp-formaat. Overgenomen uit clean_and_parse.py.
    """
    parent = Path(parent_dir)
    if not parent.is_dir():
        return None
    cands = [p for p in parent.iterdir() if p.is_dir() and CLEANED_RE.match(p.name)]
    if not cands:
        return None
    return sorted(cands, key=lambda p: p.name)[-1].resolve()


def label_only_directory(
    cleaned_dir: Optional[str | Path] = None,
    *,
    parent_dir: Optional[str | Path] = None,
    verbose: bool = False,
    mode: str = "both",
) -> Path:
    """
    Voer ALLEEN de outlier-labelling uit op een al-geschoonde map (stap 5).

    Dit is het pure-Python equivalent van `clean_and_parse.py --mode label_only`
    (zonder SLURM-array / OPERATOR_PREFIX-lookup). De labelling draait in-place op
    de `WOK_Player_Profile*`-bestanden in de cleaned-map en gebruikt `only_label=True`
    zodat de (al gebeurde) schoonmaakstap wordt overgeslagen.

    Geef óf `cleaned_dir` (een concrete cleaned_<stamp>/ map) óf `parent_dir`
    (dan wordt de nieuwste cleaned_<stamp>/ daarin automatisch gekozen).
    """
    if cleaned_dir is None and parent_dir is None:
        raise ValueError("Geef óf cleaned_dir óf parent_dir mee.")

    if cleaned_dir is None:
        target = newest_cleaned_dir(parent_dir)
        if target is None:
            raise FileNotFoundError(
                f"Geen cleaned_<stamp>/ map gevonden onder {parent_dir}."
            )
        print(f"🔎 Nieuwste cleaned-map gekozen: {target}")
    else:
        target = Path(cleaned_dir).resolve()
        if not target.is_dir():
            raise FileNotFoundError(f"cleaned-map bestaat niet: {target}")

    print(f"\n🏷️  Label-only op: {target}")
    label_outliers(str(target), verbose=verbose, inplace=True, only_label=True, mode=mode)
    print(f"\n✅ Label-only klaar. Map: {target}")
    return target
