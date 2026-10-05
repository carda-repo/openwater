#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
relational_joins.py
===================

Gedeelde helpers voor de feature-laag in het **relationele organisatie-formaat**.

Waar de oude data de koppelingen als JSON binnen één rij had (``Bet_Transactions``,
``Game_Transactions``, ``Bet_Parts``), zijn dat in het organisatie-formaat losse tabellen die je via
``wok_*_pk_id`` → ``pk_id`` joint. Deze helpers bouwen één keer een in-memory mapping van die
relationele tabellen, zodat de features die kunnen opzoeken i.p.v. JSON te parsen.

NB: snake_case kolommen worden hier expliciet gevraagd (de organisatie-data ís snake_case); de
case-tolerante ``iter_csv_chunks`` laat ze ongemoeid.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from path_finding import iter_csv_chunks


def _notnull(v) -> bool:
    return v is not None and not (isinstance(v, float) and pd.isna(v))


def build_session_tx_map(session_tx_paths, chunksize: int = 200_000) -> Dict[object, List[Tuple]]:
    """``WOK_Game_Session_Transaction`` → {wok_game_session_pk_id: [(player_profile_id, transaction_id), ...]}.

    Vervangt het oude ``iter_transaction_ids_from_Game_Transactions(json)`` per sessie-rij:
    sla de transacties per sessie (via de ``wok_game_session_pk_id``-join) op.
    """
    m: Dict[object, List[Tuple]] = defaultdict(list)
    if not session_tx_paths:
        return m
    for df in iter_csv_chunks(
        paths=session_tx_paths,
        usecols=["wok_game_session_pk_id", "player_profile_id", "transaction_id"],
        chunksize=chunksize, verbose=False,
    ):
        if df.empty:
            continue
        for spk, pid, txid in zip(df["wok_game_session_pk_id"].tolist(),
                                  df["player_profile_id"].tolist(),
                                  df["transaction_id"].tolist()):
            m[spk].append((pid, txid))
    return m


def build_bet_tx_map(bet_tx_paths, chunksize: int = 200_000) -> Dict[object, List[Tuple]]:
    """``WOK_Bet_Transaction`` → {wok_bet_pk_id: [(player_profile_id, transactions_id), ...]}.

    Vervangt ``iter_player_profile_ids_from_Bet_Transactions`` / ``iter_transaction_ids_from_Bet_Transactions``.
    """
    m: Dict[object, List[Tuple]] = defaultdict(list)
    if not bet_tx_paths:
        return m
    for df in iter_csv_chunks(
        paths=bet_tx_paths,
        usecols=["wok_bet_pk_id", "player_profile_id", "transactions_id"],
        chunksize=chunksize, verbose=False,
    ):
        if df.empty:
            continue
        for bpk, pid, txid in zip(df["wok_bet_pk_id"].tolist(),
                                  df["player_profile_id"].tolist(),
                                  df["transactions_id"].tolist()):
            m[bpk].append((pid, txid))
    return m


def build_bet_players_map(bet_tx_paths, chunksize: int = 200_000) -> Dict[object, set]:
    """``WOK_Bet_Transaction`` → {wok_bet_pk_id: set(player_profile_id)}.

    ``WOK_Bet`` zelf heeft geen speler-kolom; de attributie van een bet aan zijn speler(s) loopt via
    ``wok_bet_pk_id`` → ``WOK_Bet.pk_id``. Deze map levert alleen die speler-koppeling (geen tijd):
    f46 gebruikt ``WOK_Bet.Extraction_Date`` van een afgewikkelde/geannuleerde rapportage
    als benadering van de afwikkeltijd, zonder ``created_at`` of transactietijden te gebruiken.
    """
    players: Dict[object, set] = defaultdict(set)
    if not bet_tx_paths:
        return players
    for df in iter_csv_chunks(
        paths=bet_tx_paths,
        usecols=["wok_bet_pk_id", "player_profile_id"],
        chunksize=chunksize, verbose=False,
    ):
        if df.empty:
            continue
        for bpk, pid in zip(df["wok_bet_pk_id"].tolist(), df["player_profile_id"].tolist()):
            if _notnull(pid):
                players[bpk].add(str(pid))
    return players


def build_limits_player_map(limits_parent_paths, chunksize: int = 200_000) -> Dict[object, str]:
    """``WOK_Player_Limits`` (parent) → {pk_id: player_profile_id}.

    De sub-tabellen (Deposit/Game_Type/Login) hebben géén speler-kolom; die join je via
    ``wok_player_limit_pk_id`` → ``WOK_Player_Limits.pk_id`` om bij de speler te komen.
    """
    m: Dict[object, str] = {}
    if not limits_parent_paths:
        return m
    for df in iter_csv_chunks(
        paths=limits_parent_paths,
        usecols=["pk_id", "player_profile_id"],
        chunksize=chunksize, verbose=False,
    ):
        if df.empty:
            continue
        for pk, pid in zip(df["pk_id"].tolist(), df["player_profile_id"].tolist()):
            if _notnull(pid):
                m[pk] = str(pid)
    return m


def build_limits_events_map(
    limits_parent_paths,
    *,
    participation_paths=None,
    deposit_paths=None,
    login_paths=None,
    game_type_paths=None,
    balance_paths=None,
    chunksize: int = 200_000,
) -> Dict[str, List[Tuple]]:
    """Bouw per speler een lijst limiet-wijzig-events ``(ts, kind, value, window)``.

    Vervangt het oude per-rij JSON-parsen van ``Limit_Participation`` / ``Limit_Deposit`` /
    ``Limit_Login`` / ``Limit_Balance`` / ``Limit_Game_Type`` in f22/f23. In het organisatie-formaat zijn dit
    losse sub-tabellen die via ``wok_player_limit_pk_id`` → ``WOK_Player_Limits.pk_id`` →
    ``player_profile_id`` koppelen — één sub-document per limiet-soort.

    ``Limit_Participation`` is in het organisatie-formaat de losse tabel ``WOK_Player_Limits_Participation``
    en is — net als in de root — de **primaire** bron voor f22/f23 (kind="participation"). ``value``
    is numeriek voor participation (``participation_amount``), deposit (``deposit_amount``), login
    (``login_duration``) en balance (``balance_amount``) → die tellen mee als increase/decrease.
    ``game_type`` is vrije tekst zonder bedrag → geen increase/decrease (wordt bij de numerieke
    vergelijking overgeslagen). Net als in het origineel krijgt **balance window=None** (alle
    balance-events vormen één stream per speler).
    """
    pid_by_pk = build_limits_player_map(limits_parent_paths, chunksize=chunksize)
    events: Dict[str, List[Tuple]] = defaultdict(list)

    def _harvest(paths, *, kind, ts_col, val_col, win_col):
        # win_col=None → window=None (zoals het origineel voor balance doet).
        if not paths:
            return
        cols = ["wok_player_limit_pk_id", ts_col, val_col] + ([win_col] if win_col else [])
        for df in iter_csv_chunks(paths=paths, usecols=cols, chunksize=chunksize, verbose=False):
            if df.empty:
                continue
            wins = df[win_col].tolist() if win_col else [None] * len(df)
            for lpk, ts, val, win in zip(df["wok_player_limit_pk_id"].tolist(),
                                         df[ts_col].tolist(),
                                         df[val_col].tolist(),
                                         wins):
                pid = pid_by_pk.get(lpk)
                if pid is None:
                    continue
                win_s = str(win).upper() if _notnull(win) else None
                events[pid].append((ts, kind, val, win_s))

    _harvest(participation_paths, kind="participation", ts_col="participation_request_datetime",
             val_col="participation_amount", win_col="participation_time_window")
    _harvest(deposit_paths, kind="deposit", ts_col="deposit_request_datetime",
             val_col="deposit_amount", win_col="deposit_time_window")
    _harvest(login_paths, kind="login", ts_col="login_request_datetime",
             val_col="login_duration", win_col="login_time_window")
    _harvest(balance_paths, kind="balance", ts_col="balance_request_datetime",
             val_col="balance_amount", win_col=None)   # window=None, 1-op-1 met het origineel
    _harvest(game_type_paths, kind="game_type", ts_col="game_type_request_datetime",
             val_col="game_type_type", win_col="game_type_time_window")
    return events


def build_bet_parts_map(bet_parts_paths, chunksize: int = 200_000) -> Dict[object, List[dict]]:
    """``WOK_Bet_Parts`` → {wok_bet_pk_id: [ {sport, live, odds, stake, ...}, ... ]}.

    Vervangt ``iter_part_ids_from_Bet_Parts`` / ``iter_part_live_flags_from_Bet_Parts`` e.d.
    """
    m: Dict[object, List[dict]] = defaultdict(list)
    if not bet_parts_paths:
        return m
    cols = ["wok_bet_pk_id", "part_id", "part_sport", "part_live", "part_odds", "part_stake",
            "part_prognosis_value", "part_event"]
    for df in iter_csv_chunks(paths=bet_parts_paths, usecols=cols, chunksize=chunksize, verbose=False):
        if df.empty:
            continue
        present = [c for c in cols if c in df.columns]
        for rec in df[present].to_dict("records"):
            m[rec["wok_bet_pk_id"]].append(rec)
    return m
