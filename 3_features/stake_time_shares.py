"""F44/F45: net stakes, attributed to placement time rather than refund time."""

from collections import defaultdict
import logging
from numbers import Real

import numpy as np
import pandas as pd

from path_finding import iter_csv_chunks
from local_time import local_time


def _id(value):
    if pd.isna(value) or not str(value).strip():
        return None
    if isinstance(value, Real) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def net_stake_time_shares(tables, *, x_tijdspad=None, chunksize=200_000, verbose=False, logger=None):
    """Return morning/evening shares after successful VOID_BET/VOID_STAKE refunds.

    References join on (player, transaction), parents on pk_id. Stable bet/session
    IDs combine update versions. Parent tables are optional: without them, references
    sharing the same parent pk_id can still be used. Unmatched/ambiguous refunds
    leave the player's shares unknown rather than deducting from unrelated stakes.

    Placement hours use Europe/Amsterdam, including summer/winter time.
    Refunds reduce their bet/session's stakes proportionally, including stakes before
    the window. This is an explicit allocation rule: the export has no original-stake
    ID on refunds. Only transactions before the exclusive window end are used, so
    later refunds cannot leak future information. Each input is scanned once; only
    reference mappings and per-player/bet/session sums are retained, not event lists.
    """
    start = pd.to_datetime(x_tijdspad[0], format="%d%m%Y") if x_tijdspad else None
    end = pd.to_datetime(x_tijdspad[1], format="%d%m%Y") if x_tijdspad else None
    links = defaultdict(set)
    for scope, parent_table, parent_id, reference_table, fk, tx_col in (
        ("bet", "WOK_Bet", "Bet_ID", "WOK_Bet_Transaction", "wok_bet_pk_id", "transactions_id"),
        ("session", "WOK_Game_Session", "Game_Session_ID", "WOK_Game_Session_Transaction",
         "wok_game_session_pk_id", "transaction_id"),
    ):
        parents = {}
        for df in iter_csv_chunks(tables.get(parent_table) or [], usecols=["pk_id", parent_id],
                                  chunksize=chunksize, verbose=verbose):
            for pk, identity in zip(df["pk_id"], df.get(parent_id, pd.Series(index=df.index, dtype=object))):
                if _id(pk) is not None:
                    parents[_id(pk)] = ("id", _id(identity)) if _id(identity) is not None else ("pk", _id(pk))
        for df in iter_csv_chunks(tables.get(reference_table) or [], usecols=[fk, "player_profile_id", tx_col],
                                  chunksize=chunksize, verbose=verbose):
            for pk, pid, txid in zip(df[fk], df["player_profile_id"], df[tx_col]):
                pk, pid, txid = _id(pk), _id(pid), _id(txid)
                if pk is not None and pid is not None and txid is not None:
                    links[(pid, txid)].add((scope, parents.get(pk, ("pk", pk))))

    # [all stakes, refunds, in-window night stakes, morning stakes, evening stakes]
    amounts = defaultdict(lambda: np.zeros(5, dtype=float))
    players = set()
    unknown = set()
    for df in iter_csv_chunks(tables.get("WOK_Player_Account_Transaction") or [],
                              usecols=["Player_Profile_ID", "Transaction_ID", "Transaction_Type",
                                       "Transaction_Status", "Transaction_Amount", "Transaction_Datetime"],
                              chunksize=chunksize, verbose=verbose):
        typ = df["Transaction_Type"].fillna("").astype(str).str.strip().str.upper()
        status = df["Transaction_Status"].fillna("").astype(str).str.strip().str.upper()
        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
        hours = local_time(ts).dt.hour
        money = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        mask = typ.isin(["STAKE", "VOID_BET", "VOID_STAKE"]) & status.eq("SUCCESSFUL") & ts.notna() & np.isfinite(money)
        if end is not None:
            mask &= ts < end
        txids = df.get("Transaction_ID", pd.Series(index=df.index, dtype=object))
        for pid, txid, kind, timestamp, value, hour in zip(df.loc[mask, "Player_Profile_ID"], txids[mask],
                                                    typ[mask], ts[mask], money[mask], hours[mask]):
            pid, txid = _id(pid), _id(txid)
            if pid is None:
                continue
            if kind != "STAKE" and value == 0:
                continue
            groups = links.get((pid, txid), set())
            if kind != "STAKE":
                scope = "bet" if kind == "VOID_BET" else "session"
                groups = {group for group in groups if group[0] == scope}
            in_window = start is None or timestamp >= start
            if kind == "STAKE" and in_window:
                players.add(pid)
            if len(groups) > 1 or (kind != "STAKE" and not groups):
                unknown.add(pid)
                continue
            # Unreferenced stakes still count; never assign a refund by player alone.
            group = next(iter(groups)) if groups else ("unlinked", txid)
            acc = amounts[(pid, group)]
            if kind == "STAKE":
                acc[0] += abs(value)
                if in_window:
                    bucket = 0 if hour < 8 else (1 if hour < 16 else 2)
                    acc[2 + bucket] += abs(value)
            else:
                acc[1] += abs(value)

    totals = defaultdict(lambda: np.zeros(3, dtype=float))
    for (pid, _), acc in amounts.items():
        stake, refund = acc[:2]
        if refund > stake and not np.isclose(refund, stake):
            unknown.add(pid)
            continue
        if stake > 0:
            totals[pid] += acc[2:] * max(0.0, 1.0 - refund / stake)
    if unknown & players:
        (logger or logging.getLogger(__name__)).warning(
            "Net stake shares unknown for %d players: unmatched/ambiguous or excessive refunds.",
            len(unknown & players))
    records = []
    for pid in sorted(players):
        total = totals[pid].sum()
        valid = pid not in unknown and total > 0
        records.append({"Player_Profile_ID": pid,
                        "f44_morning_stakes_percentage": totals[pid][1] / total if valid else np.nan,
                        "f45_evening_stakes_percentage": totals[pid][2] / total if valid else np.nan})
    return pd.DataFrame.from_records(records, columns=["Player_Profile_ID", "f44_morning_stakes_percentage",
                                                      "f45_evening_stakes_percentage"])
