"""F51: match resolved sports bets to the next accepted stake of the same player."""

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from path_finding import iter_csv_chunks


FEATURE = "f51_median_seconds_loss_to_next_bet"


def _identifier(value):
    if pd.isna(value) or not str(value).strip():
        return None
    if isinstance(value, Real) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _timestamps(values):
    return pd.to_datetime(values, errors="coerce", utc=True, format="mixed").dt.tz_localize(None)


@dataclass(slots=True)
class _Bet:
    placed: object = None
    first_settled: object = None
    latest_status: str | None = ""
    latest_status_at: object = None


@dataclass(slots=True)
class _Outcome:
    stake: bool = False
    prize: bool = False
    cashout: bool = False
    void_at: object = None
    unknown: bool = False


def median_loss_to_next_bet(tables, *, x_tijdspad=None, chunksize=200_000, verbose=False, logger=None):
    """Read each table once, retaining reference indexes and per-bet state.

    Logical bet identity is (operator, Bet_ID); parent pk_ids only join references.
    Transaction identity is (operator, player, Transaction_ID). History before
    the window is used to establish ownership, stakes and prizes. Closures and
    subsequent placements must be in [start, end); future transactions/statuses
    do not determine historical losses. Successful positive winnings or cash-outs
    exclude their own bet, irrespective of other bets placed in the meantime.

    Regular losses require final observed status BET_SETTLED and a successful stake.
    Conflicting final statuses at the latest Extraction_Date leave regular losses
    unknown. A strictly newer status can resolve that ambiguity. VOID_BET refunds
    remain independent evidence of closure.
    The first settled Extraction_Date is a proxy, not an exact resolution timestamp.
    VOID_BET is included by project choice: its successful positive refund closes
    the interval at Transaction_Datetime, independently of the reported bet status.
    No missing timestamp is replaced with the original placement time.

    RAM scales with references and player/bet pairs; event rows are not buffered.
    Sorting placement times and binary searches are performed per operator/player.
    Input transactions are expected to have been deduplicated by cleaning.
    """
    empty = pd.DataFrame(columns=["Player_Profile_ID", FEATURE])
    if not tables.get("WOK_Bet") or not tables.get("WOK_Bet_Transaction"):
        return empty
    start = pd.to_datetime(x_tijdspad[0], format="%d%m%Y") if x_tijdspad else None
    end = pd.to_datetime(x_tijdspad[1], format="%d%m%Y") if x_tijdspad else None
    # KSA CDB v1.11 p.57: only BET_SETTLED establishes settlement.
    # Whether a settled bet lost is determined from its own transactions.
    settled_statuses = {"BET_SETTLED"}
    parents = {}
    bets = {}

    # 1. Combine all update versions; the earliest settled report is not overwritten.
    for df in iter_csv_chunks(tables.get("WOK_Bet"),
                              usecols=["pk_id", "Bet_ID", "Bet_Start_Datetime", "Bet_Status",
                                       "Extraction_Date", "Operator_ID"],
                              chunksize=chunksize, verbose=verbose):
        placed = _timestamps(df["Bet_Start_Datetime"])
        extracted = _timestamps(df.get("Extraction_Date", pd.Series(pd.NaT, index=df.index)))
        operators = df.get("Operator_ID", pd.Series(index=df.index, dtype=object))
        statuses = df["Bet_Status"].fillna("").astype(str).str.strip().str.upper()
        for pk, bet_id, operator, placement, extraction, status in zip(
                df["pk_id"], df["Bet_ID"], operators, placed, extracted, statuses):
            pk, bet_id = _identifier(pk), _identifier(bet_id)
            if pk is None or bet_id is None:
                continue
            identity = (_identifier(operator), bet_id)
            parents[pk] = identity
            bet = bets.setdefault(identity, _Bet())
            if pd.notna(placement) and (end is None or placement < end):
                if bet.placed is None or placement < bet.placed:
                    bet.placed = placement
            # A future version may supply a parent link/placement, never a past outcome.
            if pd.isna(extraction) or (end is not None and extraction >= end):
                continue
            if bet.latest_status_at is None or extraction > bet.latest_status_at:
                bet.latest_status, bet.latest_status_at = status, extraction
            elif extraction == bet.latest_status_at and status != bet.latest_status:
                # Equal extraction times cannot establish which status is newer.
                bet.latest_status = None
            if status in settled_statuses:
                if bet.first_settled is None or extraction < bet.first_settled:
                    bet.first_settled = extraction

    # 2. A compact lookup stores one bet per transaction in the usual case.
    tx_to_bet = {}
    ambiguous = {}
    outcomes = {}
    players = set()
    for df in iter_csv_chunks(tables.get("WOK_Bet_Transaction"),
                              usecols=["wok_bet_pk_id", "player_profile_id", "transactions_id"],
                              chunksize=chunksize, verbose=verbose):
        for pk, pid, txid in zip(df["wok_bet_pk_id"], df["player_profile_id"], df["transactions_id"]):
            identity = parents.get(_identifier(pk))
            pid, txid = _identifier(pid), _identifier(txid)
            if pid is None:
                continue
            players.add(pid)
            if identity is None:
                continue
            outcomes.setdefault((identity, pid), _Outcome())
            if txid is None:
                continue
            key = (pid, txid)
            previous = tx_to_bet.setdefault(key, identity)
            if previous != identity:
                ambiguous.setdefault(key, {previous}).add(identity)
    del parents

    # 3. Only the outcome flags/refund timestamp for each player/bet survive a chunk.
    for df in iter_csv_chunks(tables.get("WOK_Player_Account_Transaction") or [],
                              usecols=["Player_Profile_ID", "Transaction_ID", "Transaction_Type",
                                       "Transaction_Status", "Transaction_Amount", "Transaction_Datetime", "Operator_ID"],
                              chunksize=chunksize, verbose=verbose):
        typ = df["Transaction_Type"].fillna("").astype(str).str.strip().str.upper()
        status = df["Transaction_Status"].fillna("").astype(str).str.strip().str.upper()
        mask = status.eq("SUCCESSFUL") & typ.isin(["STAKE", "WINNING", "CASH_OUT", "VOID_BET"])
        df, typ = df.loc[mask], typ.loc[mask]
        timestamps = _timestamps(df["Transaction_Datetime"])
        amounts = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        operators = df.get("Operator_ID", pd.Series(index=df.index, dtype=object))
        for pid, txid, operator, kind, timestamp, amount in zip(
                df["Player_Profile_ID"], df["Transaction_ID"], operators, typ, timestamps, amounts):
            pid, txid, operator = _identifier(pid), _identifier(txid), _identifier(operator)
            key = (pid, txid)
            identity = tx_to_bet.get(key)
            if identity is None or (pd.notna(timestamp) and end is not None and timestamp >= end):
                continue
            candidates = ambiguous.get(key, (identity,))
            if operator is not None:
                candidates = [candidate for candidate in candidates if candidate[0] in (None, operator)]
            if len(candidates) != 1:
                for candidate in candidates:
                    outcomes[(candidate, pid)].unknown = True
                continue
            identity = next(iter(candidates))
            outcome = outcomes[(identity, pid)]
            placement = bets[identity].placed
            if pd.isna(timestamp) or (placement is not None and timestamp < placement):
                outcome.unknown = True
                continue
            if kind == "CASH_OUT":
                outcome.cashout = True
            elif not np.isfinite(amount) or (kind == "WINNING" and amount < 0):
                outcome.unknown = True
            elif kind == "STAKE" and abs(amount) > 0:
                outcome.stake = True
            elif kind == "WINNING" and amount > 0:
                outcome.prize = True
            elif kind == "VOID_BET" and abs(amount) > 0:
                if outcome.void_at is None or timestamp < outcome.void_at:
                    outcome.void_at = timestamp
    reference_count = len(tx_to_bet)
    del tx_to_bet, ambiguous

    placements = defaultdict(list)
    for (identity, pid), outcome in outcomes.items():
        placement = bets[identity].placed
        if outcome.stake and placement is not None and (start is None or placement >= start):
            placements[(identity[0], pid)].append(placement)
    for times in placements.values():
        times.sort()

    deltas = defaultdict(list)
    proxies = voids = 0
    for (identity, pid), outcome in outcomes.items():
        bet = bets[identity]
        if (not outcome.stake or outcome.prize or outcome.cashout or outcome.unknown
                or bet.placed is None):
            continue
        if outcome.void_at is not None:
            closed = outcome.void_at
            is_void = True
        elif bet.latest_status in settled_statuses:
            closed = bet.first_settled
            is_void = False
        else:
            continue
        if closed is None or closed < bet.placed or (start is not None and closed < start):
            continue
        times = placements.get((identity[0], pid), [])
        idx = bisect_right(times, closed)  # Strictly after closure; never the preceding open bet.
        if idx < len(times):
            deltas[pid].append((times[idx] - closed).total_seconds())
            voids += is_void
            proxies += not is_void
    if logger:
        logger.info("F51: %d logical bets, %d transaction references, %d player/bet pairs; "
                    "%d settlement-proxy intervals, %d void-refund intervals, %d unknown outcomes.",
                    len(bets), reference_count, len(outcomes), proxies, voids,
                    sum(outcome.unknown for outcome in outcomes.values()))
    return pd.DataFrame.from_records([
        {"Player_Profile_ID": pid, FEATURE: float(np.median(deltas[pid])) if deltas[pid] else np.nan}
        for pid in sorted(players)
    ], columns=["Player_Profile_ID", FEATURE])
