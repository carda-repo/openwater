"""Balance events at observed timestamps, without inventing an order within ties."""

from dataclasses import dataclass, field
from datetime import datetime
from math import fsum

import numpy as np
import pandas as pd

from path_finding import iter_csv_chunks


@dataclass
class BalanceEvents:
    balance: float
    drops: int = 0
    deposits: int = 0
    pending_drop: object = None
    intervals: list = field(default_factory=list)
    unknown: bool = False


def balance_events(tables, start_balances, *, x_tijdspad=None, chunksize=200_000,
                   verbose=False, threshold=2.0, collect_intervals=False):
    """Scan sorted account movements once, keeping one unfinished moment per player.

    Sorting must be chronological per player across files/chunks, as provided by
    the sorting pipeline. A backward timestamp makes that player's result unknown.
    State scales with players, the largest tied group and observed deposit intervals;
    entire transaction histories are not buffered. fsum avoids sum-order roundoff.
    Deposits at a later moment count against the balance BEFORE that whole moment.
    Multiple deposits at that moment count separately for F27 and once for F28.
    """
    states = {str(pid): BalanceEvents(balance) for pid, balance in start_balances.items()}
    pending = {}
    if x_tijdspad:
        start, end = (pd.Timestamp(datetime.strptime(v, "%d%m%Y")) for v in x_tijdspad)
    else:
        start = end = None

    def apply(pid):
        timestamp, amounts, deposits = pending[pid]
        state = states[pid]
        before = state.balance
        after = before + fsum(amounts)
        if deposits and before < threshold and state.drops:
            state.deposits += deposits
            if state.pending_drop is not None and timestamp > state.pending_drop:
                if collect_intervals:
                    state.intervals.append((timestamp - state.pending_drop).total_seconds())
                state.pending_drop = None
        if before >= threshold and after < threshold:
            state.drops += 1
            state.pending_drop = timestamp
        state.balance = after

    for df in iter_csv_chunks(
        tables.get("WOK_Player_Account_Transaction") or [],
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Amount",
                 "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize, verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna() & df["Transaction_Status"].eq("SUCCESSFUL")].copy()
        df["pid"] = df["Player_Profile_ID"].astype(str)
        df = df[df["pid"].isin(states)].copy()
        df["ts"] = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True,
                                   format="mixed").dt.tz_localize(None)
        df["amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df = df[df["ts"].notna() & np.isfinite(df["amount"])].copy()
        if start is not None:
            df = df[(df["ts"] >= start) & (df["ts"] < end)].copy()
        stake = df["Transaction_Type"].eq("STAKE")
        df.loc[stake, "amount"] = -df.loc[stake, "amount"].abs()
        for pid, ts, amount, kind in df[["pid", "ts", "amount", "Transaction_Type"]].itertuples(index=False, name=None):
            if states[pid].unknown:
                continue
            previous = pending.get(pid)
            if previous is not None and ts < previous[0]:
                states[pid].unknown = True
                continue
            if previous is not None and ts > previous[0]:
                apply(pid)
                previous = None
            if previous is None:
                pending[pid] = (ts, [float(amount)], int(kind == "DEPOSIT"))
            else:
                previous[1].append(float(amount))
                pending[pid] = (ts, previous[1], previous[2] + int(kind == "DEPOSIT"))
    for pid in pending:
        if not states[pid].unknown:
            apply(pid)
    return states
