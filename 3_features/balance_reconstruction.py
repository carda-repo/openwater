"""Reconstruct opening balances with chunked reads and O(players) state.

Inputs must contain complete, deduplicated account movements between the chosen
snapshot and the analysis start. Snapshot timestamps are taken from Extraction_Date;
transactions at that exact timestamp are treated as occurring after the snapshot.
Only snapshots before the analysis end are used, to avoid reading target-period balances.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from path_finding import iter_csv_chunks


def reconstruct_start_balances(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    verbose: bool = False,
) -> Dict[str, float]:
    """Prefer the latest snapshot <= start, otherwise the earliest within the window.

    Older snapshots are advanced to the start; later snapshots are reversed to it.
    Without a window, reverse the earliest snapshot using all earlier available
    transactions, giving the opening balance of the available transaction history.
    Missing anchors or unusable bridging transactions leave the balance unknown.
    Read profiles once, and transactions at most once; no transaction sort/buffer.
    """
    profile_paths = tables.get("WOK_Player_Profile") or []
    if not profile_paths:
        return {}
    if x_tijdspad:
        start, end = (pd.Timestamp(datetime.strptime(value, "%d%m%Y")) for value in x_tijdspad)
    else:
        start, end = None, None
    before: Dict[str, tuple] = {}
    after: Dict[str, tuple] = {}

    for df in iter_csv_chunks(
        profile_paths,
        usecols=["Player_Profile_ID", "Player_Profile_EOD_Balance", "Extraction_Date"],
        chunksize=chunksize, verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        df["pid"] = df["Player_Profile_ID"].astype(str)
        df["ts"] = pd.to_datetime(df["Extraction_Date"], errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
        df["bal"] = pd.to_numeric(df["Player_Profile_EOD_Balance"], errors="coerce")
        df = df[df["pid"].str.strip().ne("") & df["ts"].notna() & np.isfinite(df["bal"])]
        if end is not None:
            df = df[df["ts"] < end]
        past = df[df["ts"] <= start] if start is not None else df.iloc[:0]
        future = df[df["ts"] > start] if start is not None else df
        for subset, target, latest in ((past, before, True), (future, after, False)):
            if subset.empty:
                continue
            grouped = subset.groupby("pid")["ts"]
            chosen = grouped.transform("max" if latest else "min")
            candidates = subset[subset["ts"].eq(chosen)]
            for (pid, ts), values in candidates.groupby(["pid", "ts"])["bal"]:
                bal = float(values.iloc[0]) if values.nunique() == 1 else None
                previous = target.get(pid)
                if previous is None or (ts > previous[0] if latest else ts < previous[0]):
                    target[pid] = (ts, bal)
                elif ts == previous[0] and bal != previous[1]:
                    target[pid] = (ts, None)

    anchors = {**after, **before}
    # A conflicting chosen anchor is unknown; do not silently pick an older one.
    anchors = {pid: anchor for pid, anchor in anchors.items() if anchor[1] is not None}
    balances = {pid: bal for pid, (ts, bal) in anchors.items() if start is not None and ts == start}
    bridge = {pid: ts for pid, (ts, _) in anchors.items() if pid not in balances}
    if not bridge:
        return balances
    tx_paths = tables.get("WOK_Player_Account_Transaction") or []
    if not tx_paths:
        return balances
    deltas = dict.fromkeys(bridge, 0.0)
    unusable = set()

    for df in iter_csv_chunks(
        tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Amount", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize, verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna() & df["Transaction_Status"].eq("SUCCESSFUL")].copy()
        df["pid"] = df["Player_Profile_ID"].astype(str)
        df = df[df["pid"].isin(bridge)].copy()
        if df.empty:
            continue
        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
        unusable.update(df.loc[ts.isna(), "pid"])
        anchor_ts = pd.to_datetime(df["pid"].map(bridge))
        if start is None:
            in_bridge = ts < anchor_ts
        else:
            in_bridge = (
                ((anchor_ts < start) & (ts >= anchor_ts) & (ts < start))
                | ((anchor_ts > start) & (ts >= start) & (ts < anchor_ts))
            )
        df = df.loc[in_bridge].copy()
        df["delta"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        valid = np.isfinite(df["delta"])
        unusable.update(df.loc[~valid, "pid"])
        df = df.loc[valid].copy()
        stake = df["Transaction_Type"].eq("STAKE")
        df.loc[stake, "delta"] = -df.loc[stake, "delta"].abs()
        for pid, delta in df.groupby("pid")["delta"].sum().items():
            deltas[pid] += float(delta)

    for pid, ts in bridge.items():
        if pid not in unusable and np.isfinite(deltas[pid]):
            direction = 1 if start is not None and ts < start else -1
            balances[pid] = anchors[pid][1] + direction * deltas[pid]
    return balances
