# -*- coding: utf-8 -*-
"""portfolio_gui_io.py

I/O helpers for the Portfolio Simulator GUI.

This module is intentionally small and conservative:
  - Load a user-provided .pkl (pickle) file
  - Extract a list of "sheet-like" DataFrames to drive Event construction

Security note
-------------
Pickles are not safe to load from untrusted sources. This GUI assumes the user
is loading their own locally-generated .pkl files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import io
import pickle

import pandas as pd


@dataclass
class SheetItem:
    """A single sheet-like input to the event builder."""

    name: str
    df: pd.DataFrame
    meta: Dict[str, Any] | None = None


def load_pkl_bytes(pkl_bytes: bytes) -> Any:
    """Load a pickle payload from bytes."""
    return pickle.loads(pkl_bytes)


def load_pkl_path(path: str) -> Any:
    """Load a pickle payload from a filesystem path."""
    with open(path, "rb") as f:
        return pickle.load(f)


def summarize_object(obj: Any) -> Dict[str, Any]:
    """Return a lightweight summary for UI display."""
    out: Dict[str, Any] = {"type": type(obj).__name__}
    if isinstance(obj, dict):
        out["keys"] = list(obj.keys())
        out["n_keys"] = len(obj)
    elif isinstance(obj, (list, tuple)):
        out["length"] = len(obj)
        out["first_type"] = type(obj[0]).__name__ if len(obj) else None
    else:
        # best-effort: show attributes a user might recognize
        for k in ("sheets", "sheet_dfs", "dfs", "data"):
            if hasattr(obj, k):
                out["has_attr"] = k
                break
    return out


def _as_dataframe(x: Any) -> Optional[pd.DataFrame]:
    if isinstance(x, pd.DataFrame):
        return x
    return None


def extract_sheet_items(obj: Any) -> List[SheetItem]:
    """Extract a list of SheetItem from a variety of reasonable .pkl structures.

    Supported patterns
    ------------------
    1) dict with key 'sheets' or 'sheet_dfs' or 'dfs' holding:
         - list[DataFrame]
         - dict[str, DataFrame]

    2) dict[str, DataFrame] (treated as name->df)

    3) list[DataFrame]

    4) object with attribute .sheets / .sheet_dfs / .dfs containing one of the above
    """

    # 1) direct list
    if isinstance(obj, list) and all(isinstance(x, pd.DataFrame) for x in obj):
        return [SheetItem(name=f"sheet_{i:03d}", df=df) for i, df in enumerate(obj)]

    # 2) direct mapping name->df
    if isinstance(obj, dict) and obj and all(isinstance(v, pd.DataFrame) for v in obj.values()):
        return [SheetItem(name=str(k), df=v) for k, v in obj.items()]

    # 3) dict wrapper
    if isinstance(obj, dict):
        for key in ("sheets", "sheet_dfs", "dfs", "data"):
            if key not in obj:
                continue
            payload = obj[key]
            if isinstance(payload, list) and all(isinstance(x, pd.DataFrame) for x in payload):
                return [SheetItem(name=f"{key}_{i:03d}", df=df) for i, df in enumerate(payload)]
            if isinstance(payload, dict) and payload and all(isinstance(v, pd.DataFrame) for v in payload.values()):
                return [SheetItem(name=str(k), df=v) for k, v in payload.items()]

    # 4) object wrapper
    for attr in ("sheets", "sheet_dfs", "dfs", "data"):
        if hasattr(obj, attr):
            try:
                return extract_sheet_items(getattr(obj, attr))
            except Exception:
                pass

    return []


def infer_column_candidates(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Infer candidate columns for ticker, datetime, numeric metrics."""
    if df is None or df.empty:
        return {"ticker": [], "datetime": [], "numeric": []}

    cols = list(df.columns)

    # ticker: object-like columns with low-ish cardinality strings
    ticker_cands: List[str] = []
    for c in cols:
        s = df[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            ticker_cands.append(c)

    # datetime: columns parseable by to_datetime with reasonable success
    dt_cands: List[str] = []
    for c in cols:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            ok = parsed.notna().mean() if len(parsed) else 0.0
            if ok >= 0.8:  # heuristic
                dt_cands.append(c)
        except Exception:
            continue

    # numeric
    num_cands = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    # also include columns coercible to numeric with decent success
    for c in cols:
        if c in num_cands:
            continue
        try:
            coerced = pd.to_numeric(df[c], errors="coerce")
            ok = coerced.notna().mean() if len(coerced) else 0.0
            if ok >= 0.8:
                num_cands.append(c)
        except Exception:
            continue

    # mild preference ordering
    def prefer(names: Sequence[str], preferred: Sequence[str]) -> List[str]:
        out = []
        for p in preferred:
            if p in names and p not in out:
                out.append(p)
        for n in names:
            if n not in out:
                out.append(n)
        return out

    ticker_cands = prefer(ticker_cands, ["yf_Ticker", "Valid Ticker", "ticker", "Ticker", "Symbol", "symbol"])
    dt_cands = prefer(dt_cands, ["datetime", "date", "Date", "event_date", "Event Date"])

    return {"ticker": ticker_cands, "datetime": dt_cands, "numeric": num_cands}
