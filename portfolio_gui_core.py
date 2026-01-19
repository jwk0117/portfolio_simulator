# -*- coding: utf-8 -*-
"""portfolio_gui_core.py

Core orchestration logic for the Portfolio Simulator GUI.

This module sits "next to" portfolio_sim.py and avoids modifying engine logic.
It provides:
  - Sheet normalization for event building
  - Calls into build_events_from_sheets
  - Simulation construction and run helpers
  - Export helpers for reproducibility

The GUI layer (portfolio_gui_app.py) should call this module, not portfolio_sim
directly, so that GUI concerns stay isolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import json
import math

import numpy as np
import pandas as pd

import portfolio_sim as ps

from portfolio_gui_io import SheetItem


@dataclass
class EventBuildConfig:
    """Parameters to convert sheet data into an ordered Event sequence."""

    ticker_col: str
    datetime_col: str
    ordering_col: str
    direction: str = "ascend"  # 'ascend' | 'descend'
    top_n: int = 10

    # Weighting
    weight_mode: str = "equal"  # equal | proportional | softmax | inverse_rank
    weight_col: Optional[str] = None
    softmax_tau: float = 1.0
    min_weight: float = 0.0
    max_weight: Optional[float] = None
    round_weights: Optional[int] = None

    # Selection controls
    include: Optional[Sequence[str]] = None
    exclude: Optional[Sequence[str]] = None
    dedupe: str = "first"          # first | none
    tie_breaker: str = "stable"    # stable | random
    random_state: Optional[int] = None


@dataclass
class SimulationConfig:
    """Simulation parameters exposed in the GUI."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float = 100_000.0

    # Ensemble/randomization
    shuffle_events: bool = False
    num_shuffle: int = 100
    shuffle_window_days: Optional[float] = None
    weight_method: Optional[str] = None  # None | 'random' | 'uniform'
    random_state: Optional[int] = None

    # Cash policy
    cash_policy: Optional[str] = None  # None | 'fixed' | 'proportion'
    cash_fixed_amount: float = 0.0
    cash_pct: float = 0.0

    # Trading/execution controls
    rebalance_mode: str = "adjust"  # adjust | rebuild
    use_adj_close: bool = False

    # Leverage / margin
    max_leverage: float = 1.0
    margin_rate_apr: float = 0.0

    # Benchmarks
    benchmark_symbols: Tuple[str, ...] = ("SPY", "NDX", "GLD")

    # Dollar-neutral overlay
    dollar_neutral: bool = False
    hedge_symbol: str = "SPY"
    hedge_notional_base: str = "total"  # total | gross_long
    hedge_rounding: str = "floor"       # floor | round


def _coerce_date(x: Any) -> pd.Timestamp:
    return pd.Timestamp(x).tz_localize(None).normalize()


def normalize_sheets(
    sheet_items: Sequence[SheetItem],
    *,
    ticker_col: str,
    datetime_col: str,
    ordering_col: str,
    weight_col: Optional[str] = None,
) -> List[pd.DataFrame]:
    """Prepare sheet DataFrames for ps.build_events_from_sheets.

    build_events_from_sheets expects each sheet to contain:
      - 'yf_Ticker'
      - a datetime column (group_col)
      - the ordering column (metric)

    We rename only ticker and datetime columns (if needed) and preserve the
    ordering column's original name so it propagates into Event.selection_column.
    """

    out: List[pd.DataFrame] = []
    for it in sheet_items:
        df = it.df.copy()

        # basic existence checks
        for required in (ticker_col, datetime_col, ordering_col):
            if required not in df.columns:
                raise KeyError(f"Sheet '{it.name}' missing required column: {required!r}")

        # rename ticker/datetime to canonical column names
        if ticker_col != "yf_Ticker":
            df = df.rename(columns={ticker_col: "yf_Ticker"})
        if datetime_col != "datetime":
            df = df.rename(columns={datetime_col: "datetime"})

        # strip / sanitize tickers (avoid placeholders like 'NONE' or '$NONE')
        def _clean_ticker(x: Any) -> Optional[str]:
            try:
                s = str(x).strip()
            except Exception:
                return None
            if not s:
                return None
            su = s.upper()
            if su in {"", "NONE", "NAN", "NULL", "N/A", "NA", "<NA>"}:
                return None
            if s.startswith("$"):
                s = s[1:].strip()
                if not s or s.upper() in {"NONE", "NAN", "NULL", "N/A", "NA", "<NA>"}:
                    return None
            # conservative whitelist: letters, numbers, '.', '-', '^', '='
            import re

            if not re.fullmatch(r"[A-Za-z0-9\.\-\^=]+", s):
                return None
            return s

        df["yf_Ticker"] = df["yf_Ticker"].apply(_clean_ticker)
        df = df.dropna(subset=["yf_Ticker"]).copy()

        # datetime coercion
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        # coerce ordering/weight numeric if possible (let build_events handle NaNs)
        df[ordering_col] = pd.to_numeric(df[ordering_col], errors="coerce")
        if weight_col is not None and weight_col in df.columns:
            df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")

        out.append(df)

    return out


def build_events(
    sheet_items: Sequence[SheetItem],
    cfg: EventBuildConfig,
) -> Tuple[List[ps.Event], pd.DataFrame, Dict[str, Any]]:
    """Build Events from sheet_items using cfg.

    Returns
    -------
    events : List[ps.Event]
    detail_df : pd.DataFrame
        Row-level detail about which tickers were selected per event.
    summary : dict
        High-level summary produced by ps.build_events_from_sheets.
    """

    sheets_norm = normalize_sheets(
        sheet_items,
        ticker_col=cfg.ticker_col,
        datetime_col=cfg.datetime_col,
        ordering_col=cfg.ordering_col,
        weight_col=cfg.weight_col,
    )

    events, detail_df, summary = ps.build_events_from_sheets(
        sheets_norm,
        column=cfg.ordering_col,
        direction=cfg.direction,
        top_n=int(cfg.top_n),
        group_col="datetime",
        weight_mode=cfg.weight_mode,
        weight_column=cfg.weight_col,
        softmax_tau=float(cfg.softmax_tau),
        min_weight=float(cfg.min_weight),
        max_weight=None if cfg.max_weight is None else float(cfg.max_weight),
        round_weights=cfg.round_weights,
        include=cfg.include,
        exclude=cfg.exclude,
        dedupe=cfg.dedupe,
        tie_breaker=cfg.tie_breaker,
        random_state=cfg.random_state,
        return_detail=True,
        return_summary=True,
        lookup_meta=True,
    )

    # Attach selection variable name to events explicitly (helps plotting)
    for e in events:
        try:
            e.selection_column = cfg.ordering_col
        except Exception:
            pass

    return events, detail_df, summary


def infer_sim_date_range_from_events(events: Sequence[ps.Event]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if not events:
        raise ValueError("No events available")
    d0 = min(_coerce_date(e.date) for e in events)
    d1 = max(_coerce_date(e.date) for e in events)
    return d0, d1


def make_simulation(events: Sequence[ps.Event], cfg: SimulationConfig) -> ps.Simulation:
    """Construct a ps.Simulation from cfg."""
    sim = ps.Simulation(
        start_date=_coerce_date(cfg.start_date),
        end_date=_coerce_date(cfg.end_date),
        initial_capital=float(cfg.initial_capital),
        events=list(events),
        shuffle_events=bool(cfg.shuffle_events),
        num_shuffle=int(cfg.num_shuffle),
        weight_method=cfg.weight_method,
        cash_policy=cfg.cash_policy,
        cash_fixed_amount=float(cfg.cash_fixed_amount),
        cash_pct=float(cfg.cash_pct),
        max_leverage=float(cfg.max_leverage),
        margin_rate_apr=float(cfg.margin_rate_apr),
        random_state=cfg.random_state,
        benchmark_symbols=tuple(cfg.benchmark_symbols),
        shuffle_window_days=cfg.shuffle_window_days,
        dollar_neutral=bool(cfg.dollar_neutral),
        hedge_symbol=str(cfg.hedge_symbol),
        hedge_notional_base=str(cfg.hedge_notional_base),
        hedge_rounding=str(cfg.hedge_rounding),
        rebalance_mode=str(cfg.rebalance_mode),
        use_adj_close=bool(cfg.use_adj_close),
        capture_trades=True,
    )
    return sim


def run_simulation(sim: ps.Simulation) -> Dict[str, pd.DataFrame]:
    """Run the simulation and return the engine result dict."""
    return sim.run()


def analyze_result(result: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute an Analysis.summary() table."""
    an = ps.Analysis(result)
    return an.summary()


def events_to_jsonable(events: Sequence[ps.Event]) -> List[Dict[str, Any]]:
    """Serialize events to a JSON-friendly list of dicts."""
    out: List[Dict[str, Any]] = []
    for e in events:
        row: Dict[str, Any] = {
            "date": pd.Timestamp(getattr(e, "date", e)).strftime("%Y-%m-%d"),
            "target_weights": dict(getattr(e, "target_weights", {}) or {}),
        }
        # Optional metadata if present
        for attr in ("selection_column", "selection_direction", "selection_top_n", "selection_order"):
            if hasattr(e, attr):
                try:
                    row[attr] = getattr(e, attr)
                except Exception:
                    pass
        out.append(row)
    return out


def dump_events_json(events: Sequence[ps.Event], *, indent: int = 2) -> str:
    return json.dumps(events_to_jsonable(events), indent=indent)


def result_to_parquet_bytes(result: Dict[str, pd.DataFrame]) -> bytes:
    """Package result frames into a single Parquet bytes payload (zip-like via bytes).

    Notes
    -----
    Parquet is convenient for download/export in Streamlit. We store each key
    as a separate parquet blob inside a dict serialized with pickle.
    """
    payload: Dict[str, bytes] = {}
    for k, df in result.items():
        if not isinstance(df, pd.DataFrame):
            continue
        payload[k] = df.to_parquet(index=True)
    import pickle

    return pickle.dumps(payload)
