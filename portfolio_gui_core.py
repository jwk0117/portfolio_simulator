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


# =============================================================================
# NYSE Trading Calendar Utilities
# =============================================================================

def _get_nyse_holidays(start_year: int = 2000, end_year: int = 2030) -> set:
    """Generate a set of NYSE holiday dates.
    
    This is a simplified holiday calendar. For production use, consider
    using the `exchange_calendars` or `pandas_market_calendars` package.
    """
    holidays = set()
    
    for year in range(start_year, end_year + 1):
        # New Year's Day (observed)
        nyd = pd.Timestamp(f"{year}-01-01")
        if nyd.weekday() == 6:  # Sunday -> Monday
            holidays.add(pd.Timestamp(f"{year}-01-02"))
        elif nyd.weekday() == 5:  # Saturday -> previous Friday (prior year)
            pass  # Skip, handled by prior year
        else:
            holidays.add(nyd)
        
        # MLK Day (3rd Monday of January)
        jan1 = pd.Timestamp(f"{year}-01-01")
        first_monday = jan1 + pd.Timedelta(days=(7 - jan1.weekday()) % 7)
        if first_monday.day > 1:
            first_monday = jan1 + pd.Timedelta(days=(0 - jan1.weekday()) % 7)
        mlk = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=(0 - pd.Timestamp(f"{year}-01-01").weekday()) % 7)
        # Find 3rd Monday
        d = pd.Timestamp(f"{year}-01-01")
        mondays = 0
        while mondays < 3:
            if d.weekday() == 0:
                mondays += 1
                if mondays == 3:
                    holidays.add(d)
                    break
            d += pd.Timedelta(days=1)
        
        # Presidents Day (3rd Monday of February)
        d = pd.Timestamp(f"{year}-02-01")
        mondays = 0
        while mondays < 3:
            if d.weekday() == 0:
                mondays += 1
                if mondays == 3:
                    holidays.add(d)
                    break
            d += pd.Timedelta(days=1)
        
        # Good Friday (variable - approximate)
        # This is complex to calculate; we'll use a simplified approach
        # For accuracy, consider using a library
        
        # Memorial Day (last Monday of May)
        d = pd.Timestamp(f"{year}-05-31")
        while d.weekday() != 0:
            d -= pd.Timedelta(days=1)
        holidays.add(d)
        
        # Juneteenth (June 19, observed)
        if year >= 2021:
            jt = pd.Timestamp(f"{year}-06-19")
            if jt.weekday() == 6:
                holidays.add(jt + pd.Timedelta(days=1))
            elif jt.weekday() == 5:
                holidays.add(jt - pd.Timedelta(days=1))
            else:
                holidays.add(jt)
        
        # Independence Day (July 4, observed)
        july4 = pd.Timestamp(f"{year}-07-04")
        if july4.weekday() == 6:
            holidays.add(july4 + pd.Timedelta(days=1))
        elif july4.weekday() == 5:
            holidays.add(july4 - pd.Timedelta(days=1))
        else:
            holidays.add(july4)
        
        # Labor Day (1st Monday of September)
        d = pd.Timestamp(f"{year}-09-01")
        while d.weekday() != 0:
            d += pd.Timedelta(days=1)
        holidays.add(d)
        
        # Thanksgiving (4th Thursday of November)
        d = pd.Timestamp(f"{year}-11-01")
        thursdays = 0
        while thursdays < 4:
            if d.weekday() == 3:
                thursdays += 1
                if thursdays == 4:
                    holidays.add(d)
                    break
            d += pd.Timedelta(days=1)
        
        # Christmas (December 25, observed)
        xmas = pd.Timestamp(f"{year}-12-25")
        if xmas.weekday() == 6:
            holidays.add(xmas + pd.Timedelta(days=1))
        elif xmas.weekday() == 5:
            holidays.add(xmas - pd.Timedelta(days=1))
        else:
            holidays.add(xmas)
    
    return holidays


_NYSE_HOLIDAYS = _get_nyse_holidays()


def _is_nyse_trading_day(d: pd.Timestamp) -> bool:
    """Check if a date is a NYSE trading day."""
    d = pd.Timestamp(d).normalize()
    # Weekend check
    if d.weekday() >= 5:
        return False
    # Holiday check
    if d in _NYSE_HOLIDAYS:
        return False
    return True


def _next_trading_day(d: pd.Timestamp, max_attempts: int = 10) -> pd.Timestamp:
    """Find the next NYSE trading day on or after d."""
    d = pd.Timestamp(d).normalize()
    for _ in range(max_attempts):
        if _is_nyse_trading_day(d):
            return d
        d += pd.Timedelta(days=1)
    # Fallback: return original + offset
    return d


def shift_events_by_calendar_days(
    events: Sequence[ps.Event],
    days: int,
    *,
    snap_to_trading_day: bool = True,
) -> List[ps.Event]:
    """Shift event dates by calendar days, optionally snapping to trading days.
    
    Parameters
    ----------
    events : Sequence[Event]
        Original events (not mutated).
    days : int
        Number of calendar days to shift. Positive = future, negative = past.
    snap_to_trading_day : bool
        If True, shifted dates that land on weekends/holidays are moved to
        the next valid NYSE trading day.
    
    Returns
    -------
    List[Event]
        New events with shifted dates, sorted chronologically.
    """
    shifted = []
    delta = pd.Timedelta(days=int(days))
    
    for e in events:
        new_date = (pd.Timestamp(e.date) + delta).normalize()
        
        if snap_to_trading_day:
            new_date = _next_trading_day(new_date)
        
        # Copy event with new date, preserving all attributes
        new_event = ps.Event(new_date, dict(e.target_weights))
        
        # Copy any extra attributes (like selection_column)
        for attr in ("selection_column", "selection_direction", "selection_top_n", "selection_order"):
            if hasattr(e, attr):
                try:
                    setattr(new_event, attr, getattr(e, attr))
                except Exception:
                    pass
        
        shifted.append(new_event)
    
    shifted.sort(key=lambda ev: ev.date)
    return shifted


# =============================================================================
# Monte Carlo Random Stock Selection
# =============================================================================

def build_random_events_from_sheets(
    sheets: List[pd.DataFrame],
    n_stocks: int,
    *,
    group_col: str = "datetime",
    random_state: Optional[int] = None,
) -> List[ps.Event]:
    """Build events by randomly selecting n_stocks from each sheet/event.
    
    Parameters
    ----------
    sheets : List[pd.DataFrame]
        Normalized sheets with 'yf_Ticker' and group_col columns.
    n_stocks : int
        Number of stocks to randomly select per event.
    group_col : str
        Column name for grouping (event dates).
    random_state : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    List[Event]
        Events with equal-weighted random stock selections.
    """
    rng = np.random.default_rng(random_state)
    events = []
    
    for sheet in sheets:
        if group_col not in sheet.columns:
            continue
        if "yf_Ticker" not in sheet.columns:
            continue
        
        # Group by event date
        for dt, group in sheet.groupby(group_col):
            tickers = group["yf_Ticker"].dropna().unique().tolist()
            
            if len(tickers) == 0:
                continue
            
            # Randomly select n_stocks (or all if fewer available)
            n_select = min(n_stocks, len(tickers))
            selected = rng.choice(tickers, size=n_select, replace=False).tolist()
            
            # Equal weights
            weight = 1.0 / n_select
            target_weights = {t: weight for t in selected}
            
            events.append(ps.Event(pd.Timestamp(dt).normalize(), target_weights))
    
    events.sort(key=lambda e: e.date)
    return events


def run_monte_carlo_null_distribution(
    sheet_items: Sequence[SheetItem],
    cfg: "EventBuildConfig",
    sim_cfg: "SimulationConfig",
    *,
    n_trials: int = 100,
    n_stocks: Optional[int] = None,
    random_state: Optional[int] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """Run Monte Carlo simulation with random stock selection.
    
    Parameters
    ----------
    sheet_items : Sequence[SheetItem]
        Original sheet data.
    cfg : EventBuildConfig
        Event configuration (used for column mapping).
    sim_cfg : SimulationConfig
        Simulation configuration.
    n_trials : int
        Number of random trials to run.
    n_stocks : int, optional
        Stocks per event. If None, uses cfg.top_n.
    random_state : int, optional
        Base random seed (each trial uses random_state + trial_idx).
    progress_callback : callable, optional
        Called with (current_trial, total_trials) for progress updates.
    
    Returns
    -------
    dict with keys:
        - 'final_values': list of final portfolio values
        - 'final_returns_pct': list of total returns (%)
        - 'all_series': list of portfolio value Series (for percentile bands)
        - 'percentiles': dict with p5, p10, p25, p50, p75, p90, p95 Series
    """
    from portfolio_gui_core import normalize_sheets
    
    if n_stocks is None:
        n_stocks = cfg.top_n
    
    sheets_norm = normalize_sheets(
        sheet_items,
        ticker_col=cfg.ticker_col,
        datetime_col=cfg.datetime_col,
        ordering_col=cfg.ordering_col,
        weight_col=cfg.weight_col,
    )
    
    final_values = []
    final_returns_pct = []
    all_series = []
    
    base_seed = random_state if random_state is not None else 42
    
    for trial in range(n_trials):
        if progress_callback:
            progress_callback(trial, n_trials)
        
        trial_seed = base_seed + trial
        
        # Build random events
        random_events = build_random_events_from_sheets(
            sheets_norm,
            n_stocks=n_stocks,
            group_col="datetime",
            random_state=trial_seed,
        )
        
        if not random_events:
            continue
        
        # Create and run simulation
        try:
            sim = ps.Simulation(
                start_date=_coerce_date(sim_cfg.start_date),
                end_date=_coerce_date(sim_cfg.end_date),
                initial_capital=float(sim_cfg.initial_capital),
                events=random_events,
                shuffle_events=False,  # No shuffling for null distribution
                cash_policy=sim_cfg.cash_policy,
                cash_fixed_amount=float(sim_cfg.cash_fixed_amount),
                cash_pct=float(sim_cfg.cash_pct),
                max_leverage=float(sim_cfg.max_leverage),
                margin_rate_apr=float(sim_cfg.margin_rate_apr),
                benchmark_symbols=(),  # Skip benchmarks for speed
                use_adj_close=bool(sim_cfg.use_adj_close),
                rebalance_mode=str(sim_cfg.rebalance_mode),
            )
            
            result = sim.run()
            
            # Extract portfolio value series
            strat_df = None
            for k in ("with_cash", "no_cash", "strategy"):
                if k in result and isinstance(result[k], pd.DataFrame):
                    strat_df = result[k]
                    break
            
            if strat_df is not None and "Portfolio Value" in strat_df.columns:
                pv = strat_df["Portfolio Value"].dropna()
                if len(pv) > 0:
                    all_series.append(pv)
                    final_val = float(pv.iloc[-1])
                    start_val = float(pv.iloc[0])
                    final_values.append(final_val)
                    final_returns_pct.append((final_val / start_val - 1.0) * 100.0)
        
        except Exception:
            # Skip failed trials
            continue
    
    if progress_callback:
        progress_callback(n_trials, n_trials)
    
    # Compute percentile bands across all series
    percentiles = {}
    if all_series:
        # Align all series to common index
        common_idx = all_series[0].index
        for s in all_series[1:]:
            common_idx = common_idx.union(s.index)
        common_idx = common_idx.sort_values()
        
        aligned = []
        for s in all_series:
            aligned.append(s.reindex(common_idx).ffill().bfill())
        
        df_all = pd.DataFrame(aligned).T
        df_all.index = common_idx
        
        for p in [5, 10, 25, 50, 75, 90, 95]:
            percentiles[f"p{p}"] = df_all.quantile(p / 100.0, axis=1)
        
        percentiles["mean"] = df_all.mean(axis=1)
    
    return {
        "final_values": final_values,
        "final_returns_pct": final_returns_pct,
        "all_series": all_series,
        "percentiles": percentiles,
        "n_trials": len(final_values),
    }


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
