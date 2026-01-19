# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Iterable, Optional, Tuple, List, Dict
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math

from typing import Dict, List, Optional, Tuple

__all__ = ["MarketData", "Benchmarks", "Event", "Portfolio", "Simulation", "plot_simulation","build_events_from_sheets", "Analysis", "compute_holdings_heatmap"]


# =========================
# Ticker sanitation / robustness
# =========================

# Common placeholder / invalid "tickers" that can leak in from spreadsheets or parsing.
_YF_INVALID_TICKERS = {
    "",
    "NONE",
    "NAN",
    "NULL",
    "N/A",
    "NA",
    "<NA>",
}


def clean_yf_ticker(t: object) -> Optional[str]:
    """Return a sanitized yfinance-compatible ticker string or None.

    This prevents placeholder values (e.g., 'NONE') or illegal strings (e.g., '$NONE')
    from propagating into yfinance calls and causing hard failures.

    Rules
    -----
    - Strip whitespace.
    - Drop blanks and common placeholders.
    - Drop leading '$' (common spreadsheet artifact) and reject if empty afterward.
    - Permit typical Yahoo ticker characters: letters, numbers, '.', '-', '^', '='.
    """
    if t is None:
        return None
    s = str(t).strip()
    if not s:
        return None
    s_up = s.upper()
    if s_up in _YF_INVALID_TICKERS:
        return None
    if s.startswith("$"):
        s = s[1:].strip()
        if not s:
            return None
        if s.upper() in _YF_INVALID_TICKERS:
            return None

    # Basic character whitelist. Keep this conservative; reject obviously invalid inputs.
    # Examples allowed: AAPL, BRK-B, 0700.HK, ^NDX, JPY=X
    import re

    if not re.fullmatch(r"[A-Za-z0-9\.\-\^=]+", s):
        return None
    return s

warnings.simplefilter(action='ignore', category=FutureWarning)



try:
    __all__.append("event_date_range")
except NameError:
    __all__ = ["event_date_range"]

def event_date_range(events: List["Event"]) -> Tuple[datetime, datetime]:
    """
    Return (earliest_event_datetime, latest_event_datetime) and print summary stats.

    Prints:
      - First/last event dates
      - Span in days (fractional)
      - Total events and intervals (n-1)
      - Average days/event = span_days / intervals
    """
    if not events:
        raise ValueError("No events provided.")

    # Accept Event objects with .date or raw datetime-like entries
    dates = [pd.Timestamp(getattr(e, "date", e)) for e in events]
    start = min(dates).to_pydatetime()
    end   = max(dates).to_pydatetime()

    n_events = len(dates)
    intervals = max(n_events - 1, 0)

    span_days = (end - start).total_seconds() / 86400.0

    print(f"First event: {start.date()}")
    print(f"Last event:  {end.date()}")
    print(f"Span: {span_days:.6f} day(s)")
    print(f"Total events: {n_events} | Intervals: {intervals}")

    if intervals > 0 and span_days > 0:
        days_per_event = span_days / intervals
        print(f"Average days/event (span/intervals): {days_per_event:.6g} days/event")
    else:
        print("Average days/event: n/a (need at least two events and nonzero span)")

    return start, end

# near the top of portfolio_sim.py
try:
    __all__.append("shift_events_by_days")
except NameError:
    __all__ = ["shift_events_by_days"]

def shift_events_by_days(events: List[Event], days: int) -> List[Event]:
    """
    Return a new list of Events with each event's date shifted by `days` calendar days.

    Parameters
    ----------
    events : List[Event]
        Existing events (will NOT be mutated).
    days : int
        Number of calendar days to shift. Positive moves forward; negative moves backward.

    Notes
    -----
    - Dates are normalized to midnight (consistent with Event).
    - Event weights are copied as-is.
    - Result is sorted chronologically by date.
    """
    if not isinstance(days, (int, np.integer)):
        raise TypeError("`days` must be an integer (calendar days).")
    
    shifted = []
    delta = pd.Timedelta(days=int(days))
    for e in events:
        new_date = (pd.Timestamp(e.date) + delta).normalize()
        shifted.append(Event(new_date, dict(e.target_weights)))
    shifted.sort(key=lambda ev: ev.date)
    return shifted


def _is_ensemble(df: pd.DataFrame) -> bool:
    return isinstance(df.index, pd.MultiIndex) and "sim_id" in df.index.names

def _plot_single_path(ax, df: pd.DataFrame, label: str, lw: float = 2.0, alpha: float = 1.0, *, y_as: str = "dollar"):
    if "Portfolio Value" not in df.columns or df.empty:
        return
    s = df["Portfolio Value"].copy()
    if y_as.lower() in {"pct", "percent", "percentage"}:
        base = s.dropna().iloc[0]
        if base and np.isfinite(base):
            s = (s / base - 1.0) * 100.0
    s.plot(ax=ax, label=label, linewidth=lw, alpha=alpha)

def _plot_ensemble(ax,
                   edf: pd.DataFrame,
                   label_prefix: str,
                   quantiles=(0.05, 0.5, 0.95),
                   shade_alpha: float = 0.15,
                   sample_paths: int = 10,
                   lw: float = 2.0,
                   *,
                   y_as: str = "dollar"):
    if "Portfolio Value" not in edf.columns or edf.empty:
        return
    # pivot sims: index=Date, columns=sim_id
    qdf = edf["Portfolio Value"].unstack("sim_id").sort_index()
    # Rebase to % if requested: (each sim column independently)
    if y_as.lower() in {"pct", "percent", "percentage"}:
        def _rebase(col):
            b = col.dropna().iloc[0] if col.dropna().size else np.nan
            return (col / b - 1.0) * 100.0 if b and np.isfinite(b) else col * np.nan
        qdf = qdf.apply(_rebase, axis=0)

    qs = qdf.quantile(list(quantiles), axis=1)
    if len(quantiles) >= 3:
        lower, median, upper = quantiles[0], quantiles[len(quantiles)//2], quantiles[-1]
        ax.fill_between(qdf.index, qs.loc[lower], qs.loc[upper],
                        alpha=shade_alpha, label=f"{label_prefix} {int(lower*100)}–{int(upper*100)}%")
        ax.plot(qdf.index, qs.loc[median], label=f"{label_prefix} median", linewidth=lw)
    else:
        ax.plot(qdf.index, qs.loc[quantiles[-1]], label=f"{label_prefix} q{int(quantiles[-1]*100)}", linewidth=lw)

    # A few sample paths
    for sid in qdf.columns[:max(0, sample_paths)]:
        ax.plot(qdf.index, qdf[sid], alpha=0.25, linewidth=1.0)


def compute_holdings_heatmap(
    *,
    sim: "Simulation",
    strategy_df: pd.DataFrame,
    top_n: Optional[int] = None,
    clip: Optional[float] = 20.0,
) -> Dict[str, object]:
    """Compute holdings heatmap data for use in interactive plotting.

    This function extracts the heatmap computation logic from plot_simulation
    so it can be called independently by Plotly-based plotting code.

    Parameters
    ----------
    sim : Simulation
        The simulation object containing events and market data.
    strategy_df : pd.DataFrame
        The strategy DataFrame with 'Portfolio Value' and '<TICKER> Shares' columns.
    top_n : int, optional
        Maximum number of slots (tickers per event) to display. If None, inferred
        from the events.
    clip : float, optional
        Clip returns to [-clip, +clip] percent. Default is 20.

    Returns
    -------
    dict
        A dictionary containing:
        - 'returns_pct': 2D numpy array of shape (n_slots, n_events) with % returns
        - 'tickers': 2D numpy array of shape (n_slots, n_events) with ticker symbols
        - 'x_dates': list of event execution dates
        - 'y_labels': list of slot labels (e.g., ['1', '2', ...])
        - 'ordering_label': the name of the selection ordering variable
        - 'clip': the clip value used
        - 'avg_returns_pct': list of average returns per slot (row)
    """
    if sim is None:
        raise ValueError("sim is required to compute holdings heatmap")

    if strategy_df is None or strategy_df.empty:
        raise ValueError("strategy_df is required and must not be empty")

    if "Portfolio Value" not in strategy_df.columns:
        raise ValueError("strategy_df must contain 'Portfolio Value' column")

    # Check for ensemble DataFrame (not supported)
    if {"Ensemble Mean", "Ensemble P10", "Ensemble P90"}.issubset(strategy_df.columns):
        raise ValueError("Holdings heatmap requires a non-ensemble strategy dataframe")

    # Get price data from sim
    price_df = None
    if getattr(sim, "market_data", None) is not None:
        try:
            price_df = getattr(sim.market_data, "usd_data", None)
        except Exception:
            price_df = None

    if price_df is None or not isinstance(price_df, pd.DataFrame) or price_df.empty:
        raise ValueError("Price data is unavailable from sim.market_data.usd_data")

    # Get share columns
    share_cols = [c for c in strategy_df.columns if c.endswith(" Shares")]
    if not share_cols:
        raise ValueError("strategy_df must contain '<TICKER> Shares' columns")

    idx = strategy_df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)

    def _map_to_execution_date(idx: pd.DatetimeIndex, d: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Map an intended event date to the first index date on/after it."""
        if idx is None or len(idx) == 0:
            return None
        d = pd.Timestamp(d)
        pos = idx.searchsorted(d, side="left")
        if pos >= len(idx):
            return None
        return pd.Timestamp(idx[pos])

    # Determine event execution dates
    event_exec_dates = []
    if getattr(sim, "events", None) and idx is not None:
        for e in getattr(sim, "events", []):
            d0 = pd.Timestamp(getattr(e, "date", e)).normalize()
            d_exec = _map_to_execution_date(idx, d0)
            if d_exec is not None:
                event_exec_dates.append(d_exec)
    else:
        # Fallback: infer from share changes
        sh = strategy_df[share_cols].copy()
        changed = (sh.diff().fillna(0).abs().sum(axis=1) > 0)
        if len(changed) > 0:
            changed.iloc[0] = True
        event_exec_dates = [pd.Timestamp(d) for d in sh.index[changed]]

    event_exec_dates = sorted({pd.Timestamp(d) for d in event_exec_dates})

    if not event_exec_dates:
        raise ValueError("Could not determine event dates for holdings heatmap")

    # Windows: each event is a start; the corresponding end is the next event (or last date)
    last_date = pd.Timestamp(idx.max())
    window_starts = event_exec_dates
    window_ends = window_starts[1:] + [last_date]

    # Determine how many rows (slots) to display
    n_slots = top_n
    if n_slots is None:
        if getattr(sim, "events", None):
            try:
                n_slots = max(
                    len([t for t, w in (e.target_weights or {}).items() if (w is not None and float(w) > 0.0)])
                    for e in getattr(sim, "events", [])
                )
            except Exception:
                n_slots = None
        if n_slots is None or n_slots <= 0:
            # Fallback: max count of positive-share tickers at window starts
            n_slots = 0
            for d0 in window_starts:
                if d0 not in strategy_df.index:
                    continue
                row = strategy_df.loc[d0, share_cols]
                n_pos = int((row > 0).sum())
                n_slots = max(n_slots, n_pos)
            n_slots = max(n_slots, 1)

    hedge_symbol = getattr(sim, "hedge_symbol", None) if (getattr(sim, "dollar_neutral", False)) else None

    # Determine the label for the selection-ordering variable
    order_var = None
    for _attr in ("selection_column", "selection_metric", "ordering_column", "rank_column", "order_by"):
        if hasattr(sim, _attr):
            _val = getattr(sim, _attr, None)
            if _val is not None and str(_val) != "":
                order_var = str(_val)
                break
    if order_var is None and getattr(sim, "events", None):
        _e0 = getattr(sim, "events", [None])[0]
        if _e0 is not None:
            for _attr in ("selection_column", "selection_metric", "ordering_column", "rank_column", "order_by"):
                _val = getattr(_e0, _attr, None)
                if _val is not None and str(_val) != "":
                    order_var = str(_val)
                    break
    if order_var is None:
        order_var = "selection metric"

    # Build a mapping from execution date -> Event to preserve the original selection ordering
    event_exec_to_event: Dict[pd.Timestamp, "Event"] = {}
    if getattr(sim, "events", None):
        for _e in getattr(sim, "events", []):
            _d0 = pd.Timestamp(getattr(_e, "date", _e)).normalize()
            _d_exec = _map_to_execution_date(idx, _d0)
            if _d_exec is not None:
                event_exec_to_event[pd.Timestamp(_d_exec)] = _e

    col_to_ticker = {c: c[: -len(" Shares")] for c in share_cols}

    heat = np.full((int(n_slots), len(window_starts)), np.nan, dtype=float)
    tickers_grid = np.full((int(n_slots), len(window_starts)), "", dtype=object)

    for j, (d_start, d_end) in enumerate(zip(window_starts, window_ends)):
        d_end = min(pd.Timestamp(d_end), last_date)

        if d_start not in strategy_df.index:
            continue

        shares_row = strategy_df.loc[d_start, share_cols]
        tickers_held = []
        for c in share_cols:
            t = col_to_ticker.get(c, None)
            if t is None:
                continue
            if hedge_symbol is not None and t == hedge_symbol:
                continue
            try:
                sh = float(shares_row[c])
            except Exception:
                sh = 0.0
            if np.isfinite(sh) and sh > 0:
                tickers_held.append(t)

        if not tickers_held:
            continue

        # Prefer the original selection ordering when we can map this window start to an Event
        tickers_sorted = None
        eobj = event_exec_to_event.get(pd.Timestamp(d_start), None)
        if eobj is not None:
            tw = getattr(eobj, 'target_weights', {}) or {}
            tickers_sorted = [
                t for t, w in tw.items()
                if (w is not None and np.isfinite(float(w)) and float(w) > 0.0)
            ]
            if hedge_symbol is not None:
                tickers_sorted = [t for t in tickers_sorted if t != hedge_symbol]

        if tickers_sorted is None:
            # Fallback: sort tickers by position value at the start (descending)
            pv = {}
            for t in tickers_held:
                p0 = price_df.get(t, pd.Series(index=price_df.index, dtype=float)).reindex([d_start]).iloc[0] if t in price_df.columns else np.nan
                sh0 = strategy_df.loc[d_start, f"{t} Shares"] if f"{t} Shares" in strategy_df.columns else np.nan
                try:
                    pv[t] = float(sh0) * float(p0) if np.isfinite(sh0) and np.isfinite(p0) else -np.inf
                except Exception:
                    pv[t] = -np.inf
            tickers_sorted = sorted(tickers_held, key=lambda t: pv.get(t, -np.inf), reverse=True)

        tickers_sorted = tickers_sorted[: int(n_slots)]

        for i, t in enumerate(tickers_sorted):
            tickers_grid[i, j] = t
            if t not in price_df.columns:
                continue
            p0 = price_df.at[d_start, t] if (d_start in price_df.index) else np.nan
            p1 = price_df.at[d_end, t] if (d_end in price_df.index) else np.nan
            if p0 is None or p1 is None or np.isnan(p0) or np.isnan(p1) or float(p0) <= 0:
                continue
            heat[i, j] = (float(p1) / float(p0) - 1.0) * 100.0

    # Apply clipping
    clip_val = float(clip) if clip is not None and np.isfinite(float(clip)) else 20.0
    heat = np.clip(heat, -clip_val, clip_val)

    # Interpolate NaN values along the time axis (within each row/slot)
    # For each NaN, use the average of the nearest non-NaN values before and after
    def _interpolate_row(row: np.ndarray) -> np.ndarray:
        """Interpolate NaN values in a 1D array using nearest neighbors in time."""
        result = row.copy()
        n = len(row)
        for i in range(n):
            if np.isfinite(row[i]):
                continue
            # Find nearest non-NaN before
            before_val = None
            for j in range(i - 1, -1, -1):
                if np.isfinite(row[j]):
                    before_val = row[j]
                    break
            # Find nearest non-NaN after
            after_val = None
            for j in range(i + 1, n):
                if np.isfinite(row[j]):
                    after_val = row[j]
                    break
            # Compute interpolated value
            if before_val is not None and after_val is not None:
                result[i] = (before_val + after_val) / 2.0
            elif before_val is not None:
                result[i] = before_val
            elif after_val is not None:
                result[i] = after_val
            # else: leave as NaN (entire row is NaN)
        return result

    for i in range(heat.shape[0]):
        heat[i, :] = _interpolate_row(heat[i, :])

    # Compute average returns per slot (row)
    avg_returns = []
    for i in range(int(n_slots)):
        row_vals = heat[i, :]
        finite_vals = row_vals[np.isfinite(row_vals)]
        if len(finite_vals) > 0:
            avg_returns.append(float(np.mean(finite_vals)))
        else:
            avg_returns.append(float("nan"))

    y_labels = [str(i + 1) for i in range(int(n_slots))]

    return {
        "returns_pct": heat,
        "tickers": tickers_grid,
        "x_dates": window_starts,
        "y_labels": y_labels,
        "ordering_label": order_var,
        "clip": clip_val,
        "avg_returns_pct": avg_returns,
    }


def plot_simulation(
    *args,
    result: Optional[Dict[str, pd.DataFrame]] = None,
    sim: Optional["Simulation"] = None,
    show_benchmarks: bool = True,
    show_band: bool = True,
    show_median: bool = False,
    legend_loc: str = "best",
    figsize=(11, 5),
    title: Optional[str] = None,
    y_as: str = "value",                         # 'value' | 'pct' | 'index' | 'log'
    ensemble_quantiles: Tuple[float, ...] = (0.1, 0.9),
    # NEW: event/holding-window visualization
    show_events: bool = True,
    show_holdings_heatmap: bool = True,
    event_linewidth: float = 0.8,
    event_alpha: float = 0.35,
    heatmap_cmap: str = "red_white_green",
    heatmap_clip: Optional[float] = None,         # e.g., 15 => clip to [-15, +15]
    heatmap_xtick_fontsize: int = 7,
    heatmap_ordering_label: Optional[str] = None,   # label for the selection-ordering variable
    **kwargs  # accept deprecated/extra args like ensemble_samples
):
    """Plot the simulation result.

    Enhancements
    ------------
    1) If `show_events=True`, draw thin vertical event markers (linestyle=':') on the main plot.
    2) If `show_holdings_heatmap=True` and the plotted strategy frame contains per-ticker share columns,
       draw a second subplot below showing per-window per-stock % change during the holding interval.

    Notes
    -----
    - Event markers are placed on the *execution date* (the first trading day on/after the Event.date).
    - The heatmap requires per-ticker share columns (e.g., '<TICKER> Shares') and price data.
      If `sim` is provided, price data is sourced from `sim.market_data.usd_data`.
    """

    # ---- deprecation/compat ----
    if "ensemble_samples" in kwargs:
        warnings.warn("plot_simulation: 'ensemble_samples' is ignored (no sampling done in this plot).", RuntimeWarning)

    # ---- flexible arg binding (supports plot_simulation(result, sim=...) or plot_simulation(sim, result=...) ) ----
    for obj in args:
        if isinstance(obj, dict) and result is None:
            result = obj
        elif hasattr(obj, "__class__") and obj.__class__.__name__ == "Simulation" and sim is None:
            sim = obj
        else:
            pass

    if result is None:
        raise ValueError("plot_simulation: 'result' dict is required.")

    def _is_ensemble_df(df: pd.DataFrame) -> bool:
        return {"Ensemble Mean", "Ensemble P10", "Ensemble P90"}.issubset(df.columns)

    def _reexpress(df: pd.DataFrame) -> pd.DataFrame:
        """Re-express values according to y_as, column-wise base on first non-NaN."""
        if y_as not in ("value", "pct", "index", "log"):
            raise ValueError("y_as must be one of: 'value', 'pct', 'index', 'log'")
        if y_as in ("value", "log"):
            return df
        out = df.copy()
        for col in out.columns:
            s = out[col].dropna()
            if s.empty:
                continue
            base = s.iloc[0]
            if base == 0:
                continue
            if y_as == "pct":
                out[col] = (out[col] / base - 1.0) * 100.0
            elif y_as == "index":
                out[col] = (out[col] / base) * 100.0
        return out

    def _plot_single_curve(ax, df: pd.DataFrame, label: str):
        if "Portfolio Value" not in df.columns:
            raise ValueError(f"{label}: missing 'Portfolio Value'.")
        s = df["Portfolio Value"].to_frame()
        s = _reexpress(s)
        s["Portfolio Value"].plot(ax=ax, lw=2, label=label)

    def _plot_ensemble(ax, df: pd.DataFrame, label_prefix: str):
        if not _is_ensemble_df(df):
            raise ValueError(f"{label_prefix}: not an ensemble DataFrame.")
        # Map requested quantiles to available columns
        q_low, q_med, q_hi = None, None, None
        if len(ensemble_quantiles) == 3 and tuple(ensemble_quantiles) == (0.1, 0.5, 0.9):
            q_low, q_med, q_hi = "Ensemble P10", "Ensemble Median", "Ensemble P90"
        elif len(ensemble_quantiles) == 2 and tuple(ensemble_quantiles) == (0.1, 0.9):
            q_low, q_hi = "Ensemble P10", "Ensemble P90"
        else:
            warnings.warn("Requested 'ensemble_quantiles' not available; using P10–P90 and Mean.", RuntimeWarning)
            q_low, q_hi = "Ensemble P10", "Ensemble P90"

        plot_cols = ["Ensemble Mean"]
        if q_low:
            plot_cols.append(q_low)
        if q_hi and q_hi not in plot_cols:
            plot_cols.append(q_hi)
        if q_med and q_med not in plot_cols:
            plot_cols.append(q_med)

        df2 = _reexpress(df[plot_cols])

        if show_band and q_low and q_hi:
            ax.fill_between(df.index, df2[q_low], df2[q_hi], alpha=0.15,
                            label=f"{label_prefix} {q_low.split()[-1]}–{q_hi.split()[-1]}")

        df2["Ensemble Mean"].plot(ax=ax, lw=2, label=f"{label_prefix} Mean")
        if show_median and q_med and q_med in df2.columns:
            df2[q_med].plot(ax=ax, lw=1.5, linestyle="--", label=f"{label_prefix} Median")

    def _plot_benchmarks(ax, bench: pd.DataFrame):
        if bench is None or bench.empty:
            return
        bench2 = _reexpress(bench)
        for col in bench2.columns:
            bench2[col].plot(ax=ax, lw=1, alpha=0.8, label=f"Benchmark: {col}")

    def _pick_primary_strategy_df(res: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Choose a strategy DataFrame to use for event mapping/heatmap.

        Preference order mirrors plotting preference: with_cash -> no_cash -> strategy.
        """
        for k in ("with_cash", "no_cash", "strategy"):
            df = res.get(k, None)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        return None

    def _map_to_execution_date(idx: pd.DatetimeIndex, d: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Map an intended event date to the first index date on/after it."""
        if idx is None or len(idx) == 0:
            return None
        d = pd.Timestamp(d)
        # idx must be sorted
        pos = idx.searchsorted(d, side="left")
        if pos >= len(idx):
            return None
        return pd.Timestamp(idx[pos])

    # ---------------------- select what to plot ----------------------
    strat_items: list[tuple[str, pd.DataFrame]] = []
    if "with_cash" in result or "no_cash" in result:
        if "with_cash" in result:
            strat_items.append(("With Cash", result["with_cash"]))
        if "no_cash" in result:
            strat_items.append(("No Cash", result["no_cash"]))
    elif "strategy" in result:
        strat_items.append(("Strategy", result["strategy"]))
    else:
        raise ValueError("Result dict missing expected keys.")

    bench_df = result.get("benchmarks", None)

    # Determine figure geometry: add height when heatmap is requested and feasible
    fig_w, fig_h = float(figsize[0]), float(figsize[1])
    primary_df = _pick_primary_strategy_df(result)
    shares_available = False
    if isinstance(primary_df, pd.DataFrame) and ("Portfolio Value" in primary_df.columns):
        share_cols = [c for c in primary_df.columns if c.endswith(" Shares")]
        shares_available = len(share_cols) > 0

    want_heatmap = bool(show_holdings_heatmap and shares_available)
    if want_heatmap:
        fig_h = fig_h * 1.45
        fig, (ax, ax_hm) = plt.subplots(
            2, 1,
            figsize=(fig_w, fig_h),
            gridspec_kw={"height_ratios": [3.2, 1.5]},
            constrained_layout=True,
        )
    else:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax_hm = None

    # ---------------------- main plot ----------------------
    for label, df in strat_items:
        if _is_ensemble_df(df):
            _plot_ensemble(ax, df, label_prefix=label)
        else:
            _plot_single_curve(ax, df, label=label)

    if show_benchmarks and isinstance(bench_df, pd.DataFrame) and not bench_df.empty:
        _plot_benchmarks(ax, bench_df)

    ax.set_xlabel("Date")
    if y_as == "pct":
        ax.set_ylabel("Return (%)")
    elif y_as == "index":
        ax.set_ylabel("Index (base = 100)")
    else:
        ax.set_ylabel("Value (USD)")
    if y_as == "log":
        ax.set_yscale("log")

    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc=legend_loc)

    if title is None:
        title = "Simulation Results"
    ax.set_title(title, fontsize=12, pad=10)

    # Subtitle with shuffle-window awareness
    subtitle_bits = []
    if sim is not None:
        if getattr(sim, "shuffle_events", False):
            if getattr(sim, "shuffle_window_days", None) is None:
                subtitle_bits.append("Shuffling: enabled (no time window)")
            else:
                w = sim.shuffle_window_days
                window_txt = f"±{int(w) if float(w).is_integer() else w:g} days"
                subtitle_bits.append(f"Shuffling: enabled ({window_txt})")
            subtitle_bits.append(f"Members: {getattr(sim, 'num_shuffle', 'N/A')}")
        else:
            subtitle_bits.append("Shuffling: disabled")
        if getattr(sim, "weight_method", None):
            subtitle_bits.append(f"Weights: {sim.weight_method}")
        if getattr(sim, "cash_policy", None):
            subtitle_bits.append(f"Cash policy: {sim.cash_policy}")

        lev = getattr(sim, "max_leverage", 1.0)
        apr = getattr(sim, "margin_rate_apr", 0.0)
        try:
            lev_f = float(lev)
        except Exception:
            lev_f = 1.0
        try:
            apr_f = float(apr)
        except Exception:
            apr_f = 0.0
        if np.isfinite(lev_f) and lev_f > 1.0:
            subtitle_bits.append(f"Max leverage: {lev_f:g}x")
        if np.isfinite(apr_f) and apr_f > 0.0:
            subtitle_bits.append(f"Margin APR: {apr_f*100:.2f}%")
        if getattr(sim, "dollar_neutral", False):
            hedge = getattr(sim, "hedge_symbol", "SPY")
            base = getattr(sim, "hedge_notional_base", "total")
            subtitle_bits.append(f"Dollar neutral: short {hedge} ({base})")

    if subtitle_bits:
        ax.annotate(
            " | ".join(subtitle_bits),
            xy=(0, 1), xycoords="axes fraction",
            xytext=(0, -0.10), textcoords="axes fraction",
            ha="left", va="top", fontsize=9,
        )

    if sim is not None and getattr(sim, "shuffle_events", False) and getattr(sim, "shuffle_window_days", None) is not None:
        ax.annotate(
            "Note: Events without a valid target within the window remain unshifted.",
            xy=(0, 0), xycoords="axes fraction",
            xytext=(0, -0.18), textcoords="axes fraction",
            ha="left", va="top", fontsize=8, alpha=0.8,
        )

    # Force x tick labels on the main plot to be 90 degrees (vertical)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(90)
        lbl.set_ha("center")
        lbl.set_va("top")

    # ---------------------- event markers ----------------------
    event_exec_dates: list[pd.Timestamp] = []
    if show_events:
        # Prefer sim.events if available; else infer from share changes.
        idx = None
        if primary_df is not None and isinstance(primary_df.index, pd.DatetimeIndex):
            idx = primary_df.index
        if sim is not None and getattr(sim, "events", None) and idx is not None:
            for e in getattr(sim, "events", []):
                d0 = pd.Timestamp(getattr(e, "date", e)).normalize()
                d_exec = _map_to_execution_date(idx, d0)
                if d_exec is not None:
                    event_exec_dates.append(d_exec)
        elif shares_available and idx is not None and primary_df is not None:
            share_cols = [c for c in primary_df.columns if c.endswith(" Shares")]
            if share_cols:
                sh = primary_df[share_cols].copy()
                # Event dates where any share changes (including first date)
                changed = (sh.diff().fillna(0).abs().sum(axis=1) > 0)
                if len(changed) > 0:
                    changed.iloc[0] = True
                event_exec_dates = [pd.Timestamp(d) for d in sh.index[changed]]

        # De-dupe and sort
        if event_exec_dates:
            event_exec_dates = sorted({pd.Timestamp(d) for d in event_exec_dates})

        # Draw thin event lines with ':' linestyle
        first = True
        for d in event_exec_dates:
            ax.axvline(
                d,
                linestyle=":",
                linewidth=float(event_linewidth),
                alpha=float(event_alpha),
                color="k",
                label="Event" if first else None,
                zorder=0,
            )
            first = False

        # If we added a legend label for Event, refresh legend
        if event_exec_dates:
            ax.legend(loc=legend_loc)

    # ---------------------- holdings heatmap subplot ----------------------
    if want_heatmap and ax_hm is not None and primary_df is not None:
        # Heatmap requires price data; source from sim.market_data.usd_data when available.
        price_df = None
        if sim is not None and getattr(sim, "market_data", None) is not None:
            try:
                price_df = getattr(sim.market_data, "usd_data", None)
            except Exception:
                price_df = None

        if price_df is None or not isinstance(price_df, pd.DataFrame) or price_df.empty:
            warnings.warn(
                "plot_simulation: holdings heatmap requested but price data is unavailable; skipping heatmap.",
                RuntimeWarning,
            )
        else:
            # Use event dates inferred above; if none, infer from share changes now.
            if not event_exec_dates:
                idx = primary_df.index
                share_cols = [c for c in primary_df.columns if c.endswith(" Shares")]
                sh = primary_df[share_cols].copy()
                changed = (sh.diff().fillna(0).abs().sum(axis=1) > 0)
                if len(changed) > 0:
                    changed.iloc[0] = True
                event_exec_dates = [pd.Timestamp(d) for d in sh.index[changed]]
                event_exec_dates = sorted({pd.Timestamp(d) for d in event_exec_dates})

            if not event_exec_dates:
                warnings.warn(
                    "plot_simulation: could not determine event dates for holdings heatmap; skipping heatmap.",
                    RuntimeWarning,
                )
            else:
                # Windows: each event is a start; the corresponding end is the next event (or last date)
                idx = primary_df.index
                last_date = pd.Timestamp(idx.max())

                window_starts = event_exec_dates
                window_ends = window_starts[1:] + [last_date]

                # Determine how many rows (slots) to display
                n_slots = None
                if sim is not None and getattr(sim, "events", None):
                    try:
                        n_slots = max(
                            len([t for t, w in (e.target_weights or {}).items() if (w is not None and float(w) > 0.0)])
                            for e in getattr(sim, "events", [])
                        )
                    except Exception:
                        n_slots = None
                if n_slots is None or n_slots <= 0:
                    # Fallback: max count of positive-share tickers at window starts
                    share_cols = [c for c in primary_df.columns if c.endswith(" Shares")]
                    n_slots = 0
                    for d0 in window_starts:
                        if d0 not in primary_df.index:
                            continue
                        row = primary_df.loc[d0, share_cols]
                        n_pos = int((row > 0).sum())
                        n_slots = max(n_slots, n_pos)
                    n_slots = max(n_slots, 1)

                hedge_symbol = getattr(sim, "hedge_symbol", None) if (sim is not None and getattr(sim, "dollar_neutral", False)) else None

                # Determine the label for the selection-ordering variable (used in the y-axis label).
                order_var = heatmap_ordering_label
                if order_var is None:
                    # Common attribute names that upstream code may use.
                    for _attr in ("selection_column", "selection_metric", "ordering_column", "rank_column", "order_by"):
                        if sim is not None and hasattr(sim, _attr):
                            _val = getattr(sim, _attr, None)
                            if _val is not None and str(_val) != "":
                                order_var = str(_val)
                                break
                if order_var is None and sim is not None and getattr(sim, "events", None):
                    _e0 = getattr(sim, "events", [None])[0]
                    for _attr in ("selection_column", "selection_metric", "ordering_column", "rank_column", "order_by"):
                        _val = getattr(_e0, _attr, None)
                        if _val is not None and str(_val) != "":
                            order_var = str(_val)
                            break
                if order_var is None:
                    order_var = "selection metric"

                # Build a mapping from execution date -> Event to preserve the original selection ordering.
                event_exec_to_event: Dict[pd.Timestamp, Event] = {}
                if sim is not None and getattr(sim, "events", None):
                    for _e in getattr(sim, "events", []):
                        _d0 = pd.Timestamp(getattr(_e, "date", _e)).normalize()
                        _d_exec = _map_to_execution_date(idx, _d0)
                        if _d_exec is not None:
                            event_exec_to_event[pd.Timestamp(_d_exec)] = _e


                share_cols = [c for c in primary_df.columns if c.endswith(" Shares")]
                # map share col -> ticker
                col_to_ticker = {c: c[: -len(" Shares")] for c in share_cols}

                heat = np.full((int(n_slots), len(window_starts)), np.nan, dtype=float)

                for j, (d_start, d_end) in enumerate(zip(window_starts, window_ends)):
                    # Clamp end to last_date
                    d_end = min(pd.Timestamp(d_end), last_date)

                    # Pull tickers held immediately after event (positive shares)
                    if d_start not in primary_df.index:
                        continue

                    shares_row = primary_df.loc[d_start, share_cols]
                    tickers_held = []
                    for c in share_cols:
                        t = col_to_ticker.get(c, None)
                        if t is None:
                            continue
                        if hedge_symbol is not None and t == hedge_symbol:
                            continue
                        try:
                            sh = float(shares_row[c])
                        except Exception:
                            sh = 0.0
                        if np.isfinite(sh) and sh > 0:
                            tickers_held.append(t)

                    if not tickers_held:
                        continue

                    # Prefer the *original selection ordering* (as requested) when we can map
                    # this window start to an Event; otherwise fall back to sorting by
                    # start-of-window position value.
                    tickers_sorted = None
                    eobj = event_exec_to_event.get(pd.Timestamp(d_start), None)
                    if eobj is not None:
                        tw = getattr(eobj, 'target_weights', {}) or {}
                        tickers_sorted = [
                            t for t, w in tw.items()
                            if (w is not None and np.isfinite(float(w)) and float(w) > 0.0)
                        ]
                        if hedge_symbol is not None:
                            tickers_sorted = [t for t in tickers_sorted if t != hedge_symbol]

                    if tickers_sorted is None:
                        # Fallback: sort tickers by position value at the start (descending)
                        pv = {}
                        for t in tickers_held:
                            p0 = price_df.get(t, pd.Series(index=price_df.index, dtype=float)).reindex([d_start]).iloc[0] if t in price_df.columns else np.nan
                            sh0 = primary_df.loc[d_start, f"{t} Shares"] if f"{t} Shares" in primary_df.columns else np.nan
                            try:
                                pv[t] = float(sh0) * float(p0) if np.isfinite(sh0) and np.isfinite(p0) else -np.inf
                            except Exception:
                                pv[t] = -np.inf
                        tickers_sorted = sorted(tickers_held, key=lambda t: pv.get(t, -np.inf), reverse=True)

                    tickers_sorted = tickers_sorted[: int(n_slots)]

                    for i, t in enumerate(tickers_sorted):
                        if t not in price_df.columns:
                            continue
                        p0 = price_df.at[d_start, t] if (d_start in price_df.index) else np.nan
                        p1 = price_df.at[d_end, t] if (d_end in price_df.index) else np.nan
                        if p0 is None or p1 is None or np.isnan(p0) or np.isnan(p1) or float(p0) <= 0:
                            continue
                        heat[i, j] = (float(p1) / float(p0) - 1.0) * 100.0
                # Optional clipping; also sets the symmetric color range when provided.
                max_abs = None
                if heatmap_clip is not None and np.isfinite(float(heatmap_clip)):
                    clip = float(heatmap_clip)
                    heat = np.clip(heat, -clip, clip)
                    max_abs = clip
                else:
                    try:
                        max_abs = float(np.nanmax(np.abs(heat)))
                    except Exception:
                        max_abs = float('nan')
                    if not np.isfinite(max_abs) or max_abs <= 0.0:
                        max_abs = 1.0

                # Build a red -> white -> green colormap with white exactly at 0.
                from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
                if isinstance(heatmap_cmap, str) and str(heatmap_cmap).lower() in {"red_white_green", "redwhitegreen", "rwg"}:
                    cmap = LinearSegmentedColormap.from_list(
                        "RedWhiteGreen",
                        ["#d73027", "#ffffff", "#1a9850"],
                        N=256,
                    )
                else:
                    cmap = plt.get_cmap(heatmap_cmap) if isinstance(heatmap_cmap, str) else heatmap_cmap

                norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

                # Plot heatmap
                hmask = np.ma.masked_invalid(heat)
                im = ax_hm.imshow(hmask, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
                ax_hm.set_title("Per-stock % change over holding window", fontsize=11)
                ax_hm.set_ylabel(f"Selected stocks\n(ordered by {order_var})")
                ax_hm.set_xlabel("Event date (start of window)")

                # One xtick per event, labeled with event date
                ax_hm.set_xticks(np.arange(len(window_starts)))
                ax_hm.set_xticklabels(
                    [pd.Timestamp(d).strftime("%Y-%m-%d") for d in window_starts],
                    rotation=90,
                    ha="center",
                    va="top",
                    fontsize=int(heatmap_xtick_fontsize),
                )

                ax_hm.set_yticks(np.arange(int(n_slots)))
                ax_hm.set_yticklabels([str(i + 1) for i in range(int(n_slots))])

                cbar = fig.colorbar(im, ax=ax_hm, orientation="vertical", fraction=0.02, pad=0.02)
                cbar.set_label("Return (%)")

    # Final layout
    # IMPORTANT: Do not call `tight_layout()` on a figure that was created with
    # `constrained_layout=True` *after* a colorbar has been created. Matplotlib
    # can attempt to switch the layout engine (constrained -> tight), which
    # raises:
    #   RuntimeError: Colorbar layout of new layout engine not compatible...
    #
    # We therefore only apply `tight_layout()` for the single-panel figure.
    if not want_heatmap:
        try:
            fig.tight_layout()
        except Exception:
            pass
    plt.show()

# =========================
# Build event sequence from list of dataframes
# =========================
# Make sure this is exported
# try:
#     __all__.append("build_events_from_sheets")
# except NameError:
#     __all__ = ["build_events_from_sheets"]

def build_events_from_sheets(
    sheets: List[pd.DataFrame],
    column: str,
    direction: str = "ascend",
    top_n: int = 10,
    *,
    group_col: str = "datetime",
    # weighting
    weight_mode: str = "equal",                 # "equal" | "proportional" | "softmax" | "inverse_rank"
    weight_column: Optional[str] = None,        # if None, uses `column`
    softmax_tau: float = 1.0,                   # temperature for softmax
    min_weight: float = 0.0,                    # clamp before renorm
    max_weight: Optional[float] = None,
    round_weights: Optional[int] = None,        # e.g. 6 -> round then renorm
    # selection controls
    include: Optional[Iterable[str]] = None,    # allowlist of tickers
    exclude: Optional[Iterable[str]] = None,    # blocklist of tickers
    dedupe: str = "first",                      # "first" | "none"
    tie_breaker: str = "stable",                # "stable" | "random"
    random_state: Optional[int] = None,
    # outputs
    return_detail: bool = False,
    return_summary: bool = True,                # <— NEW: include summary by default
    lookup_meta: bool = True,                   # <— NEW: fetch currency/exchange via yfinance
) -> (
    List["Event"]
    | Tuple[List["Event"], pd.DataFrame]
    | Tuple[List["Event"], Dict[str, object]]
    | Tuple[List["Event"], pd.DataFrame, Dict[str, object]]
):
    import warnings
    import numpy as np
    import pandas as pd
    import yfinance as yf

    if top_n <= 0:
        raise ValueError("top_n must be a positive integer")

    d = str(direction).strip().lower()
    if d in {"ascend", "asc", "ascending", "low_is_better"}:
        ascending = True
    elif d in {"descend", "desc", "descending", "high_is_better"}:
        ascending = False
    else:
        raise ValueError("direction must be 'ascend' or 'descend'")

    weight_mode = str(weight_mode).lower().strip()
    if weight_mode not in {"equal", "proportional", "softmax", "inverse_rank"}:
        raise ValueError("weight_mode must be one of: 'equal', 'proportional', 'softmax', 'inverse_rank'")

    tie_breaker = str(tie_breaker).lower().strip()
    if tie_breaker not in {"stable", "random"}:
        raise ValueError("tie_breaker must be 'stable' or 'random'")

    dedupe = str(dedupe).lower().strip()
    if dedupe not in {"first", "none"}:
        raise ValueError("dedupe must be 'first' or 'none'")

    allow = set(t.strip() for t in include) if include is not None else None
    block = set(t.strip() for t in exclude) if exclude is not None else set()

    required_cols = {"yf_Ticker", group_col, column}
    rng = np.random.default_rng(random_state)

    events: List["Event"] = []
    detail_rows = []

    def _finalize_weights(vals: pd.Series) -> pd.Series:
        v = vals.clip(lower=min_weight)
        if max_weight is not None:
            v = v.clip(upper=max_weight)
        s = v.sum()
        if s <= 0 or not np.isfinite(s):
            v = pd.Series(1.0, index=v.index)
            s = v.sum()
        v = v / s
        if round_weights is not None:
            v = v.round(round_weights)
            s2 = v.sum()
            if s2 > 0:
                v = v / s2
        return v

    def _compute_weights(df_sel: pd.DataFrame) -> pd.Series:
        base_col = weight_column if (weight_column is not None and weight_column in df_sel.columns) else column
        s = df_sel[base_col].astype(float)

        if weight_mode == "equal":
            w = pd.Series(1.0, index=df_sel.index)
        elif weight_mode == "proportional":
            if ascending:
                s2 = (s.max() - s).replace({0.0: 1e-12})
            else:
                s2 = (s - s.min()).replace({0.0: 1e-12})
            s2 = s2.clip(lower=1e-12)
            w = s2
        elif weight_mode == "softmax":
            logits = (-s / softmax_tau) if ascending else (s / softmax_tau)
            exps = np.exp(logits - np.max(logits))
            w = pd.Series(exps, index=df_sel.index)
        elif weight_mode == "inverse_rank":
            r = s.rank(method="first", ascending=ascending)
            w = 1.0 / r
        else:
            raise AssertionError("unreachable")

        return _finalize_weights(w)

    for idx, sheet in enumerate(sheets):
        if not required_cols.issubset(sheet.columns):
            missing = required_cols - set(sheet.columns)
            raise KeyError(f"Sheet {idx} is missing required columns: {sorted(missing)}")

        df = sheet.copy()

        # Clean tickers (robust against placeholders like 'NONE' or '$NONE')
        df["yf_Ticker"] = df["yf_Ticker"].apply(clean_yf_ticker)
        df = df.dropna(subset=["yf_Ticker"])
        df = df.loc[df["yf_Ticker"] != ""]

        # Include/exclude
        if allow is not None:
            df = df.loc[df["yf_Ticker"].isin(allow)]
        if block:
            df = df.loc[~df["yf_Ticker"].isin(block)]

        # Metric coercion
        for colname in {column, weight_column} - {None}:
            df[colname] = pd.to_numeric(df[colname], errors="coerce")
        df = df.dropna(subset=[column])

        # Parse event times
        dt_series = pd.to_datetime(df[group_col], errors="coerce")
        df = df.loc[~dt_series.isna()].copy()
        if df.empty:
            warnings.warn(f"Sheet {idx} has no valid rows after cleaning; skipping.")
            continue
        df["__dt__"] = pd.to_datetime(df[group_col]).dt.tz_localize(None).dt.normalize()

        # Optional random tie-breaker
        if tie_breaker == "random":
            df["__rand__"] = rng.random(len(df))
            sort_cols = [column, "__rand__"]
            sort_asc = [ascending, True]
            sort_kind = "quicksort"
        else:
            sort_cols = [column]
            sort_asc = [ascending]
            sort_kind = "mergesort"  # stable

        # Build events per datetime
        for dt_val, grp in df.groupby("__dt__", sort=True):
            grp_sorted = grp.sort_values(by=sort_cols, ascending=sort_asc, kind=sort_kind)

            # Ensure one row per ticker for selection when dedupe='first'
            if dedupe == "first":
                grp_for_pick = grp_sorted.drop_duplicates(subset="yf_Ticker", keep="first")
            else:
                grp_for_pick = grp_sorted  # may contain duplicates

            # Select top_n tickers (may include duplicates if dedupe='none')
            tickers_series = grp_for_pick["yf_Ticker"].astype(str).str.strip()
            sel_tickers = tickers_series.head(top_n).tolist()
            if len(sel_tickers) == 0:
                warnings.warn(f"Sheet {idx} @ {dt_val.date()}: no valid tickers for column '{column}'. Skipping.")
                continue

            # Rows for those selections (could be >top_n rows if duplicates allowed)
            sel_df = grp_sorted.loc[grp_sorted["yf_Ticker"].isin(sel_tickers)].copy()

            # Build per-ticker weights (aggregate duplicates safely)
            event_tickers = list(dict.fromkeys(sel_tickers))  # unique in order

            if weight_mode == "equal":
                weights_by_ticker = pd.Series(1.0, index=event_tickers)
                weights_by_ticker = _finalize_weights(weights_by_ticker)
            else:
                per_row_w = _compute_weights(sel_df)
                # label by ticker, then aggregate duplicates by MEAN (robust default)
                per_row_w.index = sel_df["yf_Ticker"].astype(str).tolist()
                weights_by_ticker = per_row_w.groupby(level=0).mean()
                weights_by_ticker = _finalize_weights(weights_by_ticker)

            # Create mapping (guard against any missing keys)
            weights_map = {
                t: float(weights_by_ticker.loc[t])
                for t in event_tickers
                if t in weights_by_ticker.index
            }

            if not weights_map:
                warnings.warn(f"Sheet {idx} @ {dt_val.date()}: empty weights after filtering; skipping.")
                continue

            ev = Event(dt_val, weights_map)

            # Attach selection metadata (useful for plotting/debugging).
            try:
                ev.selection_column = column
                ev.selection_direction = direction
                ev.selection_top_n = int(top_n)
                ev.selection_order = list(event_tickers)
            except Exception:
                pass

            events.append(ev)

            if return_detail:
                metric_by_ticker = sel_df.groupby("yf_Ticker")[column].mean()
                for t in event_tickers:
                    if t in weights_map:
                        detail_rows.append(
                            {
                                "datetime": dt_val,
                                "yf_Ticker": t,
                                "metric_column": column,
                                "metric_value": float(metric_by_ticker.get(t, np.nan)),
                                "weight_mode": weight_mode,
                                "final_weight": float(weights_map[t]),
                                "sheet_index": idx,
                            }
                        )

    # Ensure chronological order
    events.sort(key=lambda e: e.date)

    # Build detail DF if requested
    detail_df = None
    if return_detail:
        detail_df = pd.DataFrame(detail_rows)
        if not detail_df.empty:
            detail_df = detail_df.sort_values(
                ["datetime", "sheet_index", "final_weight"],
                ascending=[True, True, False]
            ).reset_index(drop=True)

    # Build summary if requested
    summary = None
    if return_summary:
        if events:
            start_date = min(e.date for e in events)
            end_date = max(e.date for e in events)
        else:
            start_date = end_date = None

        uniq_tickers = sorted({t for e in events for t in e.target_weights})
        # Final safety pass: remove any placeholders that slipped through.
        uniq_tickers = [t for t in uniq_tickers if clean_yf_ticker(t) is not None]
        # Defensive cleanup in case external code created Events with dirty tickers.
        uniq_tickers = [t for t in (clean_yf_ticker(x) for x in uniq_tickers) if t]

        ticker_meta: Dict[str, Dict[str, str]] = {}
        currencies = set()
        exchanges = set()

        if lookup_meta and uniq_tickers:
            for t in uniq_tickers:
                cur = exch = None
                try:
                    tk = yf.Ticker(t)
                    fi = getattr(tk, "fast_info", None)
                    if fi is not None:
                        try:
                            cur = fi.get("currency") if hasattr(fi, "get") else None
                            exch = fi.get("exchange") if hasattr(fi, "get") else None
                        except Exception:
                            pass
                    if cur is None or exch is None:
                        info = getattr(tk, "info", {})
                        if cur is None:
                            cur = info.get("currency")
                        if exch is None:
                            exch = info.get("exchange") or info.get("market")
                except Exception:
                    pass
                cur = cur or "UNKNOWN"
                exch = exch or "UNKNOWN"
                ticker_meta[t] = {"currency": cur, "exchange": exch}
                currencies.add(cur)
                exchanges.add(exch)

        summary = {
            "start_date": start_date,
            "end_date": end_date,
            "num_events": len(events),
            "num_unique_tickers": len(uniq_tickers),
            "unique_tickers": uniq_tickers,
            "num_unique_currencies": len(currencies) if lookup_meta else None,
            "unique_currencies": sorted(currencies) if lookup_meta else None,
            "num_unique_exchanges": len(exchanges) if lookup_meta else None,
            "unique_exchanges": sorted(exchanges) if lookup_meta else None,
            "ticker_meta": ticker_meta if lookup_meta else None,
        }

    if return_detail and return_summary:
        return events, detail_df, summary
    if return_detail and not return_summary:
        return events, detail_df
    if (not return_detail) and return_summary:
        return events, summary
    return events



    
# =========================
# Market data (USD-based)
# =========================
class MarketData:
    """
    Loads close prices for tickers and converts to USD when needed.
    - Detects listing currency for each ticker.
    - Downloads FX series and builds a USD conversion panel.
    - Provides get_prices(date, in_usd=True) to retrieve a dict of prices (USD by default).
    """

    def __init__(
        self,
        tickers,
        start_date,
        end_date,
        base_currency: str = "USD",
        *,
        use_adj_close: bool = False,
    ):
        raw = list(tickers) if isinstance(tickers, (list, tuple, set)) else [tickers]
        cleaned: List[str] = []
        dropped: List[str] = []
        seen = set()
        for t in raw:
            ct = clean_yf_ticker(t)
            if ct is None:
                if t is not None and str(t).strip():
                    dropped.append(str(t).strip())
                continue
            if ct in seen:
                continue
            seen.add(ct)
            cleaned.append(ct)

        self.tickers: List[str] = cleaned
        self.dropped_tickers: List[str] = dropped
        self.use_adj_close: bool = bool(use_adj_close)
        self.start_date: pd.Timestamp = pd.Timestamp(start_date)
        self.end_date: pd.Timestamp = pd.Timestamp(end_date)
        self.base_currency: str = base_currency

        self.data: pd.DataFrame = self._download_closes()              # local currency closes
        self.currency_map: Dict[str, str] = self._detect_currencies()  # ticker -> currency (e.g., USD, JPY, GBp)
        self.fx_rates: Dict[str, pd.Series] = self._build_fx_panel()   # currency -> USD per 1 unit of currency
        self.usd_data: pd.DataFrame = self._convert_all_to_usd()       # closes converted to USD

    # ---------- download helpers ----------
    def _download_closes(self) -> pd.DataFrame:
        # If no valid tickers, return empty panel.
        if not self.tickers:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))

        # Prefer batch download for speed; fall back to per-ticker downloads if
        # the batch download errors or returns empty.
        try:
            raw = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date + pd.Timedelta(days=1),
                progress=False,
            )
        except Exception:
            raw = pd.DataFrame()

        if raw.empty:
            # Fallback: download tickers one-by-one so a single bad ticker does not
            # poison the whole run.
            panels: Dict[str, pd.Series] = {}
            for t in self.tickers:
                try:
                    one = yf.download(
                        t,
                        start=self.start_date,
                        end=self.end_date + pd.Timedelta(days=1),
                        progress=False,
                    )
                except Exception:
                    continue
                if one is None or getattr(one, "empty", True):
                    continue
                field = "Adj Close" if self.use_adj_close and "Adj Close" in one.columns else "Close"
                if field not in one.columns:
                    continue
                s = one[field].copy()
                s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
                panels[t] = s
            if not panels:
                print("Warning: No data downloaded for market prices.")
                return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))
            df = pd.DataFrame(panels)
        else:
            df = raw

        # Choose price field
        price_field = "Adj Close" if bool(self.use_adj_close) else "Close"

        if isinstance(df.columns, pd.MultiIndex):
            # yfinance returns OHLCV in level 0 and tickers in level 1
            if price_field not in df.columns.levels[0]:
                # fallback to Close if Adj Close missing
                if price_field != "Close" and "Close" in df.columns.levels[0]:
                    price_field = "Close"
                else:
                    print(f"Warning: '{price_field}' not found in downloaded columns.")
                    return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))
            df = df[price_field]
        else:
            # single-ticker download returns a flat DataFrame
            if price_field not in df.columns:
                if price_field != "Close" and "Close" in df.columns:
                    price_field = "Close"
                else:
                    print(f"Warning: '{price_field}' column not found.")
                    return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))
            df = df[[price_field]]
            if len(self.tickers) == 1:
                df.columns = self.tickers

        df.index = df.index.tz_localize(None).normalize()

        # Ensure all requested columns exist (create empty if missing) and in requested order
        for t in self.tickers:
            if t not in df.columns:
                df[t] = np.nan
        return df[self.tickers]

    def _detect_currencies(self) -> Dict[str, str]:
        out = {}
        for t in self.tickers:
            cur = None
            try:
                fi = yf.Ticker(t).fast_info
                cur = fi.get("currency", None) if hasattr(fi, "get") else None
            except Exception:
                cur = None
            if not cur:
                try:
                    cur = yf.Ticker(t).info.get("currency")
                except Exception:
                    cur = None
            out[t] = cur or "USD"
        return out

    def _as_close_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Given a DataFrame from yf.download (one symbol), return a Close price Series
        aligned to self.data.index (timezone-naive, normalized).
        """
        if df is None or df.empty:
            return pd.Series(index=self.data.index, dtype=float)

        # Extract 'Close' regardless of shape
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" not in df.columns.levels[0]:
                return pd.Series(index=self.data.index, dtype=float)
            df = df["Close"]
        else:
            if "Close" not in df.columns:
                return pd.Series(index=self.data.index, dtype=float)
            df = df[["Close"]]

        # Squeeze to 1D
        if isinstance(df, pd.DataFrame):
            if df.shape[1] >= 1:
                s = df.iloc[:, 0]
            else:
                return pd.Series(index=self.data.index, dtype=float)
        else:
            s = df

        s.index = s.index.tz_localize(None).normalize()
        s = s.reindex(self.data.index).ffill()
        return s.astype(float)

    def _download_fx(self, ccy: str) -> pd.Series:
        """
        Return a USD conversion series for 'ccy' (USD per 1 unit of 'ccy').
        Tries CCYUSD=X first (multiply), then USDCCY=X (invert). JPY has a Yahoo shortcut.
        """
        if ccy == self.base_currency:
            return pd.Series(1.0, index=self.data.index, name=f"{ccy}->USD")

        # 1) Try CCYUSD=X (direct USD per CCY)
        pair1 = f"{ccy}{self.base_currency}=X"
        df1 = yf.download(pair1, start=self.start_date, end=self.end_date + pd.Timedelta(days=1), progress=False)
        s1 = self._as_close_series(df1)
        if not s1.empty and bool(np.nan_to_num(s1).any()):
            s1.name = f"{ccy}->USD"
            return s1

        # 2) Try USDCCY=X (invert)
        pair2 = f"{self.base_currency}{ccy}=X"
        df2 = yf.download(pair2, start=self.start_date, end=self.end_date + pd.Timedelta(days=1), progress=False)
        s2 = self._as_close_series(df2)
        if not s2.empty and bool(np.nan_to_num(s2).any()):
            inv = (1.0 / s2.replace(0, np.nan)).rename(f"{ccy}->USD")
            return inv

        # 3) Special: 'JPY=X' equals USD/JPY
        if ccy == "JPY":
            dfj = yf.download("JPY=X", start=self.start_date, end=self.end_date + pd.Timedelta(days=1), progress=False)
            sj = self._as_close_series(dfj)
            if not sj.empty and bool(np.nan_to_num(sj).any()):
                inv = (1.0 / sj.replace(0, np.nan)).rename(f"{ccy}->USD")
                return inv

        print(f"Warning: FX rate for {ccy} not found. Assuming 1 {ccy} = 1 USD.")
        return pd.Series(1.0, index=self.data.index, name=f"{ccy}->USD")

    def _build_fx_panel(self) -> Dict[str, pd.Series]:
        # Collect unique currencies; treat GBp specially (use GBP series)
        ccys = {c for c in self.currency_map.values() if c}
        panel: Dict[str, pd.Series] = {}
        for ccy in ccys:
            base = "GBP" if ccy == "GBp" else ccy
            panel[ccy] = self._download_fx(base)
        return panel

    def _convert_all_to_usd(self) -> pd.DataFrame:
        if self.data.empty:
            return self.data.copy()
        usd_df = pd.DataFrame(index=self.data.index)
        for t in self.tickers:
            series = self.data[t].astype(float)
            ccy = self.currency_map.get(t, "USD")
            fx = self.fx_rates.get(ccy, pd.Series(1.0, index=self.data.index))
            # LSE GBp: convert pence -> GBP first
            if ccy == "GBp":
                usd_df[t] = (series / 100.0) * fx
            else:
                usd_df[t] = series * fx
        return usd_df

    # ---------- public ----------
    def get_prices(self, date, in_usd: bool = True) -> Optional[Dict[str, float]]:
        """
        Returns dict of prices for the given date (closest on/before) with
        per-ticker forward-fill so holidays/market closures don't produce NaNs.
        """
        target = pd.Timestamp(date).normalize()
        df = self.usd_data if in_usd else self.data
        if df.empty:
            return None
    
        # slice up to the target date, then forward-fill each column independently
        df2 = df.loc[:target]
        if df2.empty:
            return None
    
        row = df2.ffill().iloc[-1]
        if pd.isna(row).all():
            return None
        return row.to_dict()


# =========================
# Benchmarks (buy & hold)
# =========================
class Benchmarks:
    """
    Buy-and-hold benchmarks: invest 100% on the first available trading day,
    hold to the end (fractional via price normalization). Returns USD values.
    """
    MAP = {"SPY": "SPY", "NDX": "^NDX", "GLD": "GLD"}

    def __init__(self, symbols: Tuple[str, ...], start_date, end_date, *, use_adj_close: bool = False):
        self.display_symbols: List[str] = list(symbols)
        self.use_adj_close: bool = bool(use_adj_close)
        yf_symbols = [self.MAP.get(s, s) for s in self.display_symbols]
        # sanitize
        yf_symbols = [s for s in (clean_yf_ticker(x) for x in yf_symbols) if s is not None]
        self.start_date: pd.Timestamp = pd.Timestamp(start_date)
        self.end_date: pd.Timestamp = pd.Timestamp(end_date)
        self.prices: pd.DataFrame = self._download_closes(yf_symbols)

        # Rename columns back to display names where possible
        rename_map = {yf_sym: disp for yf_sym, disp in zip(self.prices.columns, self.prices.columns)}
        # If Yahoo returned the exact yf_symbols, remap to display order
        col_map = {}
        for yf_sym, disp in zip(yf_symbols, self.display_symbols):
            if yf_sym in self.prices.columns:
                col_map[yf_sym] = disp
        self.prices = self.prices.rename(columns=col_map)

        # Ensure all requested symbols exist as columns (even if NaN)
        for name in self.display_symbols:
            if name not in self.prices.columns:
                self.prices[name] = np.nan
        self.prices = self.prices[self.display_symbols]

    def _download_closes(self, tickers: List[str]) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))

        try:
            raw = yf.download(
                tickers,
                start=self.start_date,
                end=self.end_date + pd.Timedelta(days=1),
                progress=False,
            )
        except Exception:
            raw = pd.DataFrame()

        if raw.empty:
            # fallback to per-ticker downloads
            panels: Dict[str, pd.Series] = {}
            for t in tickers:
                try:
                    one = yf.download(
                        t,
                        start=self.start_date,
                        end=self.end_date + pd.Timedelta(days=1),
                        progress=False,
                    )
                except Exception:
                    continue
                if one is None or getattr(one, "empty", True):
                    continue
                field = "Adj Close" if self.use_adj_close and "Adj Close" in one.columns else "Close"
                if field not in one.columns:
                    continue
                s = one[field].copy()
                s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
                panels[t] = s
            if not panels:
                print("Warning: No benchmark data downloaded.")
                return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))
            df = pd.DataFrame(panels)
        else:
            df = raw

        price_field = "Adj Close" if bool(self.use_adj_close) else "Close"
        if isinstance(df.columns, pd.MultiIndex):
            if price_field not in df.columns.levels[0]:
                if price_field != "Close" and "Close" in df.columns.levels[0]:
                    price_field = "Close"
                else:
                    print(f"Warning: '{price_field}' not found for benchmarks.")
                    return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))
            df = df[price_field]
        else:
            if price_field not in df.columns:
                if price_field != "Close" and "Close" in df.columns:
                    price_field = "Close"
                else:
                    print(f"Warning: '{price_field}' missing for benchmarks.")
                    return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))
            df = df[[price_field]]
            if len(tickers) == 1:
                df.columns = tickers

        df.index = df.index.tz_localize(None).normalize()
        # Keep only requested columns in the same order; create missing
        for t in tickers:
            if t not in df.columns:
                df[t] = np.nan
        return df[tickers]

    def buy_and_hold_values(self, initial_capital: float, index_ref: pd.DatetimeIndex) -> pd.DataFrame:
        """Return DataFrame with benchmark portfolio values aligned to index_ref."""
        if self.prices.empty:
            return pd.DataFrame(index=index_ref, columns=self.display_symbols, dtype=float)

        vals = {}
        for col in self.prices.columns:
            s = self.prices[col].dropna()
            if s.empty:
                vals[col] = pd.Series(index=index_ref, dtype=float)
                continue
            base = s.iloc[0]
            value_series = (s / base) * float(initial_capital)
            aligned = value_series.reindex(index_ref).ffill()
            vals[col] = aligned

        out = pd.DataFrame(vals, index=index_ref)
        # Ensure column order matches requested display symbols
        for name in self.display_symbols:
            if name not in out.columns:
                out[name] = np.nan
        return out[self.display_symbols]


# =========================
# Event & Portfolio
# =========================
class Event:
    """Rebalance instruction at a specific date with target weights."""
    def __init__(self, date, target_weights: Dict[str, float]):
        self.date: pd.Timestamp = pd.Timestamp(date).normalize()
        self.target_weights: Dict[str, float] = dict(target_weights)


class Portfolio:
    """Simple long-only portfolio with integer shares and optional cash policy."""
    def __init__(self, capital: float):
        self.capital: float = float(capital)
        self.positions: Dict[str, int] = {}
        self.cash: float = float(capital)

    def value(self, prices: Dict[str, float]) -> float:
        invested_value = 0.0
        for t, sh in self.positions.items():
            p = prices.get(t, np.nan)
            if not np.isnan(p):
                invested_value += sh * p
        return float(invested_value + self.cash)

    def set_position_shares(self, ticker: str, target_shares: int, price: float) -> None:
        """Set a single position to a target share count (can be negative for shorts).

        Updates cash consistently at the provided execution price.

        Notes
        -----
        - Buying shares reduces cash; selling shares increases cash.
        - Shorting shares (target_shares < 0) increases cash; covering reduces cash.
        - This is intended for overlays (e.g., hedges) applied on top of a rebalance.
        """
        if price is None or np.isnan(price) or float(price) <= 0.0:
            raise ValueError("price must be a positive number")

        t = str(ticker)
        target = int(target_shares)
        current = int(self.positions.get(t, 0))
        delta = target - current

        # Buying (delta>0) spends cash; selling/shorting (delta<0) raises cash.
        self.cash -= float(delta) * float(price)

        if target == 0:
            self.positions.pop(t, None)
        else:
            self.positions[t] = target


    def rebalance(
            self,
            prices: Dict[str, float],
            target_weights: Dict[str, float],
            *,
            rebalance_mode: str = "rebuild",     # 'rebuild' | 'adjust' (currently both map to the same mechanics)
            cash_policy: Optional[str] = None,   # None | "fixed" | "proportion"
            cash_fixed_amount: float = 0.0,      # used when cash_policy == "fixed"
            cash_pct: float = 0.0,               # used when cash_policy == "proportion", e.g., 0.20 for 20%
            max_leverage: float = 1.0            # NEW: allow gross long up to max_leverage * equity (equity = positions + cash)
        ) -> None:
            """Rebalance to target_weights while enforcing the cash policy at this event.

            Cash policy (interpreted on *equity*):
            - fixed:      keep min(fixed_amount, total_equity) in cash
            - proportion: keep total_equity * cash_pct in cash
            - None:       keep 0 in cash

            Margin / leverage:
            - If max_leverage > 1.0, the rebalance is allowed to invest more than equity.
              This is modeled as negative cash (a margin loan). No maintenance margin is enforced.
            """

            mode = str(rebalance_mode).strip().lower()
            if mode not in {"rebuild", "adjust"}:
                raise ValueError("rebalance_mode must be 'rebuild' or 'adjust'")

            total_equity = float(self.value(prices))

            # Validate leverage
            try:
                lev = float(max_leverage)
            except Exception as e:
                raise TypeError("max_leverage must be a float >= 1.0") from e
            if not np.isfinite(lev) or lev < 1.0:
                raise ValueError("max_leverage must be a finite float >= 1.0")

            # determine target cash (clamped to non-negative)
            cp = None if (cash_policy is None or str(cash_policy).lower() == "none") else str(cash_policy).lower()
            if cp is None:
                target_cash = 0.0
            elif cp == "fixed":
                target_cash = float(max(0.0, min(total_equity, cash_fixed_amount)))
            elif cp == "proportion":
                pct = float(max(0.0, min(1.0, cash_pct)))
                target_cash = float(max(0.0, total_equity * pct))
            else:
                raise ValueError("cash_policy must be None, 'fixed', or 'proportion'")

            # Determine investable budget.
            # - lev==1: this reduces to max(total_equity - target_cash, 0)
            # - lev>1: allows borrowing, i.e., investable can exceed equity.
            investable = float(max(0.0, total_equity * lev - target_cash))

            # normalize weights
            w_items = [(t, float(w)) for t, w in target_weights.items() if not np.isnan(w) and w > 0.0]
            w_sum = sum(w for _, w in w_items)
            if w_sum <= 0.0 or investable <= 0.0 or total_equity <= 0.0:
                self.positions.clear()
                self.cash = float(total_equity)
                return
            norm_weights = {t: w / w_sum for t, w in w_items}

            # reset positions and buy integer shares
            self.positions.clear()
            invested = 0.0
            for ticker, weight in norm_weights.items():
                price = prices.get(ticker, np.nan)
                if np.isnan(price) or price <= 0.0:
                    continue
                allocation = investable * weight
                shares = int(allocation // price)
                if shares > 0:
                    self.positions[ticker] = shares
                    invested += shares * price

            # Cash can be negative when lev > 1.0 (borrowed funds).
            self.cash = float(total_equity - invested)


# =========================
# Simulation
# =========================
class Simulation:
    """
    Supports:
    - Event shuffling (shuffle_events) with optional time window constraint
    - Weight methods: None (as provided) | 'random' (Dirichlet) | 'uniform'
    - Cash policies: None | 'fixed' | 'proportion'
    - Benchmarks: SPY, NDX (^NDX), GLD buy-and-hold (USD)
    - Optional dollar-neutral overlay: short hedge_symbol dollar-for-dollar at each event

    run() returns:
        - If cash_policy is None and no ensemble:
            {"strategy": DataFrame, "benchmarks": DataFrame}
        - If cash_policy is not None and no ensemble:
            {"with_cash": DataFrame, "no_cash": DataFrame, "benchmarks": DataFrame}
        - If ensemble (shuffle_events or weight_method set):
            - cash_policy None:
                {"strategy": EnsembleDF, "benchmarks": DataFrame}
            - cash_policy not None:
                {"with_cash": EnsembleDF, "no_cash": EnsembleDF, "benchmarks": DataFrame}
    """

    def __init__(
        self,
        start_date,
        end_date,
        initial_capital: float,
        events: List["Event"],
        *,
        shuffle_events: bool = False,
        num_shuffle: int = 100,
        weight_method: Optional[str] = None,     # None | 'random' | 'uniform'
        cash_policy: Optional[str] = None,       # None | 'fixed' | 'proportion'
        cash_fixed_amount: float = 0.0,
        cash_pct: float = 0.0,
        # Margin / leverage (optional)
        max_leverage: float = 1.0,              # 1.0 => no margin; 1.5 => 50% margin; 2.0 => 2x gross long
        margin_rate_apr: float = 0.0,           # APR charged on borrowed cash (negative cash), e.g., 0.08 for 8%
        random_state: Optional[int] = None,
        benchmark_symbols: Tuple[str, ...] = ("SPY", "NDX", "GLD"),
        shuffle_window_days: Optional[float] = None,  # NEW: None/np.inf means unlimited
        dollar_neutral: bool = False,                 # NEW: optional dollar-neutral overlay via shorting hedge_symbol
        hedge_symbol: str = "SPY",                    # NEW: instrument to short for the dollar-neutral overlay
        hedge_notional_base: str = "total",      # NEW: 'total' (include cash) or 'gross_long' (exclude cash)
        hedge_rounding: str = "floor",                # NEW: 'floor' (conservative) or 'round'
        rebalance_mode: str = "adjust",               # NEW: 'adjust' | 'rebuild'
        use_adj_close: bool = False,                   # NEW: use 'Adj Close' when available
        capture_trades: bool = False,                  # NEW: include a trade log in run() output for non-ensemble runs
    ):
        self.start_date = pd.Timestamp(start_date).normalize()
        self.end_date = pd.Timestamp(end_date).normalize()
        self.initial_capital = float(initial_capital)
        # Expect Event(date: pd.Timestamp-like, target_weights: Dict[str,float])
        self.events: List[Event] = sorted(events, key=lambda e: pd.Timestamp(getattr(e, "date", e)).normalize())

        # Optional: capture the selection-ordering variable name (if events were constructed
        # via build_events_from_sheets or another pipeline that attaches metadata).
        self.selection_column = None
        self.selection_direction = None
        try:
            if self.events:
                cols = {getattr(e, 'selection_column', None) for e in self.events}
                cols.discard(None)
                if len(cols) == 1:
                    self.selection_column = next(iter(cols))
                dirs = {getattr(e, 'selection_direction', None) for e in self.events}
                dirs.discard(None)
                if len(dirs) == 1:
                    self.selection_direction = next(iter(dirs))
        except Exception:
            pass
        self.shuffle_events = bool(shuffle_events)
        self.num_shuffle = int(num_shuffle)
        wm = None if weight_method is None else str(weight_method).lower()
        self.weight_method = None if wm in (None, "none", "") else wm
        cp = None if cash_policy is None else str(cash_policy).lower()
        self.cash_policy = None if cp in (None, "none", "") else cp
        self.cash_fixed_amount = float(cash_fixed_amount)
        self.cash_pct = float(cash_pct)

        # Execution / data options
        rm = str(rebalance_mode).strip().lower() if rebalance_mode is not None else "adjust"
        if rm not in {"adjust", "rebuild"}:
            raise ValueError("rebalance_mode must be 'adjust' or 'rebuild'")
        self.rebalance_mode = rm
        self.use_adj_close = bool(use_adj_close)
        self.capture_trades = bool(capture_trades)
        self._last_trades: Optional[pd.DataFrame] = None

        # Margin / leverage configuration
        self.max_leverage = float(max_leverage)
        if not np.isfinite(self.max_leverage) or self.max_leverage < 1.0:
            raise ValueError("max_leverage must be a finite float >= 1.0")
        self.margin_rate_apr = float(margin_rate_apr)
        if not np.isfinite(self.margin_rate_apr) or self.margin_rate_apr < 0.0:
            raise ValueError("margin_rate_apr must be a finite float >= 0.0")
        self._margin_periods_per_year = 252

        self.random_state = random_state
        # NEW
        if shuffle_window_days is None or (isinstance(shuffle_window_days, (int, float)) and np.isinf(shuffle_window_days)):
            self.shuffle_window_days = None
        else:
            self.shuffle_window_days = float(shuffle_window_days)

        # Dollar-neutral overlay configuration (optional)
        self.dollar_neutral = bool(dollar_neutral)
        # sanitize hedge symbol to avoid poisoning yfinance downloads
        hs = clean_yf_ticker(hedge_symbol)
        self.hedge_symbol = hs if hs is not None else str(hedge_symbol).strip()
        self.hedge_notional_base = str(hedge_notional_base).strip().lower()
        self.hedge_rounding = str(hedge_rounding).strip().lower()

        if self.dollar_neutral:
            if not self.hedge_symbol or clean_yf_ticker(self.hedge_symbol) is None:
                raise ValueError("hedge_symbol must be a non-empty ticker when dollar_neutral=True")
            if self.hedge_notional_base not in {"gross_long", "total"}:
                raise ValueError("hedge_notional_base must be 'gross_long' or 'total'")
            if self.hedge_rounding not in {"floor", "round"}:
                raise ValueError("hedge_rounding must be 'floor' or 'round'")

            # Prevent conflicts: strategy events should not directly target the hedge symbol.
            for e in self.events:
                tw = getattr(e, "target_weights", {}) or {}
                if self.hedge_symbol in tw:
                    raise ValueError(
                        f"hedge_symbol '{self.hedge_symbol}' appears in an Event.target_weights; "
                        "this conflicts with dollar_neutral overlay."
                    )

        # Build market data universe from events (plus hedge instrument, if enabled)
        self.strategy_tickers = sorted({t for e in self.events for t in (getattr(e, "target_weights", {}) or {}).keys()})
        self.strategy_tickers = [t for t in self.strategy_tickers if clean_yf_ticker(t) is not None]
        tickers = list(self.strategy_tickers)
        if self.dollar_neutral and self.hedge_symbol not in tickers:
            tickers.append(self.hedge_symbol)
        # final clean + stable de-dup
        _seen = set()
        clean_universe: List[str] = []
        for t in tickers:
            ct = clean_yf_ticker(t)
            if ct is None or ct in _seen:
                continue
            _seen.add(ct)
            clean_universe.append(ct)
        tickers = sorted(clean_universe)

        self.market_data = MarketData(tickers, self.start_date, self.end_date, use_adj_close=self.use_adj_close)
        self.benchmarks = Benchmarks(benchmark_symbols, self.start_date, self.end_date, use_adj_close=self.use_adj_close)

    # ---------- internal helpers ----------

    def _margin_rate_per_period(self) -> float:
        """Convert margin_rate_apr (APR) to a per-trading-day rate.

        Uses geometric conversion consistent with other annual-to-period conversions:
            r_day = (1 + APR) ** (1 / 252) - 1

        Returns 0.0 when margin_rate_apr is 0.
        """
        apr = float(getattr(self, "margin_rate_apr", 0.0))
        if not np.isfinite(apr) or apr <= 0.0:
            return 0.0
        if apr <= -1.0:
            raise ValueError("margin_rate_apr must be > -1.0")
        n = int(getattr(self, "_margin_periods_per_year", 252))
        n = 252 if n <= 0 else n
        return (1.0 + apr) ** (1.0 / n) - 1.0

    def _run_with_events(self, events_for_run: List["Event"], cash_policy, cash_fixed_amount, cash_pct) -> pd.DataFrame:
        portfolio = Portfolio(self.initial_capital)
        results = []
        trade_log: List[Dict[str, object]] = []
        event_queue = list(sorted(events_for_run, key=lambda e: e.date))

        # Margin interest is applied on borrowed cash (negative cash) once per trading day.
        margin_rate_per_day = self._margin_rate_per_period()
        cumulative_interest_paid = 0.0

        for current_date in self.market_data.usd_data.index:
            prices_today = self.market_data.get_prices(current_date, in_usd=True)
            if prices_today is None:
                continue

            # Apply margin interest for this period (on borrowed cash from prior close)
            interest_paid = 0.0
            if margin_rate_per_day > 0.0 and portfolio.cash < 0.0:
                borrow = -float(portfolio.cash)
                interest_paid = borrow * float(margin_rate_per_day)
                portfolio.cash -= float(interest_paid)
                cumulative_interest_paid += float(interest_paid)

            # Trigger all events scheduled up to and including current_date
            event_fired_today = False
            while event_queue and current_date >= event_queue[0].date:
                event = event_queue.pop(0)
                event_fired_today = True
                # Trade capture: snapshot positions/cash before rebalance
                before_pos = dict(portfolio.positions)
                before_cash = float(portfolio.cash)
                # Possibly mutate weights per weight_method
                weights = self._generate_weights(event.target_weights)
                portfolio.rebalance(
                    prices_today,
                    weights,
                    rebalance_mode=self.rebalance_mode,
                    cash_policy=cash_policy,
                    cash_fixed_amount=cash_fixed_amount,
                    cash_pct=cash_pct,
                    max_leverage=self.max_leverage,
                )

                if self.capture_trades:
                    after_pos = dict(portfolio.positions)
                    after_cash = float(portfolio.cash)
                    tick_set = set(before_pos) | set(after_pos)
                    for t in sorted(tick_set):
                        d = int(after_pos.get(t, 0)) - int(before_pos.get(t, 0))
                        if d == 0:
                            continue
                        px = prices_today.get(t, np.nan)
                        trade_log.append(
                            {
                                "Date": current_date,
                                "Ticker": t,
                                "Action": "BUY" if d > 0 else "SELL",
                                "Shares": abs(int(d)),
                                "Signed Shares": int(d),
                                "Price": float(px) if (px is not None and np.isfinite(px)) else np.nan,
                                "Notional": float(d) * float(px) if (px is not None and np.isfinite(px)) else np.nan,
                                "Cash Delta": after_cash - before_cash,
                                "Source": "rebalance",
                            }
                        )

            # Optional overlay: short hedge_symbol dollar-for-dollar after event-driven rebalance(s)
            if event_fired_today and getattr(self, "dollar_neutral", False):
                if self.capture_trades:
                    before_pos = dict(portfolio.positions)
                    before_cash = float(portfolio.cash)
                self._apply_dollar_neutral_hedge(portfolio, prices_today)
                if self.capture_trades:
                    after_pos = dict(portfolio.positions)
                    after_cash = float(portfolio.cash)
                    tick_set = set(before_pos) | set(after_pos)
                    for t in sorted(tick_set):
                        d = int(after_pos.get(t, 0)) - int(before_pos.get(t, 0))
                        if d == 0:
                            continue
                        px = prices_today.get(t, np.nan)
                        trade_log.append(
                            {
                                "Date": current_date,
                                "Ticker": t,
                                "Action": "BUY" if d > 0 else "SELL",
                                "Shares": abs(int(d)),
                                "Signed Shares": int(d),
                                "Price": float(px) if (px is not None and np.isfinite(px)) else np.nan,
                                "Notional": float(d) * float(px) if (px is not None and np.isfinite(px)) else np.nan,
                                "Cash Delta": after_cash - before_cash,
                                "Source": "hedge",
                            }
                        )

            portfolio_val = portfolio.value(prices_today)
            row = {
                "Date": current_date,
                "Portfolio Value": portfolio_val,
                "Cash": portfolio.cash,
                **{f"{t} Shares": portfolio.positions.get(t, 0) for t in self.market_data.tickers},
            }

            # Margin diagnostics (only meaningful if leverage or margin rate is enabled)
            if self.max_leverage > 1.0 or self.margin_rate_apr > 0.0:
                borrowed = float(max(-portfolio.cash, 0.0))
                gross_exposure = 0.0
                net_exposure = 0.0
                for t, sh in portfolio.positions.items():
                    p = prices_today.get(t, np.nan)
                    if np.isnan(p) or p <= 0.0:
                        continue
                    pos_val = float(sh) * float(p)
                    net_exposure += pos_val
                    gross_exposure += abs(pos_val)
                eff_lev = (gross_exposure / float(portfolio_val)) if (portfolio_val is not None and np.isfinite(portfolio_val) and portfolio_val > 0.0) else np.nan
                row["Borrowed"] = borrowed
                row["Gross Exposure"] = float(gross_exposure)
                row["Net Exposure"] = float(net_exposure)
                row["Effective Leverage"] = eff_lev
                row["Interest Paid"] = float(interest_paid)
                row["Cumulative Interest Paid"] = float(cumulative_interest_paid)

            # Diagnostics for dollar-neutral overlay (if enabled)
            if getattr(self, "dollar_neutral", False):
                hedge = getattr(self, "hedge_symbol", "SPY")

                gross_long_value = 0.0
                for t, sh in portfolio.positions.items():
                    if t == hedge:
                        continue
                    if sh <= 0:
                        continue
                    p = prices_today.get(t, np.nan)
                    if not np.isnan(p) and p > 0:
                        gross_long_value += float(sh) * float(p)

                hedge_price = prices_today.get(hedge, np.nan)
                hedge_shares = int(portfolio.positions.get(hedge, 0))

                hedge_notional = np.nan
                net_market_value = np.nan
                if not np.isnan(hedge_price) and hedge_price > 0:
                    hedge_notional = -float(hedge_shares) * float(hedge_price)   # positive when short
                    net_market_value = float(gross_long_value + hedge_shares * float(hedge_price))

                row["Gross Long Value"] = float(gross_long_value)
                row[f"{hedge} Hedge Notional"] = hedge_notional
                row["Net Market Value"] = net_market_value

            results.append(row)

        # Save last trades for UI use (single-pass runs)
        if self.capture_trades:
            if trade_log:
                self._last_trades = pd.DataFrame(trade_log).sort_values(["Date", "Ticker"])
            else:
                self._last_trades = pd.DataFrame(columns=[
                    "Date", "Ticker", "Action", "Shares", "Signed Shares", "Price", "Notional", "Cash Delta", "Source"
                ])

        if not results:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))
        return pd.DataFrame(results).set_index("Date")

    def _apply_dollar_neutral_hedge(self, portfolio: "Portfolio", prices_today: Dict[str, float]) -> None:
        """Apply a dollar-for-dollar short hedge in self.hedge_symbol.

        Triggered at event dates (after the portfolio has been rebalanced).

        The target hedge notional is:
            - 'total' (default): total portfolio value (includes cash)
            - 'gross_long': sum of long positions (excludes cash and excludes hedge_symbol)
            - 'total': total portfolio value (includes cash)

        Shares are sized using either floor (conservative) or round.
        """
        if not getattr(self, "dollar_neutral", False):
            return

        hedge = getattr(self, "hedge_symbol", "SPY")
        hedge_price = prices_today.get(hedge, np.nan)
        if hedge_price is None or np.isnan(hedge_price) or float(hedge_price) <= 0.0:
            return

        # Compute gross long value (excluding hedge symbol)
        gross_long_value = 0.0
        for t, sh in portfolio.positions.items():
            if t == hedge:
                continue
            if sh <= 0:
                continue
            p = prices_today.get(t, np.nan)
            if not np.isnan(p) and p > 0:
                gross_long_value += float(sh) * float(p)

        base = gross_long_value
        if getattr(self, "hedge_notional_base", "total") == "total":
            base = float(portfolio.value(prices_today))

        base = float(max(0.0, base))
        if base <= 0.0:
            target_shares = 0
        else:
            shares_float = base / float(hedge_price)
            if getattr(self, "hedge_rounding", "floor") == "round":
                shares = int(np.round(shares_float))
            else:
                shares = int(np.floor(shares_float))
            target_shares = -int(max(0, shares))

        portfolio.set_position_shares(hedge, target_shares, float(hedge_price))

    def _generate_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply weight_method to base weights, returning a new weight dict that sums to 1 (if non-empty)."""
        if self.weight_method is None:
            return dict(base_weights)

        keys = list(base_weights.keys())
        n = len(keys)
        if n == 0:
            return {}

        if self.weight_method == "uniform":
            w = np.ones(n, dtype=float) / n
        elif self.weight_method == "random":
            rng = np.random.default_rng(self.random_state)
            # Dirichlet with alpha=1 (flat); skew slightly by base_weights magnitudes if provided
            base = np.array([max(float(base_weights[k]), 0.0) for k in keys], dtype=float)
            alpha = np.where(base > 0, 1.0 + base / (base.sum() if base.sum() > 0 else 1.0), 1.0)
            w = rng.dirichlet(alpha)
        else:
            warnings.warn(f"Unknown weight_method={self.weight_method!r}; using base weights.")
            w = np.array([base_weights[k] for k in keys], dtype=float)
            s = w.sum()
            if s > 0:
                w = w / s

        return {k: float(wi) for k, wi in zip(keys, w)}

    def _run_ensemble(self, *, cash_policy, cash_fixed_amount, cash_pct) -> pd.DataFrame:
        """Run multiple randomized variants and return an ensemble summary DataFrame."""
        rng = np.random.default_rng(self.random_state)
        members = []
        for _ in range(self.num_shuffle):
            evs = self.events
            if self.shuffle_events:
                evs = self._make_shuffled_events(rng)
            # NOTE: _generate_weights is applied inside _run_with_events at each event
            df = self._run_with_events(evs, cash_policy=cash_policy, cash_fixed_amount=cash_fixed_amount, cash_pct=cash_pct)
            members.append(df["Portfolio Value"].rename(f"member_{len(members)}"))

        if not members:
            return pd.DataFrame()

        # Align on Date and aggregate
        M = pd.concat(members, axis=1)
        out = pd.DataFrame({
            "Ensemble Mean": M.mean(axis=1),
            "Ensemble Median": M.median(axis=1),
            "Ensemble P10": M.quantile(0.10, axis=1),
            "Ensemble P90": M.quantile(0.90, axis=1),
        })
        out.index.name = "Date"
        return out

    # --------- shuffling ---------
    def _make_shuffled_events(self, rng: np.random.Generator) -> List["Event"]:
        """Shuffle event target_weights across event dates. Optionally constrained by a time window."""
        dates = [pd.Timestamp(e.date).normalize() for e in self.events]
        payloads = [e.target_weights for e in self.events]

        if self.shuffle_window_days is None:
            perm = rng.permutation(len(payloads))
        else:
            perm = self._windowed_permutation(dates, self.shuffle_window_days, rng)

        # Reconstruct Events with original date and permuted payload
        new_events = [Event(dates[i], payloads[int(perm[i])]) for i in range(len(dates))]
        return sorted(new_events, key=lambda e: e.date)

    def _windowed_permutation(
        self,
        dates: List[pd.Timestamp],
        window_days: float,
        rng: Optional[np.random.Generator] = None
    ) -> List[int]:
        """
        Return a random permutation 'perm' under the constraint that each index i
        may map only to indices j whose dates are within ±window_days of dates[i].
        If no valid alternative is available, the event may map to itself (j=i).

        Uses a randomized Kuhn matching to create a one-to-one assignment.
        """
        if rng is None:
            rng = np.random.default_rng()

        n = len(dates)
        ts = [pd.Timestamp(d).normalize() for d in dates]
        window = pd.Timedelta(days=float(window_days))

        # Build adjacency: i -> candidate js (within window, including itself)
        edges: List[List[int]] = []
        for i in range(n):
            nbrs = [j for j in range(n) if abs(ts[j] - ts[i]) <= window]
            # randomize neighbor order
            if len(nbrs) > 1:
                nbrs = list(rng.permutation(nbrs))
            edges.append(nbrs)

        match_r = [-1] * n  # right side matches

        # Harder nodes first: fewest options
        order_left = sorted(range(n), key=lambda i: (len(edges[i]), rng.random()))

        # DFS for augmenting paths
        import sys
        sys.setrecursionlimit(max(10000, 10 * n))

        def dfs(i: int, seen: set) -> bool:
            for j in edges[i]:
                if j in seen:
                    continue
                seen.add(j)
                if match_r[j] == -1 or dfs(match_r[j], seen):
                    match_r[j] = i
                    return True
            return False

        for i in order_left:
            dfs(i, set())  # it's fine if it fails; we'll patch below

        # Patch unmatched rights or lefts if any (rare)
        # Convert to left->right mapping; fallback to identity where needed
        perm = [None] * n
        for j, i in enumerate(match_r):
            if i is not None and i != -1:
                perm[i] = j
        for i in range(n):
            if perm[i] is None:
                # choose any neighbor (prefer itself if allowed)
                nbrs = edges[i]
                if i in nbrs:
                    perm[i] = i
                else:
                    perm[i] = nbrs[0] if nbrs else i
        return [int(x) for x in perm]

    # ---------- public ----------
    def run(self) -> Dict[str, pd.DataFrame]:
        """Run the simulation and return strategy and benchmarks."""
        bench_df = self.benchmarks.buy_and_hold_values(self.initial_capital, self.market_data.usd_data.index)

        is_ensemble = self.shuffle_events or (self.weight_method is not None)
        if is_ensemble:
            if self.cash_policy is None:
                strat = self._run_ensemble(cash_policy=None, cash_fixed_amount=0.0, cash_pct=0.0)
                return {"strategy": strat, "benchmarks": bench_df}
            else:
                with_c = self._run_ensemble(
                    cash_policy=self.cash_policy,
                    cash_fixed_amount=self.cash_fixed_amount,
                    cash_pct=self.cash_pct,
                )
                no_c = self._run_ensemble(cash_policy=None, cash_fixed_amount=0.0, cash_pct=0.0)
                return {"with_cash": with_c, "no_cash": no_c, "benchmarks": bench_df}

        # Single pass (no ensemble)
        if self.cash_policy is None:
            df = self._run_with_events(self.events, cash_policy=None, cash_fixed_amount=0.0, cash_pct=0.0)
            if self.capture_trades:
                return {"strategy": df, "benchmarks": bench_df, "trades": (self._last_trades if self._last_trades is not None else pd.DataFrame())}
            return {"strategy": df, "benchmarks": bench_df}
        else:
            df_with = self._run_with_events(
                self.events,
                cash_policy=self.cash_policy,
                cash_fixed_amount=self.cash_fixed_amount,
                cash_pct=self.cash_pct,
            )
            trades_df = None
            if self.capture_trades:
                trades_df = (self._last_trades.copy() if self._last_trades is not None else pd.DataFrame())
            df_no = self._run_with_events(self.events, cash_policy=None, cash_fixed_amount=0.0, cash_pct=0.0)
            if self.capture_trades:
                return {"with_cash": df_with, "no_cash": df_no, "benchmarks": bench_df, "trades": trades_df}
            return {"with_cash": df_with, "no_cash": df_no, "benchmarks": bench_df}

    def plot(self, **kwargs):
        """Convenience wrapper around plot_simulation(self, **kwargs)."""
        return plot_simulation(self, **kwargs)


# =========================
# Analysis
# =========================
class Analysis:
    """
    Compute industry-standard performance and risk statistics for a given simulation result
    (from Simulation.run) or for a portfolio value series/DataFrame.

    Parameters
    ----------
    sim_result_or_values : dict | pd.Series | pd.DataFrame
        - dict: result returned by Simulation.run(), e.g.
            {"strategy": df, "benchmarks": bench_df}
            {"with_cash": df1, "no_cash": df2, "benchmarks": bench_df}
            or their ensemble variants (DataFrames with "Ensemble Mean", ...).
        - pd.Series: portfolio value indexed by Date
        - pd.DataFrame: either a single-path frame with a "Portfolio Value" column,
            or an ensemble frame with columns including "Ensemble Mean".
    benchmarks : dict[str, pd.Series] | pd.DataFrame | None
        Optional benchmarks. If sim_result_or_values is a dict containing "benchmarks",
        that frame is used automatically.
    rf : float | pd.Series, default 0.0
        Risk-free rate. If float, interpreted as annual rate (APR). If Series, should be
        per-period risk-free (already aligned to the index) or annual rate; both are accepted
        (we detect and convert if needed when possible).
    periods_per_year : int | None, default 252
        Used for annualization of moment-based stats (Sharpe, vol, Sortino, TE, etc.).
        If None, will be inferred from the index spacing (approximate). CAGR does not use this.
    use_log_returns : bool, default False
        If True, use log returns; otherwise use simple returns.
    primary_benchmark : str | None
        If provided, compute benchmark-relative metrics vs this column from the benchmarks frame.
        If None and a benchmarks frame exists, the first column is used.
    """

    def __init__(
        self,
        sim_result_or_values,
        *,
        benchmarks: dict | pd.DataFrame | None = None,
        rf: float | pd.Series = 0.0,
        periods_per_year: int | None = 252,
        use_log_returns: bool = False,
        primary_benchmark: str | None = None,
    ) -> None:
        self.use_log_returns = bool(use_log_returns)
        self.periods_per_year = periods_per_year
        self.primary_benchmark = primary_benchmark

        # Extract value series (or series map) and benchmarks
        self.value_map: dict[str, pd.Series] = {}
        self.benchmarks: pd.DataFrame | None = None
        self.diagnostics_map: dict[str, pd.DataFrame] = {}

        if isinstance(sim_result_or_values, dict):
            # Try keys in priority order
            for key in ("with_cash", "no_cash", "strategy"):
                if key in sim_result_or_values and isinstance(sim_result_or_values[key], pd.DataFrame):
                    df_k = sim_result_or_values[key]
                    s = self._series_from_df(df_k)
                    if s is not None:
                        self.value_map[key] = s
                    diag = self._diagnostics_from_df(df_k)
                    if diag is not None and not diag.empty:
                        self.diagnostics_map[key] = diag

            # Benchmarks inside result
            if "benchmarks" in sim_result_or_values and isinstance(sim_result_or_values["benchmarks"], pd.DataFrame):
                self.benchmarks = sim_result_or_values["benchmarks"].copy()

        elif isinstance(sim_result_or_values, pd.Series):
            self.value_map["value"] = self._coerce_series(sim_result_or_values)

        elif isinstance(sim_result_or_values, pd.DataFrame):
            df0 = sim_result_or_values
            s = self._series_from_df(df0)
            if s is not None:
                self.value_map["value"] = s
            diag = self._diagnostics_from_df(df0)
            if diag is not None and not diag.empty:
                self.diagnostics_map["value"] = diag

        # External benchmarks override if passed explicitly
        if benchmarks is not None:
            if isinstance(benchmarks, dict):
                self.benchmarks = pd.DataFrame(benchmarks)
            elif isinstance(benchmarks, pd.DataFrame):
                self.benchmarks = benchmarks.copy()

        # Ensure we have at least one series
        if not self.value_map:
            raise ValueError("Analysis: could not extract a portfolio value series from the input.")

        # Normalize indices and drop NaNs
        for k, s in list(self.value_map.items()):
            s = self._coerce_series(s)
            s = s.dropna()
            self.value_map[k] = s

        if self.benchmarks is not None:
            self.benchmarks = self._coerce_frame(self.benchmarks)

        # Infer periods/year if not provided
        if self.periods_per_year is None:
            self.periods_per_year = self._infer_periods_per_year(next(iter(self.value_map.values())).index)

        # Risk-free handling
        self.rf = rf

    # ---------------------- public API ----------------------
    def returns(self) -> pd.DataFrame:
        """Return a DataFrame of period returns for each tracked series in value_map."""
        out = {}
        for k, s in self.value_map.items():
            out[k] = self._value_to_returns(s)
        return pd.DataFrame(out)

    def summary(self) -> pd.DataFrame:
        """
        Compute summary metrics for each portfolio series in the input.
        If benchmarks are available, include benchmark-relative metrics vs the primary benchmark.
        """
        rows = []
        for name, s in self.value_map.items():
            metr = self._compute_all_metrics(s, name=name)

            # If diagnostics were available in the original simulation output, include summarized diagnostics
            if hasattr(self, "diagnostics_map") and name in getattr(self, "diagnostics_map", {}):
                diag = self.diagnostics_map.get(name)
                if isinstance(diag, pd.DataFrame) and not diag.empty:
                    v0 = float(s.iloc[0]) if len(s) else None
                    metr.update(self._diagnostic_summary(diag, start_value=v0))

            rows.append(metr)
        df = pd.DataFrame(rows).set_index("Series")

        # Optionally add benchmark summary rows too
        if self.benchmarks is not None and not self.benchmarks.empty:
            # Compute metrics for each benchmark column by treating it as a value curve (already in USD value units)
            # If the benchmark frame is in value units already (from Benchmarks.buy_and_hold_values), we can use directly.
            for col in self.benchmarks.columns:
                s = self.benchmarks[col].dropna()
                if s.empty:
                    continue
                metr = self._compute_all_metrics(s, name=f"Benchmark:{col}")
                df = pd.concat([df, pd.DataFrame([metr]).set_index("Series")], axis=0)

        return df

    def drawdowns(self, top_n: int = 10) -> pd.DataFrame:
        """Return top-N drawdowns for the first portfolio series (for quick inspection)."""
        name, s = next(iter(self.value_map.items()))
        _, dd_tbl = self._drawdown_path_and_table(s, top_n=top_n)
        dd_tbl.insert(0, "Series", name)
        return dd_tbl

    def rolling(self, window: int = 252) -> pd.DataFrame:
        """Return rolling metrics (Sharpe, Vol, MaxDD) for the first portfolio series."""
        name, s = next(iter(self.value_map.items()))
        r = self._value_to_returns(s)
        rf_per = self._rf_to_periodic(r.index)
        ex = r - rf_per.reindex_like(r).fillna(0.0)
        win = max(int(window), 1)

        roll = pd.DataFrame(index=r.index)
        roll["Rolling Vol (ann)"] = r.rolling(win).std() * math.sqrt(self.periods_per_year)
        down = ex.where(ex < 0.0, 0.0)
        roll["Rolling Sortino (ann)"] = ex.rolling(win).mean() / down.rolling(win).std().replace(0.0, np.nan) * math.sqrt(self.periods_per_year)
        roll["Rolling Sharpe (ann)"] = ex.rolling(win).mean() / r.rolling(win).std().replace(0.0, np.nan) * math.sqrt(self.periods_per_year)
        # Windowed max drawdown approximated by computing on rolling cumprod
        val = (1.0 + r).fillna(1.0)
        cum = val.cumprod()
        rolling_peak = cum.rolling(win, min_periods=1).max()
        roll["Rolling MaxDD"] = (cum / rolling_peak - 1.0).rolling(win, min_periods=1).min()
        return roll

    # ---------------------- internals ----------------------
    def _compute_all_metrics(self, value: pd.Series, *, name: str) -> dict:
        value = self._coerce_series(value)
        if value.empty:
            raise ValueError("Empty value series")

        start, end = value.index[0], value.index[-1]
        v0, v1 = float(value.iloc[0]), float(value.iloc[-1])
        elapsed_years = max((end - start).days / 365.25, 0.0)

        r = self._value_to_returns(value)
        rf_per = self._rf_to_periodic(r.index)
        ex = r - rf_per.reindex_like(r).fillna(0.0)

        # Basic stats
        total_ret = (v1 / v0 - 1.0) if v0 > 0 else np.nan
        cagr = (v1 / v0) ** (1.0 / elapsed_years) - 1.0 if (v0 > 0 and elapsed_years > 0) else np.nan

        mu = r.mean()
        sigma = r.std(ddof=1)
        vol_ann = sigma * math.sqrt(self.periods_per_year) if not np.isnan(sigma) else np.nan

        ex_mu = ex.mean()
        sharpe = (ex_mu / sigma) * math.sqrt(self.periods_per_year) if sigma not in (0.0, np.nan) else np.nan

        # Sortino
        downside = ex.where(ex < 0.0, 0.0)
        dd_sigma = downside.std(ddof=1)
        sortino = (ex_mu / dd_sigma) * math.sqrt(self.periods_per_year) if dd_sigma not in (0.0, np.nan) else np.nan

        # Drawdowns
        dd_path, dd_tbl = self._drawdown_path_and_table(value, top_n=10)
        maxdd = float(dd_tbl["Drawdown"].min()) if not dd_tbl.empty else 0.0
        calmar = (cagr / abs(maxdd)) if (not np.isnan(cagr) and abs(maxdd) > 1e-12) else np.nan

        # Other diagnostics
        skew = r.skew()
        kurt = r.kurt()  # excess kurtosis in pandas
        hitrate = (r > 0.0).mean()

        # Best/worst period
        best = r.max()
        worst = r.min()
        best_dt = r.idxmax() if not r.empty else pd.NaT
        worst_dt = r.idxmin() if not r.empty else pd.NaT

        # VaR/ES (historical, 1-period)
        def var_cvar(series: pd.Series, q: float) -> tuple[float, float]:
            if series.empty:
                return (np.nan, np.nan)
            var = series.quantile(q)
            cvar = series[series <= var].mean() if q <= 0.5 else series[series >= var].mean()
            return (var, cvar)

        var95, es95 = var_cvar(r, 0.05)
        var99, es99 = var_cvar(r, 0.01)

        # Benchmark-relative (if available)
        alpha_ann = beta = r2 = te_ann = ir = active_ann = np.nan
        bm_name = None
        if self.benchmarks is not None and not self.benchmarks.empty:
            # Choose primary benchmark
            if self.primary_benchmark in self.benchmarks.columns:
                bm_name = self.primary_benchmark
            else:
                bm_name = self.benchmarks.columns[0]
            bm_val = self.benchmarks[bm_name].dropna().reindex(value.index).interpolate(limit_direction="both")
            bm_r = self._value_to_returns(bm_val)
            bm_ex = bm_r - rf_per.reindex_like(bm_r).fillna(0.0)

            # Active and TE/IR
            active = r.reindex_like(bm_r) - bm_r
            te_ann = active.std(ddof=1) * math.sqrt(self.periods_per_year)
            active_ann = active.mean() * self.periods_per_year if not np.isnan(active.mean()) else np.nan
            ir = (active_ann / te_ann) if (te_ann not in (0.0, np.nan)) else np.nan

            # CAPM OLS: re_p = a/PY + b * re_m + eps
            X = bm_ex.values.reshape(-1, 1)
            Y = ex.reindex_like(bm_ex).values.reshape(-1, 1)
            mask = ~np.isnan(X).ravel() & ~np.isnan(Y).ravel()
            Xm = X[mask].ravel()
            Ym = Y[mask].ravel()
            if Xm.size >= 2:
                xm = Xm.mean(); ym = Ym.mean()
                cov = ((Xm - xm) * (Ym - ym)).sum() / (Xm.size - 1)
                var = ((Xm - xm) ** 2).sum() / (Xm.size - 1)
                beta = cov / var if var != 0 else np.nan
                alpha_per_period = ym - beta * xm
                alpha_ann = alpha_per_period * self.periods_per_year
                # R^2
                ss_tot = ((Ym - ym) ** 2).sum()
                ss_res = ((Ym - (alpha_per_period + beta * Xm)) ** 2).sum()
                r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan

        return {
            "Series": name,
            "Start": start,
            "End": end,
            "Periods": len(r),
            "Years": elapsed_years,
            "Start Value": v0,
            "End Value": v1,
            "Total Return": total_ret,
            "CAGR": cagr,
            "Vol (ann)": vol_ann,
            "Sharpe (ann)": sharpe,
            "Sortino (ann)": sortino,
            "Downside Dev (ann)": dd_sigma * math.sqrt(self.periods_per_year) if not np.isnan(dd_sigma) else np.nan,
            "Upside Dev (ann)": ex.where(ex > 0.0, 0.0).std(ddof=1) * math.sqrt(self.periods_per_year),
            "Max Drawdown": maxdd,
            "Calmar": calmar,
            "Skew": skew,
            "Kurtosis (excess)": kurt,
            "Hit Rate": hitrate,
            "Best Period Return": best,
            "Best Period Date": best_dt,
            "Worst Period Return": worst,
            "Worst Period Date": worst_dt,
            "VaR 95%": var95,
            "ES 95%": es95,
            "VaR 99%": var99,
            "ES 99%": es99,
            # Benchmark-rel
            "Primary Benchmark": bm_name,
            "Active Return (ann)": active_ann,
            "Tracking Error (ann)": te_ann,
            "Information Ratio": ir,
            "Alpha (ann)": alpha_ann,
            "Beta": beta,
            "R^2": r2,
        }

    # ---- helpers ----
    @staticmethod
    def _coerce_series(s: pd.Series) -> pd.Series:
        s = pd.Series(s).copy()
        # Ensure datetime index, monotonic increasing, drop duplicates
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        s.index = s.index.tz_localize(None)
        s = s[~s.index.duplicated(keep="first")]
        s = s.sort_index()
        return s

    @staticmethod
    def _coerce_frame(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index)
        out.index = out.index.tz_localize(None)
        out = out[~out.index.duplicated(keep="first")]
        out = out.sort_index()
        return out

    
    def _diagnostics_from_df(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """
        Extract margin/overlay diagnostic time series from a simulation output DataFrame, if present.
        Returns a DataFrame indexed by Date with a subset of known diagnostic columns.
        """
        if df is None:
            return None
        try:
            dfc = self._coerce_frame(df)
        except Exception:
            return None

        diag_cols = [
            "Borrowed",
            "Gross Exposure",
            "Net Exposure",
            "Effective Leverage",
            "Interest Paid",
            "Cumulative Interest Paid",
        ]
        cols = [c for c in diag_cols if c in dfc.columns]
        if not cols:
            return None
        return dfc[cols].copy()

    def _diagnostic_summary(self, diag: pd.DataFrame, *, start_value: float | None = None) -> dict:
        """Compute simple aggregates of diagnostics suitable for the summary table."""
        out: dict[str, float] = {}

        def _mean(col: str) -> float:
            return float(diag[col].mean()) if col in diag.columns else float("nan")

        def _max(col: str) -> float:
            return float(diag[col].max()) if col in diag.columns else float("nan")

        out["Avg Borrowed"] = _mean("Borrowed")
        out["Max Borrowed"] = _max("Borrowed")

        out["Avg Gross Exposure"] = _mean("Gross Exposure")
        out["Max Gross Exposure"] = _max("Gross Exposure")

        out["Avg Net Exposure"] = _mean("Net Exposure")

        out["Avg Effective Leverage"] = _mean("Effective Leverage")
        out["Max Effective Leverage"] = _max("Effective Leverage")

        total_interest = float(diag["Interest Paid"].sum()) if "Interest Paid" in diag.columns else float("nan")
        out["Total Interest Paid"] = total_interest

        end_cum_interest = float(diag["Cumulative Interest Paid"].iloc[-1]) if "Cumulative Interest Paid" in diag.columns and not diag.empty else float("nan")
        out["Cumulative Interest Paid (end)"] = end_cum_interest

        if start_value is not None and np.isfinite(start_value) and start_value != 0.0 and np.isfinite(total_interest):
            out["Interest Paid (% start)"] = total_interest / float(start_value)
        else:
            out["Interest Paid (% start)"] = float("nan")

        return out

    def _series_from_df(self, df: pd.DataFrame) -> pd.Series | None:
        df = self._coerce_frame(df)
        # Ensemble frame
        if {"Ensemble Mean", "Ensemble P10", "Ensemble P90"}.issubset(df.columns):
            return df["Ensemble Mean"]
        # Single-pass frame
        for col in ("Portfolio Value", "Value", "Total Value"):
            if col in df.columns:
                return df[col]
        # If only one column, use it
        if df.shape[1] == 1:
            return df.iloc[:, 0]
        return None

    def _infer_periods_per_year(self, idx: pd.DatetimeIndex) -> int:
        if len(idx) < 2:
            return 252
        deltas = np.diff(idx.values.astype("datetime64[ns]")).astype("timedelta64[D]").astype(float)
        median_days = np.nanmedian(deltas) if deltas.size else 1.0
        if not np.isfinite(median_days) or median_days <= 0:
            return 252
        approx = int(round(365.25 / median_days))
        # snap to common values
        for cand in (252, 365, 52, 12):
            if abs(approx - cand) <= 5:
                return cand
        return max(1, approx)

    def _rf_to_periodic(self, idx: pd.DatetimeIndex) -> pd.Series:
        """Return a Series of per-period RF matching index. Accepts scalar APR or Series (APR or per-period)."""
        if isinstance(self.rf, (int, float)):
            apr = float(self.rf)
            per = (1.0 + apr) ** (1.0 / self.periods_per_year) - 1.0
            return pd.Series(per, index=idx)
        elif isinstance(self.rf, pd.Series):
            s = self._coerce_series(self.rf).reindex(idx).ffill()
            # Heuristic: if median value > 0.2, it looks annual; convert. Otherwise assume per-period already.
            med = s.median()
            if pd.notna(med) and med > 0.2:
                per = (1.0 + s) ** (1.0 / self.periods_per_year) - 1.0
                return per
            return s
        else:
            return pd.Series(0.0, index=idx)

    def _value_to_returns(self, value: pd.Series) -> pd.Series:
        v = self._coerce_series(value).astype(float)
        if self.use_log_returns:
            # log returns: ln(V_t) - ln(V_{t-1})
            r = np.log(v).diff()
        else:
            r = v.pct_change()
        return r.dropna()

    @staticmethod
    def _drawdown_path_and_table(value: pd.Series, *, top_n: int = 10):
        """Return (path_df, top_table) for drawdowns from the given value curve."""
        v = pd.Series(value).dropna()
        if not isinstance(v.index, pd.DatetimeIndex):
            v.index = pd.to_datetime(v.index)
        v.index = v.index.tz_localize(None)
        v = v.sort_index()

        running_max = v.cummax()
        dd = v / running_max - 1.0

        # Identify drawdown episodes
        episodes = []
        in_dd = False
        start = None
        trough = None
        for dt, d in dd.items():
            if not in_dd:
                if d < 0:
                    in_dd = True
                    start = dt
                    trough = dt
                continue
            # in a drawdown
            if d < dd[trough]:
                trough = dt
            if d == 0.0:
                # recovery
                episodes.append((start, trough, dt, float(dd[start:dt].min())))
                in_dd = False
                start = trough = None
        # If unfinished drawdown at the end
        if in_dd and start is not None:
            episodes.append((start, trough, pd.NaT, float(dd[start:].min())))

        tbl = pd.DataFrame(episodes, columns=["Start", "Trough", "Recovery", "Drawdown"]).sort_values("Drawdown")
        # Durations
        tbl["Trough Length (days)"] = (tbl["Trough"] - tbl["Start"]).dt.days
        tbl["Total Length (days)"] = (tbl["Recovery"].fillna(v.index[-1]) - tbl["Start"]).dt.days

        # Path df
        path = pd.DataFrame({"Value": v, "Peak": running_max, "Drawdown": dd})

        # Top-N
        return path, tbl.head(top_n)
