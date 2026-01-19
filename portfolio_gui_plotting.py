"""Plotly plotting utilities for the Portfolio Simulator GUI.

This module provides Plotly (interactive) figures. It is designed so the GUI can
still run with Matplotlib fallback if Plotly is not installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _import_plotly():
    """Import Plotly lazily.

    Returns
    -------
    (go, make_subplots) or (None, None) if Plotly is unavailable.
    """
    try:
        import plotly.graph_objects as go  # type: ignore
        from plotly.subplots import make_subplots  # type: ignore
        return go, make_subplots
    except Exception:
        return None, None


def _pick_strategy_df(result: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
    for k in ("with_cash", "no_cash", "strategy"):
        df = result.get(k, None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return k, df
    raise ValueError("No strategy dataframe found in result")


def _is_ensemble_df(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return {"Ensemble Mean", "Ensemble P10", "Ensemble P90"}.issubset(cols)


def _reexpress_series(s: pd.Series, y_as: str) -> pd.Series:
    y_as = str(y_as).lower().strip()
    if y_as not in {"value", "pct", "index", "log"}:
        raise ValueError("y_as must be one of: value, pct, index, log")
    s2 = s.astype(float).copy()
    if y_as in {"value", "log"}:
        return s2
    base = s2.dropna().iloc[0] if s2.dropna().size else float("nan")
    if not (base and pd.notna(base)):
        return s2 * float("nan")
    if y_as == "pct":
        return (s2 / base - 1.0) * 100.0
    return (s2 / base) * 100.0


def make_performance_figure(
    *,
    result: Dict[str, pd.DataFrame],
    sim=None,
    y_as: str = "pct",
    show_benchmarks: bool = True,
    show_events: bool = True,
    title: Optional[str] = None,
    template: str = "plotly_dark",
    monte_carlo_result: Optional[Dict] = None,
) -> "object":
    """Interactive performance plot (Plotly).

    Notes
    -----
    - Event markers are rendered as dots on the primary strategy curve at the
      first trading day on/after each event date (matching Simulation's trigger logic).
    - The Plotly figure is styled with a dark template by default.
    - If monte_carlo_result is provided, percentile bands from random selection
      trials are shown as a shaded region.
    """
    go, _ = _import_plotly()
    if go is None:
        raise ImportError("Plotly is not available")

    key, df = _pick_strategy_df(result)

    # Strategy series used for event markers + primary hover
    series_for_events: Optional[pd.Series] = None
    series_label: str = key

    fig = go.Figure()

    # Title: default to empty (user requested no visible title)
    if title is None:
        title = ""

    # Add Monte Carlo percentile bands first (so they appear behind strategy line)
    if monte_carlo_result is not None and monte_carlo_result.get("percentiles"):
        percentiles = monte_carlo_result["percentiles"]
        
        # Get p10 and p90 for the band
        if "p10" in percentiles and "p90" in percentiles:
            p10 = _reexpress_series(percentiles["p10"], y_as)
            p90 = _reexpress_series(percentiles["p90"], y_as)
            
            # Add lower bound (invisible)
            fig.add_trace(
                go.Scatter(
                    x=p10.index,
                    y=p10.values,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Add upper bound with fill
            fig.add_trace(
                go.Scatter(
                    x=p90.index,
                    y=p90.values,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(128, 128, 128, 0.25)",
                    line=dict(width=0),
                    name="MC Random P10–P90",
                    hoverinfo="skip",
                )
            )
        
        # Add p25-p75 band (darker)
        if "p25" in percentiles and "p75" in percentiles:
            p25 = _reexpress_series(percentiles["p25"], y_as)
            p75 = _reexpress_series(percentiles["p75"], y_as)
            
            fig.add_trace(
                go.Scatter(
                    x=p25.index,
                    y=p25.values,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=p75.index,
                    y=p75.values,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(128, 128, 128, 0.35)",
                    line=dict(width=0),
                    name="MC Random P25–P75",
                    hoverinfo="skip",
                )
            )
        
        # Add median line
        if "p50" in percentiles:
            p50 = _reexpress_series(percentiles["p50"], y_as)
            fig.add_trace(
                go.Scatter(
                    x=p50.index,
                    y=p50.values,
                    mode="lines",
                    name="MC Random Median",
                    line=dict(width=1.5, dash="dash", color="rgba(150, 150, 150, 0.8)"),
                    hovertemplate="MC Median<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
                )
            )

    if _is_ensemble_df(df):
        mean = _reexpress_series(df["Ensemble Mean"], y_as)
        series_for_events = mean
        series_label = f"{key} mean"

        fig.add_trace(
            go.Scatter(
                x=mean.index,
                y=mean.values,
                mode="lines",
                name=series_label,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
            )
        )

        if "Ensemble P10" in df.columns and "Ensemble P90" in df.columns:
            lo = _reexpress_series(df["Ensemble P10"], y_as)
            hi = _reexpress_series(df["Ensemble P90"], y_as)
            fig.add_trace(
                go.Scatter(
                    x=lo.index,
                    y=lo.values,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=hi.index,
                    y=hi.values,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(255,255,255,0.10)",
                    line=dict(width=0),
                    name=f"{key} P10–P90",
                    hoverinfo="skip",
                )
            )
    else:
        s = _reexpress_series(df["Portfolio Value"], y_as)
        series_for_events = s
        series_label = key

        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name="Your Strategy",
                line=dict(width=2.5, color="#00cc96"),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
            )
        )

    # Benchmarks
    if show_benchmarks:
        bench = result.get("benchmarks", None)
        if isinstance(bench, pd.DataFrame) and not bench.empty:
            for col in bench.columns:
                try:
                    bs = _reexpress_series(bench[col], y_as)
                except Exception:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=bs.index,
                        y=bs.values,
                        mode="lines",
                        name=f"Benchmark: {col}",
                        line=dict(width=1),
                        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
                    )
                )

    # Event markers: dots placed on the strategy curve
    if show_events and sim is not None and series_for_events is not None:
        try:
            idx = pd.DatetimeIndex(series_for_events.index).tz_localize(None)
            xs = []
            ys = []
            for e in getattr(sim, "events", []) or []:
                ed = pd.Timestamp(getattr(e, "date", e)).tz_localize(None).normalize()
                pos = idx.searchsorted(ed)
                if pos >= len(idx):
                    continue
                d_exec = pd.Timestamp(idx[pos])
                xs.append(d_exec)
                ys.append(float(series_for_events.iloc[pos]))
            if xs:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers",
                        name="Events",
                        marker=dict(size=7, symbol="circle", color="rgba(255, 165, 0, 0.9)", line=dict(width=0)),
                        hovertemplate="Event<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
                    )
                )
        except Exception:
            pass

    ytitle = {
        "value": "Portfolio Value",
        "pct": "Return (%)",
        "index": "Index (Base=100)",
        "log": "Portfolio Value (log)",
    }[str(y_as).lower().strip()]

    fig.update_layout(
        template=template,
        title=title,
        xaxis_title="Date",
        yaxis_title=ytitle,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=20 if not title else 50, b=40),
    )
    if str(y_as).lower().strip() == "log":
        fig.update_yaxes(type="log")

    return fig


def make_comparison_figure(
    *,
    runs: "list[dict]",
    y_as: str = "pct",
    show_benchmarks: bool = True,
    title: Optional[str] = None,
    template: str = "plotly_dark",
) -> "object":
    """Build an interactive Plotly figure overlaying multiple stored runs.

    Parameters
    ----------
    runs : list[dict]
        List of run records (as stored in st.session_state.runs). Each run record
        is expected to include at least:
          - "name": display label
          - "result": dict[str, DataFrame] simulation result
        If a run is missing a usable strategy series, it is skipped.
    y_as : {"pct", "value", "index", "log"}
        Output scale (consistent with make_performance_figure).
    show_benchmarks : bool
        If True, include benchmark traces (added once, sourced from the first
        run that contains a non-empty "benchmarks" frame).
    title : str | None
        Figure title. If None, defaults to empty.
    template : str
        Plotly template name.
    """
    go, _ = _import_plotly()
    if go is None:
        raise ImportError("Plotly is not available")

    if title is None:
        title = ""

    fig = go.Figure()

    # Add each run trace
    for r in runs or []:
        res = r.get("result", None)
        if not isinstance(res, dict):
            continue

        try:
            _, df = _pick_strategy_df(res)
        except Exception:
            continue

        # Select a representative series
        if _is_ensemble_df(df):
            if "Ensemble Mean" not in df.columns:
                continue
            s = _reexpress_series(df["Ensemble Mean"], y_as)
        else:
            if "Portfolio Value" not in df.columns:
                continue
            s = _reexpress_series(df["Portfolio Value"], y_as)

        label = str(r.get("name", "Run"))
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=label,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
            )
        )

    # Benchmarks (added once)
    if show_benchmarks:
        bench_df = None
        for r in runs or []:
            res = r.get("result", None)
            if isinstance(res, dict):
                b = res.get("benchmarks", None)
                if isinstance(b, pd.DataFrame) and not b.empty:
                    bench_df = b
                    break

        if isinstance(bench_df, pd.DataFrame) and not bench_df.empty:
            for col in bench_df.columns:
                try:
                    bs = _reexpress_series(bench_df[col], y_as)
                except Exception:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=bs.index,
                        y=bs.values,
                        mode="lines",
                        name=f"Benchmark: {col}",
                        line=dict(width=1),
                        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
                    )
                )

    ytitle = {
        "value": "Portfolio Value",
        "pct": "Return (%)",
        "index": "Index (Base=100)",
        "log": "Portfolio Value (log)",
    }[str(y_as).lower().strip()]

    fig.update_layout(
        template=template,
        title=title,
        xaxis_title="Date",
        yaxis_title=ytitle,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=20 if not title else 50, b=40),
    )
    if str(y_as).lower().strip() == "log":
        fig.update_yaxes(type="log")

    return fig


def make_holdings_heatmap_figure(
    *,
    result: Dict[str, pd.DataFrame],
    sim=None,
    top_n: Optional[int] = None,
    clip: Optional[float] = 20.0,
) -> "object":
    """Interactive holdings heatmap (Plotly) with hover tooltips.

    Hover shows:
      - event date
      - rank slot
      - ticker
      - % return in the inter-event interval

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go, make_subplots = _import_plotly()
    if go is None or make_subplots is None:
        raise ImportError("Plotly is not available")

    if sim is None:
        raise ValueError("sim is required to compute holdings heatmap")

    # Lazily import the engine helper
    import portfolio_sim as ps  # local import to avoid circular dependency

    _, df = _pick_strategy_df(result)
    if _is_ensemble_df(df):
        raise ValueError("Holdings heatmap requires a non-ensemble strategy dataframe")

    hm = ps.compute_holdings_heatmap(sim=sim, strategy_df=df, top_n=top_n, clip=clip)
    ret = hm["returns_pct"]
    tickers = hm["tickers"]
    x_dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in hm["x_dates"]]
    y_labels = hm["y_labels"]
    ordering_label = str(hm.get("ordering_label", "Selection metric"))
    clip_val = float(hm["clip"])
    avg = hm["avg_returns_pct"]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.82, 0.18],
        horizontal_spacing=0.05,
        specs=[[{"type": "heatmap"}, {"type": "bar"}]],
    )

    colorscale = [[0.0, "rgb(192,57,43)"], [0.5, "rgb(255,255,255)"], [1.0, "rgb(30,132,73)"]]

    fig.add_trace(
        go.Heatmap(
            z=ret,
            x=x_dates,
            y=y_labels,
            zmin=-clip_val,
            zmax=clip_val,
            zmid=0.0,
            colorscale=colorscale,
            colorbar=dict(title="Return (%)", x=1.02, thickness=12),
            customdata=tickers,
            hovertemplate=(
                "Event: %{x}<br>"
                "Slot: %{y}<br>"
                "Ticker: %{customdata}<br>"
                "Return: %{z:.2f}%<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # Avg bars (independent x scale)
    avg_series = pd.Series(avg, index=y_labels, dtype=float)
    fig.add_trace(
        go.Bar(
            x=avg_series.values,
            y=avg_series.index,
            orientation="h",
            name="Avg",
            hovertemplate="Slot: %{y}<br>Avg Return: %{x:.2f}%<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Independent x range for avg bars
    finite_avg = avg_series.dropna().values
    if finite_avg.size:
        mn, mx = float(finite_avg.min()), float(finite_avg.max())
        span = max(1e-9, mx - mn)
        pad = 0.15 * span
        lo = min(mn - pad, 0.0)
        hi = max(mx + pad, 0.0)
        fig.update_xaxes(range=[lo, hi], row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=30, t=30, b=80),
        showlegend=False,
    )

    fig.update_xaxes(tickangle=90, row=1, col=1)
    fig.update_yaxes(title_text=f"Selection order by: {ordering_label}", row=1, col=1)
    fig.update_xaxes(title_text="Avg Return (%)", row=1, col=2)

    # Invert y-axis so that lower ranks (e.g., 1) appear at the top.
    # Apply to both the heatmap panel and the avg-bar panel so the rank
    # ordering is consistent across columns.
    fig.update_yaxes(
        autorange="reversed",
        categoryorder="array",
        categoryarray=y_labels,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        autorange="reversed",
        categoryorder="array",
        categoryarray=y_labels,
        row=1,
        col=2,
    )

    return fig


def make_rank_trajectory_overlay_figure(
    *,
    events_detail: pd.DataFrame,
    events: Sequence[Any],
    metric_column: Optional[str] = None,
    max_tickers: Optional[int] = None,
    focus_ticker: Optional[str] = None,
    invert_y: bool = True,
    template: str = "plotly_dark",
) -> "object":
    """Overlay plot of per-ticker selection metric across event times.

    This is intended for the GUI "4) Run & Analyze" page, as a companion to the
    holdings heatmap. Each ticker is a colored line+markers trace.

    Key behavior: If a ticker is absent at an intermediate event time and later
    re-enters, the line should NOT connect across the missing period. This is
    achieved by inserting None values at missing event times.

    Parameters
    ----------
    events_detail : pd.DataFrame
        Output from build_events(..., return_detail=True) stored in session_state
        as events_detail. Expected columns: ['datetime', 'yf_Ticker', 'metric_value'].
    events : Sequence
        The (possibly shifted) events used for the simulation. Used to define the
        x-axis event times.
    metric_column : str, optional
        Label for the y-axis. If None, will attempt to infer from events_detail.
    max_tickers : int, optional
        Optional cap on the number of tickers plotted (chosen by frequency of appearance).
        If None, all available tickers in events_detail are plotted.
    focus_ticker : str, optional
        If provided, visually emphasize this ticker and dim others.
    invert_y : bool
        If True, reverse the y-axis (useful when a lower rank is "better").
    template : str
        Plotly template.
    """
    go, _ = _import_plotly()
    if go is None:
        raise ImportError("Plotly is not available")

    if not isinstance(events_detail, pd.DataFrame) or events_detail.empty:
        raise ValueError("events_detail is required and must not be empty")

    required = {"datetime", "yf_Ticker", "metric_value"}
    missing = required - set(events_detail.columns)
    if missing:
        raise KeyError(f"events_detail missing required columns: {sorted(missing)}")

    df = events_detail.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["datetime", "yf_Ticker"]).copy()
    if df.empty:
        raise ValueError("events_detail has no usable rows after cleaning")

    # Determine x-axis event dates from the provided events.
    event_dates: List[pd.Timestamp] = []
    for e in (events or []):
        try:
            d = pd.Timestamp(getattr(e, "date", e)).tz_localize(None).normalize()
        except Exception:
            continue
        event_dates.append(d)
    # De-duplicate while preserving order
    seen = set()
    event_dates = [d for d in event_dates if not (d in seen or seen.add(d))]

    # If the provided events length matches the unique detail datetimes, align by position.
    # This is important when the GUI shifts events by a fixed number of days.
    detail_dates = sorted(df["datetime"].dropna().unique())
    if len(event_dates) == len(detail_dates) and len(event_dates) > 0:
        date_map = {pd.Timestamp(o): pd.Timestamp(n) for o, n in zip(detail_dates, event_dates)}
        df["datetime"] = df["datetime"].map(lambda x: date_map.get(pd.Timestamp(x), pd.Timestamp(x)))
        x_dates = event_dates
    else:
        # Fall back to the datetimes present in the detail dataframe.
        x_dates = detail_dates

    # Infer metric label when possible
    if metric_column is None:
        if "metric_column" in df.columns:
            uniq = [str(x) for x in df["metric_column"].dropna().unique()]
            metric_column = uniq[0] if len(uniq) == 1 else (uniq[0] if uniq else "Selection metric")
        else:
            metric_column = "Selection metric"
    metric_label = str(metric_column)

    # Choose tickers to plot by frequency across events (most common first)
    counts = df["yf_Ticker"].astype(str).value_counts()
    tickers = counts.index.tolist()
    if max_tickers is not None:
        try:
            k = int(max_tickers)
        except Exception:
            k = None
        if k is not None and k > 0 and len(tickers) > k:
            tickers = tickers[:k]
    if focus_ticker is not None and str(focus_ticker).strip() != "":
        ft = str(focus_ticker).strip()
        if ft not in tickers and ft in counts.index:
            tickers = [ft] + tickers

    # Precompute a date->metric mapping per ticker.
    # If duplicates exist for (datetime, ticker), take the mean.
    df["yf_Ticker"] = df["yf_Ticker"].astype(str)
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
    df = df.dropna(subset=["metric_value"]).copy()

    fig = go.Figure()
    x = x_dates

    # Styling: allow a "focus" ticker to be emphasized
    focus = str(focus_ticker).strip() if (focus_ticker is not None and str(focus_ticker).strip()) else None

    # Base opacity/width
    base_opacity = 0.35 if focus is None else 0.12
    base_width = 1.0
    base_ms = 5
    focus_opacity = 1.0
    focus_width = 3.5
    focus_ms = 7

    for t in tickers:
        dsub = df.loc[df["yf_Ticker"] == t, ["datetime", "metric_value"]]
        if dsub.empty:
            continue
        s = dsub.groupby("datetime")["metric_value"].mean()
        s = s.reindex(x)
        y = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in s.values]

        is_focus = (focus is not None and t == focus)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                connectgaps=False,
                showlegend=False,
                name=t,
                opacity=(focus_opacity if is_focus else base_opacity),
                line=dict(width=(focus_width if is_focus else base_width)),
                marker=dict(size=(focus_ms if is_focus else base_ms)),
                customdata=[t] * len(x),
                hovertemplate=(
                    "Ticker: %{customdata}<br>"
                    "Event: %{x|%Y-%m-%d}<br>"
                    f"{metric_label}: %{{y:.2f}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template=template,
        margin=dict(l=50, r=20, t=30, b=60),
        hovermode="closest",
        xaxis_title="Event date (each sheet)",
        yaxis_title=metric_label,
    )
    if invert_y:
        fig.update_yaxes(autorange="reversed")

    return fig


def make_monte_carlo_histogram(
    mc_result: Dict,
    *,
    strategy_return: Optional[float] = None,
    title: Optional[str] = None,
    template: str = "plotly_dark",
) -> "object":
    """Create a histogram of Monte Carlo final returns.
    
    Parameters
    ----------
    mc_result : dict
        Result from run_monte_carlo_null_distribution containing 'final_returns_pct'.
    strategy_return : float, optional
        Your strategy's final return (%) to show as a vertical line.
    title : str, optional
        Plot title.
    template : str
        Plotly template.
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    go, _ = _import_plotly()
    if go is None:
        raise ImportError("Plotly is not available")
    
    final_returns = mc_result.get("final_returns_pct", [])
    if not final_returns:
        raise ValueError("No final returns data in mc_result")
    
    import numpy as np
    
    fig = go.Figure()
    
    # Histogram of random strategy returns
    fig.add_trace(
        go.Histogram(
            x=final_returns,
            nbinsx=30,
            name="Random Selection Returns",
            marker_color="rgba(100, 100, 100, 0.7)",
            hovertemplate="Return: %{x:.1f}%<br>Count: %{y}<extra></extra>",
        )
    )
    
    # Add vertical line for strategy return
    if strategy_return is not None:
        # Calculate percentile
        pct_rank = (np.array(final_returns) < strategy_return).mean() * 100
        
        # Determine color based on performance
        if pct_rank >= 75:
            color = "#00cc96"  # Green
        elif pct_rank >= 50:
            color = "#ffa500"  # Orange
        else:
            color = "#ef553b"  # Red
        
        fig.add_vline(
            x=strategy_return,
            line_width=3,
            line_dash="solid",
            line_color=color,
            annotation_text=f"Your Strategy: {strategy_return:.1f}% (P{pct_rank:.0f})",
            annotation_position="top",
            annotation_font_color=color,
        )
    
    # Add mean line
    mean_return = np.mean(final_returns)
    fig.add_vline(
        x=mean_return,
        line_width=2,
        line_dash="dash",
        line_color="rgba(255, 255, 255, 0.5)",
        annotation_text=f"MC Mean: {mean_return:.1f}%",
        annotation_position="bottom",
        annotation_font_color="rgba(255, 255, 255, 0.7)",
    )
    
    if title is None:
        title = ""
    
    fig.update_layout(
        template=template,
        title=title,
        xaxis_title="Total Return (%)",
        yaxis_title="Frequency",
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    
    return fig
