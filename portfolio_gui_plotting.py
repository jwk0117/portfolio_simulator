"""Plotly plotting utilities for the Portfolio Simulator GUI.

This module provides Plotly (interactive) figures. It is designed so the GUI can
still run with Matplotlib fallback if Plotly is not installed.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

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
) -> "object":
    """Interactive performance plot (Plotly).

    Notes
    -----
    - Event markers are rendered as dots on the primary strategy curve at the
      first trading day on/after each event date (matching Simulation's trigger logic).
    - The Plotly figure is styled with a dark template by default.
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
                name=key,
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

    return fig
