# -*- coding: utf-8 -*-
"""portfolio_gui_app.py

Streamlit GUI for the portfolio_sim backtesting engine.

How to run
----------
1) Install dependencies (recommended in a fresh environment):
      pip install -r requirements_gui.txt

2) Ensure portfolio_sim.py is in the same folder as this file.

3) Start the app:
      streamlit run portfolio_gui_app.py

This app expects the input dataset as a preprocessed .pkl. The .pkl should
contain "sheet-like" DataFrames that drive Event construction.

Implementation note
-------------------
portfolio_sim.plot_simulation() calls plt.show() and does not return the Figure.
In a GUI context we suppress plt.show() temporarily and capture active figures.
"""

from __future__ import annotations

import io
import json
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import inspect

import portfolio_sim as ps
import portfolio_gui_plotting as pgp
from portfolio_gui_io import SheetItem, extract_sheet_items, infer_column_candidates, load_pkl_bytes, load_pkl_path, summarize_object
from portfolio_gui_core import (
    EventBuildConfig,
    SimulationConfig,
    analyze_result,
    build_events,
    dump_events_json,
    infer_sim_date_range_from_events,
    make_simulation,
    result_to_parquet_bytes,
    run_simulation,
    shift_events_by_calendar_days,
    run_monte_carlo_null_distribution,
)


# -----------------------------
# Streamlit setup
# -----------------------------

st.set_page_config(page_title="Portfolio Simulator GUI", layout="wide")

st.markdown("""
<style>
/* Dark theme overrides */
:root {
  --bg: #0e1117;
  --fg: #fafafa;
}
[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stHeader"], [data-testid="stToolbar"] { background: var(--bg); }
[data-testid="stSidebar"] { background: #111827; }
html, body, [class*="css"] { color: var(--fg); }
</style>
""", unsafe_allow_html=True)


def _ss_init(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


_ss_init("raw_obj", None)
_ss_init("sheets", [])                 # List[SheetItem]
_ss_init("sheet_table", None)          # user edited selection/order table
_ss_init("colmap", {})                 # mapping config
_ss_init("event_cfg", None)            # EventBuildConfig
_ss_init("events", None)               # List[Event]
_ss_init("events_detail", None)        # pd.DataFrame
_ss_init("events_summary", None)       # dict
_ss_init("sim_cfg", None)              # SimulationConfig
_ss_init("sim", None)                  # Simulation
_ss_init("result", None)               # dict[str, DataFrame]
_ss_init("analysis_summary", None)     # pd.DataFrame
_ss_init("runs", [])                   # list of run dicts
_ss_init("event_shift_days", 0)        # event time shift in calendar days
_ss_init("monte_carlo_result", None)   # Monte Carlo null distribution result
_ss_init("run_name_draft", "")         # optional user-provided run name
_ss_init("_clear_run_name_draft", False)  # internal flag to clear text_input safely on next rerun


def _safe_str_list(text: str) -> Optional[List[str]]:
    """Parse a newline/comma separated string list."""
    if text is None:
        return None
    items = []
    for line in str(text).replace(",", "\n").splitlines():
        s = line.strip()
        if s:
            items.append(s)
    return items or None


def _make_unique_run_name(desired: str, existing: Sequence[str]) -> str:
    """Ensure a user-visible run name is unique among existing names.

    If desired is empty/whitespace, the caller should provide a default.
    If desired already exists, append " (2)", " (3)", ...
    """
    base = str(desired).strip()
    if not base:
        base = "Run"
    existing_set = set(map(str, existing or []))
    if base not in existing_set:
        return base
    i = 2
    while True:
        cand = f"{base} ({i})"
        if cand not in existing_set:
            return cand
        i += 1


def _sheet_selector_ui(sheet_items: Sequence[SheetItem]) -> List[SheetItem]:
    """UI to select and order sheets.

    Returns the ordered subset.
    """
    if not sheet_items:
        return []

    df = pd.DataFrame(
        {
            "Use": True,
            "Order": np.arange(len(sheet_items), dtype=int) + 1,
            "Name": [s.name for s in sheet_items],
            "Rows": [int(len(s.df)) for s in sheet_items],
            "Cols": [int(s.df.shape[1]) for s in sheet_items],
        }
    )

    st.caption("Select which sheets to include and define their order.")
    edited = st.data_editor(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "Use": st.column_config.CheckboxColumn(required=True),
            "Order": st.column_config.NumberColumn(min_value=1, step=1),
        },
        key="sheet_table_editor",
    )

    # Keep for downstream steps
    st.session_state.sheet_table = edited

    use = edited[edited["Use"] == True].copy()  # noqa: E712
    use = use.sort_values(["Order", "Name"], ascending=[True, True])
    names = use["Name"].tolist()
    out = [s for s in sheet_items if s.name in set(names)]
    # reorder to match names order
    out = sorted(out, key=lambda s: names.index(s.name))
    return out


def _pick_columns_ui(sheet_items: Sequence[SheetItem]) -> Dict[str, str]:
    """UI for selecting the ticker/datetime/ordering columns."""
    if not sheet_items:
        return {}

    st.subheader("Column mapping")
    st.write(
        "Choose which columns in your sheets correspond to:"
        "\n- Ticker (Yahoo Finance / yfinance ticker)"
        "\n- Event datetime"
        "\n- Ordering metric (used to rank/select top-N tickers per event)"
    )

    # Use the first selected sheet as a template
    df0 = sheet_items[0].df
    cands = infer_column_candidates(df0)
    cols = list(df0.columns)

    ticker_col = st.selectbox("Ticker column", options=cols, index=cols.index(cands["ticker"][0]) if cands["ticker"] and cands["ticker"][0] in cols else 0)
    datetime_col = st.selectbox("Datetime column", options=cols, index=cols.index(cands["datetime"][0]) if cands["datetime"] and cands["datetime"][0] in cols else 0)

    numeric_cols = cands["numeric"] if cands["numeric"] else cols
    
    # Default to "Master Rank" if available, otherwise first numeric column
    ordering_default_idx = 0
    for i, col in enumerate(numeric_cols):
        if col == "Master Rank":
            ordering_default_idx = i
            break
    
    ordering_col = st.selectbox("Ordering metric column", options=numeric_cols, index=ordering_default_idx)

    weight_col = st.selectbox(
        "Weight column (optional)",
        options=["(same as ordering metric)"] + list(numeric_cols),
        index=0,
    )
    weight_col = None if weight_col == "(same as ordering metric)" else weight_col

    mapping = {
        "ticker_col": ticker_col,
        "datetime_col": datetime_col,
        "ordering_col": ordering_col,
        "weight_col": weight_col,
    }

    st.session_state.colmap = mapping
    st.write("Preview (first selected sheet):")
    st.dataframe(df0.head(25), width="stretch")

    return mapping


def _capture_matplotlib_figures(func, *args, **kwargs):
    """Run a plotting function and capture all created matplotlib figures."""
    prev_show = plt.show
    plt.show = lambda *a, **k: None  # type: ignore
    try:
        before = set(plt.get_fignums())
        func(*args, **kwargs)
        after = set(plt.get_fignums())
        new_nums = sorted(list(after - before))
        if not new_nums:
            # Fallback: capture current figure, if any
            new_nums = sorted(list(after))
        figs = [plt.figure(n) for n in new_nums]
        return figs
    finally:
        plt.show = prev_show


# -----------------------------
# Sidebar navigation
# -----------------------------

st.sidebar.title("Portfolio Simulator GUI")
page = st.sidebar.radio(
    "Workflow",
    [
        "1) Load & Validate",
        "2) Build Events",
        "3) Simulation Parameters",
        "4) Run & Analyze",
        "5) Compare & Export",
    ],
)


# -----------------------------
# Page 1: Load & Validate
# -----------------------------

if page.startswith("1"):
    st.title("1) Load & Validate (.pkl)")

    st.info(
        "This GUI loads a preprocessed .pkl and extracts one or more sheet-like DataFrames. "
        "Pickle files should only be loaded from sources you trust."
    )

    colA, colB = st.columns([2, 1])
    with colA:
        uploaded = st.file_uploader("Upload a preprocessed .pkl", type=["pkl", "pickle"]) 
    with colB:
        path = st.text_input("Or enter a local path", value="")

    load_clicked = st.button("Load")

    if load_clicked:
        obj = None
        err = None
        try:
            if uploaded is not None:
                obj = load_pkl_bytes(uploaded.getvalue())
            elif path.strip():
                obj = load_pkl_path(path.strip())
            else:
                err = "Provide an uploaded file or a valid local path."
        except Exception as e:
            err = f"Failed to load pickle: {e}"

        if err:
            st.error(err)
        else:
            st.session_state.raw_obj = obj
            sheets = extract_sheet_items(obj)
            st.session_state.sheets = sheets
            st.session_state.events = None
            st.session_state.result = None
            st.session_state.analysis_summary = None

    raw = st.session_state.raw_obj
    if raw is not None:
        st.subheader("Detected structure")
        st.json(summarize_object(raw))

    sheets_all: List[SheetItem] = st.session_state.sheets
    if sheets_all:
        st.subheader("Sheets detected")
        st.write(f"Found {len(sheets_all)} sheet(s).")

        selected_sheets = _sheet_selector_ui(sheets_all)
        if selected_sheets:
            st.subheader("Column mapping")
            mapping = _pick_columns_ui(selected_sheets)
            st.success(
                "Load step complete. Proceed to '2) Build Events' once your sheet selection and column mapping look correct."
            )
        else:
            st.warning("No sheets selected.")
    elif raw is not None:
        st.warning(
            "No sheet-like DataFrames were detected. "
            "Expected a list of DataFrames, a dict of name->DataFrame, or a dict with keys like 'sheets'."
        )


# -----------------------------
# Page 2: Build Events
# -----------------------------

if page.startswith("2"):
    st.title("2) Build Events")

    sheets_all: List[SheetItem] = st.session_state.sheets
    if not sheets_all:
        st.warning("Load your .pkl first (Page 1).")
        st.stop()

    # Reconstruct selected sheets in the same way as page 1, based on the edited table.
    selected_sheets = _sheet_selector_ui(sheets_all)
    if not selected_sheets:
        st.warning("Select at least one sheet.")
        st.stop()

    mapping = st.session_state.colmap or _pick_columns_ui(selected_sheets)
    if not mapping:
        st.warning("Define column mapping first.")
        st.stop()

    st.subheader("Selection and weighting")
    c1, c2, c3 = st.columns(3)
    with c1:
        direction = st.selectbox("Direction", ["ascend", "descend"], index=0)
        top_n = st.number_input("Top N", min_value=1, max_value=200, value=10, step=1)
    with c2:
        weight_mode = st.selectbox("Weight mode", ["equal", "proportional", "softmax", "inverse_rank"], index=0)
        softmax_tau = st.number_input("Softmax temperature (tau)", min_value=0.01, value=1.0, step=0.25)
    with c3:
        dedupe = st.selectbox("De-dupe", ["first", "none"], index=0)
        tie_breaker = st.selectbox("Tie breaker", ["stable", "random"], index=0)

    include_txt = st.text_area("Include tickers (optional; comma/newline separated)", value="")
    exclude_txt = st.text_area("Exclude tickers (optional; comma/newline separated)", value="")
    include = _safe_str_list(include_txt)
    exclude = _safe_str_list(exclude_txt)

    random_state = st.number_input("Random seed (optional)", min_value=0, value=0, step=1)
    use_seed = st.checkbox("Use random seed", value=False)
    rs = int(random_state) if use_seed else None

    cfg = EventBuildConfig(
        ticker_col=mapping["ticker_col"],
        datetime_col=mapping["datetime_col"],
        ordering_col=mapping["ordering_col"],
        direction=direction,
        top_n=int(top_n),
        weight_mode=weight_mode,
        weight_col=mapping.get("weight_col", None),
        softmax_tau=float(softmax_tau),
        include=include,
        exclude=exclude,
        dedupe=dedupe,
        tie_breaker=tie_breaker,
        random_state=rs,
    )
    st.session_state.event_cfg = cfg

    if st.button("Build events"):
        try:
            events, detail_df, summary = build_events(selected_sheets, cfg)
        except Exception as e:
            st.error(f"Failed to build events: {e}")
        else:
            st.session_state.events = events
            st.session_state.events_detail = detail_df
            st.session_state.events_summary = summary
            st.session_state.result = None
            st.session_state.analysis_summary = None
            st.success(f"Built {len(events)} event(s).")

    events = st.session_state.events
    if events:
        st.subheader("Event summary")
        try:
            d0, d1 = infer_sim_date_range_from_events(events)
            st.write({"first_event": str(d0.date()), "last_event": str(d1.date()), "num_events": len(events)})
        except Exception:
            st.write({"num_events": len(events)})

        summary = st.session_state.events_summary
        if isinstance(summary, dict):
            with st.expander("Build summary"):
                st.json({k: (str(v) if isinstance(v, (pd.Timestamp,)) else v) for k, v in summary.items() if k != "ticker_meta"})

        detail_df = st.session_state.events_detail
        if isinstance(detail_df, pd.DataFrame) and not detail_df.empty:
            with st.expander("Selection detail"):
                st.dataframe(detail_df, width="stretch")

        # Download events as JSON
        st.download_button(
            "Download events.json",
            data=dump_events_json(events).encode("utf-8"),
            file_name="events.json",
            mime="application/json",
        )


# -----------------------------
# Page 3: Simulation Parameters
# -----------------------------

if page.startswith("3"):
    st.title("3) Simulation Parameters")

    events = st.session_state.events
    if not events:
        st.warning("Build events first (Page 2).")
        st.stop()

    d0, d1 = infer_sim_date_range_from_events(events)
    st.caption(f"Default date range inferred from events: {d0.date()} to {d1.date()}")

    # -------------------------
    # Event Time Shifting
    # -------------------------
    st.subheader("Event Time Shifting")
    st.caption("Shift all event dates by a fixed number of calendar days. Dates landing on weekends/holidays are moved to the next NYSE trading day.")
    
    shift_col1, shift_col2 = st.columns([2, 1])
    with shift_col1:
        event_shift_days = st.slider(
            "Shift events by (calendar days)",
            min_value=-60,
            max_value=60,
            value=int(st.session_state.event_shift_days),
            step=1,
            help="Positive = shift into future, Negative = shift into past"
        )
    with shift_col2:
        if event_shift_days != 0:
            st.info(f"Events will be shifted by **{event_shift_days:+d}** days")
        else:
            st.success("No shift applied")
    
    st.session_state.event_shift_days = event_shift_days
    
    # Apply shift to get effective events for simulation
    if event_shift_days != 0:
        effective_events = shift_events_by_calendar_days(events, event_shift_days, snap_to_trading_day=True)
        d0_eff, d1_eff = infer_sim_date_range_from_events(effective_events)
        st.caption(f"Effective date range after shift: {d0_eff.date()} to {d1_eff.date()}")
    else:
        effective_events = events
        d0_eff, d1_eff = d0, d1
    
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start date", value=d0_eff.date())
        end_date = st.date_input("End date", value=d1_eff.date())
        initial_capital = st.number_input("Initial capital", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
    with col2:
        cash_policy = st.selectbox("Cash policy", ["None", "fixed", "proportion"], index=0)
        cash_fixed_amount = st.number_input("Cash fixed amount", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
        cash_pct = st.number_input("Cash proportion (0-1)", min_value=0.0, max_value=1.0, value=0.0, step=0.05, format="%.2f")
    with col3:
        max_leverage = st.number_input("Max leverage", min_value=1.0, value=1.0, step=0.1, format="%.2f")
        margin_rate_apr = st.number_input("Margin APR (e.g., 0.05 = 5%)", min_value=0.0, value=0.0, step=0.01, format="%.4f")
        rebalance_mode = st.selectbox("Rebalance mode", ["adjust", "rebuild"], index=0)
        use_adj_close = st.checkbox("Use Adj Close (Yahoo Adj Close)", value=True)

    # -------------------------
    # Shuffling section (commented out)
    # -------------------------
    # st.subheader("Ensemble / randomization")
    # e1, e2, e3, e4 = st.columns(4)
    # with e1:
    #     shuffle_events = st.checkbox("Shuffle events", value=False)
    #     num_shuffle = st.number_input("# shuffles", min_value=1, value=100, step=10)
    # with e2:
    #     shuffle_window_days = st.number_input("Shuffle window days (optional)", min_value=0.0, value=0.0, step=1.0, format="%.1f")
    #     use_shuffle_window = st.checkbox("Enable shuffle window", value=False)
    # with e3:
    #     weight_method = st.selectbox("Weight method", ["None", "random", "uniform"], index=0)
    # with e4:
    #     use_seed = st.checkbox("Use random seed", value=False, key="sim_use_seed")
    #     seed = st.number_input("Seed", min_value=0, value=0, step=1, key="sim_seed")
    
    # Default values for commented-out shuffling options
    shuffle_events = False
    num_shuffle = 100
    shuffle_window_days = 0.0
    use_shuffle_window = False
    weight_method = "None"
    use_seed = False
    seed = 0

    st.subheader("Dollar-neutral overlay")
    dn1, dn2, dn3, dn4 = st.columns(4)
    with dn1:
        dollar_neutral = st.checkbox("Dollar neutral", value=False)
    with dn2:
        hedge_symbol = st.text_input("Hedge symbol", value="SPY")
    with dn3:
        hedge_notional_base = st.selectbox("Hedge base", ["total", "gross_long"], index=0)
    with dn4:
        hedge_rounding = st.selectbox("Hedge rounding", ["floor", "round"], index=0)

    scfg = SimulationConfig(
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date),
        initial_capital=float(initial_capital),
        shuffle_events=bool(shuffle_events),
        num_shuffle=int(num_shuffle),
        shuffle_window_days=float(shuffle_window_days) if use_shuffle_window else None,
        weight_method=None if weight_method == "None" else weight_method,
        random_state=int(seed) if use_seed else None,
        cash_policy=None if cash_policy == "None" else cash_policy,
        cash_fixed_amount=float(cash_fixed_amount),
        cash_pct=float(cash_pct),
        rebalance_mode=str(rebalance_mode),
        use_adj_close=bool(use_adj_close),
        max_leverage=float(max_leverage),
        margin_rate_apr=float(margin_rate_apr),
        dollar_neutral=bool(dollar_neutral),
        hedge_symbol=str(hedge_symbol).strip() or "SPY",
        hedge_notional_base=str(hedge_notional_base),
        hedge_rounding=str(hedge_rounding),
    )

    st.session_state.sim_cfg = scfg
    
    # -------------------------
    # Monte Carlo Null Distribution
    # -------------------------
    st.markdown("---")
    st.subheader("Monte Carlo Null Distribution")
    st.caption("Test how your strategy compares to randomly selecting stocks from your universe.")
    
    mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)
    with mc_col1:
        run_monte_carlo = st.checkbox("Enable Monte Carlo", value=False, key="enable_mc")
    with mc_col2:
        mc_n_trials = st.number_input("Number of trials", min_value=10, max_value=1000, value=100, step=10)
    with mc_col3:
        event_cfg = st.session_state.event_cfg
        default_n = event_cfg.top_n if event_cfg else 10
        mc_n_stocks = st.number_input("Stocks per event", min_value=1, max_value=100, value=default_n, step=1,
                                       help="Number of randomly selected stocks per event (default: same as your strategy)")
    with mc_col4:
        mc_seed = st.number_input("MC random seed", min_value=0, value=42, step=1)
    
    if run_monte_carlo:
        st.info(f"Monte Carlo will run **{mc_n_trials}** trials, each selecting **{mc_n_stocks}** random stocks per event with equal weights.")
    
    # Store MC settings in session state
    st.session_state.mc_enabled = run_monte_carlo
    st.session_state.mc_n_trials = mc_n_trials
    st.session_state.mc_n_stocks = mc_n_stocks
    st.session_state.mc_seed = mc_seed
    
    st.success("Simulation parameters saved in session. Proceed to '4) Run & Analyze'.")


# -----------------------------
# Page 4: Run & Analyze
# -----------------------------

if page.startswith("4"):
    st.title("4) Run & Analyze")

    events = st.session_state.events
    scfg: SimulationConfig = st.session_state.sim_cfg
    if not events:
        st.warning("Build events first (Page 2).")
        st.stop()
    if scfg is None:
        st.warning("Set simulation parameters first (Page 3).")
        st.stop()

    # Apply event time shift if configured
    event_shift_days = st.session_state.get("event_shift_days", 0)
    if event_shift_days != 0:
        effective_events = shift_events_by_calendar_days(events, event_shift_days, snap_to_trading_day=True)
        st.info(f"Events shifted by {event_shift_days:+d} calendar days")
    else:
        effective_events = events

    st.subheader("Run simulation")

    # Clear the draft run name on the next rerun (safe pattern for Streamlit widgets).
    # Streamlit raises if you mutate a widget-backed key after the widget is instantiated.
    if st.session_state.get("_clear_run_name_draft", False):
        st.session_state["run_name_draft"] = ""
        st.session_state["_clear_run_name_draft"] = False

    # Optional: user-defined name for this run (used for Page 5 comparisons)
    st.text_input(
        "Run name (optional)",
        key="run_name_draft",
        help=(
            "Provide a descriptive label for this simulation run (e.g., 'Top10 weekly, margin=1.5, SPY hedge'). "
            "If omitted, an automatic Run N name is used. Names are auto-deduplicated if reused."
        ),
    )
    
    # Check if Monte Carlo is enabled
    mc_enabled = st.session_state.get("mc_enabled", False)
    if mc_enabled:
        mc_n_trials = st.session_state.get("mc_n_trials", 100)
        mc_n_stocks = st.session_state.get("mc_n_stocks", 10)
        mc_seed = st.session_state.get("mc_seed", 42)
        st.caption(f"Monte Carlo enabled: {mc_n_trials} trials, {mc_n_stocks} stocks/event")

    if st.button("Run"):
        import traceback

        existing_names = [r.get("name", "") for r in (st.session_state.runs or [])]
        desired_name = str(st.session_state.get("run_name_draft", "") or "").strip()
        default_name = f"Run {len(existing_names) + 1}"
        final_name = _make_unique_run_name(desired_name or default_name, existing_names)
        run_id = str(uuid.uuid4())

        run_record: Dict[str, Any] = {
            "run_id": run_id,
            "name": final_name,
            "sim_cfg": asdict(scfg),
            "event_cfg": asdict(st.session_state.event_cfg) if st.session_state.event_cfg else None,
            "analysis_summary": None,
            "result": None,
            "ok": False,
            "error": None,
            "traceback": None,
        }

        try:
            # Run main strategy simulation
            sim = make_simulation(effective_events, scfg)
            res = run_simulation(sim)
            summ = analyze_result(res)

            st.session_state.sim = sim
            st.session_state.result = res
            st.session_state.analysis_summary = summ

            run_record.update({"analysis_summary": summ, "result": res, "ok": True})

            # Run Monte Carlo if enabled
            if mc_enabled:
                sheets_all = st.session_state.sheets
                event_cfg = st.session_state.event_cfg

                # Get selected sheets (same logic as page 2)
                sheet_table = st.session_state.sheet_table
                if sheet_table is not None:
                    use = sheet_table[sheet_table["Use"] == True].copy()  # noqa: E712
                    use = use.sort_values(["Order", "Name"], ascending=[True, True])
                    names = use["Name"].tolist()
                    selected_sheets = [s for s in sheets_all if s.name in set(names)]
                    selected_sheets = sorted(selected_sheets, key=lambda s: names.index(s.name))
                else:
                    selected_sheets = sheets_all

                if selected_sheets and event_cfg:
                    progress_bar = st.progress(0, text="Running Monte Carlo simulations...")

                    def update_progress(current, total):
                        progress_bar.progress(current / total, text=f"Monte Carlo: {current}/{total} trials")

                    mc_result = run_monte_carlo_null_distribution(
                        selected_sheets,
                        event_cfg,
                        scfg,
                        n_trials=mc_n_trials,
                        n_stocks=mc_n_stocks,
                        random_state=mc_seed,
                        progress_callback=update_progress,
                    )

                    st.session_state.monte_carlo_result = mc_result
                    progress_bar.empty()
                    st.success(f"Monte Carlo completed: {mc_result['n_trials']} successful trials")
                else:
                    st.warning("Could not run Monte Carlo - missing sheet data or event config")
            else:
                st.session_state.monte_carlo_result = None

        except Exception as e:
            run_record.update({
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            st.error(f"Simulation failed: {e}")
            st.code(run_record["traceback"])
        finally:
            # Always store the attempt in run history so Page 5 can see it.
            st.session_state.runs.append(run_record)
            # Clear the draft name after each run attempt to reduce accidental reuse.
            # Do this on the *next* rerun to avoid mutating a widget-backed key post-instantiation.
            st.session_state["_clear_run_name_draft"] = True

    res = st.session_state.result
    sim = st.session_state.sim
    summ = st.session_state.analysis_summary

    if isinstance(summ, pd.DataFrame):
        st.subheader("Performance summary")
        st.dataframe(summ, width="stretch")

        st.download_button(
            "Download summary.csv",
            data=summ.to_csv(index=True).encode("utf-8"),
            file_name="analysis_summary.csv",
            mime="text/csv",
        )

    if isinstance(res, dict) and sim is not None:
        tabs = st.tabs(["Visualization - Performance", "Monte Carlo Analysis", "Trades", "Data Health"])

        # -------------------------
        # Tab 1: Performance + Heatmap
        # -------------------------
        with tabs[0]:
            st.subheader("Visualization")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                backend = st.radio("Plot backend", ["Interactive (Plotly)", "Matplotlib"], index=0, horizontal=False)
            with c2:
                y_as = st.selectbox("Y-axis", ["pct", "value", "index", "log"], index=0)
                show_bench = st.checkbox("Benchmarks", value=True)
            with c3:
                show_events = st.checkbox("Event markers (dots)", value=True)
                show_mc_bands = st.checkbox("Show MC percentile bands", value=True) if st.session_state.monte_carlo_result else False
            with c4:
                top_n = st.number_input("Heatmap slots (top N)", min_value=1, value=10, step=1)
                hm_clip = st.number_input("Heatmap clip (abs %; 0=auto)", min_value=0.0, value=20.0, step=1.0)
                hm_clip = None if hm_clip <= 0 else float(hm_clip)

            if backend.startswith("Interactive"):
                try:
                    mc_result = st.session_state.monte_carlo_result if show_mc_bands else None
                    fig = pgp.make_performance_figure(
                        result=res,
                        sim=sim,
                        y_as=y_as,
                        show_benchmarks=bool(show_bench),
                        show_events=bool(show_events),
                        title="",
                        template="plotly_dark",
                        monte_carlo_result=mc_result,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Interactive plot failed; falling back to Matplotlib. Error: {e}")
                    backend = "Matplotlib"

            if backend == "Matplotlib":
                plot_kwargs = dict(
                    result=res,
                    sim=sim,
                    y_as=y_as,
                    show_benchmarks=show_bench,
                    # Matplotlib fallback: event markers and heatmap are handled in the Plotly plotting module.
                )
                figs = _capture_matplotlib_figures(ps.plot_simulation, **plot_kwargs)
                for fig in figs:
                    st.pyplot(fig, clear_figure=False, use_container_width=True)
                    plt.close(fig)

            st.markdown("---")
            st.subheader("Holdings Heatmap")
            try:
                fig_hm = pgp.make_holdings_heatmap_figure(
                    result=res,
                    sim=sim,
                    top_n=int(top_n),
                    clip=hm_clip,
                )
                st.plotly_chart(fig_hm, use_container_width=True)
            except Exception as e:
                st.error(f"Holdings heatmap failed: {e}")

        # -------------------------
        # Tab 2: Monte Carlo Analysis
        # -------------------------
        with tabs[1]:
            st.subheader("Monte Carlo Null Distribution Analysis")
            
            mc_result = st.session_state.monte_carlo_result
            if mc_result is None:
                st.info("Monte Carlo analysis not enabled. Enable it in Page 3 (Simulation Parameters) and re-run.")
            else:
                st.success(f"Completed {mc_result['n_trials']} random selection trials")
                
                # Get strategy's final return for comparison
                strat_final_return = None
                try:
                    _, strat_df = pgp._pick_strategy_df(res)
                    if "Portfolio Value" in strat_df.columns:
                        pv = strat_df["Portfolio Value"].dropna()
                        if len(pv) > 0:
                            strat_final_return = (pv.iloc[-1] / pv.iloc[0] - 1.0) * 100.0
                    elif "Ensemble Mean" in strat_df.columns:
                        pv = strat_df["Ensemble Mean"].dropna()
                        if len(pv) > 0:
                            strat_final_return = (pv.iloc[-1] / pv.iloc[0] - 1.0) * 100.0
                except Exception:
                    pass
                
                # Summary statistics
                final_returns = mc_result["final_returns_pct"]
                if final_returns:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MC Mean Return", f"{np.mean(final_returns):.2f}%")
                        st.metric("MC Median Return", f"{np.median(final_returns):.2f}%")
                    with col2:
                        st.metric("MC Std Dev", f"{np.std(final_returns):.2f}%")
                        st.metric("MC Min Return", f"{np.min(final_returns):.2f}%")
                    with col3:
                        st.metric("MC Max Return", f"{np.max(final_returns):.2f}%")
                        if strat_final_return is not None:
                            # Calculate percentile rank
                            pct_rank = (np.array(final_returns) < strat_final_return).mean() * 100
                            st.metric("Your Strategy Percentile", f"{pct_rank:.1f}%",
                                     help="Percentage of random trials your strategy beat")
                
                # Histogram
                st.markdown("#### Distribution of Random Strategy Returns")
                try:
                    fig_hist = pgp.make_monte_carlo_histogram(
                        mc_result,
                        strategy_return=strat_final_return,
                        title="",
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                except Exception as e:
                    st.error(f"Histogram failed: {e}")
                
                # Show strategy return vs null distribution
                if strat_final_return is not None:
                    st.markdown("#### Your Strategy vs Random Selection")
                    pct_rank = (np.array(final_returns) < strat_final_return).mean() * 100
                    
                    if pct_rank >= 95:
                        st.success(f"🎯 Your strategy return ({strat_final_return:.2f}%) is in the **top {100-pct_rank:.1f}%** of random selections!")
                    elif pct_rank >= 75:
                        st.info(f"📈 Your strategy return ({strat_final_return:.2f}%) beats **{pct_rank:.1f}%** of random selections.")
                    elif pct_rank >= 50:
                        st.warning(f"📊 Your strategy return ({strat_final_return:.2f}%) is **above median** (beats {pct_rank:.1f}% of random).")
                    else:
                        st.error(f"📉 Your strategy return ({strat_final_return:.2f}%) is **below median** (beats only {pct_rank:.1f}% of random).")

        # -------------------------
        # Tab 3: Trades
        # -------------------------
        with tabs[2]:
            st.subheader("Trade Log")
            trades_df = None
            if isinstance(res.get("trades"), pd.DataFrame):
                trades_df = res.get("trades")
            elif isinstance(res.get("trades_with_cash"), pd.DataFrame):
                trades_df = res.get("trades_with_cash")
            elif isinstance(res.get("trades_no_cash"), pd.DataFrame):
                trades_df = res.get("trades_no_cash")

            if trades_df is None or trades_df.empty:
                st.info("No trade log available in this result")
            else:
                st.dataframe(trades_df, use_container_width=True)
                st.download_button(
                    "Download trades.csv",
                    data=trades_df.to_csv(index=False).encode("utf-8"),
                    file_name="trade_log.csv",
                    mime="text/csv",
                )

        # -------------------------
        # Tab 4: Data Health
        # -------------------------
        with tabs[3]:
            st.subheader("Data Health")
            dh = res.get("data_health", None)
            if not isinstance(dh, pd.DataFrame) or dh.empty:
                st.info("No data health report available in this result")
            else:
                st.dataframe(dh, width="stretch")
                st.download_button(
                    "Download data_health.csv",
                    data=dh.to_csv(index=True).encode("utf-8"),
                    file_name="data_health.csv",
                    mime="text/csv",
                )

            # Heatmap-only NaN imputations (if any)
            try:
                # Only defined for non-ensemble primary strategy
                _, df = pgp._pick_strategy_df(res)
                if not pgp._is_ensemble_df(df):
                    hm = ps.compute_holdings_heatmap(sim=sim, strategy_df=df, top_n=int(top_n), clip=hm_clip)
                    imps = hm.get("imputation_points", []) or []
                    if imps:
                        st.markdown("#### Heatmap NaN imputations")
                        imp_df = pd.DataFrame(imps)
                        st.dataframe(imp_df, width="stretch")
                        counts = hm.get("imputation_counts", {}) or {}
                        if counts:
                            st.markdown("**Imputation counts by ticker**")
                            st.dataframe(pd.Series(counts, name="n_imputations").sort_values(ascending=False).to_frame(), width="stretch")
            except Exception:
                pass

        # Export result (as parquet blobs inside a pickle)

        st.download_button(
            "Download result (parquet bundle)",
            data=result_to_parquet_bytes(res),
            file_name="simulation_result_parquet_bundle.pkl",
            mime="application/octet-stream",
        )


# -----------------------------
# Page 5: Compare & Export
# -----------------------------

if page.startswith("5"):
    st.title("5) Compare & Export")

    runs = st.session_state.runs

    # Backfill: if the user has a current result loaded in session but no run history,
    # create a single run record so this page is usable without re-running.
    if (not runs) and (isinstance(st.session_state.get("result"), dict)):
        st.session_state.runs.append(
            {
                "run_id": str(uuid.uuid4()),
                "name": "Run 1",
                "sim_cfg": asdict(st.session_state.sim_cfg) if st.session_state.sim_cfg else None,
                "event_cfg": asdict(st.session_state.event_cfg) if st.session_state.event_cfg else None,
                "analysis_summary": st.session_state.get("analysis_summary"),
                "result": st.session_state.get("result"),
                "ok": True,
                "error": None,
                "traceback": None,
            }
        )
        runs = st.session_state.runs

    if not runs:
        st.info("No runs yet. Run at least one simulation in Page 4.")
        st.stop()

    st.subheader("Run history")

    ok_runs = [
        r for r in (runs or [])
        if bool(r.get("ok", False)) and isinstance(r.get("result"), dict)
    ]
    if not ok_runs:
        st.info("No successful runs yet. Run at least one simulation in Page 4.")
        # Still allow config export if there are failed attempts
        st.stop()

    run_names = [str(r.get("name", "Run")) for r in ok_runs]
    pick = st.multiselect(
        "Select runs to compare",
        options=run_names,
        default=run_names,  # include all so legend toggles work for every stored run
    )

    picked = [r for r in ok_runs if str(r.get("name", "")) in set(pick)]
    if not picked:
        st.warning("Select at least one run.")
        st.stop()

    st.subheader("Overlay portfolio curves (interactive)")

    c1, c2 = st.columns([1, 1])
    with c1:
        y_as = st.selectbox("Y-axis", ["pct", "value", "index", "log"], index=0)
    with c2:
        show_bench = st.checkbox("Benchmarks", value=True)

    try:
        fig = pgp.make_comparison_figure(
            runs=picked,
            y_as=y_as,
            show_benchmarks=bool(show_bench),
            title="",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Interactive comparison plot failed: {e}")

    st.subheader("Download performance summaries (runs + benchmarks)")
    summary_rows: List[pd.DataFrame] = []
    for r in picked:
        summ = r.get("analysis_summary", None)
        if not isinstance(summ, pd.DataFrame):
            try:
                res = r.get("result", None)
                if isinstance(res, dict):
                    summ = analyze_result(res)
                    r["analysis_summary"] = summ
            except Exception:
                summ = None

        if isinstance(summ, pd.DataFrame) and not summ.empty:
            tmp = summ.copy().reset_index().rename(columns={"Series": "Series"})
            tmp.insert(0, "run_id", str(r.get("run_id", "")))
            tmp.insert(1, "run_name", str(r.get("name", "Run")))
            summary_rows.append(tmp)

    if summary_rows:
        summary_all = pd.concat(summary_rows, axis=0, ignore_index=True)
        st.dataframe(summary_all, width="stretch")
        st.download_button(
            "Download performance_summaries.csv",
            data=summary_all.to_csv(index=False).encode("utf-8"),
            file_name="performance_summaries.csv",
            mime="text/csv",
        )
    else:
        st.info("No performance summaries available for the selected runs.")

    st.subheader("Export run configs")
    export_payload = {
        "runs": [
            {
                "run_id": r.get("run_id"),
                "name": r["name"],
                "sim_cfg": r.get("sim_cfg"),
                "event_cfg": r.get("event_cfg"),
            }
            for r in picked
        ]
    }
    st.download_button(
        "Download configs.json",
        data=json.dumps(export_payload, indent=2, default=str).encode("utf-8"),
        file_name="run_configs.json",
        mime="application/json",
    )
