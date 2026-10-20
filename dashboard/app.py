"""Streamlit dashboard for hedgeFund regime scans. Reads cached runs; never touches the network.

Launch with ``uv run hedgefund dashboard`` or ``uv run streamlit run dashboard/app.py``.
Every number on the page comes from ``./jobs/<run_id>/`` (the scan's result JSON) or, when the
sidebar controls differ from the scan's parameters, from the parquet price cache the scan wrote.
No data source module is imported here, so the app cannot open a connection even by accident.

Backtests are historical, not forward-looking.
"""

import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from hedgefund.dashboard.data import (
    STATES,
    Controls,
    LoadedRun,
    SymbolView,
    compute_symbol_view,
    control_options,
    controls_from_run,
    convergence_distance,
    list_run_ids,
    load_run,
    nstep_sequence,
    resolve_jobs_dir,
)
from hedgefund.report.html import CURVE_COLOURS, CURVE_LABELS, STATE_FILLS, STATE_STROKES

# ``st.set_page_config`` must be the first Streamlit call in the script.
st.set_page_config(page_title="hedgeFund regimes", layout="wide")

# Seconds between frames of the P**n animation: fast enough to feel like motion, slow enough
# that the eye can follow the rows converging (60 frames at most, so under ten seconds).
ANIMATION_FRAME_SECONDS = 0.15
# Null paths re-simulated on every control change. 200 keeps a re-render under a few
# seconds. The band always uses the scan's own seed, so at the run's full count it is the
# same draw the JSON null percentile ranks against; with fewer paths it is the leading
# subset of that draw, a visual guide rather than the ranking statistic.
NULL_PATHS_DEFAULT = 200
NULL_PATHS_MIN = 20  # below this the 2.5 / 97.5 percentiles are not defined by distinct paths
SAME_DRAW_TEXT = "the same draw the JSON null percentile ranks against"
# Upper bound on the P**n slider: twice the mixing time shows the convergence completing,
# with a floor so a fast-mixing chain still has a visible path and a cap so the
# animation stays short.
NSTEP_MIN_MAX = 10
NSTEP_MAX_CAP = 60
NSTEP_FALLBACK = 20  # used when the chain has no finite mixing time
# Equity curves plotted in view 5, in legend order; the others stay in the JSON.
EQUITY_CURVE_KEYS: tuple[str, ...] = ("strategy_net", "strategy_gross", "buy_and_hold")
NULL_BAND_LABEL = "random-sign null, 2.5-97.5 pct, net of costs"
NULL_BAND_COLOUR = "#b3cde3"  # light blue, distinct from every curve colour
BADGE_FROM_JSON = "From the scan's result JSON"
BADGE_RECOMPUTED = "Recomputed from cached prices for these controls"
# Heatmap shading: white at p = 0 to this blue at p = 1, text flips to white above the cutoff.
HEAT_RANGE = ("#ffffff", "#1f77b4")
HEAT_TEXT_FLIP = 0.6
HEATMAP_CELL_PX = 90
RIBBON_HEIGHT_PX = 320
EQUITY_HEIGHT_PX = 360
SMALL_CHART_HEIGHT_PX = 220


# ----------------------------------------------------------------------------------------------
# Pure helpers (module level so tests can import them without running the app)
# ----------------------------------------------------------------------------------------------


def ribbon_segments(labels: pd.Series) -> pd.DataFrame:
    """Collapse a date-indexed series of state codes into contiguous runs.

    Returns columns ``start``, ``end``, ``code``, ``state``. ``end`` is the first date of the
    next run (so rects tile without gaps); the final run ends on the last date of the series.
    An empty input returns an empty frame with the same columns.
    """
    columns = ["start", "end", "code", "state"]
    if labels.empty:
        return pd.DataFrame(columns=columns)
    codes = labels.to_numpy().astype(int)
    dates = labels.index
    # A new run starts wherever the code differs from the previous bar.
    boundaries = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[boundaries[1:], len(codes) - 1]
    return pd.DataFrame(
        {
            "start": dates[boundaries],
            "end": dates[ends],
            "code": codes[boundaries],
            "state": [STATES[c] for c in codes[boundaries]],
        }
    )


def heatmap_frame(
    matrix: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> pd.DataFrame:
    """Long-form frame for a 3x3 transition heatmap.

    One row per cell with ``from_state``, ``to_state`` (STATES order), ``p``, ``lower``,
    ``upper`` and a display ``label`` of the form ``0.85 [0.80, 0.90]`` (just ``0.85`` when
    no interval is given).
    """
    m = np.asarray(matrix, dtype=float)
    lo = None if lower is None else np.asarray(lower, dtype=float)
    hi = None if upper is None else np.asarray(upper, dtype=float)
    rows = []
    for i, from_state in enumerate(STATES):
        for j, to_state in enumerate(STATES):
            p = float(m[i, j])
            cell_lo = None if lo is None else float(lo[i, j])
            cell_hi = None if hi is None else float(hi[i, j])
            label = f"{p:.2f}"
            if cell_lo is not None and cell_hi is not None:
                label = f"{p:.2f} [{cell_lo:.2f}, {cell_hi:.2f}]"
            rows.append(
                {
                    "from_state": from_state,
                    "to_state": to_state,
                    "p": p,
                    "lower": cell_lo,
                    "upper": cell_hi,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def band_frame(
    curves: dict[str, pd.Series],
    lower: pd.Series | None,
    upper: pd.Series | None,
) -> pd.DataFrame:
    """Wide frame for the equity view.

    Columns: ``date``, one per curve, ``null_lower``, ``null_upper``.

    Curves are aligned on the union of their indices; the null band is reindexed onto that
    index (missing values stay NaN so Altair leaves a gap rather than inventing a value).
    Returns an empty frame when there are no curves.
    """
    if not curves:
        return pd.DataFrame(columns=["date", "null_lower", "null_upper"])
    frame = pd.DataFrame(curves)
    frame["null_lower"] = lower.reindex(frame.index) if lower is not None else np.nan
    frame["null_upper"] = upper.reindex(frame.index) if upper is not None else np.nan
    frame.index.name = "date"
    return frame.reset_index()


def stationary_frame(
    stationary: np.ndarray, lower: np.ndarray | None, upper: np.ndarray | None
) -> pd.DataFrame:
    """One row per state with the stationary probability and its interval."""
    pi = np.asarray(stationary, dtype=float)
    lo = pi if lower is None else np.asarray(lower, dtype=float)
    hi = pi if upper is None else np.asarray(upper, dtype=float)
    return pd.DataFrame({"state": list(STATES), "p": pi, "lower": lo, "upper": hi})


def nstep_max(mixing_time: int | None) -> int:
    """Slider ceiling for the P**n view (see the NSTEP_* constants)."""
    horizon = 2 * mixing_time if mixing_time else NSTEP_FALLBACK
    return min(NSTEP_MAX_CAP, max(NSTEP_MIN_MAX, horizon))


def with_default(options: list, default: object) -> list:
    """Return ``options`` with ``default`` present (prepended when the run used a value the
    config does not list) so the selectbox can always show the run's own parameter."""
    opts = list(options)
    if default not in opts:
        opts.insert(0, default)
    return opts


# ----------------------------------------------------------------------------------------------
# Chart builders
# ----------------------------------------------------------------------------------------------


def _state_colour_scale(fills: dict[str, str]) -> alt.Scale:
    return alt.Scale(domain=list(STATES), range=[fills[s] for s in STATES])


def ribbon_chart(close: pd.Series, labels: pd.Series, *, log_scale: bool) -> alt.LayerChart:
    """Close line over one coloured rect per regime run."""
    segments = ribbon_segments(labels)
    price = pd.DataFrame({"date": close.index, "close": close.to_numpy()})
    rects = (
        alt.Chart(segments)
        .mark_rect(opacity=0.8)
        .encode(
            x=alt.X("start:T", title="Date"),
            x2="end:T",
            color=alt.Color(
                "state:N", scale=_state_colour_scale(STATE_FILLS), legend=alt.Legend(title="Regime")
            ),
            tooltip=["state:N", "start:T", "end:T"],
        )
    )
    y_scale = alt.Scale(type="log") if log_scale else alt.Scale(zero=False)
    line = (
        alt.Chart(price)
        .mark_line(color="#1a1a1a", strokeWidth=1.2)
        .encode(x="date:T", y=alt.Y("close:Q", title="Close", scale=y_scale))
    )
    return (rects + line).properties(height=RIBBON_HEIGHT_PX).resolve_scale(color="independent")


def heatmap_chart(frame: pd.DataFrame, *, title: str) -> alt.LayerChart:
    """3x3 rect heatmap with a text label per cell."""
    base = alt.Chart(frame).encode(
        x=alt.X("to_state:N", sort=list(STATES), title="To"),
        y=alt.Y("from_state:N", sort=list(STATES), title="From"),
    )
    rects = base.mark_rect().encode(
        color=alt.Color("p:Q", scale=alt.Scale(domain=[0, 1], range=list(HEAT_RANGE)), legend=None),
        tooltip=["from_state:N", "to_state:N", "label:N"],
    )
    text = base.mark_text(fontSize=11).encode(
        text="label:N",
        color=alt.condition(alt.datum.p > HEAT_TEXT_FLIP, alt.value("white"), alt.value("black")),
    )
    size = HEATMAP_CELL_PX * len(STATES)
    return (rects + text).properties(width=size, height=size, title=title)


def stationary_row_chart(stationary: np.ndarray) -> alt.LayerChart:
    """1x3 heatmap of the stationary distribution, aligned with the P**n heatmap columns."""
    frame = pd.DataFrame(
        {
            "to_state": list(STATES),
            "row": ["stationary"] * len(STATES),
            "p": np.asarray(stationary, dtype=float),
        }
    )
    frame["label"] = frame["p"].map(lambda p: f"{p:.2f}")
    base = alt.Chart(frame).encode(
        x=alt.X("to_state:N", sort=list(STATES), title="State"),
        y=alt.Y("row:N", title=None),
    )
    rects = base.mark_rect().encode(
        color=alt.Color("p:Q", scale=alt.Scale(domain=[0, 1], range=list(HEAT_RANGE)), legend=None)
    )
    text = base.mark_text(fontSize=11).encode(
        text="label:N",
        color=alt.condition(alt.datum.p > HEAT_TEXT_FLIP, alt.value("white"), alt.value("black")),
    )
    return (rects + text).properties(
        width=HEATMAP_CELL_PX * len(STATES), height=HEATMAP_CELL_PX, title="Stationary π"
    )


def stationary_chart(frame: pd.DataFrame) -> alt.LayerChart:
    """Bar chart of the stationary mix with bootstrap error bars."""
    base = alt.Chart(frame).encode(x=alt.X("state:N", sort=list(STATES), title="State"))
    bars = base.mark_bar().encode(
        y=alt.Y("p:Q", title="Long-run probability", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("state:N", scale=_state_colour_scale(STATE_STROKES), legend=None),
        tooltip=["state:N", alt.Tooltip("p:Q", format=".3f")],
    )
    errors = base.mark_rule(strokeWidth=2).encode(y="lower:Q", y2="upper:Q")
    return (bars + errors).properties(height=SMALL_CHART_HEIGHT_PX)


def convergence_chart(distances: np.ndarray) -> alt.Chart:
    """Max-row total-variation distance from P**n to the stationary distribution."""
    frame = pd.DataFrame({"n": np.arange(len(distances)), "distance": distances})
    return (
        alt.Chart(frame)
        .mark_line(point=True, color=CURVE_COLOURS["strategy_net"])
        .encode(
            x=alt.X("n:Q", title="n (steps)"),
            y=alt.Y("distance:Q", title="max row TVD to π"),
            tooltip=["n:Q", alt.Tooltip("distance:Q", format=".4f")],
        )
        .properties(height=SMALL_CHART_HEIGHT_PX)
    )


def equity_chart(frame: pd.DataFrame) -> alt.LayerChart:
    """Equity lines over the shaded null band."""
    keys = [k for k in EQUITY_CURVE_KEYS if k in frame.columns]
    labels = [CURVE_LABELS[k] for k in keys]
    colours = [CURVE_COLOURS[k] for k in keys]
    lines_frame = frame[["date", *keys]].rename(columns=dict(zip(keys, labels, strict=True)))
    band = (
        alt.Chart(frame.assign(band=NULL_BAND_LABEL))
        .mark_area(opacity=0.5)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("null_lower:Q", title="Equity (start = 1.0)"),
            y2="null_upper:Q",
            color=alt.Color(
                "band:N",
                scale=alt.Scale(domain=[NULL_BAND_LABEL], range=[NULL_BAND_COLOUR]),
                legend=alt.Legend(title="Null band"),
            ),
        )
    )
    lines = (
        alt.Chart(lines_frame)
        .transform_fold(labels, as_=["series", "equity"])
        .mark_line()
        .encode(
            x="date:T",
            y="equity:Q",
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=labels, range=colours),
                legend=alt.Legend(title="Curve"),
            ),
            tooltip=["date:T", "series:N", alt.Tooltip("equity:Q", format=".3f")],
        )
    )
    return (band + lines).properties(height=EQUITY_HEIGHT_PX).resolve_scale(color="independent")


# ----------------------------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------------------------


# ``cache_resource`` rather than ``cache_data``: a LoadedRun holds DataFrames, the manifest
# dataclass and per-symbol result dicts. ``cache_data`` would pickle and copy all of that on
# every rerun; the run is read-only once written, so sharing one object is safe and cheaper.
@st.cache_resource(show_spinner="Loading cached run…")
def cached_run(run_id: str, jobs_dir: str) -> LoadedRun:
    return load_run(run_id, Path(jobs_dir))


# Keyed on primitives (not the Controls dataclass) so Streamlit's hasher has nothing exotic
# to digest. ``seed`` is the scan's seed (part of the key so a rerun with another seed is
# not served a stale band).
@st.cache_resource(show_spinner="Computing view for these controls…")
def cached_view(
    run_id: str,
    jobs_dir: str,
    symbol: str,
    window: int,
    threshold: float,
    refit: str,
    rolling_bars: int,
    cost_profile: str,
    return_type: str,
    null_paths: int,
    seed: int,
) -> SymbolView:
    run = cached_run(run_id, jobs_dir)
    controls = Controls(
        symbol=symbol,
        window=window,
        threshold=threshold,
        refit=refit,
        rolling_bars=rolling_bars,
        cost_profile=cost_profile,
        return_type=return_type,
    )
    return compute_symbol_view(run, controls, null_paths=null_paths, seed=seed)


# ----------------------------------------------------------------------------------------------
# Page sections
# ----------------------------------------------------------------------------------------------


def run_seed(run: LoadedRun) -> int:
    """The seed the scan used for its null paths (``manifest.params["seed"]``)."""
    return int(run.manifest.params["seed"])


def json_null_percentile(result: dict) -> float | None:
    """``backtest.null.random_sign.percentile_net`` from the result JSON, or None when absent."""
    node: object = result
    for key in ("backtest", "null", "random_sign", "percentile_net"):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return None if node is None else float(node)


def null_band_caption(null_paths: int, run_paths: int, seed: int) -> str:
    """Caption naming how many of the scan's paths and which seed drew the band."""
    text = f"{NULL_BAND_LABEL}; {null_paths} of the scan's {run_paths} paths, seed {seed}"
    if null_paths == run_paths:
        text += f"; {SAME_DRAW_TEXT}"
    return text


def run_null_paths(run: LoadedRun, symbol: str) -> int:
    """Null paths the scan simulated, from the manifest or the symbol's result JSON."""
    from_manifest = run.manifest.params.get("null_paths")
    if from_manifest is not None:
        return int(from_manifest)
    result = run.results.get(symbol) or {}
    stats = result.get("statistics_params") or {}
    return int(stats.get("null_paths", NULL_PATHS_MIN))


def sidebar(jobs_dir: Path, run_ids: list[str]) -> tuple[LoadedRun, Controls, int]:
    """Render the sidebar widgets and return the run, the chosen controls and null paths."""
    st.sidebar.header("Controls")
    run_id = st.sidebar.selectbox("Run", run_ids, index=0, key="run_id")
    run = cached_run(run_id, str(jobs_dir))

    symbols = list(run.symbols)
    symbol = st.sidebar.selectbox("Symbol", symbols, index=0, key="symbol")
    defaults = controls_from_run(run, symbol)
    options = control_options(run.config, run)

    windows = with_default(options["windows"], defaults.window)
    window = st.sidebar.selectbox(
        "Window (trading days)", windows, index=windows.index(defaults.window), key="window"
    )
    thresholds = with_default(options["thresholds"], defaults.threshold)
    threshold = st.sidebar.selectbox(
        "Threshold",
        thresholds,
        index=thresholds.index(defaults.threshold),
        key="threshold",
        format_func=lambda t: f"±{t:.1%}",
    )
    refits = list(options["refits"])
    refit = st.sidebar.radio(
        "Refit mode", refits, index=refits.index(defaults.refit), key="refit", horizontal=True
    )
    rolling_bars = defaults.rolling_bars or int(options["rolling_bars"])
    if refit == "rolling":
        rolling_bars = int(
            st.sidebar.number_input(
                "Rolling bars", min_value=1, value=int(rolling_bars), step=1, key="rolling_bars"
            )
        )
    profiles = with_default(options["cost_profiles"], defaults.cost_profile)
    cost_profile = st.sidebar.selectbox(
        "Cost profile", profiles, index=profiles.index(defaults.cost_profile), key="cost_profile"
    )

    max_paths = run_null_paths(run, symbol)
    if max_paths > NULL_PATHS_MIN:
        null_paths = int(
            st.sidebar.slider(
                "Null paths",
                min_value=NULL_PATHS_MIN,
                max_value=max_paths,
                value=min(NULL_PATHS_DEFAULT, max_paths),
                key="null_paths",
            )
        )
    else:
        null_paths = max_paths
        st.sidebar.caption(f"Null paths: {null_paths} (the scan's count)")

    controls = Controls(
        symbol=symbol,
        window=int(window),
        threshold=float(threshold),
        refit=str(refit),
        rolling_bars=int(rolling_bars),
        cost_profile=str(cost_profile),
        return_type=defaults.return_type,
    )
    return run, controls, null_paths


def header(view: SymbolView) -> None:
    st.title(f"{view.symbol}: Markov regime dashboard")
    cols = st.columns(4)
    cols[0].metric("Current regime", STATES[view.current_state])
    cols[1].metric("Signal", f"{view.signal:+.2f}")
    mixing = "n/a" if view.mixing_time is None else f"{view.mixing_time} days"
    cols[2].metric("Mixing time", mixing)
    cols[3].metric("Null paths (band)", str(view.null_paths))
    if view.recomputed:
        st.info(BADGE_RECOMPUTED)
    else:
        st.success(BADGE_FROM_JSON)
    probs = ", ".join(
        f"{s} {p:.0%}" for s, p in zip(STATES, view.next_state_probabilities, strict=True)
    )
    st.caption(f"Next-step probabilities from the current state: {probs}.")


def view_ribbon(view: SymbolView) -> None:
    st.subheader("1. Regime ribbon")
    log_scale = st.checkbox("Log price scale", value=False, key="log_scale")
    st.altair_chart(ribbon_chart(view.close, view.labels, log_scale=log_scale), width="stretch")
    st.caption(
        "Each bar is coloured by the regime assigned on that day using only data up to that "
        "day (no lookahead)."
    )


def view_matrix(view: SymbolView) -> None:
    st.subheader("2. Transition matrix with confidence intervals")
    frame = heatmap_frame(view.matrix, view.matrix_lower, view.matrix_upper)
    st.altair_chart(heatmap_chart(frame, title="P(from → to)"))
    ci = view.result.get("confidence_intervals") or {}
    level = ci.get("ci")
    level_text = f"{float(level):.0%}" if level is not None else "n/a"
    st.caption(
        f"Cells show p [lower, upper]: {level_text} block-bootstrap intervals "
        f"(block_days={ci.get('block_days', 'n/a')}, n_resamples={ci.get('n_resamples', 'n/a')})."
    )


def view_stationary(view: SymbolView) -> None:
    st.subheader("3. Stationary mix")
    frame = stationary_frame(view.stationary, view.stationary_lower, view.stationary_upper)
    st.altair_chart(stationary_chart(frame), width="stretch")
    st.caption("Long-run mix implied by the current matrix, not a forecast.")


def view_convergence(view: SymbolView) -> None:
    st.subheader("4. P**n convergence")
    max_n = nstep_max(view.mixing_time)
    sequence = nstep_sequence(view.matrix, max_n)
    distances = convergence_distance(view.matrix, max_n)
    n = st.slider("n", min_value=0, max_value=max_n, value=0, step=1, key="nstep")
    play = st.button("Play", key="play")

    left, right = st.columns([2, 1])
    matrix_slot = left.empty()
    pi_slot = right.empty()
    label_slot = st.empty()

    def render(step: int) -> None:
        matrix_slot.altair_chart(heatmap_chart(heatmap_frame(sequence[step]), title=f"P**{step}"))
        pi_slot.altair_chart(stationary_row_chart(view.stationary))
        label_slot.caption(f"n = {step}: max row TVD to π = {distances[step]:.4f}")

    if play:
        # Animate from 0 to max_n and leave the final frame on screen.
        for step in range(max_n + 1):
            render(step)
            time.sleep(ANIMATION_FRAME_SECONDS)
    else:
        render(n)
    st.altair_chart(convergence_chart(distances), width="stretch")


def view_equity(view: SymbolView, *, run_paths: int, seed: int) -> None:
    st.subheader("5. Walk-forward equity")
    if not view.curves:
        st.warning("history too short for the walk-forward")
        return
    chart_col, metric_col = st.columns([4, 1])
    frame = band_frame(view.curves, view.null_lower, view.null_upper)
    chart_col.altair_chart(equity_chart(frame), width="stretch")
    percentile = json_null_percentile(view.result)
    metric_col.metric(
        "Random-sign null percentile (net, JSON)",
        # Named after its null model: the scan table's null_percentile is the permutation null.
        "n/a" if percentile is None else f"{percentile:.1f}",
        help=f"Rank of the net strategy among the scan's {run_paths} random-sign null "
        f"paths, from the result JSON (not recomputed here).",
    )
    cm = view.cost_model
    st.caption(
        f"Strategy is net of the '{view.controls.cost_profile}' cost profile: "
        f"fee {cm.fee_bps:g} bps + slippage {cm.slippage_bps:g} bps per side, "
        f"borrow {cm.borrow_bps:g} bps/yr, funding {cm.funding_bps:g} bps/yr. "
        f"Shaded band: {null_band_caption(view.null_paths, run_paths, seed)}."
    )


def footer(run: LoadedRun) -> None:
    st.divider()
    manifest = run.manifest
    st.caption(manifest.disclaimer)
    sources = ", ".join(
        f"{sym}: {(manifest.data.get(sym) or {}).get('source', 'n/a')}" for sym in run.symbols
    )
    st.caption(
        f"Run {manifest.run_id} · created {manifest.created_at} · universe {manifest.universe} "
        f"· cost profile {manifest.cost_profile} · sources: {sources}"
    )


def main() -> None:
    jobs_dir = resolve_jobs_dir()
    run_ids = list_run_ids(jobs_dir)
    if not run_ids:
        st.warning(
            f"No cached runs under {jobs_dir}. Run `uv run hedgefund scan --universe us_etfs` "
            "first, then reopen the dashboard."
        )
        st.stop()

    run, controls, null_paths = sidebar(jobs_dir, run_ids)
    seed = run_seed(run)
    view = cached_view(
        run.run_id,
        str(run.jobs_dir),
        controls.symbol,
        controls.window,
        controls.threshold,
        controls.refit,
        controls.rolling_bars,
        controls.cost_profile,
        controls.return_type,
        null_paths,
        seed,
    )

    header(view)
    view_ribbon(view)
    left, right = st.columns(2)
    with left:
        view_matrix(view)
    with right:
        view_stationary(view)
    view_convergence(view)
    view_equity(view, run_paths=run_null_paths(run, controls.symbol), seed=seed)
    footer(run)


# Streamlit (``streamlit run`` and ``AppTest``) executes the script as a module named
# ``__main__``, so this guard runs the app there but not when a test imports the file through
# ``importlib`` under another name to reach the pure helpers.
if __name__ == "__main__":
    main()
