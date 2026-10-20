"""Static HTML report for one ``analyze`` result and the scan index (plan.md Phase 5).

Every chart is an inline SVG string built from the standard library so the pages are
self-contained files with no JavaScript, no external assets and no plotting
dependency. One document per symbol (``render_symbol_report``) and one ranked index
per run (``render_scan_index``); ``write_run_html`` writes both under
``<run_dir>/html``. All text that comes from data (symbols, dates, error messages,
sources) passes through ``html.escape``.

Honesty rules the pages follow (``.claude/rules/no-lookahead.md``):

- The regime ribbon colours each bar with the label ``label_regimes`` assigned at that
  bar, which only uses closes at or before it. No hindsight is applied.
- ``equity_curves`` re-runs the same walk-forward the result was produced with, so the
  curves are the actual no-lookahead positions compounded day by day, net of the same
  cost model as the ``backtest`` block. Benchmarks pay the same costs (a constant
  position pays its entry cost on day one), matching the Phase 3 convention.
- The stride diagonal is shown next to the overlapping one, the labeller-induced null
  excess next to the raw persistence, and the permutation p-values next to the Sharpe.
- The stationary distribution is labelled as the long-run mix implied by the current
  matrix, not a forecast, and every page carries the disclaimer.
"""

import html as html_lib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from hedgefund.backtest.benchmarks import (
    DEFAULT_EXPOSURE,
    buy_and_hold,
    constant_exposure,
    long_only_positions,
)
from hedgefund.backtest.costs import CostModel, apply_costs
from hedgefund.backtest.walk_forward import run_walk_forward
from hedgefund.regime.labels import STATES, label_regimes
from hedgefund.report.backtest_report import DISCLAIMER

__all__ = [
    "CURVE_KEYS",
    "DISCLAIMER",
    "ci_bars_svg",
    "equity_curve_svg",
    "equity_curves",
    "matrix_heatmap_svg",
    "regime_ribbon_svg",
    "regime_series",
    "render_scan_index",
    "render_symbol_report",
    "write_run_html",
]

STATE_KEYS: tuple[str, ...] = ("bear", "sideways", "bull")  # JSON keys in STATES order
N_STATES = len(STATES)

# Keys of ``equity_curves``, in legend order. ``constant_50`` is the Phase 3
# ``DEFAULT_EXPOSURE`` benchmark; the name is frozen by the backtest block.
CURVE_KEYS: tuple[str, ...] = (
    "strategy_gross",
    "strategy_net",
    "buy_and_hold",
    "constant_50",
    "long_only",
)
CURVE_LABELS: dict[str, str] = {
    "strategy_gross": "Strategy (gross)",
    "strategy_net": "Strategy (net of costs)",
    "buy_and_hold": "Buy and hold",
    "constant_50": f"Constant {DEFAULT_EXPOSURE:.0%} exposure",
    "long_only": "Signal long-only",
}
# Colour-blind-safe palette (Okabe-Ito) for the five curves.
CURVE_COLOURS: dict[str, str] = {
    "strategy_gross": "#999999",
    "strategy_net": "#0072b2",
    "buy_and_hold": "#000000",
    "constant_50": "#e69f00",
    "long_only": "#009e73",
}
# Band colours for the regime ribbon, in STATES order; light so the price line stays legible.
STATE_FILLS: dict[str, str] = {"Bear": "#f4c7c3", "Sideways": "#e6e6e6", "Bull": "#c6e9c0"}
STATE_STROKES: dict[str, str] = {"Bear": "#c0392b", "Sideways": "#7f8c8d", "Bull": "#1e8449"}
PRICE_STROKE = "#1a1a1a"

# Heatmap shading interpolates from white (p = 0) to this blue (p = 1).
HEAT_RGB = (31, 119, 180)
HEAT_TEXT_FLIP = 0.6  # above this probability the cell is dark enough for white text

# SVG geometry in pixels. One set of constants so every chart lines up on the page.
CHART_WIDTH = 760
CHART_HEIGHT = 260
MARGIN_LEFT = 64  # room for a 6-character y label
MARGIN_RIGHT = 16
MARGIN_TOP = 16
MARGIN_BOTTOM = 32  # room for the date labels
LEGEND_ROW_HEIGHT = 22
LEGEND_SWATCH = 28
LEGEND_ITEM_WIDTH = 190  # five legend items fit in one CHART_WIDTH row
HEAT_CELL_WIDTH = 120
HEAT_CELL_HEIGHT = 64
HEAT_HEADER_WIDTH = 84
HEAT_HEADER_HEIGHT = 28
BAR_CHART_WIDTH = 400
BAR_WIDTH = 60
WHISKER_CAP = 10  # half-width of the whisker end caps
N_Y_TICKS = 5  # gridlines per chart: enough to read a level, few enough to stay clean
FONT = "font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
FONT_SIZE = 12
COORD_DECIMALS = 1  # SVG coordinates rounded to 0.1 px keep the files small
# A curve must stay strictly positive for the log scale; an equity that compounds to
# zero or below is clipped here (a position cannot lose more than 100% in this repo).
MIN_LOG_EQUITY = 1e-6
NOT_AVAILABLE = "n/a"
# Everything other than letters, digits, dot, underscore and dash in a symbol becomes a
# dash in the file name, so "BTC/USD" -> "BTC-USD" and nothing escapes the html directory.
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]")

STYLE = """
body { max-width: 980px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a;
       font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; line-height: 1.45; }
h1 { font-size: 1.6rem; margin-bottom: 0.2rem; }
h2 { font-size: 1.2rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.2rem; }
table { border-collapse: collapse; margin: 0.5rem 0; }
th, td { border: 1px solid #ddd; padding: 0.25rem 0.6rem; text-align: right; }
th { background: #f5f5f5; }
td.text, th.text { text-align: left; }
tr.error td { color: #a33; }
.note { color: #555; font-size: 0.92rem; }
.disclaimer { margin-top: 2.5rem; padding: 0.6rem; border-top: 2px solid #c0392b;
              font-weight: bold; }
.regime-Bear { color: #c0392b; }
.regime-Sideways { color: #7f8c8d; }
.regime-Bull { color: #1e8449; }
svg { display: block; margin: 0.5rem 0; max-width: 100%; height: auto; }
"""

NO_HINDSIGHT_NOTE = (
    "Each band shows the label known at the bar it covers: the trailing-return labeller "
    "only sees closes at or before that bar, so there is no hindsight in the colouring. "
    "A regime that looks obvious on this chart was only identifiable once the trailing "
    "return had crossed the threshold."
)
STATIONARY_NOTE = (
    "The stationary distribution is the long-run mix implied by the current matrix, "
    "not a forecast. Whiskers are the block-bootstrap interval."
)
HISTORY_TOO_SHORT = (
    "History too short for the walk-forward: the equity curves need min_train plus "
    "30 labelled days, so no out-of-sample equity is shown."
)


# --------------------------------------------------------------------------- helpers


def _esc(value: object) -> str:
    """``html.escape`` of ``str(value)``; the only way text reaches a page."""
    return html_lib.escape(str(value), quote=True)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _finite(value: object) -> float | None:
    """A finite float, or ``None`` for ``None`` / NaN / inf / non-numbers."""
    if not _is_number(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _num(value: object, digits: int = 2) -> str:
    number = _finite(value)
    return NOT_AVAILABLE if number is None else f"{number:.{digits}f}"


def _pct(value: object, digits: int = 1) -> str:
    number = _finite(value)
    return NOT_AVAILABLE if number is None else f"{number:.{digits}%}"


def _range(lower: object, upper: object, digits: int = 2) -> str:
    return f"[{_num(lower, digits)}, {_num(upper, digits)}]"


def _get(node: object, *keys: str) -> object:
    """Nested ``dict`` lookup that returns ``None`` at the first missing key."""
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _px(value: float) -> str:
    return f"{value:.{COORD_DECIMALS}f}"


def _svg_open(width: float, height: float, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_px(width)} {_px(height)}" '
        f'width="{_px(width)}" height="{_px(height)}" role="img" aria-label="{_esc(title)}" '
        f'style="{FONT}; font-size: {FONT_SIZE}px">'
        f"<title>{_esc(title)}</title>"
    )


def _text(x: float, y: float, text: str, anchor: str = "start", extra: str = "") -> str:
    return (
        f'<text x="{_px(x)}" y="{_px(y)}" text-anchor="{anchor}" '
        f'dominant-baseline="middle"{extra}>{_esc(text)}</text>'
    )


def _empty_chart(message: str) -> str:
    """A chart-sized SVG carrying only a message, for inputs with nothing to draw."""
    return (
        _svg_open(CHART_WIDTH, CHART_HEIGHT, message)
        + _text(CHART_WIDTH / 2, CHART_HEIGHT / 2, message, anchor="middle")
        + "</svg>"
    )


def _heat_fill(value: float) -> str:
    """White to ``HEAT_RGB`` as ``value`` goes 0 to 1; non-finite values draw white."""
    p = min(max(value, 0.0), 1.0) if math.isfinite(value) else 0.0
    r, g, b = (round(255 + (c - 255) * p) for c in HEAT_RGB)
    return f"rgb({r},{g},{b})"


def _check_vector(values: dict, name: str) -> None:
    missing = [key for key in STATE_KEYS if key not in values]
    if missing:
        raise ValueError(f"{name} must have keys {STATE_KEYS}, missing {missing}")


def _check_matrix(matrix: object, name: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (N_STATES, N_STATES):
        raise ValueError(f"{name} must be a {N_STATES}x{N_STATES} matrix, got shape {arr.shape}")
    return arr


def _cost_model_from_params(costs: dict) -> CostModel:
    """``CostModel`` from the ``backtest.params.costs`` echo (``per_side_bps`` is derived)."""
    return CostModel(
        fee_bps=float(costs["fee_bps"]),
        slippage_bps=float(costs["slippage_bps"]),
        borrow_bps=float(costs["borrow_bps"]),
        funding_bps=float(costs["funding_bps"]),
    )


def _equity(returns: np.ndarray, start_date: pd.Timestamp, index: pd.Index) -> pd.Series:
    """Cumulative-product equity that starts at exactly 1.0 on ``start_date``."""
    values = np.concatenate(([1.0], np.cumprod(1.0 + np.asarray(returns, dtype=float))))
    full_index = pd.DatetimeIndex([start_date]).append(pd.DatetimeIndex(index))
    return pd.Series(values, index=full_index, dtype=float)


# ----------------------------------------------------------------------------- series


def regime_series(close: pd.Series, result: dict) -> pd.Series:
    """``label_regimes`` with the labeller parameters echoed in ``result["params"]``.

    These are the labels the result was computed from (same window, threshold and
    return type), each known at its own bar.
    """
    params = result["params"]
    return label_regimes(
        close.dropna(),
        window=int(params["window"]),
        threshold=float(params["threshold"]),
        return_type=str(params["return_type"]),
    )


def equity_curves(
    close: pd.Series, result: dict, cost_model: CostModel | None = None
) -> dict[str, pd.Series]:
    """Equity curves (start 1.0) of the strategy and its benchmarks, keyed by ``CURVE_KEYS``.

    Re-runs ``run_walk_forward`` with the parameters echoed in
    ``result["backtest"]["params"]`` (``min_train``, ``refit``, ``rolling_bars``,
    ``rule``, ``dead_band``) on the labels from ``result["params"]``, so the curves are
    the no-lookahead positions the result reports. ``strategy_net`` and every benchmark
    are net of ``cost_model`` (``None`` builds one from the echoed ``costs``); a constant
    position pays its entry cost on day one, the Phase 3 benchmark convention.
    ``strategy_gross`` pays nothing. All five series share one index: the first decision
    date (equity 1.0) followed by the dates whose return was earned. Every series is
    empty when the history is too short for the walk-forward.
    """
    params = result["backtest"]["params"]
    model = _cost_model_from_params(params["costs"]) if cost_model is None else cost_model
    close = close.dropna()
    labels = regime_series(close, result)
    wf = run_walk_forward(
        close,
        labels,
        min_train=int(params["min_train"]),
        refit=str(params["refit"]),
        rolling_bars=int(params["rolling_bars"]),
        rule=str(params["rule"]),
        dead_band=float(params["dead_band"]),
    )
    gross = wf.strategy_returns.to_numpy(dtype=float)
    if gross.size == 0:
        empty = pd.Series(np.empty(0, dtype=float), index=pd.DatetimeIndex([]), dtype=float)
        return {key: empty.copy() for key in CURVE_KEYS}

    positions = wf.positions.to_numpy(dtype=float)
    asset = wf.asset_returns.to_numpy(dtype=float)
    ones = np.ones_like(asset)
    half = np.full_like(asset, DEFAULT_EXPOSURE)
    lo_pos = long_only_positions(positions)
    returns = {
        "strategy_gross": gross,
        "strategy_net": apply_costs(gross, positions, model),
        "buy_and_hold": apply_costs(buy_and_hold(asset), ones, model),
        "constant_50": apply_costs(constant_exposure(asset, DEFAULT_EXPOSURE), half, model),
        "long_only": apply_costs(lo_pos * asset, lo_pos, model),
    }
    start = wf.positions.index[0]
    earned = wf.strategy_returns.index
    return {key: _equity(returns[key], start, earned) for key in CURVE_KEYS}


# ----------------------------------------------------------------------------- charts


def matrix_heatmap_svg(
    matrix: list[list[float]],
    lower: list[list[float]] | None = None,
    upper: list[list[float]] | None = None,
) -> str:
    """3x3 transition matrix as a heatmap; rows are "from", columns "to", STATES order.

    Each cell is shaded by its probability and labelled with the value; when ``lower``
    and ``upper`` are both given the cell also shows ``[lo, hi]``. Raises ``ValueError``
    for a matrix or interval that is not 3x3.
    """
    arr = _check_matrix(matrix, "matrix")
    has_ci = lower is not None and upper is not None
    lo = _check_matrix(lower, "lower") if has_ci else None
    hi = _check_matrix(upper, "upper") if has_ci else None
    width = HEAT_HEADER_WIDTH + N_STATES * HEAT_CELL_WIDTH + MARGIN_RIGHT
    height = HEAT_HEADER_HEIGHT + N_STATES * HEAT_CELL_HEIGHT + MARGIN_TOP
    parts = [_svg_open(width, height, "Transition matrix heatmap (rows: from, columns: to)")]
    parts.append(_text(HEAT_HEADER_WIDTH / 2, HEAT_HEADER_HEIGHT / 2, "from \\ to", "middle"))
    for j, state in enumerate(STATES):
        x = HEAT_HEADER_WIDTH + (j + 0.5) * HEAT_CELL_WIDTH
        parts.append(_text(x, HEAT_HEADER_HEIGHT / 2, state, "middle", ' font-weight="bold"'))
    for i, state in enumerate(STATES):
        y_top = HEAT_HEADER_HEIGHT + i * HEAT_CELL_HEIGHT
        parts.append(
            _text(
                HEAT_HEADER_WIDTH - 8,
                y_top + HEAT_CELL_HEIGHT / 2,
                state,
                "end",
                ' font-weight="bold"',
            )
        )
        for j in range(N_STATES):
            value = float(arr[i, j])
            x_left = HEAT_HEADER_WIDTH + j * HEAT_CELL_WIDTH
            colour = "#ffffff" if math.isfinite(value) and value > HEAT_TEXT_FLIP else "#1a1a1a"
            parts.append(
                f'<rect x="{_px(x_left)}" y="{_px(y_top)}" width="{HEAT_CELL_WIDTH}" '
                f'height="{HEAT_CELL_HEIGHT}" fill="{_heat_fill(value)}" stroke="#ffffff"/>'
            )
            cx = x_left + HEAT_CELL_WIDTH / 2
            if has_ci and lo is not None and hi is not None:
                parts.append(
                    _text(
                        cx,
                        y_top + HEAT_CELL_HEIGHT * 0.36,
                        _num(value),
                        "middle",
                        f' fill="{colour}"',
                    )
                )
                parts.append(
                    _text(
                        cx,
                        y_top + HEAT_CELL_HEIGHT * 0.7,
                        _range(lo[i, j], hi[i, j]),
                        "middle",
                        f' fill="{colour}"',
                    )
                )
            else:
                parts.append(
                    _text(
                        cx, y_top + HEAT_CELL_HEIGHT / 2, _num(value), "middle", f' fill="{colour}"'
                    )
                )
    parts.append("</svg>")
    return "".join(parts)


def _plot_box() -> tuple[float, float, float, float]:
    """``(left, top, width, height)`` of the drawing area inside a standard chart."""
    return (
        MARGIN_LEFT,
        MARGIN_TOP,
        CHART_WIDTH - MARGIN_LEFT - MARGIN_RIGHT,
        CHART_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM,
    )


def _runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    """Maximal runs of equal values as ``(start, end_exclusive, value)``."""
    if values.size == 0:
        return []
    change = np.flatnonzero(np.diff(values)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [values.size]))
    return [(int(s), int(e), int(values[s])) for s, e in zip(starts, ends, strict=True)]


def _date_axis(
    parts: list[str], index: pd.Index, left: float, top: float, width: float, height: float
) -> None:
    """First and last date under the plot plus the plot frame."""
    y = top + height + MARGIN_BOTTOM / 2
    parts.append(
        f'<rect x="{_px(left)}" y="{_px(top)}" width="{_px(width)}" height="{_px(height)}" '
        'fill="none" stroke="#999"/>'
    )
    first = pd.Timestamp(index[0]).date()
    last = pd.Timestamp(index[-1]).date()
    parts.append(_text(left, y, str(first), "start"))
    parts.append(_text(left + width, y, str(last), "end"))


def regime_ribbon_svg(close: pd.Series, labels: pd.Series) -> str:
    """Price line over background bands coloured by the regime label at each bar.

    ``labels`` is a ``label_regimes`` output (ints in ``STATES`` order, indexed by the
    bars where a label exists); ``close`` is read on that index. Band ``i`` covers the
    bar it was computed at, so the colouring uses no hindsight. Raises ``ValueError``
    for a label outside ``0..2``.
    """
    if labels.empty:
        return _empty_chart("No labelled bars to draw")
    lab = np.asarray(labels, dtype=int)
    if lab.min() < 0 or lab.max() >= N_STATES:
        raise ValueError(f"labels must be integers in 0..{N_STATES - 1} (order {STATES})")
    prices = close.reindex(labels.index).to_numpy(dtype=float)
    finite = np.isfinite(prices)
    if not finite.any():
        return _empty_chart("No finite close on the labelled bars")

    left, top, width, height = _plot_box()
    n = lab.size
    step = width / max(n - 1, 1)
    p_min, p_max = float(prices[finite].min()), float(prices[finite].max())
    span = (p_max - p_min) or max(abs(p_max), 1.0)  # a flat series still gets a scale

    def x_at(i: int) -> float:
        return left + i * step

    def y_at(price: float) -> float:
        return top + height - (price - p_min) / span * height

    parts = [
        _svg_open(CHART_WIDTH, CHART_HEIGHT, "Close with regime bands (labels known at the bar)")
    ]
    for start, end, state in _runs(lab):
        x0 = x_at(start) - step / 2
        x1 = x_at(end - 1) + step / 2
        x0, x1 = max(x0, left), min(x1, left + width)
        parts.append(
            f'<rect x="{_px(x0)}" y="{_px(top)}" width="{_px(x1 - x0)}" height="{_px(height)}" '
            f'fill="{STATE_FILLS[STATES[state]]}"/>'
        )
    points = " ".join(f"{_px(x_at(i))},{_px(y_at(prices[i]))}" for i in range(n) if finite[i])
    parts.append(
        f'<polyline points="{points}" fill="none" stroke="{PRICE_STROKE}" stroke-width="1.2"/>'
    )
    for tick in np.linspace(p_min, p_max, N_Y_TICKS):
        parts.append(_text(left - 6, y_at(float(tick)), _num(tick), "end"))
    _date_axis(parts, labels.index, left, top, width, height)
    # Legend: one swatch per state in STATES order, top-left inside the margin.
    for k, state in enumerate(STATES):
        x = left + k * 100
        parts.append(
            f'<rect x="{_px(x)}" y="{_px(MARGIN_TOP / 2 - 5)}" width="14" height="10" '
            f'fill="{STATE_FILLS[state]}" stroke="{STATE_STROKES[state]}"/>'
        )
        parts.append(_text(x + 18, MARGIN_TOP / 2, state))
    parts.append("</svg>")
    return "".join(parts)


def equity_curve_svg(curves: dict[str, pd.Series]) -> str:
    """Log-scaled equity lines, one per curve, with a legend row underneath.

    Curves are drawn in ``CURVE_KEYS`` order (unknown keys after them) on the union of
    their indices. Empty curves are skipped; with nothing to draw the chart says so.
    """
    drawn = {key: s for key, s in curves.items() if len(s) > 0}
    if not drawn:
        return _empty_chart(HISTORY_TOO_SHORT)
    order = [key for key in CURVE_KEYS if key in drawn] + sorted(set(drawn) - set(CURVE_KEYS))
    union = drawn[order[0]].index
    for key in order[1:]:
        union = union.union(drawn[key].index)
    union = union.sort_values()
    position = {ts: i for i, ts in enumerate(union)}
    all_values = np.concatenate([np.asarray(drawn[key], dtype=float) for key in order])
    all_values = np.maximum(all_values[np.isfinite(all_values)], MIN_LOG_EQUITY)
    if all_values.size == 0:
        return _empty_chart("No finite equity values to draw")
    log_min, log_max = float(np.log(all_values.min())), float(np.log(all_values.max()))
    if log_max == log_min:  # flat curves: pad so the line sits mid-chart
        log_min, log_max = log_min - 0.05, log_max + 0.05

    left, top, width, height = _plot_box()
    legend_rows = math.ceil(len(order) * LEGEND_ITEM_WIDTH / width)
    total_height = CHART_HEIGHT + legend_rows * LEGEND_ROW_HEIGHT
    step = width / max(len(union) - 1, 1)

    def y_at(value: float) -> float:
        v = math.log(max(value, MIN_LOG_EQUITY))
        return top + height - (v - log_min) / (log_max - log_min) * height

    parts = [_svg_open(CHART_WIDTH, total_height, "Equity curves, log scale, start 1.0")]
    for tick in np.linspace(log_min, log_max, N_Y_TICKS):
        y = y_at(math.exp(float(tick)))
        parts.append(
            f'<line x1="{_px(left)}" x2="{_px(left + width)}" y1="{_px(y)}" y2="{_px(y)}" '
            'stroke="#e5e5e5"/>'
        )
        parts.append(_text(left - 6, y, _num(math.exp(float(tick))), "end"))
    if log_min <= 0.0 <= log_max:
        y1 = y_at(1.0)
        parts.append(
            f'<line x1="{_px(left)}" x2="{_px(left + width)}" y1="{_px(y1)}" y2="{_px(y1)}" '
            'stroke="#999" stroke-dasharray="4 3"/>'
        )
    for key in order:
        series = drawn[key]
        values = np.asarray(series, dtype=float)
        points = " ".join(
            f"{_px(left + position[ts] * step)},{_px(y_at(float(v)))}"
            for ts, v in zip(series.index, values, strict=True)
            if math.isfinite(v)
        )
        colour = CURVE_COLOURS.get(key, PRICE_STROKE)
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="1.4"/>'
        )
    _date_axis(parts, union, left, top, width, height)
    for k, key in enumerate(order):
        row, col = divmod(k, max(int(width // LEGEND_ITEM_WIDTH), 1))
        x = left + col * LEGEND_ITEM_WIDTH
        y = CHART_HEIGHT + (row + 0.5) * LEGEND_ROW_HEIGHT
        colour = CURVE_COLOURS.get(key, PRICE_STROKE)
        parts.append(
            f'<line x1="{_px(x)}" x2="{_px(x + LEGEND_SWATCH)}" y1="{_px(y)}" y2="{_px(y)}" '
            f'stroke="{colour}" stroke-width="2.5"/>'
        )
        parts.append(_text(x + LEGEND_SWATCH + 6, y, CURVE_LABELS.get(key, key)))
    parts.append("</svg>")
    return "".join(parts)


def ci_bars_svg(point: dict, lower: dict, upper: dict) -> str:
    """Bar per state (STATES order) at ``point`` with a whisker from ``lower`` to ``upper``.

    All three are ``{"bear", "sideways", "bull"}`` objects of probabilities; a ``None``
    or non-finite bound draws no whisker for that state. Raises ``ValueError`` when a
    key is missing.
    """
    _check_vector(point, "point")
    _check_vector(lower, "lower")
    _check_vector(upper, "upper")
    left, top = MARGIN_LEFT, MARGIN_TOP
    width = BAR_CHART_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    height = CHART_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    slot = width / N_STATES

    def y_at(prob: float) -> float:
        return top + height - min(max(prob, 0.0), 1.0) * height

    parts = [_svg_open(BAR_CHART_WIDTH, CHART_HEIGHT, "Stationary distribution with bootstrap CI")]
    for tick in np.linspace(0.0, 1.0, N_Y_TICKS):
        y = y_at(float(tick))
        parts.append(
            f'<line x1="{_px(left)}" x2="{_px(left + width)}" y1="{_px(y)}" y2="{_px(y)}" '
            'stroke="#e5e5e5"/>'
        )
        parts.append(_text(left - 6, y, _num(tick), "end"))
    for k, (state, key) in enumerate(zip(STATES, STATE_KEYS, strict=True)):
        cx = left + (k + 0.5) * slot
        value = _finite(point[key])
        if value is not None:
            y = y_at(value)
            parts.append(
                f'<rect x="{_px(cx - BAR_WIDTH / 2)}" y="{_px(y)}" width="{BAR_WIDTH}" '
                f'height="{_px(top + height - y)}" fill="{STATE_FILLS[state]}" '
                f'stroke="{STATE_STROKES[state]}"/>'
            )
            parts.append(_text(cx, y - 9, _num(value), "middle"))
        lo, hi = _finite(lower[key]), _finite(upper[key])
        if lo is not None and hi is not None:
            y_lo, y_hi = y_at(lo), y_at(hi)
            parts.append(
                f'<line x1="{_px(cx)}" x2="{_px(cx)}" y1="{_px(y_lo)}" y2="{_px(y_hi)}" '
                'stroke="#1a1a1a" stroke-width="1.5"/>'
            )
            for y in (y_lo, y_hi):
                parts.append(
                    f'<line x1="{_px(cx - WHISKER_CAP)}" x2="{_px(cx + WHISKER_CAP)}" '
                    f'y1="{_px(y)}" y2="{_px(y)}" stroke="#1a1a1a" stroke-width="1.5"/>'
                )
        parts.append(_text(cx, top + height + MARGIN_BOTTOM / 2, state, "middle"))
    parts.append(
        f'<rect x="{_px(left)}" y="{_px(top)}" width="{_px(width)}" height="{_px(height)}" '
        'fill="none" stroke="#999"/>'
    )
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------------------------ pages


def _page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{STYLE}</style></head>\n"
        f"<body>\n{body}\n"
        f'<p class="disclaimer">{_esc(DISCLAIMER)}</p>\n'
        "</body></html>\n"
    )


def _kv_table(rows: list[tuple[str, object]]) -> str:
    cells = "".join(
        f'<tr><th class="text">{_esc(k)}</th><td class="text">{_esc(v)}</td></tr>' for k, v in rows
    )
    return f"<table>{cells}</table>"


def _state_table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(
        f'<th class="text">{_esc(h)}</th>' if i == 0 else f"<th>{_esc(h)}</th>"
        for i, h in enumerate(header)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="text">{_esc(c)}</td>' if i == 0 else f"<td>{_esc(c)}</td>"
            for i, c in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def _header_section(result: dict) -> str:
    params = result.get("params", {})
    bt_params = _get(result, "backtest", "params") or {}
    costs = bt_params.get("costs", {}) if isinstance(bt_params, dict) else {}
    rows: list[tuple[str, object]] = [
        ("Source", result.get("source", NOT_AVAILABLE)),
        (
            "Dates",
            f"{result.get('date_start', NOT_AVAILABLE)} to {result.get('date_end', NOT_AVAILABLE)}",
        ),
        ("Rows", result.get("rows", NOT_AVAILABLE)),
        ("Labeller window", params.get("window", NOT_AVAILABLE)),
        ("Labeller threshold", _pct(params.get("threshold"))),
        ("Return type", params.get("return_type", NOT_AVAILABLE)),
        ("Minimum training days", params.get("min_train", NOT_AVAILABLE)),
    ]
    if isinstance(bt_params, dict) and bt_params:
        rows += [
            ("Walk-forward refit", bt_params.get("refit", NOT_AVAILABLE)),
            ("Rolling bars", bt_params.get("rolling_bars", NOT_AVAILABLE)),
            ("Position rule", bt_params.get("rule", NOT_AVAILABLE)),
            ("Dead band", _num(bt_params.get("dead_band"), 3)),
        ]
    if costs:
        rows.append(
            (
                "Costs (bps)",
                f"fee {_num(costs.get('fee_bps'), 1)}, "
                f"slippage {_num(costs.get('slippage_bps'), 1)} per side; "
                f"borrow {_num(costs.get('borrow_bps'), 1)}, "
                f"funding {_num(costs.get('funding_bps'), 1)} per year",
            )
        )
    return _kv_table(rows)


def _regime_section(result: dict) -> str:
    regime = str(result.get("current_regime", NOT_AVAILABLE))
    probs = result.get("next_state_probabilities", {})
    rows = [[state, _pct(probs.get(key))] for state, key in zip(STATES, STATE_KEYS, strict=True)]
    signal = _num(result.get("signal"), 3)
    return (
        f'<p>Current regime: <strong class="regime-{_esc(regime)}">{_esc(regime)}</strong>. '
        f"Signal (next-day bull minus bear probability): <strong>{_esc(signal)}</strong>.</p>"
        + _state_table(["Next state", "Probability"], rows)
    )


def _honest_section(result: dict) -> str:
    overlap = result.get("persistence_diagonal", {})
    stride = result.get("stride_diagonal") or {}
    implied = result.get("implied_window_step_diagonal") or {}
    window = _get(result, "params", "window")
    rows = [
        [state, _pct(overlap.get(key)), _pct(implied.get(key)), _pct(stride.get(key))]
        for state, key in zip(STATES, STATE_KEYS, strict=True)
    ]
    parts = [
        "<h3>Persistence: overlapping versus stride</h3>",
        _state_table(
            [
                "State",
                "Overlapping 1-step",
                f"Implied P^{_esc(str(window))}",
                "Stride (non-overlapping)",
            ],
            rows,
        ),
        '<p class="note">The overlapping diagonal counts daily labels that share '
        f"{_esc(str(window))} - 1 bars with their neighbours, so it is inflated by window "
        "autocorrelation. The stride diagonal subsamples every window-th label and is the "
        "honest persistence estimate.</p>",
    ]
    induced = _get(result, "null_percentile", "labeller_induced")
    if isinstance(induced, dict):
        rows = []
        for state, key in zip(STATES, STATE_KEYS, strict=True):
            ci = (induced.get("excess_ci") or {}).get(key) or [None, None]
            rows.append(
                [
                    state,
                    _pct((induced.get("null_diagonal_mean") or {}).get(key)),
                    _pct((induced.get("excess_persistence") or {}).get(key)),
                    f"[{_pct(ci[0])}, {_pct(ci[1])}]",
                ]
            )
        parts += [
            "<h3>Labeller-induced null</h3>",
            _state_table(
                ["State", "Null mean persistence", "Excess over null", "Excess 95% CI"], rows
            ),
            '<p class="note">The same labeller applied to iid price paths '
            f"({_esc(str(induced.get('n_paths', NOT_AVAILABLE)))} paths) produces persistence "
            "by construction; the excess is what the real series adds.</p>",
        ]
    null = _get(result, "backtest", "null")
    if isinstance(null, dict):
        rows = []
        for name, label in (
            ("permutation", "Return permutation (full pipeline)"),
            ("label_shuffle", "Label shuffle"),
            ("random_sign", "Random sign"),
        ):
            block = null.get(name)
            if isinstance(block, dict):
                rows.append(
                    [
                        label,
                        _num(block.get("p_value_gross"), 3),
                        _num(block.get("p_value_net"), 3),
                        _num(block.get("percentile_net"), 1),
                    ]
                )
        parts += [
            "<h3>Sharpe significance</h3>",
            _state_table(
                ["Null model", "p-value (gross)", "p-value (net)", "Net percentile"], rows
            ),
            '<p class="note">One-sided permutation p-values: the share of null paths whose '
            "Sharpe reaches the real one, counting the real path as one more draw.</p>",
        ]
    return "".join(parts)


def _equity_section(result: dict, curves: dict[str, pd.Series]) -> str:
    bt = result.get("backtest") or {}
    costs = _get(bt, "params", "costs") or {}
    cost_text = (
        f"{_num(costs.get('per_side_bps'), 1)} bps per side, "
        f"borrow {_num(costs.get('borrow_bps'), 1)} and "
        f"funding {_num(costs.get('funding_bps'), 1)} bps per year"
        if costs
        else NOT_AVAILABLE
    )
    rows = [
        [
            "Strategy gross",
            _num(_get(bt, "gross", "sharpe")),
            _pct(_get(bt, "gross", "max_drawdown")),
        ],
        ["Strategy net", _num(_get(bt, "net", "sharpe")), _pct(_get(bt, "net", "max_drawdown"))],
    ]
    for key in ("buy_and_hold", "constant_50", "long_only"):
        block = _get(bt, "benchmarks", key)
        if isinstance(block, dict):
            rows.append(
                [CURVE_LABELS[key], _num(block.get("sharpe")), _pct(block.get("max_drawdown"))]
            )
    table = _state_table(["Series", "Sharpe", "Max drawdown"], rows)
    if all(len(s) == 0 for s in curves.values()):
        chart = f'<p class="note">{_esc(HISTORY_TOO_SHORT)}</p>'
    else:
        chart = equity_curve_svg(curves)
    return (
        chart
        + table
        + f'<p class="note">Net figures charge {_esc(cost_text)}; benchmarks pay the same '
        "costs, a constant position paying its entry on day one. Equity starts at 1.0 on "
        "the first decision date and compounds the out-of-sample daily returns.</p>"
    )


def render_symbol_report(
    result: dict, close: pd.Series, *, cost_model: CostModel | None = None
) -> str:
    """One self-contained HTML document for an ``analyze`` result and its close series.

    Sections: header, current regime and signal, transition matrix heatmap with
    block-bootstrap intervals, stationary distribution with CI whiskers, regime ribbon
    (labels known at the bar), equity curves with the net and gross Sharpe, the
    honest-statistics summary and the disclaimer. ``cost_model`` ``None`` uses the cost
    assumptions echoed in the result.
    """
    source = str(result.get("source", "symbol"))
    labels = regime_series(close, result)
    curves = equity_curves(close, result, cost_model)
    ci = result.get("confidence_intervals") or {}
    matrix_ci = ci.get("transition_matrix") or {}
    stationary_ci = ci.get("stationary_distribution") or {}
    stationary = result.get("stationary_distribution", {})
    ci_level = _finite(ci.get("ci"))
    ci_text = f"{ci_level:.0%}" if ci_level is not None else NOT_AVAILABLE

    heatmap = matrix_heatmap_svg(
        result["transition_matrix"], matrix_ci.get("lower"), matrix_ci.get("upper")
    )
    if all(key in stationary_ci for key in ("lower", "upper")):
        bars = ci_bars_svg(stationary, stationary_ci["lower"], stationary_ci["upper"])
    else:
        none_vector = dict.fromkeys(STATE_KEYS)
        bars = ci_bars_svg(stationary, none_vector, none_vector)

    body = "\n".join(
        [
            f"<h1>{_esc(source)}: Markov regime report</h1>",
            f'<p class="note">{_esc(result.get("framework", ""))}</p>',
            _header_section(result),
            "<h2>Current regime and signal</h2>",
            _regime_section(result),
            "<h2>Transition matrix</h2>",
            f'<p class="note">Overlapping daily labels; cells show the probability and the '
            f"{_esc(ci_text)} block-bootstrap interval.</p>",
            heatmap,
            "<h2>Stationary distribution</h2>",
            bars,
            f'<p class="note">{_esc(STATIONARY_NOTE)}</p>',
            "<h2>Regime ribbon</h2>",
            regime_ribbon_svg(close, labels),
            f'<p class="note">{_esc(NO_HINDSIGHT_NOTE)}</p>',
            "<h2>Equity curves</h2>",
            _equity_section(result, curves),
            "<h2>Honest statistics</h2>",
            _honest_section(result),
        ]
    )
    return _page(f"{source} regime report", body)


INDEX_COLUMNS: tuple[str, ...] = (
    "rank",
    "symbol",
    "current_regime",
    "signal",
    "persistence",
    "stationary_bear",
    "net_sharpe",
    "null_percentile",
)
INDEX_HEADERS: dict[str, str] = {
    "rank": "Rank",
    "symbol": "Symbol",
    "current_regime": "Regime",
    "signal": "Signal",
    "persistence": "Persistence",
    "stationary_bear": "Stationary bear",
    "net_sharpe": "Net Sharpe",
    "null_percentile": "Null percentile",
}


def _index_cell(row: dict, column: str, symbol_pages: dict[str, str]) -> str:
    value = row.get(column)
    if column == "symbol":
        symbol = str(value)
        page = symbol_pages.get(symbol)
        text = _esc(symbol)
        if page is not None:
            text = f'<a href="{_esc(page)}">{text}</a>'
        return f'<td class="text">{text}</td>'
    if column == "current_regime":
        regime = NOT_AVAILABLE if value is None else str(value)
        return f'<td class="text regime-{_esc(regime)}">{_esc(regime)}</td>'
    if column == "rank":
        return f"<td>{_esc(NOT_AVAILABLE if value is None else value)}</td>"
    if column in ("persistence", "stationary_bear"):
        return f"<td>{_esc(_pct(value))}</td>"
    if column == "null_percentile":
        return f"<td>{_esc(_num(value, 1))}</td>"
    return f"<td>{_esc(_num(value, 3 if column == 'signal' else 2))}</td>"


def render_scan_index(table: list[dict], manifest: dict, symbol_pages: dict[str, str]) -> str:
    """The run's ranked table as an HTML page with a link per symbol.

    ``table`` rows are ``hedgefund.jobs.scan`` rows (``INDEX_COLUMNS`` plus ``error``);
    a row with an ``error`` shows the message instead of its numbers. ``manifest`` is a
    ``RunManifest.to_dict()``; ``symbol_pages`` maps symbol to the relative file name of
    its page (symbols without a page are shown without a link).
    """
    header_rows: list[tuple[str, object]] = [
        ("Run", manifest.get("run_id", NOT_AVAILABLE)),
        ("Created", manifest.get("created_at", NOT_AVAILABLE)),
        ("Universe", manifest.get("universe", NOT_AVAILABLE)),
        ("Cost profile", manifest.get("cost_profile", NOT_AVAILABLE)),
        ("Symbols", len(manifest.get("symbols", []) or [])),
    ]
    head = "".join(
        f'<th class="text">{_esc(INDEX_HEADERS[c])}</th>'
        if c in ("symbol", "current_regime")
        else f"<th>{_esc(INDEX_HEADERS[c])}</th>"
        for c in INDEX_COLUMNS
    )
    body_rows = []
    for row in table:
        error = row.get("error")
        if error:
            symbol_cell = _index_cell(row, "symbol", symbol_pages)
            body_rows.append(
                f'<tr class="error"><td>{_esc(NOT_AVAILABLE)}</td>{symbol_cell}'
                f'<td class="text" colspan="{len(INDEX_COLUMNS) - 2}">'
                f"error: {_esc(error)}</td></tr>"
            )
        else:
            cells = "".join(_index_cell(row, c, symbol_pages) for c in INDEX_COLUMNS)
            body_rows.append(f"<tr>{cells}</tr>")
    table_html = f"<table><tr>{head}</tr>{''.join(body_rows)}</table>"
    errors = manifest.get("errors") or {}
    error_html = ""
    if errors:
        items = "".join(
            f"<li><strong>{_esc(s)}</strong>: {_esc(m)}</li>" for s, m in errors.items()
        )
        error_html = f"<h2>Errors</h2><ul>{items}</ul>"
    body = "\n".join(
        [
            f"<h1>Scan {_esc(manifest.get('run_id', ''))}</h1>",
            _kv_table(header_rows),
            "<h2>Ranked table</h2>",
            '<p class="note">Ranked by net Sharpe, errors last. Persistence is the stride '
            "diagonal of the current regime where available (overlapping otherwise); the "
            "null percentile is the net Sharpe's rank against the return-permutation null.</p>",
            table_html,
            error_html,
        ]
    )
    return _page(f"Scan {manifest.get('run_id', '')}", body)


def symbol_filename(symbol: str) -> str:
    """``<symbol>.html`` with every unsafe character (including ``/``) replaced by ``-``."""
    safe = _UNSAFE_FILENAME.sub("-", symbol).strip(".") or "symbol"
    return f"{safe}.html"


def write_run_html(
    run_dir: Path,
    manifest: dict,
    results: dict[str, dict],
    closes: dict[str, pd.Series],
    table: list[dict],
) -> Path:
    """Write ``<run_dir>/html/index.html`` and one page per result; return the index path.

    ``closes`` must hold a close series for every symbol in ``results`` (the pages
    rebuild the ribbon and equity curves from it); a missing one raises ``ValueError``.
    """
    missing = sorted(set(results) - set(closes))
    if missing:
        raise ValueError(f"closes is missing a series for {missing}")
    html_dir = Path(run_dir) / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    symbol_pages: dict[str, str] = {}
    for symbol, result in results.items():
        name = symbol_filename(symbol)
        (html_dir / name).write_text(render_symbol_report(result, closes[symbol]), encoding="utf-8")
        symbol_pages[symbol] = name
    index_path = html_dir / "index.html"
    index_path.write_text(render_scan_index(table, manifest, symbol_pages), encoding="utf-8")
    return index_path
