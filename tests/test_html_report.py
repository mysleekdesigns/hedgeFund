"""Tests for ``hedgefund.report.html`` (plan.md Phase 5: matrix heatmap, regime
ribbon, equity curves vs benchmarks, CI bars, symbol page, scan index).

One synthetic geometric random walk (fixed seed, no network) is analysed once per
module with small Monte Carlo sizes; every assertion is structural (labels present,
escaping applied, curves start at 1.0, net below gross) rather than numeric.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hedgefund.backtest.costs import ZERO_COSTS, CostModel
from hedgefund.regime.labels import STATES, label_regimes
from hedgefund.report.html import (
    CURVE_KEYS,
    DISCLAIMER,
    ci_bars_svg,
    equity_curve_svg,
    equity_curves,
    matrix_heatmap_svg,
    regime_ribbon_svg,
    regime_series,
    render_scan_index,
    render_symbol_report,
    symbol_filename,
    write_run_html,
)
from hedgefund.report.json_contract import analyze

SEED = 11
LONG_ROWS = 600  # enough labels for the walk-forward (252 + 30) to produce curves
SHORT_ROWS = 100  # fewer labels than min_train + 30 -> empty walk-forward
DAILY_VOL = 0.01
START_PRICE = 100.0
FAST_NULL_PATHS = 20  # small Monte Carlo keeps the module fast; assertions are structural
FAST_RESAMPLES = 20
SYMBOL = "<X>"  # a symbol that must be escaped everywhere it is printed
N_CHARTS = 4  # heatmap, CI bars, ribbon, equity curves
STATE_KEYS = ("bear", "sideways", "bull")


def _synthetic_close(rows: int, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, DAILY_VOL, rows)
    index = pd.bdate_range("2020-01-01", periods=rows, name="date")
    return pd.Series(START_PRICE * np.cumprod(1.0 + returns), index=index, name="close")


def _fast_analyze(close: pd.Series, source: str = SYMBOL) -> dict:
    return analyze(
        close,
        source=source,
        hmm=False,
        null_paths=FAST_NULL_PATHS,
        bootstrap_resamples=FAST_RESAMPLES,
    )


@pytest.fixture(scope="module")
def close() -> pd.Series:
    return _synthetic_close(LONG_ROWS)


@pytest.fixture(scope="module")
def result(close: pd.Series) -> dict:
    return _fast_analyze(close)


@pytest.fixture(scope="module")
def short_close() -> pd.Series:
    return _synthetic_close(SHORT_ROWS, seed=SEED + 1)


@pytest.fixture(scope="module")
def short_result(short_close: pd.Series) -> dict:
    return _fast_analyze(short_close, source="short")


@pytest.fixture(scope="module")
def curves(close: pd.Series, result: dict) -> dict[str, pd.Series]:
    return equity_curves(close, result, None)


def _assert_svg(svg: str) -> None:
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    ET.fromstring(svg)  # well-formed XML, so browsers render it


# ---------------------------------------------------------------------------- series


def test_regime_series_reproduces_label_regimes_with_result_params(
    close: pd.Series, result: dict
) -> None:
    params = result["params"]
    expected = label_regimes(
        close,
        window=params["window"],
        threshold=params["threshold"],
        return_type=params["return_type"],
    )
    pd.testing.assert_series_equal(regime_series(close, result), expected)


def test_equity_curves_has_every_key_and_starts_at_one(curves: dict[str, pd.Series]) -> None:
    assert tuple(curves) == CURVE_KEYS
    for key in CURVE_KEYS:
        assert len(curves[key]) > 1, key
        assert curves[key].iloc[0] == 1.0, key
        assert np.isfinite(curves[key].to_numpy()).all(), key


def test_equity_curves_share_one_index(curves: dict[str, pd.Series]) -> None:
    first = curves[CURVE_KEYS[0]].index
    for key in CURVE_KEYS[1:]:
        assert curves[key].index.equals(first), key


def test_equity_curves_net_ends_at_or_below_gross_with_positive_costs(
    curves: dict[str, pd.Series], result: dict
) -> None:
    assert result["backtest"]["params"]["costs"]["per_side_bps"] > 0
    assert curves["strategy_net"].iloc[-1] <= curves["strategy_gross"].iloc[-1]


def test_equity_curves_zero_costs_make_net_equal_gross(close: pd.Series, result: dict) -> None:
    free = equity_curves(close, result, ZERO_COSTS)
    pd.testing.assert_series_equal(free["strategy_net"], free["strategy_gross"])


def test_equity_curves_net_matches_backtest_block_total_return(
    curves: dict[str, pd.Series], result: dict
) -> None:
    # The curves re-run the same walk-forward and cost model as the ``backtest`` block,
    # so the net equity's final value is 1 + its reported total return.
    expected = 1.0 + result["backtest"]["net"]["total_return"]
    assert math.isclose(curves["strategy_net"].iloc[-1], expected, rel_tol=1e-9)


def test_equity_curves_empty_on_short_history(short_close: pd.Series, short_result: dict) -> None:
    short = equity_curves(short_close, short_result, None)
    assert tuple(short) == CURVE_KEYS
    assert all(len(s) == 0 for s in short.values())


def test_equity_curves_builds_cost_model_from_result_when_none(
    close: pd.Series, result: dict, curves: dict[str, pd.Series]
) -> None:
    costs = result["backtest"]["params"]["costs"]
    explicit = equity_curves(
        close,
        result,
        CostModel(
            fee_bps=costs["fee_bps"],
            slippage_bps=costs["slippage_bps"],
            borrow_bps=costs["borrow_bps"],
            funding_bps=costs["funding_bps"],
        ),
    )
    pd.testing.assert_series_equal(explicit["strategy_net"], curves["strategy_net"])


# ---------------------------------------------------------------------------- charts


def test_matrix_heatmap_svg_has_state_headers_and_values(result: dict) -> None:
    svg = matrix_heatmap_svg(result["transition_matrix"])
    _assert_svg(svg)
    for state in STATES:
        assert svg.count(state) >= 2, state  # row and column header
    for row in result["transition_matrix"]:
        for value in row:
            assert f"{value:.2f}" in svg


def test_matrix_heatmap_svg_shows_interval_text_when_bounds_given(result: dict) -> None:
    ci = result["confidence_intervals"]["transition_matrix"]
    svg = matrix_heatmap_svg(result["transition_matrix"], ci["lower"], ci["upper"])
    _assert_svg(svg)
    lo, hi = ci["lower"][0][0], ci["upper"][0][0]
    assert f"[{lo:.2f}, {hi:.2f}]" in svg
    assert svg.count("[") == 9  # one interval per cell


def test_matrix_heatmap_svg_rejects_non_3x3() -> None:
    with pytest.raises(ValueError, match="3x3"):
        matrix_heatmap_svg([[0.5, 0.5], [0.5, 0.5]])


def test_regime_ribbon_svg_has_bands_price_line_and_legend(close: pd.Series, result: dict) -> None:
    labels = regime_series(close, result)
    svg = regime_ribbon_svg(close, labels)
    _assert_svg(svg)
    assert "<polyline" in svg
    assert "<rect" in svg
    for state in STATES:
        assert state in svg
    assert str(labels.index[0].date()) in svg
    assert str(labels.index[-1].date()) in svg


def test_regime_ribbon_svg_one_band_per_label_run(close: pd.Series, result: dict) -> None:
    labels = regime_series(close, result)
    runs = 1 + int((np.diff(labels.to_numpy()) != 0).sum())
    svg = regime_ribbon_svg(close, labels)
    # Band rectangles plus three legend swatches and the frame.
    assert svg.count("<rect") == runs + len(STATES) + 1


def test_regime_ribbon_svg_rejects_bad_labels(close: pd.Series) -> None:
    bad = pd.Series([0, 1, 3], index=close.index[:3], dtype=int)
    with pytest.raises(ValueError, match="labels"):
        regime_ribbon_svg(close, bad)


def test_equity_curve_svg_has_a_line_and_legend_entry_per_curve(
    curves: dict[str, pd.Series],
) -> None:
    svg = equity_curve_svg(curves)
    _assert_svg(svg)
    assert svg.count("<polyline") == len(CURVE_KEYS)
    for label in ("Strategy (gross)", "Strategy (net of costs)", "Buy and hold"):
        assert label in svg


def test_equity_curve_svg_with_empty_curves_says_history_too_short() -> None:
    empty = pd.Series(np.empty(0), index=pd.DatetimeIndex([]), dtype=float)
    svg = equity_curve_svg(dict.fromkeys(CURVE_KEYS, empty))
    _assert_svg(svg)
    assert "History too short" in svg
    assert "<polyline" not in svg


def test_ci_bars_svg_has_bars_whiskers_and_state_labels(result: dict) -> None:
    ci = result["confidence_intervals"]["stationary_distribution"]
    svg = ci_bars_svg(result["stationary_distribution"], ci["lower"], ci["upper"])
    _assert_svg(svg)
    for state in STATES:
        assert state in svg
    assert svg.count("<rect") == len(STATES) + 1  # three bars plus the frame
    assert svg.count("<line") >= 3 * len(STATES)  # whisker plus two caps per state


def test_ci_bars_svg_skips_whiskers_for_missing_bounds(result: dict) -> None:
    none = dict.fromkeys(STATE_KEYS)
    svg = ci_bars_svg(result["stationary_distribution"], none, none)
    _assert_svg(svg)
    assert 'stroke="#1a1a1a"' not in svg


def test_ci_bars_svg_requires_all_state_keys(result: dict) -> None:
    with pytest.raises(ValueError, match="bull"):
        ci_bars_svg({"bear": 0.3, "sideways": 0.4}, {}, {})


# ----------------------------------------------------------------------------- pages


@pytest.fixture(scope="module")
def page(result: dict, close: pd.Series) -> str:
    return render_symbol_report(result, close)


def test_symbol_report_carries_disclaimer(page: str) -> None:
    assert DISCLAIMER in page


def test_symbol_report_names_current_regime_and_signal(page: str, result: dict) -> None:
    assert result["current_regime"] in page
    assert f"{result['signal']:.3f}" in page


def test_symbol_report_says_ribbon_uses_labels_known_at_the_bar(page: str) -> None:
    assert "known at the bar" in page
    assert "no hindsight" in page


def test_symbol_report_embeds_four_charts(page: str) -> None:
    assert page.count("<svg") == N_CHARTS


def test_symbol_report_shows_net_and_gross_sharpe(page: str, result: dict) -> None:
    bt = result["backtest"]
    assert f"{bt['net']['sharpe']:.2f}" in page
    assert f"{bt['gross']['sharpe']:.2f}" in page
    assert "Strategy net" in page and "Strategy gross" in page


def test_symbol_report_shows_honest_statistics(page: str, result: dict) -> None:
    assert "Stride (non-overlapping)" in page
    assert "Labeller-induced null" in page
    p_net = result["backtest"]["null"]["permutation"]["p_value_net"]
    assert f"{p_net:.3f}" in page


def test_symbol_report_escapes_source(page: str) -> None:
    assert SYMBOL not in page
    assert "&lt;X&gt;" in page


def test_symbol_report_short_history_renders_note_instead_of_equity_chart(
    short_result: dict, short_close: pd.Series
) -> None:
    short_page = render_symbol_report(short_result, short_close)
    assert "History too short" in short_page
    assert short_page.count("<svg") == N_CHARTS - 1
    assert DISCLAIMER in short_page


def test_symbol_report_accepts_explicit_cost_model(result: dict, close: pd.Series) -> None:
    free = render_symbol_report(result, close, cost_model=ZERO_COSTS)
    assert DISCLAIMER in free


def _table(result: dict) -> list[dict]:
    return [
        {
            "rank": 1,
            "symbol": SYMBOL,
            "current_regime": result["current_regime"],
            "signal": result["signal"],
            "persistence": result["persistence_diagonal"]["bull"],
            "persistence_source": "overlapping",
            "stationary_bear": result["stationary_distribution"]["bear"],
            "net_sharpe": result["backtest"]["net"]["sharpe"],
            "null_percentile": result["backtest"]["null"]["permutation"]["percentile_net"],
            "rows": result["rows"],
            "date_end": result["date_end"],
            "error": None,
        },
        {
            "rank": None,
            "symbol": "BAD/Y",
            "current_regime": None,
            "signal": None,
            "persistence": None,
            "stationary_bear": None,
            "net_sharpe": None,
            "null_percentile": None,
            "rows": None,
            "date_end": None,
            "error": "boom <b>",
        },
    ]


def _manifest() -> dict:
    return {
        "run_id": "20260101T000000Z_test",
        "created_at": "2026-01-01T00:00:00+00:00",
        "universe": "test",
        "cost_profile": "us_etfs",
        "symbols": [SYMBOL, "BAD/Y"],
        "errors": {"BAD/Y": "boom <b>"},
    }


def test_scan_index_links_symbols_and_shows_manifest_header(result: dict) -> None:
    index = render_scan_index(_table(result), _manifest(), {SYMBOL: "-X-.html"})
    assert 'href="-X-.html"' in index
    assert "&lt;X&gt;" in index and SYMBOL not in index
    for text in ("20260101T000000Z_test", "2026-01-01T00:00:00+00:00", "us_etfs"):
        assert text in index
    for header in ("Rank", "Symbol", "Regime", "Signal", "Persistence", "Net Sharpe"):
        assert header in index
    assert DISCLAIMER in index


def test_scan_index_shows_errored_rows_with_escaped_error_text(result: dict) -> None:
    index = render_scan_index(_table(result), _manifest(), {})
    assert "boom &lt;b&gt;" in index
    assert "boom <b>" not in index
    assert 'class="error"' in index


def test_symbol_filename_replaces_slash() -> None:
    assert symbol_filename("BTC/USD") == "BTC-USD.html"
    assert symbol_filename("SPY") == "SPY.html"
    assert "/" not in symbol_filename("a/b/c") and "<" not in symbol_filename(SYMBOL)


def test_write_run_html_writes_index_and_symbol_pages(
    tmp_path: Path, result: dict, close: pd.Series
) -> None:
    index_path = write_run_html(
        tmp_path, _manifest(), {SYMBOL: result}, {SYMBOL: close}, _table(result)
    )
    assert index_path == tmp_path / "html" / "index.html"
    page_path = tmp_path / "html" / symbol_filename(SYMBOL)
    assert page_path.exists()
    index = index_path.read_text(encoding="utf-8")
    assert f'href="{page_path.name}"' in index
    symbol_page = page_path.read_text(encoding="utf-8")
    assert DISCLAIMER in symbol_page
    assert symbol_page.count("<svg") == N_CHARTS


def test_write_run_html_requires_a_close_for_every_result(tmp_path: Path, result: dict) -> None:
    with pytest.raises(ValueError, match="closes"):
        write_run_html(tmp_path, _manifest(), {SYMBOL: result}, {}, _table(result))
