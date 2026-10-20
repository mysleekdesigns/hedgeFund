"""The ``analyze()`` dict is exactly the companion skill's contract (plus additive fields).

The companion keys come first, in the companion's order; the Phase 2 honest-statistics
keys follow in ``PHASE2_FIELDS`` order, then the Phase 3 ``backtest`` block
(``PHASE3_FIELDS``). Fixtures use small ``null_paths`` /
``bootstrap_resamples`` so the file stays fast; the values are checked for internal
consistency against the ``regime`` estimators, not for statistical power.
"""

import copy
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from hedgefund.backtest.costs import ZERO_COSTS, CostModel
from hedgefund.cli import app
from hedgefund.regime.labels import label_regimes
from hedgefund.regime.markov import (
    build_transition_matrix,
    drift_tvd,
    nstep_forecast,
    rolling_transition_matrix,
    run_length_persistence,
)
from hedgefund.regime.stats import homogeneity_test
from hedgefund.report.json_contract import (
    COMPANION_FIELDS,
    DISCLAIMER,
    FRAMEWORK,
    PHASE2_FIELDS,
    PHASE3_FIELDS,
    analyze,
    backtest_only,
    to_json,
)
from hedgefund.report.terminal import BACKTEST_HEADER, render

SEED = 7
LONG_ROWS = 600  # enough labels for the walk-forward (252 + 30) to produce numbers
SHORT_ROWS = 100  # fewer labels than min_train + 30 -> NaN Sharpe / drawdown
FAST_ROWS = 400  # the "fast path" series from the Phase 2 brief
DAILY_VOL = 0.01
START_PRICE = 100.0
WINDOW = 20  # labeller window = stride

# Small Monte Carlo sizes keep the module fast; every assertion below is structural.
FAST_NULL_PATHS = 20
FAST_RESAMPLES = 20

PROB_KEYS = ["bear", "sideways", "bull"]
COMPANION_PARAMS = ["window", "threshold", "min_train"]
MATRIX_SIDE = 3
ROW_SUM_TOL = 1e-9
EXPECTED_PHASE2_FIELDS = [
    "stride_matrix",
    "stride_diagonal",
    "implied_window_step_diagonal",
    "stride_n_transitions",
    "run_length_persistence",
    "effective_n",
    "null_percentile",
    "confidence_intervals",
    "rolling_matrix",
    "rolling_bars",
    "drift_tvd",
    "ew_matrix",
    "ew_half_life",
    "shrunk_matrix",
    "shrinkage_alpha",
    "mixing_time",
    "mixing_epsilon",
    "tests",
    "statistics_params",
]
EXPECTED_PHASE3_FIELDS = ["backtest"]
BACKTEST_ONLY_KEYS = [
    "source",
    "rows",
    "date_start",
    "date_end",
    "params",
    "backtest",
    "framework",
    "disclaimer",
]


def _synthetic_close(rows: int, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, DAILY_VOL, rows)
    index = pd.bdate_range("2020-01-01", periods=rows, name="date")
    return pd.Series(START_PRICE * np.cumprod(1.0 + returns), index=index, name="close")


def _fast_analyze(close: pd.Series, source: str = "synthetic", **kwargs: object) -> dict:
    return analyze(
        close,
        source=source,
        hmm=False,
        null_paths=FAST_NULL_PATHS,
        bootstrap_resamples=FAST_RESAMPLES,
        **kwargs,
    )


def _walk(node: object, path: str, visit: Callable[[object, str], None]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            _walk(value, f"{path}.{key}", visit)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _walk(value, f"{path}[{i}]", visit)
    else:
        visit(node, path)


def _assert_no_nan(node: object, path: str) -> None:
    def visit(value: object, where: str) -> None:
        if isinstance(value, float):
            assert math.isfinite(value), where

    _walk(node, path, visit)


@pytest.fixture(scope="module")
def close() -> pd.Series:
    return _synthetic_close(LONG_ROWS)


@pytest.fixture(scope="module")
def result(close: pd.Series) -> dict:
    return _fast_analyze(close)


# --------------------------------------------------------------------------- #
# Companion prefix (frozen)
# --------------------------------------------------------------------------- #
def test_top_level_keys_start_with_companion_fields_then_phase2_then_phase3(result: dict) -> None:
    keys = list(result.keys())
    assert keys[: len(COMPANION_FIELDS)] == COMPANION_FIELDS
    phase2_end = len(COMPANION_FIELDS) + len(EXPECTED_PHASE2_FIELDS)
    assert keys[len(COMPANION_FIELDS) : phase2_end] == EXPECTED_PHASE2_FIELDS
    assert keys[phase2_end:] == EXPECTED_PHASE3_FIELDS
    assert PHASE2_FIELDS == EXPECTED_PHASE2_FIELDS
    assert PHASE3_FIELDS == EXPECTED_PHASE3_FIELDS


def test_nested_key_names_match_companion(result: dict) -> None:
    assert list(result["params"].keys()) == [*COMPANION_PARAMS, "return_type"]
    for field in ("next_state_probabilities", "persistence_diagonal", "stationary_distribution"):
        assert list(result[field].keys()) == PROB_KEYS, field
    assert list(result["walk_forward"].keys()) == ["sharpe", "max_drawdown", "n_trades"]
    assert result["hmm"] == {"available": False, "reason": "disabled via --no-hmm"}


def test_states_order_framework_and_disclaimer_are_frozen(result: dict) -> None:
    assert result["states"] == ["Bear", "Sideways", "Bull"]
    assert result["framework"] == FRAMEWORK == "Roan (@RohOnChain)"
    assert result["disclaimer"] == DISCLAIMER == "Backtests are historical, not forward-looking."


def test_value_types_match_companion(result: dict) -> None:
    assert isinstance(result["rows"], int) and result["rows"] == LONG_ROWS
    assert isinstance(result["signal"], float)
    assert len(result["transition_matrix"]) == MATRIX_SIDE
    assert all(isinstance(x, float) for row in result["transition_matrix"] for x in row)
    assert isinstance(result["walk_forward"]["n_trades"], int)
    assert isinstance(result["walk_forward"]["sharpe"], float)
    assert result["params"]["return_type"] == "simple"


def test_dates_are_iso_strings(result: dict) -> None:
    assert result["date_start"] == "2020-01-01"
    assert result["date_end"] == str(pd.bdate_range("2020-01-01", periods=LONG_ROWS)[-1].date())


def test_to_json_round_trips(result: dict) -> None:
    text = to_json(result)
    assert json.loads(text) == result


def test_to_json_keeps_nan_only_for_walk_forward_on_short_history() -> None:
    short = _fast_analyze(_synthetic_close(SHORT_ROWS), source="short")
    parsed = json.loads(to_json(short))
    assert math.isnan(parsed["walk_forward"]["sharpe"])
    assert math.isnan(parsed["walk_forward"]["max_drawdown"])
    assert parsed["walk_forward"]["n_trades"] == 0
    for field in parsed:
        if field != "walk_forward":
            _assert_no_nan(parsed[field], field)
    assert parsed["backtest"]["gross"]["sharpe"] is None  # same short history, null not NaN


def test_analyze_raises_value_error_on_too_few_rows() -> None:
    with pytest.raises(ValueError, match="Not enough data to label regimes"):
        analyze(_synthetic_close(10), source="tiny", window=WINDOW, hmm=False)


def test_cli_json_emits_exactly_one_object(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.csv"
    _synthetic_close(LONG_ROWS).to_csv(path)
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "--csv",
            str(path),
            "--no-hmm",
            "--json",
            "--null-paths",
            str(FAST_NULL_PATHS),
            "--bootstrap-resamples",
            str(FAST_RESAMPLES),
        ],
    )
    assert result.exit_code == 0, result.output
    stdout = result.stdout
    assert stdout.count("\n") == 1 and stdout.endswith("\n")
    parsed = json.loads(stdout)
    assert list(parsed.keys()) == [
        *COMPANION_FIELDS,
        *EXPECTED_PHASE2_FIELDS,
        *EXPECTED_PHASE3_FIELDS,
    ]
    assert parsed["source"] == str(path)
    assert parsed["statistics_params"]["null_paths"] == FAST_NULL_PATHS
    assert parsed["statistics_params"]["bootstrap_resamples"] == FAST_RESAMPLES


# --------------------------------------------------------------------------- #
# Phase 2 fields: NaN-free, internally consistent with the regime estimators
# --------------------------------------------------------------------------- #
def test_every_new_numeric_field_is_finite_or_none(result: dict) -> None:
    for field in EXPECTED_PHASE2_FIELDS:
        _assert_no_nan(result[field], field)


def test_to_json_has_no_nan_token_outside_walk_forward(result: dict) -> None:
    without_wf = {k: v for k, v in result.items() if k != "walk_forward"}
    text = to_json(without_wf)
    assert "NaN" not in text and "Infinity" not in text


def test_stride_matrix_rows_sum_to_one_or_zero(result: dict) -> None:
    for row in result["stride_matrix"]:
        assert len(row) == MATRIX_SIDE
        total = sum(row)
        assert math.isclose(total, 1.0, abs_tol=ROW_SUM_TOL) or total == 0.0, row
        assert all(0.0 <= x <= 1.0 for x in row)


def test_stride_matrix_matches_regime_estimator(close: pd.Series, result: dict) -> None:
    labels = label_regimes(close, window=WINDOW, threshold=0.05, return_type="simple")
    expected = build_transition_matrix(labels, stride=WINDOW)
    assert np.allclose(np.array(result["stride_matrix"]), expected)
    assert result["stride_n_transitions"] == len(labels.to_numpy()[::WINDOW]) - 1


def test_stride_diagonal_equals_diagonal_of_stride_matrix_or_null(result: dict) -> None:
    m = result["stride_matrix"]
    for i, key in enumerate(PROB_KEYS):
        if sum(m[i]) > 0.0:
            assert result["stride_diagonal"][key] == m[i][i]
        else:
            assert result["stride_diagonal"][key] is None


def test_stride_diagonal_is_null_for_a_state_absent_from_the_stride_sequence() -> None:
    # A steady drift with tiny noise never labels Bear, so the Bear stride row is all
    # zeros: the diagonal reports null (missing data), as the null blocks do, not 0.0.
    rows = FAST_ROWS
    index = pd.bdate_range("2020-01-01", periods=rows, name="date")
    trend = START_PRICE * np.exp(np.linspace(0.0, 2.0, rows))
    out = _fast_analyze(pd.Series(trend, index=index, name="close"), source="trend")
    m = out["stride_matrix"]
    assert sum(m[0]) == 0.0 and out["stride_diagonal"]["bear"] is None
    for i, key in enumerate(PROB_KEYS):
        if sum(m[i]) > 0.0:
            assert out["stride_diagonal"][key] == m[i][i]
        else:
            assert out["stride_diagonal"][key] is None


def test_implied_window_step_diagonal_is_nstep_forecast_diagonal(result: dict) -> None:
    implied = np.diag(nstep_forecast(np.array(result["transition_matrix"]), WINDOW))
    for i, key in enumerate(PROB_KEYS):
        assert math.isclose(result["implied_window_step_diagonal"][key], implied[i])


def test_run_length_persistence_is_computed_on_the_stride_sequence(
    close: pd.Series, result: dict
) -> None:
    labels = label_regimes(close, window=WINDOW, threshold=0.05, return_type="simple")
    on_stride = run_length_persistence(labels.to_numpy()[::WINDOW])
    on_daily = run_length_persistence(labels.to_numpy())
    for i, key in enumerate(PROB_KEYS):
        got = result["run_length_persistence"][key]
        if math.isnan(on_stride[i]):
            assert got is None
        else:
            assert math.isclose(got, on_stride[i])
    # Daily-label run lengths equal the overlapping diagonal and must not be what we emit.
    assert not np.allclose(
        [result["run_length_persistence"][k] or 0.0 for k in PROB_KEYS],
        np.nan_to_num(on_daily),
    )


def test_effective_n_is_transitions_over_window(close: pd.Series, result: dict) -> None:
    labels = label_regimes(close, window=WINDOW, threshold=0.05, return_type="simple").to_numpy()
    counts = np.bincount(labels[:-1], minlength=MATRIX_SIDE)
    for i, key in enumerate(PROB_KEYS):
        assert math.isclose(result["effective_n"][key], counts[i] / WINDOW)


def test_confidence_interval_brackets_the_point_diagonal(result: dict) -> None:
    ci = result["confidence_intervals"]
    assert ci["ci"] == 0.95
    assert ci["n_resamples"] == FAST_RESAMPLES
    assert ci["block_days"] == WINDOW
    lower, upper = ci["transition_matrix"]["lower"], ci["transition_matrix"]["upper"]
    for i in range(MATRIX_SIDE):
        assert lower[i][i] <= result["transition_matrix"][i][i] <= upper[i][i]
    for key in PROB_KEYS:
        lo = ci["stationary_distribution"]["lower"][key]
        hi = ci["stationary_distribution"]["upper"][key]
        assert lo <= hi


def test_null_percentile_structure_and_excess_persistence(result: dict) -> None:
    null = result["null_percentile"]
    assert list(null.keys()) == ["label_shuffle", "iid", "labeller_induced"]
    for model in ("label_shuffle", "iid"):
        assert list(null[model].keys()) == ["diagonal_percentile", "null_diagonal_mean", "n_paths"]
        assert null[model]["n_paths"] == FAST_NULL_PATHS
        for key in PROB_KEYS:
            pct = null[model]["diagonal_percentile"][key]
            assert pct is None or 0.0 <= pct <= 100.0
    induced = null["labeller_induced"]
    assert induced["n_paths"] == FAST_NULL_PATHS
    for key in PROB_KEYS:
        real = result["persistence_diagonal"][key]
        mean = induced["null_diagonal_mean"][key]
        excess = induced["excess_persistence"][key]
        lo, hi = induced["excess_ci"][key]
        if mean is None:
            assert excess is None
            continue
        assert math.isclose(excess, real - mean)
        assert lo <= excess <= hi


def test_rolling_ew_shrunk_matrices_are_row_stochastic(result: dict) -> None:
    for field in ("rolling_matrix", "ew_matrix", "shrunk_matrix"):
        for row in result[field]:
            total = sum(row)
            assert math.isclose(total, 1.0, abs_tol=ROW_SUM_TOL) or total == 0.0, (field, row)
    assert 0.0 <= result["drift_tvd"] <= 1.0
    assert result["rolling_bars"] == 756


def test_drift_tvd_skips_states_missing_from_the_rolling_window(close: pd.Series) -> None:
    # A 2-bar rolling window holds a single transition, so two rolling rows are all
    # zeros. Drift must be the TVD of the one populated row, not the 0.5 that a
    # populated full-history row scores against an empty row.
    rolling_bars = 2
    out = _fast_analyze(close, rolling_bars=rolling_bars)
    labels = label_regimes(close, window=WINDOW, threshold=0.05, return_type="simple")
    rolling = rolling_transition_matrix(labels.to_numpy(), window_bars=rolling_bars)
    full = np.array(out["transition_matrix"])
    populated = [i for i in range(MATRIX_SIDE) if rolling[i].sum() > 0]
    assert len(populated) == 1
    expected = 0.5 * np.abs(full[populated[0]] - rolling[populated[0]]).sum()
    assert out["drift_tvd"] == pytest.approx(expected)
    assert out["drift_tvd"] == pytest.approx(drift_tvd(full, rolling))
    assert out["drift_tvd"] < 0.5 or expected >= 0.5


def test_drift_tvd_field_is_null_when_no_row_is_comparable(result: dict) -> None:
    # The contract routes ``drift_tvd`` through the NaN-to-null helper; an all-zero
    # rolling matrix is the case that makes the estimator return NaN.
    from hedgefund.report.json_contract import _finite_or_none

    full = np.array(result["transition_matrix"])
    assert _finite_or_none(drift_tvd(full, np.zeros_like(full))) is None
    assert result["ew_half_life"] == 250.0
    assert result["shrinkage_alpha"] == 20.0


def test_mixing_time_is_int_or_none(result: dict) -> None:
    assert result["mixing_time"] is None or isinstance(result["mixing_time"], int)
    assert result["mixing_epsilon"] == 0.01


def test_tests_block_shape(result: dict) -> None:
    ag = result["tests"]["anderson_goodman"]
    assert list(ag.keys()) == ["statistic", "p_value", "df", "n_transitions", "note"]
    assert ag["note"] == "on the stride sequence"
    assert 0.0 <= ag["p_value"] <= 1.0
    hg = result["tests"]["homogeneity"]
    assert list(hg.keys()) == ["statistic", "p_value", "df", "n_periods", "stable", "note"]
    assert hg["note"] == "on the stride sequence"
    assert hg["n_periods"] == 2
    assert isinstance(hg["stable"], bool)
    assert hg["stable"] == (hg["p_value"] >= 0.05)


def test_homogeneity_test_is_computed_on_the_stride_sequence(
    close: pd.Series, result: dict
) -> None:
    # Overlapping daily labels are autocorrelated, so the chi-square p-value is only
    # calibrated on the stride sequence ``labels[::window]`` (the Anderson-Goodman input).
    labels = label_regimes(close, window=WINDOW, threshold=0.05, return_type="simple")
    stride = labels.to_numpy()[::WINDOW]
    expected = homogeneity_test(stride, n_periods=2)
    hg = result["tests"]["homogeneity"]
    assert hg["statistic"] == pytest.approx(expected["statistic"])
    assert hg["p_value"] == pytest.approx(expected["p_value"])
    assert hg["df"] == expected["df"]
    assert hg["stable"] == expected["stable"]
    on_daily = homogeneity_test(labels.to_numpy(), n_periods=2)
    assert hg["statistic"] != pytest.approx(on_daily["statistic"])


def test_statistics_params_echo_inputs(result: dict) -> None:
    assert result["statistics_params"] == {
        "null_paths": FAST_NULL_PATHS,
        "bootstrap_resamples": FAST_RESAMPLES,
        "block_days": 20,
        "rolling_bars": 756,
        "ew_half_life": 250.0,
        "shrinkage_alpha": 20.0,
        "mixing_epsilon": 0.01,
        "homogeneity_periods": 2,
        "seed": 0,
    }


def test_analysis_is_deterministic_for_the_same_seed(close: pd.Series, result: dict) -> None:
    again = _fast_analyze(close)
    assert again == result


def test_changing_the_seed_changes_nothing_in_the_companion_fields(
    close: pd.Series, result: dict
) -> None:
    other = _fast_analyze(close, seed=1)
    for field in COMPANION_FIELDS:
        if field == "walk_forward":
            assert other[field]["n_trades"] == result[field]["n_trades"]
            continue
        assert other[field] == result[field], field
    assert other["statistics_params"]["seed"] == 1
    # The stochastic blocks should move with the seed (20 paths is enough to differ).
    assert other["null_percentile"] != result["null_percentile"]


def test_fast_path_runs_on_a_400_day_series() -> None:
    out = _fast_analyze(_synthetic_close(FAST_ROWS), source="fast")
    assert list(out.keys()) == [
        *COMPANION_FIELDS,
        *EXPECTED_PHASE2_FIELDS,
        *EXPECTED_PHASE3_FIELDS,
    ]
    _assert_no_nan({k: out[k] for k in EXPECTED_PHASE2_FIELDS}, "phase2")
    _assert_no_nan(out["backtest"], "backtest")


def test_short_series_emits_null_for_fields_it_cannot_compute() -> None:
    # 22 rows -> 2 labels: the stride sequence has one label, too short for the stride
    # diagonal and both tests, so those fields are null while the companion fields ship.
    tiny = _fast_analyze(_synthetic_close(WINDOW + 2), source="tiny")
    assert list(tiny.keys()) == [
        *COMPANION_FIELDS,
        *EXPECTED_PHASE2_FIELDS,
        *EXPECTED_PHASE3_FIELDS,
    ]
    _assert_no_nan(tiny["backtest"], "backtest")
    assert tiny["stride_n_transitions"] == 0
    assert tiny["stride_diagonal"] == {"bear": None, "sideways": None, "bull": None}
    assert tiny["tests"]["anderson_goodman"] is None
    assert tiny["tests"]["homogeneity"] is None
    _assert_no_nan({k: tiny[k] for k in EXPECTED_PHASE2_FIELDS}, "phase2")


def test_result_is_plain_json_types(result: dict) -> None:
    """No numpy scalars leak into the dict (they would break ``json.dumps``)."""

    def visit(value: object, where: str) -> None:
        assert value is None or isinstance(value, bool | int | float | str), (where, type(value))

    _walk(copy.deepcopy(result), "result", visit)


# --------------------------------------------------------------------------- #
# Phase 3: the ``backtest`` block and ``backtest_only``
# --------------------------------------------------------------------------- #
def test_backtest_gross_sharpe_equals_the_companion_walk_forward(result: dict) -> None:
    bt = result["backtest"]
    assert bt["gross"]["sharpe"] == result["walk_forward"]["sharpe"]
    assert bt["gross"]["max_drawdown"] == result["walk_forward"]["max_drawdown"]
    assert bt["gross"]["n_days"] == result["walk_forward"]["n_trades"]
    assert bt["net"]["sharpe"] < bt["gross"]["sharpe"]  # default us_etfs costs apply


def test_backtest_block_shares_null_paths_and_seed_with_the_statistics(result: dict) -> None:
    params = result["backtest"]["params"]
    assert params["null_paths"] == result["statistics_params"]["null_paths"] == FAST_NULL_PATHS
    assert params["seed"] == result["statistics_params"]["seed"]
    assert result["backtest"]["null"]["permutation"]["n_paths"] == FAST_NULL_PATHS
    assert params["refit"] == "expanding" and params["rule"] == "sign"
    assert params["dead_band"] == 0.0 and params["rolling_bars"] == 756
    assert params["costs"]["per_side_bps"] == CostModel().per_side_bps


def test_backtest_block_is_finite_and_round_trips(result: dict) -> None:
    _assert_no_nan(result["backtest"], "backtest")
    assert "NaN" not in to_json(result["backtest"])


def test_changing_cost_model_or_seed_changes_nothing_in_companion_or_phase2_fields(
    close: pd.Series, result: dict
) -> None:
    other = _fast_analyze(
        close,
        cost_model=ZERO_COSTS,
        rule="long_only",
        refit="rolling",
        backtest_rolling_bars=100,
        dead_band=0.5,
    )
    for field in [*COMPANION_FIELDS, *PHASE2_FIELDS]:
        assert other[field] == result[field], field
    assert other["backtest"]["params"]["rule"] == "long_only"
    assert other["backtest"]["params"]["refit"] == "rolling"
    assert other["backtest"]["params"]["rolling_bars"] == 100
    assert other["backtest"]["net"] == other["backtest"]["gross"]  # zero costs
    assert other["backtest"] != result["backtest"]


def test_statistics_rolling_bars_and_backtest_rolling_bars_are_separate_knobs(
    close: pd.Series,
) -> None:
    out = _fast_analyze(close, rolling_bars=100, backtest_rolling_bars=300)
    assert out["rolling_bars"] == 100  # Phase 2 rolling matrix window
    assert out["statistics_params"]["rolling_bars"] == 100
    assert out["backtest"]["params"]["rolling_bars"] == 300  # walk-forward rolling refit


def test_backtest_only_returns_the_slim_dict(close: pd.Series, result: dict) -> None:
    slim = backtest_only(close, source="synthetic", null_paths=FAST_NULL_PATHS)
    assert list(slim.keys()) == BACKTEST_ONLY_KEYS
    assert slim["params"] == result["params"]
    assert slim["framework"] == FRAMEWORK and slim["disclaimer"] == DISCLAIMER
    assert slim["rows"] == result["rows"]
    assert slim["date_start"] == result["date_start"] and slim["date_end"] == result["date_end"]
    assert slim["backtest"] == result["backtest"]  # same inputs, same block
    assert json.loads(to_json(slim)) == slim


def test_backtest_only_raises_the_companion_error_on_too_few_rows() -> None:
    with pytest.raises(ValueError, match="Not enough data to label regimes"):
        backtest_only(_synthetic_close(10), source="tiny", window=WINDOW)


def test_render_consumes_the_backtest_block(result: dict) -> None:
    text = render(result)
    assert BACKTEST_HEADER in text
    assert "buy-and-hold" in text and "percentile" in text
    without = copy.deepcopy(result)
    del without["backtest"]
    assert BACKTEST_HEADER not in render(without)
