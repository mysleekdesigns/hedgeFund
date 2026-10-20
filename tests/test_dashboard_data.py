"""Phase 8: the dashboard's offline data layer (``hedgefund.dashboard.data``).

Every test reads the ``dashboard_run`` fixture (a cached ``hedgefund scan`` over two
synthetic CSVs) or synthetic arrays; the autouse ``no_network`` guard in ``conftest.py``
fails any test that opens a socket, so there is nothing to patch here.
"""

import copy
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import hedgefund.dashboard.data as data
from hedgefund.backtest.costs import ZERO_COSTS, CostModel
from hedgefund.dashboard import JOBS_DIR_ENV
from hedgefund.jobs.manifest import DEFAULT_JOBS_DIR, REPO_ROOT, RunManifest
from hedgefund.regime.labels import STATES, label_regimes
from hedgefund.regime.markov import build_transition_matrix
from hedgefund.report.html import CURVE_KEYS

N = len(STATES)
# An ergodic chain: every entry positive, so ``P**n`` converges geometrically to ``pi``.
ERGODIC = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])
MAX_N = 200  # the second eigenvalue is 0.75 at most here, 0.75**200 ~ 1e-25: converged
# Recomputing the matrix from the labels the JSON was built from is the same integer
# counts divided by the same row sums; 1e-12 leaves room only for summation order.
REBUILD_TOL = 1e-12
EXACT = 0.0  # JSON floats round-trip exactly, so the matrix comparison is bit-for-bit
SEED = 3
N_RETURNS = 120
DAILY_VOL = 0.01
HEAVY_COSTS = CostModel(fee_bps=50.0, slippage_bps=50.0, borrow_bps=0.0, funding_bps=0.0)


@pytest.fixture(scope="module")
def run(dashboard_run: dict) -> data.LoadedRun:
    return data.load_run(dashboard_run["run_id"], dashboard_run["jobs_dir"])


# ------------------------------------------------------------------------- locating runs


def test_resolve_jobs_dir_precedence(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    from_env = tmp_path / "from_env"
    env = {JOBS_DIR_ENV: str(from_env)}
    assert data.resolve_jobs_dir(explicit, env) == explicit
    assert data.resolve_jobs_dir(None, env) == from_env
    assert data.resolve_jobs_dir(None, {}) == DEFAULT_JOBS_DIR


def test_list_run_ids_newest_first_and_empty_when_missing(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    for run_id in ("20240101T000000Z_a", "20250101T000000Z_b", "20230101T000000Z_c"):
        (jobs / run_id).mkdir(parents=True)
        (jobs / run_id / "manifest.json").write_text("{}")
    assert data.list_run_ids(jobs) == [
        "20250101T000000Z_b",
        "20240101T000000Z_a",
        "20230101T000000Z_c",
    ]
    assert data.list_run_ids(tmp_path / "nowhere") == []


def test_resolve_cache_path_relative_joins_repo_root_absolute_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "root"
    assert data.resolve_cache_path("cache/csv/AAA.parquet", root) == root / "cache/csv/AAA.parquet"
    absolute = tmp_path / "elsewhere" / "AAA.parquet"
    assert data.resolve_cache_path(str(absolute), root) == absolute
    assert data.resolve_cache_path("cache/x.parquet") == REPO_ROOT / "cache" / "x.parquet"


# ------------------------------------------------------------------------------ loading


def test_load_run_symbols_and_closes(run: data.LoadedRun, dashboard_run: dict) -> None:
    assert run.run_id == dashboard_run["run_id"]
    assert run.symbols == list(dashboard_run["symbols"])
    assert set(run.results) == set(run.symbols)
    assert run.table
    assert "costs" in run.config
    for symbol in run.symbols:
        close = run.closes[symbol]
        assert isinstance(close.index, pd.DatetimeIndex)
        assert close.dtype == np.float64
        assert close.size == run.manifest.data[symbol]["rows"]


def _broken_manifest(manifest: RunManifest, symbol: str, cache_path: object) -> RunManifest:
    broken = copy.deepcopy(manifest)
    broken.data[symbol]["cache_path"] = cache_path
    return broken


def test_load_closes_raises_naming_symbol_when_cache_path_missing(
    run: data.LoadedRun, tmp_path: Path
) -> None:
    symbol = run.symbols[0]
    results = {symbol: run.results[symbol]}
    with pytest.raises(ValueError, match=symbol):
        data.load_closes(_broken_manifest(run.manifest, symbol, None), results)
    missing = tmp_path / "gone.parquet"
    with pytest.raises(ValueError, match=f"{symbol}.*{missing.name}"):
        data.load_closes(_broken_manifest(run.manifest, symbol, str(missing)), results)


# -------------------------------------------------------------------------------- controls


def test_controls_from_run_match_manifest_and_mismatch_after_change(run: data.LoadedRun) -> None:
    symbol = run.symbols[0]
    params = run.manifest.params
    controls = data.controls_from_run(run, symbol)
    assert controls.symbol == symbol
    assert controls.window == params["window"]
    assert controls.threshold == params["threshold"]
    assert controls.refit == params["refit"]
    assert controls.rolling_bars == params["backtest_rolling_bars"]
    assert controls.return_type == params["return_type"]
    assert controls.cost_profile == run.manifest.cost_profile
    assert data.controls_match_run(run, controls)
    changed = dataclasses.replace(controls, window=controls.window + 1)
    assert not data.controls_match_run(run, changed)
    other_profile = dataclasses.replace(controls, cost_profile="crypto")
    assert not data.controls_match_run(run, other_profile)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("window", 1),
        ("threshold", 0.0),
        ("threshold", 1.0),
        ("refit", "sliding"),
        ("rolling_bars", 0),
        ("symbol", ""),
        ("cost_profile", ""),
    ],
)
def test_controls_validation_names_field(field: str, value: object) -> None:
    kwargs = {
        "symbol": "AAA",
        "window": 20,
        "threshold": 0.05,
        "refit": "expanding",
        "rolling_bars": 756,
        "cost_profile": "us_etfs",
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        data.Controls(**kwargs)


def test_control_options_union_config_and_run(run: data.LoadedRun) -> None:
    config = {
        "backtest": {
            "rolling_bars": 500,
            "sensitivity": {"windows": [10, 60], "thresholds": [0.02, 0.10]},
        },
        "costs": {"profiles": {"crypto": {"fee_bps": 10, "slippage_bps": 10}}},
    }
    options = data.control_options(config, run)
    params = run.manifest.params
    assert options["windows"] == sorted({10, 60, int(params["window"])})
    assert options["thresholds"] == sorted({0.02, 0.10, float(params["threshold"])})
    assert options["refits"] == ["expanding", "rolling"]
    assert options["cost_profiles"] == sorted({"crypto", run.manifest.cost_profile})
    assert options["rolling_bars"] == 500
    without_run = data.control_options(config)
    assert without_run["windows"] == [10, 60]
    assert without_run["cost_profiles"] == ["crypto"]


# ----------------------------------------------------------------------------- symbol view


def test_symbol_view_matching_controls_reads_json_and_agrees_with_recomputation(
    run: data.LoadedRun,
) -> None:
    symbol = run.symbols[0]
    controls = data.controls_from_run(run, symbol)
    view = data.compute_symbol_view(run, controls)
    result = run.results[symbol]
    assert view.recomputed is False
    np.testing.assert_allclose(view.matrix, np.array(result["transition_matrix"]), atol=EXACT)
    expected_pi = [result["stationary_distribution"][k] for k in ("bear", "sideways", "bull")]
    np.testing.assert_allclose(view.stationary, expected_pi, atol=EXACT)
    ci = result["confidence_intervals"]
    np.testing.assert_allclose(view.matrix_lower, ci["transition_matrix"]["lower"], atol=EXACT)
    np.testing.assert_allclose(view.matrix_upper, ci["transition_matrix"]["upper"], atol=EXACT)
    assert view.signal == result["signal"]
    assert view.mixing_time == result["mixing_time"]
    assert STATES[view.current_state] == result["current_regime"]
    assert view.cost_model.fee_bps == result["backtest"]["params"]["costs"]["fee_bps"]

    expected_labels = label_regimes(
        view.close,
        window=controls.window,
        threshold=controls.threshold,
        return_type=controls.return_type,
    )
    pd.testing.assert_series_equal(view.labels, expected_labels)
    rebuilt = build_transition_matrix(view.labels)
    np.testing.assert_allclose(rebuilt, view.matrix, atol=REBUILD_TOL)
    assert view.labels.iloc[-1] == view.current_state
    np.testing.assert_allclose(view.next_state_probabilities, view.matrix[view.current_state])
    assert tuple(view.curves) == CURVE_KEYS
    assert view.null_paths == run.manifest.params["null_paths"]


def test_force_recompute_reproduces_json_exactly(run: data.LoadedRun) -> None:
    """The recompute branch, fed the run's own controls, equals the JSON bit-for-bit.

    Zero tolerance is justified: the labels are the same series the scan built, the
    bootstrap uses the same ``n_resamples``, ``block_days``, ``ci`` and ``seed`` (so the
    same ``default_rng`` stream), and ``mixing_time`` and ``signal_from_matrix`` are the
    very functions ``analyze`` called. JSON round-trips Python floats exactly.
    """
    symbol = run.symbols[0]
    controls = data.controls_from_run(run, symbol)
    from_json = data.compute_symbol_view(run, controls)
    recomputed = data.compute_symbol_view(run, controls, force_recompute=True)
    assert from_json.recomputed is False
    assert recomputed.recomputed is True
    for field in (
        "matrix",
        "matrix_lower",
        "matrix_upper",
        "stationary",
        "stationary_lower",
        "stationary_upper",
        "next_state_probabilities",
    ):
        np.testing.assert_allclose(
            getattr(recomputed, field), getattr(from_json, field), rtol=EXACT, atol=EXACT
        )
    assert recomputed.signal == from_json.signal
    assert recomputed.mixing_time == from_json.mixing_time
    assert recomputed.current_state == from_json.current_state
    assert recomputed.cost_model == from_json.cost_model
    for key in CURVE_KEYS:
        pd.testing.assert_series_equal(recomputed.curves[key], from_json.curves[key])
    pd.testing.assert_series_equal(recomputed.null_lower, from_json.null_lower)
    pd.testing.assert_series_equal(recomputed.null_upper, from_json.null_upper)


def _config_with_fee(run: data.LoadedRun, profile: str, fee_bps: float) -> dict:
    edited = copy.deepcopy(run.config)
    edited["costs"]["profiles"][profile]["fee_bps"] = fee_bps
    return edited


EDITED_FEE_BPS = 50.0  # far from the recorded 1 bp so a wrong model is unmistakable


def test_symbol_view_charges_recorded_cost_model_not_edited_config(run: data.LoadedRun) -> None:
    symbol = run.symbols[0]
    profile = run.manifest.cost_profile
    edited_run = dataclasses.replace(run, config=_config_with_fee(run, profile, EDITED_FEE_BPS))
    controls = data.controls_from_run(edited_run, symbol)
    view = data.compute_symbol_view(edited_run, controls)
    recorded = run.manifest.params["cost_model"]
    assert view.cost_model == CostModel(**recorded)
    assert view.cost_model.fee_bps != EDITED_FEE_BPS
    # The net curve compounds the same costed returns the JSON ``net`` block summarised:
    # ``total_return`` is ``prod(1 + r) - 1`` and the curve's last point ``cumprod``'s
    # last element; both multiply the same floats in the same order.
    net_total = run.results[symbol]["backtest"]["net"]["total_return"]
    assert view.curves["strategy_net"].iloc[-1] == 1.0 + net_total


def test_controls_match_run_false_when_config_profile_edited(run: data.LoadedRun) -> None:
    symbol = run.symbols[0]
    profile = run.manifest.cost_profile
    controls = data.controls_from_run(run, symbol)
    assert data.controls_match_run(run, controls)
    edited_run = dataclasses.replace(run, config=_config_with_fee(run, profile, EDITED_FEE_BPS))
    assert not data.controls_match_run(edited_run, controls)
    view = data.compute_symbol_view(edited_run, controls)
    assert view.recomputed is True


def test_other_cost_profile_comes_from_live_config(run: data.LoadedRun) -> None:
    symbol = run.symbols[0]
    controls = dataclasses.replace(data.controls_from_run(run, symbol), cost_profile="crypto")
    view = data.compute_symbol_view(run, controls)
    expected = run.config["costs"]["profiles"]["crypto"]
    assert view.cost_model.fee_bps == float(expected["fee_bps"])
    assert view.cost_model.slippage_bps == float(expected["slippage_bps"])
    assert view.recomputed is True
    with pytest.raises(ValueError, match="nope"):
        data.compute_symbol_view(run, dataclasses.replace(controls, cost_profile="nope"))


def test_recorded_cost_model_falls_back_to_result_costs(run: data.LoadedRun) -> None:
    symbol = run.symbols[0]
    manifest = copy.deepcopy(run.manifest)
    recorded = manifest.params.pop("cost_model")
    old_run = dataclasses.replace(run, manifest=manifest)
    assert data.recorded_cost_model(old_run, symbol) == CostModel(**recorded)


def test_symbol_view_changed_window_recomputes_consistently(run: data.LoadedRun) -> None:
    symbol = run.symbols[1]
    base = data.controls_from_run(run, symbol)
    controls = dataclasses.replace(base, window=base.window + 5, refit="rolling", rolling_bars=100)
    view = data.compute_symbol_view(run, controls)
    assert view.recomputed is True
    assert view.matrix.shape == (N, N)
    np.testing.assert_allclose(view.matrix.sum(axis=1), np.ones(N), atol=REBUILD_TOL)
    assert ((view.matrix >= 0.0) & (view.matrix <= 1.0)).all()
    assert (view.matrix_lower <= view.matrix + REBUILD_TOL).all()
    assert (view.matrix_upper >= view.matrix - REBUILD_TOL).all()
    assert (view.stationary_lower <= view.stationary + REBUILD_TOL).all()
    assert (view.stationary_upper >= view.stationary - REBUILD_TOL).all()
    assert view.stationary.sum() == pytest.approx(1.0)
    assert -1.0 <= view.signal <= 1.0
    assert tuple(view.curves) == CURVE_KEYS
    strategy = view.curves["strategy_net"]
    assert not strategy.empty
    assert strategy.iloc[0] == 1.0
    assert (view.null_lower <= view.null_upper).all()
    pd.testing.assert_index_equal(view.null_lower.index, strategy.index)
    pd.testing.assert_index_equal(view.null_upper.index, strategy.index)
    assert view.null_lower.iloc[0] == 1.0
    assert view.null_upper.iloc[0] == 1.0
    # The JSON is untouched: the curves were built on a deep copy.
    assert run.results[symbol]["params"]["window"] == base.window
    assert run.results[symbol]["backtest"]["params"]["refit"] == base.refit


def test_symbol_view_short_history_gives_empty_curves_and_band(run: data.LoadedRun) -> None:
    symbol = run.symbols[0]
    controls = data.controls_from_run(run, symbol)
    short_closes = {symbol: run.closes[symbol].iloc[:100]}  # < min_train + 30
    short_run = dataclasses.replace(run, closes=short_closes)
    view = data.compute_symbol_view(short_run, controls)
    assert all(series.empty for series in view.curves.values())
    assert view.null_lower.empty and view.null_upper.empty


def test_symbol_view_unknown_symbol_raises(run: data.LoadedRun) -> None:
    controls = dataclasses.replace(data.controls_from_run(run, run.symbols[0]), symbol="ZZZ")
    with pytest.raises(ValueError, match="ZZZ"):
        data.compute_symbol_view(run, controls)


# ------------------------------------------------------------------------------ null band


def _returns() -> np.ndarray:
    return np.random.default_rng(SEED).normal(0.0, DAILY_VOL, N_RETURNS)


def test_null_equity_band_single_path_zero_costs_reproduces_random_sign_draw() -> None:
    rets = _returns()
    rng = np.random.default_rng(SEED)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(1, rets.size))
    expected = np.cumprod(1.0 + signs[0] * rets)
    lower, upper = data.null_equity_band(rets, ZERO_COSTS, n_paths=1, seed=SEED)
    np.testing.assert_array_equal(lower, expected)
    np.testing.assert_array_equal(upper, expected)


def test_null_equity_band_costs_lower_the_band() -> None:
    rets = _returns()
    free_lo, free_hi = data.null_equity_band(rets, ZERO_COSTS, n_paths=50, seed=SEED)
    paid_lo, paid_hi = data.null_equity_band(rets, HEAVY_COSTS, n_paths=50, seed=SEED)
    assert (free_lo <= free_hi).all()
    assert (paid_hi < free_hi).all()
    assert (paid_lo < free_lo).all()


def test_null_equity_band_empty_and_invalid() -> None:
    lower, upper = data.null_equity_band(np.empty(0), ZERO_COSTS, n_paths=5, seed=SEED)
    assert lower.size == 0 and upper.size == 0
    with pytest.raises(ValueError, match="n_paths"):
        data.null_equity_band(_returns(), ZERO_COSTS, n_paths=0, seed=SEED)


# ------------------------------------------------------------------------ P**n convergence


def test_nstep_sequence_identity_matrix_and_row_sums() -> None:
    seq = data.nstep_sequence(ERGODIC, 10)
    assert seq.shape == (11, N, N)
    np.testing.assert_array_equal(seq[0], np.eye(N))
    np.testing.assert_array_equal(seq[1], ERGODIC)
    np.testing.assert_allclose(seq.sum(axis=2), np.ones((11, N)), atol=REBUILD_TOL)
    with pytest.raises(ValueError, match="max_n"):
        data.nstep_sequence(ERGODIC, -1)


@settings(max_examples=25, deadline=None)
@given(
    rows=st.lists(
        st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=N, max_size=N),
        min_size=N,
        max_size=N,
    ),
    k=st.integers(min_value=0, max_value=30),
)
def test_nstep_sequence_rows_sum_to_one_for_any_stochastic_matrix(
    rows: list[list[float]], k: int
) -> None:
    matrix = np.array(rows) / np.array(rows).sum(axis=1, keepdims=True)
    seq = data.nstep_sequence(matrix, k)
    # Each power multiplies at most 30 row-stochastic matrices; float error grows roughly
    # linearly in the step count, far below 1e-9.
    np.testing.assert_allclose(seq[k].sum(axis=1), np.ones(N), atol=1e-9)


def test_convergence_distance_non_increasing_and_vanishes_for_ergodic_chain() -> None:
    dist = data.convergence_distance(ERGODIC, MAX_N)
    assert dist.shape == (MAX_N + 1,)
    assert dist[0] > 0.0  # the identity is never within epsilon of pi
    # TVD to the stationary vector is non-increasing for any Markov chain (contraction);
    # 1e-12 only absorbs rounding in the repeated products.
    assert (np.diff(dist) <= REBUILD_TOL).all()
    # By step 200 the slowest mode (|lambda_2| <= 0.75) has decayed to ~1e-25.
    assert dist[-1] < REBUILD_TOL
