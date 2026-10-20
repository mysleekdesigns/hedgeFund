"""Tests for hedgefund.backtest: metrics and the incremental walk-forward."""

import io
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hedgefund.backtest.metrics import (
    TRADING_DAYS_PER_YEAR,
    equity_curve,
    max_drawdown,
    sharpe_ratio,
)
from hedgefund.backtest.walk_forward import (
    DEFAULT_DEAD_BAND,
    DEFAULT_REFIT,
    DEFAULT_ROLLING_BARS,
    MIN_TEST_DAYS,
    POSITION_RULES,
    REFIT_MODES,
    WalkForwardResult,
    banded_position,
    run_walk_forward,
    walk_forward_backtest,
)
from hedgefund.regime.labels import BEAR, BULL, SIDEWAYS, label_regimes
from hedgefund.regime.markov import build_transition_matrix
from hedgefund.regime.signal import sign_position, signal_from_matrix

FIXTURES = Path(__file__).parent / "fixtures"
REPRODUCTION_DECIMALS = 6  # testing rule: companion reproduction is compared to 6 decimals

# Walk-forward parameters for synthetic tests; small so the tests stay fast.
SMALL_MIN_TRAIN = 40
SYNTHETIC_DAYS = 400
DAILY_VOL = 0.01  # 1% daily vol, roughly SPY-like
LABEL_WINDOW = 20
LABEL_THRESHOLD = 0.02
# Phase 3: a rolling window short enough to actually roll inside SYNTHETIC_DAYS, and a dead
# band wide enough to swallow most day-to-day changes of a conviction signal in [-1, 1].
SHORT_ROLLING_BARS = 60
WIDE_DEAD_BAND = 0.25
# Review fix W2: the sign rules apply the band as hysteresis on the signal. A 700-day walk
# with a 60-day training window emits enough zero crossings for a 0.1 band to bite.
HYSTERESIS_DAYS = 700
HYSTERESIS_MIN_TRAIN = 60
SIGNAL_DEAD_BAND = 0.1


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def synthetic_close(n_days: int, seed: int, drift: float = 0.0002) -> pd.Series:
    """Geometric random walk with business-day index and an explicit seed."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, DAILY_VOL, size=n_days)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    index = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.Series(prices, index=index, name="SYNTH")


def reference_walk_forward(close: pd.Series, labels: pd.Series, min_train: int) -> dict:
    """The companion script's walk_forward_backtest, copied verbatim as the oracle."""
    daily_returns = close.pct_change().dropna()
    common_index = labels.index.intersection(daily_returns.index)
    labels = labels.loc[common_index]
    daily_returns = daily_returns.loc[common_index]

    if len(labels) < min_train + 30:
        return {"sharpe": float("nan"), "max_drawdown": float("nan"), "n_trades": 0}

    lab = np.asarray(labels, dtype=int)
    rets = daily_returns.to_numpy(dtype=float)

    counts = np.zeros((3, 3), dtype=float)
    for i in range(min_train - 1):
        counts[lab[i], lab[i + 1]] += 1.0

    strategy_returns = np.empty(len(lab) - 1 - min_train, dtype=float)
    for k, t in enumerate(range(min_train, len(lab) - 1)):
        row_sums = counts.sum(axis=1, keepdims=True)
        safe = np.where(row_sums == 0, 1.0, row_sums)
        p_t = counts / safe
        current_state = lab[t]
        signal = float(p_t[current_state, 2] - p_t[current_state, 0])
        position = float(np.sign(signal))
        strategy_returns[k] = position * rets[t + 1]
        counts[lab[t - 1], lab[t]] += 1.0

    sr = strategy_returns
    std = sr.std(ddof=1) if len(sr) > 1 else 0.0
    sharpe = float("nan") if std == 0 or not np.isfinite(std) else float(sr.mean() / std * 252**0.5)
    equity = (1.0 + sr).cumprod()
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min()) if len(drawdown) else float("nan")
    return {"sharpe": sharpe, "max_drawdown": max_dd, "n_trades": len(sr)}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_equity_curve_is_cumprod() -> None:
    returns = np.array([0.1, -0.5, 1.0])
    assert np.array_equal(equity_curve(returns), np.array([1.1, 0.55, 1.1]))


def test_equity_curve_empty() -> None:
    assert equity_curve(np.array([])).shape == (0,)


def test_sharpe_constant_returns_is_nan() -> None:
    """Zero sample std gives NaN. Use exactly representable constants: for a value like
    0.001 numpy's pairwise mean leaves a std of ~1e-19, which the companion (and we,
    for bit-identity) treat as a real, enormous Sharpe rather than NaN."""
    assert math.isnan(sharpe_ratio(np.full(100, 0.25)))
    assert math.isnan(sharpe_ratio(np.zeros(100)))


def test_sharpe_single_observation_is_nan() -> None:
    assert math.isnan(sharpe_ratio(np.array([0.01])))


def test_sharpe_empty_is_nan() -> None:
    assert math.isnan(sharpe_ratio(np.array([])))


def test_sharpe_nan_input_is_nan() -> None:
    assert math.isnan(sharpe_ratio(np.array([0.01, float("nan"), 0.02])))


def test_sharpe_hand_value() -> None:
    # Returns alternate +1% / -1% with one extra +1%: mean 0.002, sample std known.
    returns = np.array([0.01, -0.01, 0.01, -0.01, 0.01])
    expected = returns.mean() / returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert sharpe_ratio(returns) == expected


def test_sharpe_periods_per_year_scales_by_sqrt() -> None:
    returns = np.array([0.01, -0.02, 0.03, 0.0, 0.01])
    assert sharpe_ratio(returns, periods_per_year=4) == pytest.approx(
        sharpe_ratio(returns, periods_per_year=1) * 2.0
    )


def test_sharpe_matches_companion_formula_bitwise() -> None:
    rng = np.random.default_rng(3)
    sr = rng.normal(0.0005, 0.01, size=500)
    std = sr.std(ddof=1)
    assert sharpe_ratio(sr) == float(sr.mean() / std * np.sqrt(252))


def test_max_drawdown_down_then_up() -> None:
    # Equity 1.0 -> 0.9 -> 0.99: the trough is 10% below the peak.
    assert max_drawdown(np.array([0.0, -0.10, 0.10])) == pytest.approx(-0.10)


def test_max_drawdown_ignores_a_loss_on_the_very_first_day() -> None:
    """Companion behaviour kept for bit-identity: the equity curve starts at the first
    return's equity (no leading 1.0), so a first-day loss sets the initial peak and is
    not counted as a drawdown. Equity 0.9 -> 0.99 has zero drawdown."""
    assert max_drawdown(np.array([-0.10, 0.10])) == 0.0


def test_max_drawdown_monotone_rise_is_zero() -> None:
    assert max_drawdown(np.array([0.01, 0.02, 0.03])) == 0.0


def test_max_drawdown_is_deepest_not_last() -> None:
    # -20% early, full recovery, then -5%: the answer is the early -20%.
    returns = np.array([0.0, -0.2, 0.25, 0.0, -0.05])
    assert max_drawdown(returns) == pytest.approx(-0.2)


def test_max_drawdown_empty_is_nan() -> None:
    assert math.isnan(max_drawdown(np.array([])))


def test_max_drawdown_matches_companion_formula_bitwise() -> None:
    rng = np.random.default_rng(5)
    sr = rng.normal(0.0, 0.01, size=500)
    equity = (1.0 + sr).cumprod()
    running_max = np.maximum.accumulate(equity)
    assert max_drawdown(sr) == float(((equity - running_max) / running_max).min())


@pytest.mark.parametrize("func", [sharpe_ratio, max_drawdown, equity_curve])
def test_metrics_reject_2d_input(func: Callable[[np.ndarray], object]) -> None:
    with pytest.raises(ValueError, match="1-D"):
        func(np.ones((3, 2)))


# --------------------------------------------------------------------------- #
# walk-forward
# --------------------------------------------------------------------------- #
def test_short_history_short_circuits_to_nan() -> None:
    window = 20
    # The labeller drops `window` rows, leaving len(labels) one short of the cutoff.
    close = synthetic_close(SMALL_MIN_TRAIN + MIN_TEST_DAYS + window - 1, seed=1)
    labels = label_regimes(close, window=window, threshold=0.02)
    assert len(labels) == SMALL_MIN_TRAIN + MIN_TEST_DAYS - 1
    out = walk_forward_backtest(close, labels, min_train=SMALL_MIN_TRAIN)
    assert math.isnan(out["sharpe"])
    assert math.isnan(out["max_drawdown"])
    assert out["n_trades"] == 0


def test_short_history_returns_empty_series() -> None:
    close = synthetic_close(SMALL_MIN_TRAIN + 10, seed=1)
    labels = label_regimes(close, window=20, threshold=0.02)
    result = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN)
    assert result.strategy_returns.empty
    assert result.positions.empty
    assert result.signals.empty
    summary = result.summary()
    assert math.isnan(summary["sharpe"])
    assert math.isnan(summary["max_drawdown"])
    assert summary["n_trades"] == 0


def test_exact_cutoff_runs() -> None:
    """len(labels) == min_train + MIN_TEST_DAYS is enough to run (strict less-than rule)."""
    window = 20
    close = synthetic_close(SMALL_MIN_TRAIN + MIN_TEST_DAYS + window, seed=2)
    labels = label_regimes(close, window=window, threshold=0.02)
    assert len(labels) == SMALL_MIN_TRAIN + MIN_TEST_DAYS
    out = walk_forward_backtest(close, labels, min_train=SMALL_MIN_TRAIN)
    assert out["n_trades"] == len(labels) - 1 - SMALL_MIN_TRAIN


def test_n_trades_counts_every_decision_day() -> None:
    close = synthetic_close(SYNTHETIC_DAYS, seed=3)
    labels = label_regimes(close, window=20, threshold=0.02)
    out = walk_forward_backtest(close, labels, min_train=SMALL_MIN_TRAIN)
    assert out["n_trades"] == len(labels) - 1 - SMALL_MIN_TRAIN


def test_incremental_counts_equal_from_scratch_matrix() -> None:
    """The O(n) running counts reproduce build_transition_matrix(labels[:t]) bit for bit.

    Re-derive the per-step matrix the walk-forward used from the signal it emitted:
    for each decision day t, signal_t must equal signal_from_matrix(P(labels[:t]), lab[t])
    where P is the from-scratch MLE estimate. Checked at every t, not a sample.
    """
    close = synthetic_close(SYNTHETIC_DAYS, seed=4)
    labels = label_regimes(close, window=20, threshold=0.02)
    result = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN)

    daily_returns = close.pct_change().dropna()
    common = labels.index.intersection(daily_returns.index)
    lab = labels.loc[common].to_numpy(dtype=int)

    checked_ts = [SMALL_MIN_TRAIN, SMALL_MIN_TRAIN + 1, SMALL_MIN_TRAIN + 57, len(lab) - 2]
    for t in checked_ts:
        matrix_scratch = build_transition_matrix(lab[:t])
        expected = signal_from_matrix(matrix_scratch, int(lab[t]))
        assert result.signals.loc[common[t]] == expected  # exact, not approx

    # And the full signal vector equals the from-scratch rebuild at every t.
    scratch_signals = np.array(
        [
            signal_from_matrix(build_transition_matrix(lab[:t]), int(lab[t]))
            for t in range(SMALL_MIN_TRAIN, len(lab) - 1)
        ]
    )
    assert np.array_equal(result.signals.to_numpy(), scratch_signals)


def test_running_counts_track_label_slices() -> None:
    """Replay the companion's seeding and sliding rule and compare raw counts with np.add.at."""
    rng = np.random.default_rng(11)
    lab = rng.integers(0, 3, size=300)
    min_train = SMALL_MIN_TRAIN

    counts = np.zeros((3, 3), dtype=float)
    for i in range(min_train - 1):
        counts[lab[i], lab[i + 1]] += 1.0
    for t in range(min_train, len(lab) - 1):
        scratch = np.zeros((3, 3), dtype=float)
        np.add.at(scratch, (lab[:t][:-1], lab[:t][1:]), 1.0)
        assert np.array_equal(counts, scratch), f"counts diverge at t={t}"
        counts[lab[t - 1], lab[t]] += 1.0


def test_positions_are_sign_of_signals() -> None:
    close = synthetic_close(SYNTHETIC_DAYS, seed=5)
    labels = label_regimes(close, window=20, threshold=0.02)
    result = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN)
    expected = np.array([sign_position(s) for s in result.signals.to_numpy()])
    assert np.array_equal(result.positions.to_numpy(), expected)
    assert set(np.unique(result.positions)) <= {-1.0, 0.0, 1.0}
    assert result.positions.index.equals(result.signals.index)


def test_strategy_return_is_position_times_next_day_return() -> None:
    close = synthetic_close(SYNTHETIC_DAYS, seed=6)
    labels = label_regimes(close, window=20, threshold=0.02)
    result = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN)

    daily_returns = close.pct_change().dropna()
    common = labels.index.intersection(daily_returns.index)
    rets = daily_returns.loc[common]

    # strategy_returns is indexed by the earned date t+1; positions by the decision date t.
    earned_dates = result.strategy_returns.index
    decision_dates = result.positions.index
    assert len(earned_dates) == len(decision_dates)
    assert earned_dates[0] == common[SMALL_MIN_TRAIN + 1]
    assert decision_dates[0] == common[SMALL_MIN_TRAIN]
    assert earned_dates[-1] == common[-1]
    expected = result.positions.to_numpy() * rets.loc[earned_dates].to_numpy()
    assert np.array_equal(result.strategy_returns.to_numpy(), expected)


def test_summary_matches_walk_forward_backtest() -> None:
    close = synthetic_close(SYNTHETIC_DAYS, seed=8)
    labels = label_regimes(close, window=20, threshold=0.02)
    summary = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN).summary()
    direct = walk_forward_backtest(close, labels, min_train=SMALL_MIN_TRAIN)
    assert summary == direct
    assert set(summary) == {"sharpe", "max_drawdown", "n_trades"}
    assert isinstance(summary["n_trades"], int)


def test_matches_companion_reference_on_synthetic_data() -> None:
    """Against a verbatim copy of the companion's loop: every number identical."""
    for seed in (21, 22, 23):
        close = synthetic_close(SYNTHETIC_DAYS, seed=seed)
        labels = label_regimes(close, window=20, threshold=0.02)
        ours = walk_forward_backtest(close, labels, min_train=SMALL_MIN_TRAIN)
        ref = reference_walk_forward(close, labels, SMALL_MIN_TRAIN)
        assert ours["n_trades"] == ref["n_trades"]
        assert ours["sharpe"] == ref["sharpe"]
        assert ours["max_drawdown"] == ref["max_drawdown"]


def test_alignment_uses_intersection_of_label_and_return_index() -> None:
    """Labels on dates with no return (the first close) are dropped, as the companion does."""
    close = synthetic_close(SYNTHETIC_DAYS, seed=9)
    # Labels on every close date, including the very first one, which has no return.
    labels = pd.Series(SIDEWAYS, index=close.index, dtype=int)
    labels.iloc[::3] = BULL
    labels.iloc[1::7] = BEAR
    ours = walk_forward_backtest(close, labels, min_train=SMALL_MIN_TRAIN)
    ref = reference_walk_forward(close, labels, SMALL_MIN_TRAIN)
    assert ours == ref
    assert ours["n_trades"] == len(close) - 1 - 1 - SMALL_MIN_TRAIN


def test_result_is_frozen() -> None:
    close = synthetic_close(SYNTHETIC_DAYS, seed=10)
    labels = label_regimes(close, window=20, threshold=0.02)
    result = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN)
    assert isinstance(result, WalkForwardResult)
    with pytest.raises(AttributeError):
        result.signals = result.signals  # type: ignore[misc]


def test_rejects_non_positive_min_train() -> None:
    close = synthetic_close(100, seed=12)
    labels = label_regimes(close, window=20, threshold=0.02)
    with pytest.raises(ValueError, match="min_train"):
        run_walk_forward(close, labels, min_train=0)


def test_rejects_out_of_range_labels() -> None:
    close = synthetic_close(SYNTHETIC_DAYS, seed=13)
    labels = pd.Series(SIDEWAYS, index=close.index, dtype=int)
    labels.iloc[50] = 3
    with pytest.raises(ValueError, match="labels"):
        run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN)


def test_default_min_train_matches_config() -> None:
    import yaml

    config = yaml.safe_load((Path(__file__).parents[1] / "config" / "default.yaml").read_text())
    from hedgefund.backtest.walk_forward import DEFAULT_MIN_TRAIN

    assert config["backtest"]["min_train"] == DEFAULT_MIN_TRAIN


# --------------------------------------------------------------------------- #
# Frozen-fixture reproduction of the companion's walk-forward numbers
# --------------------------------------------------------------------------- #
def test_walk_forward_reproduces_companion_on_spy_fixture() -> None:
    companion = json.loads((FIXTURES / "spy_10y.companion.json").read_text())
    params = companion["result"]["params"]
    expected = companion["result"]["walk_forward"]

    close = pd.read_parquet(FIXTURES / "spy_10y.parquet")["close"].dropna()
    labels = label_regimes(close, window=params["window"], threshold=params["threshold"])
    out = walk_forward_backtest(close, labels, min_train=params["min_train"])

    assert out["n_trades"] == expected["n_trades"]
    assert out["sharpe"] == pytest.approx(expected["sharpe"], abs=10**-REPRODUCTION_DECIMALS)
    assert out["max_drawdown"] == pytest.approx(
        expected["max_drawdown"], abs=10**-REPRODUCTION_DECIMALS
    )


def test_walk_forward_is_bit_identical_to_companion_after_its_csv_round_trip() -> None:
    """Exact equality, not 6 decimals.

    The frozen JSON was produced by exporting the parquet close to CSV and running the
    companion on that CSV. The text round trip shifts closes by ~6e-14, which moves the
    Sharpe and drawdown in the 15th digit. Feeding the same round-tripped series through
    our port reproduces the frozen numbers to the last bit, which pins down that the
    only difference between fixture and port is that export, not the arithmetic.
    """
    companion = json.loads((FIXTURES / "spy_10y.companion.json").read_text())
    params = companion["result"]["params"]
    expected = companion["result"]["walk_forward"]

    close = pd.read_parquet(FIXTURES / "spy_10y.parquet")["close"].dropna()
    buffer = io.StringIO()
    pd.DataFrame({"date": close.index, "close": close.to_numpy()}).to_csv(buffer, index=False)
    frame = pd.read_csv(io.StringIO(buffer.getvalue()))
    round_tripped = pd.Series(
        frame["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(pd.to_datetime(frame["date"]))
    )

    labels = label_regimes(round_tripped, window=params["window"], threshold=params["threshold"])
    out = walk_forward_backtest(round_tripped, labels, min_train=params["min_train"])
    assert out == expected


# --------------------------------------------------------------------------- #
# Phase 3: refit modes, position rules, dead band (plan.md Phase 3, Fix 6)
# --------------------------------------------------------------------------- #
def reference_daily_series(
    close: pd.Series, labels: pd.Series, min_train: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The companion loop again, returning its per-day arrays instead of the summary.

    Phase 1 oracle for the per-day series: (strategy_returns, positions, signals).
    """
    daily_returns = close.pct_change().dropna()
    common_index = labels.index.intersection(daily_returns.index)
    lab = np.asarray(labels.loc[common_index], dtype=int)
    rets = daily_returns.loc[common_index].to_numpy(dtype=float)

    counts = np.zeros((3, 3), dtype=float)
    for i in range(min_train - 1):
        counts[lab[i], lab[i + 1]] += 1.0

    n_steps = len(lab) - 1 - min_train
    strategy_returns = np.empty(n_steps, dtype=float)
    positions = np.empty(n_steps, dtype=float)
    signals = np.empty(n_steps, dtype=float)
    for k, t in enumerate(range(min_train, len(lab) - 1)):
        row_sums = counts.sum(axis=1, keepdims=True)
        safe = np.where(row_sums == 0, 1.0, row_sums)
        p_t = counts / safe
        current_state = lab[t]
        signal = float(p_t[current_state, 2] - p_t[current_state, 0])
        position = float(np.sign(signal))
        signals[k] = signal
        positions[k] = position
        strategy_returns[k] = position * rets[t + 1]
        counts[lab[t - 1], lab[t]] += 1.0
    return strategy_returns, positions, signals


def _synthetic_case(seed: int) -> tuple[pd.Series, pd.Series, np.ndarray, pd.DatetimeIndex]:
    """Close, labels, the aligned integer label array and its index for one seed."""
    close = synthetic_close(SYNTHETIC_DAYS, seed=seed)
    labels = label_regimes(close, window=LABEL_WINDOW, threshold=LABEL_THRESHOLD)
    common = labels.index.intersection(close.pct_change().dropna().index)
    return close, labels, labels.loc[common].to_numpy(dtype=int), common


def _assert_results_identical(a: WalkForwardResult, b: WalkForwardResult) -> None:
    for name in ("strategy_returns", "positions", "signals", "asset_returns"):
        left, right = getattr(a, name), getattr(b, name)
        assert left.index.equals(right.index), name
        assert np.array_equal(left.to_numpy(), right.to_numpy(), equal_nan=True), name


def test_phase3_defaults_are_phase1_values() -> None:
    assert DEFAULT_REFIT == "expanding"
    assert REFIT_MODES == ("expanding", "rolling")
    assert DEFAULT_DEAD_BAND == 0.0
    assert POSITION_RULES[0] == "sign"


def test_default_kwargs_reproduce_phase1_per_day_series_bit_for_bit() -> None:
    """The Phase 3 walk-forward with defaults is the Phase 1 loop, every entry identical."""
    for seed in (31, 32, 33):
        close, labels, _, _ = _synthetic_case(seed)
        ref_returns, ref_positions, ref_signals = reference_daily_series(
            close, labels, SMALL_MIN_TRAIN
        )
        result = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN)
        assert np.array_equal(result.strategy_returns.to_numpy(), ref_returns)
        assert np.array_equal(result.positions.to_numpy(), ref_positions)
        assert np.array_equal(result.signals.to_numpy(), ref_signals)


def test_explicit_phase1_kwargs_equal_defaults() -> None:
    close, labels, _, _ = _synthetic_case(34)
    implicit = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN)
    explicit = run_walk_forward(
        close,
        labels,
        min_train=SMALL_MIN_TRAIN,
        refit="expanding",
        rolling_bars=DEFAULT_ROLLING_BARS,
        rule="sign",
        dead_band=0.0,
    )
    _assert_results_identical(implicit, explicit)
    assert implicit.summary() == walk_forward_backtest(close, labels, min_train=SMALL_MIN_TRAIN)


def test_asset_returns_are_the_next_day_simple_returns_on_the_earned_index() -> None:
    close, labels, _, common = _synthetic_case(35)
    result = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN)
    rets = close.pct_change().dropna().loc[common]
    assert result.asset_returns.index.equals(result.strategy_returns.index)
    assert len(result.asset_returns) == len(result.positions)
    expected = rets.loc[result.strategy_returns.index].to_numpy()
    assert np.array_equal(result.asset_returns.to_numpy(), expected)


@pytest.mark.parametrize("rule", POSITION_RULES)
def test_strategy_return_is_held_position_times_asset_return_for_every_rule(rule: str) -> None:
    close, labels, _, _ = _synthetic_case(36)
    result = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN, rule=rule)
    expected = result.positions.to_numpy() * result.asset_returns.to_numpy()
    assert np.array_equal(result.strategy_returns.to_numpy(), expected)


def test_empty_result_has_empty_asset_returns() -> None:
    close = synthetic_close(SMALL_MIN_TRAIN + 10, seed=1)
    labels = label_regimes(close, window=LABEL_WINDOW, threshold=LABEL_THRESHOLD)
    result = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN, refit="rolling")
    assert result.asset_returns.empty
    assert result.strategy_returns.empty


def test_summary_keys_unchanged_for_every_rule_and_refit() -> None:
    close, labels, _, _ = _synthetic_case(37)
    for refit in REFIT_MODES:
        for rule in POSITION_RULES:
            summary = run_walk_forward(
                close,
                labels,
                min_train=SMALL_MIN_TRAIN,
                refit=refit,
                rolling_bars=SHORT_ROLLING_BARS,
                rule=rule,
            ).summary()
            assert set(summary) == {"sharpe", "max_drawdown", "n_trades"}
            assert summary["n_trades"] == len(labels) - 1 - SMALL_MIN_TRAIN


# ---- rolling refit -------------------------------------------------------- #
def test_rolling_matrix_equals_from_scratch_mle_on_trailing_window() -> None:
    """Rolling refit: at every t the matrix is build_transition_matrix on
    labels[max(0, t - rolling_bars):t], recovered through the emitted signal."""
    close, labels, lab, common = _synthetic_case(41)
    result = run_walk_forward(
        close, labels, min_train=SMALL_MIN_TRAIN, refit="rolling", rolling_bars=SHORT_ROLLING_BARS
    )

    checked_ts = [
        SMALL_MIN_TRAIN,  # window still partial: starts at 0
        SHORT_ROLLING_BARS - 1,  # last decision before the window is full
        SHORT_ROLLING_BARS,  # first decision with a full window
        SHORT_ROLLING_BARS + 1,  # first decision after a transition dropped out
        SMALL_MIN_TRAIN + 157,
        len(lab) - 2,
    ]
    for t in checked_ts:
        window = lab[max(0, t - SHORT_ROLLING_BARS) : t]
        expected = signal_from_matrix(build_transition_matrix(window), int(lab[t]))
        assert result.signals.loc[common[t]] == expected, f"t={t}"  # exact, not approx

    scratch_signals = np.array(
        [
            signal_from_matrix(
                build_transition_matrix(lab[max(0, t - SHORT_ROLLING_BARS) : t]), int(lab[t])
            )
            for t in range(SMALL_MIN_TRAIN, len(lab) - 1)
        ]
    )
    assert np.array_equal(result.signals.to_numpy(), scratch_signals)


def test_rolling_counts_track_window_slices_with_random_labels() -> None:
    """Replay the add-one / drop-one sliding rule on random labels and compare with
    np.add.at on the exact window slice at every step, including a window shorter than
    min_train so the seed itself must be truncated."""
    rng = np.random.default_rng(42)
    lab = rng.integers(0, 3, size=300)
    for min_train, rolling_bars in ((SMALL_MIN_TRAIN, 25), (SMALL_MIN_TRAIN, 60), (10, 2)):
        counts = np.zeros((3, 3), dtype=float)
        for i in range(max(0, min_train - rolling_bars), min_train - 1):
            counts[lab[i], lab[i + 1]] += 1.0
        for t in range(min_train, len(lab) - 1):
            window = lab[max(0, t - rolling_bars) : t]
            scratch = np.zeros((3, 3), dtype=float)
            np.add.at(scratch, (window[:-1], window[1:]), 1.0)
            assert np.array_equal(counts, scratch), f"rolling_bars={rolling_bars}, t={t}"
            counts[lab[t - 1], lab[t]] += 1.0
            if t >= rolling_bars:
                counts[lab[t - rolling_bars], lab[t - rolling_bars + 1]] -= 1.0


def test_rolling_window_longer_than_history_equals_expanding_bit_for_bit() -> None:
    close, labels, lab, _ = _synthetic_case(43)
    expanding = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN)
    for rolling_bars in (len(lab), len(lab) + 1, DEFAULT_ROLLING_BARS):
        assert rolling_bars >= len(lab)
        rolling = run_walk_forward(
            close, labels, min_train=SMALL_MIN_TRAIN, refit="rolling", rolling_bars=rolling_bars
        )
        _assert_results_identical(expanding, rolling)


def test_short_rolling_window_actually_differs_from_expanding() -> None:
    close, labels, _, _ = _synthetic_case(44)
    expanding = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN)
    rolling = run_walk_forward(
        close, labels, min_train=SMALL_MIN_TRAIN, refit="rolling", rolling_bars=SHORT_ROLLING_BARS
    )
    assert not np.array_equal(expanding.signals.to_numpy(), rolling.signals.to_numpy())
    # Before any transition has dropped out (t < rolling_bars) the two agree exactly.
    first_full = rolling.signals.index[SHORT_ROLLING_BARS - SMALL_MIN_TRAIN]
    assert np.array_equal(
        expanding.signals.loc[:first_full].to_numpy(), rolling.signals.loc[:first_full].to_numpy()
    )


def test_rolling_matrix_counts_stay_non_negative_and_rows_stochastic() -> None:
    """Dropping transitions must never drive a count below zero; signals stay in [-1, 1]."""
    close, labels, _, _ = _synthetic_case(45)
    result = run_walk_forward(
        close, labels, min_train=SMALL_MIN_TRAIN, refit="rolling", rolling_bars=SHORT_ROLLING_BARS
    )
    signals = result.signals.to_numpy()
    assert np.all(np.isfinite(signals))
    assert np.all(signals >= -1.0) and np.all(signals <= 1.0)


@pytest.mark.parametrize("bad_refit", ["expanding ", "Rolling", "rolling:756", "", None])
def test_rejects_unknown_refit_mode(bad_refit: object) -> None:
    close, labels, _, _ = _synthetic_case(46)
    with pytest.raises(ValueError, match="refit"):
        run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN, refit=bad_refit)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_bars", [1, 0, -5])
def test_rejects_rolling_bars_below_two(bad_bars: int) -> None:
    close, labels, _, _ = _synthetic_case(47)
    with pytest.raises(ValueError, match="rolling_bars"):
        run_walk_forward(
            close, labels, min_train=SMALL_MIN_TRAIN, refit="rolling", rolling_bars=bad_bars
        )
    # The bound is checked even in expanding mode: a bad option is a bad option.
    with pytest.raises(ValueError, match="rolling_bars"):
        run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN, rolling_bars=bad_bars)


def test_rolling_bars_of_two_is_the_smallest_valid_window() -> None:
    """A two-label window holds exactly one transition: the matrix is one-hot per step."""
    close, labels, lab, _ = _synthetic_case(48)
    result = run_walk_forward(
        close, labels, min_train=SMALL_MIN_TRAIN, refit="rolling", rolling_bars=2
    )
    expected = np.array(
        [
            signal_from_matrix(build_transition_matrix(lab[t - 2 : t]), int(lab[t]))
            for t in range(SMALL_MIN_TRAIN, len(lab) - 1)
        ]
    )
    assert np.array_equal(result.signals.to_numpy(), expected)
    assert set(np.unique(result.signals)) <= {-1.0, 0.0, 1.0}


# ---- position rules ------------------------------------------------------- #
def test_conviction_positions_equal_signals() -> None:
    close, labels, _, _ = _synthetic_case(51)
    result = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN, rule="conviction")
    assert np.array_equal(result.positions.to_numpy(), result.signals.to_numpy())
    assert result.positions.abs().max() <= 1.0
    # A real conviction series is not just signs.
    assert not set(np.unique(result.positions)) <= {-1.0, 0.0, 1.0}


def test_persistence_positions_equal_signal_times_stay_probability() -> None:
    close, labels, lab, _ = _synthetic_case(52)
    result = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN, rule="persistence")
    expected = []
    for t in range(SMALL_MIN_TRAIN, len(lab) - 1):
        matrix = build_transition_matrix(lab[:t])
        state = int(lab[t])
        expected.append(signal_from_matrix(matrix, state) * matrix[state, state])
    assert np.array_equal(result.positions.to_numpy(), np.array(expected))
    assert np.all(np.abs(result.positions.to_numpy()) <= np.abs(result.signals.to_numpy()))


def test_long_only_positions_are_sign_floored_at_zero() -> None:
    close, labels, _, _ = _synthetic_case(53)
    sign = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN, rule="sign")
    long_only = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN, rule="long_only")
    assert set(np.unique(long_only.positions)) <= {0.0, 1.0}
    assert np.array_equal(
        long_only.positions.to_numpy(), np.maximum(sign.positions.to_numpy(), 0.0)
    )
    assert np.array_equal(long_only.signals.to_numpy(), sign.signals.to_numpy())
    # The sign rule did go short at least once, so long-only is genuinely different.
    assert (sign.positions < 0).any()


def test_signals_do_not_depend_on_the_rule() -> None:
    close, labels, _, _ = _synthetic_case(54)
    base = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN)
    for rule in POSITION_RULES:
        other = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN, rule=rule)
        assert np.array_equal(other.signals.to_numpy(), base.signals.to_numpy())
        assert np.array_equal(other.asset_returns.to_numpy(), base.asset_returns.to_numpy())


@pytest.mark.parametrize("bad_rule", ["signs", "Conviction", "", None])
def test_rejects_unknown_rule(bad_rule: object) -> None:
    close, labels, _, _ = _synthetic_case(55)
    with pytest.raises(ValueError, match="rule"):
        run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN, rule=bad_rule)  # type: ignore[arg-type]


# ---- dead band (Fix 6 threshold rebalancing) ------------------------------ #
def _position_changes(positions: pd.Series) -> int:
    """Number of steps where the held position differs from the previous one (first from flat)."""
    held = positions.to_numpy()
    previous = np.concatenate(([0.0], held[:-1]))
    return int(np.count_nonzero(held != previous))


def test_zero_dead_band_is_identical_to_phase1() -> None:
    close, labels, _, _ = _synthetic_case(61)
    base = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN)
    banded = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN, dead_band=0.0)
    _assert_results_identical(base, banded)
    ref_returns, ref_positions, _ = reference_daily_series(close, labels, SMALL_MIN_TRAIN)
    assert np.array_equal(banded.positions.to_numpy(), ref_positions)
    assert np.array_equal(banded.strategy_returns.to_numpy(), ref_returns)


@pytest.mark.parametrize("rule", ["conviction", "persistence"])
def test_wide_dead_band_strictly_reduces_position_changes(rule: str) -> None:
    close, labels, _, _ = _synthetic_case(62)
    tight = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN, rule=rule, dead_band=0.0)
    wide = run_walk_forward(
        close, labels, min_train=SMALL_MIN_TRAIN, rule=rule, dead_band=WIDE_DEAD_BAND
    )
    assert _position_changes(wide.positions) < _position_changes(tight.positions)
    assert _position_changes(wide.positions) > 0
    # Turnover in notional terms shrinks too.
    assert np.abs(np.diff(wide.positions.to_numpy())).sum() < (
        np.abs(np.diff(tight.positions.to_numpy())).sum()
    )


def test_dead_band_positions_only_take_values_that_were_targets() -> None:
    close, labels, _, _ = _synthetic_case(63)
    result = run_walk_forward(
        close, labels, min_train=SMALL_MIN_TRAIN, rule="conviction", dead_band=WIDE_DEAD_BAND
    )
    targets = set(result.signals.to_numpy().tolist()) | {0.0}  # conviction target == signal
    assert set(result.positions.to_numpy().tolist()) <= targets


def test_dead_band_follows_the_held_target_recurrence() -> None:
    """Replay held = target when |target - held| > band, starting flat, on the emitted
    conviction signals (which are the targets) and compare position by position."""
    close, labels, _, _ = _synthetic_case(64)
    result = run_walk_forward(
        close, labels, min_train=SMALL_MIN_TRAIN, rule="conviction", dead_band=WIDE_DEAD_BAND
    )
    held = 0.0
    expected = np.empty(len(result.signals), dtype=float)
    for k, target in enumerate(result.signals.to_numpy()):
        if abs(target - held) > WIDE_DEAD_BAND:
            held = target
        expected[k] = held
    assert np.array_equal(result.positions.to_numpy(), expected)
    assert np.array_equal(
        result.strategy_returns.to_numpy(), expected * result.asset_returns.to_numpy()
    )


def test_dead_band_holds_flat_until_the_first_target_clears_it() -> None:
    close, labels, _, _ = _synthetic_case(65)
    result = run_walk_forward(
        close, labels, min_train=SMALL_MIN_TRAIN, rule="conviction", dead_band=WIDE_DEAD_BAND
    )
    signals = result.signals.to_numpy()
    positions = result.positions.to_numpy()
    first_clear = int(np.argmax(np.abs(signals) > WIDE_DEAD_BAND))
    assert np.abs(signals[first_clear]) > WIDE_DEAD_BAND
    assert np.all(positions[:first_clear] == 0.0)
    assert positions[first_clear] == signals[first_clear]


def test_dead_band_wider_than_two_never_trades() -> None:
    """Every target lies in [-1, 1], so a band of 2 can never be exceeded from flat."""
    close, labels, _, _ = _synthetic_case(66)
    result = run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN, dead_band=2.0)
    assert np.all(result.positions.to_numpy() == 0.0)
    assert np.all(result.strategy_returns.to_numpy() == 0.0)


def _one_way_turnover(positions: pd.Series) -> float:
    """Sum of |Δposition| with the first position entered from flat."""
    held = positions.to_numpy()
    return float(np.abs(np.diff(np.concatenate(([0.0], held)))).sum())


def _hysteresis_case(seed: int) -> tuple[pd.Series, pd.Series]:
    close = synthetic_close(HYSTERESIS_DAYS, seed=seed)
    return close, label_regimes(close, window=LABEL_WINDOW, threshold=LABEL_THRESHOLD)


# ---- W2: the sign rules apply the band as hysteresis on the signal --------- #
def test_sign_rule_signal_dead_band_lowers_turnover_and_keeps_unit_positions() -> None:
    """A band below 1 used to be a no-op in position space; on the signal it must bite."""
    close, labels = _hysteresis_case(71)
    plain = run_walk_forward(close, labels, min_train=HYSTERESIS_MIN_TRAIN, rule="sign")
    banded = run_walk_forward(
        close, labels, min_train=HYSTERESIS_MIN_TRAIN, rule="sign", dead_band=SIGNAL_DEAD_BAND
    )
    assert _one_way_turnover(banded.positions) < _one_way_turnover(plain.positions)
    assert _one_way_turnover(banded.positions) > 0.0
    assert set(np.unique(banded.positions)) <= {-1.0, 0.0, 1.0}
    assert np.array_equal(banded.signals.to_numpy(), plain.signals.to_numpy())


def test_long_only_signal_dead_band_lowers_turnover_and_stays_in_zero_one() -> None:
    close, labels = _hysteresis_case(72)
    plain = run_walk_forward(close, labels, min_train=HYSTERESIS_MIN_TRAIN, rule="long_only")
    banded = run_walk_forward(
        close, labels, min_train=HYSTERESIS_MIN_TRAIN, rule="long_only", dead_band=SIGNAL_DEAD_BAND
    )
    assert _one_way_turnover(banded.positions) < _one_way_turnover(plain.positions)
    assert _one_way_turnover(banded.positions) > 0.0
    assert set(np.unique(banded.positions)) <= {0.0, 1.0}


def test_banded_position_sign_hysteresis_on_a_hand_sequence() -> None:
    """Hand-traced: small signals do not move a flat or held position, only a crossing
    of the opposite band flips it, and a zero signal keeps the current side."""
    band = 0.1
    signals = [0.05, -0.05, 0.2, 0.05, 0.0, -0.05, -0.2, 0.0, 0.05, 0.2]
    expected = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0]
    held = 0.0
    path = []
    for signal in signals:
        held = banded_position(held, sign_position(signal), signal, "sign", band)
        path.append(held)
    assert path == expected


def test_banded_position_long_only_hysteresis_on_a_hand_sequence() -> None:
    band = 0.1
    signals = [0.05, 0.2, -0.05, 0.0, -0.2, 0.05, 0.2]
    expected = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    held = 0.0
    path = []
    for signal in signals:
        target = max(sign_position(signal), 0.0)
        held = banded_position(held, target, signal, "long_only", band)
        path.append(held)
    assert path == expected


def test_banded_position_zero_band_applies_every_target_for_every_rule() -> None:
    for rule in POSITION_RULES:
        for held, target, signal in ((0.0, 1.0, 0.01), (1.0, -1.0, -0.01), (0.3, 0.31, 0.31)):
            assert banded_position(held, target, signal, rule, 0.0) == target


def test_banded_position_continuous_rules_keep_position_space_band() -> None:
    for rule in ("conviction", "persistence"):
        assert banded_position(0.0, 0.2, 0.2, rule, 0.25) == 0.0  # inside the band: hold
        assert banded_position(0.0, 0.3, 0.3, rule, 0.25) == 0.3  # outside: move
        assert banded_position(0.5, 0.3, 0.3, rule, 0.25) == 0.5  # |0.3 - 0.5| < 0.25: hold


def test_banded_position_rejects_unknown_rule() -> None:
    with pytest.raises(ValueError, match="rule"):
        banded_position(0.0, 1.0, 0.5, "signs", 0.1)


@pytest.mark.parametrize("rule", ["sign", "long_only"])
def test_sign_rules_dead_band_follows_the_signal_hysteresis_recurrence(rule: str) -> None:
    """Replay the per-rule hysteresis on the emitted signals and compare every position."""
    close, labels = _hysteresis_case(73)
    result = run_walk_forward(
        close, labels, min_train=HYSTERESIS_MIN_TRAIN, rule=rule, dead_band=SIGNAL_DEAD_BAND
    )
    held = 0.0
    expected = np.empty(len(result.signals), dtype=float)
    for k, signal in enumerate(result.signals.to_numpy()):
        if rule == "sign":
            if held == 0.0 and abs(signal) > SIGNAL_DEAD_BAND:
                held = float(np.sign(signal))
            elif held == 1.0 and signal < -SIGNAL_DEAD_BAND:
                held = -1.0
            elif held == -1.0 and signal > SIGNAL_DEAD_BAND:
                held = 1.0
        else:
            if held == 0.0 and signal > SIGNAL_DEAD_BAND:
                held = 1.0
            elif held == 1.0 and signal < -SIGNAL_DEAD_BAND:
                held = 0.0
        expected[k] = held
    assert np.array_equal(result.positions.to_numpy(), expected)
    assert np.array_equal(
        result.strategy_returns.to_numpy(), expected * result.asset_returns.to_numpy()
    )


def test_sign_dead_band_holds_long_through_an_empty_row_on_a_hand_built_label_path() -> None:
    """Tiny label sequence, traced by hand. 20 Bull labels then 20 Bear, min_train 5.

    While the state is Bull the Bull row is all Bull->Bull, so the signal is +1 and both
    variants are long. At the first two Bear bars the Bear row has no counts yet, so the
    signal is exactly 0: the Phase 1 sign rule goes flat, the banded rule has no
    reason to leave long and stays. From the third Bear bar the Bear row holds a
    Bear->Bear transition, the signal is -1 and both are short.
    """
    n_each = 20
    min_train = 5
    close = synthetic_close(2 * n_each + 1, seed=74)
    labels = pd.Series([BULL] * n_each + [BEAR] * n_each, index=close.index[1:], dtype=int)
    plain = run_walk_forward(close, labels, min_train=min_train, rule="sign")
    banded = run_walk_forward(
        close, labels, min_train=min_train, rule="sign", dead_band=SIGNAL_DEAD_BAND
    )
    decisions = labels.index[min_train:-1]
    first_bear = labels.index[n_each]
    second_bear = labels.index[n_each + 1]
    expected_plain = pd.Series(1.0, index=decisions)
    expected_plain.loc[first_bear:] = -1.0
    expected_plain.loc[[first_bear, second_bear]] = 0.0
    expected_banded = expected_plain.replace(0.0, 1.0)
    assert np.array_equal(plain.positions.to_numpy(), expected_plain.to_numpy())
    assert np.array_equal(banded.positions.to_numpy(), expected_banded.to_numpy())
    assert plain.signals.loc[first_bear] == 0.0 and plain.signals.loc[second_bear] == 0.0


@pytest.mark.parametrize("bad_band", [-0.1, float("nan"), float("inf")])
def test_rejects_negative_or_non_finite_dead_band(bad_band: float) -> None:
    close, labels, _, _ = _synthetic_case(68)
    with pytest.raises(ValueError, match="dead_band"):
        run_walk_forward(close, labels, min_train=SMALL_MIN_TRAIN, dead_band=bad_band)


def test_phase3_defaults_match_config() -> None:
    import yaml

    config = yaml.safe_load((Path(__file__).parents[1] / "config" / "default.yaml").read_text())
    assert config["backtest"]["refit"] == DEFAULT_REFIT
    # Package E adds backtest.rolling_bars / backtest.dead_band; until then the Fix 4
    # statistics window is the same 3-year number.
    assert config.get("backtest", {}).get("rolling_bars", DEFAULT_ROLLING_BARS) == (
        DEFAULT_ROLLING_BARS
    )
    assert config.get("backtest", {}).get("dead_band", DEFAULT_DEAD_BAND) == DEFAULT_DEAD_BAND
    assert config["statistics"]["rolling_bars"] == DEFAULT_ROLLING_BARS
