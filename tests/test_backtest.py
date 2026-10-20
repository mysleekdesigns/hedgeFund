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
    MIN_TEST_DAYS,
    WalkForwardResult,
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
