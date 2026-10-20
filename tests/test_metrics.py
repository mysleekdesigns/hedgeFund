"""Tests for the Phase 3 additions to ``hedgefund.backtest.metrics`` (plan Fix 6).

Hand-computed values on tiny arrays. The Phase 1 functions (``equity_curve``,
``sharpe_ratio``, ``max_drawdown``) are covered by ``tests/test_backtest.py`` and the
reproduction test; here we only check they are still the same objects the new code builds on.
"""

import math

import numpy as np
import pytest

from hedgefund.backtest import metrics
from hedgefund.backtest.metrics import (
    DEFAULT_HORIZONS,
    TRADING_DAYS_PER_YEAR,
    annual_turnover,
    annualised_return,
    average_loss,
    average_win,
    calmar_ratio,
    hit_rate,
    horizon_pnl,
    max_drawdown,
    metric_set,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    time_in_market,
    total_return,
)

RETURNS = np.array([0.10, -0.05, 0.0, 0.02, -0.01])
EMPTY = np.array([])


# --------------------------------------------------------------------------- #
# total_return / annualised_return
# --------------------------------------------------------------------------- #
def test_total_return_compounds() -> None:
    assert total_return(np.array([0.1, 0.1])) == pytest.approx(0.21)
    assert total_return(RETURNS) == pytest.approx(1.10 * 0.95 * 1.0 * 1.02 * 0.99 - 1.0)


def test_total_return_empty_is_nan() -> None:
    assert math.isnan(total_return(EMPTY))


def test_annualised_return_geometric() -> None:
    # 1% a step for 252 steps annualises to exactly the compounded total.
    daily = np.full(252, 0.01)
    assert annualised_return(daily) == pytest.approx(1.01**252 - 1.0)
    # Half a year at the same rate still annualises to the same figure.
    assert annualised_return(daily[:126]) == pytest.approx(1.01**252 - 1.0)
    # Different calendar: 365 periods.
    assert annualised_return(np.array([0.1]), periods_per_year=365) == pytest.approx(1.1**365 - 1.0)


def test_annualised_return_edge_cases() -> None:
    assert math.isnan(annualised_return(EMPTY))
    assert math.isnan(annualised_return(np.array([-1.0])))  # wiped out: no real growth rate
    assert math.isnan(annualised_return(np.array([-1.5])))


# --------------------------------------------------------------------------- #
# sortino_ratio / calmar_ratio
# --------------------------------------------------------------------------- #
def test_sortino_hand_computed() -> None:
    r = np.array([0.02, -0.01, 0.03, -0.02])
    downside = math.sqrt((0.01**2 + 0.02**2) / 4)  # gains count as zero in the downside sum
    expected = r.mean() / downside * math.sqrt(252)
    assert sortino_ratio(r) == pytest.approx(expected)
    assert sortino_ratio(r, periods_per_year=12) == pytest.approx(
        r.mean() / downside * math.sqrt(12)
    )


def test_sortino_nan_cases() -> None:
    assert math.isnan(sortino_ratio(EMPTY))
    assert math.isnan(sortino_ratio(np.array([0.01])))  # n < 2
    assert math.isnan(sortino_ratio(np.array([0.01, 0.02, 0.0])))  # no downside


def test_sortino_exceeds_sharpe_for_positively_skewed_returns() -> None:
    r = np.array([0.05, -0.01, 0.05, -0.01, 0.05, -0.01])
    assert sortino_ratio(r) > sharpe_ratio(r)


def test_calmar_hand_computed() -> None:
    r = np.array([0.10, -0.20, 0.05])
    assert max_drawdown(r) == pytest.approx(-0.20)
    assert calmar_ratio(r) == pytest.approx(annualised_return(r) / 0.20)


def test_calmar_nan_cases() -> None:
    assert math.isnan(calmar_ratio(EMPTY))
    assert math.isnan(calmar_ratio(np.array([0.01, 0.02, 0.03])))  # no drawdown


# --------------------------------------------------------------------------- #
# hit_rate / average_win / average_loss / profit_factor
# --------------------------------------------------------------------------- #
def test_hit_rate_ignores_flat_periods() -> None:
    # 2 wins, 2 losses, 1 flat: 50%, not 40%.
    assert hit_rate(RETURNS) == pytest.approx(0.5)
    assert hit_rate(np.array([0.01, 0.02, -0.01])) == pytest.approx(2 / 3)


def test_hit_rate_nan_when_nothing_active() -> None:
    assert math.isnan(hit_rate(EMPTY))
    assert math.isnan(hit_rate(np.zeros(5)))


def test_average_win_and_loss() -> None:
    assert average_win(RETURNS) == pytest.approx(0.06)
    assert average_loss(RETURNS) == pytest.approx(-0.03)
    assert average_loss(RETURNS) < 0


def test_average_win_and_loss_nan_when_absent() -> None:
    assert math.isnan(average_win(np.array([-0.01, 0.0])))
    assert math.isnan(average_loss(np.array([0.01, 0.0])))
    assert math.isnan(average_win(EMPTY)) and math.isnan(average_loss(EMPTY))


def test_profit_factor_hand_computed() -> None:
    assert profit_factor(RETURNS) == pytest.approx(0.12 / 0.06)
    assert profit_factor(np.array([0.01, -0.02])) == pytest.approx(0.5)


def test_profit_factor_nan_without_losses() -> None:
    assert math.isnan(profit_factor(np.array([0.01, 0.0, 0.02])))
    assert math.isnan(profit_factor(EMPTY))
    assert profit_factor(np.array([-0.01, 0.0])) == 0.0


# --------------------------------------------------------------------------- #
# time_in_market / annual_turnover
# --------------------------------------------------------------------------- #
def test_time_in_market_counts_either_sign() -> None:
    assert time_in_market(np.array([1.0, -1.0, 0.0, 0.5])) == pytest.approx(0.75)
    assert time_in_market(np.zeros(3)) == 0.0
    assert time_in_market(np.ones(3)) == 1.0


def test_time_in_market_empty_is_nan() -> None:
    assert math.isnan(time_in_market(EMPTY))


def test_annual_turnover_constant_position_is_entry_only() -> None:
    # Buy and hold for a year: one unit of notional traded, once.
    assert annual_turnover(np.ones(252)) == pytest.approx(1.0)
    # Hold for two years: half a unit per year.
    assert annual_turnover(np.ones(504)) == pytest.approx(0.5)


def test_annual_turnover_daily_flipping() -> None:
    positions = np.tile([1.0, -1.0], 126)
    # 1 unit to enter + 2 units per flip for 251 flips, over 252 days, times 252.
    assert annual_turnover(positions) == pytest.approx(1.0 + 2.0 * 251)


def test_annual_turnover_edge_cases() -> None:
    assert math.isnan(annual_turnover(EMPTY))
    assert annual_turnover(np.array([0.0, 0.0])) == 0.0
    assert annual_turnover(np.array([1.0]), periods_per_year=12) == pytest.approx(12.0)


# --------------------------------------------------------------------------- #
# horizon_pnl
# --------------------------------------------------------------------------- #
def test_horizon_pnl_h1_reproduces_gross_strategy_returns() -> None:
    rng = np.random.default_rng(3)
    asset = rng.normal(0.0, 0.01, 50)
    positions = rng.choice([-1.0, 0.0, 1.0], size=50)
    gross = positions * asset
    result = horizon_pnl(positions, asset, horizons=(1,))[1]
    assert result["n"] == 50
    assert result["mean"] == pytest.approx(gross.mean())
    assert result["median"] == pytest.approx(np.median(gross))
    assert result["hit_rate"] == pytest.approx(hit_rate(gross))


def test_horizon_pnl_hand_computed_h2() -> None:
    positions = np.array([1.0, -1.0, 0.5])
    asset = np.array([0.10, 0.20, -0.10])
    out = horizon_pnl(positions, asset, horizons=(2,))[2]
    # k=0: 1.0 * (1.1*1.2 - 1) = 0.32 ; k=1: -1.0 * (1.2*0.9 - 1) = -0.08
    assert out["n"] == 2
    assert out["mean"] == pytest.approx((0.32 - 0.08) / 2)
    assert out["median"] == pytest.approx((0.32 - 0.08) / 2)
    assert out["hit_rate"] == pytest.approx(0.5)


def test_horizon_pnl_sample_count_drops_by_h_minus_one() -> None:
    n = 30
    out = horizon_pnl(np.ones(n), np.full(n, 0.01))
    assert set(out) == set(DEFAULT_HORIZONS) == {1, 5, 10, 20}
    for h in DEFAULT_HORIZONS:
        assert out[h]["n"] == n - h + 1
        assert out[h]["mean"] == pytest.approx(1.01**h - 1.0)


def test_horizon_pnl_short_series_is_nan_with_n_zero() -> None:
    out = horizon_pnl(np.ones(3), np.full(3, 0.01), horizons=(1, 5))
    assert out[1]["n"] == 3
    assert out[5] == {
        "mean": float("nan"),
        "median": float("nan"),
        "hit_rate": float("nan"),
        "n": 0,
    } or (out[5]["n"] == 0 and all(math.isnan(out[5][k]) for k in ("mean", "median", "hit_rate")))
    empty = horizon_pnl(EMPTY, EMPTY, horizons=(1,))[1]
    assert empty["n"] == 0 and math.isnan(empty["mean"])


def test_horizon_pnl_uses_only_future_steps_of_each_decision() -> None:
    # Changing the asset return before the decision must not change that decision's P&L.
    positions = np.array([0.0, 1.0, 1.0, 1.0])
    base = np.array([0.5, 0.01, 0.02, 0.03])
    mutated = base.copy()
    mutated[0] = -0.5
    out_a = horizon_pnl(positions, base, horizons=(3,))[3]
    out_b = horizon_pnl(positions, mutated, horizons=(3,))[3]
    # k=0 has position 0 so contributes 0 either way; k=1 uses returns[1:4] only.
    assert out_a == out_b


def test_horizon_pnl_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="equal length"):
        horizon_pnl(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="positions must be a 1-D array"):
        horizon_pnl(np.ones((3, 1)), np.ones(3))
    with pytest.raises(ValueError, match="positive integers"):
        horizon_pnl(np.ones(3), np.ones(3), horizons=(0,))


# --------------------------------------------------------------------------- #
# metric_set
# --------------------------------------------------------------------------- #
EXPECTED_KEYS = {
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "total_return",
    "annual_return",
    "hit_rate",
    "avg_win",
    "avg_loss",
    "profit_factor",
    "turnover",
    "time_in_market",
    "n_days",
}


def test_metric_set_keys_and_values_match_components() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0005, 0.01, 300)
    positions = rng.choice([-1.0, 0.0, 1.0], size=300)
    out = metric_set(returns, positions)
    assert set(out) == EXPECTED_KEYS
    assert out["sharpe"] == sharpe_ratio(returns)
    assert out["sortino"] == sortino_ratio(returns)
    assert out["calmar"] == calmar_ratio(returns)
    assert out["max_drawdown"] == max_drawdown(returns)
    assert out["total_return"] == total_return(returns)
    assert out["annual_return"] == annualised_return(returns)
    assert out["hit_rate"] == hit_rate(returns)
    assert out["avg_win"] == average_win(returns)
    assert out["avg_loss"] == average_loss(returns)
    assert out["profit_factor"] == profit_factor(returns)
    assert out["turnover"] == annual_turnover(positions)
    assert out["time_in_market"] == time_in_market(positions)
    assert out["n_days"] == 300 and isinstance(out["n_days"], int)


def test_metric_set_passes_periods_per_year_through() -> None:
    returns = np.array([0.01, -0.02, 0.03, 0.0])
    positions = np.array([1.0, 1.0, -1.0, 0.0])
    out = metric_set(returns, positions, periods_per_year=52)
    assert out["sharpe"] == sharpe_ratio(returns, 52)
    assert out["annual_return"] == annualised_return(returns, 52)
    assert out["turnover"] == annual_turnover(positions, 52)


def test_metric_set_empty_is_all_nan_with_zero_days() -> None:
    out = metric_set(EMPTY, EMPTY)
    assert out["n_days"] == 0
    assert all(math.isnan(v) for k, v in out.items() if k != "n_days")


def test_metric_set_rejects_mismatch_and_2d() -> None:
    with pytest.raises(ValueError, match="equal length"):
        metric_set(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="1-D"):
        metric_set(np.zeros((3, 1)), np.zeros(3))
    with pytest.raises(ValueError, match="positions must be a 1-D array"):
        metric_set(np.zeros(3), np.zeros((3, 1)))


# --------------------------------------------------------------------------- #
# Phase 1 surface is intact
# --------------------------------------------------------------------------- #
def test_phase_1_metrics_still_exported() -> None:
    assert metrics.TRADING_DAYS_PER_YEAR == TRADING_DAYS_PER_YEAR == 252
    assert metrics.equity_curve(np.array([0.1, 0.1])).tolist() == pytest.approx([1.1, 1.21])
    assert math.isnan(metrics.sharpe_ratio(np.array([0.01])))
    assert metrics.max_drawdown(np.array([0.1, -0.1])) == pytest.approx(-0.1)


@pytest.mark.parametrize(
    "fn",
    [
        total_return,
        annualised_return,
        sortino_ratio,
        calmar_ratio,
        hit_rate,
        average_win,
        average_loss,
        profit_factor,
    ],
)
def test_return_metrics_reject_2d(fn) -> None:
    with pytest.raises(ValueError, match="1-D"):
        fn(np.zeros((4, 2)))


@pytest.mark.parametrize("fn", [time_in_market, annual_turnover])
def test_position_metrics_reject_2d(fn) -> None:
    with pytest.raises(ValueError, match="positions must be a 1-D array"):
        fn(np.zeros((4, 2)))
