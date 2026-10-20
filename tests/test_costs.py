"""Tests for ``hedgefund.backtest.costs`` (Phase 3, plan Fix 6).

Synthetic positions and returns with hand-computed answers. Offline, no fixtures.
"""

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from hedgefund.backtest.costs import (
    SENSITIVITY_BPS,
    ZERO_COSTS,
    CostModel,
    apply_costs,
    cost_model_from_config,
    cost_sensitivity,
    cost_series,
    turnover,
)
from hedgefund.backtest.metrics import (
    TRADING_DAYS_PER_YEAR,
    max_drawdown,
    sharpe_ratio,
    total_return,
)
from hedgefund.config import load_config

BPS = 1e4


# --------------------------------------------------------------------------- #
# CostModel
# --------------------------------------------------------------------------- #
def test_cost_model_defaults_match_us_etf_profile() -> None:
    model = CostModel()
    # config/default.yaml costs.profiles.us_etfs: 1 bp fee, 2 bps slippage, 30 bps/yr borrow
    # on short notional, no perpetual funding (review fix N1: defaults match config).
    assert (model.fee_bps, model.slippage_bps, model.borrow_bps, model.funding_bps) == (
        1.0,
        2.0,
        30.0,
        0.0,
    )
    assert model.per_side_bps == 3.0


def test_cost_model_default_equals_us_etfs_config_profile() -> None:
    """python.md: parameter defaults match config/default.yaml, field for field."""
    assert CostModel() == cost_model_from_config(load_config(), "us_etfs")


def test_cost_model_default_charges_borrow_on_short_days() -> None:
    """The default is no longer borrow-free: a short day costs 30 / 252 bps."""
    costs = cost_series(np.array([-1.0, -1.0]), CostModel(), periods_per_year=252)
    assert costs[1] == pytest.approx(30.0 / 252 / BPS)
    assert costs[1] > 0.0


def test_zero_costs_is_all_zero() -> None:
    assert ZERO_COSTS.per_side_bps == 0.0
    assert ZERO_COSTS.borrow_bps == 0.0 and ZERO_COSTS.funding_bps == 0.0


@pytest.mark.parametrize("field", ["fee_bps", "slippage_bps", "borrow_bps", "funding_bps"])
def test_cost_model_rejects_negative_bps(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        CostModel(**{field: -1.0})


def test_cost_model_rejects_nan_bps() -> None:
    with pytest.raises(ValueError, match="fee_bps"):
        CostModel(fee_bps=float("nan"))


def test_sensitivity_levels_are_the_fix_6_levels() -> None:
    assert SENSITIVITY_BPS == (0, 5, 10, 20, 50)


# --------------------------------------------------------------------------- #
# cost_model_from_config
# --------------------------------------------------------------------------- #
def test_cost_model_from_default_config_us_etfs() -> None:
    model = cost_model_from_config(load_config(), "us_etfs")
    assert (model.fee_bps, model.slippage_bps) == (1.0, 2.0)
    # config/default.yaml: 30 bps/yr general-collateral borrow on short ETF notional, no funding
    assert (model.borrow_bps, model.funding_bps) == (30.0, 0.0)


def test_cost_model_from_default_config_crypto() -> None:
    model = cost_model_from_config(load_config(), "crypto")
    assert (model.fee_bps, model.slippage_bps) == (10.0, 10.0)
    assert (model.borrow_bps, model.funding_bps) == (0.0, 0.0)


def test_cost_model_from_config_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="unknown cost profile 'nope'"):
        cost_model_from_config(load_config(), "nope")


def test_cost_model_from_config_reads_optional_borrow_and_funding() -> None:
    config = {
        "costs": {
            "profiles": {
                "perps": {"fee_bps": 5, "slippage_bps": 5, "borrow_bps": 100, "funding_bps": 1000}
            }
        }
    }
    model = cost_model_from_config(config, "perps")
    assert model == CostModel(5.0, 5.0, 100.0, 1000.0)


def test_cost_model_from_config_missing_required_key_raises() -> None:
    config = {"costs": {"profiles": {"half": {"fee_bps": 5}}}}
    with pytest.raises(ValueError, match="slippage_bps"):
        cost_model_from_config(config, "half")


# --------------------------------------------------------------------------- #
# turnover
# --------------------------------------------------------------------------- #
def test_turnover_constant_position_is_entry_then_zero() -> None:
    assert turnover(np.array([0.7, 0.7, 0.7, 0.7])).tolist() == [0.7, 0.0, 0.0, 0.0]
    assert turnover(np.array([-1.0, -1.0, -1.0])).tolist() == [1.0, 0.0, 0.0]


def test_turnover_sign_flip_costs_two_units() -> None:
    assert turnover(np.array([1.0, -1.0, 1.0, 0.0])).tolist() == [1.0, 2.0, 2.0, 1.0]


def test_turnover_empty_is_empty() -> None:
    assert turnover(np.array([])).shape == (0,)


def test_turnover_rejects_2d() -> None:
    with pytest.raises(ValueError, match="positions must be a 1-D array"):
        turnover(np.ones((3, 2)))


# --------------------------------------------------------------------------- #
# cost_series
# --------------------------------------------------------------------------- #
def test_cost_series_trading_charge_is_turnover_times_per_side() -> None:
    model = CostModel(fee_bps=1.0, slippage_bps=2.0, borrow_bps=0.0)  # isolate the trading charge
    positions = np.array([1.0, 1.0, -1.0, 0.0])
    expected = np.array([1.0, 0.0, 2.0, 1.0]) * 3.0 / BPS
    np.testing.assert_allclose(cost_series(positions, model), expected)


def test_cost_series_borrow_applies_only_on_short_steps() -> None:
    model = CostModel(fee_bps=0.0, slippage_bps=0.0, borrow_bps=252.0)
    positions = np.array([1.0, -1.0, -0.5, 0.0, 1.0])
    # 252 annual bps / 252 periods = 1 bp per step on the short notional only.
    expected = np.array([0.0, 1.0, 0.5, 0.0, 0.0]) / BPS
    np.testing.assert_allclose(cost_series(positions, model, periods_per_year=252), expected)


def test_cost_series_funding_applies_on_absolute_position() -> None:
    model = CostModel(fee_bps=0.0, slippage_bps=0.0, borrow_bps=0.0, funding_bps=365.0 * 2)
    positions = np.array([1.0, -1.0, 0.5, 0.0])
    # 730 annual bps / 365 periods = 2 bps per step on |position| either side.
    expected = np.array([1.0, 1.0, 0.5, 0.0]) * 2.0 / BPS
    np.testing.assert_allclose(cost_series(positions, model, periods_per_year=365), expected)


def test_cost_series_components_add() -> None:
    model = CostModel(fee_bps=2.0, slippage_bps=3.0, borrow_bps=252.0, funding_bps=504.0)
    positions = np.array([-1.0, -1.0])
    # step 0: enter short (1 unit * 5 bps) + borrow 1 bp + funding 2 bps = 8 bps
    # step 1: no trade + borrow 1 bp + funding 2 bps = 3 bps
    np.testing.assert_allclose(cost_series(positions, model), np.array([8.0, 3.0]) / BPS)


def test_cost_series_rejects_non_positive_periods() -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        cost_series(np.array([1.0]), CostModel(), periods_per_year=0)


# --------------------------------------------------------------------------- #
# apply_costs
# --------------------------------------------------------------------------- #
def test_apply_costs_zero_costs_is_bit_identical() -> None:
    rng = np.random.default_rng(0)
    gross = rng.normal(0.0, 0.01, 500)
    positions = rng.choice([-1.0, 0.0, 1.0], size=500)
    net = apply_costs(gross, positions, ZERO_COSTS)
    assert np.array_equal(net, gross)
    assert net.dtype == gross.dtype


def test_apply_costs_subtracts_cost_series() -> None:
    model = CostModel(fee_bps=5.0, slippage_bps=5.0, borrow_bps=0.0)  # trading charge only
    gross = np.array([0.01, -0.02, 0.0])
    positions = np.array([1.0, 1.0, -1.0])
    expected = gross - np.array([1.0, 0.0, 2.0]) * 10.0 / BPS
    np.testing.assert_allclose(apply_costs(gross, positions, model), expected)


def test_apply_costs_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal length"):
        apply_costs(np.zeros(3), np.zeros(4), CostModel())


def test_apply_costs_rejects_2d_returns() -> None:
    with pytest.raises(ValueError, match="gross_returns must be a 1-D array"):
        apply_costs(np.zeros((3, 1)), np.zeros(3), CostModel())


# --------------------------------------------------------------------------- #
# Plan item: costs reduce returns monotonically as bps increase
# --------------------------------------------------------------------------- #
# Bounded so 1 + net stays positive at every level: |asset return| <= 50%, per-side
# cost <= 2 units of turnover * 500 bps = 10%, so the compounded equity never crosses zero
# and cumulative returns stay ordered.
_series = hnp.arrays(
    dtype=np.float64,
    shape=st.integers(min_value=1, max_value=60),
    elements=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
)
_positions = hnp.arrays(
    dtype=np.float64,
    shape=st.shared(st.integers(min_value=1, max_value=60), key="n"),
    elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
_asset_returns = hnp.arrays(
    dtype=np.float64,
    shape=st.shared(st.integers(min_value=1, max_value=60), key="n"),
    elements=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
)
_bps_pairs = st.tuples(
    st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
).map(sorted)


@settings(deadline=None, max_examples=200)
@given(positions=_positions, asset_returns=_asset_returns, bps=_bps_pairs)
def test_costs_reduce_returns_monotonically(
    positions: np.ndarray, asset_returns: np.ndarray, bps: list[float]
) -> None:
    gross = positions * asset_returns
    low, high = bps
    net_low = apply_costs(gross, positions, CostModel(fee_bps=low, slippage_bps=0.0))
    net_high = apply_costs(gross, positions, CostModel(fee_bps=high, slippage_bps=0.0))
    # Every per-step net return is weakly lower at the higher cost level.
    assert (net_high <= net_low).all()
    # So is every cumulative net return along the path (equity stays positive by construction).
    cum_low = np.cumprod(1.0 + net_low) - 1.0
    cum_high = np.cumprod(1.0 + net_high) - 1.0
    assert (cum_high <= cum_low + 1e-12).all()
    assert total_return(net_high) <= total_return(net_low) + 1e-12


@settings(deadline=None, max_examples=100)
@given(positions=_positions, asset_returns=_asset_returns)
def test_cost_sensitivity_total_return_is_non_increasing_across_levels(
    positions: np.ndarray, asset_returns: np.ndarray
) -> None:
    rows = cost_sensitivity(positions * asset_returns, positions)
    totals = [row["total_return"] for row in rows]
    assert all(later <= earlier + 1e-12 for earlier, later in itertools.pairwise(totals))


@settings(deadline=None, max_examples=100)
@given(positions=_positions, borrow=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False))
def test_borrow_and_funding_never_increase_returns(positions: np.ndarray, borrow: float) -> None:
    gross = np.zeros_like(positions)
    net = apply_costs(gross, positions, CostModel(0.0, 0.0, borrow_bps=borrow, funding_bps=borrow))
    assert (net <= 0.0).all()


def test_sign_rule_daily_flipping_costs_two_sides_per_day() -> None:
    # A rule that flips every day pays 2 * per_side each day: 3 bps per side -> 6 bps a day,
    # roughly 15% of notional a year. This is why Fix 6 calls for a dead-band.
    positions = np.tile([1.0, -1.0], 126)
    costs = cost_series(positions, CostModel(fee_bps=1.0, slippage_bps=2.0, borrow_bps=0.0))
    assert costs[0] == pytest.approx(3.0 / BPS)
    assert np.allclose(costs[1:], 6.0 / BPS)
    assert costs.sum() * TRADING_DAYS_PER_YEAR / len(positions) == pytest.approx(0.1509, abs=1e-3)


# --------------------------------------------------------------------------- #
# cost_sensitivity
# --------------------------------------------------------------------------- #
def _sample_strategy() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    asset = rng.normal(0.0005, 0.01, 400)
    positions = np.where(np.cumsum(rng.choice([-1.0, 1.0], size=400)) > 0, 1.0, -1.0)
    return positions * asset, positions


def test_cost_sensitivity_returns_exactly_the_five_levels_in_order() -> None:
    gross, positions = _sample_strategy()
    rows = cost_sensitivity(gross, positions)
    assert [row["bps_per_side"] for row in rows] == [0, 5, 10, 20, 50]
    assert all(
        set(row) == {"bps_per_side", "sharpe", "total_return", "max_drawdown", "annual_cost"}
        for row in rows
    )
    assert all(isinstance(row["bps_per_side"], int) for row in rows)


def test_cost_sensitivity_level_zero_equals_gross_metrics() -> None:
    gross, positions = _sample_strategy()
    row = cost_sensitivity(gross, positions)[0]
    assert row["sharpe"] == sharpe_ratio(gross)
    assert row["total_return"] == total_return(gross)
    assert row["max_drawdown"] == max_drawdown(gross)
    assert row["annual_cost"] == 0.0


def test_cost_sensitivity_annual_cost_is_turnover_times_bps() -> None:
    gross, positions = _sample_strategy()
    rows = cost_sensitivity(gross, positions)
    one_way = turnover(positions).sum() / len(positions) * TRADING_DAYS_PER_YEAR
    for row in rows:
        assert row["annual_cost"] == pytest.approx(one_way * row["bps_per_side"] / BPS)


def test_cost_sensitivity_sharpe_falls_with_costs_on_a_trading_strategy() -> None:
    gross, positions = _sample_strategy()
    sharpes = [row["sharpe"] for row in cost_sensitivity(gross, positions)]
    assert all(later < earlier for earlier, later in itertools.pairwise(sharpes))


def test_cost_sensitivity_custom_levels_and_empty_input() -> None:
    rows = cost_sensitivity(np.array([0.01, 0.02]), np.array([1.0, 1.0]), bps_levels=(0, 100))
    assert [row["bps_per_side"] for row in rows] == [0, 100]
    assert rows[1]["total_return"] == pytest.approx((1.0 + 0.01 - 0.01) * 1.02 - 1.0)
    empty = cost_sensitivity(np.array([]), np.array([]))
    assert len(empty) == len(SENSITIVITY_BPS)
    assert all(math.isnan(row["annual_cost"]) and math.isnan(row["sharpe"]) for row in empty)
