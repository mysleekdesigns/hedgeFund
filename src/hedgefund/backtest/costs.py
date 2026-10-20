"""Transaction cost model: fees, slippage, short borrow and perpetual funding.

Costs are charged on ``|Δposition|`` per side (``.claude/rules/no-lookahead.md``). The
first position is entered from flat, so it pays ``|positions[0]| * per_side``. Borrow and
funding are annual bps spread evenly over ``periods_per_year`` and charged on the position
held into each step. Pure numpy: arrays in, arrays or dicts out, no I/O.

Backtests are historical, not forward-looking.
"""

from dataclasses import dataclass

import numpy as np

from hedgefund.backtest.metrics import (
    TRADING_DAYS_PER_YEAR,
    _as_1d,
    _one_way_turnover,
    max_drawdown,
    sharpe_ratio,
    total_return,
)

BPS = 1e4  # one basis point is 1/10,000 of notional
SENSITIVITY_BPS: tuple[int, ...] = (0, 5, 10, 20, 50)  # plan Fix 6: per-side levels to report


@dataclass(frozen=True)
class CostModel:
    """Per-universe cost assumptions. All figures are basis points of notional.

    ``fee_bps`` and ``slippage_bps`` are per side and charged on every unit of turnover.
    ``borrow_bps`` and ``funding_bps`` are ANNUAL rates: borrow applies to the short
    notional on days the position is negative, funding to the gross notional every day
    (crypto perpetual funding, which is paid whether long or short).
    """

    fee_bps: float = 1.0  # config costs.profiles.us_etfs.fee_bps: commission for liquid US ETFs
    slippage_bps: float = 2.0  # config costs.profiles.us_etfs.slippage_bps: half spread + impact
    # config costs.profiles.us_etfs.borrow_bps: annual general-collateral borrow on short notional
    borrow_bps: float = 30.0
    funding_bps: float = 0.0  # annual perpetual funding; 0 for spot ETFs, set for crypto perps

    def __post_init__(self) -> None:
        for name in ("fee_bps", "slippage_bps", "borrow_bps", "funding_bps"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite non-negative bps, got {value!r}")

    @property
    def per_side_bps(self) -> float:
        """Total per-side charge on turnover: fee plus slippage."""
        return self.fee_bps + self.slippage_bps


ZERO_COSTS = CostModel(0.0, 0.0, 0.0, 0.0)


def cost_model_from_config(config: dict, profile: str) -> CostModel:
    """Build a ``CostModel`` from ``config["costs"]["profiles"][profile]``.

    ``fee_bps`` and ``slippage_bps`` are required; ``borrow_bps`` and ``funding_bps``
    default to 0 when the profile omits them. Raises ``ValueError`` on an unknown profile.
    """
    profiles = config.get("costs", {}).get("profiles", {})
    if profile not in profiles:
        raise ValueError(
            f"unknown cost profile {profile!r}; config costs.profiles has {sorted(profiles)}"
        )
    entry = profiles[profile]
    for key in ("fee_bps", "slippage_bps"):
        if key not in entry:
            raise ValueError(f"cost profile {profile!r} is missing required key {key!r}")
    return CostModel(
        fee_bps=float(entry["fee_bps"]),
        slippage_bps=float(entry["slippage_bps"]),
        borrow_bps=float(entry.get("borrow_bps", 0.0)),
        funding_bps=float(entry.get("funding_bps", 0.0)),
    )


def turnover(positions: np.ndarray) -> np.ndarray:
    """One-way turnover ``|Δposition|`` per step; the first entry is ``|positions[0]|``."""
    return _one_way_turnover(positions)


def cost_series(
    positions: np.ndarray,
    model: CostModel,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Cost paid at each step as a fraction of notional, aligned with ``positions``.

    ``turnover * per_side + max(-position, 0) * borrow / ppy + |position| * funding / ppy``
    with every rate converted from bps. Entry ``k`` is charged against the return earned on
    the step that ``positions[k]`` is held into, so costs never use future positions.
    """
    pos = _as_1d(positions, "positions")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    trading = turnover(pos) * (model.per_side_bps / BPS)
    borrow = np.maximum(-pos, 0.0) * (model.borrow_bps / BPS / periods_per_year)
    funding = np.abs(pos) * (model.funding_bps / BPS / periods_per_year)
    return trading + borrow + funding


def apply_costs(
    gross_returns: np.ndarray,
    positions: np.ndarray,
    model: CostModel,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Net returns: ``gross - cost_series``. ``ZERO_COSTS`` returns ``gross`` bit-identical."""
    gross = _as_1d(gross_returns, "gross_returns")
    pos = _as_1d(positions, "positions")
    if len(gross) != len(pos):
        raise ValueError(
            f"gross_returns and positions must have equal length, got {len(gross)} and {len(pos)}"
        )
    return gross - cost_series(pos, model, periods_per_year)


def cost_sensitivity(
    gross_returns: np.ndarray,
    positions: np.ndarray,
    bps_levels: tuple[int, ...] = SENSITIVITY_BPS,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> list[dict]:
    """Net Sharpe, total return and max drawdown at each per-side cost level (plan Fix 6).

    Each level is applied as a pure per-side trading charge (fee = level, slippage,
    borrow and funding all 0). ``annual_cost`` is the average cost per step times
    ``periods_per_year``, the yearly return drag at that level. Level 0 reproduces the
    gross metrics exactly.
    """
    gross = _as_1d(gross_returns, "gross_returns")
    pos = _as_1d(positions, "positions")
    rows: list[dict] = []
    for level in bps_levels:
        model = CostModel(fee_bps=float(level), slippage_bps=0.0, borrow_bps=0.0, funding_bps=0.0)
        costs = cost_series(pos, model, periods_per_year)
        net = apply_costs(gross, pos, model, periods_per_year)
        annual_cost = float(costs.mean() * periods_per_year) if len(costs) else float("nan")
        rows.append(
            {
                "bps_per_side": int(level),
                "sharpe": sharpe_ratio(net, periods_per_year),
                "total_return": total_return(net),
                "max_drawdown": max_drawdown(net),
                "annual_cost": annual_cost,
            }
        )
    return rows
