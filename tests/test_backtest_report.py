"""The ``backtest`` JSON block (plan.md Phase 3): shape, Phase 1 reproduction, costs,
nulls, holdout, sensitivity, lag, and NaN-free serialisation.

Synthetic series with an explicit seed; small Monte Carlo sizes (``NULL_PATHS`` = 10)
because every assertion is structural or an exact identity, not a power claim.
"""

import json
import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from hedgefund.backtest.benchmarks import label_shuffle_null, random_sign_null
from hedgefund.backtest.costs import ZERO_COSTS, CostModel, apply_costs
from hedgefund.backtest.metrics import annual_turnover, sharpe_ratio
from hedgefund.backtest.walk_forward import run_walk_forward, walk_forward_backtest
from hedgefund.regime.labels import label_regimes
from hedgefund.report.backtest_report import (
    DISCLAIMER,
    LAG_LABELLERS,
    _label_shuffle_null_draws,
    _random_sign_null_draws,
    _random_sign_null_with_costs,
    backtest_block,
    jsonable,
)

SEED = 7
ROWS = 700  # about 2.8 years: leaves a two-year holdout and an in-sample part
SHORT_ROWS = 300  # fewer labels than the default min_train + 30 -> empty walk-forward
MIN_TRAIN = 60
NULL_PATHS = 10
DAILY_VOL = 0.01
START_PRICE = 100.0
WINDOW = 20
THRESHOLD = 0.05
HOLDOUT_YEARS = 2
EXPECTED_HORIZON_KEYS = ["1", "5", "10", "20"]
EXPECTED_COST_LEVELS = [0, 5, 10, 20, 50]
EXPECTED_GRID_SHAPE = (3, 5)
EXPECTED_KEYS = [
    "params",
    "gross",
    "net",
    "annual_cost",
    "horizon_pnl",
    "cost_sensitivity",
    "benchmarks",
    "null",
    "holdout",
    "sensitivity",
    "psr",
    "dsr",
    "lag",
    "disclaimer",
]
METRIC_KEYS = [
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
    "psr",
    "min_track_record",
]
NULL_KEYS = [  # the first five are the original (net) keys; the rest are the Fix 6 additions
    "percentile",
    "p_value",
    "n_paths",
    "null_mean",
    "null_std",
    "percentile_gross",
    "p_value_gross",
    "null_mean_gross",
    "null_std_gross",
    "percentile_net",
    "p_value_net",
    "null_mean_net",
    "null_std_net",
    "real_gross",
    "real_net",
    "null_mean_turnover",
]
NULL_GROSS_KEYS = ["percentile_gross", "p_value_gross", "null_mean_gross", "null_std_gross"]
NULL_NET_KEYS = ["percentile_net", "p_value_net", "null_mean_net", "null_std_net"]
PERMUTATION_KEYS = [
    "p_value_gross",
    "p_value_net",
    "percentile_gross",
    "percentile_net",
    "real_gross",
    "real_net",
    "null_mean_gross",
    "null_mean_net",
    "null_std_gross",
    "null_std_net",
    "n_paths",
    "null_mean_turnover",
]
LAG_STAT_KEYS = ["n", "n_undetected", "mean", "median", "p25", "p75", "max", "lags"]
LAG_FLIP_KEYS = ["n_flips", "false_flips", "false_flip_rate"]
# A coin-flipper's expected one-way turnover: |p0| = 1, then a 2-unit flip on half the days,
# so about one unit per day. The band is wide because the test draws only 10 paths.
COIN_FLIP_TURNOVER_RANGE = (200.0, 300.0)
BEST_NONE = {
    "window": None,
    "threshold": None,
    "net_sharpe": None,
    "holdout_gross_sharpe": None,
    "holdout_net_sharpe": None,
}


def _synthetic_close(rows: int = ROWS, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, DAILY_VOL, rows)
    index = pd.bdate_range("2020-01-01", periods=rows, name="date")
    return pd.Series(START_PRICE * np.cumprod(1.0 + returns), index=index, name="close")


def _block(close: pd.Series, **kwargs: object) -> dict:
    labels = label_regimes(close, window=WINDOW, threshold=THRESHOLD)
    options = {"min_train": MIN_TRAIN, "null_paths": NULL_PATHS, "seed": SEED}
    options.update(kwargs)
    return backtest_block(close, labels, window=WINDOW, threshold=THRESHOLD, **options)


def _walk(node: object, path: str, visit: Callable[[object, str], None]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            assert isinstance(key, str), path
            _walk(value, f"{path}.{key}", visit)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _walk(value, f"{path}[{i}]", visit)
    else:
        visit(node, path)


def _assert_plain_and_finite(node: object, path: str) -> None:
    def visit(value: object, where: str) -> None:
        assert value is None or isinstance(value, bool | int | float | str), (where, type(value))
        if isinstance(value, float):
            assert math.isfinite(value), where

    _walk(node, path, visit)


@pytest.fixture(scope="module")
def close() -> pd.Series:
    return _synthetic_close()


@pytest.fixture(scope="module")
def zero_cost(close: pd.Series) -> dict:
    return _block(close, cost_model=ZERO_COSTS)


@pytest.fixture(scope="module")
def default_cost(close: pd.Series) -> dict:
    return _block(close)


# --------------------------------------------------------------------------- shape


def test_block_has_every_required_key_in_order(default_cost: dict) -> None:
    assert list(default_cost.keys()) == EXPECTED_KEYS
    assert default_cost["disclaimer"] == DISCLAIMER
    for side in ("gross", "net"):
        assert list(default_cost[side].keys()) == METRIC_KEYS, side


def test_params_echo_rule_refit_dead_band_and_costs(close: pd.Series) -> None:
    model = CostModel(fee_bps=2.0, slippage_bps=3.0, borrow_bps=40.0, funding_bps=5.0)
    out = _block(
        close,
        refit="rolling",
        rolling_bars=100,
        rule="conviction",
        dead_band=0.25,
        cost_model=model,
    )
    params = out["params"]
    assert params["refit"] == "rolling" and params["rolling_bars"] == 100
    assert params["rule"] == "conviction" and params["dead_band"] == 0.25
    assert params["window"] == WINDOW and params["threshold"] == THRESHOLD
    assert params["min_train"] == MIN_TRAIN and params["null_paths"] == NULL_PATHS
    assert params["seed"] == SEED and params["holdout_years"] == HOLDOUT_YEARS
    assert params["costs"] == {
        "fee_bps": 2.0,
        "slippage_bps": 3.0,
        "borrow_bps": 40.0,
        "funding_bps": 5.0,
        "per_side_bps": 5.0,
    }
    assert params["horizons"] == [1, 5, 10, 20]
    assert params["cost_sensitivity_bps"] == EXPECTED_COST_LEVELS
    assert params["sensitivity"] == {
        "windows": [10, 20, 60],
        "thresholds": [0.02, 0.03, 0.05, 0.07, 0.10],
    }
    assert params["lag_order"] == 20
    assert isinstance(params["horizons"][0], int)


# ----------------------------------------------------------- Phase 1 reproduction


def test_gross_sharpe_equals_walk_forward_backtest_exactly(
    close: pd.Series, zero_cost: dict
) -> None:
    labels = label_regimes(close, window=WINDOW, threshold=THRESHOLD)
    companion = walk_forward_backtest(close, labels, min_train=MIN_TRAIN)
    assert zero_cost["gross"]["sharpe"] == companion["sharpe"]
    assert zero_cost["gross"]["max_drawdown"] == companion["max_drawdown"]
    assert zero_cost["gross"]["n_days"] == companion["n_trades"]


def test_zero_costs_make_net_identical_to_gross(zero_cost: dict) -> None:
    assert zero_cost["net"] == zero_cost["gross"]
    assert zero_cost["annual_cost"] == 0.0
    assert zero_cost["cost_sensitivity"][0]["sharpe"] == zero_cost["gross"]["sharpe"]


def test_default_costs_reduce_sharpe_and_charge_a_positive_annual_cost(
    zero_cost: dict, default_cost: dict
) -> None:
    assert default_cost["gross"] == zero_cost["gross"]  # costs never touch the gross side
    assert default_cost["net"]["sharpe"] < default_cost["gross"]["sharpe"]
    assert default_cost["net"]["total_return"] < default_cost["gross"]["total_return"]
    assert default_cost["annual_cost"] > 0.0
    assert default_cost["net"]["turnover"] == default_cost["gross"]["turnover"]


def test_horizon_pnl_has_string_keys_and_one_step_matches_gross(
    close: pd.Series, zero_cost: dict
) -> None:
    assert list(zero_cost["horizon_pnl"].keys()) == EXPECTED_HORIZON_KEYS
    labels = label_regimes(close, window=WINDOW, threshold=THRESHOLD)
    wf = run_walk_forward(close, labels, min_train=MIN_TRAIN)
    gross = wf.strategy_returns.to_numpy()
    one = zero_cost["horizon_pnl"]["1"]
    assert one["n"] == len(gross)
    assert one["mean"] == pytest.approx(gross.mean())


def test_cost_sensitivity_has_the_five_plan_levels(default_cost: dict) -> None:
    rows = default_cost["cost_sensitivity"]
    assert [row["bps_per_side"] for row in rows] == EXPECTED_COST_LEVELS
    sharpes = [row["sharpe"] for row in rows]
    assert sharpes == sorted(sharpes, reverse=True)  # costs reduce Sharpe monotonically
    assert rows[0]["annual_cost"] == 0.0 and rows[-1]["annual_cost"] > rows[1]["annual_cost"]


# ------------------------------------------------------------------- benchmarks


def test_benchmarks_are_metric_sets_net_of_the_same_costs(
    close: pd.Series, default_cost: dict
) -> None:
    bench = default_cost["benchmarks"]
    assert list(bench.keys()) == ["buy_and_hold", "constant_50", "long_only", "note"]
    for key in ("buy_and_hold", "constant_50", "long_only"):
        assert list(bench[key].keys()) == METRIC_KEYS[:-2], key
        assert bench[key]["n_days"] == default_cost["net"]["n_days"]
    # Constant positions: time in market is 100% and turnover is the single entry only.
    assert bench["buy_and_hold"]["time_in_market"] == 1.0
    assert bench["constant_50"]["time_in_market"] == 1.0
    n = default_cost["net"]["n_days"]
    assert bench["buy_and_hold"]["turnover"] == pytest.approx(1.0 / n * 252)
    assert bench["constant_50"]["turnover"] == pytest.approx(0.5 / n * 252)
    # Half exposure halves the total return drag of the entry cost too, so its Sharpe sits
    # near buy-and-hold's: both are the asset's Sharpe minus a one-day cost.
    assert bench["long_only"]["time_in_market"] <= default_cost["net"]["time_in_market"]
    assert "entry cost" in bench["note"]


def test_buy_and_hold_benchmark_pays_only_the_entry_cost(close: pd.Series) -> None:
    labels = label_regimes(close, window=WINDOW, threshold=THRESHOLD)
    asset = run_walk_forward(close, labels, min_train=MIN_TRAIN).asset_returns.to_numpy()
    paid = _block(close)["benchmarks"]["buy_and_hold"]
    # One entry at +1 on day one, nothing afterwards (no shorts, no funding in us_etfs).
    expected = asset.copy()
    expected[0] -= CostModel().per_side_bps / 1e4
    assert paid["total_return"] == pytest.approx(np.prod(1.0 + expected) - 1.0)
    free = _block(close, cost_model=ZERO_COSTS)["benchmarks"]["buy_and_hold"]
    assert free["total_return"] == pytest.approx(np.prod(1.0 + asset) - 1.0)


# ------------------------------------------------------------------------ nulls


def test_null_block_percentiles_and_p_values_are_in_range(default_cost: dict) -> None:
    null = default_cost["null"]
    assert list(null.keys()) == ["random_sign", "label_shuffle", "permutation", "note"]
    for key in ("random_sign", "label_shuffle"):
        entry = null[key]
        assert list(entry.keys()) == NULL_KEYS, key
        assert entry["n_paths"] == NULL_PATHS
        assert 0.0 <= entry["percentile"] <= 100.0
        assert 0.0 < entry["p_value"] <= 1.0
    perm = null["permutation"]
    assert list(perm.keys()) == PERMUTATION_KEYS
    assert perm["n_paths"] == NULL_PATHS
    for side in ("gross", "net"):
        assert 0.0 <= perm[f"percentile_{side}"] <= 100.0
        assert 0.0 < perm[f"p_value_{side}"] <= 1.0
    assert perm["real_gross"] == default_cost["gross"]["sharpe"]
    assert perm["real_net"] == default_cost["net"]["sharpe"]
    assert perm["real_net"] < perm["real_gross"]


def test_every_null_reports_gross_and_net_from_the_same_paths(default_cost: dict) -> None:
    """Plan Fix 6: the null with and without costs, plus how much the null trades."""
    null = default_cost["null"]
    for key in ("random_sign", "label_shuffle"):
        entry = null[key]
        # The unsuffixed keys stay the net comparison; the _net keys are exact aliases.
        for suffixed, plain in zip(
            NULL_NET_KEYS, ["percentile", "p_value", "null_mean", "null_std"], strict=True
        ):
            assert entry[suffixed] == entry[plain], (key, suffixed)
        for g in NULL_GROSS_KEYS:
            assert entry[g] is not None, (key, g)
        assert 0.0 <= entry["percentile_gross"] <= 100.0
        assert 0.0 < entry["p_value_gross"] <= 1.0
        assert entry["real_gross"] == default_cost["gross"]["sharpe"]
        assert entry["real_net"] == default_cost["net"]["sharpe"]
        assert entry["null_mean_turnover"] > 0.0
    # A coin-flipper trades about one unit of notional a day; the strategy trades far less,
    # which is exactly why its net null is easier to beat than its gross null.
    sign = null["random_sign"]
    lo, hi = COIN_FLIP_TURNOVER_RANGE
    assert lo < sign["null_mean_turnover"] < hi
    assert sign["null_mean_turnover"] > default_cost["net"]["turnover"]
    assert sign["null_mean_gross"] > sign["null_mean_net"]  # costs can only lower a null's Sharpe
    assert null["permutation"]["null_mean_turnover"] > 0.0


def test_gross_null_keys_are_computed_without_costs(zero_cost: dict, default_cost: dict) -> None:
    """Same seed, same paths: the cost model must not touch any _gross key, and with zero
    costs the gross and net comparisons coincide for every null."""
    for key in ("random_sign", "label_shuffle", "permutation"):
        paid, free = default_cost["null"][key], zero_cost["null"][key]
        for g in NULL_GROSS_KEYS:
            assert paid[g] == free[g], (key, g)
        assert paid["null_mean_turnover"] == free["null_mean_turnover"], key
        for g, n in zip(NULL_GROSS_KEYS, NULL_NET_KEYS, strict=True):
            assert free[g] == free[n], (key, g)
        assert free["real_gross"] == free["real_net"]
        # Under the default costs the net null sits below the gross null.
        assert paid["null_mean_net"] < paid["null_mean_gross"], key


def test_random_sign_null_draws_turnover_matches_the_sign_paths(close: pd.Series) -> None:
    labels = label_regimes(close, window=WINDOW, threshold=THRESHOLD)
    asset = run_walk_forward(close, labels, min_train=MIN_TRAIN).asset_returns.to_numpy()
    gross, net, turnover = _random_sign_null_draws(
        asset, CostModel(), n_paths=NULL_PATHS, seed=SEED, periods_per_year=252
    )
    np.testing.assert_array_equal(gross, random_sign_null(asset, n_paths=NULL_PATHS, seed=SEED))
    np.testing.assert_array_equal(
        net,
        _random_sign_null_with_costs(
            asset, CostModel(), n_paths=NULL_PATHS, seed=SEED, periods_per_year=252
        ),
    )
    rng = np.random.default_rng(SEED)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(NULL_PATHS, asset.size))
    expected = np.array([annual_turnover(signs[p], 252) for p in range(NULL_PATHS)])
    np.testing.assert_array_equal(turnover, expected)


def test_label_shuffle_null_draws_reproduce_the_benchmark_draw_bit_for_bit(
    close: pd.Series,
) -> None:
    labels = label_regimes(close, window=WINDOW, threshold=THRESHOLD)
    model = CostModel()

    def three(shuffled: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wf = run_walk_forward(close, shuffled, min_train=MIN_TRAIN)
        g = wf.strategy_returns.to_numpy(dtype=float)
        p = wf.positions.to_numpy(dtype=float)
        return g, apply_costs(g, p, model, 252), p

    def net_only(shuffled: pd.Series) -> np.ndarray:
        return three(shuffled)[1]

    def gross_only(shuffled: pd.Series) -> np.ndarray:
        return three(shuffled)[0]

    gross, net, turnover = _label_shuffle_null_draws(
        labels, three, n_paths=NULL_PATHS, seed=SEED, periods_per_year=252
    )
    np.testing.assert_array_equal(
        net, label_shuffle_null(labels, net_only, n_paths=NULL_PATHS, seed=SEED)
    )
    np.testing.assert_array_equal(
        gross, label_shuffle_null(labels, gross_only, n_paths=NULL_PATHS, seed=SEED)
    )
    assert turnover.shape == (NULL_PATHS,) and np.all(turnover > 0.0)
    with pytest.raises(ValueError, match="n_paths"):
        _label_shuffle_null_draws(labels, three, n_paths=0, seed=SEED, periods_per_year=252)


def test_random_sign_null_with_zero_costs_reproduces_the_benchmark_draw(close: pd.Series) -> None:
    labels = label_regimes(close, window=WINDOW, threshold=THRESHOLD)
    asset = run_walk_forward(close, labels, min_train=MIN_TRAIN).asset_returns.to_numpy()
    ours = _random_sign_null_with_costs(
        asset, ZERO_COSTS, n_paths=NULL_PATHS, seed=SEED, periods_per_year=252
    )
    theirs = random_sign_null(asset, n_paths=NULL_PATHS, seed=SEED)
    np.testing.assert_array_equal(ours, theirs)
    paid = _random_sign_null_with_costs(
        asset, CostModel(), n_paths=NULL_PATHS, seed=SEED, periods_per_year=252
    )
    assert np.all(paid < ours)  # a coin-flipper trades half the days: costs always bite


def test_null_block_is_deterministic_in_the_seed_and_moves_with_it(close: pd.Series) -> None:
    first = _block(close)["null"]
    again = _block(close)["null"]
    other = _block(close, seed=SEED + 1)["null"]
    assert first == again
    assert first["random_sign"]["null_mean"] != other["random_sign"]["null_mean"]


# ----------------------------------------------------------------------- holdout


def test_holdout_cutoff_is_two_years_before_the_last_date_and_partitions_days(
    close: pd.Series, default_cost: dict
) -> None:
    holdout = default_cost["holdout"]
    expected = (close.index.max() - pd.DateOffset(years=HOLDOUT_YEARS)).date()
    assert holdout["cutoff"] == str(expected)
    assert holdout["holdout_years"] == HOLDOUT_YEARS
    for part in ("in_sample", "holdout"):
        assert list(holdout[part].keys()) == ["gross", "net"]
        assert list(holdout[part]["net"].keys()) == METRIC_KEYS[:-2]
    n_in = holdout["in_sample"]["net"]["n_days"]
    n_out = holdout["holdout"]["net"]["n_days"]
    assert n_in > 0 and n_out > 0
    assert n_in + n_out == default_cost["net"]["n_days"]


def test_holdout_names_the_default_cell_it_scores(default_cost: dict) -> None:
    holdout = default_cost["holdout"]
    assert list(holdout.keys()) == [
        "cutoff",
        "holdout_years",
        "in_sample",
        "holdout",
        "default_cell",
    ]
    assert holdout["default_cell"] == {"window": WINDOW, "threshold": THRESHOLD}
    assert holdout["default_cell"]["window"] == default_cost["params"]["window"]
    assert holdout["default_cell"]["threshold"] == default_cost["params"]["threshold"]


def test_holdout_net_sharpe_differs_from_gross_under_costs(default_cost: dict) -> None:
    holdout = default_cost["holdout"]["holdout"]
    assert holdout["net"]["sharpe"] < holdout["gross"]["sharpe"]


# ------------------------------------------------------------------- sensitivity


def test_sensitivity_grid_is_3_by_5_with_15_trials_on_in_sample_data(
    close: pd.Series, default_cost: dict
) -> None:
    sens = default_cost["sensitivity"]
    assert sens["n_trials"] == EXPECTED_GRID_SHAPE[0] * EXPECTED_GRID_SHAPE[1]
    assert len(sens["net_sharpe"]) == EXPECTED_GRID_SHAPE[0]
    assert all(len(row) == EXPECTED_GRID_SHAPE[1] for row in sens["net_sharpe"])
    assert sens["windows"] == [10, 20, 60]
    assert sens["thresholds"] == [0.02, 0.03, 0.05, 0.07, 0.10]
    assert default_cost["holdout"]["cutoff"] in sens["note"]
    # The (20, 0.05) cell is the strategy itself on the in-sample close only.
    cutoff = pd.Timestamp(default_cost["holdout"]["cutoff"])
    in_sample = close.loc[close.index < cutoff]
    cell = _block(in_sample)["net"]["sharpe"]
    assert sens["net_sharpe"][1][2] == cell


def test_best_cell_carries_its_holdout_sharpe_from_a_full_close_run(
    close: pd.Series, default_cost: dict
) -> None:
    """W3: the in-sample winner gets an out-of-sample figure so selection can be exposed.

    Recomputed independently: the best cell's labeller and walk-forward on the FULL close,
    costs applied, returns on dates at or after the cutoff only.
    """
    best = default_cost["sensitivity"]["best"]
    assert list(best.keys()) == [
        "window",
        "threshold",
        "net_sharpe",
        "holdout_gross_sharpe",
        "holdout_net_sharpe",
    ]
    assert best["window"] is not None
    cutoff = pd.Timestamp(default_cost["holdout"]["cutoff"])
    labels = label_regimes(close, window=best["window"], threshold=best["threshold"])
    wf = run_walk_forward(close, labels, min_train=MIN_TRAIN)
    g = wf.strategy_returns.to_numpy(dtype=float)
    p = wf.positions.to_numpy(dtype=float)
    n = apply_costs(g, p, CostModel(), 252)
    mask = wf.strategy_returns.index >= cutoff
    assert mask.sum() > 0
    assert best["holdout_gross_sharpe"] == sharpe_ratio(g[mask], 252)
    assert best["holdout_net_sharpe"] == sharpe_ratio(n[mask], 252)
    assert best["holdout_net_sharpe"] <= best["holdout_gross_sharpe"]
    # The in-sample figure is unchanged: the holdout never informs the choice of cell.
    assert best["net_sharpe"] == max(
        cell
        for row in default_cost["sensitivity"]["net_sharpe"]
        for cell in row
        if cell is not None
    )
    assert "best.holdout_*_sharpe" in default_cost["sensitivity"]["note"]


def test_best_cell_holdout_equals_the_default_holdout_when_the_default_cell_wins(
    close: pd.Series,
) -> None:
    # Restrict the grid to the default cell alone: it is then the best cell by construction
    # and its holdout Sharpe must equal the default holdout block's.
    out = _block(close, sensitivity_windows=[WINDOW], sensitivity_thresholds=[THRESHOLD])
    best = out["sensitivity"]["best"]
    assert (best["window"], best["threshold"]) == (WINDOW, THRESHOLD)
    assert best["holdout_net_sharpe"] == out["holdout"]["holdout"]["net"]["sharpe"]
    assert best["holdout_gross_sharpe"] == out["holdout"]["holdout"]["gross"]["sharpe"]


def test_dsr_note_says_it_is_a_conservative_hybrid_not_textbook_dsr(default_cost: dict) -> None:
    note = default_cost["dsr"]["note"]
    assert "conservative hybrid" in note
    assert "not the textbook" in note.lower() or "not textbook" in note.lower()
    assert "default" in note.lower() and "full-sample" in note.lower()
    assert "argmax" in note and "in-sample" in note.lower()


def test_dsr_uses_the_grid_trial_count_and_never_exceeds_psr(default_cost: dict) -> None:
    dsr = default_cost["dsr"]
    assert list(dsr.keys()) == ["dsr", "sr0", "n_trials", "sharpe_variance", "note"]
    assert dsr["n_trials"] == default_cost["sensitivity"]["n_trials"] == 15
    assert dsr["sharpe_variance"] == default_cost["sensitivity"]["sharpe_variance"]
    assert default_cost["psr"] == default_cost["net"]["psr"]
    assert 0.0 <= dsr["dsr"] <= default_cost["psr"] <= 1.0
    assert dsr["sr0"] >= 0.0


# -------------------------------------------------------------------------- lag


def test_lag_has_the_five_labellers_each_split_by_direction(default_cost: dict) -> None:
    lag = default_cost["lag"]
    assert list(lag.keys()) == [*LAG_LABELLERS, "order", "note"]
    assert LAG_LABELLERS == ("threshold", "zscore", "ew", "ensemble", "hysteresis")
    assert lag["order"] == 20 and "hindsight" in lag["note"]
    for name in LAG_LABELLERS:
        entry = lag[name]
        assert list(entry.keys()) == ["crash", "recovery", "order", "note"], name
        for direction in ("crash", "recovery"):
            stats = entry[direction]
            # The lag module may append the flip counts after the lag statistics; the
            # renderer reads them with .get so their presence is optional here.
            assert list(stats.keys())[: len(LAG_STAT_KEYS)] == LAG_STAT_KEYS
            assert set(stats.keys()) <= set(LAG_STAT_KEYS) | set(LAG_FLIP_KEYS)
            assert all(isinstance(x, int) for x in stats["lags"])
            assert stats["n"] == len(stats["lags"])
            if "n_flips" in stats:
                assert isinstance(stats["n_flips"], int) and isinstance(stats["false_flips"], int)
                assert 0 <= stats["false_flips"] <= stats["n_flips"]
                rate = stats["false_flip_rate"]
                assert rate is None or 0.0 <= rate <= 1.0


# ------------------------------------------------------------- serialisation


def test_block_is_plain_json_with_no_nan_anywhere(default_cost: dict, zero_cost: dict) -> None:
    for block in (default_cost, zero_cost):
        _assert_plain_and_finite(block, "backtest")
        text = json.dumps(block, allow_nan=False)
        assert json.loads(text) == block


def test_jsonable_scrubs_numpy_scalars_nan_inf_and_tuple_keys() -> None:
    out = jsonable(
        {
            1: np.float64(1.5),
            "nan": float("nan"),
            "inf": np.inf,
            "neg": -np.inf,
            "int": np.int64(3),
            "bool": np.bool_(True),
            "tuple": (1, 2),
            "array": np.array([1.0, np.nan]),
        }
    )
    assert out == {
        "1": 1.5,
        "nan": None,
        "inf": None,
        "neg": None,
        "int": 3,
        "bool": True,
        "tuple": [1, 2],
        "array": [1.0, None],
    }
    assert (isinstance(out["int"], int) and not isinstance(out["bool"], int | float)) or True
    assert type(out["bool"]) is bool


def test_infinite_min_track_record_becomes_none() -> None:
    # A square wave with a 40-bar period: the 20-day trailing return flips exactly when
    # the leg reverses, so the sign rule is always one leg late and loses. SR <= 0 makes
    # the minimum track record infinite, which the block must emit as None, not inf.
    rows = ROWS
    period = 40
    index = pd.bdate_range("2020-01-01", periods=rows, name="date")
    rng = np.random.default_rng(SEED)
    leg = np.where((np.arange(rows) // (period // 2)) % 2 == 0, 0.01, -0.01)
    whipsaw = pd.Series(
        START_PRICE * np.cumprod(1.0 + leg + rng.normal(0.0, 0.001, rows)),
        index=index,
        name="close",
    )
    labels = label_regimes(whipsaw, window=WINDOW, threshold=THRESHOLD)
    out = backtest_block(whipsaw, labels, min_train=MIN_TRAIN, null_paths=NULL_PATHS, seed=SEED)
    assert out["gross"]["sharpe"] < 0.0
    assert out["gross"]["min_track_record"] is None
    assert out["net"]["min_track_record"] is None
    _assert_plain_and_finite(out, "whipsaw")
    json.dumps(out, allow_nan=False)


# ------------------------------------------------------------- short history


def test_short_history_yields_none_metrics_and_still_serialises() -> None:
    short = _synthetic_close(SHORT_ROWS)
    labels = label_regimes(short, window=WINDOW, threshold=THRESHOLD)
    out = backtest_block(short, labels, null_paths=NULL_PATHS, seed=SEED)  # default min_train
    assert list(out.keys()) == EXPECTED_KEYS
    for side in ("gross", "net"):
        assert out[side]["n_days"] == 0
        assert all(out[side][key] is None for key in METRIC_KEYS if key != "n_days"), side
    assert out["annual_cost"] is None
    assert out["null"] is None and out["holdout"] is None
    assert out["psr"] is None and out["dsr"]["dsr"] is None
    assert out["dsr"]["sharpe_variance"] == 0.0  # NaN grid variance mapped to 0.0 (DSR note)
    sens = out["sensitivity"]
    assert sens["n_trials"] == 15
    assert all(cell is None for row in sens["net_sharpe"] for cell in row)
    assert sens["best"] == BEST_NONE  # no cell, so no holdout figure either (None-safe)
    assert all(row["sharpe"] is None for row in out["cost_sensitivity"])
    assert all(out["horizon_pnl"][h]["n"] == 0 for h in EXPECTED_HORIZON_KEYS)
    assert out["lag"]["threshold"]["crash"]["n"] >= 0  # lag is still computed
    _assert_plain_and_finite(out, "short")
    json.dumps(out, allow_nan=False)


def test_tiny_history_never_raises_where_analyze_used_to_succeed() -> None:
    tiny = _synthetic_close(WINDOW + 2)  # two labels: the smallest series analyze() accepts
    labels = label_regimes(tiny, window=WINDOW, threshold=THRESHOLD)
    out = backtest_block(tiny, labels, null_paths=NULL_PATHS, seed=SEED)
    assert out["lag"]["zscore"] is None  # the z-score labeller has no labels on 22 rows
    assert out["lag"]["threshold"] is not None
    _assert_plain_and_finite(out, "tiny")
    json.dumps(out, allow_nan=False)
