"""The ``backtest`` JSON block: Phase 3 metrics, costs, benchmarks, nulls, holdout, lag.

``backtest_block`` wires the pure ``hedgefund.backtest`` modules into one JSON-ready dict
nested under the top-level key ``backtest`` (``.claude/rules/json-contract.md``). Every
number is a plain Python ``int`` / ``float`` / ``str`` / ``None``; NaN and inf become
``None`` so ``json.dumps(..., allow_nan=False)`` always succeeds.

Honesty rules enforced here (``.claude/rules/no-lookahead.md``, plan.md Fix 6):

* gross and net metric sets are always emitted together with the cost assumption echoed;
* every null model is reported with and without costs: the net keys charge each null
  path the strategy's cost model, the ``_gross`` keys charge nothing, and
  ``null_mean_turnover`` shows how much more a null trades (a coin-flipper turns over
  roughly ``periods_per_year`` units a year, so charging it costs would manufacture
  significance on its own);
* the two-year holdout is split off before any parameter is scored and the window x
  threshold sensitivity grid sees the in-sample close only; the in-sample best cell is
  then re-scored on the holdout so selection has an out-of-sample figure to answer to;
* the lag diagnostic uses future bars to find turning points and says so.

Short history (the walk-forward returns empty series) never raises: metrics come out as
``None`` with ``n_days`` 0, the null and holdout blocks are ``None`` and the sensitivity
grid is all ``None`` cells, so ``analyze()`` keeps working wherever it used to.

Backtests are historical, not forward-looking.
"""

import math
from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
import pandas as pd

from hedgefund.backtest.benchmarks import (
    DEFAULT_EXPOSURE,
    DEFAULT_NULL_PATHS,
    buy_and_hold,
    constant_exposure,
    long_only_positions,
    percentile_rank,
    permutation_p_value,
)
from hedgefund.backtest.costs import (
    SENSITIVITY_BPS,
    CostModel,
    apply_costs,
    cost_sensitivity,
    cost_series,
)
from hedgefund.backtest.lag import DEFAULT_LAG_ORDER, LAG_NOTE, detection_lag
from hedgefund.backtest.metrics import (
    DEFAULT_HORIZONS,
    TRADING_DAYS_PER_YEAR,
    annual_turnover,
    horizon_pnl,
    metric_set,
    sharpe_ratio,
)
from hedgefund.backtest.sensitivity import (
    DEFAULT_HOLDOUT_YEARS,
    DEFAULT_SENSITIVITY_THRESHOLDS,
    DEFAULT_SENSITIVITY_WINDOWS,
    holdout_cutoff,
    sensitivity_grid,
)
from hedgefund.backtest.significance import (
    deflated_sharpe_ratio,
    min_track_record_length,
    permutation_test,
    probabilistic_sharpe_ratio,
)
from hedgefund.backtest.walk_forward import (
    DEFAULT_DEAD_BAND,
    DEFAULT_MIN_TRAIN,
    DEFAULT_REFIT,
    DEFAULT_ROLLING_BARS,
    WalkForwardResult,
    run_walk_forward,
)
from hedgefund.regime.labels import (
    ensemble_label_regimes,
    ew_label_regimes,
    hysteresis_label_regimes,
    label_regimes,
    zscore_label_regimes,
)

DISCLAIMER = "Backtests are historical, not forward-looking."

DEFAULT_WINDOW = 20  # config labelling.window
DEFAULT_THRESHOLD = 0.05  # config labelling.threshold
DEFAULT_RETURN_TYPE = "simple"  # config labelling.return_type
DEFAULT_RULE = "sign"  # config backtest.rule: the Phase 1 position rule
DEFAULT_SEED = 0  # config statistics.seed
# Hysteresis exit threshold as a fraction of the entry threshold: plan Fix 3 pairs
# "enter Bear below -5%" with "leave Bear above -2%", so a little under half; exactly
# half keeps the pair tied to the single config threshold.
HYSTERESIS_EXIT_FRACTION = 0.5
# Labeller variants scored by the lag diagnostic (plan Fix 3), in report order.
LAG_LABELLERS: tuple[str, ...] = ("threshold", "zscore", "ew", "ensemble", "hysteresis")
BENCHMARK_NOTE = (
    "all benchmarks are net of the same cost model; buy-and-hold and constant 50% hold a "
    "fixed position so they pay the entry cost on day one plus any funding, never turnover"
)
NULL_NOTE = (
    "percentile = share of null draws strictly below the real Sharpe; "
    "p = (count(null >= real) + 1) / (N + 1). Each null is reported twice from the same "
    "seeded paths: the _net keys (and the unsuffixed percentile / p_value / null_mean / "
    "null_std) charge every null path the strategy's cost model and rank the real NET "
    "Sharpe; the _gross keys charge nothing and rank the real GROSS Sharpe. "
    "null_mean_turnover is the mean one-way annual turnover of the null paths, "
    "comparable to net.turnover: a null that trades far more than the strategy pays far "
    "more in costs, so its net figures flatter the strategy; read the gross pair first"
)
DSR_NOTE = (
    "conservative hybrid, not the textbook Deflated Sharpe Ratio: dsr is the PSR of the "
    "DEFAULT (window, threshold) cell's FULL-SAMPLE net returns against SR0, the expected "
    "maximum Sharpe of n_trials zero-skill trials whose variance is the per-period variance "
    "across the IN-SAMPLE sensitivity grid. The default cell is not the in-sample argmax "
    "(see sensitivity.best) and its sample is longer than the grid cells', so the two "
    "sides of the comparison are not the same experiment. When fewer than two cells could "
    "be scored sharpe_variance is taken as 0.0, so SR0 = 0 and dsr equals psr"
)


# ------------------------------------------------------------------ JSON scrubbing


def jsonable(node: object) -> object:
    """Recursively turn numpy scalars into Python numbers and NaN / inf into ``None``.

    Dict keys become strings, tuples become lists, integer-valued numpy ints stay ints.
    The result serialises with ``json.dumps(..., allow_nan=False)``.
    """
    if isinstance(node, dict):
        return {str(key): jsonable(value) for key, value in node.items()}
    if isinstance(node, list | tuple):
        return [jsonable(value) for value in node]
    if isinstance(node, np.ndarray):
        return [jsonable(value) for value in node.tolist()]
    if isinstance(node, bool | np.bool_):
        return bool(node)
    if isinstance(node, int | np.integer):
        return int(node)
    if isinstance(node, float | np.floating):
        number = float(node)
        return number if math.isfinite(number) else None
    return node


# --------------------------------------------------------------------- pipelines


def _labels_for(close: pd.Series, window: int, threshold: float, return_type: str) -> pd.Series:
    return label_regimes(close, window=window, threshold=threshold, return_type=return_type)


def _walk(
    close: pd.Series,
    labels: pd.Series,
    *,
    min_train: int,
    refit: str,
    rolling_bars: int,
    rule: str,
    dead_band: float,
) -> WalkForwardResult:
    return run_walk_forward(
        close,
        labels,
        min_train=min_train,
        refit=refit,
        rolling_bars=rolling_bars,
        rule=rule,
        dead_band=dead_band,
    )


def _metrics_with_track_record(
    returns: np.ndarray, positions: np.ndarray, periods_per_year: int
) -> dict:
    """``metric_set`` plus the Bailey / Lopez de Prado PSR and minimum track record."""
    out = metric_set(returns, positions, periods_per_year)
    out["psr"] = probabilistic_sharpe_ratio(returns)
    out["min_track_record"] = min_track_record_length(returns)
    return out


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _finite_std(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.std(ddof=1)) if finite.size > 1 else float("nan")


def _null_summary(real: float, null: np.ndarray, n_paths: int) -> dict:
    return {
        "percentile": percentile_rank(real, null),
        "p_value": permutation_p_value(real, null),
        "n_paths": int(n_paths),
        "null_mean": _finite_mean(null),
        "null_std": _finite_std(null),
    }


def _null_summary_gross_and_net(
    real_gross: float,
    real_net: float,
    null_gross: np.ndarray,
    null_net: np.ndarray,
    null_turnover: np.ndarray,
    n_paths: int,
) -> dict:
    """One null block with and without costs (plan Fix 6) from the same seeded paths.

    The unsuffixed keys are the NET comparison (real net Sharpe against null net Sharpes)
    exactly as before; ``*_net`` repeats them under a symmetric name, ``*_gross`` ranks
    the real gross Sharpe against the null's gross Sharpes, and ``null_mean_turnover``
    is the mean annual one-way turnover of the null paths (comparable to
    ``net.turnover``), which is why the two pairs differ.
    """
    net = _null_summary(real_net, null_net, n_paths)
    gross = _null_summary(real_gross, null_gross, n_paths)
    return {
        **net,
        "percentile_gross": gross["percentile"],
        "p_value_gross": gross["p_value"],
        "null_mean_gross": gross["null_mean"],
        "null_std_gross": gross["null_std"],
        "percentile_net": net["percentile"],
        "p_value_net": net["p_value"],
        "null_mean_net": net["null_mean"],
        "null_std_net": net["null_std"],
        "real_gross": float(real_gross),
        "real_net": float(real_net),
        "null_mean_turnover": _finite_mean(null_turnover),
    }


def _random_sign_null_draws(
    asset_returns: np.ndarray,
    model: CostModel,
    *,
    n_paths: int,
    seed: int,
    periods_per_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random-sign null with and without costs: ``(gross, net, turnover)`` per path.

    ``benchmarks.random_sign_null`` only exposes the path returns to its statistic, not
    the positions, so the cost model cannot be applied through it. This reproduces its
    draw exactly (same ``default_rng(seed)`` and ``rng.choice`` call) so ``gross`` is
    bit-identical to ``random_sign_null(asset_returns, n_paths, seed)``; ``net`` charges
    ``apply_costs(signs * r, signs, model)`` on the same signs and ``turnover`` is each
    path's ``annual_turnover`` (a coin-flipper flips on about half the days).
    """
    rets = np.asarray(asset_returns, dtype=float)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_paths, rets.size))
    gross = np.empty(n_paths, dtype=float)
    net = np.empty(n_paths, dtype=float)
    turnover = np.empty(n_paths, dtype=float)
    for p in range(n_paths):
        path = signs[p] * rets
        gross[p] = sharpe_ratio(path, periods_per_year)
        charged = apply_costs(path, signs[p], model, periods_per_year)
        net[p] = sharpe_ratio(charged, periods_per_year)
        turnover[p] = annual_turnover(signs[p], periods_per_year)
    return gross, net, turnover


def _random_sign_null_with_costs(
    asset_returns: np.ndarray,
    model: CostModel,
    *,
    n_paths: int,
    seed: int,
    periods_per_year: int,
) -> np.ndarray:
    """The NET Sharpe draw of ``_random_sign_null_draws`` (kept for its callers)."""
    return _random_sign_null_draws(
        asset_returns, model, n_paths=n_paths, seed=seed, periods_per_year=periods_per_year
    )[1]


ShufflePipeline = Callable[[pd.Series], tuple[np.ndarray, np.ndarray, np.ndarray]]


def _label_shuffle_null_draws(
    labels: pd.Series,
    pipeline: ShufflePipeline,
    *,
    n_paths: int,
    seed: int,
    periods_per_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label-shuffle null with and without costs: ``(gross, net, turnover)`` per path.

    Reproduces ``benchmarks.label_shuffle_null``'s draw exactly (same ``default_rng(seed)``
    and ``rng.permutation`` per path, index kept) but asks the pipeline for
    ``(gross_returns, net_returns, positions)`` so one permutation yields the gross Sharpe,
    the net Sharpe and the path's annual turnover together. ``net`` is bit-identical to
    ``label_shuffle_null(labels, net_pipeline, n_paths, seed)``.
    """
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    if labels.size < 2:
        raise ValueError(f"need at least 2 labels for a null model, got {labels.size}")
    values = labels.to_numpy()
    rng = np.random.default_rng(seed)
    gross = np.empty(n_paths, dtype=float)
    net = np.empty(n_paths, dtype=float)
    turnover = np.empty(n_paths, dtype=float)
    for p in range(n_paths):  # one full walk-forward per permutation
        shuffled = pd.Series(rng.permutation(values), index=labels.index, name=labels.name)
        g, n, pos = pipeline(shuffled)
        gross[p] = sharpe_ratio(np.asarray(g, dtype=float), periods_per_year)
        net[p] = sharpe_ratio(np.asarray(n, dtype=float), periods_per_year)
        turnover[p] = annual_turnover(np.asarray(pos, dtype=float), periods_per_year)
    return gross, net, turnover


def _lag_block(
    close: pd.Series, *, window: int, threshold: float, return_type: str, order: int
) -> dict:
    """``detection_lag`` per labeller variant (plan Fix 3); ``None`` where a variant has
    no labels on this history (``detection_lag`` raises ``ValueError``).

    Equivalent to ``lag.lag_by_labeller`` but tolerant of one variant failing on a short
    series, so the block never breaks a report that used to succeed.
    """
    labellers: dict[str, Callable[[pd.Series], pd.Series]] = {
        "threshold": partial(
            label_regimes, window=window, threshold=threshold, return_type=return_type
        ),
        "zscore": partial(zscore_label_regimes, window=window, return_type=return_type),
        "ew": partial(
            ew_label_regimes, window=window, threshold=threshold, return_type=return_type
        ),
        "ensemble": partial(ensemble_label_regimes, threshold=threshold, return_type=return_type),
        "hysteresis": partial(
            hysteresis_label_regimes,
            window=window,
            enter_threshold=threshold,
            exit_threshold=threshold * HYSTERESIS_EXIT_FRACTION,
            return_type=return_type,
        ),
    }
    out: dict = {}
    for name in LAG_LABELLERS:
        try:
            out[name] = detection_lag(close, labellers[name](close), order)
        except ValueError:
            out[name] = None
    out["order"] = int(order)
    out["note"] = LAG_NOTE
    return out


def _best_cell_holdout(
    close: pd.Series,
    best: dict,
    *,
    cutoff: pd.Timestamp,
    return_type: str,
    walk_kwargs: dict,
    model: CostModel,
    periods_per_year: int,
) -> tuple[float, float]:
    """``(gross, net)`` Sharpe of the in-sample best cell on the holdout dates only.

    The best cell was chosen on the in-sample close; this runs its labeller and
    walk-forward on the FULL close (so the holdout decisions use all history up to each
    day, as the strategy itself does), applies the cost model, and keeps the returns
    earned on dates at or after ``cutoff``. NaN when no best cell exists, when the cell
    cannot run, or when the holdout slice is too short for a Sharpe.
    """
    nan = float("nan")
    if best["window"] is None or best["threshold"] is None:
        return nan, nan
    try:
        cell_labels = _labels_for(close, int(best["window"]), float(best["threshold"]), return_type)
        result = _walk(close, cell_labels, **walk_kwargs)
    except ValueError:
        return nan, nan
    g = result.strategy_returns.to_numpy(dtype=float)
    p = result.positions.to_numpy(dtype=float)
    n = apply_costs(g, p, model, periods_per_year)
    holdout_mask = result.strategy_returns.index >= cutoff
    return (
        sharpe_ratio(g[holdout_mask], periods_per_year),
        sharpe_ratio(n[holdout_mask], periods_per_year),
    )


# ------------------------------------------------------------------- the block


def backtest_block(
    close: pd.Series,
    labels: pd.Series,
    *,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    return_type: str = DEFAULT_RETURN_TYPE,
    min_train: int = DEFAULT_MIN_TRAIN,
    refit: str = DEFAULT_REFIT,
    rolling_bars: int = DEFAULT_ROLLING_BARS,
    rule: str = DEFAULT_RULE,
    dead_band: float = DEFAULT_DEAD_BAND,
    cost_model: CostModel | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    null_paths: int = DEFAULT_NULL_PATHS,
    seed: int = DEFAULT_SEED,
    holdout_years: int = DEFAULT_HOLDOUT_YEARS,
    sensitivity_windows: Sequence[int] = DEFAULT_SENSITIVITY_WINDOWS,
    sensitivity_thresholds: Sequence[float] = DEFAULT_SENSITIVITY_THRESHOLDS,
    cost_sensitivity_bps: Sequence[int] = SENSITIVITY_BPS,
    lag_order: int = DEFAULT_LAG_ORDER,
) -> dict:
    """Compute the ``backtest`` JSON block for ``close`` and its regime ``labels``.

    ``labels`` must be ``label_regimes(close, window, threshold, return_type)``: the
    null models and the sensitivity grid re-run that labeller on shuffled or
    in-sample data, so the parameters have to describe the labels passed in.
    ``cost_model`` ``None`` means ``CostModel()`` (the ``us_etfs`` profile).

    Keys, in order: ``params`` (echo of every argument, ``costs`` nested), ``gross`` and
    ``net`` (``metric_set`` plus ``psr`` and ``min_track_record``), ``annual_cost``,
    ``horizon_pnl`` (string horizon keys), ``cost_sensitivity``, ``benchmarks``
    (``buy_and_hold``, ``constant_50``, ``long_only``, ``note``), ``null``
    (``random_sign``, ``label_shuffle``, ``permutation``, ``note``; each null carries
    gross and net percentile / p-value and its mean turnover), ``holdout``
    (``cutoff``, ``in_sample``, ``holdout`` with gross / net metric sets, and
    ``default_cell`` naming the (window, threshold) it scores), ``sensitivity`` (in-sample
    window x threshold grid of net Sharpe; ``best`` also carries the best cell's holdout
    gross / net Sharpe from a full-close run sliced at the cutoff), ``psr``, ``dsr``,
    ``lag`` and ``disclaimer``. ``gross["sharpe"]`` with the default rule and zero costs equals the
    companion ``walk_forward["sharpe"]`` exactly.

    The two-year holdout cutoff is ``holdout_cutoff(close.index, holdout_years)``, the
    same date the strategy return index yields because both end on the last close.
    """
    model = CostModel() if cost_model is None else cost_model
    close = close.dropna()
    walk_kwargs = {
        "min_train": min_train,
        "refit": refit,
        "rolling_bars": rolling_bars,
        "rule": rule,
        "dead_band": dead_band,
    }
    wf = _walk(close, labels, **walk_kwargs)
    gross = wf.strategy_returns.to_numpy(dtype=float)
    positions = wf.positions.to_numpy(dtype=float)
    asset = wf.asset_returns.to_numpy(dtype=float)
    net = apply_costs(gross, positions, model, periods_per_year)
    has_history = gross.size > 0

    costs = cost_series(positions, model, periods_per_year)
    annual_cost = float(costs.mean() * periods_per_year) if has_history else float("nan")

    def gross_net_positions(result: WalkForwardResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = result.strategy_returns.to_numpy(dtype=float)
        p = result.positions.to_numpy(dtype=float)
        return g, apply_costs(g, p, model, periods_per_year), p

    # ``permutation_test`` calls the pipeline once on the real close and then once per
    # null path, exposing only the returns; the positions are recorded here so the
    # permutation null can report its turnover next to the other two nulls.
    permutation_turnovers: list[float] = []

    def full_pipeline(close_path: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Labeller -> walk-forward -> costs on any price path (permutation null)."""
        path_labels = _labels_for(close_path, window, threshold, return_type)
        g, n, p = gross_net_positions(_walk(close_path, path_labels, **walk_kwargs))
        permutation_turnovers.append(annual_turnover(p, periods_per_year))
        return g, n

    def label_pipeline(shuffled: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Walk-forward -> costs on shuffled labels (label-shuffle null)."""
        return gross_net_positions(_walk(close, shuffled, **walk_kwargs))

    # Benchmarks: constant positions pay entry cost (and funding) through apply_costs.
    ones = np.ones_like(asset)
    bh_net = apply_costs(buy_and_hold(asset), ones, model, periods_per_year)
    half = np.full_like(asset, DEFAULT_EXPOSURE)
    c50_net = apply_costs(constant_exposure(asset, DEFAULT_EXPOSURE), half, model, periods_per_year)
    lo_pos = long_only_positions(positions)
    lo_net = apply_costs(lo_pos * asset, lo_pos, model, periods_per_year)
    benchmarks = {
        "buy_and_hold": metric_set(bh_net, ones, periods_per_year),
        "constant_50": metric_set(c50_net, half, periods_per_year),
        "long_only": metric_set(lo_net, lo_pos, periods_per_year),
        "note": BENCHMARK_NOTE,
    }

    null: dict | None = None
    if has_history:
        gross_sharpe = sharpe_ratio(gross, periods_per_year)
        net_sharpe = sharpe_ratio(net, periods_per_year)
        sign_draws = _random_sign_null_draws(
            asset, model, n_paths=null_paths, seed=seed, periods_per_year=periods_per_year
        )
        shuffle_draws = _label_shuffle_null_draws(
            labels, label_pipeline, n_paths=null_paths, seed=seed, periods_per_year=periods_per_year
        )
        permutation = permutation_test(
            close,
            full_pipeline,
            n_paths=null_paths,
            seed=seed,
            statistic=partial(sharpe_ratio, periods_per_year=periods_per_year),
        )
        # The first recorded turnover is the real path's; the rest are the null paths'.
        null_turnovers = np.asarray(permutation_turnovers[1:], dtype=float)
        permutation["null_mean_turnover"] = (
            _finite_mean(null_turnovers) if null_turnovers.size == null_paths else float("nan")
        )
        null = {
            "random_sign": _null_summary_gross_and_net(
                gross_sharpe, net_sharpe, *sign_draws, null_paths
            ),
            "label_shuffle": _null_summary_gross_and_net(
                gross_sharpe, net_sharpe, *shuffle_draws, null_paths
            ),
            "permutation": permutation,
            "note": NULL_NOTE,
        }

    # Holdout split on the earned-return index; positions are split by the same step
    # mask so each half keeps returns and positions aligned entry for entry.
    cutoff = holdout_cutoff(close.index, holdout_years)
    holdout: dict | None = None
    if has_history:
        in_mask = wf.strategy_returns.index < cutoff
        holdout = {
            "cutoff": str(cutoff.date()),
            "holdout_years": int(holdout_years),
            "in_sample": {
                "gross": metric_set(gross[in_mask], positions[in_mask], periods_per_year),
                "net": metric_set(net[in_mask], positions[in_mask], periods_per_year),
            },
            "holdout": {
                "gross": metric_set(gross[~in_mask], positions[~in_mask], periods_per_year),
                "net": metric_set(net[~in_mask], positions[~in_mask], periods_per_year),
            },
            # Which (window, threshold) the in_sample / holdout metric sets score: the
            # default cell, never the in-sample best (that one lives in sensitivity.best).
            "default_cell": {"window": int(window), "threshold": float(threshold)},
        }

    in_sample_close = close.loc[close.index < cutoff]

    def score(close_is: pd.Series, w: int, t: float) -> float:
        """Net Sharpe of one grid cell on the in-sample close; NaN when it cannot run."""
        try:
            cell_labels = _labels_for(close_is, w, t, return_type)
            result = _walk(close_is, cell_labels, **walk_kwargs)
        except ValueError:
            return float("nan")
        g = result.strategy_returns.to_numpy(dtype=float)
        p = result.positions.to_numpy(dtype=float)
        return sharpe_ratio(apply_costs(g, p, model, periods_per_year), periods_per_year)

    sensitivity = sensitivity_grid(
        in_sample_close, score, windows=sensitivity_windows, thresholds=sensitivity_thresholds
    )
    sensitivity["note"] = (
        f"in-sample only (dates before {cutoff.date()}); rows = windows, cols = thresholds; "
        "best.holdout_*_sharpe re-runs the best cell on the full close and scores dates at "
        "or after the cutoff only"
    )
    best_holdout_gross, best_holdout_net = _best_cell_holdout(
        close,
        sensitivity["best"],
        cutoff=cutoff,
        return_type=return_type,
        walk_kwargs=walk_kwargs,
        model=model,
        periods_per_year=periods_per_year,
    )
    sensitivity["best"]["holdout_gross_sharpe"] = best_holdout_gross
    sensitivity["best"]["holdout_net_sharpe"] = best_holdout_net
    sharpe_variance = sensitivity["sharpe_variance"]
    if not np.isfinite(sharpe_variance):
        sharpe_variance = 0.0  # see DSR_NOTE
    dsr = deflated_sharpe_ratio(
        net, n_trials=sensitivity["n_trials"], sharpe_variance=sharpe_variance
    )
    dsr["note"] = DSR_NOTE

    block = {
        "params": {
            "window": int(window),
            "threshold": float(threshold),
            "return_type": return_type,
            "min_train": int(min_train),
            "refit": refit,
            "rolling_bars": int(rolling_bars),
            "rule": rule,
            "dead_band": float(dead_band),
            "periods_per_year": int(periods_per_year),
            "horizons": [int(h) for h in horizons],
            "null_paths": int(null_paths),
            "seed": int(seed),
            "holdout_years": int(holdout_years),
            "sensitivity": {
                "windows": [int(w) for w in sensitivity_windows],
                "thresholds": [float(t) for t in sensitivity_thresholds],
            },
            "cost_sensitivity_bps": [int(b) for b in cost_sensitivity_bps],
            "lag_order": int(lag_order),
            "costs": {
                "fee_bps": float(model.fee_bps),
                "slippage_bps": float(model.slippage_bps),
                "borrow_bps": float(model.borrow_bps),
                "funding_bps": float(model.funding_bps),
                "per_side_bps": float(model.per_side_bps),
            },
        },
        "gross": _metrics_with_track_record(gross, positions, periods_per_year),
        "net": _metrics_with_track_record(net, positions, periods_per_year),
        "annual_cost": annual_cost,
        "horizon_pnl": {
            str(h): stats
            for h, stats in horizon_pnl(positions, asset, tuple(int(h) for h in horizons)).items()
        },
        "cost_sensitivity": cost_sensitivity(
            gross, positions, tuple(int(b) for b in cost_sensitivity_bps), periods_per_year
        ),
        "benchmarks": benchmarks,
        "null": null,
        "holdout": holdout,
        "sensitivity": sensitivity,
        "psr": probabilistic_sharpe_ratio(net),
        "dsr": dsr,
        "lag": _lag_block(
            close, window=window, threshold=threshold, return_type=return_type, order=lag_order
        ),
        "disclaimer": DISCLAIMER,
    }
    return jsonable(block)
