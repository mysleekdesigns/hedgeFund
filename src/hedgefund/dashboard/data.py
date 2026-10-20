"""Offline data layer for the Streamlit dashboard (plan.md Phase 8).

Everything the app shows comes from a cached scan run under ``jobs/<run_id>/`` and the
parquet price cache that scan wrote. Nothing here imports Streamlit and nothing opens a
network connection, so the layer is testable on its own and the Phase 8 gate ("dashboard
opens from a cached scan with no network calls") holds by construction.

Two paths produce a ``SymbolView``:

- The controls equal the run's own parameters: the matrix, its intervals, the stationary
  mix, the mixing time and the signal are read straight from the result JSON, so the
  dashboard shows exactly what ``hedgefund scan`` reported.
- The user moves a control: the same pure functions the scan used are re-run on the cached
  closes (``label_regimes`` -> ``build_transition_matrix`` -> ``block_bootstrap_ci`` -> ...),
  with the run's own bootstrap size, block length and seed, so the two paths are
  consistent (a test proves the recomputed matrix equals the JSON one).

Equity curves and the null band are always rebuilt from the closes because the result
JSON stores no daily series. The curves come from ``report.html.equity_curves`` and the
band from the same seeded random-sign draw the ``backtest`` percentiles used, net of the
selected cost profile. Labels are known at their own bar and the walk-forward decides at
``t`` for the return of ``t+1``; no hindsight anywhere.

The stationary distribution shown is the long-run mix implied by the current matrix, not
a forecast. Backtests are historical, not forward-looking.
"""

import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hedgefund.backtest.costs import CostModel, apply_costs, cost_model_from_config
from hedgefund.backtest.walk_forward import REFIT_MODES, run_walk_forward
from hedgefund.config import load_config
from hedgefund.dashboard import JOBS_DIR_ENV
from hedgefund.data.base import close_series
from hedgefund.data.cache import read_cached
from hedgefund.jobs.manifest import (
    DEFAULT_JOBS_DIR,
    REPO_ROOT,
    RunManifest,
    list_runs,
    read_manifest,
    read_results,
    read_table,
)
from hedgefund.regime.labels import STATES, label_regimes
from hedgefund.regime.markov import (
    build_transition_matrix,
    mixing_time,
    nstep_forecast,
    stationary_distribution,
)
from hedgefund.regime.signal import next_state_probabilities, signal_from_matrix
from hedgefund.regime.stats import (
    NULL_BAND_LOWER_PCT,
    NULL_BAND_UPPER_PCT,
    block_bootstrap_ci,
)
from hedgefund.report.html import CURVE_KEYS, equity_curves

__all__ = [
    "CURVE_KEYS",
    "JOBS_DIR_ENV",
    "REFIT_MODES",
    "STATES",
    "Controls",
    "LoadedRun",
    "SymbolView",
    "compute_symbol_view",
    "control_options",
    "controls_from_run",
    "controls_match_run",
    "convergence_distance",
    "list_run_ids",
    "load_closes",
    "load_run",
    "nstep_sequence",
    "null_equity_band",
    "recorded_cost_model",
    "resolve_cache_path",
    "resolve_jobs_dir",
    "select_cost_model",
]

N_STATES = len(STATES)
STATE_KEYS: tuple[str, ...] = ("bear", "sideways", "bull")  # JSON keys in STATES order
# The scan's interval level when the result carries none; json_contract.BOOTSTRAP_CI.
DEFAULT_CI = 0.95
MIN_WINDOW = 2  # a trailing return needs at least two closes
MIN_ROLLING_BARS = 1  # ``run_walk_forward`` validates the real floor; this guards the control
START_EQUITY = 1.0  # every curve and the band start at 1.0 on the first decision date
EQUITY_ONE = 1.0  # compounding: equity_t = prod(1 + r)
_EMPTY_INDEX = pd.DatetimeIndex([])


# ------------------------------------------------------------------------- locating runs


def resolve_jobs_dir(
    jobs_dir: Path | str | None = None, env: Mapping[str, str] | None = None
) -> Path:
    """Jobs directory: explicit argument, then ``env[JOBS_DIR_ENV]``, then the default.

    ``env`` defaults to ``os.environ``; the ``hedgefund dashboard`` command sets the
    variable so the Streamlit subprocess (which swallows script argv) reads the same
    directory the user passed on the command line.
    """
    if jobs_dir is not None:
        return Path(jobs_dir)
    mapping = os.environ if env is None else env
    from_env = mapping.get(JOBS_DIR_ENV)
    if from_env:
        return Path(from_env)
    return DEFAULT_JOBS_DIR


def list_run_ids(jobs_dir: Path | str | None = None) -> list[str]:
    """Run ids under ``jobs_dir`` newest first; ``[]`` when the directory is missing."""
    return list(reversed(list_runs(resolve_jobs_dir(jobs_dir))))


def resolve_cache_path(cache_path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Absolute path of a manifest ``cache_path``; a relative one is joined to ``repo_root``.

    ``hedgefund scan`` records ``str(path)`` and the default ``data.cache_dir`` is the
    relative ``cache``, so a manifest written from the repo root holds relative paths
    that only resolve from there. The dashboard may start anywhere.
    """
    path = Path(cache_path)
    return path if path.is_absolute() else Path(repo_root) / path


def load_closes(
    manifest: RunManifest, results: dict[str, dict], repo_root: Path = REPO_ROOT
) -> dict[str, pd.Series]:
    """Close series for every symbol in ``results``, read from the run's parquet cache.

    Offline by construction. Raises ``ValueError`` naming the symbol and path when the
    manifest records no ``cache_path`` or the file cannot be read, so the user knows to
    re-run ``hedgefund scan`` (the cache, not the run, is what is missing).
    """
    closes: dict[str, pd.Series] = {}
    for symbol in results:
        entry = manifest.data.get(symbol) or {}
        cache_path = entry.get("cache_path")
        if not cache_path:
            raise ValueError(
                f"run {manifest.run_id} records no cache_path for {symbol}; re-run hedgefund scan."
            )
        path = resolve_cache_path(cache_path, repo_root)
        try:
            frame = read_cached(path)
        except Exception as exc:
            raise ValueError(
                f"cannot read cached prices for {symbol} at {path}: {exc}. "
                "Re-run hedgefund scan to refresh the cache."
            ) from exc
        closes[symbol] = close_series(frame, symbol)
    return closes


@dataclass(frozen=True)
class LoadedRun:
    """One scan run with its manifest, per-symbol results, ranked table, closes and config."""

    run_id: str
    jobs_dir: Path
    manifest: RunManifest
    results: dict[str, dict]
    table: list[dict]
    closes: dict[str, pd.Series]
    config: dict

    @property
    def symbols(self) -> list[str]:
        """Symbols with both a result and closes, in manifest order."""
        return [s for s in self.manifest.symbols if s in self.results and s in self.closes]


def load_run(
    run_id: str,
    jobs_dir: Path | str | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    config: dict | None = None,
) -> LoadedRun:
    """Read manifest, results, table and cached closes of one run; ``config`` defaults to
    ``config/default.yaml``. Raises ``ValueError`` (from the readers) when a file is missing.
    """
    directory = resolve_jobs_dir(jobs_dir)
    manifest = read_manifest(run_id, directory)
    results = read_results(run_id, directory)
    table = read_table(run_id, directory)
    closes = load_closes(manifest, results, repo_root)
    return LoadedRun(
        run_id=run_id,
        jobs_dir=directory,
        manifest=manifest,
        results=results,
        table=table,
        closes=closes,
        config=load_config() if config is None else config,
    )


# -------------------------------------------------------------------------------- controls


@dataclass(frozen=True)
class Controls:
    """The dashboard's user-settable parameters for one symbol.

    ``return_type`` is carried so a recomputation labels exactly like the run did.
    Validation raises ``ValueError`` naming the field.
    """

    symbol: str
    window: int
    threshold: float
    refit: str
    rolling_bars: int
    cost_profile: str
    return_type: str = "simple"  # config/default.yaml labelling.return_type

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must be a non-empty string")
        if int(self.window) != self.window or self.window < MIN_WINDOW:
            raise ValueError(f"window must be an integer >= {MIN_WINDOW}, got {self.window!r}")
        if not np.isfinite(self.threshold) or not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must lie strictly in (0, 1), got {self.threshold!r}")
        if self.refit not in REFIT_MODES:
            raise ValueError(f"refit must be one of {REFIT_MODES}, got {self.refit!r}")
        if int(self.rolling_bars) != self.rolling_bars or self.rolling_bars < MIN_ROLLING_BARS:
            raise ValueError(
                f"rolling_bars must be an integer >= {MIN_ROLLING_BARS}, got {self.rolling_bars!r}"
            )
        if not self.cost_profile:
            raise ValueError("cost_profile must be a non-empty string")


def _config_get(config: dict, *keys: str, default: object = None) -> object:
    node: object = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def control_options(config: dict, run: LoadedRun | None = None) -> dict:
    """Selectable values per control, from the config plus the run's own parameters.

    ``windows`` and ``thresholds`` come from ``backtest.sensitivity``, ``cost_profiles``
    from ``costs.profiles`` and ``rolling_bars`` from ``backtest.rolling_bars``. The run's
    own window, threshold and cost profile are added so its defaults are always
    selectable even when the scan used values outside the sensitivity grid.
    """
    windows = {
        int(w) for w in _config_get(config, "backtest", "sensitivity", "windows", default=[])
    }
    thresholds = {
        float(t) for t in _config_get(config, "backtest", "sensitivity", "thresholds", default=[])
    }
    profiles = set(_config_get(config, "costs", "profiles", default={}))
    rolling_bars = _config_get(config, "backtest", "rolling_bars")
    if run is not None:
        params = run.manifest.params
        windows.add(int(params["window"]))
        thresholds.add(float(params["threshold"]))
        profiles.add(str(run.manifest.cost_profile))
        if rolling_bars is None:
            rolling_bars = params.get("backtest_rolling_bars")
    if rolling_bars is None:
        raise ValueError("config backtest.rolling_bars is missing and the run records none")
    return {
        "windows": sorted(windows),
        "thresholds": sorted(thresholds),
        "refits": list(REFIT_MODES),
        "cost_profiles": sorted(profiles),
        "rolling_bars": int(rolling_bars),
    }


def controls_from_run(run: LoadedRun, symbol: str) -> Controls:
    """The controls that reproduce the run: ``manifest.params`` plus its cost profile."""
    params = run.manifest.params
    return Controls(
        symbol=symbol,
        window=int(params["window"]),
        threshold=float(params["threshold"]),
        refit=str(params["refit"]),
        rolling_bars=int(params["backtest_rolling_bars"]),
        cost_profile=str(run.manifest.cost_profile),
        return_type=str(params["return_type"]),
    )


COST_FIELDS: tuple[str, ...] = ("fee_bps", "slippage_bps", "borrow_bps", "funding_bps")


def _cost_model_from_dict(values: Mapping[str, object], what: str) -> CostModel:
    """``CostModel`` from a JSON echo; extra keys (``per_side_bps``) are ignored."""
    missing = [key for key in COST_FIELDS if key not in values]
    if missing:
        raise ValueError(f"{what} must carry {COST_FIELDS}, missing {missing}")
    return CostModel(**{key: float(values[key]) for key in COST_FIELDS})


def recorded_cost_model(run: LoadedRun, symbol: str | None = None) -> CostModel:
    """The cost model the scan actually charged, from ``manifest.params["cost_model"]``.

    Older manifests without that key fall back to the ``backtest.params.costs`` echo of
    ``symbol``'s result (or the first result). This is what the result JSON was computed
    with, so it is the model to show whenever the run's own profile is selected, even if
    ``config/default.yaml`` has changed since the scan.
    """
    recorded = run.manifest.params.get("cost_model")
    if isinstance(recorded, Mapping):
        return _cost_model_from_dict(recorded, f"run {run.run_id} params.cost_model")
    key = symbol if symbol is not None and symbol in run.results else next(iter(run.results), None)
    if key is None:
        raise ValueError(
            f"run {run.run_id} records no cost_model and has no result to read it from"
        )
    costs = run.results[key].get("backtest", {}).get("params", {}).get("costs")
    if not isinstance(costs, Mapping):
        raise ValueError(f"run {run.run_id} records no cost_model and {key} echoes no costs")
    return _cost_model_from_dict(costs, f"result {key} backtest.params.costs")


def _live_cost_model(run: LoadedRun, profile: str) -> CostModel | None:
    """``cost_model_from_config`` for ``profile``, ``None`` when the live config lacks it."""
    try:
        return cost_model_from_config(run.config, profile)
    except ValueError:
        return None


def select_cost_model(run: LoadedRun, controls: Controls) -> CostModel:
    """The model the view charges: the recorded one for the run's own profile, else the
    live config profile. Raises ``ValueError`` for a profile the config does not define.
    """
    if controls.cost_profile == str(run.manifest.cost_profile):
        return recorded_cost_model(run, controls.symbol)
    return cost_model_from_config(run.config, controls.cost_profile)


def controls_match_run(run: LoadedRun, controls: Controls) -> bool:
    """True when every control equals the run's own parameter, so the JSON can be shown as is.

    Also False when the live config's entry for the run's profile no longer equals the
    model the scan recorded (someone edited ``costs.profiles`` since), so the "from the
    scan" badge is never shown next to a cost assumption the config now contradicts.
    """
    params = run.manifest.params
    same_params = (
        int(params["window"]) == controls.window
        and float(params["threshold"]) == controls.threshold
        and str(params["return_type"]) == controls.return_type
        and str(params["refit"]) == controls.refit
        and int(params["backtest_rolling_bars"]) == controls.rolling_bars
        and str(run.manifest.cost_profile) == controls.cost_profile
    )
    if not same_params:
        return False
    return _live_cost_model(run, controls.cost_profile) == recorded_cost_model(run, controls.symbol)


# ----------------------------------------------------------------------------- symbol view


@dataclass(frozen=True)
class SymbolView:
    """Everything the dashboard draws for one symbol under one set of controls.

    Arrays use the ``STATES`` order. ``labels`` are known at their own bar. ``curves`` are
    keyed exactly by ``CURVE_KEYS`` and share one index with ``null_lower`` /
    ``null_upper``: the first decision date (equity 1.0) then the dates whose return was
    earned. ``recomputed`` is False when the numbers came from the result JSON.
    ``stationary`` is the long-run mix implied by ``matrix``, not a forecast.
    """

    symbol: str
    controls: Controls
    close: pd.Series
    labels: pd.Series
    matrix: np.ndarray
    matrix_lower: np.ndarray
    matrix_upper: np.ndarray
    stationary: np.ndarray
    stationary_lower: np.ndarray
    stationary_upper: np.ndarray
    current_state: int
    signal: float
    next_state_probabilities: np.ndarray
    mixing_time: int | None
    curves: dict[str, pd.Series]
    null_lower: pd.Series
    null_upper: pd.Series
    null_paths: int
    cost_model: CostModel
    recomputed: bool
    result: dict


def _vector(values: Mapping[str, object]) -> np.ndarray:
    """``{"bear", "sideways", "bull"}`` -> array in ``STATES`` order."""
    missing = [key for key in STATE_KEYS if key not in values]
    if missing:
        raise ValueError(f"state vector must have keys {STATE_KEYS}, missing {missing}")
    return np.array([float(values[key]) for key in STATE_KEYS], dtype=float)


def _matrix(values: object) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (N_STATES, N_STATES):
        raise ValueError(f"transition matrix must be {N_STATES}x{N_STATES}, got {arr.shape}")
    return arr


def _result_with_controls(result: dict, controls: Controls, cost_model: CostModel) -> dict:
    """Deep copy of ``result`` whose labeller, walk-forward and cost echoes carry the selection."""
    out = copy.deepcopy(result)
    out["params"]["window"] = int(controls.window)
    out["params"]["threshold"] = float(controls.threshold)
    out["params"]["return_type"] = controls.return_type
    bt = out["backtest"]["params"]
    bt["window"] = int(controls.window)
    bt["threshold"] = float(controls.threshold)
    bt["return_type"] = controls.return_type
    bt["refit"] = controls.refit
    bt["rolling_bars"] = int(controls.rolling_bars)
    bt["costs"] = {
        **{key: float(getattr(cost_model, key)) for key in COST_FIELDS},
        "per_side_bps": float(cost_model.per_side_bps),
    }
    return out


def _with_start(values: np.ndarray, start: pd.Timestamp, index: pd.Index) -> pd.Series:
    """Series starting at ``START_EQUITY`` on ``start`` followed by ``values`` on ``index``."""
    full = np.concatenate(([START_EQUITY], np.asarray(values, dtype=float)))
    full_index = pd.DatetimeIndex([start]).append(pd.DatetimeIndex(index))
    return pd.Series(full, index=full_index, dtype=float)


def _empty_series() -> pd.Series:
    return pd.Series(np.empty(0, dtype=float), index=_EMPTY_INDEX.copy(), dtype=float)


def null_equity_band(
    asset_returns: np.ndarray,
    cost_model: CostModel,
    *,
    n_paths: int,
    seed: int,
    lower_pct: float = NULL_BAND_LOWER_PCT,
    upper_pct: float = NULL_BAND_UPPER_PCT,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-date percentile band of the compounded equity of costed random-sign paths.

    Reproduces the draw of ``backtest_report._random_sign_null_draws`` exactly (same
    ``default_rng(seed)`` and ``rng.choice`` call, same ``apply_costs(signs * r, signs,
    model)``) so the band is the equity of the very null the JSON ``percentile_net`` was
    ranked against. Returns ``(lower, upper)`` each of length ``len(asset_returns)``;
    both empty when there are no returns. Raises ``ValueError`` for ``n_paths < 1`` or
    percentiles outside ``[0, 100]`` or out of order.

    The band is not prefix-stable as history grows: the row-major
    ``rng.choice(size=(n_paths, T))`` draw reassigns signs when ``T`` changes. That is
    fine because the band is a null reference, not a decision input; no position is
    derived from it, so this is not lookahead.
    """
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    if not 0.0 <= lower_pct <= upper_pct <= 100.0:
        raise ValueError(
            f"percentiles must satisfy 0 <= lower <= upper <= 100, got {lower_pct}, {upper_pct}"
        )
    rets = np.asarray(asset_returns, dtype=float)
    if rets.ndim != 1:
        raise ValueError(f"asset_returns must be 1-D, got shape {rets.shape}")
    if rets.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_paths, rets.size))
    equity = np.empty((n_paths, rets.size), dtype=float)
    for p in range(n_paths):
        net = apply_costs(signs[p] * rets, signs[p], cost_model)
        equity[p] = np.cumprod(EQUITY_ONE + net)
    lower = np.percentile(equity, lower_pct, axis=0)
    upper = np.percentile(equity, upper_pct, axis=0)
    return lower, upper


def compute_symbol_view(
    run: LoadedRun,
    controls: Controls,
    *,
    null_paths: int | None = None,
    seed: int | None = None,
    force_recompute: bool = False,
) -> SymbolView:
    """Build the ``SymbolView`` for ``controls.symbol``.

    When ``controls_match_run`` (and not ``force_recompute``) the matrix, intervals,
    stationary mix, mixing time and signal come straight from the result JSON
    (``recomputed=False``). Otherwise they are recomputed from the cached closes with the
    run's own bootstrap resamples, block length, seed, interval level and mixing epsilon
    (``recomputed=True``); with the run's own controls that reproduces the JSON exactly.
    Curves and the null band are always rebuilt from the closes, charged
    ``select_cost_model``: the model the scan recorded when the run's own profile is
    selected, the live config profile otherwise. ``null_paths`` and ``seed`` default to
    the run's. A history too short for the walk-forward yields empty curves and an empty
    band rather than an error. Raises ``ValueError`` when the symbol has no result or no
    closes in the run.
    """
    symbol = controls.symbol
    if symbol not in run.results:
        raise ValueError(f"run {run.run_id} has no result for {symbol!r}")
    if symbol not in run.closes:
        raise ValueError(f"run {run.run_id} has no cached closes for {symbol!r}")
    result = run.results[symbol]
    params = run.manifest.params
    close = run.closes[symbol].dropna()
    labels = label_regimes(
        close,
        window=controls.window,
        threshold=controls.threshold,
        return_type=controls.return_type,
    )
    if labels.empty:
        raise ValueError(
            f"{symbol}: {close.size} closes are too few to label with window {controls.window}"
        )
    current_state = int(labels.iloc[-1])
    matching = controls_match_run(run, controls) and not force_recompute

    if matching:
        matrix = _matrix(result["transition_matrix"])
        ci = result["confidence_intervals"]
        matrix_lower = _matrix(ci["transition_matrix"]["lower"])
        matrix_upper = _matrix(ci["transition_matrix"]["upper"])
        stationary = _vector(result["stationary_distribution"])
        stationary_lower = _vector(ci["stationary_distribution"]["lower"])
        stationary_upper = _vector(ci["stationary_distribution"]["upper"])
        mix = result.get("mixing_time")
        mixing = None if mix is None else int(mix)
        signal = float(result["signal"])
        current_state = STATES.index(result["current_regime"])
    else:
        matrix = build_transition_matrix(labels)
        ci_level = float(result.get("confidence_intervals", {}).get("ci", DEFAULT_CI))
        boot = block_bootstrap_ci(
            labels,
            n_resamples=int(params["bootstrap_resamples"]),
            block_days=int(params["block_days"]),
            ci=ci_level,
            seed=int(params["seed"]),
        )
        matrix_lower = boot["matrix_lower"]
        matrix_upper = boot["matrix_upper"]
        stationary = stationary_distribution(matrix)
        stationary_lower = boot["stationary_lower"]
        stationary_upper = boot["stationary_upper"]
        mixing = mixing_time(matrix, epsilon=float(params["mixing_epsilon"]))
        signal = signal_from_matrix(matrix, current_state)

    cost_model = select_cost_model(run, controls)
    overridden = _result_with_controls(result, controls, cost_model)
    curves = equity_curves(close, overridden, cost_model=cost_model)

    # ``equity_curves`` runs the walk-forward internally but exposes no asset returns, so
    # the band needs a second (identical, deterministic) run. Cheap next to the bootstrap.

    bt = overridden["backtest"]["params"]
    wf = run_walk_forward(
        close,
        labels,
        min_train=int(bt["min_train"]),
        refit=controls.refit,
        rolling_bars=controls.rolling_bars,
        rule=str(bt["rule"]),
        dead_band=float(bt["dead_band"]),
    )
    paths = int(params["null_paths"]) if null_paths is None else int(null_paths)
    draw_seed = int(params["seed"]) if seed is None else int(seed)
    asset = wf.asset_returns.to_numpy(dtype=float)
    if asset.size == 0:
        null_lower, null_upper = _empty_series(), _empty_series()
    else:
        lower, upper = null_equity_band(asset, cost_model, n_paths=paths, seed=draw_seed)
        start = wf.positions.index[0]
        earned = wf.asset_returns.index
        null_lower = _with_start(lower, start, earned)
        null_upper = _with_start(upper, start, earned)

    return SymbolView(
        symbol=symbol,
        controls=controls,
        close=close,
        labels=labels,
        matrix=matrix,
        matrix_lower=matrix_lower,
        matrix_upper=matrix_upper,
        stationary=stationary,
        stationary_lower=stationary_lower,
        stationary_upper=stationary_upper,
        current_state=current_state,
        signal=signal,
        next_state_probabilities=next_state_probabilities(matrix, current_state),
        mixing_time=mixing,
        curves=curves,
        null_lower=null_lower,
        null_upper=null_upper,
        null_paths=paths,
        cost_model=cost_model,
        recomputed=not matching,
        result=result,
    )


# ------------------------------------------------------------------------ P**n convergence


def nstep_sequence(matrix: np.ndarray, max_n: int) -> np.ndarray:
    """``P**0 .. P**max_n`` stacked as shape ``(max_n + 1, n, n)`` for the animation.

    Raises ``ValueError`` for ``max_n < 0`` or a non-square matrix.
    """
    if max_n < 0:
        raise ValueError(f"max_n must be >= 0, got {max_n}")
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"matrix must be square, got shape {arr.shape}")
    return np.stack([nstep_forecast(arr, n) for n in range(max_n + 1)])


def convergence_distance(matrix: np.ndarray, max_n: int) -> np.ndarray:
    """Max over start states of ``TVD(row of P**n, stationary)`` for ``n = 0 .. max_n``.

    The same distance ``mixing_time`` thresholds, so the first index at or below the
    epsilon equals the reported mixing time. Shape ``(max_n + 1,)``.
    """
    powers = nstep_sequence(matrix, max_n)
    pi = stationary_distribution(matrix)
    tvd = 0.5 * np.abs(powers - pi).sum(axis=2)
    return tvd.max(axis=1)
