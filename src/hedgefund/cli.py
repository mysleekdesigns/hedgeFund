"""Command-line entry point for hedgefund.

``hedgefund analyze --ticker SYM | --csv PATH [--json]`` mirrors the companion script's
CLI and appends the honest statistics, the ``backtest`` block and the Phase 4
walk-forward ``hmm`` block (``--hmm-seeds``, ``--hmm-refit-every``, ``--hmm-emission``
override the ``hmm`` config block; ``--no-hmm`` skips it and then the block is not read).
``hedgefund backtest`` runs the cost-aware walk-forward alone (plan.md Phase 3) with cost
and rule flags. ``hedgefund scan --universe NAME`` runs the full ``analyze`` pipeline per
symbol through the parquet cache, writes a run manifest, results and a ranked table under
``jobs/<run-id>/`` and renders the HTML report (Phase 5). ``hedgefund report RUN_ID``
regenerates that HTML from the cached run without touching the network. ``hedgefund
daily`` is the Phase 7 scheduled job: a scan plus a signal file and a summary POST to the
consumers under ``webhooks/``; ``hedgefund propose`` turns that signal file into a paper
allocation and prints the proposed orders without submitting anything. ``hedgefund
dashboard`` launches the optional Streamlit app over the cached runs (Phase 8). Config
over flags: every default comes from ``config/default.yaml``; a flag overrides it.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from hedgefund import __version__
from hedgefund.backtest.costs import CostModel, cost_model_from_config
from hedgefund.backtest.walk_forward import REFIT_MODES
from hedgefund.config import DEFAULT_CONFIG_PATH, load_config
from hedgefund.dashboard import APP_PATH, JOBS_DIR_ENV
from hedgefund.data.base import PriceSource, close_series
from hedgefund.data.cache import CachedSource, read_cached
from hedgefund.data.csv_source import CsvDirSource, load_csv
from hedgefund.data.yfinance_source import YFinanceSource, fetch_ticker
from hedgefund.jobs.daily import (
    DEFAULT_WEBHOOKS_DIR,
    LATEST_FILE,
    ORDERS_NOTE,
    read_signal,
    render_webhook_result,
    run_daily,
)
from hedgefund.jobs.manifest import (
    DEFAULT_JOBS_DIR,
    RunManifest,
    read_manifest,
    read_results,
    read_table,
    run_dir,
)
from hedgefund.jobs.scan import ScanResult, render_table, scan_json, scan_universe
from hedgefund.jobs.universe import load_universe
from hedgefund.regime.hmm import EMISSION_CHOICES
from hedgefund.regime.signal import POSITION_RULES
from hedgefund.report.html import write_run_html
from hedgefund.report.json_contract import analyze, backtest_only, to_json
from hedgefund.report.terminal import print_backtest, print_pretty

app = typer.Typer(
    help="Markov regime detection (Bull / Bear / Sideways) with honest, no-lookahead backtests.",
    no_args_is_help=True,
)

ROLLING_PREFIX = "rolling:"  # ``--refit rolling:N`` sets the rolling window to N bars
SEEDS_SEPARATOR = ","  # ``--hmm-seeds 0,1,2``
UNIVERSES_DIR = DEFAULT_CONFIG_PATH.parent / "universes"  # config/universes/<name>.yaml
SCAN_SOURCES = ("yfinance", "csv", "alpaca")  # ``--source`` choices for ``scan`` / ``daily``
# Defaults used when a config predates the Phase 7 keys (config over flags, but never crash).
DEFAULT_ALPACA_FEED = "iex"  # the free IEX feed; "sip" needs a paid data subscription
DEFAULT_ALPACA_ADJUSTMENT = "all"  # splits and dividends adjusted, matching yfinance auto_adjust
DEFAULT_PAPER_MAX_GROSS = 1.0  # gross exposure capped at 100% of equity: no leverage by default
DEFAULT_PAPER_ALLOW_SHORT = True  # the sign rule emits -1 in Bear; a long-only book sets False
DEFAULT_PAPER_MIN_TRADE_NOTIONAL = 1.0  # skip proposed trades below one dollar of notional
DEFAULT_DASHBOARD_PORT = 8501  # Streamlit's own default port
STREAMLIT_MISSING = "streamlit is not installed; run `uv sync --extra dashboard`"


@app.callback()
def root() -> None:
    """Keep ``app`` a command group even while ``version`` is the only command."""


@app.command()
def version() -> None:
    """Print the installed hedgefund version."""
    typer.echo(__version__)


def _config_value(config: dict, *keys: str) -> object:
    """Walk ``config[keys[0]][keys[1]]...`` and name the missing key on failure."""
    node: object = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"config is missing {'.'.join(keys)}")
        node = node[key]
    return node


def _load_close(
    ticker: str | None, csv: str | None, years: int, json_output: bool
) -> tuple[object, str]:
    """Resolve the price series from exactly one of ``--ticker`` / ``--csv``."""
    if (ticker is None) == (csv is None):
        raise ValueError("Provide exactly one of --ticker or --csv.")
    if ticker is not None:
        if not json_output:
            typer.echo(f"  fetching {ticker} from Yahoo Finance...", err=True)
        return fetch_ticker(ticker, years=years), ticker
    return load_csv(csv), csv


def _parse_refit(refit: str | None, rolling_bars: int | None, cfg: dict) -> tuple[str, int]:
    """Resolve ``(refit, rolling_bars)`` from ``--refit``, ``--rolling-bars`` and the config.

    ``--refit`` accepts ``expanding``, ``rolling`` or ``rolling:N`` (N bars). An explicit
    ``--rolling-bars`` wins over the ``:N`` suffix, which wins over ``backtest.rolling_bars``.
    """
    bars_from_refit: int | None = None
    if refit is None:
        mode = str(_config_value(cfg, "backtest", "refit"))
        if mode.startswith(ROLLING_PREFIX):  # config may also use the rolling:N spelling
            bars_from_refit = int(mode[len(ROLLING_PREFIX) :])
            mode = "rolling"
    elif refit.startswith(ROLLING_PREFIX):
        mode = "rolling"
        try:
            bars_from_refit = int(refit[len(ROLLING_PREFIX) :])
        except ValueError as exc:
            raise ValueError(
                f"--refit rolling:N needs an integer number of bars, got {refit!r}"
            ) from exc
    else:
        mode = refit
    if mode not in REFIT_MODES:
        raise ValueError(f"--refit must be one of {REFIT_MODES} or rolling:N, got {refit!r}")
    if rolling_bars is not None:
        bars = int(rolling_bars)
    elif bars_from_refit is not None:
        bars = bars_from_refit
    else:
        bars = int(_config_value(cfg, "backtest", "rolling_bars"))
    return mode, bars


def _default_cost_profile(cfg: dict) -> str:
    """The cost profile of the config's default ``universe`` (``config/universes/<name>.yaml``).

    Falls back to the universe name itself when no universe file exists next to the
    default config (a custom ``--config`` with a universe that is also a profile name).
    """
    universe = str(_config_value(cfg, "universe"))
    universe_file = UNIVERSES_DIR / f"{universe}.yaml"
    if universe_file.is_file():
        return str(_config_value(load_config(universe_file), "cost_profile"))
    return universe


def _resolve_cost_model(
    cfg: dict,
    cost_profile: str | None,
    fee_bps: float | None,
    slippage_bps: float | None,
    borrow_bps: float | None,
    funding_bps: float | None,
) -> tuple[CostModel, str]:
    """Cost model from ``--cost-profile`` (default: the universe's), each bps flag overriding."""
    profile = cost_profile if cost_profile is not None else _default_cost_profile(cfg)
    base = cost_model_from_config(cfg, profile)
    model = CostModel(
        fee_bps=base.fee_bps if fee_bps is None else float(fee_bps),
        slippage_bps=base.slippage_bps if slippage_bps is None else float(slippage_bps),
        borrow_bps=base.borrow_bps if borrow_bps is None else float(borrow_bps),
        funding_bps=base.funding_bps if funding_bps is None else float(funding_bps),
    )
    return model, profile


def _validate_rule(rule: str) -> str:
    if rule not in POSITION_RULES:
        raise ValueError(f"--rule must be one of {POSITION_RULES}, got {rule!r}")
    return rule


def _walk_forward_kwargs(
    cfg: dict,
    *,
    refit: str | None,
    rolling_bars: int | None,
    rule: str | None,
    dead_band: float | None,
) -> dict:
    """The Phase 3 keyword arguments shared by ``analyze`` and ``backtest_only``, cost model aside.

    The cost model is resolved separately: ``analyze`` and ``backtest`` take it from
    ``--cost-profile`` (default: the config universe's), ``scan`` from the universe file.
    """
    refit_used, rolling_used = _parse_refit(refit, rolling_bars, cfg)
    rule_used = _validate_rule(
        str(rule if rule is not None else _config_value(cfg, "backtest", "rule"))
    )
    dead_band_used = float(
        dead_band if dead_band is not None else _config_value(cfg, "backtest", "dead_band")
    )
    sensitivity = _config_value(cfg, "backtest", "sensitivity")
    return {
        "refit": refit_used,
        "rolling_bars": rolling_used,
        "rule": rule_used,
        "dead_band": dead_band_used,
        "horizons": [int(h) for h in _config_value(cfg, "backtest", "horizons")],
        "holdout_years": int(_config_value(cfg, "backtest", "holdout_years")),
        "sensitivity_windows": [int(w) for w in _config_value(sensitivity, "windows")],
        "sensitivity_thresholds": [float(t) for t in _config_value(sensitivity, "thresholds")],
        "cost_sensitivity_bps": [
            int(b) for b in _config_value(cfg, "backtest", "cost_sensitivity_bps")
        ],
        "lag_order": int(_config_value(cfg, "backtest", "lag_order")),
    }


def _backtest_kwargs(
    cfg: dict,
    *,
    refit: str | None,
    rolling_bars: int | None,
    rule: str | None,
    dead_band: float | None,
    cost_profile: str | None,
    fee_bps: float | None,
    slippage_bps: float | None,
    borrow_bps: float | None,
    funding_bps: float | None,
) -> dict:
    """The Phase 3 keyword arguments of ``backtest_only`` including the resolved cost model."""
    kwargs = _walk_forward_kwargs(
        cfg, refit=refit, rolling_bars=rolling_bars, rule=rule, dead_band=dead_band
    )
    model, _profile = _resolve_cost_model(
        cfg, cost_profile, fee_bps, slippage_bps, borrow_bps, funding_bps
    )
    kwargs["cost_model"] = model
    return kwargs


def _parse_seeds(seeds: str) -> tuple[int, ...]:
    """``"0,1,2"`` -> ``(0, 1, 2)``; name the bad token on failure."""
    tokens = [token.strip() for token in seeds.split(SEEDS_SEPARATOR) if token.strip()]
    if not tokens:
        raise ValueError(f"--hmm-seeds needs at least one integer, got {seeds!r}")
    try:
        return tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise ValueError(
            f"--hmm-seeds must be comma-separated integers such as 0,1,2, got {seeds!r}"
        ) from exc


def _validate_emission(emission: str) -> str:
    if emission not in EMISSION_CHOICES:
        raise ValueError(f"--hmm-emission must be one of {EMISSION_CHOICES}, got {emission!r}")
    return emission


def _hmm_kwargs(
    cfg: dict, *, seeds: str | None, refit_every: int | None, emission: str | None
) -> dict:
    """The Phase 4 ``hmm_*`` keyword arguments of ``analyze`` from the ``hmm`` config block.

    ``--hmm-seeds``, ``--hmm-refit-every`` and ``--hmm-emission`` override their config
    values; everything else comes from the config only. ``min_train`` is shared with the
    backtest (``backtest.min_train``) and is not read here.
    """
    seeds_used = (
        _parse_seeds(seeds)
        if seeds is not None
        else tuple(int(s) for s in _config_value(cfg, "hmm", "seeds"))
    )
    refit_used = int(
        refit_every if refit_every is not None else _config_value(cfg, "hmm", "refit_every")
    )
    emission_used = _validate_emission(
        str(emission if emission is not None else _config_value(cfg, "hmm", "emission"))
    )
    return {
        "hmm_seeds": seeds_used,
        "hmm_n_iter": int(_config_value(cfg, "hmm", "n_iter")),
        "hmm_emission": emission_used,
        "hmm_enter": float(_config_value(cfg, "hmm", "enter")),
        "hmm_exit": float(_config_value(cfg, "hmm", "exit")),
        "hmm_confirm_bars": int(_config_value(cfg, "hmm", "confirm_bars")),
        "hmm_refit_every": refit_used,
        "hmm_train_bars": int(_config_value(cfg, "hmm", "train_bars")),
        "hmm_vol_short": int(_config_value(cfg, "hmm", "vol_short")),
        "hmm_vol_long": int(_config_value(cfg, "hmm", "vol_long")),
        "hmm_match_window": int(_config_value(cfg, "hmm", "match_window")),
    }


def _analyze_kwargs(
    cfg: dict,
    *,
    window: int | None,
    threshold: float | None,
    min_train: int | None,
    no_hmm: bool,
    hmm_seeds: str | None,
    hmm_refit_every: int | None,
    hmm_emission: str | None,
    null_paths: int | None,
    bootstrap_resamples: int | None,
    seed: int | None,
    refit: str | None,
    rule: str | None,
    dead_band: float | None,
) -> dict:
    """Every keyword ``analyze`` takes except ``close``, ``source`` and ``cost_model``.

    Shared by ``analyze`` (one series, cost model from ``--cost-profile``) and ``scan``
    (one call per symbol, cost model from the universe file). Each flag overrides its
    config value; ``None`` means "use the config".
    """
    window_used = int(window if window is not None else _config_value(cfg, "labelling", "window"))
    threshold_used = float(
        threshold if threshold is not None else _config_value(cfg, "labelling", "threshold")
    )
    min_train_used = int(
        min_train if min_train is not None else _config_value(cfg, "backtest", "min_train")
    )
    return_type = str(_config_value(cfg, "labelling", "return_type"))
    null_paths_used = int(
        null_paths if null_paths is not None else _config_value(cfg, "statistics", "null_paths")
    )
    resamples_used = int(
        bootstrap_resamples
        if bootstrap_resamples is not None
        else _config_value(cfg, "statistics", "bootstrap", "resamples")
    )
    seed_used = int(seed if seed is not None else _config_value(cfg, "statistics", "seed"))
    statistics_kwargs = {
        "block_days": int(_config_value(cfg, "statistics", "bootstrap", "block_days")),
        "rolling_bars": int(_config_value(cfg, "statistics", "rolling_bars")),
        "ew_half_life": float(_config_value(cfg, "statistics", "ew_half_life")),
        "shrinkage_alpha": float(_config_value(cfg, "statistics", "shrinkage_alpha")),
        "mixing_epsilon": float(_config_value(cfg, "statistics", "mixing_epsilon")),
        "homogeneity_periods": int(_config_value(cfg, "statistics", "homogeneity_periods")),
    }
    backtest_kwargs = _walk_forward_kwargs(
        cfg, refit=refit, rolling_bars=None, rule=rule, dead_band=dead_band
    )
    backtest_kwargs["backtest_rolling_bars"] = backtest_kwargs.pop("rolling_bars")
    # With --no-hmm the hmm block is never run, so a config without it still works.
    hmm_kwargs = (
        {}
        if no_hmm
        else _hmm_kwargs(cfg, seeds=hmm_seeds, refit_every=hmm_refit_every, emission=hmm_emission)
    )
    return {
        "window": window_used,
        "threshold": threshold_used,
        "min_train": min_train_used,
        "hmm": not no_hmm,
        "return_type": return_type,
        "null_paths": null_paths_used,
        "bootstrap_resamples": resamples_used,
        "seed": seed_used,
        **statistics_kwargs,
        **backtest_kwargs,
        **hmm_kwargs,
    }


def _config_get(config: dict | None, *keys: str, default: object) -> object:
    """``_config_value`` that falls back to ``default`` when the key path is absent."""
    try:
        return _config_value(config if config is not None else {}, *keys)
    except ValueError:
        return default


def _build_scan_source(
    source_name: str,
    csv_dir: Path | None,
    cache_dir: Path,
    years: int,
    offline: bool,
    cfg: dict | None = None,
) -> CachedSource:
    """``--source`` wrapped in the parquet cache; ``--offline`` turns a cache miss into an error.

    ``cfg`` supplies ``data.alpaca.feed`` / ``data.alpaca.adjustment`` for the alpaca
    source; alpaca-py is imported lazily so the other sources work without it.
    """
    if source_name == "yfinance":
        inner: PriceSource = YFinanceSource(years=years)
    elif source_name == "csv":
        if csv_dir is None:
            raise ValueError("--source csv needs --csv-dir DIR, a directory of <SYMBOL>.csv files.")
        inner = CsvDirSource(csv_dir)
    elif source_name == "alpaca":
        from hedgefund.data.alpaca_source import AlpacaSource, read_credentials

        if not offline:
            # Fail fast with the variable name instead of one error row per symbol; a warm
            # cache under --offline needs no credentials at all.
            read_credentials()
        inner = AlpacaSource(
            years=years,
            feed=str(_config_get(cfg, "data", "alpaca", "feed", default=DEFAULT_ALPACA_FEED)),
            adjustment=str(
                _config_get(cfg, "data", "alpaca", "adjustment", default=DEFAULT_ALPACA_ADJUSTMENT)
            ),
        )
    else:
        raise ValueError(f"--source must be one of {SCAN_SOURCES}, got {source_name!r}")
    return CachedSource(inner, cache_dir, years=years, offline=offline)


def _closes_from_manifest(manifest: RunManifest, results: dict[str, dict]) -> dict:
    """Close series for every symbol with a result, read from the run's cached parquet files.

    A missing or unreadable cache file is a ``ValueError`` naming the symbol and path so
    the user knows to re-run ``hedgefund scan`` (the cache, not the run, is what expired).
    """
    closes = {}
    for symbol in results:
        entry = manifest.data.get(symbol, {})
        cache_path = entry.get("cache_path")
        if not cache_path:
            raise ValueError(
                f"run {manifest.run_id} records no cache_path for {symbol}; re-run hedgefund scan."
            )
        path = Path(cache_path)
        try:
            frame = read_cached(path)
        except Exception as exc:
            raise ValueError(
                f"cannot read cached prices for {symbol} at {path}: {exc}. "
                "Re-run hedgefund scan to refresh the cache."
            ) from exc
        closes[symbol] = close_series(frame, symbol)
    return closes


def _write_scan_html(scan: ScanResult) -> Path:
    """Render ``jobs/<run-id>/html/`` for a finished scan and return the index path."""
    closes = _closes_from_manifest(scan.manifest, scan.results)
    return write_run_html(scan.run_dir, scan.manifest.to_dict(), scan.results, closes, scan.table)


def _fail(exc: Exception, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps({"error": str(exc)}))
    else:
        typer.echo(f"\nERROR: {exc}\n", err=True)
    raise typer.Exit(code=1) from exc


# Options shared by the commands.
TickerOpt = Annotated[
    str | None, typer.Option("--ticker", help="Symbol to fetch via yfinance, e.g. SPY")
]
CsvOpt = Annotated[
    str | None, typer.Option("--csv", help="Path to your own CSV (date + close columns)")
]
YearsOpt = Annotated[
    int | None, typer.Option("--years", help="Years of history when using --ticker")
]
WindowOpt = Annotated[int | None, typer.Option("--window", help="Rolling-return window in days")]
ThresholdOpt = Annotated[
    float | None, typer.Option("--threshold", help="Regime label threshold on rolling return")
]
MinTrainOpt = Annotated[
    int | None, typer.Option("--min-train", help="Min training rows before the walk-forward starts")
]
RefitOpt = Annotated[
    str | None,
    typer.Option("--refit", help="Walk-forward refit: expanding, rolling, or rolling:N bars"),
]
RollingBarsOpt = Annotated[
    int | None, typer.Option("--rolling-bars", help="Rolling refit window in bars (rolling mode)")
]
RuleOpt = Annotated[
    str | None,
    typer.Option("--rule", help=f"Position rule: one of {', '.join(POSITION_RULES)}"),
]
DeadBandOpt = Annotated[
    float | None,
    typer.Option(
        "--dead-band",
        help=(
            "Rebalancing band; 0 = Phase 1 behaviour. sign/long_only: hysteresis on the "
            "signal around zero; conviction/persistence: only rebalance when |target - held| "
            "exceeds the band"
        ),
    ),
]
CostProfileOpt = Annotated[
    str | None,
    typer.Option("--cost-profile", help="Name under costs.profiles (default: the universe's)"),
]
FeeOpt = Annotated[float | None, typer.Option("--fee-bps", help="Per-side fee, overrides profile")]
SlippageOpt = Annotated[
    float | None, typer.Option("--slippage-bps", help="Per-side slippage, overrides profile")
]
BorrowOpt = Annotated[
    float | None, typer.Option("--borrow-bps", help="Annual short borrow bps, overrides profile")
]
FundingOpt = Annotated[
    float | None, typer.Option("--funding-bps", help="Annual funding bps, overrides profile")
]
NullPathsOpt = Annotated[
    int | None,
    typer.Option("--null-paths", help="Null-model paths (override config for fast runs)"),
]
BootstrapResamplesOpt = Annotated[
    int | None,
    typer.Option(
        "--bootstrap-resamples", help="Block-bootstrap resamples (override config for fast runs)"
    ),
]
SeedOpt = Annotated[
    int | None, typer.Option("--seed", help="RNG seed for null models and the bootstrap")
]
NoHmmOpt = Annotated[
    bool, typer.Option("--no-hmm", help="Skip the HMM fit even if hmmlearn is available")
]
HmmSeedsOpt = Annotated[
    str | None,
    typer.Option(
        "--hmm-seeds", help="Comma-separated Baum-Welch seeds, e.g. 0,1,2 (overrides config)"
    ),
]
HmmRefitEveryOpt = Annotated[
    int | None, typer.Option("--hmm-refit-every", help="Bars between walk-forward HMM refits")
]
HmmEmissionOpt = Annotated[
    str | None,
    typer.Option("--hmm-emission", help=f"HMM emission: one of {', '.join(EMISSION_CHOICES)}"),
]
JsonOpt = Annotated[
    bool, typer.Option("--json", help="Emit the result dict as JSON to stdout, nothing else")
]
ConfigOpt = Annotated[
    Path | None, typer.Option("--config", help="YAML config overriding config/default.yaml")
]
JobsDirOpt = Annotated[
    Path | None,
    typer.Option("--jobs-dir", help="Where runs are written (default: jobs/ at the repo root)"),
]


@app.command(name="analyze")
def analyze_command(
    ticker: TickerOpt = None,
    csv: CsvOpt = None,
    years: YearsOpt = None,
    window: WindowOpt = None,
    threshold: ThresholdOpt = None,
    min_train: MinTrainOpt = None,
    no_hmm: NoHmmOpt = False,
    hmm_seeds: HmmSeedsOpt = None,
    hmm_refit_every: HmmRefitEveryOpt = None,
    hmm_emission: HmmEmissionOpt = None,
    null_paths: NullPathsOpt = None,
    bootstrap_resamples: BootstrapResamplesOpt = None,
    seed: SeedOpt = None,
    refit: RefitOpt = None,
    rule: RuleOpt = None,
    dead_band: DeadBandOpt = None,
    cost_profile: CostProfileOpt = None,
    json_output: JsonOpt = False,
    config: ConfigOpt = None,
) -> None:
    """Run the Markov regime analysis on a ticker or a CSV."""
    try:
        cfg = load_config(config)
        years_used = int(years if years is not None else _config_value(cfg, "data", "years"))
        kwargs = _analyze_kwargs(
            cfg,
            window=window,
            threshold=threshold,
            min_train=min_train,
            no_hmm=no_hmm,
            hmm_seeds=hmm_seeds,
            hmm_refit_every=hmm_refit_every,
            hmm_emission=hmm_emission,
            null_paths=null_paths,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
            refit=refit,
            rule=rule,
            dead_band=dead_band,
        )
        cost_model, _profile = _resolve_cost_model(cfg, cost_profile, None, None, None, None)

        close, source = _load_close(ticker, csv, years_used, json_output)
        result = analyze(close, source=source, cost_model=cost_model, **kwargs)
    except Exception as exc:
        _fail(exc, json_output)

    if json_output:
        typer.echo(to_json(result))
    else:
        print_pretty(result)


@app.command(name="backtest")
def backtest_command(
    ticker: TickerOpt = None,
    csv: CsvOpt = None,
    years: YearsOpt = None,
    window: WindowOpt = None,
    threshold: ThresholdOpt = None,
    min_train: MinTrainOpt = None,
    refit: RefitOpt = None,
    rolling_bars: RollingBarsOpt = None,
    rule: RuleOpt = None,
    dead_band: DeadBandOpt = None,
    cost_profile: CostProfileOpt = None,
    fee_bps: FeeOpt = None,
    slippage_bps: SlippageOpt = None,
    borrow_bps: BorrowOpt = None,
    funding_bps: FundingOpt = None,
    null_paths: NullPathsOpt = None,
    seed: SeedOpt = None,
    json_output: JsonOpt = False,
    config: ConfigOpt = None,
) -> None:
    """Run the cost-aware walk-forward backtest with benchmarks, nulls, holdout and lag."""
    try:
        cfg = load_config(config)
        years_used = int(years if years is not None else _config_value(cfg, "data", "years"))
        window_used = int(
            window if window is not None else _config_value(cfg, "labelling", "window")
        )
        threshold_used = float(
            threshold if threshold is not None else _config_value(cfg, "labelling", "threshold")
        )
        min_train_used = int(
            min_train if min_train is not None else _config_value(cfg, "backtest", "min_train")
        )
        return_type = str(_config_value(cfg, "labelling", "return_type"))
        null_paths_used = int(
            null_paths if null_paths is not None else _config_value(cfg, "statistics", "null_paths")
        )
        seed_used = int(seed if seed is not None else _config_value(cfg, "statistics", "seed"))
        backtest_kwargs = _backtest_kwargs(
            cfg,
            refit=refit,
            rolling_bars=rolling_bars,
            rule=rule,
            dead_band=dead_band,
            cost_profile=cost_profile,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            borrow_bps=borrow_bps,
            funding_bps=funding_bps,
        )

        close, source = _load_close(ticker, csv, years_used, json_output)
        result = backtest_only(
            close,
            source=source,
            window=window_used,
            threshold=threshold_used,
            min_train=min_train_used,
            return_type=return_type,
            null_paths=null_paths_used,
            seed=seed_used,
            **backtest_kwargs,
        )
    except Exception as exc:
        _fail(exc, json_output)

    if json_output:
        typer.echo(to_json(result))
    else:
        print_backtest(result)


@app.command(name="scan")
def scan_command(
    universe: Annotated[
        str | None,
        typer.Option(
            "--universe",
            help="Universe name under config/universes or a YAML path (default: config universe)",
        ),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option("--source", help=f"Price source, one of {', '.join(SCAN_SOURCES)}"),
    ] = None,
    csv_dir: Annotated[
        Path | None,
        typer.Option("--csv-dir", help="Directory of <SYMBOL>.csv files for --source csv"),
    ] = None,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Parquet cache dir (default: config data.cache_dir)"),
    ] = None,
    jobs_dir: JobsDirOpt = None,
    offline: Annotated[
        bool,
        typer.Option("--offline", help="Serve every symbol from the cache; a miss is an error"),
    ] = False,
    years: YearsOpt = None,
    window: WindowOpt = None,
    threshold: ThresholdOpt = None,
    min_train: MinTrainOpt = None,
    no_hmm: NoHmmOpt = False,
    hmm_seeds: HmmSeedsOpt = None,
    hmm_refit_every: HmmRefitEveryOpt = None,
    hmm_emission: HmmEmissionOpt = None,
    null_paths: NullPathsOpt = None,
    bootstrap_resamples: BootstrapResamplesOpt = None,
    seed: SeedOpt = None,
    refit: RefitOpt = None,
    rule: RuleOpt = None,
    dead_band: DeadBandOpt = None,
    html: Annotated[
        bool, typer.Option("--html/--no-html", help="Render the HTML report under the run")
    ] = True,
    json_output: JsonOpt = False,
    config: ConfigOpt = None,
) -> None:
    """Run the full analysis per symbol of a universe and rank the results.

    Prices go through the parquet cache, so a second run with --offline needs no network.
    Each run writes jobs/<run-id>/manifest.json, results/<symbol>.json, table.json and,
    unless --no-html, html/index.html. The cost profile comes from the universe file.
    """
    try:
        cfg = load_config(config)
        universe_loaded = load_universe(
            universe if universe is not None else str(_config_value(cfg, "universe"))
        )
        years_used = int(years if years is not None else _config_value(cfg, "data", "years"))
        source_name = str(source if source is not None else _config_value(cfg, "data", "source"))
        if cache_dir is None:
            cache_dir = Path(str(_config_value(cfg, "data", "cache_dir")))
        jobs_dir_used = jobs_dir if jobs_dir is not None else DEFAULT_JOBS_DIR
        kwargs = _analyze_kwargs(
            cfg,
            window=window,
            threshold=threshold,
            min_train=min_train,
            no_hmm=no_hmm,
            hmm_seeds=hmm_seeds,
            hmm_refit_every=hmm_refit_every,
            hmm_emission=hmm_emission,
            null_paths=null_paths,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
            refit=refit,
            rule=rule,
            dead_band=dead_band,
        )
        price_source = _build_scan_source(source_name, csv_dir, cache_dir, years_used, offline, cfg)
        progress = None if json_output else (lambda message: typer.echo(message, err=True))
        scan = scan_universe(
            universe_loaded,
            price_source,
            cfg,
            analyze_kwargs=kwargs,
            jobs_dir=jobs_dir_used,
            progress=progress,
        )
        index_path = _write_scan_html(scan) if html else None
    except Exception as exc:
        _fail(exc, json_output)

    if json_output:
        payload = scan_json(scan)
        payload["html"] = None if index_path is None else str(index_path)
        typer.echo(json.dumps(payload))
    else:
        typer.echo(render_table(scan.table))
        typer.echo(f"run id: {scan.run_id}")
        typer.echo(f"run dir: {scan.run_dir}")
        if index_path is not None:
            typer.echo(f"html: {index_path}")


@app.command(name="daily")
def daily_command(
    universe: Annotated[
        str | None,
        typer.Option(
            "--universe",
            help="Universe name under config/universes or a YAML path (default: config universe)",
        ),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option("--source", help=f"Price source, one of {', '.join(SCAN_SOURCES)}"),
    ] = None,
    csv_dir: Annotated[
        Path | None,
        typer.Option("--csv-dir", help="Directory of <SYMBOL>.csv files for --source csv"),
    ] = None,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Parquet cache dir (default: config data.cache_dir)"),
    ] = None,
    jobs_dir: JobsDirOpt = None,
    webhooks_dir: Annotated[
        Path | None,
        typer.Option(
            "--webhooks-dir",
            help="Directory of webhook registrations (default: config daily.webhooks_dir)",
        ),
    ] = None,
    no_webhooks: Annotated[
        bool, typer.Option("--no-webhooks", help="Write the signal file but post to nobody")
    ] = False,
    offline: Annotated[
        bool,
        typer.Option("--offline", help="Serve every symbol from the cache; a miss is an error"),
    ] = False,
    years: YearsOpt = None,
    window: WindowOpt = None,
    threshold: ThresholdOpt = None,
    min_train: MinTrainOpt = None,
    no_hmm: NoHmmOpt = False,
    hmm_seeds: HmmSeedsOpt = None,
    hmm_refit_every: HmmRefitEveryOpt = None,
    hmm_emission: HmmEmissionOpt = None,
    null_paths: NullPathsOpt = None,
    bootstrap_resamples: BootstrapResamplesOpt = None,
    seed: SeedOpt = None,
    refit: RefitOpt = None,
    rule: RuleOpt = None,
    dead_band: DeadBandOpt = None,
    json_output: JsonOpt = False,
    config: ConfigOpt = None,
) -> None:
    """Rescan a watchlist, write the signal file and post a summary to the webhooks.

    Writes jobs/<run-id>/signal.json and jobs/signal_latest.json, then POSTs a compact
    summary to every consumer registered under webhooks/ (see webhooks/README.md). The
    signal file is a signal, not an order: nothing here submits orders.
    """
    try:
        cfg = load_config(config)
        universe_loaded = load_universe(
            universe if universe is not None else str(_config_value(cfg, "universe"))
        )
        years_used = int(years if years is not None else _config_value(cfg, "data", "years"))
        source_name = str(source if source is not None else _config_value(cfg, "data", "source"))
        if cache_dir is None:
            cache_dir = Path(str(_config_value(cfg, "data", "cache_dir")))
        jobs_dir_used = jobs_dir if jobs_dir is not None else DEFAULT_JOBS_DIR
        if no_webhooks:
            webhooks_used: Path | None = None
        elif webhooks_dir is not None:
            webhooks_used = webhooks_dir
        else:
            webhooks_used = Path(
                str(_config_get(cfg, "daily", "webhooks_dir", default=DEFAULT_WEBHOOKS_DIR))
            )
        kwargs = _analyze_kwargs(
            cfg,
            window=window,
            threshold=threshold,
            min_train=min_train,
            no_hmm=no_hmm,
            hmm_seeds=hmm_seeds,
            hmm_refit_every=hmm_refit_every,
            hmm_emission=hmm_emission,
            null_paths=null_paths,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
            refit=refit,
            rule=rule,
            dead_band=dead_band,
        )
        price_source = _build_scan_source(source_name, csv_dir, cache_dir, years_used, offline, cfg)
        progress = None if json_output else (lambda message: typer.echo(message, err=True))
        job = run_daily(
            universe_loaded,
            price_source,
            cfg,
            analyze_kwargs=kwargs,
            jobs_dir=jobs_dir_used,
            webhooks_dir=webhooks_used,
            progress=progress,
            rule=str(kwargs["rule"]),
        )
    except Exception as exc:
        _fail(exc, json_output)

    if json_output:
        payload = dict(job.signal)
        payload["webhooks"] = [result.to_dict() for result in job.webhook_results]
        typer.echo(json.dumps(payload))
    else:
        typer.echo(render_table(job.scan.table))
        typer.echo(f"signal: {job.signal_path}")
        typer.echo(f"latest: {job.latest_path}")
        for result in job.webhook_results:
            typer.echo(render_webhook_result(result))
        typer.echo(ORDERS_NOTE)


@app.command(name="propose")
def propose_command(
    signal: Annotated[
        Path | None,
        typer.Option("--signal", help="Signal file (default: <jobs-dir>/signal_latest.json)"),
    ] = None,
    jobs_dir: JobsDirOpt = None,
    equity: Annotated[
        float | None, typer.Option("--equity", help="Account equity to size against, in dollars")
    ] = None,
    paper_equity: Annotated[
        bool,
        typer.Option("--paper-equity", help="Read equity from the Alpaca paper account"),
    ] = False,
    paper_positions: Annotated[
        bool,
        typer.Option(
            "--paper-positions",
            help="Read current positions from the Alpaca paper account (default: assume flat)",
        ),
    ] = False,
    rule: RuleOpt = None,
    max_gross: Annotated[
        float | None,
        typer.Option("--max-gross", help="Gross exposure cap as a fraction of equity"),
    ] = None,
    allow_short: Annotated[
        bool | None,
        typer.Option("--allow-short/--no-short", help="Allow negative target positions"),
    ] = None,
    json_output: JsonOpt = False,
    config: ConfigOpt = None,
) -> None:
    """Turn the latest signal file into a paper allocation and print the proposed orders.

    Nothing is submitted. The output is a proposal for a human to review; placing an
    order is a separate, explicit action outside this tool. Prices are the signal file's
    last closes. Exactly one of --equity / --paper-equity is required.
    """
    try:
        cfg = load_config(config)
        if (equity is None) == (not paper_equity):
            raise ValueError("Provide exactly one of --equity AMOUNT or --paper-equity.")
        from hedgefund.paper import sizing  # lazy: alpaca-py is optional

        jobs_dir_used = jobs_dir if jobs_dir is not None else DEFAULT_JOBS_DIR
        signal_path = signal if signal is not None else jobs_dir_used / LATEST_FILE
        signal_data = read_signal(signal_path)
        rule_used = _validate_rule(
            str(
                rule
                if rule is not None
                else signal_data.get("rule", _config_value(cfg, "backtest", "rule"))
            )
        )
        max_gross_used = float(
            max_gross
            if max_gross is not None
            else _config_get(cfg, "paper", "max_gross", default=DEFAULT_PAPER_MAX_GROSS)
        )
        allow_short_used = bool(
            allow_short
            if allow_short is not None
            else _config_get(cfg, "paper", "allow_short", default=DEFAULT_PAPER_ALLOW_SHORT)
        )
        min_notional = float(
            _config_get(
                cfg, "paper", "min_trade_notional", default=DEFAULT_PAPER_MIN_TRADE_NOTIONAL
            )
        )
        prices = {
            symbol: float(entry["last_close"])
            for symbol, entry in signal_data["symbols"].items()
            if entry.get("last_close") is not None
        }
        equity_used = float(equity) if equity is not None else float(sizing.paper_account_equity())
        positions = sizing.paper_positions() if paper_positions else None
        proposal = sizing.propose_orders(
            signal_data,
            equity=equity_used,
            prices=prices,
            current_positions=positions,
            rule=rule_used,
            max_gross=max_gross_used,
            allow_short=allow_short_used,
            min_trade_notional=min_notional,
        )
    except Exception as exc:
        _fail(exc, json_output)

    if json_output:
        typer.echo(json.dumps(proposal.to_dict()))
    else:
        typer.echo(sizing.render_proposal(proposal))


@app.command(name="report")
def report_command(
    run_id: Annotated[str, typer.Argument(help="Run id, the directory name under jobs/")],
    jobs_dir: JobsDirOpt = None,
    json_output: JsonOpt = False,
) -> None:
    """Regenerate the HTML report of a finished scan from its cached prices and results."""
    try:
        jobs_dir_used = jobs_dir if jobs_dir is not None else DEFAULT_JOBS_DIR
        manifest = read_manifest(run_id, jobs_dir_used)
        results = read_results(run_id, jobs_dir_used)
        table = read_table(run_id, jobs_dir_used)
        closes = _closes_from_manifest(manifest, results)
        index_path = write_run_html(
            run_dir(jobs_dir_used, run_id), manifest.to_dict(), results, closes, table
        )
    except Exception as exc:
        _fail(exc, json_output)

    if json_output:
        typer.echo(json.dumps({"run_id": run_id, "html": str(index_path)}))
    else:
        typer.echo(str(index_path))


def dashboard_command_line(
    jobs_dir: Path | None, port: int, headless: bool
) -> tuple[list[str], dict[str, str]]:
    """Build the ``streamlit run`` argv and child environment for ``hedgefund dashboard``.

    Pure so the launch is testable without starting a server. ``--jobs-dir`` travels through
    the ``HEDGEFUND_JOBS_DIR`` environment variable because Streamlit swallows script argv;
    when it is not given the child inherits the parent's environment unchanged and the app
    falls back to its own default (``jobs/`` at the repo root).
    """
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",  # no telemetry from a research tool
        "false",
        # Bind loopback only. With an explicit address Streamlit prints a single URL and
        # skips both of its startup lookups: the UDP probe to 8.8.8.8 for the internal IP
        # and (in headless mode) the HTTP GET to checkip.amazonaws.com for the external IP.
        "--server.address",
        "localhost",
        # A first run without ~/.streamlit/credentials.toml would otherwise block on an
        # interactive email prompt.
        "--server.showEmailPrompt",
        "false",
    ]
    if headless:
        cmd += ["--server.headless", "true"]  # do not open a browser (servers, CI)
    env = dict(os.environ)
    if jobs_dir is not None:
        env[JOBS_DIR_ENV] = str(jobs_dir.resolve())
    return cmd, env


@app.command(name="dashboard")
def dashboard_command(
    jobs_dir: JobsDirOpt = None,
    port: Annotated[
        int, typer.Option("--port", help="Local port for the Streamlit server")
    ] = DEFAULT_DASHBOARD_PORT,
    headless: Annotated[
        bool, typer.Option("--headless", help="Do not open a browser (servers, CI)")
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Dry run: print the command and environment as one JSON object, do not launch",
        ),
    ] = False,
) -> None:
    """Launch the local Streamlit dashboard over cached scan runs (no network calls).

    Needs the optional extra: ``uv sync --extra dashboard``.

    ``--json`` is a dry run: it prints the command and exits without starting a server.
    """
    try:
        if importlib.util.find_spec("streamlit") is None:
            raise RuntimeError(STREAMLIT_MISSING)
        if not APP_PATH.is_file():
            raise FileNotFoundError(f"dashboard app not found at {APP_PATH}")
        cmd, env = dashboard_command_line(jobs_dir, port, headless)
    except Exception as exc:
        _fail(exc, json_output)

    jobs_dir_used = env.get(JOBS_DIR_ENV, str(DEFAULT_JOBS_DIR))
    if json_output:
        typer.echo(json.dumps({"command": cmd, "jobs_dir": jobs_dir_used, "port": port}))
        return
    typer.echo(f"Launching dashboard at http://localhost:{port} (jobs dir: {jobs_dir_used})")
    completed = subprocess.run(cmd, env=env, check=False)
    raise typer.Exit(code=completed.returncode)


def main() -> None:
    """Console-script entry point (``hedgefund = "hedgefund.cli:main"``)."""
    app()
