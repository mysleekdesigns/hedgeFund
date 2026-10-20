"""Command-line entry point for hedgefund.

``hedgefund analyze --ticker SYM | --csv PATH [--json]`` mirrors the companion script's
CLI and appends the honest statistics and the ``backtest`` block. ``hedgefund backtest``
runs the cost-aware walk-forward alone (plan.md Phase 3) with cost and rule flags.
``scan`` and ``report`` arrive in later phases. Config over flags: every default comes
from ``config/default.yaml``; a flag overrides it.
"""

import json
from pathlib import Path
from typing import Annotated

import typer

from hedgefund import __version__
from hedgefund.backtest.costs import CostModel, cost_model_from_config
from hedgefund.backtest.walk_forward import REFIT_MODES
from hedgefund.config import DEFAULT_CONFIG_PATH, load_config
from hedgefund.data.csv_source import load_csv
from hedgefund.data.yfinance_source import fetch_ticker
from hedgefund.regime.signal import POSITION_RULES
from hedgefund.report.json_contract import analyze, backtest_only, to_json
from hedgefund.report.terminal import print_backtest, print_pretty

app = typer.Typer(
    help="Markov regime detection (Bull / Bear / Sideways) with honest, no-lookahead backtests.",
    no_args_is_help=True,
)

ROLLING_PREFIX = "rolling:"  # ``--refit rolling:N`` sets the rolling window to N bars
UNIVERSES_DIR = DEFAULT_CONFIG_PATH.parent / "universes"  # config/universes/<name>.yaml


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
    """The Phase 3 keyword arguments shared by ``analyze`` and ``backtest_only``."""
    refit_used, rolling_used = _parse_refit(refit, rolling_bars, cfg)
    rule_used = _validate_rule(
        str(rule if rule is not None else _config_value(cfg, "backtest", "rule"))
    )
    dead_band_used = float(
        dead_band if dead_band is not None else _config_value(cfg, "backtest", "dead_band")
    )
    model, _profile = _resolve_cost_model(
        cfg, cost_profile, fee_bps, slippage_bps, borrow_bps, funding_bps
    )
    sensitivity = _config_value(cfg, "backtest", "sensitivity")
    return {
        "refit": refit_used,
        "rolling_bars": rolling_used,
        "rule": rule_used,
        "dead_band": dead_band_used,
        "cost_model": model,
        "horizons": [int(h) for h in _config_value(cfg, "backtest", "horizons")],
        "holdout_years": int(_config_value(cfg, "backtest", "holdout_years")),
        "sensitivity_windows": [int(w) for w in _config_value(sensitivity, "windows")],
        "sensitivity_thresholds": [float(t) for t in _config_value(sensitivity, "thresholds")],
        "cost_sensitivity_bps": [
            int(b) for b in _config_value(cfg, "backtest", "cost_sensitivity_bps")
        ],
        "lag_order": int(_config_value(cfg, "backtest", "lag_order")),
    }


def _fail(exc: Exception, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps({"error": str(exc)}))
    else:
        typer.echo(f"\nERROR: {exc}\n", err=True)
    raise typer.Exit(code=1) from exc


# Options shared by both commands.
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
SeedOpt = Annotated[
    int | None, typer.Option("--seed", help="RNG seed for null models and the bootstrap")
]
JsonOpt = Annotated[
    bool, typer.Option("--json", help="Emit the result dict as JSON to stdout, nothing else")
]
ConfigOpt = Annotated[
    Path | None, typer.Option("--config", help="YAML config overriding config/default.yaml")
]


@app.command(name="analyze")
def analyze_command(
    ticker: TickerOpt = None,
    csv: CsvOpt = None,
    years: YearsOpt = None,
    window: WindowOpt = None,
    threshold: ThresholdOpt = None,
    min_train: MinTrainOpt = None,
    no_hmm: Annotated[
        bool, typer.Option("--no-hmm", help="Skip the HMM fit even if hmmlearn is available")
    ] = False,
    null_paths: NullPathsOpt = None,
    bootstrap_resamples: Annotated[
        int | None,
        typer.Option(
            "--bootstrap-resamples",
            help="Block-bootstrap resamples (override config for fast runs)",
        ),
    ] = None,
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
        backtest_kwargs = _backtest_kwargs(
            cfg,
            refit=refit,
            rolling_bars=None,
            rule=rule,
            dead_band=dead_band,
            cost_profile=cost_profile,
            fee_bps=None,
            slippage_bps=None,
            borrow_bps=None,
            funding_bps=None,
        )
        backtest_kwargs["backtest_rolling_bars"] = backtest_kwargs.pop("rolling_bars")

        close, source = _load_close(ticker, csv, years_used, json_output)
        result = analyze(
            close,
            source=source,
            window=window_used,
            threshold=threshold_used,
            min_train=min_train_used,
            hmm=not no_hmm,
            return_type=return_type,
            null_paths=null_paths_used,
            bootstrap_resamples=resamples_used,
            seed=seed_used,
            **statistics_kwargs,
            **backtest_kwargs,
        )
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


def main() -> None:
    """Console-script entry point (``hedgefund = "hedgefund.cli:main"``)."""
    app()
