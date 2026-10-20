"""Command-line entry point for hedgefund.

``hedgefund analyze --ticker SYM | --csv PATH [--json]`` mirrors the companion script's
CLI. ``backtest``, ``scan`` and ``report`` arrive in later phases.
"""

import json
from pathlib import Path
from typing import Annotated

import typer

from hedgefund import __version__
from hedgefund.config import load_config
from hedgefund.data.csv_source import load_csv
from hedgefund.data.yfinance_source import fetch_ticker
from hedgefund.report.json_contract import analyze, to_json
from hedgefund.report.terminal import print_pretty

app = typer.Typer(
    help="Markov regime detection (Bull / Bear / Sideways) with honest, no-lookahead backtests.",
    no_args_is_help=True,
)


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


@app.command(name="analyze")
def analyze_command(
    ticker: Annotated[
        str | None, typer.Option("--ticker", help="Symbol to fetch via yfinance, e.g. SPY")
    ] = None,
    csv: Annotated[
        str | None, typer.Option("--csv", help="Path to your own CSV (date + close columns)")
    ] = None,
    years: Annotated[
        int | None, typer.Option("--years", help="Years of history when using --ticker")
    ] = None,
    window: Annotated[
        int | None, typer.Option("--window", help="Rolling-return window in days")
    ] = None,
    threshold: Annotated[
        float | None, typer.Option("--threshold", help="Regime label threshold on rolling return")
    ] = None,
    min_train: Annotated[
        int | None,
        typer.Option("--min-train", help="Min training rows before the walk-forward starts"),
    ] = None,
    no_hmm: Annotated[
        bool, typer.Option("--no-hmm", help="Skip the HMM fit even if hmmlearn is available")
    ] = False,
    null_paths: Annotated[
        int | None,
        typer.Option("--null-paths", help="Null-model paths (override config for fast runs)"),
    ] = None,
    bootstrap_resamples: Annotated[
        int | None,
        typer.Option(
            "--bootstrap-resamples",
            help="Block-bootstrap resamples (override config for fast runs)",
        ),
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", help="RNG seed for null models and the bootstrap")
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit the analyze() dict as JSON to stdout, nothing else")
    ] = False,
    config: Annotated[
        Path | None, typer.Option("--config", help="YAML config overriding config/default.yaml")
    ] = None,
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
            block_days=int(_config_value(cfg, "statistics", "bootstrap", "block_days")),
            rolling_bars=int(_config_value(cfg, "statistics", "rolling_bars")),
            ew_half_life=float(_config_value(cfg, "statistics", "ew_half_life")),
            shrinkage_alpha=float(_config_value(cfg, "statistics", "shrinkage_alpha")),
            mixing_epsilon=float(_config_value(cfg, "statistics", "mixing_epsilon")),
            homogeneity_periods=int(_config_value(cfg, "statistics", "homogeneity_periods")),
            seed=seed_used,
        )
    except Exception as exc:
        if json_output:
            typer.echo(json.dumps({"error": str(exc)}))
        else:
            typer.echo(f"\nERROR: {exc}\n", err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(to_json(result))
    else:
        print_pretty(result)


def main() -> None:
    """Console-script entry point (``hedgefund = "hedgefund.cli:main"``)."""
    app()
