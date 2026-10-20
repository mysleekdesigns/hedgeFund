"""Command-line entry point for hedgefund.

Phase 0 ships only ``version``. ``analyze``, ``backtest``, ``scan`` and ``report``
arrive in later phases.
"""

import typer

from hedgefund import __version__

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


def main() -> None:
    """Console-script entry point (``hedgefund = "hedgefund.cli:main"``)."""
    app()
