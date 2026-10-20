"""Phase 0 scaffold tests: the package imports and the console script resolves."""

import importlib

import pytest
from typer.testing import CliRunner

import hedgefund
from hedgefund.cli import app

EXPECTED_VERSION = "0.1.0"
SUBPACKAGES = ["data", "regime", "backtest", "report", "jobs"]


def test_package_exposes_version() -> None:
    assert hedgefund.__version__ == EXPECTED_VERSION


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    module = importlib.import_module(f"hedgefund.{name}")
    assert module.__name__ == f"hedgefund.{name}"


def test_cli_version_command_prints_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == EXPECTED_VERSION
