"""Phase 1 reproduction gate: ``analyze()`` on the frozen SPY fixture matches the
companion script's frozen ``--json`` output to 6 decimals, both as a library call and
through the ``hedgefund analyze --csv ... --no-hmm --json`` CLI.

The tolerance is 5e-7 (6 decimals, ``.claude/rules/testing.md``); do not loosen it.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from hedgefund.cli import app
from hedgefund.report.json_contract import analyze

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SOURCE_NAME = "spy_10y.csv"
ABS_TOL = 5e-7  # 6 decimals

FLOAT_FIELDS = [
    "transition_matrix",
    "stationary_distribution",
    "persistence_diagonal",
    "next_state_probabilities",
    "signal",
]
EXACT_FIELDS = [
    "rows",
    "date_start",
    "date_end",
    "current_regime",
    "states",
    "framework",
    "disclaimer",
    "hmm",
]
EXACT_PARAMS = ["window", "threshold", "min_train"]


@pytest.fixture(scope="module")
def close() -> pd.Series:
    return pd.read_parquet(FIXTURES / "spy_10y.parquet")["close"]


@pytest.fixture(scope="module")
def expected() -> dict:
    with (FIXTURES / "spy_10y.companion.json").open() as fh:
        return json.load(fh)["result"]


def _assert_close(ours: object, theirs: object, path: str) -> None:
    """Recursive ``abs`` comparison that reports the offending path and numbers."""
    if isinstance(theirs, dict):
        assert set(ours) == set(theirs), f"{path}: keys {set(ours)} != {set(theirs)}"
        for key in theirs:
            _assert_close(ours[key], theirs[key], f"{path}.{key}")
    elif isinstance(theirs, list):
        assert len(ours) == len(theirs), f"{path}: length {len(ours)} != {len(theirs)}"
        for i, (a, b) in enumerate(zip(ours, theirs, strict=True)):
            _assert_close(a, b, f"{path}[{i}]")
    else:
        assert isinstance(theirs, int | float), f"{path}: unexpected type {type(theirs)}"
        assert math.isclose(ours, theirs, abs_tol=ABS_TOL, rel_tol=0.0), (
            f"{path}: ours={ours!r} companion={theirs!r} diff={ours - theirs:.3e}"
        )


def _assert_reproduces(result: dict, expected: dict) -> None:
    for field in FLOAT_FIELDS:
        _assert_close(result[field], expected[field], field)
    for key in ("sharpe", "max_drawdown"):
        _assert_close(result["walk_forward"][key], expected["walk_forward"][key], key)
    assert result["walk_forward"]["n_trades"] == expected["walk_forward"]["n_trades"]
    for field in EXACT_FIELDS:
        assert result[field] == expected[field], field
    for key in EXACT_PARAMS:
        assert result["params"][key] == expected["params"][key], key


def test_analyze_reproduces_companion_on_frozen_spy_fixture(
    close: pd.Series, expected: dict
) -> None:
    result = analyze(close, source=SOURCE_NAME, hmm=False)
    assert result["source"] == SOURCE_NAME
    _assert_reproduces(result, expected)


def test_cli_json_reproduces_companion_on_frozen_spy_fixture(
    close: pd.Series, expected: dict, tmp_path: Path
) -> None:
    path = tmp_path / SOURCE_NAME
    close.rename("close").rename_axis("date").to_csv(path)

    result = CliRunner().invoke(app, ["analyze", "--csv", str(path), "--no-hmm", "--json"])
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)

    assert parsed["source"] == str(path)
    _assert_reproduces(parsed, expected)
