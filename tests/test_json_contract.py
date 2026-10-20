"""The ``analyze()`` dict is exactly the companion skill's contract (plus additive fields)."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from hedgefund.cli import app
from hedgefund.report.json_contract import (
    COMPANION_FIELDS,
    DISCLAIMER,
    FRAMEWORK,
    analyze,
    to_json,
)

SEED = 7
LONG_ROWS = 600  # enough labels for the walk-forward (252 + 30) to produce numbers
SHORT_ROWS = 100  # fewer labels than min_train + 30 -> NaN Sharpe / drawdown
DAILY_VOL = 0.01
START_PRICE = 100.0

PROB_KEYS = ["bear", "sideways", "bull"]
COMPANION_PARAMS = ["window", "threshold", "min_train"]


def _synthetic_close(rows: int, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, DAILY_VOL, rows)
    index = pd.bdate_range("2020-01-01", periods=rows, name="date")
    return pd.Series(START_PRICE * np.cumprod(1.0 + returns), index=index, name="close")


@pytest.fixture(scope="module")
def result() -> dict:
    return analyze(_synthetic_close(LONG_ROWS), source="synthetic", hmm=False)


def test_top_level_keys_match_companion_fields_in_order(result: dict) -> None:
    assert list(result.keys()) == COMPANION_FIELDS


def test_nested_key_names_match_companion(result: dict) -> None:
    assert list(result["params"].keys()) == [*COMPANION_PARAMS, "return_type"]
    for field in ("next_state_probabilities", "persistence_diagonal", "stationary_distribution"):
        assert list(result[field].keys()) == PROB_KEYS, field
    assert list(result["walk_forward"].keys()) == ["sharpe", "max_drawdown", "n_trades"]
    assert result["hmm"] == {"available": False, "reason": "disabled via --no-hmm"}


def test_states_order_framework_and_disclaimer_are_frozen(result: dict) -> None:
    assert result["states"] == ["Bear", "Sideways", "Bull"]
    assert result["framework"] == FRAMEWORK == "Roan (@RohOnChain)"
    assert result["disclaimer"] == DISCLAIMER == "Backtests are historical, not forward-looking."


def test_value_types_match_companion(result: dict) -> None:
    assert isinstance(result["rows"], int) and result["rows"] == LONG_ROWS
    assert isinstance(result["signal"], float)
    assert len(result["transition_matrix"]) == 3
    assert all(isinstance(x, float) for row in result["transition_matrix"] for x in row)
    assert isinstance(result["walk_forward"]["n_trades"], int)
    assert isinstance(result["walk_forward"]["sharpe"], float)
    assert result["params"]["return_type"] == "simple"


def test_dates_are_iso_strings(result: dict) -> None:
    assert result["date_start"] == "2020-01-01"
    assert result["date_end"] == str(pd.bdate_range("2020-01-01", periods=LONG_ROWS)[-1].date())


def test_to_json_round_trips(result: dict) -> None:
    text = to_json(result)
    assert json.loads(text) == result


def test_to_json_keeps_nan_only_for_walk_forward_on_short_history() -> None:
    short = analyze(_synthetic_close(SHORT_ROWS), source="short", hmm=False)
    parsed = json.loads(to_json(short))
    assert math.isnan(parsed["walk_forward"]["sharpe"])
    assert math.isnan(parsed["walk_forward"]["max_drawdown"])
    assert parsed["walk_forward"]["n_trades"] == 0

    def _no_nan(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _no_nan(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                _no_nan(value, f"{path}[{i}]")
        elif isinstance(node, float):
            assert not math.isnan(node), path

    for field in COMPANION_FIELDS:
        if field != "walk_forward":
            _no_nan(parsed[field], field)


def test_analyze_raises_value_error_on_too_few_rows() -> None:
    with pytest.raises(ValueError, match="Not enough data to label regimes"):
        analyze(_synthetic_close(10), source="tiny", window=20, hmm=False)


def test_cli_json_emits_exactly_one_object(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.csv"
    _synthetic_close(LONG_ROWS).to_csv(path)
    result = CliRunner().invoke(app, ["analyze", "--csv", str(path), "--no-hmm", "--json"])
    assert result.exit_code == 0, result.output
    stdout = result.stdout
    assert stdout.count("\n") == 1 and stdout.endswith("\n")
    parsed = json.loads(stdout)
    assert list(parsed.keys()) == COMPANION_FIELDS
    assert parsed["source"] == str(path)
