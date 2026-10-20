"""``hedgefund analyze`` CLI behaviour: argument validation, output modes, error paths."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import hedgefund.cli as cli
from hedgefund.cli import app

SEED = 11
ROWS = 300  # > window (20) but < min_train + 30, so the walk-forward reports NaN
DAILY_VOL = 0.01
START_PRICE = 50.0
SMALL_MIN_TRAIN = 60  # lets the walk-forward produce a Sharpe on 300 rows
FAST_NULL_PATHS = 20  # small Monte Carlo sizes keep CLI tests fast
FAST_RESAMPLES = 20
FAST_FLAGS = [
    "--null-paths",
    str(FAST_NULL_PATHS),
    "--bootstrap-resamples",
    str(FAST_RESAMPLES),
]
# A custom config must carry the statistics block the CLI reads (config over flags).
STATISTICS_YAML = (
    f"statistics:\n  null_paths: {FAST_NULL_PATHS}\n"
    f"  bootstrap:\n    block_days: 10\n    resamples: {FAST_RESAMPLES}\n"
    "  rolling_bars: 100\n  ew_half_life: 50\n  shrinkage_alpha: 5\n"
    "  mixing_epsilon: 0.05\n  homogeneity_periods: 2\n  seed: 9\n"
)


def _synthetic_close(rows: int = ROWS, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, DAILY_VOL, rows)
    index = pd.bdate_range("2021-01-04", periods=rows, name="date")
    return pd.Series(START_PRICE * np.cumprod(1.0 + returns), index=index, name="close")


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "prices.csv"
    _synthetic_close().to_csv(path)
    return path


def test_version_still_works() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == cli.__version__


def test_analyze_without_ticker_or_csv_exits_1() -> None:
    result = CliRunner().invoke(app, ["analyze"])
    assert result.exit_code == 1
    assert "ERROR: Provide exactly one of --ticker or --csv." in result.stderr
    assert result.stdout == ""


def test_analyze_with_both_ticker_and_csv_exits_1(csv_path: Path) -> None:
    result = CliRunner().invoke(app, ["analyze", "--ticker", "SPY", "--csv", str(csv_path)])
    assert result.exit_code == 1
    assert "ERROR: Provide exactly one of --ticker or --csv." in result.stderr


def test_analyze_csv_prints_pretty_output(csv_path: Path) -> None:
    result = CliRunner().invoke(app, ["analyze", "--csv", str(csv_path), "--no-hmm", *FAST_FLAGS])
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert out.startswith(f"\nmarkov-regime — source={csv_path} window=20 threshold=0.05\n")
    assert f"  {ROWS} rows | 2021-01-04 -> " in out
    assert "Transition matrix (rows = from, cols = to):" in out
    assert "  Sharpe: NaN (insufficient data — try a longer history)" in out
    assert "HMM skipped: disabled via --no-hmm" in out
    assert "Honest statistics (Phase 2):" in out
    assert " Framework: Roan (@RohOnChain)." in out
    assert "fetching" not in result.stderr


def test_analyze_json_on_bad_path_prints_error_object_only(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.csv"
    result = CliRunner().invoke(app, ["analyze", "--csv", str(missing), "--json"])
    assert result.exit_code == 1
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert list(parsed.keys()) == ["error"]
    assert parsed["error"]


def test_analyze_json_without_source_prints_error_object(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["analyze", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {"error": "Provide exactly one of --ticker or --csv."}


def test_analyze_pretty_on_bad_path_reports_error_on_stderr(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.csv"
    result = CliRunner().invoke(app, ["analyze", "--csv", str(missing)])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.startswith("\nERROR: ")


def test_analyze_ticker_uses_fetch_ticker_and_config_years(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_fetch(ticker: str, years: int = 0) -> pd.Series:
        calls.append((ticker, years))
        return _synthetic_close()

    monkeypatch.setattr(cli, "fetch_ticker", fake_fetch)
    result = CliRunner().invoke(
        app, ["analyze", "--ticker", "FAKE", "--no-hmm", "--json", *FAST_FLAGS]
    )
    assert result.exit_code == 0, result.output
    assert calls == [("FAKE", 10)]  # data.years from config/default.yaml
    parsed = json.loads(result.stdout)
    assert parsed["source"] == "FAKE"
    assert parsed["rows"] == ROWS
    assert "fetching" not in result.stderr


def test_analyze_ticker_pretty_mode_notes_fetch_on_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "fetch_ticker", lambda ticker, years=0: _synthetic_close())
    result = CliRunner().invoke(
        app, ["analyze", "--ticker", "FAKE", "--no-hmm", "--years", "3", *FAST_FLAGS]
    )
    assert result.exit_code == 0, result.output
    assert "  fetching FAKE from Yahoo Finance..." in result.stderr
    assert "markov-regime — source=FAKE" in result.stdout


def test_flags_override_config_defaults(csv_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "--csv",
            str(csv_path),
            "--no-hmm",
            "--json",
            "--window",
            "10",
            "--threshold",
            "0.02",
            "--min-train",
            str(SMALL_MIN_TRAIN),
            *FAST_FLAGS,
        ],
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert parsed["params"] == {
        "window": 10,
        "threshold": 0.02,
        "min_train": SMALL_MIN_TRAIN,
        "return_type": "simple",
    }
    assert parsed["walk_forward"]["n_trades"] > 0


def test_config_file_supplies_defaults(csv_path: Path, tmp_path: Path) -> None:
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "labelling:\n  window: 15\n  threshold: 0.03\n  return_type: log\n"
        f"data:\n  years: 4\nbacktest:\n  min_train: {SMALL_MIN_TRAIN}\n"
        f"{STATISTICS_YAML}"
    )
    result = CliRunner().invoke(
        app, ["analyze", "--csv", str(csv_path), "--no-hmm", "--json", "--config", str(custom)]
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert parsed["params"] == {
        "window": 15,
        "threshold": 0.03,
        "min_train": SMALL_MIN_TRAIN,
        "return_type": "log",
    }
    assert parsed["statistics_params"] == {
        "null_paths": FAST_NULL_PATHS,
        "bootstrap_resamples": FAST_RESAMPLES,
        "block_days": 10,
        "rolling_bars": 100,
        "ew_half_life": 50.0,
        "shrinkage_alpha": 5.0,
        "mixing_epsilon": 0.05,
        "homogeneity_periods": 2,
        "seed": 9,
    }


def test_missing_config_file_is_an_error(csv_path: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["analyze", "--csv", str(csv_path), "--json", "--config", str(tmp_path / "nope.yaml")]
    )
    assert result.exit_code == 1
    assert "config file not found" in json.loads(result.stdout)["error"]


def test_statistics_flags_override_config_and_json_has_new_keys(csv_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["analyze", "--csv", str(csv_path), "--no-hmm", "--json", *FAST_FLAGS, "--seed", "5"],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.count("\n") == 1
    parsed = json.loads(result.stdout)
    for key in ("stride_matrix", "null_percentile", "confidence_intervals", "rolling_matrix"):
        assert key in parsed, key
    assert "mixing_time" in parsed
    sp = parsed["statistics_params"]
    assert sp["null_paths"] == FAST_NULL_PATHS
    assert sp["bootstrap_resamples"] == FAST_RESAMPLES
    assert sp["seed"] == 5
    assert sp["rolling_bars"] == 756  # from config/default.yaml
    assert "NaN" not in json.dumps({k: v for k, v in parsed.items() if k != "walk_forward"})


def test_statistics_defaults_come_from_config(csv_path: Path) -> None:
    """Without the fast flags the CLI uses config/default.yaml's statistics block."""
    result = CliRunner().invoke(app, ["analyze", "--csv", str(csv_path), "--no-hmm", "--json"])
    assert result.exit_code == 0, result.output
    sp = json.loads(result.stdout)["statistics_params"]
    assert sp["null_paths"] == 1000
    assert sp["bootstrap_resamples"] == 500
    assert sp["seed"] == 0


def test_config_without_statistics_block_is_an_error(csv_path: Path, tmp_path: Path) -> None:
    custom = tmp_path / "partial.yaml"
    custom.write_text(
        "labelling:\n  window: 15\n  threshold: 0.03\n  return_type: log\n"
        f"data:\n  years: 4\nbacktest:\n  min_train: {SMALL_MIN_TRAIN}\n"
    )
    result = CliRunner().invoke(
        app, ["analyze", "--csv", str(csv_path), "--no-hmm", "--json", "--config", str(custom)]
    )
    assert result.exit_code == 1
    assert "config is missing statistics" in json.loads(result.stdout)["error"]
