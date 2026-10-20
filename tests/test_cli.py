"""``hedgefund analyze``, ``backtest``, ``scan`` and ``report`` CLI behaviour: argument
validation, output modes, error paths, the Phase 3 cost / rule flags and the Phase 5 cache,
jobs and HTML layout."""

import inspect
import json
import shutil
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
# ... and, since Phase 3, the backtest keys below min_train plus a costs block and universe.
BACKTEST_YAML = (
    "  refit: expanding\n  rolling_bars: 756\n  rule: sign\n  dead_band: 0.0\n"
    "  horizons: [1, 5, 10, 20]\n  holdout_years: 2\n  lag_order: 20\n"
    "  cost_sensitivity_bps: [0, 5, 10, 20, 50]\n"
    "  sensitivity:\n    windows: [10, 20, 60]\n    thresholds: [0.02, 0.03, 0.05, 0.07, 0.10]\n"
)
COSTS_YAML = (
    "costs:\n  profiles:\n    us_etfs:\n      fee_bps: 1\n      slippage_bps: 2\n"
    "      borrow_bps: 30\n      funding_bps: 0\n"
    "universe: us_etfs\n"
)
ZERO_COST_FLAGS = [
    "--fee-bps",
    "0",
    "--slippage-bps",
    "0",
    "--borrow-bps",
    "0",
    "--funding-bps",
    "0",
]
LONG_ROWS = 700  # enough for the walk-forward with SMALL_MIN_TRAIN and a two-year holdout
BACKTEST_ONLY_KEYS = [
    "source",
    "rows",
    "date_start",
    "date_end",
    "params",
    "backtest",
    "framework",
    "disclaimer",
]


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
        f"data:\n  years: 4\nbacktest:\n  min_train: {SMALL_MIN_TRAIN}\n{BACKTEST_YAML}"
        f"{STATISTICS_YAML}{COSTS_YAML}"
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
    assert parsed["backtest"]["params"]["return_type"] == "log"
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
        f"data:\n  years: 4\nbacktest:\n  min_train: {SMALL_MIN_TRAIN}\n{BACKTEST_YAML}"
        f"{COSTS_YAML}"
    )
    result = CliRunner().invoke(
        app, ["analyze", "--csv", str(csv_path), "--no-hmm", "--json", "--config", str(custom)]
    )
    assert result.exit_code == 1
    assert "config is missing statistics" in json.loads(result.stdout)["error"]


def test_config_without_backtest_rule_is_an_error(csv_path: Path, tmp_path: Path) -> None:
    custom = tmp_path / "partial.yaml"
    custom.write_text(
        "labelling:\n  window: 15\n  threshold: 0.03\n  return_type: log\n"
        f"data:\n  years: 4\nbacktest:\n  min_train: {SMALL_MIN_TRAIN}\n"
        f"{STATISTICS_YAML}{COSTS_YAML}"
    )
    result = CliRunner().invoke(
        app, ["analyze", "--csv", str(csv_path), "--no-hmm", "--json", "--config", str(custom)]
    )
    assert result.exit_code == 1
    assert "config is missing backtest." in json.loads(result.stdout)["error"]


# --------------------------------------------------------------------------- #
# Phase 3: analyze carries the backtest block; the new ``backtest`` command
# --------------------------------------------------------------------------- #
@pytest.fixture
def long_csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "long_prices.csv"
    _synthetic_close(LONG_ROWS).to_csv(path)
    return path


def _backtest_json(path: Path, *flags: str) -> dict:
    result = CliRunner().invoke(
        app,
        ["backtest", "--csv", str(path), "--json", "--null-paths", str(FAST_NULL_PATHS), *flags],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.count("\n") == 1
    return json.loads(result.stdout)


def test_analyze_json_still_has_all_old_keys_plus_backtest(csv_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["analyze", "--csv", str(csv_path), "--no-hmm", "--json", *FAST_FLAGS]
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    for key in (
        "source",
        "transition_matrix",
        "walk_forward",
        "stride_matrix",
        "statistics_params",
    ):
        assert key in parsed, key
    assert list(parsed)[-1] == "backtest"
    bt = parsed["backtest"]
    assert bt["params"]["rule"] == "sign" and bt["params"]["refit"] == "expanding"
    assert bt["params"]["costs"] == {
        "fee_bps": 1.0,
        "slippage_bps": 2.0,
        "borrow_bps": 30.0,
        "funding_bps": 0.0,
        "per_side_bps": 3.0,
    }  # config/default.yaml costs.profiles.us_etfs via universe us_etfs


def test_analyze_accepts_refit_rule_dead_band_and_cost_profile(csv_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "--csv",
            str(csv_path),
            "--no-hmm",
            "--json",
            *FAST_FLAGS,
            "--refit",
            "rolling:100",
            "--rule",
            "long_only",
            "--dead-band",
            "0.5",
            "--cost-profile",
            "crypto",
        ],
    )
    assert result.exit_code == 0, result.output
    params = json.loads(result.stdout)["backtest"]["params"]
    assert params["refit"] == "rolling" and params["rolling_bars"] == 100
    assert params["rule"] == "long_only" and params["dead_band"] == 0.5
    assert params["costs"]["fee_bps"] == 10.0 and params["costs"]["slippage_bps"] == 10.0


def test_backtest_json_prints_one_object_with_the_slim_shape(long_csv_path: Path) -> None:
    parsed = _backtest_json(long_csv_path, "--min-train", str(SMALL_MIN_TRAIN))
    assert list(parsed.keys()) == BACKTEST_ONLY_KEYS
    assert parsed["framework"] == "Roan (@RohOnChain)"
    assert parsed["disclaimer"] == "Backtests are historical, not forward-looking."
    assert parsed["source"] == str(long_csv_path) and parsed["rows"] == LONG_ROWS
    bt = parsed["backtest"]
    assert bt["net"]["n_days"] > 0
    assert bt["null"]["random_sign"]["n_paths"] == FAST_NULL_PATHS
    assert "NaN" not in json.dumps(parsed)


def test_backtest_refit_rolling_colon_sets_rolling_bars(long_csv_path: Path) -> None:
    params = _backtest_json(long_csv_path, "--refit", "rolling:100")["backtest"]["params"]
    assert params["refit"] == "rolling"
    assert params["rolling_bars"] == 100


def test_backtest_explicit_rolling_bars_flag_wins_over_the_suffix(long_csv_path: Path) -> None:
    params = _backtest_json(long_csv_path, "--refit", "rolling:100", "--rolling-bars", "200")
    assert params["backtest"]["params"]["rolling_bars"] == 200
    plain = _backtest_json(long_csv_path, "--refit", "rolling")["backtest"]["params"]
    assert plain["rolling_bars"] == 756  # config/default.yaml backtest.rolling_bars


def test_backtest_rule_flag_is_echoed(long_csv_path: Path) -> None:
    params = _backtest_json(long_csv_path, "--rule", "long_only")["backtest"]["params"]
    assert params["rule"] == "long_only"


def test_backtest_zero_cost_flags_make_net_equal_gross(long_csv_path: Path) -> None:
    bt = _backtest_json(long_csv_path, "--min-train", str(SMALL_MIN_TRAIN), *ZERO_COST_FLAGS)[
        "backtest"
    ]
    assert bt["params"]["costs"] == {
        "fee_bps": 0.0,
        "slippage_bps": 0.0,
        "borrow_bps": 0.0,
        "funding_bps": 0.0,
        "per_side_bps": 0.0,
    }
    assert bt["net"] == bt["gross"]
    assert bt["annual_cost"] == 0.0


def test_backtest_cost_flag_overrides_one_profile_value(long_csv_path: Path) -> None:
    costs = _backtest_json(long_csv_path, "--cost-profile", "crypto", "--fee-bps", "3")["backtest"][
        "params"
    ]["costs"]
    assert costs == {
        "fee_bps": 3.0,
        "slippage_bps": 10.0,
        "borrow_bps": 0.0,
        "funding_bps": 0.0,
        "per_side_bps": 13.0,
    }


def test_backtest_unknown_rule_exits_1_with_error_object(long_csv_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["backtest", "--csv", str(long_csv_path), "--json", "--rule", "yolo"]
    )
    assert result.exit_code == 1
    parsed = json.loads(result.stdout)
    assert list(parsed.keys()) == ["error"]
    assert "--rule must be one of" in parsed["error"] and "yolo" in parsed["error"]


@pytest.mark.parametrize("bad", ["monthly", "rolling:abc"])
def test_backtest_unknown_refit_exits_1(long_csv_path: Path, bad: str) -> None:
    result = CliRunner().invoke(
        app, ["backtest", "--csv", str(long_csv_path), "--json", "--refit", bad]
    )
    assert result.exit_code == 1
    assert "--refit" in json.loads(result.stdout)["error"]


def test_backtest_unknown_cost_profile_exits_1(long_csv_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["backtest", "--csv", str(long_csv_path), "--json", "--cost-profile", "moon"]
    )
    assert result.exit_code == 1
    assert "unknown cost profile" in json.loads(result.stdout)["error"]


def test_backtest_without_source_prints_error_object() -> None:
    result = CliRunner().invoke(app, ["backtest", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {"error": "Provide exactly one of --ticker or --csv."}


def test_backtest_pretty_mode_prints_the_report(long_csv_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "backtest",
            "--csv",
            str(long_csv_path),
            "--min-train",
            str(SMALL_MIN_TRAIN),
            "--null-paths",
            str(FAST_NULL_PATHS),
        ],
    )
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert out.startswith(f"\nmarkov-regime backtest — source={long_csv_path} window=20 ")
    assert "Backtest (Phase 3):" in out
    assert "buy-and-hold" in out
    assert "percentile" in out
    assert "Costs: 3.0 bps per side" in out
    assert "Turnover" in out
    assert " Framework: Roan (@RohOnChain)." in out


def test_backtest_pretty_mode_on_bad_path_reports_error_on_stderr(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["backtest", "--csv", str(tmp_path / "nope.csv")])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.startswith("\nERROR: ")


def test_backtest_ticker_uses_fetch_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_fetch(ticker: str, years: int = 0) -> pd.Series:
        calls.append((ticker, years))
        return _synthetic_close(LONG_ROWS)

    monkeypatch.setattr(cli, "fetch_ticker", fake_fetch)
    result = CliRunner().invoke(
        app, ["backtest", "--ticker", "FAKE", "--years", "3", "--json", "--null-paths", "5"]
    )
    assert result.exit_code == 0, result.output
    assert calls == [("FAKE", 3)]
    assert json.loads(result.stdout)["source"] == "FAKE"


def test_backtest_short_history_still_serialises_with_null_metrics(csv_path: Path) -> None:
    parsed = _backtest_json(csv_path)  # 300 rows < min_train 252 + 30
    bt = parsed["backtest"]
    assert bt["gross"]["sharpe"] is None and bt["net"]["n_days"] == 0
    assert bt["null"] is None and bt["holdout"] is None
    assert "NaN" not in json.dumps(parsed)


# --------------------------------------------------------------------------- #
# Phase 4: HMM flags and config block
# --------------------------------------------------------------------------- #
HMM_ROWS = 500  # ~380 decision days with a 100-bar first training window
HMM_MIN_TRAIN = 100
HMM_REFIT = 100
HMM_YAML = (
    "hmm:\n  seeds: [0, 1, 2]\n  n_iter: 200\n  emission: gaussian\n  enter: 0.6\n"
    "  exit: 0.4\n  confirm_bars: 2\n  refit_every: 21\n  train_bars: 2520\n"
    "  vol_short: 10\n  vol_long: 20\n  match_window: 20\n"
)


@pytest.fixture
def hmm_csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "hmm_prices.csv"
    _synthetic_close(HMM_ROWS).to_csv(path)
    return path


def test_analyze_hmm_flags_produce_one_json_object_with_an_available_block(
    hmm_csv_path: Path,
) -> None:
    pytest.importorskip("hmmlearn")
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "--csv",
            str(hmm_csv_path),
            "--json",
            "--hmm-seeds",
            "0",
            "--hmm-refit-every",
            str(HMM_REFIT),
            "--min-train",
            str(HMM_MIN_TRAIN),
            *FAST_FLAGS,
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.count("\n") == 1
    parsed = json.loads(result.stdout)
    hmm = parsed["hmm"]
    assert hmm["available"] is True
    assert hmm["fit"]["seeds"] == [0]
    assert hmm["walk_forward"]["params"]["refit_every"] == HMM_REFIT
    assert hmm["walk_forward"]["params"]["min_train"] == HMM_MIN_TRAIN
    assert hmm["decision"]["enter"] == 0.6  # from config/default.yaml hmm.enter


def test_analyze_bad_hmm_emission_exits_1_with_error_object(csv_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["analyze", "--csv", str(csv_path), "--json", "--hmm-emission", "bogus"]
    )
    assert result.exit_code == 1
    parsed = json.loads(result.stdout)
    assert list(parsed) == ["error"]
    assert "--hmm-emission" in parsed["error"] and "bogus" in parsed["error"]


@pytest.mark.parametrize("bad", ["", "0,x", "a"])
def test_analyze_bad_hmm_seeds_exits_1_with_error_object(csv_path: Path, bad: str) -> None:
    result = CliRunner().invoke(
        app, ["analyze", "--csv", str(csv_path), "--json", "--hmm-seeds", bad]
    )
    assert result.exit_code == 1
    assert "--hmm-seeds" in json.loads(result.stdout)["error"]


def test_analyze_hmm_refit_every_zero_exits_1_with_error_object(hmm_csv_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "--csv",
            str(hmm_csv_path),
            "--json",
            "--hmm-refit-every",
            "0",
            "--min-train",
            str(HMM_MIN_TRAIN),
            *FAST_FLAGS,
        ],
    )
    assert result.exit_code == 1
    assert result.stdout.count("\n") == 1
    parsed = json.loads(result.stdout)
    assert list(parsed) == ["error"]
    assert "refit_every" in parsed["error"]


def test_hmm_seeds_parser_accepts_spaces_and_several_seeds() -> None:
    assert cli._parse_seeds("0, 1,2") == (0, 1, 2)


def test_config_without_hmm_block_is_an_error_only_when_hmm_runs(
    csv_path: Path, tmp_path: Path
) -> None:
    custom = tmp_path / "no_hmm_block.yaml"
    custom.write_text(
        "labelling:\n  window: 15\n  threshold: 0.03\n  return_type: log\n"
        f"data:\n  years: 4\nbacktest:\n  min_train: {SMALL_MIN_TRAIN}\n{BACKTEST_YAML}"
        f"{STATISTICS_YAML}{COSTS_YAML}"
    )
    with_hmm = CliRunner().invoke(
        app, ["analyze", "--csv", str(csv_path), "--json", "--config", str(custom)]
    )
    assert with_hmm.exit_code == 1
    assert "config is missing hmm." in json.loads(with_hmm.stdout)["error"]
    without = CliRunner().invoke(
        app, ["analyze", "--csv", str(csv_path), "--no-hmm", "--json", "--config", str(custom)]
    )
    assert without.exit_code == 0, without.output


def test_hmm_config_block_supplies_the_walk_forward_params(
    hmm_csv_path: Path, tmp_path: Path
) -> None:
    pytest.importorskip("hmmlearn")
    custom = tmp_path / "custom_hmm.yaml"
    hmm_yaml = (
        HMM_YAML.replace("seeds: [0, 1, 2]", "seeds: [3]")
        .replace("n_iter: 200", "n_iter: 40")
        .replace("refit_every: 21", f"refit_every: {HMM_REFIT}")
        .replace("confirm_bars: 2", "confirm_bars: 3")
    )
    custom.write_text(
        "labelling:\n  window: 20\n  threshold: 0.05\n  return_type: simple\n"
        f"data:\n  years: 4\nbacktest:\n  min_train: {HMM_MIN_TRAIN}\n{BACKTEST_YAML}"
        f"{STATISTICS_YAML}{COSTS_YAML}{hmm_yaml}"
    )
    result = CliRunner().invoke(
        app, ["analyze", "--csv", str(hmm_csv_path), "--json", "--config", str(custom)]
    )
    assert result.exit_code == 0, result.output
    hmm = json.loads(result.stdout)["hmm"]
    assert hmm["available"] is True, hmm
    assert hmm["fit"]["seeds"] == [3]
    assert hmm["decision"]["confirm_bars"] == 3
    assert hmm["walk_forward"]["params"]["refit_every"] == HMM_REFIT


# --------------------------------------------------------------------------- #
# Phase 5: ``scan`` over a universe through the cache, and ``report`` from a run
# --------------------------------------------------------------------------- #
SCAN_ROWS = 600  # > min_train 252 + a two-year holdout so each symbol gets a net Sharpe
SCAN_SYMBOLS = ("AAA", "BBB")
SCAN_UNIVERSE_NAME = "pair"
SCAN_FLAGS = ["--no-hmm", *FAST_FLAGS]
SCAN_JSON_KEYS = ("run_id", "universe", "table", "html", "framework", "disclaimer")
UNKNOWN_RUN_ID = "19700101T000000Z_nope"


def _write_scan_csv(csv_dir: Path, symbol: str, seed: int) -> Path:
    """``<csv_dir>/<symbol>.csv`` with ``date, close`` columns ending yesterday (UTC).

    Ending yesterday keeps every row inside the cache's default window no matter when the
    test runs: the window spans ``years`` back and stops at yesterday, because the cache
    never stores today's partial bar.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, DAILY_VOL, SCAN_ROWS)
    end = pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    index = pd.bdate_range(end=end, periods=SCAN_ROWS, name="date")
    close = pd.Series(START_PRICE * np.cumprod(1.0 + returns), index=index, name="close")
    path = csv_dir / f"{symbol}.csv"
    close.to_csv(path)
    return path


def _scan_args(workspace: dict, *extra: str) -> list[str]:
    return [
        "scan",
        "--universe",
        str(workspace["universe"]),
        "--source",
        "csv",
        "--csv-dir",
        str(workspace["csv_dir"]),
        "--cache-dir",
        str(workspace["cache_dir"]),
        "--jobs-dir",
        str(workspace["jobs_dir"]),
        *SCAN_FLAGS,
        *extra,
    ]


@pytest.fixture(scope="module")
def scan_workspace(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Two synthetic CSVs, a universe file, an empty cache and jobs dir, plus one cold scan.

    Module-scoped because the cold scan is the slow part; the tests below read the files it
    wrote and run the cheap follow-ups (``--offline``, ``report``) themselves.
    """
    root = tmp_path_factory.mktemp("scan")
    csv_dir = root / "csv"
    csv_dir.mkdir()
    for offset, symbol in enumerate(SCAN_SYMBOLS):
        _write_scan_csv(csv_dir, symbol, SEED + offset)
    universe = root / f"{SCAN_UNIVERSE_NAME}.yaml"
    universe.write_text(
        f"name: {SCAN_UNIVERSE_NAME}\ncost_profile: us_etfs\nsymbols: {list(SCAN_SYMBOLS)}\n"
    )
    workspace = {
        "csv_dir": csv_dir,
        "universe": universe,
        "cache_dir": root / "cache",
        "jobs_dir": root / "jobs",
    }
    result = CliRunner().invoke(app, _scan_args(workspace, "--json"))
    assert result.exit_code == 0, result.output
    assert result.stdout.count("\n") == 1
    workspace["cold"] = json.loads(result.stdout)
    workspace["run_dir"] = workspace["jobs_dir"] / workspace["cold"]["run_id"]
    return workspace


def _read_manifest_file(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text())


def test_scan_json_prints_one_object_with_the_scan_keys(scan_workspace: dict) -> None:
    cold = scan_workspace["cold"]
    for key in SCAN_JSON_KEYS:
        assert key in cold, key
    assert cold["universe"] == SCAN_UNIVERSE_NAME
    assert cold["framework"] == "Roan (@RohOnChain)"
    assert cold["disclaimer"] == "Backtests are historical, not forward-looking."
    assert cold["html"] == str(scan_workspace["run_dir"] / "html" / "index.html")


def test_scan_two_symbols_writes_two_cached_frames_and_two_results(scan_workspace: dict) -> None:
    cached = sorted(scan_workspace["cache_dir"].rglob("*.parquet"))
    assert len(cached) == len(SCAN_SYMBOLS), cached
    results = sorted((scan_workspace["run_dir"] / "results").glob("*.json"))
    assert [p.stem for p in results] == list(SCAN_SYMBOLS)
    manifest = _read_manifest_file(scan_workspace["run_dir"])
    assert manifest["errors"] == {}
    assert sorted(manifest["data"]) == list(SCAN_SYMBOLS)
    for symbol in SCAN_SYMBOLS:
        entry = manifest["data"][symbol]
        assert entry["cache_hit"] is False  # cold cache: every symbol was a miss
        assert entry["rows"] == SCAN_ROWS
        assert Path(entry["cache_path"]) in cached


def test_scan_two_symbols_produces_one_ranked_table(scan_workspace: dict) -> None:
    table = scan_workspace["cold"]["table"]
    assert [row["rank"] for row in table] == [1, 2]
    assert sorted(row["symbol"] for row in table) == list(SCAN_SYMBOLS)
    assert all(row["error"] is None for row in table)
    assert all(row["current_regime"] in ("Bear", "Sideways", "Bull") for row in table)
    on_disk = json.loads((scan_workspace["run_dir"] / "table.json").read_text())
    assert on_disk == table


def test_scan_offline_rerun_hits_the_warm_cache_for_every_symbol(scan_workspace: dict) -> None:
    result = CliRunner().invoke(app, _scan_args(scan_workspace, "--offline", "--json"))
    assert result.exit_code == 0, result.output
    warm = json.loads(result.stdout)
    manifest = _read_manifest_file(scan_workspace["jobs_dir"] / warm["run_id"])
    assert sorted(manifest["data"]) == list(SCAN_SYMBOLS)
    assert all(entry["cache_hit"] is True for entry in manifest["data"].values())
    assert manifest["errors"] == {}
    assert [row["symbol"] for row in warm["table"]] == [
        row["symbol"] for row in scan_workspace["cold"]["table"]
    ]


def test_scan_offline_miss_is_a_per_symbol_error_row_not_an_abort(
    scan_workspace: dict, tmp_path: Path
) -> None:
    """Only the first symbol is cached: it is scored, the other gets an error row, exit 0."""
    cached, missing = SCAN_SYMBOLS
    partial_cache = tmp_path / "partial_cache"
    for parquet in scan_workspace["cache_dir"].rglob(f"{cached}_*.parquet"):
        target = partial_cache / parquet.relative_to(scan_workspace["cache_dir"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(parquet, target)
    assert list(partial_cache.rglob("*.parquet")), "the cached symbol's parquet was not copied"
    jobs_dir = tmp_path / "jobs"
    result = CliRunner().invoke(
        app,
        _scan_args(
            {**scan_workspace, "cache_dir": partial_cache, "jobs_dir": jobs_dir},
            "--offline",
            "--json",
        ),
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    rows = {row["symbol"]: row for row in parsed["table"]}
    assert rows[cached]["rank"] == 1 and rows[cached]["error"] is None
    assert rows[missing]["rank"] is None
    assert "offline" in rows[missing]["error"]
    manifest = _read_manifest_file(jobs_dir / parsed["run_id"])
    assert list(manifest["errors"]) == [missing]
    assert "offline" in manifest["errors"][missing]
    assert list(parsed["errors"]) == [missing]


def test_scan_offline_with_every_symbol_cached_has_no_errors(
    scan_workspace: dict, tmp_path: Path
) -> None:
    result = CliRunner().invoke(
        app,
        _scan_args({**scan_workspace, "jobs_dir": tmp_path / "jobs"}, "--offline", "--json"),
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert parsed["errors"] == {}
    assert _read_manifest_file(tmp_path / "jobs" / parsed["run_id"])["errors"] == {}
    assert [row["error"] for row in parsed["table"]] == [None, None]


def test_scan_pretty_mode_prints_the_table_and_run_id(scan_workspace: dict, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        _scan_args({**scan_workspace, "jobs_dir": tmp_path / "jobs"}, "--offline", "--no-html"),
    )
    assert result.exit_code == 0, result.output
    out = result.stdout
    for symbol in SCAN_SYMBOLS:
        assert symbol in out
    assert "run id: " in out
    assert "html:" not in out
    run_id = next(line for line in out.splitlines() if line.startswith("run id: ")).split(": ")[1]
    assert not (tmp_path / "jobs" / run_id / "html").exists()


def test_scan_writes_an_html_index_and_one_page_per_symbol(scan_workspace: dict) -> None:
    html_dir = scan_workspace["run_dir"] / "html"
    assert (html_dir / "index.html").is_file()
    for symbol in SCAN_SYMBOLS:
        assert (html_dir / f"{symbol}.html").is_file()
    assert "Backtests are historical, not forward-looking." in (html_dir / "index.html").read_text()


def test_scan_csv_source_without_csv_dir_is_an_error(scan_workspace: dict, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "scan",
            "--universe",
            str(scan_workspace["universe"]),
            "--source",
            "csv",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--json",
        ],
    )
    assert result.exit_code == 1
    parsed = json.loads(result.stdout)
    assert list(parsed) == ["error"]
    assert "--csv-dir" in parsed["error"]
    assert not (tmp_path / "jobs").exists()


def test_scan_unknown_universe_is_an_error(scan_workspace: dict, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        _scan_args({**scan_workspace, "universe": "no_such_universe"}, "--json"),
    )
    assert result.exit_code == 1
    parsed = json.loads(result.stdout)
    assert list(parsed) == ["error"]
    assert "no_such_universe" in parsed["error"]


def test_scan_unknown_source_is_an_error(scan_workspace: dict, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "scan",
            "--universe",
            str(scan_workspace["universe"]),
            "--source",
            "alpaca",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--json",
        ],
    )
    assert result.exit_code == 1
    assert "--source" in json.loads(result.stdout)["error"]


def test_scan_pretty_mode_error_goes_to_stderr(scan_workspace: dict, tmp_path: Path) -> None:
    result = CliRunner().invoke(app, _scan_args({**scan_workspace, "universe": "no_such_universe"}))
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.startswith("\nERROR: ")


def test_report_regenerates_the_html_from_a_cached_run(scan_workspace: dict) -> None:
    html_dir = scan_workspace["run_dir"] / "html"
    for page in html_dir.iterdir():
        page.unlink()
    html_dir.rmdir()
    assert not html_dir.exists()
    run_id = scan_workspace["cold"]["run_id"]
    result = CliRunner().invoke(
        app, ["report", run_id, "--jobs-dir", str(scan_workspace["jobs_dir"])]
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == str(html_dir / "index.html")
    assert (html_dir / "index.html").is_file()
    for symbol in SCAN_SYMBOLS:
        assert (html_dir / f"{symbol}.html").is_file()


def test_report_json_prints_run_id_and_html_path(scan_workspace: dict) -> None:
    run_id = scan_workspace["cold"]["run_id"]
    result = CliRunner().invoke(
        app, ["report", run_id, "--jobs-dir", str(scan_workspace["jobs_dir"]), "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "run_id": run_id,
        "html": str(scan_workspace["run_dir"] / "html" / "index.html"),
    }


def test_report_unknown_run_id_prints_an_error_object(scan_workspace: dict) -> None:
    result = CliRunner().invoke(
        app, ["report", UNKNOWN_RUN_ID, "--jobs-dir", str(scan_workspace["jobs_dir"]), "--json"]
    )
    assert result.exit_code == 1
    parsed = json.loads(result.stdout)
    assert list(parsed) == ["error"]
    assert UNKNOWN_RUN_ID in parsed["error"]


def test_report_names_the_symbol_when_its_cache_file_is_gone(
    scan_workspace: dict, tmp_path: Path
) -> None:
    """A run whose cache was wiped tells the user which symbol to re-scan, not a parquet trace."""
    run_dir = scan_workspace["run_dir"]
    copy = tmp_path / "jobs" / run_dir.name
    copy.parent.mkdir()
    shutil.copytree(run_dir, copy)
    manifest = _read_manifest_file(copy)
    gone = tmp_path / "gone.parquet"
    manifest["data"][SCAN_SYMBOLS[1]]["cache_path"] = str(gone)
    (copy / "manifest.json").write_text(json.dumps(manifest))
    result = CliRunner().invoke(
        app, ["report", run_dir.name, "--jobs-dir", str(tmp_path / "jobs"), "--json"]
    )
    assert result.exit_code == 1
    error = json.loads(result.stdout)["error"]
    assert SCAN_SYMBOLS[1] in error and str(gone) in error and "hedgefund scan" in error


def test_analyze_kwargs_covers_every_analyze_keyword_but_close_source_and_cost_model() -> None:
    """The helper both ``analyze`` and ``scan`` share must hand ``analyze()`` its full signature."""
    cfg = cli.load_config()
    kwargs = cli._analyze_kwargs(
        cfg,
        window=None,
        threshold=None,
        min_train=None,
        no_hmm=False,
        hmm_seeds=None,
        hmm_refit_every=None,
        hmm_emission=None,
        null_paths=None,
        bootstrap_resamples=None,
        seed=None,
        refit=None,
        rule=None,
        dead_band=None,
    )
    expected = set(inspect.signature(cli.analyze).parameters) - {"close", "source", "cost_model"}
    assert set(kwargs) == expected
    assert kwargs["window"] == 20 and kwargs["hmm"] is True and kwargs["refit"] == "expanding"
