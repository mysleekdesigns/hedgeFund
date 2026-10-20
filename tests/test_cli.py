"""``hedgefund analyze`` and ``hedgefund backtest`` CLI behaviour: argument validation,
output modes, error paths, and the Phase 3 cost / rule flags."""

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
