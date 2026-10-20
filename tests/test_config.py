"""Schema and default-value checks for ``config/default.yaml`` and the universe files.

Phase 0 has no loader module yet, so the YAML is read directly with ``yaml.safe_load``.
Expected values come from plan.md section 5.
"""

from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
UNIVERSE_FILES = sorted((CONFIG_DIR / "universes").glob("*.yaml"))


def _load(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def defaults() -> dict:
    return _load(CONFIG_DIR / "default.yaml")


def test_states_are_bear_sideways_bull_in_canonical_order(defaults: dict) -> None:
    assert defaults["states"] == ["Bear", "Sideways", "Bull"]


def test_labelling_defaults_match_section_5(defaults: dict) -> None:
    labelling = defaults["labelling"]
    assert labelling["window"] == 20
    assert labelling["threshold"] == 0.05
    assert labelling["return_type"] == "simple"


def test_asymmetric_thresholds_default_to_symmetric(defaults: dict) -> None:
    labelling = defaults["labelling"]
    assert labelling["threshold_bull"] is None
    assert labelling["threshold_bear"] is None


def test_data_defaults_match_section_5(defaults: dict) -> None:
    assert defaults["data"]["years"] == 10
    assert defaults["data"]["source"] in {"yfinance", "alpaca", "csv"}


def test_backtest_defaults_match_section_5(defaults: dict) -> None:
    backtest = defaults["backtest"]
    assert backtest["min_train"] == 252
    assert backtest["refit"] == "expanding"
    assert backtest["horizons"] == [1, 5, 10, 20]


def test_phase3_backtest_keys_match_plan_fix_6(defaults: dict) -> None:
    backtest = defaults["backtest"]
    assert backtest["rolling_bars"] == 756
    assert backtest["rule"] == "sign"
    assert backtest["dead_band"] == 0.0
    assert backtest["holdout_years"] == 2
    assert backtest["lag_order"] == 20
    assert backtest["cost_sensitivity_bps"] == [0, 5, 10, 20, 50]
    assert backtest["sensitivity"] == {
        "windows": [10, 20, 60],
        "thresholds": [0.02, 0.03, 0.05, 0.07, 0.10],
    }


def test_backtest_rule_and_refit_are_valid_choices(defaults: dict) -> None:
    from hedgefund.backtest.walk_forward import REFIT_MODES
    from hedgefund.regime.signal import POSITION_RULES

    assert defaults["backtest"]["rule"] in POSITION_RULES
    assert defaults["backtest"]["refit"] in REFIT_MODES


def test_statistics_defaults_match_section_5(defaults: dict) -> None:
    statistics = defaults["statistics"]
    assert statistics["null_paths"] == 1000
    assert statistics["bootstrap"]["block_days"] == 20
    assert statistics["bootstrap"]["resamples"] == 500


@pytest.mark.parametrize(
    ("profile", "fee_bps", "slippage_bps", "borrow_bps", "funding_bps"),
    [("us_etfs", 1, 2, 30, 0), ("crypto", 10, 10, 0, 0)],
)
def test_cost_profile_matches_section_5(
    defaults: dict,
    profile: str,
    fee_bps: int,
    slippage_bps: int,
    borrow_bps: int,
    funding_bps: int,
) -> None:
    costs = defaults["costs"]["profiles"][profile]
    assert costs["fee_bps"] == fee_bps
    assert costs["slippage_bps"] == slippage_bps
    assert costs["borrow_bps"] == borrow_bps
    assert costs["funding_bps"] == funding_bps


def test_cost_profiles_load_into_cost_models(defaults: dict) -> None:
    from hedgefund.backtest.costs import cost_model_from_config

    etfs = cost_model_from_config(defaults, "us_etfs")
    assert (etfs.fee_bps, etfs.slippage_bps, etfs.borrow_bps, etfs.funding_bps) == (1, 2, 30, 0)
    crypto = cost_model_from_config(defaults, "crypto")
    assert crypto.per_side_bps == 20


def test_at_least_one_universe_file_exists() -> None:
    assert UNIVERSE_FILES, f"no universe files under {CONFIG_DIR / 'universes'}"


@pytest.mark.parametrize("universe_path", UNIVERSE_FILES, ids=lambda p: p.stem)
def test_universe_name_matches_file_stem(universe_path: Path) -> None:
    assert _load(universe_path)["name"] == universe_path.stem


@pytest.mark.parametrize("universe_path", UNIVERSE_FILES, ids=lambda p: p.stem)
def test_universe_symbols_are_unique_upper_case_strings(universe_path: Path) -> None:
    symbols = _load(universe_path)["symbols"]
    assert isinstance(symbols, list) and symbols, "symbols must be a non-empty list"
    assert all(isinstance(s, str) and s == s.upper() for s in symbols), symbols
    assert len(set(symbols)) == len(symbols), "duplicate symbols"


@pytest.mark.parametrize("universe_path", UNIVERSE_FILES, ids=lambda p: p.stem)
def test_universe_cost_profile_exists_in_defaults(defaults: dict, universe_path: Path) -> None:
    cost_profile = _load(universe_path)["cost_profile"]
    assert cost_profile in defaults["costs"]["profiles"]


def test_default_universe_names_an_existing_file(defaults: dict) -> None:
    assert (CONFIG_DIR / "universes" / f"{defaults['universe']}.yaml").is_file()
