"""Universe files load by name and by path, and a broken file names what is wrong."""

from pathlib import Path

import pytest

from hedgefund.config import load_config
from hedgefund.jobs.universe import (
    UNIVERSES_DIR,
    Universe,
    load_universe,
    resolve_universe_path,
)

# The two universe files plan.md Phase 5 asks for, with their cost profiles.
SHIPPED_UNIVERSES = {"us_etfs": "us_etfs", "crypto": "crypto"}


def test_universes_dir_is_the_repo_config_dir() -> None:
    assert UNIVERSES_DIR.is_dir()
    assert UNIVERSES_DIR.name == "universes"
    assert UNIVERSES_DIR.parent.name == "config"


@pytest.mark.parametrize(("name", "profile"), sorted(SHIPPED_UNIVERSES.items()))
def test_shipped_universe_loads_by_name(name: str, profile: str) -> None:
    universe = load_universe(name)
    assert isinstance(universe, Universe)
    assert universe.name == name
    assert universe.cost_profile == profile
    assert universe.path == UNIVERSES_DIR / f"{name}.yaml"
    assert len(universe.symbols) >= 2
    assert all(isinstance(s, str) and s for s in universe.symbols)
    assert len(set(universe.symbols)) == len(universe.symbols)


@pytest.mark.parametrize("name", sorted(SHIPPED_UNIVERSES))
def test_shipped_universe_cost_profile_exists_in_default_config(name: str) -> None:
    universe = load_universe(name)
    assert universe.cost_profile in load_config()["costs"]["profiles"]


def test_resolve_by_name_and_by_path(tmp_path: Path) -> None:
    by_name = resolve_universe_path("us_etfs")
    assert by_name == UNIVERSES_DIR / "us_etfs.yaml"
    custom = tmp_path / "mine.yaml"
    custom.write_text("name: mine\ncost_profile: us_etfs\nsymbols: [AAA]\n")
    assert resolve_universe_path(custom) == custom
    assert resolve_universe_path(str(custom)) == custom
    loaded = load_universe(custom)
    assert loaded.symbols == ("AAA",)
    assert loaded.path == custom


def test_resolve_unknown_name_raises_naming_it(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no_such_universe"):
        resolve_universe_path("no_such_universe")
    with pytest.raises(ValueError, match=r"missing\.yaml"):
        load_universe(tmp_path / "missing.yaml")


@pytest.mark.parametrize("missing", ["name", "cost_profile", "symbols"])
def test_missing_key_is_named(tmp_path: Path, missing: str) -> None:
    content = {"name": "x", "cost_profile": "us_etfs", "symbols": ["AAA"]}
    del content[missing]
    path = tmp_path / "bad.yaml"
    path.write_text("\n".join(f"{k}: {v}" for k, v in content.items()) + "\n")
    with pytest.raises(ValueError, match=f"'{missing}'"):
        load_universe(path)


@pytest.mark.parametrize(
    "symbols_yaml",
    ["symbols: []", "symbols: SPY", "symbols: [SPY, 3]", "symbols: [SPY, '']"],
)
def test_bad_symbols_raise(tmp_path: Path, symbols_yaml: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(f"name: x\ncost_profile: us_etfs\n{symbols_yaml}\n")
    with pytest.raises(ValueError, match="symbols"):
        load_universe(path)


def test_non_mapping_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- SPY\n- QQQ\n")
    with pytest.raises(ValueError, match="mapping"):
        load_universe(path)


def test_universe_is_frozen() -> None:
    universe = load_universe("crypto")
    with pytest.raises(AttributeError):
        universe.name = "other"  # type: ignore[misc]
