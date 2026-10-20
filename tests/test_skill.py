"""Phase 6 skill package: ``skills/regime/SKILL.md``, its JSON contract reference and the
``.claude-plugin`` manifests. The gate half this file covers: the documented command, run
in one call, returns exactly one valid JSON object carrying every field the skill documents.
"""

import json
import re
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from hedgefund.cli import app

ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = ROOT / "skills" / "regime" / "SKILL.md"
REFERENCE_PATH = ROOT / "skills" / "regime" / "reference" / "json-contract.md"
PLUGIN_PATH = ROOT / ".claude-plugin" / "plugin.json"
MARKETPLACE_PATH = ROOT / ".claude-plugin" / "marketplace.json"
PYPROJECT_PATH = ROOT / "pyproject.toml"

STATES = ["Bear", "Sideways", "Bull"]
DISCLAIMER = "Backtests are historical, not forward-looking."
PATTERN_HEADINGS = [
    "### (a) Regime confirmation gate",
    "### (b) Stationary bear share as a position-size / tail-risk filter",
    "### (c) Standalone signal",
]
COMPANION_KEYS = [
    "source",
    "rows",
    "date_start",
    "date_end",
    "params",
    "states",
    "current_regime",
    "next_state_probabilities",
    "signal",
    "transition_matrix",
    "persistence_diagonal",
    "stationary_distribution",
    "walk_forward",
    "hmm",
    "framework",
    "disclaimer",
]
EXTENDED_TOP_LEVEL_KEYS = ["stride_matrix", "null_percentile", "backtest"]

SEED = 11
ROWS = 300
DAILY_VOL = 0.01
START_PRICE = 50.0
SMALL_MIN_TRAIN = 60  # lets the walk-forward, nulls and holdout populate on 300 rows
FAST_NULL_PATHS = 20
FAST_RESAMPLES = 20
FAST_FLAGS = [
    "--null-paths",
    str(FAST_NULL_PATHS),
    "--bootstrap-resamples",
    str(FAST_RESAMPLES),
]

HMM_ROWS = 500  # ~380 decision days with a 100-bar first training window (test_cli)
HMM_MIN_TRAIN = 100
HMM_REFIT = 100
HMM_FLAGS = [
    "--hmm-seeds",
    "0",
    "--hmm-refit-every",
    str(HMM_REFIT),
    "--min-train",
    str(HMM_MIN_TRAIN),
]

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
FENCED_BASH_RE = re.compile(r"```bash\n(.*?)```", re.DOTALL)
TABLE_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|", re.MULTILINE)
TABLE_HEADER = "| Field |"


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


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_PATH.read_text()


@pytest.fixture(scope="module")
def reference_text() -> str:
    return REFERENCE_PATH.read_text()


@pytest.fixture(scope="module")
def pyproject_version() -> str:
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def _frontmatter(text: str) -> dict[str, str]:
    match = FRONTMATTER_RE.match(text)
    assert match is not None, "SKILL.md must start with a YAML frontmatter block"
    fields: dict[str, str] = {}
    current = None
    for line in match.group(1).splitlines():
        if line.startswith(" ") and current is not None:
            fields[current] += " " + line.strip()
        else:
            key, _, value = line.partition(":")
            current = key.strip()
            fields[current] = value.strip()
    return fields


def _documented_paths(text: str) -> list[str]:
    """Backticked first-column names of every ``| Field |`` table in a markdown file."""
    paths: list[str] = []
    for block in text.split("\n\n"):
        if not block.lstrip().startswith(TABLE_HEADER):
            continue
        paths.extend(TABLE_ROW_RE.findall(block))
    return paths


def _walk(result: dict, path: str) -> None:
    """Assert ``path`` resolves in ``result``. A documented ``null`` on the way is the
    contract's degrade path and counts as present; so does ``hmm.available == False``."""
    node = result
    for part in path.split("."):
        if node is None:
            return
        assert isinstance(node, dict), f"{path}: {part!r} reached a non-object"
        if node.get("available") is False and part not in node:
            return
        assert part in node, f"{path}: {part!r} missing from the analyze output"
        node = node[part]


def _run_documented_analyze(csv_path: Path, hmm: bool = False) -> tuple[str, dict]:
    extra = HMM_FLAGS if hmm else ["--no-hmm", "--min-train", str(SMALL_MIN_TRAIN)]
    result = CliRunner().invoke(
        app, ["analyze", "--csv", str(csv_path), "--json", *extra, *FAST_FLAGS]
    )
    assert result.exit_code == 0, result.output
    stdout = result.stdout
    assert stdout.strip().startswith("{") and stdout.strip().endswith("}")
    parsed = json.loads(stdout)  # raises on a second object ("Extra data")
    assert isinstance(parsed, dict)
    return stdout, parsed


# --- 1. SKILL.md shape ------------------------------------------------------------------


def test_skill_file_exists_with_frontmatter(skill_text: str) -> None:
    assert SKILL_PATH.is_file()
    fields = _frontmatter(skill_text)
    assert fields["name"] == "regime"
    description = fields["description"].lstrip(">-").strip()
    assert description, "frontmatter description must be non-empty"
    assert "regime" in description.lower()


def test_skill_documents_command_patterns_states_and_disclaimer(skill_text: str) -> None:
    assert "hedgefund analyze" in skill_text
    assert "--json" in skill_text
    for heading in PATTERN_HEADINGS:
        assert heading in skill_text, f"missing composition heading {heading!r}"
    assert json.dumps(STATES, separators=(",", ":")) in skill_text
    assert DISCLAIMER in skill_text
    assert "reference/json-contract.md" in skill_text
    assert "/plugin install hedgefund@hedgefund" in skill_text


def test_skill_warns_about_hindsight_lag_and_prefers_stride(skill_text: str) -> None:
    assert "backtest.lag" in skill_text
    assert "stride_diagonal" in skill_text
    assert "p_value_net" in skill_text
    assert "paper" in skill_text.lower()


def test_skill_stays_short(skill_text: str) -> None:
    assert len(skill_text.splitlines()) <= 260


# --- 2. Plugin manifests ----------------------------------------------------------------


def test_plugin_manifest_matches_pyproject(pyproject_version: str) -> None:
    plugin = json.loads(PLUGIN_PATH.read_text())
    assert plugin["name"] == "hedgefund"
    assert plugin["version"] == pyproject_version
    assert plugin["description"]
    assert plugin["author"]["name"] == "mysleekdesigns"


def test_marketplace_manifest_matches_pyproject(pyproject_version: str) -> None:
    marketplace = json.loads(MARKETPLACE_PATH.read_text())
    assert marketplace["name"] == "hedgefund"
    assert marketplace["metadata"]["version"] == pyproject_version
    assert marketplace["owner"]["name"]
    plugins = marketplace["plugins"]
    assert len(plugins) == 1
    assert plugins[0]["name"] == "hedgefund"
    assert plugins[0]["source"] == "."


def test_plugin_root_exposes_the_skill_directory() -> None:
    """Claude Code discovers ``skills/*/SKILL.md`` under the plugin root (``source: "."``)."""
    assert (ROOT / "skills" / "regime" / "SKILL.md").is_file()
    assert REFERENCE_PATH.is_file()


# --- 3. The gate: one call, one valid JSON object, documented fields present ------------


def test_documented_command_returns_one_json_object(csv_path: Path) -> None:
    stdout, result = _run_documented_analyze(csv_path)
    assert stdout.count("\n{") == 0, "stdout must hold a single top-level object"
    for key in COMPANION_KEYS + EXTENDED_TOP_LEVEL_KEYS:
        assert key in result, f"{key} missing"
    assert result["states"] == STATES
    assert result["disclaimer"] == DISCLAIMER
    assert result["hmm"] == {"available": False, "reason": "disabled via --no-hmm"}


def test_every_documented_field_exists_in_output(
    csv_path: Path, skill_text: str, reference_text: str
) -> None:
    _, result = _run_documented_analyze(csv_path)
    skill_paths = _documented_paths(skill_text)
    reference_paths = _documented_paths(reference_text)
    assert len(skill_paths) >= len(COMPANION_KEYS)
    assert len(reference_paths) > len(skill_paths)
    for path in skill_paths + reference_paths:
        _walk(result, path)
    # The walk-forward populated on this history, so the honest fields are real values.
    assert result["backtest"]["null"] is not None
    assert "p_value_net" in result["backtest"]["null"]["permutation"]
    assert result["stride_diagonal"].keys() == {"bear", "sideways", "bull"}


def test_every_documented_hmm_field_exists_with_the_hmm_on(
    tmp_path: Path, skill_text: str, reference_text: str
) -> None:
    """The ``--no-hmm`` run above short-circuits at ``available: false``; this run has the
    walk-forward HMM on so every documented ``hmm.*`` row is checked against real output."""
    pytest.importorskip("hmmlearn")
    path = tmp_path / "hmm_prices.csv"
    _synthetic_close(HMM_ROWS).to_csv(path)
    _, result = _run_documented_analyze(path, hmm=True)
    assert result["hmm"]["available"] is True
    hmm_paths = [
        p
        for p in _documented_paths(skill_text) + _documented_paths(reference_text)
        if p == "hmm" or p.startswith("hmm.")
    ]
    assert len(hmm_paths) >= 10, hmm_paths
    for path_ in hmm_paths:
        _walk(result, path_)
    assert {"decision", "filtered_probabilities", "first_to_flip", "pnl_comparison"} <= set(
        result["hmm"]
    )


def test_skill_tables_cover_the_whole_companion_contract(skill_text: str) -> None:
    documented = set(_documented_paths(skill_text))
    assert set(COMPANION_KEYS) <= documented


def test_error_path_is_one_json_object(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["analyze", "--csv", str(tmp_path / "missing.csv"), "--json"])
    assert result.exit_code != 0
    parsed = json.loads(result.stdout)
    assert set(parsed) == {"error"}


# --- 4. The documented shell command -----------------------------------------------------


def test_first_fenced_bash_block_is_the_documented_command(skill_text: str) -> None:
    blocks = FENCED_BASH_RE.findall(skill_text)
    assert blocks, "SKILL.md needs at least one fenced bash block"
    command = blocks[0].strip().splitlines()[0]
    assert command.startswith("uv run")
    assert "hedgefund analyze" in command
    assert "--json" in command
