"""Universe files: the symbol lists a scan runs over and the cost profile they trade at.

A universe is a small YAML mapping under ``config/universes/`` with three keys:
``name``, ``cost_profile`` (a key of ``costs.profiles`` in ``config/default.yaml``) and
``symbols`` (a non-empty list of strings in the data source's symbol style). Code never
embeds tickers (``.claude/rules/data-sources.md``); the scan reads them from here.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

# src/hedgefund/jobs/universe.py -> jobs -> hedgefund -> src -> repo root
UNIVERSES_DIR = Path(__file__).resolve().parents[3] / "config" / "universes"
UNIVERSE_SUFFIX = ".yaml"
REQUIRED_KEYS = ("name", "cost_profile", "symbols")


@dataclass(frozen=True)
class Universe:
    """A named symbol list and the cost profile every symbol in it is charged."""

    name: str
    cost_profile: str
    symbols: tuple[str, ...]
    path: Path


def resolve_universe_path(name_or_path: str | Path) -> Path:
    """An existing file path is returned as-is; otherwise ``UNIVERSES_DIR/<name>.yaml``.

    Raises ``ValueError`` naming both candidates when neither exists.
    """
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    by_name = UNIVERSES_DIR / f"{name_or_path}{UNIVERSE_SUFFIX}"
    if by_name.is_file():
        return by_name
    raise ValueError(
        f"universe {str(name_or_path)!r} not found: no file at {candidate} and no "
        f"{by_name.name} under {UNIVERSES_DIR}"
    )


def load_universe(name_or_path: str | Path) -> Universe:
    """Load a universe by name (under ``config/universes/``) or by file path.

    Raises ``ValueError`` naming the missing key, a ``symbols`` value that is not a
    non-empty list of strings, or a file that does not hold a mapping.
    """
    path = resolve_universe_path(name_or_path)
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"universe file {path} must hold a YAML mapping, got {type(data).__name__}"
        )
    for key in REQUIRED_KEYS:
        if key not in data:
            raise ValueError(f"universe file {path} is missing required key {key!r}")
    symbols = data["symbols"]
    if (
        not isinstance(symbols, list)
        or not symbols
        or not all(isinstance(s, str) and s for s in symbols)
    ):
        raise ValueError(
            f"universe file {path}: 'symbols' must be a non-empty list of strings, got {symbols!r}"
        )
    for key in ("name", "cost_profile"):
        if not isinstance(data[key], str) or not data[key]:
            raise ValueError(f"universe file {path}: {key!r} must be a non-empty string")
    return Universe(
        name=data["name"],
        cost_profile=data["cost_profile"],
        symbols=tuple(symbols),
        path=path,
    )
