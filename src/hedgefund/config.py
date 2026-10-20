"""Small YAML loader for ``config/default.yaml``.

Config over flags: the YAML holds every default, CLI flags override, and nothing is
hard-coded twice (``.claude/rules/data-sources.md``).
"""

from pathlib import Path

import yaml

# src/hedgefund/config.py -> src/hedgefund -> src -> repo root
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config file as a dict. ``None`` means the repo default.

    Raises ``ValueError`` when the file is missing or does not hold a mapping.
    """
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    if not config_path.is_file():
        raise ValueError(f"config file not found: {config_path}")
    with config_path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config file {config_path} must hold a YAML mapping, got {type(data)}")
    return data
