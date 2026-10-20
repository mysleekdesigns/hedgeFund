"""Offline data layer for the Streamlit dashboard under ``dashboard/``.

The Streamlit app itself lives outside the package (``dashboard/app.py``) and imports
``hedgefund.dashboard.data`` for everything that touches disk or maths, so the data layer
is testable without Streamlit. Nothing here opens a network connection: runs come from
``./jobs/<run_id>/`` and prices from the parquet cache the scan wrote.

Backtests are historical, not forward-looking.
"""

from pathlib import Path

from hedgefund.jobs.manifest import REPO_ROOT

# Environment variable `hedgefund dashboard` sets so the Streamlit subprocess reads the same
# jobs directory the user passed on the command line (Streamlit swallows script argv).
JOBS_DIR_ENV = "HEDGEFUND_JOBS_DIR"
# The Streamlit entry point; outside ``src/`` because plan.md section 3 keeps the app there.
APP_PATH: Path = REPO_ROOT / "dashboard" / "app.py"

__all__ = ["APP_PATH", "JOBS_DIR_ENV"]
