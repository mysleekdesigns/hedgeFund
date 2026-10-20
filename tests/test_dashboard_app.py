"""Tests for the Streamlit dashboard entry point ``dashboard/app.py``.

The app is exercised through ``streamlit.testing.v1.AppTest`` against the offline scan the
``dashboard_run`` fixture produces, so nothing here starts a server or opens a connection.
The pure chart helpers are imported from the file through ``importlib`` under a name other
than ``__main__`` so ``main()`` does not execute.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import REFUSED_ATTEMPTS

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from hedgefund.dashboard import JOBS_DIR_ENV  # noqa: E402
from hedgefund.dashboard.data import STATES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_FILE = REPO_ROOT / "dashboard" / "app.py"
# A full first render recomputes the walk-forward once; generous so a slow CI box passes.
APP_TIMEOUT_SECONDS = 120

BADGE_FROM_JSON = "From the scan's result JSON"
BADGE_RECOMPUTED = "Recomputed from cached prices for these controls"
DISCLAIMER_SNIPPET = "historical, not forward-looking"


@pytest.fixture(scope="module")
def app_module():
    """``dashboard/app.py`` imported as a plain module (its ``__main__`` guard stays closed)."""
    spec = importlib.util.spec_from_file_location("dashboard_app_under_test", APP_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _app(monkeypatch: pytest.MonkeyPatch, jobs_dir: Path) -> AppTest:
    monkeypatch.setenv(JOBS_DIR_ENV, str(jobs_dir))
    return AppTest.from_file(str(APP_FILE), default_timeout=APP_TIMEOUT_SECONDS)


def _texts(at: AppTest) -> list[str]:
    return (
        [m.value for m in at.markdown]
        + [c.value for c in at.caption]
        + [t.value for t in at.text]
        + [t.value for t in at.title]
    )


def _badges(at: AppTest) -> list[str]:
    return [i.value for i in at.info] + [s.value for s in at.success]


# ----------------------------------------------------------------------------- AppTest runs


def test_app_renders_cached_scan(monkeypatch: pytest.MonkeyPatch, dashboard_run: dict) -> None:
    at = _app(monkeypatch, dashboard_run["jobs_dir"]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert list(at.selectbox(key="symbol").options) == list(dashboard_run["symbols"])
    assert at.selectbox(key="run_id").value == dashboard_run["run_id"]
    assert any(DISCLAIMER_SNIPPET in t for t in _texts(at))


def test_default_controls_use_json_then_window_change_recomputes(
    monkeypatch: pytest.MonkeyPatch, dashboard_run: dict
) -> None:
    at = _app(monkeypatch, dashboard_run["jobs_dir"]).run()
    assert not at.exception
    assert BADGE_FROM_JSON in _badges(at)
    assert BADGE_RECOMPUTED not in _badges(at)

    window = at.selectbox(key="window")
    other = next(opt for opt in window.options if opt != window.value)
    at = window.set_value(other).run()
    assert not at.exception, [e.value for e in at.exception]
    assert BADGE_RECOMPUTED in _badges(at)
    assert BADGE_FROM_JSON not in _badges(at)


def test_rolling_refit_reruns(monkeypatch: pytest.MonkeyPatch, dashboard_run: dict) -> None:
    at = _app(monkeypatch, dashboard_run["jobs_dir"]).run()
    assert at.radio(key="refit").value == "expanding"
    at = at.radio(key="refit").set_value("rolling").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.number_input(key="rolling_bars").value >= 1
    assert BADGE_RECOMPUTED in _badges(at)


def test_empty_jobs_dir_warns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    at = _app(monkeypatch, tmp_path).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.warning, "expected a warning telling the user to run a scan first"
    assert "hedgefund scan" in at.warning[0].value
    assert len(at.selectbox) == 0  # the sidebar never rendered


def test_app_makes_no_connection_attempt(
    monkeypatch: pytest.MonkeyPatch, dashboard_run: dict
) -> None:
    REFUSED_ATTEMPTS.clear()
    at = _app(monkeypatch, dashboard_run["jobs_dir"]).run()
    assert not at.exception
    assert REFUSED_ATTEMPTS == []


def test_views_render_expected_widgets(
    monkeypatch: pytest.MonkeyPatch, dashboard_run: dict
) -> None:
    at = _app(monkeypatch, dashboard_run["jobs_dir"]).run()
    assert not at.exception
    assert at.slider(key="nstep").value == 0
    assert at.button(key="play").value is False
    assert at.checkbox(key="log_scale").value is False
    captions = " ".join(c.value for c in at.caption)
    assert "block-bootstrap" in captions
    assert "not a forecast" in captions
    assert "random-sign null" in captions


def _band_caption(at: AppTest) -> str:
    return next(c.value for c in at.caption if "Shaded band" in c.value)


def test_full_path_count_band_is_the_json_null_draw(
    monkeypatch: pytest.MonkeyPatch, dashboard_run: dict
) -> None:
    """The fixture scan has 20 paths, the slider minimum, so the band uses all of them."""
    at = _app(monkeypatch, dashboard_run["jobs_dir"]).run()
    assert not at.exception
    assert all(s.key != "null_paths" for s in at.slider)
    caption = _band_caption(at)
    assert "same draw the JSON null percentile ranks against" in caption
    assert "20 of the scan's 20 paths" in caption


def test_band_caption_carries_run_seed(
    monkeypatch: pytest.MonkeyPatch, dashboard_run: dict
) -> None:
    import json

    manifest = json.loads((dashboard_run["run_dir"] / "manifest.json").read_text())
    seed = int(manifest["params"]["seed"])
    at = _app(monkeypatch, dashboard_run["jobs_dir"]).run()
    assert not at.exception
    assert f"seed {seed}" in _band_caption(at)


def test_json_null_percentile_metric_shown(
    monkeypatch: pytest.MonkeyPatch, dashboard_run: dict
) -> None:
    at = _app(monkeypatch, dashboard_run["jobs_dir"]).run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert "Random-sign null percentile (net, JSON)" in labels
    assert labels["Random-sign null percentile (net, JSON)"] != "n/a"


# ---------------------------------------------------------------------------- pure helpers


def test_null_band_caption_marks_same_draw_only_at_full_count(app_module) -> None:
    full = app_module.null_band_caption(20, 20, 7)
    assert "20 of the scan's 20 paths, seed 7" in full
    assert app_module.SAME_DRAW_TEXT in full
    partial = app_module.null_band_caption(200, 1000, 7)
    assert "200 of the scan's 1000 paths, seed 7" in partial
    assert app_module.SAME_DRAW_TEXT not in partial


def test_json_null_percentile_guards_missing_keys(app_module) -> None:
    result = {"backtest": {"null": {"random_sign": {"percentile_net": 87.5}}}}
    assert app_module.json_null_percentile(result) == 87.5
    assert app_module.json_null_percentile({}) is None
    assert app_module.json_null_percentile({"backtest": None}) is None
    none_result = {"backtest": {"null": {"random_sign": {"percentile_net": None}}}}
    assert app_module.json_null_percentile(none_result) is None


def test_ribbon_segments_collapses_runs(app_module) -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    labels = pd.Series([0, 0, 1, 2, 2, 2], index=dates)
    seg = app_module.ribbon_segments(labels)
    assert list(seg.columns) == ["start", "end", "code", "state"]
    assert seg["code"].tolist() == [0, 1, 2]
    assert seg["state"].tolist() == [STATES[0], STATES[1], STATES[2]]
    assert seg["start"].tolist() == [dates[0], dates[2], dates[3]]
    # each run ends where the next begins; the last ends on the final date
    assert seg["end"].tolist() == [dates[2], dates[3], dates[5]]


def test_ribbon_segments_empty(app_module) -> None:
    seg = app_module.ribbon_segments(pd.Series([], dtype=int))
    assert seg.empty
    assert list(seg.columns) == ["start", "end", "code", "state"]


def test_heatmap_frame_labels_and_order(app_module) -> None:
    matrix = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.05, 0.15, 0.8]])
    lower = matrix - 0.05
    upper = matrix + 0.05
    frame = app_module.heatmap_frame(matrix, lower, upper)
    assert len(frame) == 9
    assert frame["from_state"].tolist()[:3] == [STATES[0]] * 3
    assert frame["to_state"].tolist()[:3] == list(STATES)
    assert frame.loc[0, "label"] == "0.80 [0.75, 0.85]"
    assert frame["p"].to_numpy().reshape(3, 3).tolist() == matrix.tolist()
    plain = app_module.heatmap_frame(matrix)
    assert plain.loc[4, "label"] == "0.60"
    assert plain["lower"].isna().all()


def test_band_frame_aligns_curves_and_band(app_module) -> None:
    idx = pd.bdate_range("2024-01-01", periods=4)
    curves = {
        "strategy_net": pd.Series([1.0, 1.1, 1.2, 1.3], index=idx),
        "buy_and_hold": pd.Series([1.0, 0.9, 1.0, 1.1], index=idx),
    }
    lower = pd.Series([0.9, 0.9, 0.95, 0.95], index=idx)
    upper = pd.Series([1.1, 1.2, 1.25, 1.3], index=idx)
    frame = app_module.band_frame(curves, lower, upper)
    assert list(frame.columns) == [
        "date",
        "strategy_net",
        "buy_and_hold",
        "null_lower",
        "null_upper",
    ]
    assert len(frame) == 4
    assert frame["date"].tolist() == list(idx)
    assert frame["null_upper"].tolist() == upper.tolist()
    assert frame["strategy_net"].tolist() == [1.0, 1.1, 1.2, 1.3]


def test_band_frame_empty_curves(app_module) -> None:
    frame = app_module.band_frame({}, None, None)
    assert frame.empty
    assert "date" in frame.columns


def test_nstep_max_bounds(app_module) -> None:
    assert app_module.nstep_max(None) == app_module.NSTEP_FALLBACK
    assert app_module.nstep_max(2) == app_module.NSTEP_MIN_MAX
    assert app_module.nstep_max(1000) == app_module.NSTEP_MAX_CAP
    assert app_module.nstep_max(15) == 30
