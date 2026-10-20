"""Static checks on ``pine/markov_regime.pine`` (plan.md Phase 6, Fix 1 and Fix 2).

Pine cannot be compiled offline, so these tests read the script as text and assert the
properties the plan requires: every label is drawn under a ``barstate.isconfirmed`` guard,
the default label sits on ``bar_index`` with no back-dating, the hindsight and stride
toggles default to off, counts mutate only on confirmed bars, and the original tables,
ribbon and stationary iteration survive. The two guard-walk helpers are plain functions so
the strategy-variant test can reuse them (``tests`` is not a package; pytest's prepend
import mode puts ``tests/`` on ``sys.path`` so ``from test_pine import ...`` works).
"""

import re
from pathlib import Path

import pytest

PINE_PATH = Path(__file__).resolve().parents[1] / "pine" / "markov_regime.pine"
CONFIRMED_GUARD = "barstate.isconfirmed"
# The token must not be negated and must not be one side of an ``or``.
CONFIRMED_GUARD_RE = re.compile(r"(?<!not )(?<!or )barstate\.isconfirmed(?!\s+or\b)")
LABEL_MOVERS = ("label.set_x(", "label.set_xloc(", "label.set_point(", "label.set_x_y(")
LABEL_STATE_UPDATE = "last_lbl_regime :="
HINDSIGHT_TOGGLE = "show_hindsight_labels"
LABEL_CALL = "label.new("
COUNT_MUTATION = "array.set(counts"  # matches counts_daily and counts_stride, not P or C
MIN_TABLES = 3  # banner, matrix, stationary
LIVE_LABEL_X = re.compile(r"label\.new\(\s*bar_index\s*,")
BACKDATED_LABEL_X = re.compile(r"label\.new\(\s*bar_index\s*-")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_code(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("//")


def code_lines(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, line)`` for every non-blank, non-comment line."""
    return [(i + 1, line) for i, line in enumerate(text.splitlines()) if _is_code(line)]


def enclosing_block_headers(lines: list[str], index: int) -> list[str]:
    """Walk upward from ``lines[index]`` collecting the header line of every enclosing block.

    A header is the nearest earlier code line with strictly smaller indentation; the walk
    repeats from that line until indentation zero. An ``else`` header is returned as-is, so
    a call inside an ``else`` branch is *not* credited with the ``if`` that owns the branch.
    """
    headers: list[str] = []
    indent = _indent(lines[index])
    for j in range(index - 1, -1, -1):
        line = lines[j]
        if not _is_code(line):
            continue
        line_indent = _indent(line)
        if line_indent < indent:
            headers.append(line)
            indent = line_indent
            if indent == 0:
                break
    return headers


def _is_confirmed_guard(header: str) -> bool:
    """An ``if`` header that positively requires ``barstate.isconfirmed``.

    ``if not barstate.isconfirmed`` and ``if barstate.isconfirmed or ...`` do not count: the
    first inverts the guard and the second lets the branch run on an unconfirmed bar.
    """
    return header.lstrip().startswith("if ") and bool(CONFIRMED_GUARD_RE.search(header))


def _guarded_by_confirmed(lines: list[str], index: int) -> bool:
    return any(_is_confirmed_guard(header) for header in enclosing_block_headers(lines, index))


def _calls_are_confirmed_guarded(text: str, needle: str) -> bool:
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if _is_code(line) and needle in line]
    return bool(hits) and all(_guarded_by_confirmed(lines, i) for i in hits)


def labels_are_confirmed_guarded(text: str) -> bool:
    """True when every ``label.new(`` call sits under an ``if ... barstate.isconfirmed``."""
    return _calls_are_confirmed_guarded(text, LABEL_CALL)


def count_updates_are_confirmed_guarded(text: str) -> bool:
    """True when every ``array.set(counts...`` call sits under ``if ... barstate.isconfirmed``."""
    return _calls_are_confirmed_guarded(text, COUNT_MUTATION)


@pytest.fixture(scope="module")
def pine_text() -> str:
    return PINE_PATH.read_text(encoding="utf-8")


def _label_lines(text: str) -> list[str]:
    return [line for _, line in code_lines(text) if LABEL_CALL in line]


def _input_bool_lines(text: str, keyword: str) -> list[str]:
    return [line for _, line in code_lines(text) if "input.bool(" in line and keyword in line]


# --- version and declaration --------------------------------------------------


def test_first_line_is_version_5(pine_text: str) -> None:
    first = next(line for line in pine_text.splitlines() if line.strip())
    assert first.strip() == "//@version=5"


def test_declares_overlay_indicator(pine_text: str) -> None:
    indicator_lines = [line for _, line in code_lines(pine_text) if line.startswith("indicator(")]
    assert len(indicator_lines) == 1
    assert "overlay = true" in indicator_lines[0]


def test_no_strategy_call(pine_text: str) -> None:
    assert not any(line.lstrip().startswith("strategy(") for _, line in code_lines(pine_text))


# --- labels: Fix 2 ----------------------------------------------------------


def test_every_label_is_under_confirmed_guard(pine_text: str) -> None:
    assert labels_are_confirmed_guarded(pine_text)


def test_default_label_is_on_current_bar(pine_text: str) -> None:
    live = [line for line in _label_lines(pine_text) if "hindsight" not in line]
    assert len(live) == 1
    assert LIVE_LABEL_X.search(live[0])
    assert "bar_index -" not in live[0]


def test_only_hindsight_label_is_back_dated(pine_text: str) -> None:
    back_dated = [line for line in _label_lines(pine_text) if BACKDATED_LABEL_X.search(line)]
    assert len(back_dated) == 1
    assert "hindsight" in back_dated[0]


def test_no_label_is_moved_after_creation(pine_text: str) -> None:
    """A label drawn at ``bar_index`` and then moved with ``label.set_x`` is back-dated too."""
    movers = [line for _, line in code_lines(pine_text) if any(m in line for m in LABEL_MOVERS)]
    assert movers == []


def test_hindsight_label_is_inside_the_toggle_branch(pine_text: str) -> None:
    lines = pine_text.splitlines()
    hits = [i for i, line in enumerate(lines) if BACKDATED_LABEL_X.search(line)]
    assert len(hits) == 1
    headers = enclosing_block_headers(lines, hits[0])
    assert any(h.lstrip().startswith("if ") and HINDSIGHT_TOGGLE in h for h in headers)


def test_label_state_update_is_under_confirmed_guard(pine_text: str) -> None:
    assert _calls_are_confirmed_guarded(pine_text, LABEL_STATE_UPDATE)


def test_hindsight_label_text_is_tagged(pine_text: str) -> None:
    hindsight = [line for line in _label_lines(pine_text) if "hindsight" in line]
    assert len(hindsight) == 1
    assert '"hindsight' in hindsight[0]


def test_live_label_keeps_flip_offset_in_text(pine_text: str) -> None:
    assert 'flip was at bar -"' in pine_text
    assert "str.tostring(flip_off)" in pine_text or "str.tostring(min_regime_hold - 1)" in pine_text


def test_hindsight_toggle_defaults_to_false(pine_text: str) -> None:
    lines = _input_bool_lines(pine_text, "hindsight")
    assert len(lines) == 1
    assert "input.bool(false" in lines[0]


# --- stride counting: Fix 1 -------------------------------------------------


def test_stride_toggle_defaults_to_false(pine_text: str) -> None:
    lines = _input_bool_lines(pine_text, "stride")
    assert len(lines) == 1
    assert "input.bool(false" in lines[0]


def test_stride_counting_samples_one_window_apart(pine_text: str) -> None:
    code = "\n".join(line.split("//", 1)[0] for _, line in code_lines(pine_text))
    assert "% lookback_window" in code
    assert "regime[lookback_window]" in code
    assert "regime[lookback_window -" not in code and "regime[lookback_window+" not in code


def test_count_updates_are_under_confirmed_guard(pine_text: str) -> None:
    assert count_updates_are_confirmed_guarded(pine_text)


def test_matrix_header_names_the_estimator(pine_text: str) -> None:
    assert '"stride"' in pine_text
    assert '"overlapping"' in pine_text


# --- original furniture kept --------------------------------------------------


def test_keeps_three_tables(pine_text: str) -> None:
    assert pine_text.count("table.new(") >= MIN_TABLES


def test_keeps_ribbon_and_stationary_power(pine_text: str) -> None:
    assert "bgcolor(" in pine_text
    assert "stationary_power" in pine_text
    assert "matmul_3x3" in pine_text


def test_header_credits_original_authors(pine_text: str) -> None:
    assert "Roan (@RohOnChain)" in pine_text
    assert "Lewis Jackson" in pine_text


# --- the guard-walk helper itself ---------------------------------------------


def test_helper_rejects_unguarded_label() -> None:
    text = 'if not na(regime)\n    label.new(bar_index, high, "x")\n'
    assert not labels_are_confirmed_guarded(text)


def test_helper_accepts_nested_guard() -> None:
    text = (
        'if barstate.isconfirmed and held\n    if show\n        label.new(bar_index, high, "x")\n'
    )
    assert labels_are_confirmed_guarded(text)


def test_helper_rejects_else_branch_of_guard() -> None:
    text = 'if barstate.isconfirmed\n    x = 1\nelse\n    label.new(bar_index, high, "x")\n'
    assert not labels_are_confirmed_guarded(text)


def test_helper_rejects_negated_guard() -> None:
    text = 'if not barstate.isconfirmed\n    label.new(bar_index, high, "x")\n'
    assert not labels_are_confirmed_guarded(text)


def test_helper_rejects_or_guard() -> None:
    text = 'if barstate.isconfirmed or true\n    label.new(bar_index, high, "x")\n'
    assert not labels_are_confirmed_guarded(text)


def test_helper_rejects_or_before_guard() -> None:
    text = 'if held or barstate.isconfirmed\n    label.new(bar_index, high, "x")\n'
    assert not labels_are_confirmed_guarded(text)


def test_helper_accepts_and_guard() -> None:
    text = 'if barstate.isconfirmed and held\n    label.new(bar_index, high, "x")\n'
    assert labels_are_confirmed_guarded(text)


def test_helper_requires_at_least_one_call() -> None:
    assert not labels_are_confirmed_guarded("x = 1\n")
    assert not count_updates_are_confirmed_guarded("x = 1\n")


# --- stationary table reads the converged row through the display remap --------


def test_stationary_vector_is_read_through_the_display_remap(pine_text: str) -> None:
    """Row 0 of the converged matrix is in internal order (0=Side, 1=Bull, 2=Bear). The
    original overlay read M[0], M[1], M[2] as Bull, Bear, Side and printed the Sideways
    share under BULL; the three reads must go through ``disp_to_internal``."""
    reads = {
        name: line
        for _, line in code_lines(pine_text)
        for name in ("stat_bull", "stat_bear", "stat_side")
        if line.lstrip().startswith(f"{name} =")
    }
    assert set(reads) == {"stat_bull", "stat_bear", "stat_side"}
    assert "array.get(M, disp_to_internal(0))" in reads["stat_bull"]
    assert "array.get(M, disp_to_internal(1))" in reads["stat_bear"]
    assert "array.get(M, disp_to_internal(2))" in reads["stat_side"]
    assert "disp_to_internal(d) => d == 0 ? 1 : d == 1 ? 2 : 0" in pine_text
