"""Text-level checks on ``pine/markov_regime_strategy.pine`` (plan.md Fix 2).

Pine cannot run offline, so these tests pin the properties that make the strategy honest:
orders are processed at the next bar's open, every order and label sits inside a
``barstate.isconfirmed`` guard, and nothing is back-dated. The Bar Replay check on BTCUSDT
and SPY daily remains a manual step.
"""

import re
from pathlib import Path

import pytest

PINE_PATH = Path(__file__).resolve().parents[1] / "pine" / "markov_regime_strategy.pine"
CONFIRMED_TOKEN = "barstate.isconfirmed"
# Every call that acts on the chart or the tester must sit inside the confirmed guard.
GUARDED_CALLS = (
    "strategy.entry(",
    "strategy.close_all(",
    "strategy.close(",
    "label.new(",
    "last_confirmed_regime :=",
)
LABEL_MOVERS = ("label.set_x(", "label.set_xloc(", "label.set_point(", "label.set_x_y(")
# The token must not be negated and must not be one side of an ``or``.
CONFIRMED_TOKEN_RE = re.compile(r"(?<!not )(?<!or )barstate\.isconfirmed(?!\s+or\b)")
NEXT_OPEN_PATTERN = re.compile(r"process_orders_on_close\s*=\s*false")
ON_CLOSE_PATTERN = re.compile(r"process_orders_on_close\s*=\s*true")


@pytest.fixture(scope="module")
def source() -> str:
    return PINE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def lines(source: str) -> list[str]:
    return source.splitlines()


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _strategy_call(source: str) -> str:
    """Return the full ``strategy(...)`` declaration, which may span several lines."""
    match = re.search(r"^strategy\(.*?\)\s*$", source, flags=re.MULTILINE | re.DOTALL)
    assert match is not None, "no strategy( declaration found"
    return match.group(0)


def _enclosing_if_has_token(lines: list[str], index: int) -> bool:
    """Walk outward through enclosing ``if`` blocks until one mentions the confirmed token."""
    current_indent = _indent(lines[index])
    for j in range(index - 1, -1, -1):
        candidate = lines[j]
        if not candidate.strip() or candidate.lstrip().startswith("//"):
            continue
        candidate_indent = _indent(candidate)
        if candidate_indent >= current_indent:
            continue
        if candidate.lstrip().startswith("if "):
            if CONFIRMED_TOKEN_RE.search(candidate):
                return True
            current_indent = candidate_indent
            continue
        # A less-indented non-if line (for, else, top level) ends the enclosing block chain.
        if candidate_indent == 0:
            return False
        current_indent = candidate_indent
    return False


def test_first_non_empty_line_is_version_5(lines: list[str]) -> None:
    first = next(line for line in lines if line.strip())
    assert first.strip() == "//@version=5"


def test_strategy_call_processes_orders_on_next_open(source: str) -> None:
    call = _strategy_call(source)
    assert NEXT_OPEN_PATTERN.search(call) is not None
    assert "overlay = true" in call


def test_strategy_call_does_not_recalculate_every_tick(source: str) -> None:
    assert "calc_on_every_tick = false" in _strategy_call(source)
    assert "calc_on_order_fills = false" in _strategy_call(source)


def test_process_orders_on_close_true_is_absent(source: str) -> None:
    assert ON_CLOSE_PATTERN.search(source) is None


def test_every_order_and_label_sits_inside_confirmed_guard(lines: list[str]) -> None:
    offenders: list[str] = []
    seen = 0
    for i, line in enumerate(lines):
        code = line.split("//", 1)[0]
        if not any(call in code for call in GUARDED_CALLS):
            continue
        seen += 1
        if _indent(line) == 0 or not _enclosing_if_has_token(lines, i):
            offenders.append(f"{i + 1}: {line.strip()}")
    assert seen > 0, "expected at least one guarded call"
    assert offenders == [], "calls outside a barstate.isconfirmed guard:\n" + "\n".join(offenders)


def test_no_label_is_back_dated(source: str) -> None:
    assert "label.new(bar_index -" not in source


def test_no_label_is_moved_after_creation(source: str) -> None:
    assert not any(mover in source for mover in LABEL_MOVERS)


@pytest.mark.parametrize(
    "guard",
    ["if not barstate.isconfirmed", "if barstate.isconfirmed or true", "if regime == 1"],
)
def test_guard_walker_rejects_weak_guards(guard: str) -> None:
    sample = [guard, '    label.new(bar_index, high, "x")']
    assert not _enclosing_if_has_token(sample, 1)


def test_guard_walker_accepts_nested_confirmed_guard() -> None:
    sample = [
        "if barstate.isconfirmed and held",
        "    if show",
        '        label.new(bar_index, high, "x")',
    ]
    assert _enclosing_if_has_token(sample, 2)


def test_strategy_has_long_and_short_directions(source: str) -> None:
    assert "strategy.long" in source
    assert "strategy.short" in source


def test_disclaimer_present(source: str) -> None:
    assert "Backtests are historical, not forward-looking." in source
