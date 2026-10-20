"""Tests for the paper-only sizing helper (plan.md Phase 7).

Everything is offline and synthetic: hand-built signal-file dicts, fake Alpaca clients.
The forbidden-call test reads the module source so no order-placing method can sneak in.
"""

import inspect
import json
import math
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hedgefund.paper import sizing
from hedgefund.paper.sizing import (
    DEFAULT_MAX_GROSS,
    PAPER_TRADING_URL,
    PROPOSAL_NOTE,
    AllocationProposal,
    ProposedOrder,
    paper_account_equity,
    paper_positions,
    propose_orders,
    render_proposal,
    target_allocation,
)
from hedgefund.regime.signal import position_from_signal
from hedgefund.report.json_contract import DISCLAIMER

EQUITY = 100_000.0
PRICES = {"SPY": 500.0, "TLT": 100.0, "GLD": 200.0, "XLE": 80.0}
# The non-paper host, derived rather than spelled out so the repo never carries it.
LIVE_URL = PAPER_TRADING_URL.replace("paper-", "")


def _entry(**overrides: object) -> dict:
    base = {
        "current_regime": "Bull",
        "signal": 0.4,
        "stay_probability": 0.9,
        "position": 1.0,
        "persistence": 0.9,
        "persistence_source": "stride",
        "last_close": 500.0,
        "date_end": "2026-10-17",
        "rows": 2500,
        "net_sharpe": 0.3,
        "error": None,
    }
    base.update(overrides)
    return base


def _signal(symbols: dict) -> dict:
    return {
        "as_of": "2026-10-17",
        "generated_at": "2026-10-17T21:00:00+00:00",
        "run_id": "run-abc",
        "universe": "us_etfs",
        "cost_profile": "us_etfs",
        "source": "alpaca",
        "rule": "sign",
        "symbols": symbols,
        "orders_placed": False,
        "note": "n",
        "framework": "f",
        "disclaimer": DISCLAIMER,
    }


THREE_LONG = _signal({"SPY": _entry(), "TLT": _entry(position=1.0), "GLD": _entry(position=1.0)})


# --- target_allocation ----------------------------------------------------------------


def test_target_allocation_equal_weights_across_active_symbols() -> None:
    weights, skipped = target_allocation(THREE_LONG)
    assert skipped == {}
    assert weights == {
        "GLD": pytest.approx(1 / 3),
        "SPY": pytest.approx(1 / 3),
        "TLT": pytest.approx(1 / 3),
    }


def test_target_allocation_flat_symbol_counts_in_denominator() -> None:
    signal = _signal(
        {"SPY": _entry(), "TLT": _entry(signal=0.0, position=0.0, current_regime="Sideways")}
    )
    weights, _ = target_allocation(signal)
    assert weights == {"SPY": pytest.approx(0.5), "TLT": 0.0}


def test_target_allocation_scales_when_gross_exceeds_cap() -> None:
    # Positions of +-2 (a conviction rule is not clamped upstream) give gross 2.0 before
    # the cap, so max_gross=1.0 halves every weight.
    # signal=None so the stored position is used (file rule "sign" matches the default).
    signal = _signal(
        {"SPY": _entry(signal=None, position=2.0), "TLT": _entry(signal=None, position=-2.0)}
    )
    weights, _ = target_allocation(signal, max_gross=1.0)
    assert sum(abs(w) for w in weights.values()) == pytest.approx(1.0)
    assert weights["SPY"] == pytest.approx(0.5)
    assert weights["TLT"] == pytest.approx(-0.5)


@settings(max_examples=60, deadline=None)
@given(
    positions=st.lists(
        st.floats(min_value=-3.0, max_value=3.0, allow_nan=False), min_size=1, max_size=8
    ),
    max_gross=st.floats(min_value=0.1, max_value=2.0, allow_nan=False),
)
def test_target_allocation_gross_never_exceeds_cap(
    positions: list[float], max_gross: float
) -> None:
    symbols = {f"S{i}": _entry(signal=None, position=p) for i, p in enumerate(positions)}
    weights, _ = target_allocation(_signal(symbols), max_gross=max_gross)
    assert sum(abs(w) for w in weights.values()) <= max_gross + 1e-12


def test_target_allocation_allow_short_false_floors_negative_to_flat() -> None:
    signal = _signal({"SPY": _entry(), "XLE": _entry(signal=-0.3, current_regime="Bear")})
    weights, skipped = target_allocation(signal, allow_short=False)
    assert weights == {"SPY": pytest.approx(0.5), "XLE": 0.0}
    assert skipped == {}


def test_target_allocation_allow_short_true_keeps_negative() -> None:
    signal = _signal({"SPY": _entry(), "XLE": _entry(signal=-0.3)})
    weights, _ = target_allocation(signal, allow_short=True)
    assert weights["XLE"] == pytest.approx(-0.5)


def test_target_allocation_skips_error_symbols_with_reason() -> None:
    signal = _signal(
        {
            "SPY": _entry(),
            "XLE": _entry(
                current_regime=None,
                signal=None,
                stay_probability=None,
                position=None,
                error="too little history",
            ),
        }
    )
    weights, skipped = target_allocation(signal)
    assert weights == {"SPY": pytest.approx(1.0)}
    assert skipped == {"XLE": "error: too little history"}


def test_target_allocation_skips_symbol_with_no_signal_and_no_error() -> None:
    signal = _signal({"SPY": _entry(), "XLE": _entry(signal=None, position=None)})
    weights, skipped = target_allocation(signal)
    assert "XLE" not in weights
    assert skipped["XLE"] == "no signal"


def test_target_allocation_derives_position_from_signal_under_every_rule() -> None:
    # The stored position (+1.0 from the sign rule) must be ignored: the signal is re-sized.
    signal = _signal({"SPY": _entry(position=1.0, signal=0.3, stay_probability=0.8)})
    for rule in ("sign", "conviction", "persistence", "long_only"):
        weights, _ = target_allocation(signal, rule=rule)
        assert weights["SPY"] == pytest.approx(position_from_signal(0.3, 0.8, rule))


def test_target_allocation_applies_requested_rule_not_stored_position() -> None:
    signal = _signal({"SPY": _entry(signal=-0.4, stay_probability=0.9, position=-1.0)})
    long_only, _ = target_allocation(signal, rule="long_only")
    assert long_only["SPY"] == 0.0
    sign, _ = target_allocation(signal, rule="sign")
    assert sign["SPY"] == -1.0


def test_propose_orders_long_only_never_proposes_a_short() -> None:
    signal = _signal({"SPY": _entry(signal=-0.4, stay_probability=0.9, position=-1.0)})
    proposal = propose_orders(signal, equity=EQUITY, prices=PRICES, rule="long_only")
    assert proposal.rule == "long_only"
    assert proposal.target_weights == {"SPY": 0.0}
    assert proposal.orders == ()
    as_sign = propose_orders(signal, equity=EQUITY, prices=PRICES, rule="sign")
    assert as_sign.orders[0].side == "sell"
    assert as_sign.orders[0].notional == pytest.approx(EQUITY)


def test_target_allocation_uses_stored_position_only_when_file_rule_matches() -> None:
    signal = _signal({"SPY": _entry(signal=None, position=-1.0)})  # file rule is "sign"
    weights, skipped = target_allocation(signal, rule="sign")
    assert weights == {"SPY": -1.0} and skipped == {}
    weights, skipped = target_allocation(signal, rule="long_only")
    assert weights == {}
    assert skipped["SPY"] == "position from file under rule 'sign', cannot re-derive"


def test_target_allocation_missing_stay_probability_skips_under_persistence() -> None:
    signal = _signal({"SPY": _entry(signal=0.3, stay_probability=None)})
    weights, skipped = target_allocation(signal, rule="persistence")
    assert weights == {}
    assert skipped == {"SPY": "no stay_probability"}


def test_target_allocation_missing_stay_probability_is_fine_under_other_rules() -> None:
    signal = _signal({"SPY": _entry(signal=0.3, stay_probability=None)})
    for rule in ("sign", "conviction", "long_only"):
        weights, skipped = target_allocation(signal, rule=rule)
        assert skipped == {}
        assert weights["SPY"] == pytest.approx(position_from_signal(0.3, 0.0, rule))


def test_target_allocation_rejects_unknown_rule_and_bad_cap() -> None:
    with pytest.raises(ValueError, match="rule"):
        target_allocation(THREE_LONG, rule="yolo")
    with pytest.raises(ValueError, match="max_gross"):
        target_allocation(THREE_LONG, max_gross=0.0)
    with pytest.raises(ValueError, match="symbols"):
        target_allocation({"as_of": "x"})


# --- propose_orders -------------------------------------------------------------------


def test_propose_orders_flat_start_buys_target_notional() -> None:
    proposal = propose_orders(THREE_LONG, equity=EQUITY, prices=PRICES)
    by_symbol = {o.symbol: o for o in proposal.orders}
    assert set(by_symbol) == {"SPY", "TLT", "GLD"}
    spy = by_symbol["SPY"]
    assert spy.side == "buy"
    assert spy.notional == pytest.approx(EQUITY / 3)
    assert spy.qty == pytest.approx(EQUITY / 3 / PRICES["SPY"])
    assert spy.current_weight == 0.0
    assert spy.target_weight == pytest.approx(1 / 3)
    assert spy.last_price == PRICES["SPY"]
    assert proposal.skipped == {}


def test_propose_orders_existing_position_trades_only_the_delta() -> None:
    # Holding 100 SPY at $500 = $50,000 = 50% of equity; target is 1/3 -> sell the excess.
    proposal = propose_orders(
        THREE_LONG, equity=EQUITY, prices=PRICES, current_positions={"SPY": 100.0}
    )
    spy = next(o for o in proposal.orders if o.symbol == "SPY")
    assert spy.side == "sell"
    assert spy.current_weight == pytest.approx(0.5)
    assert spy.notional == pytest.approx(EQUITY * (0.5 - 1 / 3))
    assert spy.qty == pytest.approx(EQUITY * (0.5 - 1 / 3) / PRICES["SPY"])


def test_propose_orders_sell_side_when_going_short_from_flat() -> None:
    signal = _signal({"XLE": _entry(signal=-0.3, current_regime="Bear")})
    proposal = propose_orders(signal, equity=EQUITY, prices=PRICES)
    (order,) = proposal.orders
    assert order.side == "sell"
    assert order.target_weight == -1.0
    assert order.notional == pytest.approx(EQUITY)


def test_propose_orders_skips_dust_below_min_trade_notional() -> None:
    # Already at target: exactly 1/3 of equity in SPY, so the delta is ~0.
    qty = EQUITY / 3 / PRICES["SPY"]
    proposal = propose_orders(
        THREE_LONG, equity=EQUITY, prices=PRICES, current_positions={"SPY": qty}
    )
    assert "SPY" not in {o.symbol for o in proposal.orders}
    assert proposal.skipped["SPY"].startswith("dust")


def test_propose_orders_skips_symbol_without_price() -> None:
    prices = {k: v for k, v in PRICES.items() if k != "GLD"}
    proposal = propose_orders(THREE_LONG, equity=EQUITY, prices=prices)
    assert "GLD" not in {o.symbol for o in proposal.orders}
    assert proposal.skipped["GLD"] == "no price"
    assert proposal.target_weights["GLD"] == pytest.approx(1 / 3)


def test_propose_orders_leaves_held_symbol_outside_watchlist_untouched() -> None:
    proposal = propose_orders(
        THREE_LONG, equity=EQUITY, prices=PRICES, current_positions={"XLE": 10.0}
    )
    assert "XLE" not in {o.symbol for o in proposal.orders}
    assert proposal.current_weights["XLE"] == pytest.approx(10 * 80 / EQUITY)
    assert "left untouched" in proposal.skipped["XLE"]


def test_propose_orders_floored_short_reason_mentions_allow_short() -> None:
    signal = _signal({"XLE": _entry(signal=-0.3, current_regime="Bear")})
    proposal = propose_orders(
        signal,
        equity=EQUITY,
        prices=PRICES,
        current_positions={"XLE": -100.0},
        allow_short=False,
    )
    (order,) = proposal.orders
    assert order.side == "buy"  # buy back the short to get flat
    assert order.target_weight == 0.0
    assert "allow_short" in order.reason


def test_propose_orders_rejects_non_positive_or_non_finite_equity() -> None:
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="equity"):
            propose_orders(THREE_LONG, equity=bad, prices=PRICES)


def test_propose_orders_rejects_bad_price_naming_symbol() -> None:
    with pytest.raises(ValueError, match="TLT"):
        propose_orders(THREE_LONG, equity=EQUITY, prices={**PRICES, "TLT": 0.0})
    with pytest.raises(ValueError, match="TLT"):
        propose_orders(THREE_LONG, equity=EQUITY, prices={**PRICES, "TLT": math.nan})


def test_propose_orders_gross_target_respects_cap() -> None:
    signal = _signal(
        {"SPY": _entry(signal=None, position=2.0), "TLT": _entry(signal=None, position=2.0)}
    )
    proposal = propose_orders(signal, equity=EQUITY, prices=PRICES, max_gross=DEFAULT_MAX_GROSS)
    assert sum(o.notional for o in proposal.orders) == pytest.approx(EQUITY)


def test_proposal_orders_placed_is_always_false() -> None:
    proposal = propose_orders(THREE_LONG, equity=EQUITY, prices=PRICES)
    assert proposal.orders_placed is False
    assert proposal.to_dict()["orders_placed"] is False
    assert AllocationProposal.__dataclass_fields__["orders_placed"].default is False


def test_proposal_to_dict_round_trips_through_json_with_sorted_keys() -> None:
    proposal = propose_orders(
        THREE_LONG, equity=EQUITY, prices=PRICES, current_positions={"SPY": 10.0}
    )
    payload = proposal.to_dict()
    back = json.loads(json.dumps(payload))
    assert back == payload
    assert list(payload) == sorted(payload)
    assert list(payload["target_weights"]) == sorted(payload["target_weights"])
    assert isinstance(payload["orders"], list)
    assert payload["orders"][0]["symbol"] == proposal.orders[0].symbol
    assert payload["note"] == PROPOSAL_NOTE
    assert payload["disclaimer"] == DISCLAIMER
    assert payload["as_of"] == "2026-10-17"
    assert payload["run_id"] == "run-abc"


def test_proposed_order_is_frozen() -> None:
    order = ProposedOrder("SPY", "buy", 1.0, 500.0, 0.5, 0.0, 500.0, "r")
    with pytest.raises(AttributeError):
        order.qty = 2.0  # type: ignore[misc]
    assert replace(order, qty=2.0).qty == 2.0


# --- render_proposal ------------------------------------------------------------------


def test_render_proposal_ends_with_note_and_contains_disclaimer() -> None:
    proposal = propose_orders(
        THREE_LONG, equity=EQUITY, prices={k: v for k, v in PRICES.items() if k != "GLD"}
    )
    text = render_proposal(proposal)
    assert text.endswith(PROPOSAL_NOTE)
    assert text.splitlines()[-2] == DISCLAIMER
    assert "SPY" in text and "buy" in text
    assert "Skipped:" in text and "GLD: no price" in text
    assert "orders_placed=False" in text


def test_render_proposal_with_no_orders_says_so() -> None:
    text = render_proposal(propose_orders(_signal({}), equity=EQUITY, prices=PRICES))
    assert "no orders" in text
    assert text.endswith(PROPOSAL_NOTE)


# --- read-only Alpaca access ----------------------------------------------------------


class _Account:
    def __init__(self, equity: str) -> None:
        self.equity = equity


class _Position:
    def __init__(self, symbol: str, qty: str, side: object) -> None:
        self.symbol, self.qty, self.side = symbol, qty, side


class _FakeClient:
    def __init__(self, base_url: str = PAPER_TRADING_URL) -> None:
        self._base_url = base_url
        self.calls: list[str] = []

    def get_account(self) -> _Account:
        self.calls.append("get_account")
        return _Account("12345.67")

    def get_all_positions(self) -> list[_Position]:
        self.calls.append("get_all_positions")
        return [_Position("SPY", "10", "long"), _Position("XLE", "4", "short")]


def test_paper_account_equity_reads_from_injected_client() -> None:
    client = _FakeClient()
    assert paper_account_equity(trading_client=client) == pytest.approx(12345.67)
    assert client.calls == ["get_account"]


def test_paper_positions_signs_shorts_negative() -> None:
    assert paper_positions(trading_client=_FakeClient()) == {"SPY": 10.0, "XLE": -4.0}


def test_paper_positions_accepts_enum_like_side_and_negative_qty() -> None:
    class Side:
        value = "PositionSide.SHORT"

    client = _FakeClient()
    client.get_all_positions = lambda: [_Position("XLE", "-4", Side())]  # type: ignore[method-assign]
    assert paper_positions(trading_client=client) == {"XLE": -4.0}


def test_live_endpoint_client_is_refused() -> None:
    live = _FakeClient(base_url=LIVE_URL)
    with pytest.raises(RuntimeError, match="live endpoint refused"):
        paper_account_equity(trading_client=live)
    with pytest.raises(RuntimeError, match="live endpoint refused"):
        paper_positions(trading_client=live)
    assert live.calls == []


def test_client_without_base_url_is_refused() -> None:
    class NoUrl:
        def get_account(self) -> _Account:
            return _Account("1")

    with pytest.raises(RuntimeError, match="live endpoint refused"):
        paper_account_equity(trading_client=NoUrl())


def test_missing_credentials_error_never_echoes_env_values() -> None:
    leaked_key = "PKLEAKEDKEYVALUE12345"
    environ = {"APCA_API_KEY_ID": leaked_key}  # secret missing
    with pytest.raises(RuntimeError) as info:
        paper_account_equity(environ=environ)
    message = str(info.value)
    assert leaked_key not in message
    assert "APCA_API_SECRET_KEY" in message


def test_credentials_fallback_reader_unpacks_env_pair() -> None:
    environ = {"APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s"}
    assert sizing._credentials(environ, reader=sizing._env_credentials) == ("k", "s")
    as_mapping = sizing._credentials(environ, reader=lambda e: {"key_id": "k", "secret_key": "s"})
    assert as_mapping == ("k", "s")


def test_credentials_reader_exception_is_wrapped_without_its_text() -> None:
    def bad_reader(environ: object) -> object:
        raise KeyError("SECRET-TEXT-FROM-READER")

    with pytest.raises(RuntimeError) as info:
        sizing._credentials({}, reader=bad_reader)
    assert "SECRET-TEXT-FROM-READER" not in str(info.value)


def test_paper_trading_url_matches_alpaca_paper_constant() -> None:
    enums = pytest.importorskip("alpaca.common.enums")
    assert enums.BaseURL.TRADING_PAPER.value == PAPER_TRADING_URL


# --- the hard rule: no order placement anywhere in the module -------------------------


FORBIDDEN_CALLS = (
    "submit_order(",
    "cancel_orders(",
    "cancel_order_by_id(",
    "close_position(",
    "close_all_positions(",
    "replace_order_by_id(",
)


def test_sizing_source_contains_no_order_placing_call() -> None:
    source = inspect.getsource(sizing)
    for forbidden in FORBIDDEN_CALLS:
        assert forbidden not in source, forbidden
