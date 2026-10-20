"""Paper-only sizing: signal file in, target allocation and proposed orders out.

The daily job (``hedgefund.jobs``) writes a signal file with one entry per watchlist
symbol. This module turns those entries into equal-weight target weights, diffs them
against the current paper positions and prints the orders that *would* rebalance the
account. It never submits anything: order placement is a separate, explicit human
action outside hedgeFund. The Alpaca order methods (submit_order, replace_order_by_id,
cancel_orders, cancel_order_by_id, close_position, close_all_positions) are forbidden
in this module and a test greps the source to make sure none is called.

Reads from Alpaca go through ``paper_account_equity`` and ``paper_positions`` only, both
read-only, both refusing any client whose base url is not the paper endpoint. API keys
come from the environment at call time and are never logged, printed or included in an
error message.

Allocation is deliberately naive: equal weight across the watchlist, one position per
symbol from the walk-forward position rule, a gross cap and an optional long-only floor.
No optimisation, no risk model, no volatility targeting. Backtests are historical, not
forward-looking (``DISCLAIMER``).
"""

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

from hedgefund.jobs.manifest import json_ready
from hedgefund.regime.signal import POSITION_RULES, position_from_signal
from hedgefund.report.json_contract import DISCLAIMER

PAPER_TRADING_URL = "https://paper-api.alpaca.markets"
PROPOSAL_NOTE = (
    "Proposed orders only. Nothing was submitted. Placing an order is a separate, "
    "explicit human action outside hedgeFund."
)

DEFAULT_MAX_GROSS = 1.0  # config paper.max_gross: total |weight| cap, 1.0 = unlevered
DEFAULT_ALLOW_SHORT = True  # config paper.allow_short: a negative position may be held short
DEFAULT_MIN_TRADE_NOTIONAL = 1.0  # config paper.min_trade_notional: skip dust orders under $1
DEFAULT_RULE = "sign"  # config paper.rule: the Phase 1 position rule, matches the daily job

KEY_ID_ENV = "APCA_API_KEY_ID"  # alpaca-py's own variable names, read at call time only
SECRET_KEY_ENV = "APCA_API_SECRET_KEY"
CREDENTIALS_MESSAGE = (  # never interpolate a value into this: it must not leak a key
    f"Alpaca credentials missing or unreadable: set {KEY_ID_ENV} and {SECRET_KEY_ENV} "
    "in the environment (paper keys only)"
)
LIVE_ENDPOINT_MESSAGE = "live endpoint refused"

SIDE_BUY = "buy"
SIDE_SELL = "sell"
PERCENT = 100  # weights are printed as percentages


@dataclass(frozen=True)
class ProposedOrder:
    """One rebalancing trade that *would* move ``symbol`` from current to target weight."""

    symbol: str
    side: str  # "buy" or "sell"
    qty: float  # > 0, fractional shares allowed on paper
    notional: float  # > 0, qty * last_price
    target_weight: float
    current_weight: float
    last_price: float
    reason: str


@dataclass(frozen=True)
class AllocationProposal:
    """Target weights, current weights and the proposed orders between them.

    ``orders_placed`` is always ``False``: this object is a printout, not an action.
    """

    equity: float
    as_of: str | None
    run_id: str | None
    rule: str
    max_gross: float
    allow_short: bool
    target_weights: dict[str, float]
    current_weights: dict[str, float]
    orders: tuple[ProposedOrder, ...]
    skipped: dict[str, str]  # symbol -> why (error, no signal, no price, dust)
    note: str = PROPOSAL_NOTE
    disclaimer: str = DISCLAIMER
    orders_placed: bool = False

    def to_dict(self) -> dict:
        """JSON-ready dict: orders as a list of dicts, every mapping's keys sorted."""
        return _sorted_keys(json_ready(asdict(self)))


def _sorted_keys(value: object) -> object:
    if isinstance(value, dict):
        return {k: _sorted_keys(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sorted_keys(v) for v in value]
    return value


def _symbol_entries(signal: Mapping) -> dict[str, Mapping]:
    if not isinstance(signal, Mapping) or not isinstance(signal.get("symbols"), Mapping):
        raise ValueError(
            "signal must be a signal-file dict with a 'symbols' mapping, "
            f"got {type(signal).__name__}"
        )
    entries = signal["symbols"]
    for symbol, entry in entries.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"signal['symbols'][{symbol!r}] must be a dict, got {entry!r}")
    return {str(symbol): entries[symbol] for symbol in sorted(entries)}


def _validate_sizing_params(rule: str, max_gross: float) -> None:
    if rule not in POSITION_RULES:
        raise ValueError(f"rule must be one of {POSITION_RULES}, got {rule!r}")
    if not isinstance(max_gross, int | float) or not math.isfinite(max_gross) or max_gross <= 0:
        raise ValueError(f"max_gross must be a finite number > 0, got {max_gross!r}")


STAY_RULES = ("persistence",)  # the only rule position_from_signal reads stay_probability for
IGNORED_STAY = 0.0  # passed under every other rule, which never reads it


def _entry_position(
    entry: Mapping, file_rule: object, rule: str
) -> tuple[float | None, str | None]:
    """``(position, None)`` derived under ``rule``, or ``(None, reason)`` when it cannot be.

    The file's ``signal`` is always re-sized under ``rule`` so the printed rule is the
    one that was applied. The file's stored ``position`` is used only when the signal is
    missing, and only if the file was itself produced under the same rule.
    """
    raw = entry.get("signal")
    if raw is not None:
        stay = entry.get("stay_probability")
        if rule in STAY_RULES:
            if stay is None:
                return None, "no stay_probability"
            return position_from_signal(float(raw), float(stay), rule), None
        return position_from_signal(float(raw), IGNORED_STAY, rule), None
    position = entry.get("position")
    if position is None:
        return None, "no signal"
    if file_rule != rule:
        return None, f"position from file under rule {file_rule!r}, cannot re-derive"
    return float(position), None


def _positions(
    signal: Mapping, rule: str, allow_short: bool
) -> tuple[dict[str, float], dict[str, str], set[str]]:
    """Per-symbol position in [-1, 1], skipped symbols with reasons, and floored shorts."""
    positions: dict[str, float] = {}
    skipped: dict[str, str] = {}
    floored: set[str] = set()
    file_rule = signal.get("rule")
    for symbol, entry in _symbol_entries(signal).items():
        error = entry.get("error")
        if error:
            skipped[symbol] = f"error: {error}"
            continue
        position, reason = _entry_position(entry, file_rule, rule)
        if position is None:
            skipped[symbol] = reason or "no signal"
            continue
        if not math.isfinite(position):
            skipped[symbol] = f"non-finite position {position!r}"
            continue
        if position < 0 and not allow_short:
            floored.add(symbol)
            position = 0.0
        positions[symbol] = position
    return positions, skipped, floored


def target_allocation(
    signal: Mapping,
    *,
    rule: str = DEFAULT_RULE,
    max_gross: float = DEFAULT_MAX_GROSS,
    allow_short: bool = DEFAULT_ALLOW_SHORT,
) -> tuple[dict[str, float], dict[str, str]]:
    """Equal-weight target weights from a signal file, plus the symbols left out.

    Per symbol the position is ``position_from_signal(signal, stay_probability, rule)``
    so ``rule`` is always the rule applied, whatever rule the daily job used. The
    entry's stored ``position`` is a fallback only when ``signal`` is missing and the
    file's own ``rule`` equals ``rule``; otherwise the symbol is skipped. A missing
    ``stay_probability`` skips the symbol under the persistence rule (the only rule
    that reads it) rather than being coerced to zero. An entry with an ``error`` or
    no signal is skipped and the reason recorded. With ``allow_short`` False a
    negative position becomes 0.0 (flat), and the symbol stays active.

    Weights are ``position / n_active`` where ``n_active`` counts every symbol with a
    position, flat ones included (they simply get 0). If the gross exposure
    ``sum(|w|)`` exceeds ``max_gross`` every weight is scaled by ``max_gross / gross``.
    Equal weight across the watchlist is deliberate: there is no optimisation, no
    risk model and no conviction tilt beyond what the position rule already encodes,
    so the result is easy to audit by hand. Returns ``(weights, skipped)`` with
    symbols in sorted order.
    """
    _validate_sizing_params(rule, max_gross)
    positions, skipped, _ = _positions(signal, rule, allow_short)
    n_active = len(positions)
    weights = {symbol: position / n_active for symbol, position in positions.items()}
    gross = sum(abs(w) for w in weights.values())
    if gross > max_gross:
        scale = max_gross / gross
        weights = {symbol: w * scale for symbol, w in weights.items()}
    return weights, skipped


def _validate_equity(equity: float) -> float:
    if isinstance(equity, bool) or not isinstance(equity, int | float):
        raise ValueError(f"equity must be a finite number > 0, got {equity!r}")
    if not math.isfinite(equity) or equity <= 0:
        raise ValueError(f"equity must be a finite number > 0, got {equity!r}")
    return float(equity)


def _validate_prices(prices: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(prices, Mapping):
        raise ValueError(f"prices must map symbol -> last price, got {type(prices).__name__}")
    clean: dict[str, float] = {}
    for symbol, price in prices.items():
        if isinstance(price, bool) or not isinstance(price, int | float):
            raise ValueError(f"price for {symbol} must be a finite number > 0, got {price!r}")
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"price for {symbol} must be a finite number > 0, got {price!r}")
        clean[str(symbol)] = float(price)
    return clean


def _validate_min_trade_notional(min_trade_notional: float) -> float:
    if (
        isinstance(min_trade_notional, bool)
        or not isinstance(min_trade_notional, int | float)
        or not math.isfinite(min_trade_notional)
        or min_trade_notional < 0
    ):
        raise ValueError(
            f"min_trade_notional must be a finite number >= 0, got {min_trade_notional!r}"
        )
    return float(min_trade_notional)


def propose_orders(
    signal: Mapping,
    *,
    equity: float,
    prices: Mapping[str, float],
    current_positions: Mapping[str, float] | None = None,
    rule: str = DEFAULT_RULE,
    max_gross: float = DEFAULT_MAX_GROSS,
    allow_short: bool = DEFAULT_ALLOW_SHORT,
    min_trade_notional: float = DEFAULT_MIN_TRADE_NOTIONAL,
) -> AllocationProposal:
    """Diff the target allocation against current holdings and list the trades.

    ``current_positions`` maps symbol to signed share quantity (negative = short) and
    defaults to flat. ``current_weight = qty * price / equity``. For each target symbol
    ``delta = (target - current) * equity``: below ``min_trade_notional`` it is dust
    and skipped; otherwise ``qty = |delta| / price`` with side ``buy`` for a positive
    delta and ``sell`` for a negative one. A symbol without a price is skipped. Held
    symbols that are not in the signal file are reported in ``current_weights`` and
    ``skipped`` but never traded: this helper only knows about the watchlist.

    Raises ``ValueError`` when equity is not a finite number > 0 or any price is not a
    finite number > 0, naming the symbol and value. The result is a printout; nothing
    here or anywhere in hedgeFund submits it.
    """
    equity = _validate_equity(equity)
    prices = _validate_prices(prices)
    min_trade_notional = _validate_min_trade_notional(min_trade_notional)
    held = {str(s): float(q) for s, q in (current_positions or {}).items()}
    for symbol, qty in held.items():
        if not math.isfinite(qty):
            raise ValueError(f"current position for {symbol} must be finite, got {qty!r}")

    targets, skipped = target_allocation(
        signal, rule=rule, max_gross=max_gross, allow_short=allow_short
    )
    _, _, floored = _positions(signal, rule, allow_short)
    regimes = {s: e.get("current_regime") for s, e in _symbol_entries(signal).items()}

    current_weights: dict[str, float] = {}
    for symbol in sorted(set(targets) | set(held)):
        if symbol in prices:
            current_weights[symbol] = held.get(symbol, 0.0) * prices[symbol] / equity
        elif symbol in held:
            skipped.setdefault(symbol, "no price")
    for symbol in sorted(set(held) - set(targets)):
        if symbol in prices:
            skipped.setdefault(symbol, "not in signal file; left untouched")

    orders: list[ProposedOrder] = []
    for symbol, target in targets.items():
        if symbol not in prices:
            skipped[symbol] = "no price"
            continue
        current = current_weights[symbol]
        delta = (target - current) * equity
        if abs(delta) < min_trade_notional:
            skipped[symbol] = (
                f"dust: |{delta:.2f}| below min_trade_notional {min_trade_notional:.2f}"
            )
            continue
        price = prices[symbol]
        reason = f"regime {regimes.get(symbol) or 'n/a'}; {current:+.4f} -> {target:+.4f}"
        if symbol in floored:
            reason += " (short floored to flat: allow_short is False)"
        orders.append(
            ProposedOrder(
                symbol=symbol,
                side=SIDE_BUY if delta > 0 else SIDE_SELL,
                qty=abs(delta) / price,
                notional=abs(delta),
                target_weight=target,
                current_weight=current,
                last_price=price,
                reason=reason,
            )
        )

    return AllocationProposal(
        equity=equity,
        as_of=signal.get("as_of"),
        run_id=signal.get("run_id"),
        rule=rule,
        max_gross=float(max_gross),
        allow_short=bool(allow_short),
        target_weights=targets,
        current_weights=current_weights,
        orders=tuple(orders),
        skipped=dict(sorted(skipped.items())),
    )


ORDER_COLUMNS = ("symbol", "current w", "target w", "side", "qty", "notional", "reason")
ORDER_WIDTHS = (8, 10, 10, 5, 12, 14)  # the reason column is free width


def render_proposal(proposal: AllocationProposal) -> str:
    """Plain-text table of the proposal, ending with the note and the disclaimer.

    Plain ``str`` formatting like ``report/terminal.py`` so the output can be diffed.
    The last line is ``PROPOSAL_NOTE``; ``DISCLAIMER`` is the line before it.
    """
    header = " ".join(f"{c:<{w}}" for c, w in zip(ORDER_COLUMNS, ORDER_WIDTHS, strict=False))
    lines = [
        f"Proposed paper orders  as_of={proposal.as_of or 'n/a'}  "
        f"run_id={proposal.run_id or 'n/a'}",
        f"Equity ${proposal.equity:,.2f}  rule={proposal.rule}  max_gross={proposal.max_gross:.2f}"
        f"  allow_short={proposal.allow_short}",
        "",
        header + " " + ORDER_COLUMNS[-1],
    ]
    if not proposal.orders:
        lines.append("  (no orders: every symbol is at target or skipped)")
    for order in proposal.orders:
        cells = (
            order.symbol,
            f"{order.current_weight * PERCENT:+.2f}%",
            f"{order.target_weight * PERCENT:+.2f}%",
            order.side,
            f"{order.qty:.4f}",
            f"${order.notional:,.2f}",
        )
        row = " ".join(f"{c:<{w}}" for c, w in zip(cells, ORDER_WIDTHS, strict=True))
        lines.append(row + " " + order.reason)
    if proposal.skipped:
        lines.append("")
        lines.append("Skipped:")
        lines.extend(f"  {symbol}: {why}" for symbol, why in proposal.skipped.items())
    gross = sum(abs(w) for w in proposal.target_weights.values())
    lines.append("")
    lines.append(
        f"Gross target exposure {gross * PERCENT:.2f}% of equity "
        f"(cap {proposal.max_gross * PERCENT:.0f}%), "
        f"{len(proposal.orders)} proposed order(s), orders_placed={proposal.orders_placed}"
    )
    lines.append(proposal.disclaimer)
    lines.append(proposal.note)
    return "\n".join(lines)


# --- read-only Alpaca paper account access --------------------------------------------


KEY_NAMES = ("key_id", "api_key", "key", KEY_ID_ENV)  # field names a credential reader may use
SECRET_NAMES = ("secret_key", "api_secret", "secret", SECRET_KEY_ENV)


def _unpack_credentials(creds: object) -> tuple[str, str]:
    """Accept ``(key, secret)``, a mapping or an object with key/secret attributes."""
    if isinstance(creds, tuple | list) and len(creds) == 2:
        key, secret = creds
    elif isinstance(creds, Mapping):
        key = next((creds[k] for k in KEY_NAMES if k in creds), None)
        secret = next((creds[k] for k in SECRET_NAMES if k in creds), None)
    else:
        key = next((getattr(creds, k) for k in KEY_NAMES if hasattr(creds, k)), None)
        secret = next((getattr(creds, k) for k in SECRET_NAMES if hasattr(creds, k)), None)
    if not key or not secret:
        raise RuntimeError(CREDENTIALS_MESSAGE)
    return str(key), str(secret)


def _env_credentials(environ: Mapping[str, str]) -> tuple[str, str]:
    """Fallback reader: the two alpaca-py variables straight from ``environ``."""
    return environ.get(KEY_ID_ENV, ""), environ.get(SECRET_KEY_ENV, "")


def _credentials(
    environ: Mapping[str, str] | None,
    reader: Callable[[Mapping[str, str]], object] | None = None,
) -> tuple[str, str]:
    """Paper keys from ``environ`` (default ``os.environ``) without ever echoing them.

    Prefers ``hedgefund.data.alpaca_source.read_credentials`` when importable, else the
    plain environment fallback. Any failure becomes ``RuntimeError(CREDENTIALS_MESSAGE)``
    so no exception text can carry a key value.
    """
    environ = os.environ if environ is None else environ
    if reader is None:
        try:
            from hedgefund.data.alpaca_source import read_credentials as reader
        except ImportError:
            reader = _env_credentials
    try:
        return _unpack_credentials(reader(environ))
    except Exception as exc:
        raise RuntimeError(CREDENTIALS_MESSAGE) from exc


def _assert_paper_endpoint(client: object) -> None:
    """Refuse any client whose ``_base_url`` (alpaca-py's attribute) is not the paper url."""
    base = getattr(client, "_base_url", None)
    base = getattr(base, "value", base)  # alpaca's BaseURL is a str Enum
    if base is None or str(base).rstrip("/") != PAPER_TRADING_URL:
        raise RuntimeError(LIVE_ENDPOINT_MESSAGE)


def _paper_client(environ: Mapping[str, str] | None, trading_client: object | None) -> object:
    """The injected client, or a lazily built read-only ``TradingClient(paper=True)``."""
    if trading_client is None:
        key, secret = _credentials(environ)
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise RuntimeError(
                "alpaca-py is not installed; add the 'alpaca' extra (uv sync --extra alpaca)"
            ) from exc
        trading_client = TradingClient(key, secret, paper=True)
    _assert_paper_endpoint(trading_client)
    return trading_client


def paper_account_equity(
    environ: Mapping[str, str] | None = None, trading_client: object | None = None
) -> float:
    """Current paper account equity in dollars, via the read-only ``get_account`` call.

    ``trading_client`` is duck-typed (``get_account()`` returns an object with ``.equity``)
    so tests inject a fake; when ``None`` a paper ``TradingClient`` is built from the
    environment. Raises ``RuntimeError("live endpoint refused")`` for any other endpoint.
    """
    client = _paper_client(environ, trading_client)
    equity = float(client.get_account().equity)
    if not math.isfinite(equity) or equity <= 0:
        raise ValueError(f"paper account equity must be a finite number > 0, got {equity!r}")
    return equity


def paper_positions(
    environ: Mapping[str, str] | None = None, trading_client: object | None = None
) -> dict[str, float]:
    """Current paper positions as ``{symbol: signed qty}`` (short = negative), read-only.

    ``get_all_positions()`` items need ``.symbol``, ``.qty`` and ``.side``; the sign is
    taken from ``side`` so a short reported with a positive or negative qty both come
    back negative. Same client rules as ``paper_account_equity``.
    """
    client = _paper_client(environ, trading_client)
    positions: dict[str, float] = {}
    for item in client.get_all_positions():
        side = getattr(item, "side", "long")
        side = str(getattr(side, "value", side)).lower()
        qty = abs(float(item.qty))
        positions[str(item.symbol)] = -qty if "short" in side else qty
    return dict(sorted(positions.items()))
