"""Parquet cache in front of any ``PriceSource``.

``CachedSource`` wraps an inner source and keys every request by the inner source's
name, the symbol, and the concrete date window. A hit reads the parquet and never
touches the inner source, which is how the Phase 5 gate shows a warm scan makes zero
network calls. A miss fetches, writes, and returns; in ``offline`` mode a miss raises
``CacheMissError`` instead of fetching.

The cache is deliberately simple: one parquet file per (source, symbol, start, end).
Requests without bounds resolve to the companion default window (``default_window``), so
their key is concrete and stable within a day and rolls forward the next day.
"""

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hedgefund.data.base import PriceSource, validate_price_frame
from hedgefund.data.yfinance_source import DEFAULT_YEARS, ONE_DAY, default_window

# Mirrors config/default.yaml data.cache_dir; relative to the working directory unless an
# absolute path is given. The repo ships an empty ``cache/`` folder (git-ignored contents).
DEFAULT_CACHE_DIR = Path("cache")
# Parquet keeps the dtypes validate_price_frame needs; the fixture uses the same format.
CACHE_SUFFIX = ".parquet"
# Date format in cache keys; one bar per day, so day resolution is exact.
KEY_DATE_FORMAT = "%Y-%m-%d"
# Any character outside this set is replaced in file names ("BTC/USD" -> "BTC-USD").
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_REPLACEMENT = "-"
# Hex characters of a content fingerprint kept in the key: 48 bits is plenty to tell one
# edit of a file from another and keeps file names short.
FINGERPRINT_HEX_CHARS = 12


class CacheMissError(LookupError):
    """Raised in offline mode when the requested window is not in the cache."""


def sanitise_name(name: str) -> str:
    """Return ``name`` with every path-unsafe character replaced by ``-``.

    Raises ``ValueError`` when ``name`` is empty or sanitises to nothing but dots, which
    would alias the current or parent directory.
    """
    if not name:
        raise ValueError("cache key component must be a non-empty string")
    safe = _UNSAFE.sub(_REPLACEMENT, name)
    if safe.strip(".") == "":
        raise ValueError(f"cache key component {name!r} is not a usable file name")
    return safe


def cache_key(
    source: str,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fingerprint: str | None = None,
) -> str:
    """Return ``"<source>/<symbol>_<start>_<end>[_<fingerprint>]"`` with sanitised names.

    ``fingerprint`` identifies the source's content for sources whose data can change
    under the same window (a CSV the user edits); it is omitted when ``None``. Raises
    ``ValueError`` when ``start`` is after ``end``.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError(f"start {start_ts.date()} must not be after end {end_ts.date()}")
    key = (
        f"{sanitise_name(source)}/{sanitise_name(symbol)}"
        f"_{start_ts.strftime(KEY_DATE_FORMAT)}_{end_ts.strftime(KEY_DATE_FORMAT)}"
    )
    if fingerprint is not None:
        key = f"{key}_{sanitise_name(fingerprint)}"
    return key


def cache_path(
    cache_dir: Path | str,
    source: str,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fingerprint: str | None = None,
) -> Path:
    """Return the parquet path for ``(source, symbol, start, end)`` under ``cache_dir``."""
    return Path(cache_dir) / f"{cache_key(source, symbol, start, end, fingerprint)}{CACHE_SUFFIX}"


def file_fingerprint(path: Path | str) -> str:
    """Return the first ``FINGERPRINT_HEX_CHARS`` hex digits of the sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:FINGERPRINT_HEX_CHARS]


def frame_digest(df: pd.DataFrame) -> str:
    """Return a sha256 hex digest of ``df``'s dates and closes.

    The digest covers the index as int64 nanoseconds followed by the float64 ``close``
    bytes, so it is stable across runs and machines and changes when any date or close
    changes. Other columns are not hashed: the pipeline only consumes ``close``.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"frame_digest needs a DatetimeIndex, got {type(df.index).__name__}")
    if "close" not in df.columns:
        raise ValueError(f"frame_digest needs a 'close' column, saw {list(df.columns)}")
    index_ns = np.ascontiguousarray(df.index.as_unit("ns").asi8, dtype=np.int64)
    close = np.ascontiguousarray(df["close"].to_numpy(dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(index_ns.tobytes())
    digest.update(close.tobytes())
    return digest.hexdigest()


def write_cached(path: Path, df: pd.DataFrame) -> None:
    """Validate ``df`` and write it to ``path`` as parquet, creating parent directories.

    The write goes through a sibling temp file and an atomic rename so a crash mid-write
    never leaves a truncated parquet that would later read as a corrupt hit.
    """
    validate_price_frame(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        df.to_parquet(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_cached(path: Path) -> pd.DataFrame:
    """Read the parquet at ``path`` and validate it as a price frame."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no cached price frame at {path}")
    return validate_price_frame(pd.read_parquet(path))


@dataclass
class CacheEvent:
    """What one ``CachedSource.get_daily`` call did; the scan manifest records these."""

    symbol: str
    path: Path
    hit: bool
    start: pd.Timestamp
    end: pd.Timestamp
    rows: int
    digest: str


class CachedSource:
    """``PriceSource`` that serves parquet hits and fetches misses from ``inner``.

    ``name`` is the inner source's name because the cache is keyed by it: a yfinance
    frame and a CSV frame for the same symbol never collide. When ``inner`` has a
    ``fingerprint(symbol) -> str | None`` method and it returns a string, that string is
    part of the key too, so a source whose content changes under the same window (an
    edited CSV) misses instead of serving stale bars.
    """

    def __init__(
        self,
        inner: PriceSource,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        *,
        years: int = DEFAULT_YEARS,
        offline: bool = False,
    ) -> None:
        if years < 1:
            raise ValueError(f"years must be at least 1, got {years}")
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.years = years
        self.offline = offline
        self.events: list[CacheEvent] = []

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def hits(self) -> int:
        return sum(1 for event in self.events if event.hit)

    @property
    def misses(self) -> int:
        return sum(1 for event in self.events if not event.hit)

    def resolve_window(
        self, start: pd.Timestamp | None, end: pd.Timestamp | None
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Fill missing bounds from ``default_window(years)`` so the key is concrete.

        The companion window ends today *exclusive*: ``fetch_ticker`` asks Yahoo for bars
        up to yesterday, so today's partial intraday bar never enters an analysis. The
        inner source treats an explicit ``end`` as inclusive, so a missing ``end`` resolves
        to yesterday (UTC); the key is then stable within a UTC day and the scan sees the
        same last bar as ``analyze --ticker``.
        """
        default_start, default_end = default_window(self.years)
        start_ts = default_start if start is None else pd.Timestamp(start)
        end_ts = default_end - ONE_DAY if end is None else pd.Timestamp(end)
        return start_ts, end_ts

    def fingerprint_for(self, symbol: str) -> str | None:
        """Return the inner source's content fingerprint for ``symbol``, or ``None``."""
        fingerprint = getattr(self.inner, "fingerprint", None)
        if not callable(fingerprint):
            return None
        value = fingerprint(symbol)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{self.name}.fingerprint({symbol!r}) must return a non-empty str or None, "
                f"got {value!r}"
            )
        return value

    def path_for(
        self, symbol: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
    ) -> Path:
        """Return the parquet path a ``get_daily`` call with these arguments would use."""
        start_ts, end_ts = self.resolve_window(start, end)
        return cache_path(
            self.cache_dir, self.name, symbol, start_ts, end_ts, self.fingerprint_for(symbol)
        )

    def get_daily(
        self,
        symbol: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return the cached frame for the window, fetching from ``inner`` on a miss.

        The inner source receives the resolved bounds so the stored frame is exactly the
        window its key names. A hit never calls ``inner``. With ``offline=True`` a miss
        raises ``CacheMissError`` naming the symbol and the path that was looked for.
        """
        start_ts, end_ts = self.resolve_window(start, end)
        path = cache_path(
            self.cache_dir, self.name, symbol, start_ts, end_ts, self.fingerprint_for(symbol)
        )

        if path.is_file():
            frame = read_cached(path)
            hit = True
        else:
            if self.offline:
                raise CacheMissError(
                    f"offline mode: no cached data for {symbol} at {path}. "
                    "Run once without --offline to populate the cache."
                )
            frame = validate_price_frame(self.inner.get_daily(symbol, start_ts, end_ts))
            write_cached(path, frame)
            hit = False

        self.events.append(
            CacheEvent(
                symbol=symbol,
                path=path,
                hit=hit,
                start=start_ts,
                end=end_ts,
                rows=len(frame),
                digest=frame_digest(frame),
            )
        )
        return frame
