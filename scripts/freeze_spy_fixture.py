# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2", "pyarrow>=17", "yfinance>=0.2.50"]
# ///
"""Freeze the SPY ten-year daily bars used as the Phase 1 reproduction baseline.

Mirrors the companion script's fetch exactly: ``yf.download(auto_adjust=True)`` with
``end`` = today (UTC, normalised) and ``start`` = ``end`` minus ten years. Adjusted closes
drift as dividends accrue, which is why the series is frozen once and never overwritten.

Run once from the project root (network required, so never from a test):

    uv run scripts/freeze_spy_fixture.py --ticker SPY --years 10

It writes ``tests/fixtures/<ticker>_<years>y.parquet`` plus a provenance JSON next to it and
refuses to overwrite an existing fixture unless ``--force`` is given.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

# Column order the PriceSource protocol expects (close is mandatory, the rest optional).
COLUMNS = ["open", "high", "low", "close", "volume"]


def fetch_bars(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download daily bars the same way the companion script does, then normalise them."""
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,  # same as the companion script, so closes match its output
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker} between {start} and {end}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[COLUMNS]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna(subset=["close"])
    return df.astype({c: "float64" for c in COLUMNS[:-1]} | {"volume": "int64"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--end", help="YYYY-MM-DD, exclusive; default today (UTC)")
    parser.add_argument("--out-dir", default="tests/fixtures")
    parser.add_argument("--force", action="store_true", help="overwrite an existing fixture")
    args = parser.parse_args()

    end = (
        pd.Timestamp(args.end)
        if args.end
        else pd.Timestamp.now("UTC").tz_localize(None).normalize()
    )
    start = end - pd.DateOffset(years=args.years)
    out_dir = Path(args.out_dir)
    stem = f"{args.ticker.lower().replace('-', '_')}_{args.years}y"
    parquet_path = out_dir / f"{stem}.parquet"
    provenance_path = out_dir / f"{stem}.provenance.json"
    if parquet_path.exists() and not args.force:
        sys.exit(f"{parquet_path} exists; fixtures are frozen. Pass --force to overwrite.")

    df = fetch_bars(args.ticker, start, end)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, engine="pyarrow", index=True)

    import pyarrow
    import yfinance

    provenance = {
        "ticker": args.ticker,
        "source": "yfinance",
        "download_kwargs": {
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "auto_adjust": True,
        },
        "pulled_at_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        "rows": len(df),
        "first_date": str(df.index[0].date()),
        "last_date": str(df.index[-1].date()),
        "columns": COLUMNS,
        "sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
        "versions": {
            "yfinance": yfinance.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
