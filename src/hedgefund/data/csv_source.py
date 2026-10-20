"""Load a user's own price series from CSV.

``load_csv`` is a faithful port of the companion script's loader: the same date-column
and close-column resolution order, the same ``to_datetime(errors="coerce")`` parse, the
same sort and dropna, and a Series named after the file stem. Two deliberate differences:
failures raise ``ValueError`` rather than ``RuntimeError``, and floats are parsed with
``float_precision="round_trip"`` so an exported close reads back bit-identical.

``CsvSource`` wraps the same parse into the ``PriceSource`` protocol so a CSV can stand in
for any other source in the pipeline.
"""

from pathlib import Path

import pandas as pd

from hedgefund.data.base import INDEX_NAME, PRICE_COLUMNS, validate_price_frame

# Resolution order for the date column, then the first column as a fallback.
DATE_COLUMN_KEYS = ("date", "time", "timestamp", "datetime")
# Resolution order for the close column, then the sole numeric column as a fallback.
CLOSE_COLUMN_KEYS = ("close", "adj close", "adj_close", "adjclose", "price", "last")


def _read_table(path: str | Path) -> tuple[pd.DataFrame, str, str]:
    """Read ``path`` and resolve its date and close columns the companion way."""
    try:
        # round_trip parsing reads a float back bit-for-bit; pandas' default fast parser
        # can land one ulp off, which breaks exact comparisons against the parquet fixture.
        df = pd.read_csv(path, float_precision="round_trip")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{path} is empty.") from exc
    if df.empty:
        raise ValueError(f"{path} is empty.")

    cols = {str(c).lower().strip(): c for c in df.columns}

    date_col = None
    for key in DATE_COLUMN_KEYS:
        if key in cols:
            date_col = cols[key]
            break
    if date_col is None:
        date_col = df.columns[0]

    close_col = None
    for key in CLOSE_COLUMN_KEYS:
        if key in cols:
            close_col = cols[key]
            break
    if close_col is None:
        numeric = df.select_dtypes("number").columns.tolist()
        numeric = [c for c in numeric if c != date_col]
        if len(numeric) == 1:
            close_col = numeric[0]
        else:
            raise ValueError(
                f"Could not find a close column in {path}. "
                f"Add a column named one of: close, adj close, price, last. "
                f"Saw columns: {list(df.columns)}"
            )
    return df, date_col, close_col


def _parse_rows(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Coerce the date column, drop unparsable rows, and sort ascending."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], utc=False, errors="coerce")
    return out.dropna(subset=[date_col]).sort_values(date_col)


def load_csv(path: str | Path) -> pd.Series:
    """Load a close series from a CSV, companion-faithful.

    Requirements are deliberately loose so this drops into any pipeline:
      - one column that parses as dates (first match of: date, time,
        timestamp, datetime, or the first column)
      - one close column (first match of: close, adj close, adj_close,
        price, last, or, if only one numeric column exists, that column)

    Returns a float Series indexed by ``DatetimeIndex`` and named after the file stem.
    Raises ``ValueError`` naming ``path`` when it is empty, has no close column, or
    yields no usable rows.
    """
    df, date_col, close_col = _read_table(path)
    out = _parse_rows(df[[date_col, close_col]], date_col)
    close = pd.Series(
        pd.to_numeric(out[close_col], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(out[date_col]),
        name=Path(path).stem,
    ).dropna()
    if close.empty:
        raise ValueError(f"No usable rows after parsing {path}.")
    return close


class CsvSource:
    """``PriceSource`` backed by a CSV file; ``symbol`` is the path to the file."""

    name = "csv"

    def get_daily(
        self,
        symbol: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return a validated daily frame from the CSV at ``symbol``.

        ``close`` is always present; ``open``, ``high``, ``low`` and ``volume`` are included
        when the CSV has columns of those names. Duplicate dates keep the first row, as the
        frozen fixture does. ``start`` and ``end`` slice inclusively.
        """
        df, date_col, close_col = _read_table(symbol)
        cols = {str(c).lower().strip(): c for c in df.columns}
        optional = {
            key: cols[key]
            for key in PRICE_COLUMNS
            if key != "close" and key in cols and cols[key] not in (date_col, close_col)
        }
        keep = [date_col, close_col, *optional.values()]
        out = _parse_rows(df[keep], date_col)

        frame = pd.DataFrame(index=pd.DatetimeIndex(out[date_col], name=INDEX_NAME))
        for key, col in optional.items():
            frame[key] = pd.to_numeric(out[col], errors="coerce").to_numpy()
        frame["close"] = pd.to_numeric(out[close_col], errors="coerce").to_numpy()
        frame = frame[[c for c in PRICE_COLUMNS if c in frame.columns]]

        frame = frame.dropna(subset=["close"])
        frame = frame[~frame.index.duplicated(keep="first")].sort_index()
        if frame.empty:
            raise ValueError(f"No usable rows after parsing {symbol}.")
        for col in frame.columns:
            if col == "volume" and not frame[col].isna().any():
                frame[col] = frame[col].astype("int64")
            else:
                frame[col] = frame[col].astype("float64")

        if start is not None or end is not None:
            frame = frame.loc[start:end]
        if frame.empty:
            raise ValueError(f"{symbol} has no rows between {start} and {end}.")
        return validate_price_frame(frame)


class CsvDirSource:
    """``PriceSource`` over a directory of CSVs; ``symbol`` resolves to ``<dir>/<symbol>.csv``.

    This is the CSV counterpart of ``YFinanceSource`` for the scan: a universe file lists
    symbols and this source maps each to a file. File names are used as given, so a
    symbol like ``BTC/USD`` must be stored under the name the universe uses.
    """

    name = "csv"

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def path_for(self, symbol: str) -> Path:
        """Return the CSV path ``symbol`` maps to."""
        return self.directory / f"{symbol}.csv"

    def fingerprint(self, symbol: str) -> str | None:
        """Return a short sha256 of the CSV's bytes, or ``None`` when the file is absent.

        ``CachedSource`` folds this into its key so an edited CSV is a cache miss rather
        than a stale hit under the same date window.
        """
        from hedgefund.data.cache import file_fingerprint

        path = self.path_for(symbol)
        if not path.is_file():
            return None
        return file_fingerprint(path)

    def get_daily(
        self,
        symbol: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return the validated frame for ``symbol``; a missing file raises ``ValueError``."""
        path = self.path_for(symbol)
        if not path.is_file():
            raise ValueError(f"no CSV for symbol {symbol!r}: {path} does not exist")
        return CsvSource().get_daily(str(path), start, end)
