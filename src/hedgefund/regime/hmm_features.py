"""HMM feature matrix and training-window-only standardiser (plan.md Fix 5).

``build_features`` turns a close series (plus optional high / low / volume) into
the observation matrix an HMM is fitted on: the daily log return, two trailing
realised-volatility estimates and, when the inputs exist, the Parkinson range and a
trailing volume z-score. Every row ``t`` is computed from data at or before ``t``
only (trailing windows, no centred windows, no negative shifts), so perturbing the
inputs after ``t`` cannot change any feature row dated at or before ``t``.

``FeatureScaler`` / ``fit_scaler`` / ``standardise_walk_forward`` standardise the
matrix with the mean and standard deviation of the *training rows only*. A scaler
fitted on ``features.iloc[:train_end]`` carries no statistic from row ``train_end``
onward, which is the no-lookahead property ``.claude/rules/no-lookahead.md``
requires of feature scalers.

Column 0 is always the daily log return. Rolling volatility must never be the only
feature fed to the HMM: on volatility alone the model simply clusters on scale
(plan.md Fix 5, "Do not feed rolling volatility alone"). This module always emits
the return next to the volatilities; the fitter in ``regime/hmm.py`` enforces that
the return column is present and uses a full covariance matrix.

Pure module: pandas / numpy in, pandas / numpy out, no I/O.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

# The always-present columns, in this order; column 0 is ALWAYS the daily log return.
FEATURE_NAMES: tuple[str, ...] = ("log_return", "vol_10", "vol_20")
# Columns that appear only when the corresponding inputs are given, in this order.
OPTIONAL_FEATURE_NAMES: tuple[str, ...] = ("range", "volume_z")

DEFAULT_VOL_SHORT = 10  # config hmm.vol_short: 10-day realised volatility (plan.md Fix 5)
DEFAULT_VOL_LONG = 20  # config hmm.vol_long: 20-day realised volatility, the labeller window
DEFAULT_VOLUME_WINDOW = 20  # trailing window of the volume z-score, one labeller window
SCALE_EPS = 1e-12  # floor on a feature's std so a constant column cannot divide by zero

_PARKINSON_DENOMINATOR = 4.0 * np.log(2.0)  # Parkinson (1980) range-variance constant


def _validate_close(close: pd.Series) -> None:
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError(f"close must have a DatetimeIndex, got {type(close.index).__name__}")
    if close.empty:
        raise ValueError("close is empty; need at least vol_long + 1 rows to build any feature")
    values = close.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or not np.all(values > 0):
        raise ValueError("close must be finite and strictly positive everywhere")


def _validate_aligned(name: str, series: pd.Series, close: pd.Series) -> np.ndarray:
    if not series.index.equals(close.index):
        raise ValueError(f"{name} must be aligned to close's index (same dates, same order)")
    values = series.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite everywhere")
    return values


def _validate_windows(vol_short: int, vol_long: int, volume_window: int) -> None:
    if vol_short < 2:
        raise ValueError(f"vol_short must be >= 2 for a sample standard deviation, got {vol_short}")
    if vol_long <= vol_short:
        raise ValueError(
            f"vol_long must be > vol_short, got vol_long={vol_long}, vol_short={vol_short}"
        )
    if volume_window < 2:
        raise ValueError(
            f"volume_window must be >= 2 for a sample standard deviation, got {volume_window}"
        )


def build_features(
    close: pd.Series,
    *,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    volume: pd.Series | None = None,
    vol_short: int = DEFAULT_VOL_SHORT,
    vol_long: int = DEFAULT_VOL_LONG,
    volume_window: int = DEFAULT_VOLUME_WINDOW,
) -> pd.DataFrame:
    """Build the HMM observation matrix from closes and optional high / low / volume.

    Columns, in order:

    - ``log_return``      : ``ln(close_t / close_{t-1})``
    - ``vol_<vol_short>`` : trailing sample std (ddof=1) of ``log_return`` over the
      ``vol_short`` bars ending at ``t`` inclusive
    - ``vol_<vol_long>``  : the same over ``vol_long`` bars
    - ``range``           : only when ``high`` and ``low`` are given; the Parkinson
      daily range ``sqrt(ln(high/low)^2 / (4 ln 2))``
    - ``volume_z``        : only when ``volume`` is given;
      ``(volume_t - trailing mean) / trailing std (ddof=1)`` over ``volume_window``
      bars ending at ``t`` inclusive

    The index is the close index with every warm-up row (any NaN) dropped, so the
    first retained date is the ``vol_long``-th return (close position ``vol_long``)
    when ``volume_window <= vol_long``. Every row ``t`` uses data at or before ``t``
    only. All columns are float64.

    Raises ``ValueError`` when ``close`` lacks a ``DatetimeIndex``, is empty, or is
    not finite and strictly positive; when ``vol_short < 2``, ``vol_long <= vol_short``
    or ``volume_window < 2``; when only one of ``high`` / ``low`` is given, either is
    misaligned, or ``high >= low > 0`` fails; when ``volume`` is misaligned or has a
    negative entry.
    """
    _validate_windows(vol_short, vol_long, volume_window)
    _validate_close(close)
    if (high is None) != (low is None):
        raise ValueError("high and low must be given together, got only one of them")

    log_close = np.log(close.to_numpy(dtype=float))
    log_return = pd.Series(np.diff(log_close, prepend=np.nan), index=close.index)
    columns: dict[str, pd.Series] = {
        FEATURE_NAMES[0]: log_return,
        f"vol_{vol_short}": log_return.rolling(vol_short).std(ddof=1),
        f"vol_{vol_long}": log_return.rolling(vol_long).std(ddof=1),
    }

    if high is not None and low is not None:
        high_values = _validate_aligned("high", high, close)
        low_values = _validate_aligned("low", low, close)
        if not np.all(low_values > 0):
            raise ValueError("low must be strictly positive everywhere")
        if not np.all(high_values >= low_values):
            raise ValueError("high must be >= low on every bar")
        log_range = np.log(high_values / low_values)
        columns[OPTIONAL_FEATURE_NAMES[0]] = pd.Series(
            np.sqrt(log_range**2 / _PARKINSON_DENOMINATOR), index=close.index
        )

    if volume is not None:
        volume_values = _validate_aligned("volume", volume, close)
        if not np.all(volume_values >= 0):
            raise ValueError("volume must be >= 0 everywhere")
        vol_series = pd.Series(volume_values, index=close.index)
        trailing_mean = vol_series.rolling(volume_window).mean()
        trailing_std = vol_series.rolling(volume_window).std(ddof=1)
        columns[OPTIONAL_FEATURE_NAMES[1]] = (vol_series - trailing_mean) / trailing_std

    features = pd.DataFrame(columns, index=close.index).astype(np.float64)
    return features.dropna(how="any")


@dataclass(frozen=True)
class FeatureScaler:
    """Per-column affine standardiser fitted on a training block only.

    ``mean`` and ``std`` have shape ``(n_features,)``; every ``std`` is at least
    ``SCALE_EPS``. ``columns`` are the feature names in column order.
    """

    mean: np.ndarray
    std: np.ndarray
    columns: tuple[str, ...]

    def transform(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return ``(x - mean) / std`` as float64.

        A 2-D input of shape ``(n, n_features)`` returns the same shape; a 1-D row of
        length ``n_features`` returns shape ``(n_features,)``. Raises ``ValueError``
        when the column count does not match ``len(columns)``.
        """
        values = _as_float_array(x)
        n_features = len(self.columns)
        if values.ndim not in (1, 2) or values.shape[-1] != n_features:
            raise ValueError(
                f"x must have {n_features} columns {self.columns} as its last axis, "
                f"got shape {values.shape}"
            )
        return (values - self.mean) / self.std


def _as_float_array(x: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def fit_scaler(x: np.ndarray | pd.DataFrame, columns: Sequence[str] | None = None) -> FeatureScaler:
    """Fit a ``FeatureScaler`` on the rows of ``x`` only.

    Uses the column mean and population standard deviation (ddof=0) of the rows
    passed, with each std floored at ``SCALE_EPS`` so a constant column transforms to
    zero instead of inf / NaN. If ``x`` is a DataFrame its column names are used and
    ``columns`` must be omitted or match them; if ``x`` is an ndarray, ``columns`` is
    required. ``x`` must be 2-D with at least 2 rows and no NaN / inf.
    """
    if isinstance(x, pd.DataFrame):
        frame_columns = tuple(str(c) for c in x.columns)
        if columns is not None and tuple(columns) != frame_columns:
            raise ValueError(
                f"columns {tuple(columns)} do not match the DataFrame columns {frame_columns}"
            )
        names = frame_columns
    else:
        if columns is None:
            raise ValueError("columns is required when x is an ndarray")
        names = tuple(str(c) for c in columns)

    values = _as_float_array(x)
    if values.ndim != 2:
        raise ValueError(f"x must be 2-D (n_rows, n_features), got shape {values.shape}")
    if values.shape[0] < 2:
        raise ValueError(f"x must have at least 2 rows to fit a scaler, got {values.shape[0]}")
    if values.shape[1] != len(names):
        raise ValueError(
            f"x has {values.shape[1]} columns but {len(names)} names were given: {names}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("x must not contain NaN or inf; drop warm-up rows before fitting")

    mean = values.mean(axis=0)
    std = np.maximum(values.std(axis=0, ddof=0), SCALE_EPS)
    return FeatureScaler(mean=mean, std=std, columns=names)


def standardise_walk_forward(
    features: pd.DataFrame, train_end: int
) -> tuple[np.ndarray, FeatureScaler]:
    """Fit a scaler on ``features.iloc[:train_end]`` and transform the whole matrix.

    Rows strictly before ``train_end`` are the training block; no statistic from row
    ``train_end`` onward enters the scaler, so a walk-forward that scores rows
    ``train_end, train_end + 1, ...`` with the returned matrix has no lookahead in its
    scaling. Returns ``(X_scaled, scaler)`` where ``X_scaled`` has the shape of
    ``features``.

    Raises ``ValueError`` when ``train_end`` is outside ``[2, len(features)]``.
    """
    n_rows = len(features)
    if not 2 <= train_end <= n_rows:
        raise ValueError(f"train_end must be in [2, {n_rows}] (len(features)), got {train_end}")
    scaler = fit_scaler(features.iloc[:train_end])
    return scaler.transform(features), scaler
