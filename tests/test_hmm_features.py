"""Tests for the HMM feature matrix and the training-window-only scaler (Phase 4, Fix 5).

All data is synthetic with explicit seeds; nothing touches the network.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from hedgefund.regime.hmm_features import (
    DEFAULT_VOL_LONG,
    DEFAULT_VOL_SHORT,
    DEFAULT_VOLUME_WINDOW,
    FEATURE_NAMES,
    OPTIONAL_FEATURE_NAMES,
    SCALE_EPS,
    FeatureScaler,
    build_features,
    fit_scaler,
    standardise_walk_forward,
)

N_DAYS = 300
SEED = 20261020
DAILY_VOL = 0.01
INTRADAY_SPREAD = 0.005  # typical half-range of a daily bar as a fraction of price
PERTURB_SCALE = 1.5  # multiplicative shock to the future path (as in test_no_lookahead)
PERTURB_NOISE = 0.05  # additive noise on the future path, as a fraction of price
CUT_DAYS = [DEFAULT_VOL_LONG + 1, 75, 150, 240, N_DAYS - 2]
TRAIN_END = 120


def synthetic_ohlcv(
    n_days: int = N_DAYS, seed: int = SEED
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Close, high, low and volume on a business-day index; high >= close >= low > 0."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2016-01-04", periods=n_days)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0002, DAILY_VOL, n_days))), index)
    high = close * (1.0 + np.abs(rng.normal(0.0, INTRADAY_SPREAD, n_days)))
    low = close * (1.0 - np.abs(rng.normal(0.0, INTRADAY_SPREAD, n_days)))
    volume = pd.Series(rng.integers(1_000, 5_000, n_days).astype(float), index)
    return close, high, low, volume


def perturb_after(series: pd.Series, t: int, seed: int = SEED + 1) -> pd.Series:
    """Return a copy where every value strictly after position ``t`` is changed."""
    rng = np.random.default_rng(seed)
    out = series.copy()
    future = out.iloc[t + 1 :]
    noise = rng.normal(0.0, PERTURB_NOISE, size=len(future))
    out.iloc[t + 1 :] = future.to_numpy() * (PERTURB_SCALE + noise)
    assert not np.array_equal(out.to_numpy(), series.to_numpy())
    return out


def perturb_bars_after(
    close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, t: int
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Perturb every bar strictly after ``t``, keeping ``high >= close >= low > 0``.

    High and low after ``t`` are rebuilt around the shocked close with a fresh seeded
    spread, so they change too but stay a valid bar; volume gets its own shock.
    """
    rng = np.random.default_rng(SEED + 2)
    shocked_close = perturb_after(close, t)
    n_future = len(close) - (t + 1)
    shocked_high = high.copy()
    shocked_low = low.copy()
    future_close = shocked_close.iloc[t + 1 :].to_numpy()
    shocked_high.iloc[t + 1 :] = future_close * (
        1.0 + np.abs(rng.normal(0.0, INTRADAY_SPREAD, n_future))
    )
    shocked_low.iloc[t + 1 :] = future_close * (
        1.0 - np.abs(rng.normal(0.0, INTRADAY_SPREAD, n_future))
    )
    shocked_volume = perturb_after(volume, t, seed=SEED + 3)
    for name, original, shocked in (
        ("high", high, shocked_high),
        ("low", low, shocked_low),
    ):
        assert not np.array_equal(original.to_numpy(), shocked.to_numpy()), name
    return shocked_close, shocked_high, shocked_low, shocked_volume


def _full_features(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series):
    return build_features(close, high=high, low=low, volume=volume)


# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #
def test_default_columns_are_the_required_names_in_order() -> None:
    close, *_ = synthetic_ohlcv()
    features = build_features(close)
    assert tuple(features.columns) == FEATURE_NAMES
    assert FEATURE_NAMES[0] == "log_return"
    assert ("log_return", f"vol_{DEFAULT_VOL_SHORT}", f"vol_{DEFAULT_VOL_LONG}") == FEATURE_NAMES


def test_optional_columns_follow_the_required_ones_in_order() -> None:
    close, high, low, volume = synthetic_ohlcv()
    features = _full_features(close, high, low, volume)
    assert tuple(features.columns) == FEATURE_NAMES + OPTIONAL_FEATURE_NAMES


def test_only_range_when_only_high_low_given() -> None:
    close, high, low, _ = synthetic_ohlcv()
    features = build_features(close, high=high, low=low)
    assert tuple(features.columns) == (*FEATURE_NAMES, "range")


def test_only_volume_z_when_only_volume_given() -> None:
    close, _, _, volume = synthetic_ohlcv()
    features = build_features(close, volume=volume)
    assert tuple(features.columns) == (*FEATURE_NAMES, "volume_z")


def test_vol_columns_are_named_with_the_actual_windows() -> None:
    close, *_ = synthetic_ohlcv()
    features = build_features(close, vol_short=5, vol_long=30)
    assert tuple(features.columns) == ("log_return", "vol_5", "vol_30")


def test_all_columns_are_float64() -> None:
    close, high, low, volume = synthetic_ohlcv()
    features = _full_features(close, high, low, volume)
    assert all(dtype == np.float64 for dtype in features.dtypes)


# --------------------------------------------------------------------------- #
# Known answers
# --------------------------------------------------------------------------- #
def test_log_return_matches_hand_computation() -> None:
    close, *_ = synthetic_ohlcv()
    features = build_features(close)
    t = features.index[7]
    position = close.index.get_loc(t)
    expected = np.log(close.iloc[position] / close.iloc[position - 1])
    assert features.loc[t, "log_return"] == pytest.approx(expected, rel=0, abs=1e-15)


def test_vol_10_is_sample_std_of_last_ten_returns_inclusive() -> None:
    close, *_ = synthetic_ohlcv()
    features = build_features(close)
    t = features.index[5]
    position = close.index.get_loc(t)
    log_close = np.log(close.to_numpy())
    returns = np.diff(log_close)  # returns[i] is the return of close position i + 1
    last_ten = returns[position - DEFAULT_VOL_SHORT : position]  # ends at position inclusive
    assert len(last_ten) == DEFAULT_VOL_SHORT
    assert features.loc[t, "vol_10"] == pytest.approx(np.std(last_ten, ddof=1), rel=1e-12)


def test_vol_20_is_sample_std_of_last_twenty_returns_inclusive() -> None:
    close, *_ = synthetic_ohlcv()
    features = build_features(close)
    t = features.index[0]
    position = close.index.get_loc(t)
    returns = np.diff(np.log(close.to_numpy()))
    last = returns[position - DEFAULT_VOL_LONG : position]
    assert len(last) == DEFAULT_VOL_LONG
    assert features.loc[t, "vol_20"] == pytest.approx(np.std(last, ddof=1), rel=1e-12)


def test_parkinson_range_matches_formula() -> None:
    close, high, low, _ = synthetic_ohlcv()
    features = build_features(close, high=high, low=low)
    t = features.index[3]
    expected = np.sqrt(np.log(high.loc[t] / low.loc[t]) ** 2 / (4.0 * np.log(2.0)))
    assert features.loc[t, "range"] == pytest.approx(expected, rel=1e-12)


def test_volume_z_uses_trailing_window_inclusive_of_t() -> None:
    close, _, _, volume = synthetic_ohlcv()
    features = build_features(close, volume=volume)
    t = features.index[9]
    position = close.index.get_loc(t)
    window = volume.to_numpy()[position - DEFAULT_VOLUME_WINDOW + 1 : position + 1]
    assert len(window) == DEFAULT_VOLUME_WINDOW
    expected = (volume.loc[t] - window.mean()) / np.std(window, ddof=1)
    assert features.loc[t, "volume_z"] == pytest.approx(expected, rel=1e-12)


def test_constant_close_series_gives_zero_return_and_zero_vol() -> None:
    index = pd.bdate_range("2020-01-01", periods=40)
    close = pd.Series(50.0, index=index)
    features = build_features(close)
    assert np.array_equal(features.to_numpy(), np.zeros(features.shape))


# --------------------------------------------------------------------------- #
# Index and warm-up
# --------------------------------------------------------------------------- #
def test_no_nan_remains_and_index_is_suffix_of_close_index() -> None:
    close, high, low, volume = synthetic_ohlcv()
    features = _full_features(close, high, low, volume)
    assert not features.isna().any().any()
    assert features.index.equals(close.index[-len(features) :])


def test_first_retained_date_is_the_vol_long_th_return() -> None:
    close, high, low, volume = synthetic_ohlcv()
    features = _full_features(close, high, low, volume)
    # Return 1 is at close position 1, so return vol_long is at close position vol_long.
    assert features.index[0] == close.index[DEFAULT_VOL_LONG]
    assert len(features) == N_DAYS - DEFAULT_VOL_LONG


def test_longer_volume_window_delays_the_first_row() -> None:
    close, _, _, volume = synthetic_ohlcv()
    features = build_features(close, volume=volume, volume_window=30)
    # The volume z-score needs 30 bars ending at t, so the first defined row is position 29.
    assert features.index[0] == close.index[29]


def test_too_short_series_gives_empty_frame_with_columns() -> None:
    index = pd.bdate_range("2020-01-01", periods=DEFAULT_VOL_LONG)
    close = pd.Series(np.linspace(100.0, 110.0, DEFAULT_VOL_LONG), index=index)
    features = build_features(close)
    assert features.empty
    assert tuple(features.columns) == FEATURE_NAMES


# --------------------------------------------------------------------------- #
# No lookahead
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("t", CUT_DAYS)
def test_features_through_t_are_bit_identical_after_perturbing_the_future(t: int) -> None:
    close, high, low, volume = synthetic_ohlcv()
    cutoff = close.index[t]
    base = _full_features(close, high, low, volume)
    shocked = _full_features(*perturb_bars_after(close, high, low, volume, t))
    before = base.loc[:cutoff]
    after = shocked.loc[:cutoff]
    assert len(before) > 0, f"nothing to compare at t={t}"
    assert before.index.equals(after.index)
    assert np.array_equal(before.to_numpy(), after.to_numpy())


@pytest.mark.parametrize("t", CUT_DAYS)
def test_perturbation_actually_changes_later_features(t: int) -> None:
    """Sanity: the shock is large enough that rows after t differ, so the test has teeth."""
    close, high, low, volume = synthetic_ohlcv()
    cutoff = close.index[t]
    base = _full_features(close, high, low, volume)
    shocked = _full_features(perturb_after(close, t), high, low, volume)
    later_base = base.loc[cutoff:].iloc[1:].to_numpy()
    later_shocked = shocked.loc[cutoff:].iloc[1:].to_numpy()
    assert len(later_base) > 0
    assert not np.array_equal(later_base, later_shocked)


# --------------------------------------------------------------------------- #
# Scaler
# --------------------------------------------------------------------------- #
def test_fit_scaler_on_train_rows_gives_zero_mean_unit_std_on_train_block() -> None:
    close, high, low, volume = synthetic_ohlcv()
    features = _full_features(close, high, low, volume)
    train = features.iloc[:TRAIN_END]
    scaler = fit_scaler(train)
    scaled = scaler.transform(train)
    assert scaled.shape == train.shape
    assert scaled.dtype == np.float64
    # 1e-12 is a few ulps of float64 accumulation over 120 rows of O(1) values.
    assert np.all(np.abs(scaled.mean(axis=0)) < 1e-12)
    assert np.all(np.abs(scaled.std(axis=0, ddof=0) - 1.0) < 1e-12)


def test_fit_scaler_takes_columns_from_the_dataframe() -> None:
    close, high, low, volume = synthetic_ohlcv()
    features = _full_features(close, high, low, volume)
    scaler = fit_scaler(features)
    assert scaler.columns == FEATURE_NAMES + OPTIONAL_FEATURE_NAMES
    assert scaler.mean.shape == (5,)
    assert scaler.std.shape == (5,)


def test_fit_scaler_on_ndarray_matches_dataframe_fit() -> None:
    close, *_ = synthetic_ohlcv()
    features = build_features(close)
    from_frame = fit_scaler(features)
    from_array = fit_scaler(features.to_numpy(), columns=FEATURE_NAMES)
    assert from_array.columns == from_frame.columns
    assert np.array_equal(from_array.mean, from_frame.mean)
    assert np.array_equal(from_array.std, from_frame.std)


def test_transform_accepts_a_single_row() -> None:
    close, *_ = synthetic_ohlcv()
    features = build_features(close)
    scaler = fit_scaler(features)
    row = features.iloc[0]
    out = scaler.transform(row.to_numpy())
    assert out.shape == (len(FEATURE_NAMES),)
    assert np.array_equal(out, scaler.transform(features)[0])


def test_transform_uses_only_training_statistics_on_later_rows() -> None:
    """Rows after the training block are scaled with the train mean / std, not their own."""
    close, *_ = synthetic_ohlcv()
    features = build_features(close)
    scaler = fit_scaler(features.iloc[:TRAIN_END])
    test_block = features.iloc[TRAIN_END:]
    expected = (test_block.to_numpy() - scaler.mean) / scaler.std
    assert np.array_equal(scaler.transform(test_block), expected)


def test_standardise_walk_forward_scaler_equals_fit_on_train_block() -> None:
    close, high, low, volume = synthetic_ohlcv()
    features = _full_features(close, high, low, volume)
    scaled, scaler = standardise_walk_forward(features, TRAIN_END)
    direct = fit_scaler(features.iloc[:TRAIN_END])
    assert scaler.columns == direct.columns
    assert np.array_equal(scaler.mean, direct.mean)
    assert np.array_equal(scaler.std, direct.std)
    assert scaled.shape == features.shape
    assert np.array_equal(scaled, direct.transform(features))


def test_standardise_walk_forward_scaler_ignores_rows_at_and_after_train_end() -> None:
    close, high, low, volume = synthetic_ohlcv()
    features = _full_features(close, high, low, volume)
    _, base = standardise_walk_forward(features, TRAIN_END)

    rng = np.random.default_rng(SEED + 5)
    mutated = features.copy()
    tail = mutated.iloc[TRAIN_END:].to_numpy()
    mutated.iloc[TRAIN_END:] = tail * (PERTURB_SCALE + rng.normal(0.0, PERTURB_NOISE, tail.shape))
    assert not np.array_equal(mutated.to_numpy(), features.to_numpy())

    scaled, shocked = standardise_walk_forward(mutated, TRAIN_END)
    assert np.array_equal(base.mean, shocked.mean)
    assert np.array_equal(base.std, shocked.std)
    # The train block itself is scaled identically because nothing before train_end moved.
    assert np.array_equal(
        scaled[:TRAIN_END], standardise_walk_forward(features, TRAIN_END)[0][:TRAIN_END]
    )


def test_standardise_walk_forward_accepts_train_end_equal_to_length() -> None:
    close, *_ = synthetic_ohlcv()
    features = build_features(close)
    scaled, scaler = standardise_walk_forward(features, len(features))
    assert np.array_equal(scaler.mean, fit_scaler(features).mean)
    assert scaled.shape == features.shape


def test_constant_column_is_floored_and_transforms_to_zero() -> None:
    x = np.column_stack([np.full(10, 3.0), np.arange(10, dtype=float)])
    scaler = fit_scaler(x, columns=("const", "ramp"))
    assert scaler.std[0] == SCALE_EPS
    assert np.all(scaler.std >= SCALE_EPS)
    out = scaler.transform(x)
    assert np.all(np.isfinite(out))
    assert np.array_equal(out[:, 0], np.zeros(10))


def test_feature_scaler_is_frozen() -> None:
    scaler = FeatureScaler(mean=np.zeros(1), std=np.ones(1), columns=("a",))
    with pytest.raises(AttributeError):
        scaler.mean = np.ones(1)  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# ValueError paths
# --------------------------------------------------------------------------- #
def _close(n: int = 40) -> pd.Series:
    rng = np.random.default_rng(SEED + 6)
    index = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, DAILY_VOL, n))), index)


def test_build_features_rejects_non_datetime_index() -> None:
    close = pd.Series(np.linspace(100.0, 110.0, 40))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        build_features(close)


def test_build_features_rejects_empty_close() -> None:
    close = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
    with pytest.raises(ValueError, match="close is empty"):
        build_features(close)


def test_build_features_rejects_non_positive_close() -> None:
    close = _close()
    close.iloc[5] = 0.0
    with pytest.raises(ValueError, match="close must be finite and strictly positive"):
        build_features(close)


def test_build_features_rejects_nan_close() -> None:
    close = _close()
    close.iloc[5] = np.nan
    with pytest.raises(ValueError, match="close must be finite"):
        build_features(close)


def test_build_features_rejects_vol_short_below_two() -> None:
    with pytest.raises(ValueError, match="vol_short must be >= 2"):
        build_features(_close(), vol_short=1)


def test_build_features_rejects_vol_long_not_above_vol_short() -> None:
    with pytest.raises(ValueError, match="vol_long must be > vol_short"):
        build_features(_close(), vol_short=10, vol_long=10)


def test_build_features_rejects_volume_window_below_two() -> None:
    with pytest.raises(ValueError, match="volume_window must be >= 2"):
        build_features(_close(), volume_window=1)


def test_build_features_rejects_high_without_low() -> None:
    close = _close()
    with pytest.raises(ValueError, match="high and low must be given together"):
        build_features(close, high=close * 1.01)
    with pytest.raises(ValueError, match="high and low must be given together"):
        build_features(close, low=close * 0.99)


def test_build_features_rejects_misaligned_high() -> None:
    close = _close()
    high = (close * 1.01).iloc[1:]
    low = close * 0.99
    with pytest.raises(ValueError, match="high must be aligned"):
        build_features(close, high=high, low=low)


def test_build_features_rejects_misaligned_low() -> None:
    close = _close()
    high = close * 1.01
    low = (close * 0.99).iloc[::-1]
    with pytest.raises(ValueError, match="low must be aligned"):
        build_features(close, high=high, low=low)


def test_build_features_rejects_high_below_low() -> None:
    close = _close()
    with pytest.raises(ValueError, match="high must be >= low"):
        build_features(close, high=close * 0.99, low=close * 1.01)


def test_build_features_rejects_non_positive_low() -> None:
    close = _close()
    low = close * 0.99
    low.iloc[3] = 0.0
    with pytest.raises(ValueError, match="low must be strictly positive"):
        build_features(close, high=close * 1.01, low=low)


def test_build_features_rejects_misaligned_volume() -> None:
    close = _close()
    volume = pd.Series(1_000.0, index=close.index[:-1])
    with pytest.raises(ValueError, match="volume must be aligned"):
        build_features(close, volume=volume)


def test_build_features_rejects_negative_volume() -> None:
    close = _close()
    volume = pd.Series(1_000.0, index=close.index)
    volume.iloc[2] = -1.0
    with pytest.raises(ValueError, match="volume must be >= 0"):
        build_features(close, volume=volume)


def test_fit_scaler_requires_columns_for_ndarray() -> None:
    with pytest.raises(ValueError, match="columns is required"):
        fit_scaler(np.ones((5, 2)))


def test_fit_scaler_rejects_mismatched_columns_for_ndarray() -> None:
    with pytest.raises(ValueError, match="names were given"):
        fit_scaler(np.ones((5, 2)), columns=("a",))


def test_fit_scaler_rejects_mismatched_columns_for_dataframe() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    with pytest.raises(ValueError, match="do not match the DataFrame columns"):
        fit_scaler(frame, columns=("a", "c"))


def test_fit_scaler_rejects_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        fit_scaler(np.ones(5), columns=("a",))


def test_fit_scaler_rejects_fewer_than_two_rows() -> None:
    with pytest.raises(ValueError, match="at least 2 rows"):
        fit_scaler(np.ones((1, 2)), columns=("a", "b"))


def test_fit_scaler_rejects_nan() -> None:
    x = np.ones((5, 2))
    x[2, 1] = np.nan
    with pytest.raises(ValueError, match="must not contain NaN"):
        fit_scaler(x, columns=("a", "b"))


def test_transform_rejects_wrong_column_count() -> None:
    scaler = fit_scaler(np.arange(10.0).reshape(5, 2), columns=("a", "b"))
    with pytest.raises(ValueError, match="must have 2 columns"):
        scaler.transform(np.ones((3, 3)))
    with pytest.raises(ValueError, match="must have 2 columns"):
        scaler.transform(np.ones(3))


def test_standardise_walk_forward_rejects_train_end_out_of_range() -> None:
    close, *_ = synthetic_ohlcv()
    features = build_features(close)
    with pytest.raises(ValueError, match="train_end must be in"):
        standardise_walk_forward(features, 1)
    with pytest.raises(ValueError, match="train_end must be in"):
        standardise_walk_forward(features, len(features) + 1)


# --------------------------------------------------------------------------- #
# Invariant: transform is invertible
# --------------------------------------------------------------------------- #
@settings(max_examples=50, deadline=None)
@given(
    x=hnp.arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(2, 30), st.integers(1, 4)),
        elements=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
    )
)
def test_transform_then_inverse_recovers_input(x: np.ndarray) -> None:
    columns = tuple(f"f{i}" for i in range(x.shape[1]))
    scaler = fit_scaler(x, columns=columns)
    recovered = scaler.transform(x) * scaler.std + scaler.mean
    # 1e-9 relative (plus a tiny absolute floor for values near zero) covers the two
    # float64 roundings of the round trip with a wide margin.
    assert np.allclose(recovered, x, rtol=1e-9, atol=1e-9)
