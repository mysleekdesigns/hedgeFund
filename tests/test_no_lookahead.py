"""Perturbation tests: nothing after day t may change any output at or before t.

Every estimator and position rule in ``regime/`` and ``backtest/`` is registered in
``ESTIMATORS`` below as a ``(name, callable)`` pair. The callable takes a close
series and returns a dict of named output series, each indexed by the date at which
the value is *known*. The test mutates every close strictly after day ``t`` and
asserts, bit for bit, that every output up to and including the date knowable at
``t`` is unchanged.

To register a new estimator in a later phase, add one entry to ``ESTIMATORS``: a
callable ``close -> {name: pd.Series}``. Outputs that are only knowable after the
next close (the strategy return earned by the decision at ``t``, dated ``t+1``) are
indexed by the date they are knowable, so the generic check still applies.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from hedgefund.backtest.walk_forward import WalkForwardResult, run_walk_forward
from hedgefund.regime.labels import (
    ensemble_label_regimes,
    ew_label_regimes,
    hysteresis_label_regimes,
    label_regimes,
    zscore_label_regimes,
)

N_DAYS = 600
MIN_TRAIN = 60  # small training window so the walk-forward emits plenty of decisions
WINDOW = 20
THRESHOLD = 0.02
SEED = 20261019
DAILY_VOL = 0.01
PERTURB_SCALE = 1.5  # multiplicative shock to the future path
PERTURB_NOISE = 0.05  # additive noise on the future path, as a fraction of price

# Close positions t to cut at. Labels start at close index WINDOW, so the first decision
# is at close index MIN_TRAIN + WINDOW and the first earned return one day later; the
# first cut sits there so every output has at least one knowable entry. The last cut is
# the final decision day, whose earned return is the last close.
CUT_DAYS = [MIN_TRAIN + WINDOW + 1, 150, 301, 450, N_DAYS - 2]

Estimator = Callable[[pd.Series], dict[str, pd.Series]]


def synthetic_close(n_days: int = N_DAYS, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0002, DAILY_VOL, size=n_days)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, index=pd.bdate_range("2016-01-04", periods=n_days), name="SYNTH")


def perturb_after(close: pd.Series, t: int, seed: int = SEED + 1) -> pd.Series:
    """Return a copy where every close strictly after position ``t`` is changed."""
    rng = np.random.default_rng(seed)
    out = close.copy()
    future = out.iloc[t + 1 :]
    noise = rng.normal(0.0, PERTURB_NOISE, size=len(future))
    out.iloc[t + 1 :] = future.to_numpy() * (PERTURB_SCALE + noise)
    assert not np.array_equal(out.to_numpy(), close.to_numpy())
    return out


# --------------------------------------------------------------------------- #
# Estimator registry: (name, close -> {output name: series indexed by knowable date})
# --------------------------------------------------------------------------- #
def _labels(close: pd.Series) -> dict[str, pd.Series]:
    return {"labels": label_regimes(close, window=WINDOW, threshold=THRESHOLD)}


def _walk_forward_outputs(result: WalkForwardResult) -> dict[str, pd.Series]:
    return {
        "signals": result.signals,
        "positions": result.positions,
        # Indexed by t+1 already: the return earned by the decision at t is knowable at t+1,
        # and so is the asset return of t+1 that it multiplies.
        "strategy_returns": result.strategy_returns,
        "asset_returns": result.asset_returns,
    }


def _walk_forward(close: pd.Series, **options: object) -> dict[str, pd.Series]:
    labels = label_regimes(close, window=WINDOW, threshold=THRESHOLD)
    return _walk_forward_outputs(run_walk_forward(close, labels, min_train=MIN_TRAIN, **options))


# Phase 3 walk-forward variants. The rolling window is far shorter than N_DAYS so
# transitions really do drop out of the matrix between the cuts; the dead band is wide
# enough that the conviction position actually holds through small signal moves.
ROLLING_BARS = 100
DEAD_BAND = 0.2


def _walk_forward_rolling(close: pd.Series) -> dict[str, pd.Series]:
    return _walk_forward(close, refit="rolling", rolling_bars=ROLLING_BARS)


def _walk_forward_conviction(close: pd.Series) -> dict[str, pd.Series]:
    return _walk_forward(close, rule="conviction")


def _walk_forward_persistence(close: pd.Series) -> dict[str, pd.Series]:
    return _walk_forward(close, rule="persistence")


def _walk_forward_long_only(close: pd.Series) -> dict[str, pd.Series]:
    return _walk_forward(close, rule="long_only")


def _walk_forward_dead_band(close: pd.Series) -> dict[str, pd.Series]:
    return _walk_forward(close, rule="conviction", dead_band=DEAD_BAND)


# Review fix W2: the sign rules apply the band as hysteresis on the signal around zero. A
# 0.1 band is far below the unit step of a sign target, so it only has an effect through
# that signal-space path.
SIGNAL_DEAD_BAND = 0.1


def _walk_forward_sign_dead_band(close: pd.Series) -> dict[str, pd.Series]:
    return _walk_forward(close, rule="sign", dead_band=SIGNAL_DEAD_BAND)


def _walk_forward_long_only_dead_band(close: pd.Series) -> dict[str, pd.Series]:
    return _walk_forward(close, rule="long_only", dead_band=SIGNAL_DEAD_BAND)


def _walk_forward_rolling_persistence_dead_band(close: pd.Series) -> dict[str, pd.Series]:
    """Every new option at once, so their interaction is covered too."""
    return _walk_forward(
        close,
        refit="rolling",
        rolling_bars=ROLLING_BARS,
        rule="persistence",
        dead_band=DEAD_BAND,
    )


# Alternative labellers (Phase 2, Fix 3). Each must have at least one knowable value at
# the first cut (close index MIN_TRAIN + WINDOW + 1 = 81): zscore uses vol_window=WINDOW so
# its first label is at index 20; the ensemble's longest window (60) is also before 81.
ENSEMBLE_WINDOWS = (5, 10, 20, 60)
HYSTERESIS_EXIT = THRESHOLD / 2  # exit must sit strictly inside the entry threshold


def _zscore_labels(close: pd.Series) -> dict[str, pd.Series]:
    return {"labels": zscore_label_regimes(close, window=WINDOW, vol_window=WINDOW)}


def _ew_labels(close: pd.Series) -> dict[str, pd.Series]:
    return {"labels": ew_label_regimes(close, window=WINDOW, threshold=THRESHOLD)}


def _ensemble_labels(close: pd.Series) -> dict[str, pd.Series]:
    return {"labels": ensemble_label_regimes(close, windows=ENSEMBLE_WINDOWS, threshold=THRESHOLD)}


def _hysteresis_labels(close: pd.Series) -> dict[str, pd.Series]:
    return {
        "labels": hysteresis_label_regimes(
            close, window=WINDOW, enter_threshold=THRESHOLD, exit_threshold=HYSTERESIS_EXIT
        )
    }


ESTIMATORS: list[tuple[str, Estimator]] = [
    ("label_regimes", _labels),
    ("walk_forward_sign_rule", _walk_forward),
    ("zscore_label_regimes", _zscore_labels),
    ("ew_label_regimes", _ew_labels),
    ("ensemble_label_regimes", _ensemble_labels),
    ("hysteresis_label_regimes", _hysteresis_labels),
    ("walk_forward_rolling_refit", _walk_forward_rolling),
    ("walk_forward_conviction_rule", _walk_forward_conviction),
    ("walk_forward_persistence_rule", _walk_forward_persistence),
    ("walk_forward_long_only_rule", _walk_forward_long_only),
    ("walk_forward_conviction_dead_band", _walk_forward_dead_band),
    ("walk_forward_sign_dead_band", _walk_forward_sign_dead_band),
    ("walk_forward_long_only_dead_band", _walk_forward_long_only_dead_band),
    ("walk_forward_rolling_persistence_dead_band", _walk_forward_rolling_persistence_dead_band),
]


def _knowable_prefix(series: pd.Series, cutoff: pd.Timestamp) -> np.ndarray:
    return series.loc[:cutoff].to_numpy()


@pytest.mark.parametrize("t", CUT_DAYS)
@pytest.mark.parametrize(("name", "estimator"), ESTIMATORS, ids=[n for n, _ in ESTIMATORS])
def test_outputs_through_t_are_bit_identical_after_perturbing_the_future(
    name: str, estimator: Estimator, t: int
) -> None:
    close = synthetic_close()
    cutoff = close.index[t]
    base = estimator(close)
    shocked = estimator(perturb_after(close, t))

    assert base.keys() == shocked.keys()
    for output, series in base.items():
        before = _knowable_prefix(series, cutoff)
        after = _knowable_prefix(shocked[output], cutoff)
        assert len(before) > 0, f"{name}.{output}: nothing to compare at t={t}"
        assert np.array_equal(before, after, equal_nan=True), (
            f"{name}.{output} changed at or before t={t} after perturbing the future"
        )
        assert series.loc[:cutoff].index.equals(shocked[output].loc[:cutoff].index)


@pytest.mark.parametrize("t", CUT_DAYS)
def test_walk_forward_return_earned_by_decision_at_t_is_unchanged(t: int) -> None:
    """The decision at t earns the return of t+1; that single entry must also survive.

    Perturbing closes strictly after t changes close[t+1], so the return earned by the
    decision at t changes too. What must *not* change is the position that earned it.
    """
    close = synthetic_close()
    labels = label_regimes(close, window=WINDOW, threshold=THRESHOLD)
    base = run_walk_forward(close, labels, min_train=MIN_TRAIN)

    shocked_close = perturb_after(close, t)
    shocked = run_walk_forward(
        shocked_close,
        label_regimes(shocked_close, window=WINDOW, threshold=THRESHOLD),
        min_train=MIN_TRAIN,
    )

    cutoff = close.index[t]
    assert np.array_equal(
        base.positions.loc[:cutoff].to_numpy(), shocked.positions.loc[:cutoff].to_numpy()
    )
    # The return entry dated t+1 equals position_t * the (now different) return of t+1.
    next_day = close.index[t + 1]
    shocked_ret = shocked_close.pct_change().loc[next_day]
    assert shocked.strategy_returns.loc[next_day] == base.positions.loc[cutoff] * shocked_ret


@pytest.mark.parametrize("t", CUT_DAYS)
def test_perturbation_actually_changes_later_outputs(t: int) -> None:
    """Sanity: the shock is large enough that outputs after t differ, so the test has teeth."""
    close = synthetic_close()
    cutoff = close.index[t]
    base = _walk_forward(close)
    shocked = _walk_forward(perturb_after(close, t))
    later_base = base["strategy_returns"].loc[cutoff:].iloc[1:].to_numpy()
    later_shocked = shocked["strategy_returns"].loc[cutoff:].iloc[1:].to_numpy()
    assert len(later_base) > 0
    assert not np.array_equal(later_base, later_shocked)


def test_walk_forward_is_deterministic() -> None:
    close = synthetic_close()
    a = _walk_forward(close)
    b = _walk_forward(close.copy())
    for key in a:
        assert np.array_equal(a[key].to_numpy(), b[key].to_numpy())


def test_phase3_variants_have_teeth() -> None:
    """Each Phase 3 variant must produce positions that differ from the sign rule's, so
    its perturbation entry above tests a genuinely different code path; the rolling
    variant must also drop transitions, so its signals differ."""
    close = synthetic_close()
    base = _walk_forward(close)
    rolling = _walk_forward_rolling(close)
    assert not np.array_equal(base["signals"].to_numpy(), rolling["signals"].to_numpy())
    for variant in (
        _walk_forward_conviction,
        _walk_forward_persistence,
        _walk_forward_long_only,
        _walk_forward_dead_band,
        _walk_forward_sign_dead_band,
        _walk_forward_long_only_dead_band,
        _walk_forward_rolling_persistence_dead_band,
    ):
        out = variant(close)
        assert not np.array_equal(base["positions"].to_numpy(), out["positions"].to_numpy())
    # The dead band actually held a position at least once (fewer changes than no band).
    banded = _walk_forward_dead_band(close)["positions"].to_numpy()
    unbanded = _walk_forward_conviction(close)["positions"].to_numpy()
    assert np.count_nonzero(np.diff(banded)) < np.count_nonzero(np.diff(unbanded))
    # The signal-space band on the sign rules also holds positions (W2): fewer changes
    # than the same rule without a band, and the positions stay on the unit grid.
    for banded_variant, plain_variant, allowed in (
        (_walk_forward_sign_dead_band, _walk_forward, {-1.0, 0.0, 1.0}),
        (_walk_forward_long_only_dead_band, _walk_forward_long_only, {0.0, 1.0}),
    ):
        banded = banded_variant(close)["positions"].to_numpy()
        plain = plain_variant(close)["positions"].to_numpy()
        assert np.count_nonzero(np.diff(banded)) < np.count_nonzero(np.diff(plain))
        assert set(np.unique(banded)) <= allowed
