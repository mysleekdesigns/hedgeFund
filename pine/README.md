# Pine Script overlays

TradingView Pine v5 scripts for the Markov regime framework. Framework by Roan
(@RohOnChain); the original overlay is by Lewis Jackson. These are the hedgeFund
corrected versions (plan.md, Fix 1 and Fix 2).

## Files

- `markov_regime.pine` - the overlay indicator. Bull / Bear / Sideways labelling,
  regime ribbon, current-regime banner, 3x3 transition matrix table and stationary
  distribution table. Transition labels are drawn on the bar where the hold
  condition is met, never on a past bar.
- `markov_regime_strategy.pine` - the `strategy()` variant: enters on the next bar
  open after a confirmed flip, `process_orders_on_close=false`, so the Strategy
  Tester shows what the signal would actually have earned.

## Loading in TradingView

1. Open a chart (BTCUSDT daily or SPY daily are the reference charts).
2. Pine Editor (bottom panel) > Open > New indicator. Delete the template.
3. Paste the file contents and press "Add to chart". It must compile without errors.
4. Inputs are under the gear icon: "Regime logic", "Display", "Table positions".

## Manual checks (from plan.md Phase 6)

Pine cannot be compiled or run offline, so these two checks are done by hand in
TradingView and noted in the commit message when the checklist items are ticked:

1. **Compiles on Pine v5.** Add to chart with no compile errors or type warnings.
2. **No back-dated labels in Bar Replay.** On BTCUSDT daily and on SPY daily, start
   Bar Replay, step forward one bar at a time and confirm that no solid transition
   label ever appears on a bar that has already passed. Labels appear on the bar
   on which they become known, that is `min_regime_hold` bars after the flip. With
   the hindsight toggle on, hollow labels tagged "hindsight:" may appear on past
   bars; that is the point of the toggle.

Also worth eyeballing: every matrix row sums to about 100%, and the stationary row
sums to 1.

## Inputs that changed from the original overlay

- **Show hindsight labels (back-dated)**, default off. The original drew the label
  at `bar_index - (min_regime_hold - 1)`, the bar where the flip first appeared,
  which made the call look earlier than it was. Here the solid label sits on the
  current bar and its text records "flip was at bar -N". Turning this toggle on
  additionally draws the old back-dated label in a hollow style (transparent fill,
  tinted text) prefixed "hindsight:". Those are not tradeable live.
- **Stride counting (non-overlapping)**, default off. Off keeps the original
  consecutive-bar counting, where neighbouring windows share all but one return and
  the diagonal is inflated by overlap. On counts `regime[lookback_window] -> regime`
  only on bars where `bar_index % lookback_window == 0`, the same subsample as
  `labels[::window]` in the Python package. The matrix card header says "daily /
  overlapping" or "stride / N-bar step" so the chart states which it shows, and the
  footer shows the sample count `n`.

Everything else (thresholds, lookback, stationary power, palette, table positions,
text size, minimum hold) is unchanged.

## Tests

`tests/test_pine.py` reads the script as text and checks the invariants that can be
checked statically: version line, every `label.new` under a `barstate.isconfirmed`
guard, the default label at `bar_index` with no offset, both new toggles defaulting
to false, count updates guarded by `barstate.isconfirmed`, the stride construct, and
the presence of the tables, ribbon and stationary iteration. They are not a compiler.
