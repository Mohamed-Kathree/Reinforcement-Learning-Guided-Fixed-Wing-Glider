# V23 Gate D: diagnostics summary

The questions were whether the V22 policy steers its heading when it should steer its ground track (the "shortcut"), and whether the path reward pays it to drift downwind. Nothing here required training. All of it uses saved models and the benchmark's 100 paired seeds.
- **Models:** V22 best (15M steps) and V21 best (3.5M).
- **Steady wind:** stage 0 with mean wind up to 3 m/s and nothing else.
- **Code:** `analysis/wind_diagnostics.py`.
- **Results:** `results/wind_d{1..6}_*.json`, with provenance in each.

## Decision rule: third branch

The diagnosis was inferred and then tested, and the shortcut does not account for V22's failure in wind.
- **D1:** giving the policy course in place of yaw recovers none of the cross-wind excess. It recovers −17% for V22 (slightly worse) and 8% for V21, against a 40% threshold.
- **D2:** the yaw input does not dominate the course input. Its influence is 0.77× course for V22 and 0.94× for V21.
- **D5:** the policy's MLP fits the baseline's steering law almost as well from v1 as from v2. Bank error is 2.3° against 1.5°, with R² 0.95 against 0.97, out of a 24° spread.

Per the spec, Phase R comes first and Phase P is promoted. Phase O is built anyway (see the end of this summary).

## Results

| Test | V22 | V21 | Reading |
|---|---|---|---|
| D1a steady wind, no change: success / mean \|cross\| / mean \|along\| | 29% / 31 m / 48 m | 24% / 40 m / 43 m | Baseline: 39% / 20 m / 27 m |
| D1b yaw ← course | 24% (p=0.41), cross 33 m | 24% (p=1.0), cross 38 m | Using the course reading does not help |
| D1c course ← yaw | 20% (p=0.09) | 21% (p=0.69) | Losing the course reading barely matters either |
| D1d calm, yaw +15°: track shift | +4.0° [−1.0, 8.9], success 59→46% (p=0.06) | **+7.9° [1.6, 14.3], 58→26% (p<0.001)** | V21 leaned on yaw in calm air; V22 much less |
| D2 influence share, yaw / course (calm) | 0.072 / 0.093 | 0.090 / 0.096 | Inputs spread evenly; no yaw dominance |
| D3 calm success, sectors next to ±180° vs the rest | 44% vs 64% (n=27, χ² p=0.20) | 59% vs 58% (p=0.81) | No seam effect detected |
| D4 signed along-wind touchdown, policy − baseline (steady wind) | **+21.5 m [5.4, 37.5]** | +7.1 m [−6.1, 20.2] | V22 lands downwind |
| D4 r_path per second against wind speed | slope −0.001/(m/s), p≈1 | +0.008, n.s. | No measurable downwind payment |
| D5 bank error from v1 / v2 without previous action / v2 | 2.26° / 1.45° / 0.61° | — | v1 can represent the bearing error |
| D5 true wind from stacked v1, R² | north/east 0.78/0.69; tail/cross 0.87/0.73 | — | Wind is observable from v1 |
| D6 critic R², calm / steady wind / stage 1 | −1.16 / −0.36 / −0.11 | −0.95 / −0.03 / 0.02 | Poor everywhere, worst in calm |

### Three findings the spec did not anticipate
1. **V22's excess error in steady wind is along the wind as much as across it.**
   - Along-wind excess: +21.6 m over the baseline. Cross-wind excess: +10.4 m.
   - V22 lands about 21 m further downwind than the baseline on paired seeds.
   - The r_path accounting does not explain this. Per-second path reward does not rise with wind, because outbound and return legs cancel.
   - The cross-wind-dominant pattern in the spec came from V21 with gusts on. In V21's steady-wind results, the cross and along excesses are similar (19 m and 16 m).
2. **The policy doesn't fly at home.**
   - Its mean track-to-home-bearing error is 38° for V22 and −20° for V21 in wind, and 42° and −35° in calm air. The baseline's is −6° and 0°.
   - The policy flies a pattern, which r_path pays for. It loses in wind when that pattern ends up downwind or long.
3. **The critic is badly calibrated, and worst in calm air.**
   - At launch, V22's value estimate is −0.18 against a realised return of +1.41.
   - The correlation between value and return is 0.1–0.2 at launch and 0.5 at 75% of the flight.
   - A likely cause: the 15M model was trained mostly at stage 2, and the critic can't see which stage it is in. Its baseline is then biased toward stage-2 returns.
   - This is a different failure from the "critic collapses in wind" that D6 was looking for.

## Also checked
- **Baseline parity:** after the refactor, the baseline reproduces the pre-refactor V22 benchmark on all 400 episodes under v1 (`results/v23_baseline_parity.json`). Under v2 it matches 376 of 400; the difference is float32 rounding of the v2 frame. The evaluator therefore always flies the baseline on v1.
- **Spec claim that did not hold:** the spec said a heading-steering controller misses by clearly more than a course-steering one. It does not on closest approach: pure pursuit converges. It does on inbound track error, 25.6° against 2.7° median (`test_course_steering_cancels_drift`).

## Phase O is built anyway
The spec has it built on every branch: it removes the ±π seam and absolute yaw at no cost. The D1d result for V21 shows the shortcut is real for at least one of the trained models, even though it isn't what limits V22.
