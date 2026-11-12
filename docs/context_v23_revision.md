# Context: revising the V23 plan after the wind diagnostics

Follow-up to the earlier "RL glider policy can't handle steady wind" context and to the SPEC V23 plan written from it. That plan's main hypothesis has now been tested on the saved models, and it didn't hold. I need a revised plan.
- **Deadline:** resubmission on Wednesday 18 November.
- **Budget:** about four training runs of 6–8 hours each. I start the runs; the coding agent prepares and tests the code but never trains.

## Recap of the setup
- **Task:** PPO (SB3) on JSBSim. A hand-launched glider must return to its launch point and land on it.
  - About 25 m of altitude, launched 40–90 m out, flights of 40–60 s.
  - Actions: bank and airspeed commands.
  - Policy: 8 stacked frames, VecNormalize, MLP [128, 128].
  - Curriculum: stages 0–3 with wind maxima of 0, 3, 6 and 9 m/s. A quarter of episodes replay earlier stages.
- **Reward:**
  - Terminal: landing quality × precision, minus a crash penalty. Precision is `w_land·σ/(σ+d) + w_bullseye·exp(−(d/2)²)`.
  - Dense: `r_path = 0.10 × ground speed × dt` pays for metres of track in any direction, so the glider uses its altitude rather than diving in. A reachability barrier penalises flying out of glide range. There are small stall, bank and smoothness penalties.
  - Rewards reach PPO through a fixed scale of 0.02. γ = 0.999, target_kl 0.03, and the action standard deviation is floored at 0.2.
- **Baseline:** a scripted controller. It steers its GPS ground track at home, orbits to burn off excess height, then flies a final approach and flares.
- **Best model so far** (V22, 15M steps), success over 100 paired seeds:

  | Stage | Wind max | RL | Baseline |
  |---|---|---|---|
  | 0 | 0 m/s | 59% | 27% |
  | 1 | 3 m/s | 16% | 25% |
  | 2 | 6 m/s | 2% | 11% |
  | 3 | 9 m/s | 0% | 8% |

  V20, V21 and V22 all ended up with this same profile.

## The hypothesis that was tested and failed
The V23 plan assumed the policy steers its **heading** at home rather than its **ground track**. That works in calm air but is blind to cross-wind drift.

It proposed observation v2 (bearing error relative to ground track, crab angle, no absolute yaw), a wind-neutral path reward (pay air path instead of ground path), and contingent extras (a wind estimate as an input, and glide-cone potential shaping).

The tests below ran on saved models in steady wind (up to 3 m/s, nothing else), over 100 seeds paired with the baseline:

| Test | V22 | Reading |
|---|---|---|
| Replace yaw with course in the policy's input | Cross-wind excess recovered: −17% (the spec's threshold was 40%) | Feeding it course doesn't help |
| Replace course with yaw | Success 29% → 20%, p = 0.09 | Losing course barely matters either |
| Calm air, yaw biased by +15° | Track shift +4° [−1, 9] | V22 mostly isn't steering by heading. V21 was (+8°, success 58% → 26%) |
| Gradient influence, yaw vs course | 0.07 vs 0.09 | No yaw dominance |
| Fit the policy's MLP to the baseline's bank command from v1 inputs | Error 2.3° (v2 without previous action: 1.5°); bank spread 24° | v1 can represent the right law |
| Regress true wind from 8 stacked v1 frames | R² 0.7–0.87 | Wind is observable |
| Success by launch bearing (the ±180° seam) | No significant effect | — |
| Path reward per second against wind speed | Slope ≈ 0 | The path reward doesn't pay for downwind drift in practice: outbound and return legs cancel |

## What the data shows instead
1. **V22 lands downwind.**
   - In steady wind it touches down 21.5 m further downwind than the baseline on paired seeds (95% CI 5–38).
   - Its excess error is mostly along the wind: 22 m along against 10 m across.
   - Mean absolute errors: along-wind 48 m for V22 against 27 m for the baseline; cross-wind 31 m against 20 m.
2. **The policy flies a pattern, not at home.**
   - Its mean angle between ground track and home bearing is 38–42°, in calm air and in wind. The baseline's is about 0–6°.
   - The path reward pays for distance flown; the barrier only stops it flying out of glide range.
   - In calm air the pattern works: 59% against 27%. In wind the pattern drifts and the final glide ends long or downwind.
3. **The critic is miscalibrated, worst in calm air.**
   - Value-function R² against the realised discounted return is −1.16 in calm air, −0.36 in steady wind and −0.11 at stage 1.
   - At launch it estimates −0.18 against a realised +1.41.
   - The correlation between value and return is 0.1–0.2 at launch and 0.5 at 75% of the flight.
   - Likely cause: the model was saved during stage 2 training, with replay mixing stages, and the critic can't see the stage or the wind strength. Its baseline is pulled toward stage-2 returns. That would add noise to the advantage signal in every stage.
4. **Training at a stage doesn't produce lasting skill there.** V22's stage 1 training success hovered between 2% and 9% for about 10M steps. It eventually advanced on merit, but benchmarked at 16%.

## Already built and tested, available for any run
- Observation v2 behind a config flag. It's identical in flight and reward to v1, the baseline matches exactly, and V20–V22 stay loadable.
- The diagnostics module, which can rerun on any new checkpoint in about 20 minutes.
- Earlier fixes: no reward clipping, a KL limit, an action-noise floor, curriculum replay, crash-safe I/O.
- Not built yet:
  - **Phase R:** pay `r_path` on air path instead of ground path.
  - **Phase P.1:** a wind estimate as an observation, `w ← w + k(v_gps − V_cmd·[cos ψ, sin ψ] − w)`, with a time constant of 3–5 s.
  - **Phase P.2:** glide-cone potential shaping, `Φ = −k·max(0, d − G·h)` with G ≈ 8, applied as `F = γΦ(s′) − Φ(s)`. It stays zero while home is well within glide range.

## Constraints
- **No airspeed sensor.** The real airframe has no pitot tube.
- **No privileged information in the actor's observations,** because the policy must run on a Raspberry Pi. Privileged information in the critic alone would be allowed, but SB3's PPO shares one observation between actor and critic, so it needs a custom policy.
- **No "hurry home" progress reward.** It was removed earlier because it directly opposes using up altitude. Glide-cone shaping is acceptable as a bounded exception.
- **Each change sits behind a config key,** so every run is one named configuration.

## What I want help with
1. Given these findings, what's the most likely cause of the wind failure? The candidates are:
   - (a) pattern flying, paid by `r_path`, that doesn't adapt to drift;
   - (b) a noisy critic and advantage signal because the critic can't see the stage or wind;
   - (c) no learned into-wind final approach;
   - (d) something else.
2. Which cheap tests, with no training (saved models, rollouts, value probes), would separate these candidates?
3. A run plan for about four runs before 18 November.
   - Each run should be a named configuration with a falsifiable early-stop check in its first 2–3 hours.
   - The plan should include what goes into run 1 and why.
4. Specific questions:
   - Is glide-cone shaping (P.2) the right lever for the downwind and long landings?
   - Should `w_path` be lowered?
   - Is an asymmetric critic (wind and stage visible to the critic only) worth the implementation risk this close to the deadline?
   - Would training stages separately, without replay, fix the critic?
