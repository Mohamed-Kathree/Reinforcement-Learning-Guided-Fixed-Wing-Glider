# Context: RL glider policy can't handle steady wind

I'm training a PPO policy to fly a hand-launched fixed-wing glider back to its launch point and land on it. Three training runs in a row have hit the same wall: the policy beats a simple hand-written controller in calm air, but loses to it as soon as there is steady wind. I want to think through why before spending another 6–8 hours of compute on a training run.

## Setup
- **Simulator and algorithm:** JSBSim flight model, PPO from Stable-Baselines3.
  - 20 Hz policy, 2 actions: bank-angle command (±45°) and airspeed command. Inner-loop controllers fly the aircraft.
  - Episodes run about 25 m altitude, 40–90 m launch offset, and about 40–60 s of flight.
- **Observation:** 12 values, stacked over the last 8 frames, normalised with VecNormalize. MLP [128, 128].
  - `dx_home`, `dy_home` (GPS, metres)
  - ground speed, course angle (GPS track, raw radians in −π..π)
  - roll, pitch, yaw (IMU, raw radians)
  - baro altitude, vertical speed
  - ultrasonic height above ground plus a valid flag (below 4.5 m only)
  - required glide ratio (distance home ÷ altitude ÷ 15)
  - No airspeed or wind sensor, because the real airframe has no pitot tube.
- **Curriculum:** advancing a stage takes 35% / 30% / 15% rolling success at stages 0 / 1 / 2. A quarter of episodes replay earlier stages.

  | Stage | Wind | Gust | Success radius |
  |---|---|---|---|
  | 0 | 0 m/s | 0 | 25 m |
  | 1 | 3 m/s | 0.5 | 22 m |
  | 2 | 6 m/s | 1.0 | 20 m |
  | 3 | 9 m/s | 2.0 | 20 m |

- **Reward:**
  - **Terminal:** landing quality q (0–1, from touchdown speed, sink rate, roll and angle of attack) times a precision term, minus a crash penalty. Precision is `w_land·σ/(σ+d) + w_bullseye·exp(−(d/2)²)`, with σ = 10–15 m. The range is about −200 to +650.
  - **Dense terms, every step:**
    - `r_path = 0.10 × ground speed × dt`: pays for metres of ground track flown in **any** direction. It was meant to encourage using up altitude rather than diving straight in.
    - a reachability barrier, which only fires if the glider can no longer glide home;
    - stall, bank and smoothness penalties.
  - Rewards reach PPO through a fixed scale of 0.02 (no running normalisation). Settings: γ = 0.999, target_kl 0.03, and an action-noise standard deviation floor of 0.2.
- **Baseline controller:** steers its GPS **course** (ground track, not heading) toward the bearing to home, `atan2(−dy, −dx)`. It orbits when close, then flares using the ultrasonic sensor. Steering by course automatically cancels wind drift.

## Results
Each figure is the success rate over 100 seeds per stage, with episodes paired against the baseline on the same seeds.

| Stage | Best RL model (V22, 15M steps) | Baseline | Exact McNemar p |
|---|---|---|---|
| 0: calm | **59%** | 27% | <0.001 |
| 1: 3 m/s wind | 16% | 25% | 0.18 |
| 2: 6 m/s wind | 2% | 11% | 0.02 |
| 3: 9 m/s wind | 0% | 8% | 0.008 |

- **Same result three times.** V20 and V21 peaked at about 58% on stage 0 and 12% on stage 1, the same as V22.
- **A stage 1 ablation on V21** switched on one stage-1 factor at a time. Steady wind alone dropped the policy from 58% to 24% (the baseline got 39%). Gusts alone gave 37% (baseline 40%). Sensor noise and dropout barely mattered.
- **The wind miss is mostly sideways.** On stage 1 seeds, the V21 policy's median miss was 61 m against the baseline's 36 m. Its mean cross-wind touchdown error was 51 m against 22 m, while along-wind errors were similar.
- **Training at stage 1 does not fix it.** V22 trained there for 10M steps and its training success rate stayed between 2% and 9% for most of that time. It eventually advanced, but the benchmark shows the skill didn't stick.

## Already ruled out
Ruled out as the main cause, because each fix worked but the wind problem stayed:

- **Reward clipping.** Running reward normalisation was flattening the landing reward. It's now replaced by a fixed scale.
- **Unstable updates.** Update KL had reached 0.2–0.3 while action noise collapsed. Both are now held stable.
- **Forced curriculum advances,** and forgetting of earlier stages (fixed by replaying them).
- **Too little training.** V22 ran 20M steps.

## My candidate hypotheses (unverified)
1. **The policy has to compute the bearing home itself.** That means atan2 of dx and dy, minus the course angle, wrapped to ±π. Both angles are fed in raw, with a jump at ±π, so this is hard for a small MLP. In calm air, heading and course are the same, so a rough approximation works; in wind they differ by up to 10–15° of crab angle. The baseline is given exactly this quantity. A possible fix is to add sin/cos of (bearing to home − course) as an observation and to encode every angle as sin/cos.
2. **`r_path` pays for drifting downwind.** Ground speed is higher downwind, so wind makes undirected distance flown cheaper or more rewarding. That would push against holding a course into the wind.
3. **Credit assignment.** The only signal about accuracy arrives at touchdown, after about 1,000 steps. Correcting drift needs small adjustments early in the flight, and with γ = 0.999 and noisy returns those are hard to learn.
4. **Observation history.** Wind can only be inferred from course minus yaw, or from ground speed changing with heading, over 8 stacked frames (0.4 s). That may be too short a window.

## What I want help with
- Which of these hypotheses is most likely, and how to test each one cheaply, without a full training run? I can run the benchmark and analysis scripts on saved models.
- What change would most likely make wind handling learnable? For example:
  - better observation features (bearing error, crab angle, sin/cos angles);
  - directed shaping (potential-based progress toward home);
  - a residual policy on top of the baseline;
  - a recurrent policy.
- How to order the experiments, since each training run costs 6–8 hours.
