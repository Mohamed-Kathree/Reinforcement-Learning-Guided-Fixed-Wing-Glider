# SPEC V20 — decisions and modelling assumptions

Decisions taken while implementing SPEC V20 that change or extend the spec as
written. Each is stated as a modelling choice with its justification, for
carrying into the report.

## Yaw inertia: Izz = Ixx + Iyy (izz_factor removed)

**Assumption.** Yaw inertia is not an independent quantity. It is set to
Izz = Ixx + Iyy for every sampled airframe.

**Justification.**

- A rigid body satisfies Ixx + Iyy ≥ Izz, with equality only when all mass
  lies in the x–y plane (a flat plate). The originally specified
  `izz_factor` range (0.95, 1.05) was unphysical in its upper half and has
  been removed from `mass_properties.yaml` and from the domain-randomisation
  table.
- Equality is the physical upper bound on yaw inertia: the most sluggish yaw
  response, and so the harder aircraft to control. Training against this end
  is the conservative choice.
- Both estimates put the real ratio Izz / (Ixx + Iyy) between 0.90 and 1.00
  (CAD 0.28 / 0.28 = 1.00; V19 0.45 / 0.50 = 0.90). The assumption gives up
  at most about 10% on one parameter.

**Mechanism.** The aircraft file bakes in Izz = Ixx_lo + Iyy_lo. Each
per-episode inertia point-mass pair adds the same amount to Izz as to Ixx
(the y-pair) or Iyy (the x-pair), so the relation holds at every sample.

**Known extension, deliberately unbuilt.** A vertical point-mass pair at
z = ±d adds 2md² to Ixx and to Iyy but nothing to Izz. That lowers the ratio
below 1.0, the only physically available direction. Adding it would give the
solver a fifth freedom and make the ratio randomisable over roughly
0.90–1.00. It was left out to avoid a solver change late in the schedule.

## Signed ballast

At heavy, high-inertia targets the inertia pairs alone weigh more than the
mass budget, and the ballast point mass goes negative. Examples:

| Target (mass, Ixx, Iyy) | Ballast |
|---|---|
| 1.64 kg, 0.21, 0.16 | about −0.20 kg |
| 1.80 kg, 0.32, 0.26 | about −0.57 kg |

The base empty weight stays at a realistic structural figure, the low end of
the mass range. Lowering it to keep the ballast positive would leave the
baked-in base inertia inconsistent with the base mass.

This is safe because the solver (`sim/mass_solver.py`) optimises against
JSBSim's own reported mass properties, not a parallel-axis formula. Every
accepted solution is checked by reading back what JSBSim reports, and is
rejected (`UnreachableTarget`) if:

- any achieved inertia (Ixx, Iyy, Izz) is not positive (JSBSim accepts a
  negative inertia silently; a single −0.2 kg mass at y = 0.9 m drove Ixx to
  −0.018 kg·m² in testing);
- achieved total mass is not positive, or misses its target by more than 1%;
- any point mass exceeds 1.5 kg in magnitude (a conditioning guard).

Signed ballast also lets the solver reach targets well outside the training
ranges (for example a 200 mm CG shift at 1.64 kg). The guards check physical
validity, not plausibility. The domain-randomisation ranges are what keep the
sampled airframes plausible.

## ixz: fixed at the midpoint, sensitivity measured rather than randomised

The CAD rebuild gives +0.0036 kg·m² and V19 gives −0.008 kg·m², which
disagree in sign. The YAML carries
`ixz: {value: 0.0, range: [-0.010, 0.010], source: estimated}`. The previous
aircraft file pinned +0.01.

**Decision.** ixz is fixed at 0.0, the range midpoint, in the training
aircraft, and is not in the domain-randomisation table. Its sensitivity is
quantified in the Phase 5 gate sweep instead.

**Justification.**

- The term is small: |ixz| ≤ 0.010 against Ixx ≥ 0.14, under 7%.
- The two estimates straddle zero, so the midpoint is the value that favours
  neither.
- Inertia is read-only in JSBSim, and the point masses (all at z = 0)
  produce no ixz. Randomising it would mean switching between several
  loaded aircraft files per environment, a structural change to the FDM
  wrapper.
- The measured effect is inside the chaotic noise band. On the 20 fixed-seed
  baseline episodes:

  | Change | Mean shift | Worst single episode |
  |---|---|---|
  | ixz 0.0 → +0.010 | −0.66 m | 12.1 m |
  | ixz 0.0 → −0.010 | +0.15 m | 2.6 m |
  | CD0 −1e-4 (0.4% of CD0) | −0.59 m | 11.8 m |
  | CL × (1 + 1e-4) | −0.06 m | 0.9 m |

**Mechanism.** `python aircraft/generate_fdm.py --ixz-variants` writes
aircraft files at the range low, value and high to `analysis/fdm_variants/`
(gitignored). The Phase 5 sweep flies them sequentially. The environment
never loads them.

## Lift: peak and slope set independently

SPEC V20 lists `aero/clmax-scale` as a multiplier on the CLalpha function.
Applied literally, it would multiply the whole CL table, which is what
`aero/cl-mult` did. The two parameters would then be indistinguishable, and
the Phase 7 regression could not separate them.

Both now act on the table's lookup angle:

  CL(α) = s · table(α · m / s),  with s = `clmax-scale` and m = `cl-mult`

| Parameter | Pre-stall slope | CLmax | Stall angle |
|---|---|---|---|
| `clmax-scale` s | unchanged | × s | × s / m |
| `cl-mult` m | × m | unchanged | × 1 / m |

Holding the slope fixed while lowering the peak requires the stall angle to
drop: the two cannot move independently. The coupling is therefore the only
physically consistent choice, not a convenience. At s = m = 1 every
operation is exact in floating point, so the defaults reproduce the original
table (checked to 1e-9 in `tests/test_dr_hooks.py`).

**Why `cl-mult` became slope-only (Phase 4).** As a whole-table multiplier
it compounded with `clmax-scale` at the stall corner: peak lift × 0.68,
beyond both estimates. That put the worst-corner stall at 10.11 m/s, above
the 9.5 m/s command floor. Slope-only matches the property's original
documented meaning ("CL slope multiplier") and makes the two parameters
orthogonal. The worst-corner stall is now 9.17 m/s, and `cl_mult` moves it
by under 0.01 m/s.

**Table extended so the lookup never clamps.** The scaled lookup
α · m / s reaches the original ±25° table end at only 17.7° of true alpha
at the `cl-mult` 1.2 / `clmax-scale` 0.85 corner. At stage 3 the baseline
crossed it in 14 of 30 episodes (2.8% of airborne steps; 99th-percentile
|α| 32.8°). A clamped deep-stall lift would make the worst airframes behave
better than they should, exactly where the baseline crashes. With |α| ≤ 180°
and m / s ≤ 1.2 / 0.85, the lookup reaches at most ±254°, so the table now
runs to ±260° in 5° steps (109 rows).

Every new row comes from the formula that generated the original 15, which
it reproduces to 4 dp (the original `sim/aerodynamics.py` `CL()`):

  CL = a0 · α for |α| ≤ 12°, otherwise sign(α) · max(1.1519 · e^(−3 · excess), 0.9 · sin 2α)

Linear interpolation on the 5° grid is within 0.016 of the formula.
`tests/test_dr_hooks.py` checks the bound against `DR_RANGES` and
empirically, over full-range baseline and random-policy episodes.

## Control effectiveness: per surface

The retired single control multiplier scaled seven terms. Each now belongs
to the surface that produces it, and the properties are named for the
surface (Phase 4 rename) so the Phase 7 regression table reads the way the
analysis works:

| Multiplier | Surface | Terms |
|---|---|---|
| `ail-eff-mult` | aileron | Clda, Cnda (adverse yaw) |
| `elev-eff-mult` | elevator | Cmde, CLde, and the elevator term in CLtotal |
| `rud-eff-mult` | rudder | Cndr, Cldr |

**Why control derivatives group and stability derivatives do not.** A
servo, linkage or surface-area error affects every term its surface
produces, coherently. An error in a stability derivative (Cmα, Clp, Cnβ and
so on) has no such common cause. So control terms are grouped by surface,
and the stability derivatives keep independent multipliers.

**Effect on the training distribution.** Three independent ±40% draws
produce airframes a single common multiplier never could. Aileron at 1.4
with rudder at 0.6 is a genuinely odd aircraft, not just a strong or weak
one. This is intended, since it covers per-linkage build error, but it widens
the effective difficulty beyond what the old single 0.75–1.10 range implied.
Expect lower success than the single-multiplier equivalent, and do not read
that as a regression.

## Domain randomisation table (Phase 4)

`env/curriculum.py` `DR_RANGES` / `DR_ONESIDED` / `DR_SCALE`. Mass, CG, Ixx,
Iyy and wing area are read from `aircraft/mass_properties.yaml`.

- **Stage scaling.** Each stage samples every range at
  `DR_SCALE[stage]` = 0.25 / 0.50 / 0.75 / 1.00 of its width. Two-sided
  ranges shrink about their midpoint. The one-sided `cd0_add` shrinks toward
  0, so the model as written is always included.
- **Not randomised:** Izz (follows Ixx + Iyy), ixz (fixed, measured in the
  sweep), and `cm0_offset` (the hook exists, but no range was specified).
- **Iyy sampled from 0.110, not 0.10.** This is a narrowing of the *sampled*
  range only, recorded as `dr_range` with its `dr_reason` in the YAML. The
  knowledge range stays [0.10, 0.26], and the aircraft file keeps 0.10 as
  its base. 0.33% of full-range samples were unreachable by the point-mass
  solver. All of them were heavy, with |CG| ≥ 9 mm and Iyy ≤ 0.105. Moving
  the CG with ballast near the reference point always adds pitch inertia,
  and the pairs can only add, so the lowest Iyy cannot be combined with a
  large CG shift. 0.110 removes every failure and still brackets both
  estimates (CAD 0.12, V19 0.22). The unreachable region is a *joint*
  constraint on mass, CG and Iyy; narrowing Iyy globally rather than
  adding a joint rejection rule is a deliberate simplification. It gives up
  the bottom 0.010 of Iyy for every airframe, including the many that could
  reach it, in exchange for a plain rectangular range with no rejection.
- **Solver cache.** It rarely hits in training, because the targets are
  continuous. It earns its place in evaluation: the Phase 5 sweep corners
  and the 500 frozen held-out airframes repeat exactly. The hit rate is
  deliberately not optimised. A Phase 2 bug was fixed here: a cache hit
  returned the cached weights without writing them to the FDM, so after a
  reset every repeated target flew the bare 1.50 kg aircraft. It was found
  when the stall-speed calculation disagreed with itself.

## Speeds, flight times and truncation (Phase 4)

All values are from `analysis/flight_envelope.py`, which applies the sampled
parameters through the environment's own `apply_domain_rand()` and reads
lift and drag back from JSBSim. They are recorded in
`analysis/flight_envelope.json`, in `base.yaml`'s `domain_rand` section, and
as `env/glider_env.py` constants; `tests/test_domain_rand.py` recomputes and
checks them.

| Quantity | Value |
|---|---|
| Command range | 9.5–16 m/s |
| Worst-corner stall (1.80 kg, `clmax-scale` 0.85, area × 0.95) | 9.17 m/s |
| Nominal best L/D | 16.75 at 9.1 m/s (16.64 at the 9.5 m/s floor) |
| Longest flight: lightest/cleanest at min sink in range | 59.5 s from 30 m; **63.1 s** from the highest start energy (31.8 m) |
| Shortest flight: heaviest/draggiest at its best L/D | **37.5 s** from 30 m (36.1 s before Phase 5's `cd0_add` narrowing) |
| Expected flight: nominal airframe at best L/D in range | 52.5 s from 30 m |
| Truncation | 1520 steps = 76 s = 1.2 × longest |

The earlier estimates (36 s longest, 28 s shortest, truncation at 45 s)
assumed flight at the elevator-zero trim speed. The policy can command any
speed in the range, and the lightest airframe's minimum sink inside it is
0.50 m/s. Truncating at 45 s would have cut off the best airframes.
Episodes start at 20–25 m (stage `alt0_m`), so real flights are shorter
than the 30 m figures.

**Nominal glide speed not changed to 14.5 m/s.** In this aircraft model best
L/D is at 9.1 m/s (1.64 kg). At 14.5 m/s L/D is 11.35, about 32% less
range. `V_TRIM` is the inner loop's pitch-schedule reference, an empirically
tuned pair with `ALPHA_TRIM`, not a glide speed. Setting it to 14.5 made
measured speed tracking worse and non-monotonic: a 7 m/s command flew
faster than a 9 m/s one. It stays at 10. The baseline's speeds were instead
fitted to the polar: cruise 10.0 m/s (within 2% of best L/D), with
min-sink and flare on the 9.5 m/s floor (previously 9.0 and 7.0, below the
new floor).

**Touchdown-speed reward band recalibrated (reward change).** `vtd_ok` /
`vtd_bad` were 9.0 / 13.0 m/s: 1.4× and 2.0× the 1.1 kg design's 6.35 m/s
stall. Above the 9.5 m/s floor every touchdown is at 10–11 m/s, which
capped landing quality near 0.6 for every policy and airframe. The
baseline's mean quality fell from 1.0 to 0.61, with sink, roll and alpha all
clean. The same rule applied to the as-built nominal 7.75 m/s stall gives
10.85 / 15.5 m/s, changed in `env/reward.py` and `base.yaml` together. The
baseline's stage-0 quality returns to 1.00.

**Launch height not changed.** `alt0_m` (20–25 m) still reflects the
hand-launch assumption. The hi-start release height has not been measured.

## Frame stacking (Phase 4)

`training/train.py` `build_venv()` stacks `ppo.n_stack` = 8 frames
(0.4 s at 20 Hz) for both the training and eval environments, so their
observation shapes cannot diverge. `VecNormalize` stays outermost, so
`eval_venv.training = False` and `VecNormalize.save()`/`load()` act on it
directly.

The other places that load a policy (`training/evaluate.py`,
`backend/recorder.py`, `analysis/plot_trajectories.py`) build unstacked
environments. That is correct for the existing 12-observation
`deployment/` model and wrong for any model trained after Phase 4. Phase 6
replaces them with one evaluator, which must build its environment through
the same stacking.

## Geometry

Wing area (`metrics/Sw-sqft`) is writable at runtime and survives
`run_ic()`. Span (`metrics/bw-ft`) and mean chord (`metrics/cbarw-ft`) are
not: writes silently do nothing on JSBSim 1.3.1. Span and chord are
therefore fixed at their YAML values, and only area is randomised.

## Glide-ratio gate: 5 s window is start-speed sensitive

`validate_glide.py` Gate 2 takes the median glide ratio over the first 5 s.
On the same airframe it read 12.87 when started 1.8% below trim speed and
21.6 when started 5% above. The steady value (30–60 s window) is 14.1
regardless of start speed. The V19 figure of 12.87 was a transient artifact
of the fixed 10 m/s start. The start speed has been derived from mass since
Phase 1; Phase 5 adds a settling period before the measurement.

## Phase 5: physics gates at the range corners

`python validate_glide.py --sweep` gates 45 configurations: nominal, every
single-parameter extreme (each DR range at its low and high end, others
nominal), the heaviest-and-draggiest and lightest-and-least-stable corners,
and the two ixz variants. `--random 300` gates 300 airframes sampled the
way the environment samples them (seed 20261110), because the sweep does
not cover joint combinations. Results are committed as
`analysis/gate_sweep_{elevator,fixed,random}.json`.

**Nominal** means the aircraft as modelled: aerodynamic hooks at the
aircraft file's defaults, and mass, CG and inertia at the YAML nominal.

### Gate definition: trimmed for speed, not fixed at zero elevator

The original gates start every airframe from its trim with the elevator
command at zero. For sampled airframes that conflates two things:

- **Mis-trimmed.** CG ±15 mm alone moved the zero-elevator trim to the stall
  (aft: 12.95°, 8 m/s, L/D 1.3) or to 16.3 m/s (forward: L/D 9.7). Both
  airframes are statically stable and trim with about 9% elevator.
- **Uncontrollable.** No stable trim at a flyable condition exists at all.

The gates now trim each airframe's elevator so it flies at the nominal
airframe's trim **lift coefficient** (CL 0.458), the way a glider is trimmed
on the field for speed. The trim must use no more than 50% of the elevator
throw. Gates 2, 4 and 5 then test the stick-fixed glide and stability there.

An intermediate version trimmed every airframe to the nominal trim
*alpha*. That left 11 of 300 random airframes just under L/D 10, all with a
low lift slope and high drag. Trimmed to one alpha, a low-slope airframe
flies at CL 0.37, fast and far from its best glide. Trimming to one CL
removed every one of those failures. Neither narrowing boundary below
changed between the two references.

The zero-elevator definition is still available (`--trim fixed`), and its
sweep is committed for the record: 42 of 45 pass, failing exactly the two
CG extremes and the least-stable corner.

### Results and narrowings

Before narrowing, the elevator-trimmed sweep failed three configurations:

| Config | Gate | Value | Cause |
|---|---|---|---|
| `corner_lightest_least_stable` | trimmed stability | no stable pre-stall trim | negative static margin: CG 15 mm aft, `cmalpha_mult` 0.75, `cl_mult` 1.2 |
| `cd0_add_hi` (0.015) | glide ratio | 9.7 | drag |
| `corner_heaviest_draggiest` | glide ratio | 9.2 | drag |

Rule applied: narrow the offending range (preferred) rather than reject
and resample. The boundary of each was found by bisection with the gates
themselves.

| Range | Was | Now | Boundary | Recorded in |
|---|---|---|---|---|
| `cg_x_m` (+fwd) | −0.015 – +0.015 | **−0.013** – +0.015 | corner passes with CG ≤ 13.15 mm aft | YAML `dr_range` + `dr_reason` |
| `cd0_add` | 0 – 0.015 | 0 – **0.010** | corner passes with `cd0_add` ≤ 0.0109 | `env/curriculum.py` + here |

`cd0_add` is not a mass property, so its narrowing is recorded in the
curriculum table's comment and in this document rather than in
`mass_properties.yaml`.

After narrowing: **45 / 45** sweep configurations and **300 / 300** random
full-range airframes pass all five gates. Across the random set the glide
ratio runs from 10.27 to 15.16, and trims need at most 21% of the elevator
throw.

**Reading for the stage-3 baseline crashes.** The baseline crashed 10 of 20
stage-3 episodes before this phase. Only one corner of the ranges failed a
stability gate, by a small margin (13.15 mm against 15 mm). So the ranges
did not, by and large, contain uncontrollable aircraft. The crashes come
from the baseline's fixed gains meeting widely varied airframes, which is
genuine separation between baseline and policy rather than a "task too
hard" artefact. The re-measured baseline is in
`analysis/benchmark_baseline_v20.json`.


### Baseline benchmark (Gate 6)

`python -m training.benchmark --policy baseline --episodes 100`, the single
evaluator, at commit 3a8131e (clean): DeterministicRTL, 100 episodes per
stage, a fresh env per episode, Wilson 95% intervals. Recorded in
`analysis/benchmark_baseline_v20.json`. **This is the reference baseline
number.** The old 28–29% figure is void (GPS velocity bug).

| Stage | Success | Soft landing | Crash | Timeout | Mean quality |
|---|---|---|---|---|---|
| 0 | 27% [19.3, 36.4] | 73% | 0% [0, 3.7] | 0% | 1.00 |
| 1 | 25% [17.5, 34.3] | 75% | 0% [0, 3.7] | 0% | 0.91 |
| 2 | 11% [6.3, 18.6] | 82% | 6% [2.8, 12.5] | 1% | 0.69 |
| 3 | **8% [4.1, 15.0]** | 52% | **39% [30.0, 48.8]** | 1% | 0.32 |

Cross-checks:

- **Bucket logic and environment construction carried over intact.** Run
  with one env reused per stage (as the interim post-Phase-5 script did),
  the evaluator reproduces that script's 400 episodes exactly: same seeds,
  airframes and outcomes, zero difference in miss distance.
- **The fresh-env rule changes one episode in 400.** A stage-3 soft landing
  becomes a crash: one chaotic episode was sensitive to the ~1e-15 JSBSim
  state left over from earlier episodes. With a fresh env, every result
  depends on its seed alone.

No single airframe parameter separates the stage-3 crashes from the other
episodes by more than 0.42 standard deviations (largest: `clbeta_mult`,
`mass_kg`, `cl_mult`). That is consistent with the sweep's verdict: hard,
widely varied but flyable airframes, plus stage 3's wind and gusts, rather
than one uncontrollable corner. Per-episode `dr_params` are in the JSON for
the Phase 7 regression.

## Phase 6: one evaluator

`training/benchmark.py` replaced `training/evaluate.py` and five scratch
benchmark scripts. Two bugs were retired with them:

- **Soft landings counted as timeouts.** `evaluate.py` set
  `timeout = not success and not crash`, so every soft landing landed in
  the timeout bucket.
- **Paired runs that were not paired.** `evaluate.py` and
  `plot_trajectories.py` reset the RL VecEnv a second time without the
  seed, so the policy and the baseline never flew the same episode.

How it works:

- All controllers step one loop on a plain `GliderEnv` built as training
  builds it.
- A trained policy is fed through `PolicyActor`, which reproduces the
  training observation pipeline (`VecFrameStack` then `VecNormalize`) with
  `n_stack` read from the model. A test matches it to SB3's wrappers
  observation by observation. Both the pre-V20 12-observation model and
  frame-stacked models run.
- There is one outcome definition, `env/curriculum.py`
  `outcome_bucket()`, used by training, the dashboard and the evaluator.

**Ablation caveat.** `--ablate-home` zeroes obs 0, 1 and 3 (dx_home,
dy_home, course angle), as specified. obs[11] (required glide ratio =
distance to home / altitude) still carries the *distance* to home, though
not its direction. The ablation therefore removes home direction but not
home distance.

## After the first V20 run: curriculum and model selection

**What the run showed.** The 20M-step run (stopped at 18.6M by a locked
dashboard log; fixed in `7f3e924`) finished below the baseline at every
stage. A Stage 0 sweep of its checkpoints (100 paired episodes each,
`results/benchmark_v20_*_stage0.json`) located the cause:

| Checkpoint | Training stage | Stage 0 success (baseline 27%) |
|---|---|---|
| 1.5M | 0 | 24% |
| 2.0M | 0 -> 1, earned | **51%** (McNemar p = 0.0009) |
| 3.5M | 1 | 31% |
| 4.0M | 2, forced at 3.8M | 8% |
| 18.6M | 3, forced at 5.4M | 4% |

The policy learned the calm-air task and then lost it. Stage 1 never
reached the 0.60 threshold, both later advances were forced, and nothing in
training kept earlier stages in the distribution. The Stage-3-pinned
EvalCallback kept 9.0M as `best_model.zip`; it scored on reward, which
stayed flat (-110 to -200) for the whole run.

**Changes** (tests: `tests/test_curriculum_v21.py`):

1. **Replay.** `curriculum.replay_frac: 0.25`. Once past Stage 0, each
   training episode runs at an earlier stage with that probability, drawn
   uniformly (`env/curriculum.py` `episode_stage()`). `GliderEnv`
   `set_curriculum()` makes the draw at reset, from a separate RNG stream
   so the episode's own random numbers are unchanged. The stage is
   reported as `info['episode_stage']`. `CurriculumCallback` records only
   current-stage episodes toward an advance. As a side effect, episodes
   still in flight from the old stage when an advance lands are no longer
   credited to the new one. With `replay_frac` 0, or without
   `set_curriculum`, episodes are bit-identical to before (tested).
2. **No forced advances.** `max_episodes_at_stage: null`. A stall is a
   result to report. The original reason for forcing, model selection at
   Stage 3 by a policy that never trained there, is gone with (3).
3. **All-stage model selection.** `MultiStageEvalCallback` replaces SB3's
   `EvalCallback`. Every 250k steps it flies 10 deterministic episodes per
   stage through the single evaluator, scores the policy by mean success
   (ties go to the lower crash rate), and writes `best_model.zip`,
   `vecnormalize_best.pkl`, `best_model.json` and
   `eval_logs/multistage.jsonl`. The selection seeds (`eval.seed: 1000`)
   are disjoint from the benchmark's (seed 0), so selection never sees the
   test episodes. One evaluation takes about 34 s, roughly 45 min over a
   20M-step run. `best_model.json`'s figures are a max over 40-episode
   estimates and run high; report the 100-episode benchmark.
4. **Per-stage thresholds.** `advance_threshold: [0.35, 0.30, 0.15]` for
   leaving stages 0, 1 and 2, each above the baseline's 27% / 25% / 11%. A
   test pins that ordering. `rolling_window` goes back to 100 (about ±9
   points rather than ±13).

**Expectation, untested.** Under these rules the first run would have
earned Stage 1 (its best Stage 0 window was 38%) and then held at Stage 1
(best window 28% < 0.30). Whether replay and the longer stay lift Stage 1
past 0.30 is what the next run measures.

## Training defects D and E (found from the V20 and V21 runs)

The V21 run (curriculum changes above) earned Stage 1 at 1.5M steps, held
there for 14M steps at 5-14% rolling success, and degraded from its 3.5M
peak. It was stopped at 15.7M. `best_model.zip` (3.5M) benchmarks at 58%
Stage 0 success against the baseline's 27% (p < 0.001) and 12% against 25%
at Stage 1 (`results/benchmark_v21_best_3.5M.json`). The curriculum
changes did their job (no forced advance, peak kept) but did not remove the
cause. Two defects in the training setup did the damage. Neither is in
`env/reward.py`, and no reward unit test could have seen either.

### D. Reward clipping flattened the precision reward

`VecNormalize(norm_reward=True)` divides each reward by a running standard
deviation of the discounted return, then clips at ±10. That running return
resets every episode and receives the landing reward only on its last
step, so its standard deviation tracks the small dense terms, not the
landing reward. Measured from the saved `vecnormalize_<step>.pkl` files and
the training stream (V21):

| Steps | Return std | Raw rewards clipped beyond | Clean landings all paid the same inside | Success radius |
|---|---|---|---|---|
| 0.2-0.6M | 25.6 | ±256 | 14 m | 25 m |
| 0.8-1.4M | 19.1 | ±191 | 24 m | 25 m |
| 1.6-2.4M | 16.4 | ±164 | 27 m | 22 m |
| 3.0-4.0M | 14.6 | ±146 | 31 m | 22 m |
| 14.0-15.7M | 12.8 | ±128 | 38 m | 22 m |

The landing reward runs to +650 and a crash is -200. From 1.6M steps on,
the flat disc was wider than the success radius: landing on target paid
exactly what landing 30 m out paid. Crash penalties were clipped in 6-14%
of episodes. This is Defect B's symptom (no usable gradient where the
policy lands) arriving through the wrapper instead of the reward formula.

The Term 3 run had the same setting but a return std of 48.6 (clip at
±486), so it was effectively unclipped. The V20 changes lowered the return
spread enough for the same limit to bind.

**Fix.** `ppo.normalize_reward: false` and `ppo.reward_scale: 0.02`
(`training/train.py` `VecScaleReward`): one fixed constant, nothing
clipped, and a scale that does not drift with the curriculum. PPO
normalises advantages per batch, so the constant only sets the value-loss
scale. `learner_reward()` computes what PPO receives for any raw reward;
`tests/test_training_defects.py` reproduces the old flat disc with it and
requires the configured pipeline to pay strictly more for every half-metre
closer at every stage, and to pass the crash penalty in proportion. A
resume takes the normalisation flag from the config, not the checkpoint.

### E. Unbounded PPO updates as the action noise collapsed

From the training streams (`data/runs/*.jsonl`) and the saved policies:

| Steps | Action std | Update KL | Samples clipped |
|---|---|---|---|
| 0-1.5M | 1.0 -> 0.6 | 0.016 | 15% |
| 1.5-3M | 0.5 -> 0.3 | 0.034 | 27% |
| 3-4.5M | 0.25 -> 0.18 | 0.060 | 36% |
| 9-10.5M | 0.11 | 0.135 (V21), 0.280 (V20) | 50-57% |
| end | 0.08 | 0.19 (V21), 0.30 (V20) | 55-60% |

A fixed-size step in the action mean costs KL in proportion to
(step / std)², so at a constant learning rate the update grew as the noise
shrank, to 10-20 times the usual 0.01-0.03. Both policies peaked at 2-3.5M
steps, while KL was still 0.04-0.06, and degraded after. V21 shows it is
not the curriculum: it degraded while held at one stage.

**Fix.** `ppo.target_kl: 0.03` (SB3 stops a rollout's epochs at 1.5 times
that) and `ppo.min_action_std: 0.2` (`ActionStdFloorCallback` clamps the
policy's log-std before each rollout and logs `train/action_std_<k>`).
Below the limits, which is roughly the first 2M steps, training is
unchanged. `build_model()` is now the one place PPO is constructed.

### What was ruled out

- **Discounting.** gamma is 0.999, a horizon of about 50 s; the landing
  reward reaches the early decisions.
- **Raw angles in the observation.** Success does not depend on the
  direction home (40-58% in every sector, the same pattern as the
  baseline), so the ±180° wrap in course and heading is not the cause.
- **The path reward in wind.** The policy collects less path reward than
  the baseline in wind (about 29 vs 35 per episode) and flies the same
  share of its track downwind (51-59% vs 53-60%).
- **Missing wind information.** Course and heading are both observed
  (their difference is the crab angle), and ground speed and vertical
  speed give the achieved glide ratio over the ground directly.

### What Stage 1 does to the policy

`analysis/stage1_ablation.py`, V21 best model, Stage 0 plus one group of
Stage 1 settings (`results/ablation_v21_stage1_factors.json`):

| Switched on | RL success | Baseline | RL median miss |
|---|---|---|---|
| nothing (Stage 0) | 58% | 27% | 21 m |
| landing grading | 58% | 27% | 21 m |
| sensor noise + dropout | 54% | 28% | 23 m |
| airframe randomisation | 44% | 26% | 27 m |
| target geometry | 40% | 19% | 26 m |
| gusts only | 37% | 40% | 34 m |
| steady wind only | 24% | 39% | 51 m |
| wind + gusts | 18% | 25% | 61 m |
| all of Stage 1 | 12% | 25% | 60 m |

Steady wind is what the policy cannot handle; everything else leaves it
ahead of the baseline. The baseline steers its GPS ground track at home,
which cancels drift by construction. Whether the policy learns the same
once D and E are fixed is what the next run tests: at Stage 1 the reward
was flat inside 27-38 m for the whole of its stay, so it was never paid to
close the last 30 m in wind. No observation or action change has been
made.

### Also changed

- Checkpoint, VecNormalize, curriculum-state and final-model writes go
  through `retry_io()` (four attempts, then a warning and a skipped file),
  the same exposure to a locked file that ended the first run.
- `tests/test_training_defects.py::test_training_loop_end_to_end_smoke`
  runs `train()` itself for about 1,000 steps into a temp folder with
  every callback attached.
- Not changed: the smoothness penalty is still per step, not scaled by
  `dt_rl` like the other dense terms. It taxes exploration noise (about 7%
  of the early reward budget), but it is also the only thing discouraging
  bang-bang commands, so it stays until a run shows it needs to move.

### What to watch in the next run

| Signal | Healthy | Where |
|---|---|---|
| `train/approx_kl` | stays under about 0.045 | TensorBoard, dashboard |
| `train/clip_fraction` | under about 0.3 | TensorBoard, dashboard |
| `train/action_std_0`, `_1` | at or above 0.2 | TensorBoard |
| Stage 0 eval success | passes the old 50-60% peak and holds | `eval_logs/multistage.jsonl` |
| Stage 1 rolling success | climbs past the old 12% | `[Curriculum]` lines, dashboard |

## V23 Phase O: observation v2

V22 finished at 20M steps. Its best model (15M) benchmarked the same as V21's:
stage 0 59% against the baseline's 27%, stage 1 16% against 25%, and worse
than the baseline at stages 2 and 3. SPEC V23 infers that observation v1
lets the policy steer its heading instead of its ground track, which is
right in calm air and blind to cross-wind drift. Phase D tests that
inference on the saved models. This section records how Phase O was built.

### Design

- `env/nav_features.py::nav_features` is the one geometry function. The
  environment's v2 frame, the baseline (`guidance()`) and the evaluator all
  go through it.
- v2 is computed from the delayed v1 sensor vector plus the previous action
  (`obs_v2_from_v1`). Both versions see the same readings, noise, dropouts
  and delays. The reward keeps reading the v1 vector, so changing the
  version never changes what the policy is paid.
- The 17 values are:
  - `along` and `cross` to home (÷100 m);
  - sin and cos of (bearing to home − course);
  - ground speed (÷10 m/s);
  - sin and cos of (course − yaw);
  - `course_valid`;
  - roll and pitch;
  - baro altitude (÷30 m) and vertical speed;
  - ultrasonic height and its valid flag;
  - required glide ratio;
  - previous bank and speed action.
- Raw `dx`, `dy`, course and yaw are gone.
- `observation.obs_version` in `base.yaml` defaults to 1. `GliderEnv`
  reads it once, at construction.
- `_build_obs` still builds the v1 sensor vector. A new `_policy_obs`
  dispatches on the version.
- Each model carries its version in its saved attributes (`obs_version`).
  Models without that attribute are v1.
  - `PolicyActor` reads the version and raises a clear error if it is
    handed the other one.
  - `--resume` checks the checkpoint's version against the config before
    loading.
  - The backend recorder builds the environment in the model's version.

### Departures from the spec

- **The baseline always reads v1 in the evaluator,** whatever the policy's
  version. v2 frames are float32 and scaled, and that rounding moved one
  stage-3 touchdown by one step. The physical episode is the same under
  both versions, so pairing holds, and every benchmark's baseline column
  stays identical to the V20–V22 ones. The baseline also flies v2 frames
  correctly (`test_baseline_flies_v2_too`).
- **The closed-loop drift test steers on `atan2(sin_err, cos_err)`, not
  `sin_err` alone.** sin_err is zero when home is directly behind, so a
  proportional law on it never turns round: seed 8 flew away and got no
  closer than 53 m.
- **"The heading-based controller misses by clearly more" did not hold.**
  Pure pursuit reaches home whenever airspeed exceeds the wind: closest
  approach was at most 7.4 m over 12 seeds, against 1.6 m for course
  steering. What separates the two laws is the inbound track error. In a
  3 m/s crosswind its median is 2.7° when steering the ground track and
  25.6° when steering the nose. The test asserts that instead.

### Tests

`tests/test_nav_features.py` covers:
- bearing-error parity over 10,000 states;
- the home-right and crab signs;
- rotation invariance, and no seam at ±π;
- the low-speed fallback;
- the v2 layout;
- the turn sign in JSBSim;
- the crosswind drift test.

`tests/test_obs_v2.py` covers:
- the spaces for each version;
- v2 as a re-encoding of v1, with identical flight and reward;
- previous-action consistency;
- stack, VecNormalize and save/load shapes;
- loading V22 as v1;
- the version-mismatch errors in the actor, the environment and `--resume`;
- baseline parity against the pre-refactor V22 benchmark log.

`analysis/v23_baseline_parity.py` runs the parity check on all 100 seeds
of all four stages.

## V23 revision: run 1 levers

The tests behind these are in `docs/V23_horizon_tests_summary.md`. Every
lever sits behind a key whose default is the V22 behaviour.

- **`reward.path_reward_frame: ground | air`.** With `air`, the reward pays
  airspeed × dt: the same in every direction and every wind.
- **`reward.cone_shaping`.** Potential-based shaping,
  `F = γΦ(s') − Φ(s)` with `Φ = −w_cone · deficit` and Φ = 0 at the end of
  the episode.
  - The deficit is the time to reach home through the moving air, at
    10 m/s, minus the time the altitude buys at `cone_glide_ratio`,
    converted to metres.
  - It uses the true mean wind, which is allowed in the reward. The policy
    never sees it.
  - In calm air it reduces to max(0, d − G·h).
  - The discounted sum of F over an episode equals −Φ(s₀); the
    telescoping test checks this.
  - This is a deliberate, bounded exception to the Phase 4 "no progress
    reward" rule: the shaping is zero while home is comfortably in glide,
    so it never says "hurry home".
  - `cone_gamma` must equal `ppo.gamma`; a test checks this.
- **`observation.obs_version: 3`.** v2 plus the wind estimate from
  `env/wind_estimate.py`, as tailwind and crosswind relative to the track,
  each ÷10 m/s.
  - The estimator has three states: wind north, wind east, and the speed
    loop's airspeed error.
  - It skips updates while the heading turns faster than 15°/s.
- **`curriculum.stage0_wind_ms`.** A training-only wind maximum at stage 0,
  applied in `reset()` under `set_curriculum`. Evaluation pins stages with
  `set_stage()`, so the benchmark's stage 0 stays calm.
- **`ppo.gae_lambda` for run 1: 0.995,** from horizon test 1.
- **`apply_overrides` now parses true, false and null.**

## V24: training base reverted to V22, one change per run

### Bundling prevented attribution
- V23 runs 1 and 2 changed four things at once:
  - observation v3;
  - air-frame path reward;
  - `gae_lambda` 0.995;
  - light wind at stage 0.
- Neither run beat the baseline at any stage. Because the changes went in together, the failure can't be pinned on any one of them.
- From run 3 on, each run is one named configuration that changes one key.
- Runs 1 and 2 are reported as negative results:
  - tag `v23-runs-1-2-negative`;
  - `analysis/negative_results/`;
  - all their code still runs behind its config keys.

### Glide-cone shaping was not the cause
- Run 2 was run 1 without the shaping.
- Its best checkpoint scored 27 / 8 / 5 / 2% against run 1's 35 / 6 / 3 / 0%. Both are worse than the baseline at stages 1–3.

### Training base reverted; infrastructure kept
- **The restored config:**
  - `training/configs/v22_restored.yaml` is a verbatim copy of `base.yaml` at `ed65989`, the commit V22 trained at.
  - V22 archived no resolved config. Its metadata holds the commit hash, and the file's sha256 matches the hash V22's benchmark recorded.
  - The keys added after V22 are pinned in it to V22's behaviour: `observation.obs_version: 1`, `reward.path_reward_frame: ground`, `reward.cone_shaping: false` and `curriculum.stage0_wind_ms: 0.0`.
- **Checks** (`tests/test_v22_restore.py`):
  - The config matches V22's key by key; only the output paths may differ.
  - The PPO settings stored inside V22's model file agree with it.
  - Open-loop trajectories replayed in the `ed65989` tree and in this one give identical per-step rewards and observations (1e-12).
  - V22's best checkpoint reproduces its committed stage-0 benchmark on all 100 seeds: same outcome, steps, return and miss distance.
- **Resolved-config archive:** `training/train.py` now writes `resolved_config_<UTC>.yaml` into the checkpoint folder at every launch. No earlier run did, so Phase 0 had to rebuild those configs from git.

### Run 3, written before the run
- **Configuration:** `v22_restored.yaml` plus `ppo.gae_lambda=0.995`, nothing else.
- **Prediction:**
  - If the horizon hypothesis is right, run 3 matches V22 in calm air and improves at stage 1.
  - If λ 0.995 is what damaged runs 1 and 2, calm-air success falls toward their 27–35%.

### Calibration of the early check, and what it says about "V22's 59%"
V22's own stage-0 success on the 100 benchmark seeds, checkpoint by checkpoint (`results/v22_stage0_curve.json` and the committed benchmarks):

| Step | 3.0M | 3.7M (graduation) | 4.5M | 5.5M | 7.0M | 10M | 14M | 15M (best) | 20M |
|---|---|---|---|---|---|---|---|---|---|
| Stage 0 success | 2% | 24% | 13% | 2% | 2% | 26% | 57% | 59% | 0% |

- **V22 would fail the spec's own early check.** Its graduation checkpoint scored 24%, below the 45% bar, so the rule would have stopped V22 itself.
  - `analysis/v24_early_check.py` implements the rule as written; `--run checkpoints_v22` returns STOP.
- **One checkpoint's stage-0 score is too unstable to stop a run on.**
- **The "calm-air drop" from 59% to 35% and 27% compares a selected best checkpoint with others.**
  - Each run's best checkpoint was picked from about 80 noisy 10-episode evaluations.
  - Runs 1 and 2 sit inside the range V22's own checkpoints span.
  - Read that way, the evidence that λ 0.995 hurt calm-air flight is weaker than the V24 spec states.
- **Researcher's decision:** whether to stop on graduation alone, or compare run 3 against V22 at matched checkpoints.

### Wind-gated controller: on-board inputs, development-seed tuning
- **On-board inputs only.**
  - The decision lives in `baseline/wind_gate.py`. It reads the observation-v1 sensor vector (GPS ground speed and course, IMU heading) and the airspeed command it last flew.
  - It imports no flight model, wind model or environment.
  - Tests:
    - an AST check plus a run-time `sys.modules` check;
    - the same sensor stream gives identical decisions while a different true wind is live in the simulator.
  - The V23 actor never read the true wind either, but it imported `env.glider_env`.
- **Development seeds.**
  - `analysis/seeds/dev_seeds.json` holds `episode_seeds(200, 7)`, hash-checked.
  - It shares no seed with the benchmark seeds (`episode_seeds(100, 0)`) or the in-training selection seeds (`episode_seeds(10, 1000)`).
  - The V23 gate (1.0 m/s at 15 s) had already been tuned on the first 100 of these seeds, so its benchmark figures were not tuned on the benchmark seeds.
  - The repo has no held-out airframe seed set to check against.
- **Earlier-handover variant (`gate.window`).**
  - It hands over the first time the estimate exceeds the threshold within a time window.
  - Grid: thresholds 0.75–2.5 m/s, windows starting at 3–10 s and ending at 10–20 s, on 200 development seeds (`results/gate_tune_v24_dev.json`).
  - The selection rule was fixed in the script beforehand. It chose 1.25 m/s within 10–20 s: mean success 0.284 on the development seeds, against 0.259 for the single check and 0.175 for the baseline.
  - On the 200 development seeds, the best single-check setting (0.75 m/s at 10 s, 0.264) is within noise of V23's (0.259), so V23's setting was kept.
- **One evaluation on the benchmark seeds** (`results/gated_v24_benchmark.json`; success with 95% Wilson interval; McNemar p against the baseline; handover share of episodes and of flight time):

  | Stage | Baseline | V22 alone | Single (1.0 m/s, 15 s) | Window (1.25 m/s, 10–20 s) |
  |---|---|---|---|---|
  | 0 | 27% | 59% | 56% [46, 65], p < 0.001; 5% / 3% | 58% [48, 67], p < 0.001; 1% / 1% |
  | 1 | 25% | 16% | 33% [25, 43], p 0.28; 60% / 30% | 27% [19, 36], p 0.87; 66% / 40% |
  | 2 | 11% | 2% | 11% [6, 19], p 1.0; 59% / 30% | 13% [8, 21], p 0.83; 80% / 47% |
  | 3 | 8% | 0% | **1%** [0, 5], p 0.016 (worse); 52% / 27% | 6% [3, 12], p 0.77; 74% / 46% |

- **What the evaluation shows.**
  - The single check reproduces the committed V23 figures exactly, which confirms the refactor.
  - The window variant removes the significant stage 3 deficit. It is significantly better than the baseline in calm air and not significantly different at stages 1–3.
  - In wind, the baseline flies 40–47% of the flight time, so the stage 1–3 results are largely the baseline's.

### Run 3 result: λ 0.995 alone fails (negative result)
- **Configuration:** `v22_restored.yaml` plus `ppo.gae_lambda=0.995`, seed 0.
  - Started 2026-11-13 18:21 UTC; stopped at 8.6M steps under the early-stop rule.
  - Resolved config: `checkpoints_v24_run3/resolved_config_20261113T182113Z.yaml`.
- **Graduation:** stage 0 never graduated, against a limit of 5.44M steps; V22 graduated at 3.7M.
  - Training success in calm air rose to 11% by 1.6M steps, then fell back to 2–12% and stayed there.
- **Benchmark** (100 paired seeds; success, with crash rate in brackets; every stage significantly worse than the baseline's 27 / 25 / 11 / 8%, McNemar p ≤ 0.008):

  | Checkpoint | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
  |---|---|---|---|---|
  | Best by in-training checks (1.75M) | 7% (0%) | 0% (6%) | 1% (19%) | 0% (48%) |
  | Last (8.6M) | 9% (0%) | 3% (6%) | 0% (24%) | 0% (52%) |

- **Against the prediction recorded before the run:** calm-air success fell, below even runs 1 and 2. The prediction's second branch held: λ 0.995 is damaging in this setup.
- **Mechanism (probable):** the value network could not fit the near-Monte-Carlo targets that λ 0.995 produces.
  - Explained variance was 0.07–0.59 (0.24 at 8.6M), against V22's steady 0.84–0.93 over the same steps.
  - Run 2 (λ 0.995, no shaping) shows the same pattern: 0.37–0.45, then 0.18.
  - Run 1 (λ 0.995 with cone shaping, a smoother return) stayed at 0.74–0.85 until about 6M.
  - The update KL, clipping and entropy matched V22's, so the policy optimiser was not the problem.
  - Caveat: explained variance under different λ measures the fit to different targets, so this is supporting evidence, not proof.
- **Where the horizon test went wrong:**
  - Test 1 computed λ 0.995 advantages after the fact, with V22's trained critic. It showed the landing result reaching early decisions.
  - It did not test whether a critic can learn those targets during training. The horizon limit is real, but raising λ swaps it for advantage variance this setup cannot absorb.
- **Consequence:** run 4 is V22 unchanged on a second seed (`--seed 1`), testing whether V22's 59% is reproducible.

### Run 4 result: V22 reproduced on a second seed
- **Configuration:** `v22_restored.yaml`, `--seed 1`. Started 2026-11-13 20:57 UTC.
  - At 4.5M steps the run was killed by an out-of-memory error: a 10-worker early check ran beside training. It was resumed from the 4.5M checkpoint with curriculum state restored (`checkpoints_v24_run4/resolved_config_*.yaml`, two files).
  - The resume counted 20M further steps, so the run ended at 24.5M. The 20M checkpoint is the like-for-like comparison with V22's final.
  - Early checks now default to 3 workers when training is running.
- **Curriculum:** stage 1 at 1.44M steps (V22 3.6M), stage 2 at 15.1M (V22 13.2M), stage 3 never (as V22). Explained variance 0.75–0.93 throughout.
  - At the stage 0 graduation checkpoint (1.5M) calm-air success was 34% on the benchmark seeds (V22 24%).
- **Benchmark** (100 paired seeds; success, crash rate in brackets; McNemar p against the baseline):

  | Checkpoint | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
  |---|---|---|---|---|
  | Baseline | 27% (0%) | 25% (0%) | 11% (6%) | 8% (39%) |
  | V22 best (15M) | 59% (0%) | 16% (2%) | 2% (10%) | 0% (41%) |
  | Run 4 best (15.5M) | **63%** [53, 72] (0%), p < 0.001 | **25%** (6%), p 1.0 | 8% (27%), p 0.63 | 1% (54%), p 0.016 |
  | Run 4 at 20M | 17% (0%) | 11% (5%) | 7% (17%) | 0% (48%) |
  | Run 4 final (24.5M) | 1% (0%) | 10% (2%) | 6% (8%) | 2% (43%) |

- **What it shows:**
  - V22's calm-air result is reproducible: 63% against 59%, intervals overlapping almost entirely. Stage 1 matches the baseline.
  - In wind the policy crashes more than the baseline (stage 2: 27% against 6%; stage 3: 54% against 39%).
  - Calm-air skill collapses after the policy reaches stage 2, on both seeds (V22 59% to 0%; run 4 63% to 1%). The 25% replay share does not prevent it.
  - Extra steps under this curriculum did not help: training success at stage 2 was 2–7% for 9.4M steps, against the 15% bar.

### Feasibility of the benchmark seeds
- `analysis/feasibility_v24.py`, `results/feasibility_v24.json`. For every benchmark seed and stage: the episode's launch state, mean wind and randomised airframe, read from the environment; that airframe's trimmed glide polar from JSBSim's aerodynamics (airspeed within the commandable 9.5–16 m/s); best speed-to-fly with a crab into the crosswind; a 30° bank turn onto the course home; the kinetic-energy credit of the launch speed.
- The model ignores gusts and the flare, so it is optimistic: "unreachable" is firm, "reachable" is only a necessary condition. No seed it calls unreachable was ever landed by any of the 22 benchmarked controllers.

  | Stage | Reachable | Energy surplus, median (worst 10%) | Landed by at least one controller |
  |---|---|---|---|
  | 0 | 100 | 8.6× (6.4×) | 100 |
  | 1 | 100 | 9.0× (4.9×) | 80 |
  | 2 | 100 | 8.7× (4.2×) | 49 |
  | 3 | 97 | 7.6× (2.9×) | 18 |

- Energy is almost never the limit; the three unreachable cases are stage 3 headwinds of 5.8–6.7 m/s. The task is dissipating a large surplus precisely with no thrust and no drag devices.
- Gust strength separates landed from failed stage 3 seeds: 11 of 21 landed below σ 0.5 m/s, 2 of 52 at σ 1.0 m/s or above.

### Run 5, written before the run: run 4 best on a fixed stage mix
- **Problem:** no run has trained at stage 3. The promotion bars (30% to leave stage 1, 15% to leave stage 2) sit at the best any controller has achieved, so in practice they block, and training at a harder stage erodes calm-air skill.
- **Configuration:** `training/configs/v24_run5_mix.yaml` = `v22_restored.yaml` plus
  - `curriculum.stage_weights: [0.2, 0.2, 0.3, 0.3]` (new key, null by default: every episode's stage drawn from the mix; promotion and `replay_frac` play no part),
  - `ppo.total_timesteps: 10_000_000`, and its own logging directories.
  - Launched with `--init-from checkpoints_v24_run4/best_model` (new option: weights and VecNormalize statistics; hyperparameters from the config; step counter 0; no curriculum state).
  - This is two changes against run 4 (continued training and the mix), reported as one named configuration, not a single-variable test.
- **Prediction:** stage 3 crash rate falls from 54% toward the baseline's 39%; stage 3 success 3–8%; calm-air success stays above 45%. Ceiling: 18 of 100 stage 3 seeds have ever been landed by any controller.
- **Early stop** (`analysis/v24_run5_check.py`, development seeds only): at 2M and 5M steps, stop if calm-air success is below 40% or the stage 3 crash rate is not below run 4 best's on the same seeds.
- **Tests:** `tests/test_stage_mix.py`.

### Run 5 result: safer at stage 3, at the cost of every stage's success (mixed result)
- **Configuration** as recorded above; started 2026-11-14 13:54 UTC, finished at 10M steps. The 2M check was not run; the 5M check passed on the development seeds (calm-air success 44%; stage 3 crashes 33% against run 4 best's 45.5%).
- **Benchmark** (100 paired seeds; success, crash rate in brackets; McNemar p of success against the baseline):

  | Checkpoint | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
  |---|---|---|---|---|
  | Baseline | 27% (0%) | 25% (0%) | 11% (6%) | 8% (39%) |
  | Run 4 best (start point) | 63% (0%) | 25% (6%) | 8% (27%) | 1% (54%) |
  | Run 5 best by in-training checks (1.75M) | 63% (0%) | 20% (10%) | 8% (19%) | 0% (52%) |
  | Run 5 at 5M | 43% (0%), p 0.033 | 17% (1%) | 6% (15%) | 0% (46%) |
  | Run 5 final (10M) | 15% (0%), p 0.058 | 11% (0%), p 0.013 | 2% (11%), p 0.012 | 1% (37%) |

- **Against the prediction:**
  - Stage 3 crashes fell toward the baseline: 54% to 37% at 10M, significantly below run 4 best (paired McNemar on crashes, p 0.006) and level with the baseline (p 0.87). Held.
  - Stage 3 success did not rise (0–1%, predicted 3–8%). Failed.
  - Calm-air success did not hold (63% to 43% to 15%, predicted above 45%). Failed.
- **What it shows:**
  - Training on stage 3 taught the policy to avoid crashing, not to get home: crashes became soft landings far from home (stage 3 median miss 90–105 m).
  - Calm-air forgetting continued with stage 0 at 20% of episodes. Across V22, run 4 and run 5, harder-condition training erodes calm-air skill under every curriculum tried.
  - The run 5 final is the safest policy in wind but significantly worse than the baseline at stages 1 and 2. No run 5 checkpoint improves on run 4 best overall; run 4 best stays the reported policy.
