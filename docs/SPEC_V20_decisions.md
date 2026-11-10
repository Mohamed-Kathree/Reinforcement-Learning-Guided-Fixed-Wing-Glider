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
`analysis/baseline_benchmark_v20.json`.

