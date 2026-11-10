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

## clmax-scale: peak lift at constant slope

SPEC V20 lists `aero/clmax-scale` as a multiplier on the CLalpha function.
Applied literally, it would multiply the whole CL table, which is exactly
what `aero/cl-mult` already does. The two domain-randomisation parameters
would then be indistinguishable, and the Phase 7 regression could not
separate them.

It is implemented as a rescaling of the table about the origin,
CL(α) = s · table(α / s). This keeps the pre-stall lift slope and scales
both CLmax and the stall angle by s, which is what the `clmax` range
(V19 stall 8.8 m/s against the table's 7.75 m/s) describes. At s = 1 both
operations are exact in floating point. In the pre-stall region lift
changes by at most 3e-5 relative (the table's segments differ slightly in
slope), against 15% for a plain multiplier at s = 1.15.

## Control effectiveness: per surface

The retired single control multiplier scaled seven terms. Each now belongs
to the surface that produces it:

| Multiplier | Surface | Terms |
|---|---|---|
| `clda-mult` | aileron | Clda, Cnda (adverse yaw) |
| `cmde-mult` | elevator | Cmde, CLde, and the elevator term in CLtotal |
| `cndr-mult` | rudder | Cndr, Cldr |

With all three set to one value the physics is identical to the old
multiplier, which `tests/test_dr_hooks.py` checks to 1e-9 over 6 s of
doublets. Until Phase 4 the environment draws one value and applies it to
all three, which keeps the random stream and the physics unchanged.

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
