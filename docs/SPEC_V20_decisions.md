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

## ixz: sign disputed, range not pinned

The CAD rebuild gives +0.0036 kg·m² and V19 gives −0.008 kg·m², which
disagree in sign. The YAML now carries
`ixz: {value: 0.0, range: [-0.010, 0.010], source: estimated}`. The value is
the range midpoint, not a preference for either estimate. The previous file
pinned +0.01.

## Glide-ratio gate: 5 s window is start-speed sensitive

`validate_glide.py` Gate 2 takes the median glide ratio over the first 5 s.
On the same airframe it read 12.87 when started 1.8% below trim speed and
21.6 when started 5% above. The steady value (30–60 s window) is 14.1
regardless of start speed. The V19 figure of 12.87 was a transient artifact
of the fixed 10 m/s start. The start speed has been derived from mass since
Phase 1; Phase 5 adds a settling period before the measurement.
