# RL-Guided Return-to-Launch for a Hand-Launched Fixed-Wing Glider

> **COS 731/732 Honours Project — University of the Western Cape**
> Mohamed Yusuf Kathree · 4253340 · [Project Website](https://sites.google.com/myuwc.ac.za/rlglide/home)

A hand-launched glider with no motor has to find its own way back to the launch point and land
there, using only on-board sensors. This project trains a PPO reinforcement-learning policy in a
physics-validated JSBSim simulation and combines it with a deterministic return-to-launch
controller through an on-board wind gate.

**Selected controller: Run 4 – Gated.** Run 4's trained policy flies from launch. Between 8 and
15 s after launch, the first time the on-board wind estimate exceeds 1.5 m/s, control passes to
the deterministic baseline for the rest of the flight.

---

## Contents

- [Results](#results)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Dashboard](#dashboard)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reproducing the headline result](#reproducing-the-headline-result)
- [Curriculum stages](#curriculum-stages)
- [Reward](#reward)
- [Testing](#testing)
- [Documentation](#documentation)
- [Status and limitations](#status-and-limitations)

---

## Results

One evaluation on 100 paired simulated launches per wind stage (`episode_seeds(100, 0)`). Gate
settings were tuned on 200 separate development seeds and frozen before this evaluation.
Success = touchdown inside the home radius with landing quality above 0.5. p-values are exact
McNemar tests of success against the baseline on the same launches.

| Stage (max wind) | Baseline | Run 4 alone | **Run 4 – Gated** | p (gated vs baseline) |
|---|---|---|---|---|
| 0 (calm) | 27% (0% crash) | 63% (0%) | **58%** (0%) | < 0.001 |
| 1 (3 m/s) | 25% (0%) | 25% (6%) | **35%** (3%) | 0.17 |
| 2 (6 m/s) | 11% (6%) | 8% (27%) | **19%** (13%) | 0.15 |
| 3 (9 m/s) | 8% (39%) | 1% (54%) | **7%** (47%) | 1.00 |

- **Significant only in calm air**, where success more than doubles.
- **Stages 1–2:** the highest success of any controller tested, but not significant on 100 launches.
- **Stage 3:** no success gain, and a higher observed crash rate than the baseline (not
  significant, p 0.20). Strong-wind safety is **not** established.
- **Simulation only.** No flight tests yet. The simulated speed hold uses true airspeed; the
  glider has no pitot.

Source: `results/gated_v24_run4_benchmark.json`. The full record, including negative results for
runs 1–3 and 5, is in [`docs/SPEC_V20_decisions.md`](docs/SPEC_V20_decisions.md).

**Feasibility.** A per-launch energy analysis (`analysis/feasibility_v24.py`) shows home is
physically reachable on 397 of the 400 benchmark launches, usually with 7–9× the energy needed.
The task is limited by shedding that surplus precisely without thrust or airbrakes, and by gusts,
not by energy.

### Training runs (V22 configuration onwards)

| Run | Change | Best model, success by stage 0 / 1 / 2 / 3 | Outcome |
|---|---|---|---|
| V22 | reference | 59 / 16 / 2 / 0% | reference for the runs below |
| V23 run 1 | λ 0.995, cone shaping and other changes bundled | 35 / 6 / 3 / 0% | negative |
| V23 run 2 | as run 1 without cone shaping | 27 / 8 / 5 / 2% | negative |
| V24 run 3 | λ 0.995 alone | 7 / 0 / 1 / 0% | negative: the critic could not fit λ 0.995 targets |
| **V24 run 4** | V22 config, seed 1 | **63 / 25 / 8 / 1%** | V22 reproduced; used in the selected controller |
| V24 run 5 | run 4 best on a fixed stage mix | 63 / 20 / 8 / 0% (final: 15 / 11 / 2 / 1%) | fewer stage-3 crashes, lost calm-air skill |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Wind gate                                  20 Hz, on board   │
│  Wind estimate from GPS ground speed, course and heading.     │
│  8–15 s after launch: first estimate > 1.5 m/s → baseline.    │
│  One-way. baseline/wind_gate.py, baseline/wind_gated.py       │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│  Guidance: Run 4 policy  or  deterministic baseline   20 Hz   │
│  Policy: PPO, MLP 128×128, 8 stacked frames of 12 values      │
│  Output: [bank ±45°, airspeed 9.5–16 m/s]                     │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│  Attitude controller                    200 Hz, ESP32 (sim)   │
│  Roll PD + speed-scheduled pitch + coordinated rudder         │
│  Safety shield: bank ≤ 45°, pitch ≤ 20°                       │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│  JSBSim physics                                      200 Hz   │
│  6-DOF, validated aero, actuator lag and rate limits          │
│  Sensor models (GPS 5 Hz, baro 25 Hz, ultrasonic 20 Hz, IMU)  │
│  Wind: mean NED + Ornstein–Uhlenbeck gusts                    │
│  Airframe randomised per episode (mass, CG, inertia, aero)    │
└──────────────────────────────────────────────────────────────┘
```

**Airframe (as built):** flying mass ≈ 1.64 kg (estimated 1.50–1.80), span 2.0 m, wing area
0.38 m², glide ratio ≈ 14 at trim. Mass properties: `aircraft/mass_properties.yaml`.

---

## Repository layout

```
aircraft/            JSBSim aircraft (rlglider.xml), FDM generator, mass properties
sim/                 JSBSim wrapper, sensor models, wind, mass solver
env/                 GliderEnv, reward, curriculum, observation features, wind estimator
baseline/            Deterministic RTL controller; wind gate and gated controller
training/            train.py, benchmark.py (the single evaluator), configs/
  configs/base.yaml          current defaults (every change behind a key; old behaviour default)
  configs/v22_restored.yaml  V22 configuration, used by runs 3 and 4
  configs/v24_run5_mix.yaml  run 5 (fixed stage mix)
analysis/            Diagnostics, gate tuning, feasibility, dashboard data export
  seeds/dev_seeds.json       200 development seeds (tuning only; never the benchmark seeds)
  negative_results/          archived configs and hashes for the V23 negative runs
results/             Benchmark and analysis outputs (JSON)
backend/             FastAPI server for the dashboard (records flights, runs checks, streams training)
frontend/            React + Vite dashboard
flighttest/          Flight-test ground code: packet schema and codegen, safety reference, calibration, ground station (see flighttest/README.md)
firmware/            ESP32 code (see firmware/README.md):
  flight/                    flight firmware: 228 Hz attitude + inner loop, MCPWM servos, Pi link, UBX GPS, flight phases
  ground_station/            ESP-NOW <-> USB bridge for the dashboard
  bench/                     bench test menu (hardware regression suite)
  protocol/                  shared packet header, CRC and safety state machine (generated / host-tested)
  tools/                     corrected attitude visualiser and comparison tools
onboard/             Raspberry Pi flight program: numpy policy, wind gate, 20 Hz runner, flight logs (see onboard/README.md)
tests/               Pytest suite (about 450 tests)
docs/                Decisions record and diagnostic summaries
deployment/          Archived V14 deployment model (legacy)
checkpoints_*/       Model checkpoints per run (not all tracked)
validate_glide.py    Five physics validation gates
```

---

## Installation

**Requirements:** Python 3.10+ (developed on 3.14), JSBSim 1.3.1+, Node.js 20+ for the dashboard.

```bash
git clone https://github.com/Mohamed-Kathree/Reinforcement-Learning-Guided-Fixed-Wing-Glider.git
cd Reinforcement-Learning-Guided-Fixed-Wing-Glider

python -m venv venv
venv\Scripts\activate            # Windows; source venv/bin/activate elsewhere
pip install -r requirements.txt
pip install -r backend/requirements.txt   # dashboard backend only

cd frontend && npm install
```

### On a new machine: what a clone does and does not contain

A clone has all the code, the results (`results/*.json`), the dashboard data, the selected
policy (`checkpoints_v24_run4/best_model.zip` + `vecnormalize_best.pkl`), the Pi's numpy model
(`onboard/model/run4.npz`) and the archived V14 model (`deployment/`). Everything the dashboard,
the tests and the hardware need is there.

Not in git (too large), and only needed if you want them:

| Folder | Size | What you lose without it |
|---|---|---|
| `data/episodes/` | ~0.5 GB | previously recorded flights in Flight Replay (record new ones with **Record flight**) |
| `data/runs/` | ~0.2 GB | archived training streams on the Training page |
| other `checkpoints_*/`, `runs_*/` | ~180 MB per run | intermediate checkpoints of every run (re-benchmarking old runs) |

Copy those folders across by hand (USB drive, or the cloud folder they live in) into the same
places in the new clone.

- **Do not reuse a `venv/` or `node_modules/` copied from another machine:** both contain
  machine-specific paths. Create them fresh, as above.
- **If the repo sits inside a synced cloud folder** (OneDrive), clone a separate copy on the
  second machine rather than opening the synced one, so the two machines' `venv` folders do not
  overwrite each other.

**Hardware tools (only to build or flash firmware, or to update the Pi):**
- **Arduino IDE 2:** includes `arduino-cli`. Add the ESP32 board package 3.3.x and the
  libraries Adafruit BMP280 and Adafruit BusIO (and ESP32Servo for the bench sketch).
- **gcc** (MinGW on Windows), for the C host tests. Without it they skip.
- **`pip install paramiko`**, then `python onboard/deploy/push_to_pi.py` to copy the Pi program
  and restart it.

---

## Quick start

```bash
python validate_glide.py                 # 1. physics gates (must all pass)
pytest tests -q                          # 2. test suite
python -m training.benchmark --policy baseline --episodes 100   # 3. baseline benchmark
```

---

## Dashboard

```bash
python -m uvicorn backend.main:app --port 8000     # backend, from the repo root
cd frontend && npm run dev                         # frontend, http://localhost:5173
```

| Page | Purpose |
|---|---|
| Overview | What the system does, headline results and limitations, architecture and specs |
| Validation | Run 4 – Gated benchmark: charts, paired significance, per-stage readings, every benchmark launch as a clickable tile map, gate behaviour, conclusions; live physics and unit checks |
| Research | Every training run since V22 with benchmarks (negative results included), cross-run findings, calm-air forgetting, feasibility and gust analysis |
| Flight Replay | Record a flight with the gated controller (default), the baseline or the legacy V14 policy, then replay it frame by frame. Gated flights show the wind-gate trace. Benchmark launches open side by side (`#/replay/<a>/<b>`) and are checked against the stored result. **Play on aircraft**: the real servos follow the replayed surface movements in real time (REPLAY mode, firmware's calibration mapping) |
| Baseline | Fresh baseline benchmark on random launches |
| Training | Live and archived training runs |
| Analysis | Touchdown scatter and failure gallery across recorded flights |
| Readiness | Go/no-go wind limits from the results, pre-flight checklist, sim-to-real gaps, avionics as wired and open hardware items |
| Ground Station | Aircraft checklist (ground station, flight computer, Pi, calibration, GPS, barometer, ready to arm), one-click connect, flight computer panel (phase, loop rate, Pi link, console, Arm FLIGHT), guided calibration wizard (centre, direction, travel, Send to aircraft), live telemetry and the control-surface tester. Runs against a stub glider when no hardware is connected |
| Flight Logs | Replay full-rate Pi logs in the shared 3D view, optionally driving the real servos (REPLAY mode, 20 Hz cap, hold or centre on pause) |

Light appearance by default; dark via the header toggle. The backend accepts requests from any
local port; point the dashboard at another backend with `VITE_API_BASE` (e.g. in
`frontend/.env.local`). Validation figures come from `frontend/src/data/validation.json`, generated by
`python analysis/export_dashboard_data.py` from the result files;
`tests/test_dashboard_data.py` fails if it is stale.

---

## Training

```bash
# The configuration behind the selected controller (V22 settings, second seed)
python -m training.train --config training/configs/v22_restored.yaml --seed 1 \
    --set logging.checkpoint_dir=checkpoints_v24_run4/ logging.tensorboard_log=runs_v24_run4/

# Resume (restores VecNormalize stats and curriculum state)
python -m training.train --config ... --resume checkpoints_v24_run4/ppo_glider_4500000_steps

# Start a NEW run from a checkpoint's weights (config hyperparameters, step counter 0)
python -m training.train --config training/configs/v24_run5_mix.yaml \
    --init-from checkpoints_v24_run4/best_model --seed 2

tensorboard --logdir runs_v24_run4/
```

Every launch writes `resolved_config_<UTC>.yaml` (config, argv, git SHA) into its checkpoint
directory. `--resume` continues the step count, so resuming at 4.5M with a 20M budget trains to
24.5M.

| PPO parameter | Value |
|---|---|
| Environments × rollout | 8 × 2048 |
| Batch / epochs | 256 / 10 |
| Learning rate | 3 × 10⁻⁴ (constant) |
| γ / GAE λ | 0.999 / 0.95 |
| Clip / target KL | 0.2 / 0.03 |
| Entropy coefficient / min action std | 0.005 / 0.2 |
| Network | MLP 128 × 128, 8 stacked frames |
| Checkpoints | every 100k steps |

Model selection: `best_model.zip` is the checkpoint with the highest mean deterministic success
over all four stages on 10 selection seeds per stage (disjoint from the benchmark seeds).

---

## Evaluation

```bash
# Policy vs baseline, paired on identical seeds, stages 0-3, 100 episodes each
python -m training.benchmark --policy trained \
    --model checkpoints_v24_run4/best_model.zip \
    --vecnorm checkpoints_v24_run4/vecnormalize_best.pkl \
    --config training/configs/v22_restored.yaml \
    --out results/benchmark_v24run4_best_15.5M.json

python -m training.benchmark --policy random      # floor controls
python -m training.benchmark --policy zero
```

Reported per stage: four outcomes (success / soft landing / crash / timeout) with Wilson 95%
intervals, landing quality, miss distance, and for paired runs the contingency table and exact
McNemar p. Every episode's airframe, outcome and launch state are saved with a provenance block
(model, VecNormalize and config hashes, git SHA).

---

## Reproducing the headline result

```bash
# 1. Tune the gate on the 200 development seeds (writes results/gate_tune_v24_dev_run4.json)
GATE_POLICY=run4 python analysis/gate_tune_v24.py tune

# 2. Values are frozen in training/configs/base.yaml under gate.run4

# 3. One evaluation on the benchmark seeds (writes results/gated_v24_run4_benchmark.json)
GATE_POLICY=run4 python analysis/gate_tune_v24.py evaluate

# 4. Regenerate the dashboard data
python analysis/export_dashboard_data.py
```

On Windows PowerShell, set the variable with `$env:GATE_POLICY = "run4"` first. Without
`GATE_POLICY` the script runs the original V22 gate.

---

## Curriculum stages

| Stage | Max wind | Home radius | Launch height | Airframe variation |
|---|---|---|---|---|
| 0 Calm | 0 m/s | 25 m | 25 m | 25% of full range |
| 1 Light wind | 3 m/s | 22 m | 23 m | 50% |
| 2 Moderate wind | 6 m/s | 20 m | 21 m | 75% |
| 3 Strong wind | 9 m/s | 20 m | 20 m | 100% |

Wind is drawn uniformly from 0 to the maximum in a random direction, with gusts, sensor noise and
sensor dropout that grow by stage (`env/curriculum.py`). Training advances at 35% / 30% / 15%
rolling success (100-episode window), with 25% of episodes replayed at earlier stages. An
optional `curriculum.stage_weights` key replaces promotion with a fixed stage mix (run 5).

---

## Reward

```
quality     = q_sink · q_roll · q_speed · q_alpha            (each in [0,1], graded at touchdown)

r_step      = w_path · v_ground · dt
            − w_unreach · max(0, d_home/h − L/D_usable)      (h ≥ h_barrier only, capped)
            − w_stall · max(0, α − α_safe)
            − w_bank  · max(0, |φ| − φ_soft)
            − w_smooth · ‖a − a_prev‖²

r_touchdown = quality · ( w_land · σ/(σ + d_home) + w_bullseye · exp(−(d_home/2)²) )
            − (1 − quality) · w_crash
```

Weights: `training/configs/base.yaml` (`reward:`), implementation: `env/reward.py`.

---

## Testing

```bash
pytest tests -q          # about 450 tests, about 7 minutes
```

Includes physics and sensor tests, V22 restore and reward-reference tests, gate tests (the gate
reads only on-board inputs; development and benchmark seeds are disjoint), stage-mix and
`--init-from` tests, and dashboard data and gated-recording tests. The flight computer's pure-C
code (attitude filter, inner loop, Pi link, UBX parser, flight phases, safety) is compiled with
the host gcc and checked against the Python simulator; the Pi program is checked against the
benchmark controller through the real byte protocol.

---

## Documentation

- [`docs/SPEC_V20_decisions.md`](docs/SPEC_V20_decisions.md): every decision, run, prediction
  and result from V20 to V24, including negative results.
- `docs/V23_horizon_tests_summary.md`, `docs/V23_gate_D_summary.md`: diagnostic summaries.
- `docs/context_wind_problem.md`, `docs/context_v23_revision.md`: background on the wind problem.

---

## Status and limitations

- Simulation results only. No real flights yet. The flight computer (ESP32 + Raspberry Pi) runs
  the full chain on the bench: arm, launch, apex hand-off, the Pi's controller moving the servos
  (see firmware/flight/README.md for the measured results and open hardware items).
- The real aircraft has no airspeed sensor, so it flies the simulator's pitch trim schedule
  without the true-airspeed correction. Measured in simulation on 200 development seeds, that
  costs a lot (Run 4 gated success 111 -> 72, 58 -> 32, 34 -> 14, 17 -> 6 at stages 0-3);
  accepted for the prototype demonstration (results/onboard_airspeed_term_dev.json).
- Open hardware items: the board does not start by itself at battery-on (press EN); USB
  back-feeds the servo rail (servo leads off whenever USB is plugged in); surfaces not yet
  fitted, so the stored servo calibration is a trial run; GPS fix and launch/apex thresholds
  untested outdoors.
- One evaluation of 100 launches per stage: large effects only. Stage 1–2 gains and Stage 3
  crash differences need more seeds to confirm or rule out.
- Mass properties beyond the fuselage and tail are estimates; weigh the aircraft before flying.
- Calm-air skill degrades once training moves to harder stages, under every curriculum tried.
