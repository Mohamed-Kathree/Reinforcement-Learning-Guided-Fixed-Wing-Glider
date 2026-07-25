# RL-Guided Return-to-Launch for a Hand-Launched Fixed-Wing Glider

> **COS 731/732 Honours Project — University of the Western Cape**
> Mohamed Yusuf Kathree · 4253340 · [Project Website](https://sites.google.com/myuwc.ac.za/rlglide/home)

A simulation-first reinforcement learning system that autonomously guides a 1.1 kg hand-launched fixed-wing glider back to its launch point using only onboard sensors. The RL policy is trained in a physics-validated JSBSim environment and deployed on a Raspberry Pi 5, with an ESP32 handling real-time stabilisation and safety enforcement.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Physics Validation](#physics-validation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualisation](#visualisation)
- [Testing](#testing)
- [Key Design Decisions](#key-design-decisions)
- [Interface Contracts](#interface-contracts)
- [Curriculum Stages](#curriculum-stages)
- [Reward Function](#reward-function)
- [Diagnostics](#diagnostics)
- [Project Status](#project-status)
- [References](#references)

---

## Overview

Hand-launched gliders face highly variable launch conditions and outdoor disturbances that make traditional rule-based guidance difficult to generalise. This project trains a PPO reinforcement learning policy in a simulation environment that mirrors the onboard hardware stack, enabling direct sim-to-real transfer without retuning.

The task is fundamentally one of energy management: the glider can only trade altitude for airspeed while continuously losing energy to drag. The policy must learn to optimise the return-to-launch (RTL) objective while respecting hard safety constraints on stall margin, bank angle, and minimum altitude above ground level.

**Term 2 status:** Physics simulation validated, Gymnasium environment implemented, deterministic baseline benchmarked at ~20% success across all 4 stages (see [Curriculum Stages](#curriculum-stages) for the full table and correction history). Training infrastructure ready.

**Term 3 goal:** Train PPO policy to exceed 60% success at Stage 3 (0–9 m/s wind, 20% sensor noise, 15 m home radius).

---

## Architecture

The system uses a three-layer hierarchy separated by timing rate and responsibility:

```
┌──────────────────────────────────────────────────────────┐
│  Layer 1 — RL Policy                    20 Hz, Pi 5      │
│  Input:  12-element noisy observation vector             │
│  Output: [bank_cmd, speed_cmd] ∈ [-1, 1]²               │
│  Algorithm: PPO (Stable-Baselines3)                      │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│  Layer 2 — Attitude Controller         200 Hz, ESP32     │
│  Roll PD + Pitch 1/V² + Coordinated Rudder               │
│  Output: [aileron, elevator, rudder] (rad)               │
│  Safety shield: bank > 50°, AGL < 8 m                   │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│  Layer 3 — JSBSim Physics              200 Hz            │
│  6-DOF dynamics + validated aero tables                  │
│  Actuator lag/rate limits · Sensor noise models          │
│  Wind: mean NED + Ornstein–Uhlenbeck gusts               │
└──────────────────────────────────────────────────────────┘
```

This separation keeps safety deterministic and independent of the learning component. The Python simulation mirrors the ESP32 firmware exactly, so policies trained in simulation run without modification on hardware.

---

## Directory Structure

```
rl-glider/
│
├── aircraft/                       # JSBSim aircraft definitions
│   └── rlglider/
│       └── rlglider.xml           # FDM: mass, aero tables, actuator dynamics
│
├── analysis/                       # Visualisation and plotting
│   └── plot_trajectories.py       # 3D paths, ground track, time-series
│
├── baseline/                       # Deterministic RTL benchmark
│   └── deterministic_rtl.py       # Proportional heading controller
│
├── env/                            # Gymnasium environment
│   ├── glider_env.py              # GliderEnv — owns simulation loop
│   ├── reward.py                  # Hierarchical reward function
│   └── curriculum.py             # 4-stage curriculum scheduler
│
├── sim/                            # Physics simulation components
│   ├── jsbsim_fdm.py             # JSBSim wrapper (SI interface)
│   ├── sensor_models.py          # GPS, IMU, Baro, LiDAR with noise
│   ├── wind.py                   # Mean wind + OU gust generator
│   └── math_utils.py             # Quaternion/Euler conversions, state builder
│
├── tests/                          # Pytest unit test suite
│   └── test_dynamics.py          # Physics, sensors, actuators, baseline
│
├── training/                       # RL training and evaluation
│   ├── configs/
│   │   └── base.yaml             # Single source of truth for all params
│   ├── train.py                  # PPO training entry point
│   └── evaluate.py               # RL vs baseline comparison harness
│
├── scratch/                        # Diagnostic and debug scripts
│   ├── bench_rtl.py              # Quick baseline sanity check
│   ├── diag_ctrl.py              # Attitude controller episode trace
│   └── jsbsim_smoke.py           # JSBSim installation test
│
├── validate_glide.py               # 5 physics validation gates
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Installation

**Requirements:** Python 3.10+, JSBSim 1.3.1+

```bash
# Clone the repository
git clone https://github.com/CS-UWC/rl-glider.git
cd rl-glider

# Install Python dependencies
pip install -r requirements.txt

# Verify JSBSim is working
python scratch/jsbsim_smoke.py
```

**`requirements.txt` includes:**
- `jsbsim >= 1.3.1`
- `gymnasium`
- `stable-baselines3`
- `numpy`, `scipy`
- `matplotlib`
- `pytest`

---

## Quick Start

### 1 — Validate the physics simulator

All five gates must pass before any training run is permitted:

```bash
python validate_glide.py
```

Expected output:
```
[PASS] Gate 1 — Energy Conservation    drift=0.003%  (threshold < 0.1%)
[PASS] Gate 2 — Glide Ratio            L/D=14.2      (threshold 10–20)
[PASS] Gate 3 — Stall Behaviour        CL drops past alpha=12 deg
[PASS] Gate 4 — Trim Stability         max|q|=0.008 rad/s  (threshold < 0.1)
[PASS] Gate 5 — Attitude Integrity     Euler angles finite over 30s

All 5 gates PASSED. Simulator ready for training.
```

### 2 — Run the unit tests

```bash
pytest tests/test_dynamics.py -v
```

### 3 — Benchmark the baseline controller

```bash
python -m training.evaluate --baseline-only --episodes 100
```

### 4 — Start training

```bash
python -m training.train --config training/configs/base.yaml --seed 42
```

---

## Physics Validation

`validate_glide.py` runs five correctness gates before any training is allowed. Each gate targets a different class of physics error:

| Gate | Test | Criterion | Detects |
|------|------|-----------|---------|
| 1. Energy conservation | CD=0 straight glide, 10 s | drift < 0.1% | Integration errors, wrong mass/velocity |
| 2. Glide ratio | Trimmed glide, 5 s | 10 < L/D < 20 | Incorrect aerodynamic scaling |
| 3. Stall | Sweep α past 12° | CL drops | Missing post-stall regime in aero table |
| 4. Trim stability | Trimmed flight, 15 s | \|q\| < 0.1 rad/s | Longitudinal instability |
| 5. Attitude integrity | Free flight, 30 s | Euler angles finite | Quaternion normalisation failure |

All five pass with JSBSim 1.3.1 and `aircraft/rlglider/rlglider.xml`.

---

## Training

```bash
# Train from scratch
python -m training.train \
    --config training/configs/base.yaml \
    --seed 42

# Resume from checkpoint
python -m training.train \
    --resume checkpoints/ppo_glider_1000000_steps \
    --seed 42

# Override specific parameters
python -m training.train \
    --set ppo.n_envs=16 ppo.total_timesteps=50000000

# Monitor with TensorBoard
tensorboard --logdir runs/
```

**Training outputs:**
```
checkpoints/
├── ppo_glider_<step>_steps.zip    # Policy checkpoint
├── vecnormalize_<step>.pkl        # Matching normalisation stats
└── best_model.zip                 # Best evaluation checkpoint
```

### PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Rollout steps per worker | 2048 |
| Mini-batch size | 256 |
| Learning rate | 3 × 10⁻⁴ |
| Clipping range (ε) | 0.2 |
| Entropy coefficient | 0.01 |
| Discount factor (γ) | 0.99 |
| Checkpoint interval | 100k steps |
| Evaluation interval | 50k steps |

Three callbacks manage the lifecycle: `CurriculumCallback` advances stages at 80% rolling success, `CheckpointCallback` saves policy and normalisation stats every 100k steps, and `EvalCallback` tracks the best-performing checkpoint.

---

## Evaluation

```bash
# Compare RL policy vs baseline (same episode seeds)
python -m training.evaluate \
    --model checkpoints/best_model \
    --vecnorm checkpoints/vecnormalize.pkl \
    --stage 3 \
    --episodes 100

# Baseline only (no model required)
python -m training.evaluate --baseline-only --episodes 100

# Test at a specific curriculum stage
python -m training.evaluate --baseline-only --episodes 50 --stage 3
```

**Metrics reported:** success rate, crash rate, timeout rate, mean reward, mean steps, mean final distance to home, and constraint violation rates (AGL, stall, bank).

Results are saved to `results/results_<timestamp>.json`.

---

## Visualisation

```bash
# Baseline trajectory at Stage 0
python -m analysis.plot_trajectories --stage 0 --seed 42

# RL policy trajectory at Stage 3
python -m analysis.plot_trajectories \
    --model checkpoints/best_model \
    --vecnorm checkpoints/vecnormalize.pkl \
    --stage 3

# Overlay RL vs baseline on the same figures
python -m analysis.plot_trajectories \
    --model checkpoints/best_model \
    --vecnorm checkpoints/vecnormalize.pkl \
    --compare
```

Produces three figures: 3D flight path, ground track with home radius circle, and time-series of airspeed, bank angle, and AGL.

---

## Testing

```bash
# Full test suite
pytest tests/test_dynamics.py -v

# Single test
pytest tests/test_dynamics.py::test_baseline_reaches_home -v

# With coverage
pytest tests/ --cov=sim --cov=env --cov-report=html
```

### Test Coverage

| Category | Tests |
|----------|-------|
| **Sensors** | LiDAR slant-range formula, update-rate counters for all four sensors, dropout probability |
| **Control** | JSBSim actuator rate-limit enforcement (rlglider.xml) |
| **Integration** | Baseline achieves ≥ 15% success in Stage 0 over 20 episodes |

Core physics correctness (energy conservation, glide ratio, stall, trim stability, attitude integrity) is covered separately by [`validate_glide.py`](#physics-validation)'s 5 gates against the real JSBSim FDM.

---

## Key Design Decisions

### Why JSBSim?

The original custom Python 6-DOF integrator accumulated physics bugs that reduced the baseline success rate below 5%, making training impossible. JSBSim 1.3.1 was adopted because it is validated by NASA and the FAA, supports real aerodynamic coefficient tables, models actuator lag and rate limits natively, and maintains a Python control interface compatible with the existing training loop.

### Why separate the attitude controller?

Keeping stabilisation in a deterministic 200 Hz loop (mirroring the ESP32 firmware) reduces the RL action space to two values — bank command and airspeed command — and isolates the learning problem from demanding inner-loop dynamics. The policy only needs to learn navigation strategy; the attitude controller handles the physics.

### Why curriculum learning?

Stage 0 (no wind, no noise) lets the policy learn the spatial structure of the RTL task cleanly before disturbances are introduced. Each stage advances only when 80% rolling success is sustained over 100 episodes, ensuring the policy is genuinely competent rather than advancing by chance.

---

## Interface Contracts

### Observation Vector (12 elements, float32, normalised)

| Index | Signal | Source |
|-------|--------|--------|
| 0, 1 | Δx, Δy to home (m) | GPS |
| 2 | Δz to home (m) | Barometer |
| 3 | Course angle χ (rad) | GPS velocity |
| 4, 5, 6 | Body velocity [u, v, w] (m/s) | GPS |
| 7, 8 | Roll φ, pitch θ (rad) | IMU |
| 9 | Yaw rate r (rad/s) | IMU |
| 10 | LiDAR AGL (m); 0 if invalid | LiDAR |
| 11 | Airspeed (m/s) | Pitot |

### Action Vector (2 elements, float32, ∈ [−1, 1])

| Index | Signal | Denormalised range |
|-------|--------|--------------------|
| 0 | Bank command φ_cmd | ±45° |
| 1 | Airspeed command V_cmd | 5–13 m/s |

### Sensor Update Rates

| Sensor | Rate | Noise model |
|--------|------|-------------|
| GPS | 5 Hz | σ_pos = 1.5 m, σ_vel = 0.3 m/s |
| IMU | 200 Hz | σ_att = 1°, σ_rate = 0.5°/s |
| Barometer (BMP280) | 25 Hz | σ_alt = 1.0 m |
| Ultrasonic (RCWL-1655) | 20 Hz | σ_range = 0.08 m, range-gated [0.2, 4.5] m, 10% dropout |

No LiDAR is fitted on the hardware — ground proximity is sensed by a short-range ultrasonic, valid only on final approach/flare (<4.5 m), not as a continuous in-flight AGL floor. See the as-built electronics notes for the full hardware picture.

---

## Curriculum Stages

Launch altitude and offsets reflect the as-built hardware: the glider is
hand/ground-launched (peak altitude after the launch zoom-climb ~15-25 m),
not tow-released at altitude. Home radius is widened versus earlier designs
to account for the NEO-6M GPS's ~2.5 m CEP. See the as-built electronics
notes for the full hardware picture.

| Stage | Max wind | Noise scale | Home radius | Launch alt | Launch offset | Advance rule |
|-------|----------|-------------|-------------|------------|----------------|--------------|
| 0 | 0 m/s | 0% | 25 m | 25 m | 50–90 m | 80% over 100 eps |
| 1 | 3 m/s | 10% | 22 m | 23 m | 45–80 m | 80% over 100 eps |
| 2 | 6 m/s | 15% | 20 m | 21 m | 40–70 m | 80% over 100 eps |
| 3 | 9 m/s | 20% | 20 m | 20 m | 40–70 m | Final stage |

**Baseline performance** (measured, n=100/stage, `DeterministicRTL` PD heading controller):

| Stage | Conditions | Success rate | Crash rate | Timeout rate |
|-------|------------|---------------|------------|--------------|
| 0 | No wind, no noise | 21% | 79% | 0% |
| 1 | Light wind, mild noise | 19% | 81% | 0% |
| 2 | Moderate wind + gusts | 21% | 79% | 0% |
| 3 | Full domain randomisation | 23% | 77% | 0% |

At this reduced altitude budget, failures resolve as crashes rather than
timeouts (the old 100-120 m tow-release scale gave enough altitude margin
for the baseline to wander for the full episode instead of running out of
height) -- a more informative failure signature for reward shaping.

An earlier P-only heading controller (no damping) measured ~28-29% here, but
that number was produced while `sim/jsbsim_fdm.py`'s `position_ned` had a
since-fixed bug: it reported the *unsigned magnitude* of NED displacement
along each axis rather than signed displacement, so the glider's perceived
bearing to home was silently wrong whenever its net displacement crossed to
the other side of the launch point. With that fixed, the same P-only
controller's true success rate was ~4-6% -- per-episode traces showed a
genuine, non-decaying heading limit cycle (bank command saturated on ~99%
of steps), not a subtle mistuning. `DeterministicRTL` is now a PD
controller (`kp_bank=0.3`, `kd_bank=5.0`, damped by IMU yaw rate rather
than a finite difference of the 5 Hz GPS course angle, which would alias
against the 20 Hz policy rate) -- see `baseline/deterministic_rtl.py`'s
module docstring for the full derivation. The numbers above are the ones to
trust; the old ~29% figure should not be used as a reference point.

**RL target:** > 60% at Stage 3.

---

## Reward Function

```
r = w_progress · Δd_home
  + w_airtime  · Δt
  − w_agl      · max(0, h_min − h)²
  − w_stall    · max(0, α − α_safe)
  − w_bank     · max(0, |φ| − φ_soft)
  − w_smooth   · ‖a − a_prev‖²
```

| Component | Weight | Purpose |
|-----------|--------|---------|
| Progress | 1.0 | Dominant dense signal — drives the policy toward home |
| Airtime | 0.1 | Encourages survival without creating a loitering incentive |
| AGL penalty | 5.0 | Squared term creates a convex gradient near the ground |
| Stall penalty | 2.0 | Linear term discourages high angle-of-attack flight |
| Bank penalty | 1.0 | Discourages excessive bank beyond the soft limit |
| Smoothness | 0.01 | Suppresses oscillatory command sequences |
| Terminal success | +500 | Reaching home within R_home |
| Terminal crash | −100 | Ground impact |

---

## Diagnostics

```bash
# Trace a single baseline episode (step-by-step state output)
python scratch/diag_ctrl.py

# Quick 100-episode sanity check
python scratch/bench_rtl.py

# Test JSBSim installation only
python scratch/jsbsim_smoke.py
```

---

## Project Status

| Milestone | Status |
|-----------|--------|
| Requirements & architecture (Term 1) | ✅ Complete |
| JSBSim migration | ✅ Complete |
| GliderEnv implementation | ✅ Complete |
| Physics validation (5 gates) | ✅ All passing |
| Unit test suite | ✅ All passing |
| Deterministic baseline | ✅ ~20-23% across all 4 stages (PD heading controller) |
| PPO training to convergence | 🔄 Term 3 |
| ESP32 firmware integration | 🔄 Term 3 |
| Shadow-mode flight testing | 🔄 Term 3 |
| Real-world deployment | 🔄 Term 3+ |

---

## References

| Area | Reference |
|------|-----------|
| PPO | Schulman et al., *Proximal Policy Optimization Algorithms*, 2017 |
| SAC | Haarnoja et al., *Soft Actor-Critic*, 2018 |
| DDPG | Lillicrap et al., *Continuous Control with Deep RL*, 2015 |
| Safe RL | Achiam et al., *Constrained Policy Optimization*, 2017 |
| Safety layers | Dalal et al., *Safe Exploration in Continuous Action Spaces*, 2018 |
| Domain randomisation | Tobin et al., *Domain Randomization for Sim-to-Real*, 2017 |
| Dynamics randomisation | Peng et al., *Sim-to-Real Transfer with Dynamics Randomization*, 2018 |
| UAV sim-to-real | Sadeghi & Levine, *CAD2RL*, 2017 |
| JSBSim | Berndt, *JSBSim: An Open Source Flight Dynamics Model in C++*, 2004 |
| Attitude estimation | Mahony et al., *Nonlinear Complementary Filters on SO(3)*, 2008 |
| Stable-Baselines3 | Raffin et al., 2021 — [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io) |
| Gymnasium | Towers et al., 2024 — [gymnasium.farama.org](https://gymnasium.farama.org) |

---

*COS 731/732 Honours Project · Department of Computer Science · University of the Western Cape · 2026*
