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
- [Dashboard](#dashboard)
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

**Term 2 status:** Physics simulation validated, Gymnasium environment implemented, deterministic baseline benchmarked at 100% (Stage 0) down to 84% (Stage 3) success (see [Curriculum Stages](#curriculum-stages) for the full table and correction history). Training infrastructure ready.

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
│   └── benchmark.py              # the single evaluator: policy vs paired baseline
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
python -m training.benchmark --policy baseline --episodes 100
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
# Trained policy vs the deterministic baseline, paired on identical seeds,
# all four curriculum stages, 100 episodes each
python -m training.benchmark --policy trained \
    --model checkpoints_v20/best_model.zip \
    --vecnorm checkpoints_v20/vecnormalize_best.pkl

# Floor controls and the baseline alone (no model required)
python -m training.benchmark --policy random
python -m training.benchmark --policy zero
python -m training.benchmark --policy baseline --stages 3 --episodes 50

# Home-information ablation: obs[0] dx_home, obs[1] dy_home, obs[3] course zeroed
python -m training.benchmark --policy trained --model ... --vecnorm ... --ablate-home
```

**Reported:** four outcome buckets (success / soft landing / crash / timeout)
with Wilson 95% intervals, mean landing quality and miss distance, and, for
paired runs, the success contingency table with McNemar's exact p-value.
Every episode's sampled airframe (`dr_params`), outcome, miss distance and
landing quality is written to `results/benchmark_*.json` with a provenance
block (policy, VecNormalize and config SHA-256, git SHA, UTC timestamp).
