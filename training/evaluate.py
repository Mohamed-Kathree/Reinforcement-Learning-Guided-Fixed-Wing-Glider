"""
evaluate.py
===========
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Comparative evaluation harness: runs the trained RL policy and the
deterministic RTL baseline through an identical evaluation loop, then
prints a side-by-side summary and saves a JSON results file.

Both controllers are evaluated against the same episode seeds so that
the comparison is fair (same wind, launch angle, sensor noise per episode).

Usage:
    # Evaluate the best saved model against the baseline:
    python -m training.evaluate \\
        --model     checkpoints/best_model \\
        --vecnorm   checkpoints/vecnormalize.pkl \\
        --config    training/configs/base.yaml \\
        --stage     0 \\
        --episodes  100

    # Evaluate only the baseline (no RL model required):
    python -m training.evaluate --baseline-only --episodes 50

Outputs:
    results_<timestamp>.json  -- full per-episode data + aggregate statistics

Coordinate frames used:
    NED : North-East-Down inertial frame (world)
    BODY: Forward-Right-Down body frame (FRD, attached to glider)
    WIND: Stability/wind frame (x into relative wind)

Quaternion convention: [q0, q1, q2, q3] where q0 is the scalar component.
Rotation R maps NED -> BODY: v_body = R @ v_ned

Units: SI throughout (m, m/s, rad, rad/s, kg, N, N*m)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any, Protocol

import numpy as np
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.glider_env import GliderEnv
from env.curriculum import STAGES
from env.reward import DEFAULT_REWARD_CFG
from baseline.deterministic_rtl import DeterministicRTL


# ---------------------------------------------------------------------------
# Shared policy protocol
# ---------------------------------------------------------------------------

class Policy(Protocol):
    """Minimal interface shared by SB3 policies and DeterministicRTL."""
    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Any]: ...


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

_LAUNCH_KEY_MAP = {
    'speed_min_ms':  'launch_speed_min_ms',
    'speed_max_ms':  'launch_speed_max_ms',
    'pitch_min_deg': 'launch_pitch_min_deg',
    'pitch_max_deg': 'launch_pitch_max_deg',
}


def _build_eval_env(cfg: dict, stage_idx: int) -> GliderEnv:
    """Return a plain (un-vectorised) GliderEnv at the given curriculum stage."""
    stage_cfg = dict(STAGES[stage_idx])
    reward_cfg = dict(DEFAULT_REWARD_CFG)
    reward_cfg.update(cfg.get('reward', {}))
    # launch.speed_{min,max}_ms / pitch_{min,max}_deg are a fixed hardware
    # uncertainty, not curriculum-varied (see train.flat_env_cfg for the same
    # mapping) -- alt0_m is NOT pulled from here since stage_cfg (below)
    # already carries the correct per-stage value.
    launch = cfg.get('launch', {})
    for yaml_key, cfg_key in _LAUNCH_KEY_MAP.items():
        if yaml_key in launch:
            reward_cfg[cfg_key] = launch[yaml_key]
    reward_cfg.update(stage_cfg)
    return GliderEnv(cfg=reward_cfg)


def _build_normed_eval_env(
    cfg: dict,
    stage_idx: int,
    vecnorm_path: str,
) -> VecNormalize:
    """Return a VecNormalize-wrapped DummyVecEnv with frozen stats.

    The VecNormalize stats are loaded from vecnorm_path so that observations
    are normalised identically to the training environment.

    Args:
        cfg          : full config dict
        stage_idx    : curriculum stage to evaluate on
        vecnorm_path : path to the .pkl file saved during training

    Returns:
        VecNormalize with training=False (frozen stats), norm_reward=False
    """
    def _make():
        return _build_eval_env(cfg, stage_idx)

    venv = DummyVecEnv([_make])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training    = False   # do NOT update running stats during eval
    venv.norm_reward = False   # report raw rewards
    return venv


# ---------------------------------------------------------------------------
# Single-controller evaluation loop
# ---------------------------------------------------------------------------

def run_episodes(
    policy:     Policy,
    env:        GliderEnv | VecNormalize,
    n_episodes: int,
    seeds:      list[int],
    deterministic: bool = True,
) -> list[dict]:
    """Run n_episodes and collect per-episode metrics.

    Works with both a raw GliderEnv (for the deterministic baseline) and a
    VecNormalize-wrapped env (for the RL policy).  The function detects
    which type it received and adapts accordingly.

    Args:
        policy       : object with .predict(obs, deterministic) -> (action, _)
        env          : GliderEnv or VecNormalize
        n_episodes   : number of episodes to run
        seeds        : list of episode seeds (length == n_episodes)
        deterministic: whether to call policy.predict(deterministic=True)

    Returns:
        List of per-episode result dicts with keys:
            seed, success, crash, timeout, total_reward, steps,
            final_dist_home, agl_violations, stall_violations, bank_violations
    """
    is_vecenv = isinstance(env, VecNormalize)
    results: list[dict] = []

    for ep_idx in range(n_episodes):
        seed = seeds[ep_idx]

        # DeterministicRTL is stateful (yaw-rate derivative for its PD
        # heading controller) and must be reset per episode; SB3 policies
        # have no such method, hence the guard.
        if hasattr(policy, 'reset'):
            policy.reset()

        if is_vecenv:
            # VecEnv reset doesn't accept seed directly; reset the underlying env
            env.env_method('reset', seed=seed)
            obs = env.reset()
        else:
            obs, _ = env.reset(seed=seed)

        total_reward  = 0.0
        steps         = 0
        agl_v         = 0
        stall_v       = 0
        bank_v        = 0
        final_dist    = None
        success       = False
        crash         = False

        while True:
            action, _ = policy.predict(obs, deterministic=deterministic)

            if is_vecenv:
                obs, reward, done, info_arr = env.step(action)
                r    = float(reward[0])
                done = bool(done[0])
                info = info_arr[0]
            else:
                obs, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            total_reward += r
            steps        += 1
            agl_v        += int(info.get('agl_violation',   False))
            stall_v      += int(info.get('stall_violation', False))
            bank_v       += int(info.get('bank_violation',  False))
            final_dist    = float(info.get('dist_home', np.nan))

            if done:
                success = bool(info.get('success', False))
                crash   = bool(info.get('crash',   False))
                break

        results.append({
            'seed':           seed,
            'success':        success,
            'crash':          crash,
            'timeout':        not success and not crash,
            'total_reward':   total_reward,
            'steps':          steps,
            'final_dist_home':final_dist,
            'agl_violations': agl_v,
            'stall_violations': stall_v,
            'bank_violations':  bank_v,
        })

    return results


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def aggregate(results: list[dict]) -> dict:
    """Compute aggregate statistics from a list of per-episode result dicts."""
    n = len(results)
    rewards    = [r['total_reward']    for r in results]
    steps_list = [r['steps']           for r in results]
    dists      = [r['final_dist_home'] for r in results]
    agl_v      = [r['agl_violations']  for r in results]
    stall_v    = [r['stall_violations']for r in results]
    bank_v     = [r['bank_violations'] for r in results]

    return {
        'n_episodes':       n,
        'success_rate':     sum(r['success'] for r in results) / n,
        'crash_rate':       sum(r['crash']   for r in results) / n,
        'timeout_rate':     sum(r['timeout'] for r in results) / n,
        'mean_reward':      float(np.mean(rewards)),
        'std_reward':       float(np.std(rewards)),
        'mean_steps':       float(np.mean(steps_list)),
        'mean_dist_home':   float(np.nanmean(dists)),
        'mean_agl_violations':   float(np.mean(agl_v)),
        'mean_stall_violations': float(np.mean(stall_v)),
        'mean_bank_violations':  float(np.mean(bank_v)),
    }


# ---------------------------------------------------------------------------
# Pretty-print comparison table
# ---------------------------------------------------------------------------

def print_baseline_summary(
    stats:     dict,
    stage_idx: int,
    n_episodes: int,
    seed:      int,
) -> None:
    """Print a formatted single-controller summary in demo style."""
    sep = '=' * 40
    mean_steps = stats['mean_steps'] if stats['mean_steps'] > 0 else 1.0

    agl_pct   = stats['mean_agl_violations']   / mean_steps * 100.0
    stall_pct = stats['mean_stall_violations'] / mean_steps * 100.0
    bank_pct  = stats['mean_bank_violations']  / mean_steps * 100.0

    print(sep)
    print(f"  BASELINE EVALUATION — Stage {stage_idx}")
    print(f"  Episodes: {n_episodes}  |  Seed: {seed}")
    print(sep)
    print(f"  Success rate:    {stats['success_rate']:>10.1%}")
    print(f"  Crash rate:      {stats['crash_rate']:>10.1%}")
    print(f"  Timeout rate:    {stats['timeout_rate']:>10.1%}")
    print(f"  Mean reward:     {stats['mean_reward']:>10.1f}  (± {stats['std_reward']:.1f})")
    print(f"  Mean steps:      {stats['mean_steps']:>10.1f}")
    print(f"  Mean final dist: {stats['mean_dist_home']:>9.1f} m")
    print(f"  AGL violations:  {agl_pct:>9.1f}%")
    print(f"  Stall violations:{stall_pct:>9.1f}%")
    print(f"  Bank violations: {bank_pct:>9.1f}%")
    print(sep)


def print_comparison(
    rl_stats:       dict | None,
    base_stats:     dict,
    stage_idx:      int,
    n_episodes:     int,
) -> None:
    """Print a formatted comparison table to stdout."""
    sep = '-' * 62
    print(sep)
    print(f"  Evaluation  |  stage={stage_idx}  |  n={n_episodes} episodes")
    print(sep)
    fmt_head = f"  {'Metric':<26} {'RL Policy':>14} {'Deterministic':>14}"
    print(fmt_head)
    print(sep)

    metrics = [
        ('Success rate',        'success_rate',       '.1%'),
        ('Crash rate',          'crash_rate',         '.1%'),
        ('Timeout rate',        'timeout_rate',       '.1%'),
        ('Mean reward',         'mean_reward',        '.1f'),
        ('Std reward',          'std_reward',         '.1f'),
        ('Mean steps',          'mean_steps',         '.0f'),
        ('Mean dist home (m)',  'mean_dist_home',     '.1f'),
        ('Mean AGL violations', 'mean_agl_violations','.1f'),
        ('Mean stall events',   'mean_stall_violations','.1f'),
        ('Mean bank events',    'mean_bank_violations',  '.1f'),
    ]

    for label, key, fmt in metrics:
        b_val = base_stats[key]
        b_str = format(b_val, fmt)
        if rl_stats is not None:
            r_val = rl_stats[key]
            r_str = format(r_val, fmt)
        else:
            r_str = 'N/A'
        print(f"  {label:<26} {r_str:>14} {b_str:>14}")

    print(sep)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    model_path:    str | None,
    vecnorm_path:  str | None,
    config_path:   str,
    stage_idx:     int,
    n_episodes:    int,
    seed:          int,
    output_dir:    str,
    baseline_only: bool = False,
) -> dict:
    """Run full comparative evaluation and save results.

    Args:
        model_path    : path to .zip policy checkpoint (None for baseline-only)
        vecnorm_path  : path to VecNormalize .pkl (None for baseline-only)
        config_path   : path to base.yaml
        stage_idx     : curriculum stage to evaluate on (0–3)
        n_episodes    : number of evaluation episodes
        seed          : base RNG seed for reproducible episode seeds
        output_dir    : directory to write results JSON
        baseline_only : skip RL policy, evaluate baseline only

    Returns:
        Dict with 'rl', 'baseline', and 'metadata' keys.
    """
    cfg = yaml.safe_load(pathlib.Path(config_path).read_text())

    # Shared episode seeds — same for both policies so comparison is fair
    rng   = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_episodes)]

    # --- Baseline --------------------------------------------------------
    if not baseline_only:
        print(f"\nEvaluating DeterministicRTL  (stage {stage_idx}, {n_episodes} episodes) …")
    base_env  = _build_eval_env(cfg, stage_idx)
    base_ctrl = DeterministicRTL()
    base_results = run_episodes(base_ctrl, base_env, n_episodes, seeds)
    base_stats   = aggregate(base_results)
    if not baseline_only:
        print(f"  Done.  success={base_stats['success_rate']:.1%}  "
              f"reward={base_stats['mean_reward']:.1f}")

    # --- RL policy -------------------------------------------------------
    rl_stats   = None
    rl_results = None

    if not baseline_only:
        if model_path is None:
            raise ValueError("--model is required unless --baseline-only is set")
        if vecnorm_path is None:
            raise ValueError("--vecnorm is required unless --baseline-only is set")

        print(f"\nEvaluating RL policy  (stage {stage_idx}, {n_episodes} episodes) …")
        rl_venv  = _build_normed_eval_env(cfg, stage_idx, vecnorm_path)
        rl_model = PPO.load(model_path, env=rl_venv)
        rl_results = run_episodes(
            rl_model, rl_venv, n_episodes, seeds, deterministic=True
        )
        rl_stats = aggregate(rl_results)
        rl_venv.close()
        print(f"  Done.  success={rl_stats['success_rate']:.1%}  "
              f"reward={rl_stats['mean_reward']:.1f}")

    # --- Print results ---------------------------------------------------
    if baseline_only:
        print_baseline_summary(base_stats, stage_idx, n_episodes, seed)
    else:
        print_comparison(rl_stats, base_stats, stage_idx, n_episodes)

    # --- Save JSON results -----------------------------------------------
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_path  = pathlib.Path(output_dir) / f"results_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'metadata': {
            'timestamp':   timestamp,
            'stage':       stage_idx,
            'n_episodes':  n_episodes,
            'seed':        seed,
            'model_path':  model_path,
            'vecnorm_path':vecnorm_path,
            'config_path': config_path,
        },
        'baseline': {
            'aggregate': base_stats,
            'episodes':  base_results,
        },
        'rl': {
            'aggregate': rl_stats,
            'episodes':  rl_results,
        } if rl_stats is not None else None,
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved → {out_path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate RL policy vs deterministic RTL baseline'
    )
    parser.add_argument(
        '--model', default=None,
        help='path to trained PPO .zip checkpoint',
    )
    parser.add_argument(
        '--vecnorm', default=None,
        help='path to VecNormalize .pkl stats file',
    )
    parser.add_argument(
        '--config', default='training/configs/base.yaml',
        help='path to YAML config (default: training/configs/base.yaml)',
    )
    parser.add_argument(
        '--stage', type=int, default=0, choices=[0, 1, 2, 3],
        help='curriculum stage to evaluate on (default: 0)',
    )
    parser.add_argument(
        '--episodes', type=int, default=100,
        help='number of evaluation episodes (default: 100)',
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='base RNG seed for episode generation (default: 0)',
    )
    parser.add_argument(
        '--output-dir', default='results',
        help='directory for JSON results output (default: results/)',
    )
    parser.add_argument(
        '--baseline-only', action='store_true',
        help='evaluate only the deterministic baseline (no RL model needed)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    evaluate(
        model_path    = args.model,
        vecnorm_path  = args.vecnorm,
        config_path   = args.config,
        stage_idx     = args.stage,
        n_episodes    = args.episodes,
        seed          = args.seed,
        output_dir    = args.output_dir,
        baseline_only = args.baseline_only,
    )
