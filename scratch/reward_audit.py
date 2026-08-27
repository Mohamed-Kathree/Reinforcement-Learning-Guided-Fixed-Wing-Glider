"""
reward_audit.py
================
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Phase 5 (RLGlider_Phase5_Reward_Rescaling_Spec.md, Phase 0) reward-budget
instrument. Runs DeterministicRTL baseline episodes per curriculum stage,
accumulates per-component reward sums, and reports:

  - a per-stage table of episode-level means (steps, dist_home, quality,
    per-component reward sums, total return)
  - a reward-budget summary: each component's mean magnitude as a percentage
    of the summed absolute magnitude of all components (the metric that
    identified Defect A -- the reachability barrier eating ~98% of return)
  - the barrier-firing-altitude diagnostic (min/median/max AGL, % below 3 m)
  - a terminal-reward gradient probe: r_terminal at quality=1.0 swept across
    dist_home, with finite differences between adjacent entries (the metric
    that identified Defect B -- a numerically dead precision gradient)

Full results are written to scratch/reward_audit_<label>.json so before/after
runs can be diffed mechanically.

Usage:
    python -m scratch.reward_audit --n 20 --stages 0,1,2,3 --label current
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from env.glider_env import GliderEnv
from env.curriculum import STAGES
from env.reward import compute_reward, DEFAULT_REWARD_CFG
from baseline.deterministic_rtl import DeterministicRTL
from sim.math_utils import build_state


COMPONENTS: list[str] = [
    'r_path', 'penalty_unreach', 'penalty_stall', 'penalty_bank',
    'penalty_smooth', 'r_terminal',
]

GRADIENT_PROBE_DISTS: list[float] = [2, 5, 10, 15, 20, 25, 30, 40, 60, 100, 150]


# ---------------------------------------------------------------------------
# Terminal-reward gradient probe
# ---------------------------------------------------------------------------

def terminal_reward_at_distance(dist_home: float, cfg: dict) -> float:
    """r_terminal at quality=1.0 for a given dist_home, under cfg's sigma_centre_m.

    V=9.0 (== vtd_ok) and alpha=3deg (<< alpha_ok_deg) and vs=0.5 (<< sink_ok)
    and roll=0 (<< roll_ok_deg) put all four grading factors at exactly 1.0,
    i.e. quality=1.0 -- isolating the precision term's shape from the quality
    gate.
    """
    state = build_state(
        p_ned=np.array([dist_home, 0.0, 0.0]),
        v_body=np.array([9.0, 0.0, 0.5]),
    )
    obs = np.zeros(12, dtype=np.float32)
    _, _, _, info = compute_reward(
        state=state, prev_state=state, obs=obs,
        action=np.zeros(2, dtype=np.float32), prev_action=np.zeros(2, dtype=np.float32),
        home_ned=np.zeros(3), cfg=cfg,
        alpha_true=np.radians(3.0), airspeed_true=9.0,
        touched_down=True, fault=False,
    )
    assert info['quality'] == 1.0, f"gradient-probe fixture did not reach quality=1.0 (got {info['quality']})"
    return float(info['r_terminal'])


def gradient_probe(cfg: dict) -> tuple[list[float], list[float]]:
    """Returns (r_terminal values, adjacent finite differences) across GRADIENT_PROBE_DISTS."""
    values = [terminal_reward_at_distance(d, cfg) for d in GRADIENT_PROBE_DISTS]
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    return values, diffs


# ---------------------------------------------------------------------------
# Per-stage episode runner
# ---------------------------------------------------------------------------

def run_stage(stage_idx: int, n: int, seed_base: int, adversarial: bool = False) -> dict:
    """adversarial=True runs a constant [0.0, -1.0] action (wings level,
    minimum speed -- maximum-endurance drift) instead of the baseline
    controller. Used to sanity-check that barrier_min_agl_m/glide_ratio_usable
    still leave the barrier CAPABLE of firing (RLGlider_Phase5_Reward_
    Rescaling_Spec.md §1.3) -- a constraint that can never bind is not a
    constraint."""
    env = GliderEnv()
    env.set_stage(dict(STAGES[stage_idx]))
    ctrl = DeterministicRTL()
    rng = np.random.default_rng(seed_base + stage_idx)

    episodes: list[dict] = []
    barrier_altitudes: list[float] = []

    for _ in range(n):
        ctrl.reset()
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))

        sums = {c: 0.0 for c in COMPONENTS}
        total_return = 0.0
        step_count = 0
        unreach_steps = 0
        info: dict = {}
        truncated = False

        while True:
            if adversarial:
                action = np.array([0.0, -1.0], dtype=np.float32)
            else:
                action = ctrl.act(obs)
            obs, r, terminated, truncated, info = env.step(action)
            step_count += 1
            total_return += float(r)
            for c in COMPONENTS:
                sums[c] += float(info[c])
            if info['unreach_violation']:
                unreach_steps += 1
                barrier_altitudes.append(float(env._fdm.altitude))
            if terminated or truncated:
                break

        episodes.append({
            'steps':          step_count,
            'dist_home':      float(info['dist_home']),
            'quality':        float(info['quality']),
            'truncated':      bool(truncated),
            'unreach_steps':  unreach_steps,
            'total_return':   total_return,
            **sums,
        })

    cfg = {**DEFAULT_REWARD_CFG, **STAGES[stage_idx]}
    probe_values, probe_diffs = gradient_probe(cfg)

    return {
        'stage':             stage_idx,
        'sigma_centre_m':    STAGES[stage_idx]['sigma_centre_m'],
        'n_episodes':        n,
        'episodes':          episodes,
        'barrier_altitudes': barrier_altitudes,
        'gradient_probe': {
            'dist_home': GRADIENT_PROBE_DISTS,
            'r_terminal': probe_values,
            'finite_diff': probe_diffs,
        },
    }


def run_stage_policy(stage_idx: int, n: int, seed_base: int, model_path: str, vecnorm_path: str) -> dict:
    """Same per-episode audit as run_stage(), but driving a trained PPO
    policy (loaded from model_path + its matching VecNormalize stats at
    vecnorm_path) instead of the scripted DeterministicRTL baseline.

    A learned policy can find reward structure a scripted controller never
    visits, so RLGlider_Phase5_Reward_Rescaling_Spec.md §5.3's criteria 5.5/
    5.6 (barrier/terminal reward-budget share under a TRAINED policy) require
    this, not just a baseline-only audit.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def _make():
        e = GliderEnv()
        e.set_stage(dict(STAGES[stage_idx]))
        return e

    venv = DummyVecEnv([_make])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training    = False   # freeze running stats
    venv.norm_reward = False   # report raw rewards, not normalised

    model = PPO.load(model_path, env=venv)
    rng = np.random.default_rng(seed_base + stage_idx)

    episodes: list[dict] = []
    barrier_altitudes: list[float] = []

    for _ in range(n):
        seed = int(rng.integers(0, 2**31))
        venv.env_method('reset', seed=seed)
        obs = venv.reset()

        sums = {c: 0.0 for c in COMPONENTS}
        total_return = 0.0
        step_count = 0
        unreach_steps = 0
        info: dict = {}

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info_arr = venv.step(action)
            info = info_arr[0]
            step_count += 1
            total_return += float(reward[0])
            for c in COMPONENTS:
                sums[c] += float(info[c])
            if info['unreach_violation']:
                unreach_steps += 1
                barrier_altitudes.append(float(venv.get_attr('_fdm')[0].altitude))
            if bool(done[0]):
                break

        # outcome is None iff the episode ended by MAX_STEPS truncation
        # without ever touching down or faulting (env/reward.py sets it only
        # inside the touchdown/fault branches) -- the VecEnv API collapses
        # terminated/truncated into one `done` flag, so this is how the
        # distinction survives.
        truncated = info.get('outcome') is None

        episodes.append({
            'steps':          step_count,
            'dist_home':      float(info['dist_home']),
            'quality':        float(info['quality']),
            'truncated':      bool(truncated),
            'unreach_steps':  unreach_steps,
            'total_return':   total_return,
            **sums,
        })

    cfg = {**DEFAULT_REWARD_CFG, **STAGES[stage_idx]}
    probe_values, probe_diffs = gradient_probe(cfg)

    return {
        'stage':             stage_idx,
        'sigma_centre_m':    STAGES[stage_idx]['sigma_centre_m'],
        'n_episodes':        n,
        'episodes':          episodes,
        'barrier_altitudes': barrier_altitudes,
        'gradient_probe': {
            'dist_home': GRADIENT_PROBE_DISTS,
            'r_terminal': probe_values,
            'finite_diff': probe_diffs,
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _mean(episodes: list[dict], key: str) -> float:
    return float(np.mean([e[key] for e in episodes])) if episodes else float('nan')


def print_stage_report(result: dict) -> None:
    stage = result['stage']
    episodes = result['episodes']
    n = len(episodes)

    print("=" * 78)
    print(f"STAGE {stage}  (n={n} episodes, sigma_centre_m={result['sigma_centre_m']})")
    print("=" * 78)

    print(f"{'steps':>8} {'dist_home':>10} {'quality':>8} {'trunc':>6} "
          f"{'r_path':>9} {'unreach':>10} {'r_term':>9} {'return':>10}")
    for e in episodes:
        print(f"{e['steps']:>8} {e['dist_home']:>10.2f} {e['quality']:>8.2f} "
              f"{str(e['truncated']):>6} {e['r_path']:>9.2f} {e['penalty_unreach']:>10.2f} "
              f"{e['r_terminal']:>9.2f} {e['total_return']:>10.2f}")

    print("-" * 78)
    print(f"{'MEAN':>8} {_mean(episodes,'dist_home'):>10.2f} {_mean(episodes,'quality'):>8.2f} "
          f"{sum(e['truncated'] for e in episodes)/n:>6.0%} "
          f"{_mean(episodes,'r_path'):>9.2f} {_mean(episodes,'penalty_unreach'):>10.2f} "
          f"{_mean(episodes,'r_terminal'):>9.2f} {_mean(episodes,'total_return'):>10.2f}")

    # --- reward-budget summary: each component's mean |magnitude| as a % of
    # the summed |magnitude| of all components ---
    means = {c: _mean(episodes, c) for c in COMPONENTS}
    abs_sum = sum(abs(v) for v in means.values())
    print()
    print("Reward-budget summary (mean magnitude as % of total |magnitude|):")
    for c in COMPONENTS:
        pct = (abs(means[c]) / abs_sum * 100.0) if abs_sum > 0 else 0.0
        print(f"  {c:<18} mean={means[c]:>10.2f}   {pct:>5.1f}%")

    # --- barrier-firing altitude diagnostic ---
    alts = result['barrier_altitudes']
    print()
    if alts:
        alts_arr = np.array(alts)
        below3 = float(np.mean(alts_arr < 3.0)) * 100.0
        print(f"Barrier firing altitude diagnostic (n={len(alts)} violating steps):")
        print(f"  min={alts_arr.min():.2f} m  median={np.median(alts_arr):.2f} m  "
              f"max={alts_arr.max():.2f} m  {below3:.1f}% below 3 m AGL")
    else:
        print("Barrier firing altitude diagnostic: barrier never fired.")

    # --- terminal-reward gradient probe ---
    probe = result['gradient_probe']
    print()
    print("Terminal-reward gradient probe (quality=1.0, sweeping dist_home):")
    print("  " + "  ".join(f"d={d:>3}" for d in probe['dist_home']))
    print("  " + "  ".join(f"{v:>6.3f}" for v in probe['r_terminal']))
    print("  finite differences between adjacent entries:")
    print("  " + "  ".join(f"{d:>7.4f}" for d in probe['finite_diff']))
    print()


def to_jsonable(result: dict) -> dict:
    return json.loads(json.dumps(result, default=lambda o: float(o)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n',      type=int, default=20, help='episodes per stage')
    ap.add_argument('--stages', type=str, default='0,1,2,3', help='comma-separated stage indices')
    ap.add_argument('--label',  type=str, default='current', help='output label for the JSON dump')
    ap.add_argument('--seed-base', type=int, default=12345, help='base seed offset per stage')
    ap.add_argument('--adversarial', action='store_true',
                     help='run a constant [0.0, -1.0] wings-level min-speed action instead of '
                          'the baseline controller, to check the barrier can still fire')
    ap.add_argument('--policy', type=str, default=None,
                     help='path to a trained PPO .zip checkpoint -- audit this policy instead '
                          'of the scripted DeterministicRTL baseline. Requires --vecnorm.')
    ap.add_argument('--vecnorm', type=str, default=None,
                     help='path to the VecNormalize .pkl matching --policy')
    args = ap.parse_args()

    if args.policy is not None and args.vecnorm is None:
        raise SystemExit('--policy requires --vecnorm (the matching VecNormalize .pkl)')
    if args.policy is not None and args.adversarial:
        raise SystemExit('--policy and --adversarial are mutually exclusive')

    stage_indices = [int(s) for s in args.stages.split(',')]

    all_results = []
    for stage_idx in stage_indices:
        if args.policy is not None:
            result = run_stage_policy(stage_idx, args.n, args.seed_base, args.policy, args.vecnorm)
        else:
            result = run_stage(stage_idx, args.n, args.seed_base, adversarial=args.adversarial)
        print_stage_report(result)
        all_results.append(result)

    out_path = Path(__file__).parent / f"reward_audit_{args.label}.json"
    with open(out_path, 'w') as f:
        json.dump(to_jsonable({'stages': all_results, 'n': args.n, 'label': args.label}), f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
