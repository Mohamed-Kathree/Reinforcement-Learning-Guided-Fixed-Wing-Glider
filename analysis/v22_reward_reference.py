"""
v22_reward_reference.py
=======================
SPEC V24 Phase 1: per-step rewards on fixed, open-loop scripted trajectories,
so the reward of the current tree under training/configs/v22_restored.yaml
can be compared with the reward of the tree V22 trained at (ed65989).

The script only uses APIs that exist at ed65989 (GliderEnv(cfg), set_stage,
reset(seed), step; training.train.flat_env_cfg; env.curriculum.STAGES), so the
same file runs in both trees:

    git worktree add <tmp>/v22tree ed65989
    cd <tmp>/v22tree && python <this file> --config training/configs/base.yaml \
        --out <repo>/tests/data/v22_reward_reference.json
    (tests/test_v22_restore.py then replays the trajectories in this tree)

Open-loop actions (a fixed function of the step index) make the trajectory
independent of any controller code, so a reward difference can only come from
the env, the dynamics or the reward.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402

CASES = [  # (stage, seed): calm, steady wind, gusty wind with full DR
    (0, 11), (0, 12), (1, 21), (2, 31), (3, 41),
]


def scripted_action(k: int) -> np.ndarray:
    return np.array([0.6 * math.sin(k / 23.0) + 0.2 * math.sin(k / 7.0),
                     0.5 * math.cos(k / 31.0)], dtype=np.float32)


def run_case(cfg: dict, stage: int, seed: int) -> dict:
    from env.curriculum import STAGES
    from env.glider_env import GliderEnv
    from training.train import flat_env_cfg

    env = GliderEnv(cfg=flat_env_cfg(cfg))
    env.set_stage(STAGES[stage])
    obs, _ = env.reset(seed=seed)
    rewards, obs_trace = [], [np.asarray(obs, dtype=float).tolist()]
    k = 0
    while True:
        obs, r, term, trunc, info = env.step(scripted_action(k))
        rewards.append(float(r))
        obs_trace.append(np.asarray(obs, dtype=float).tolist())
        k += 1
        if term or trunc:
            break
    env.close()
    return {"stage": stage, "seed": seed, "rewards": rewards, "obs": obs_trace,
            "outcome": str(info.get("outcome", "")), "terminated": bool(term),
            "truncated": bool(trunc)}


def run_all(cfg: dict) -> list[dict]:
    return [run_case(cfg, s, sd) for s, sd in CASES]


def main(argv=None):
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"git_sha": sha, "config": a.config, "cases": run_all(cfg)}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    for c in out["cases"]:
        print(f"stage {c['stage']} seed {c['seed']}: {len(c['rewards'])} steps, "
              f"return {sum(c['rewards']):.6f}, {c['outcome']}")


if __name__ == "__main__":
    main()
