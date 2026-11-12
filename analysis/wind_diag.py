"""How does each controller fail in wind? Stage 0 plus Stage 1's wind/gusts.
Per episode: wind vector, where it touched down relative to home along and
across the wind, how much path reward it collected moving downwind vs
upwind, and its crab angle.

Usage:
    python analysis/wind_diag.py <policy|baseline> <model.zip> <vecnormalize.pkl> <episodes> <out.json>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env.curriculum import STAGES
from env.glider_env import GliderEnv
from training.benchmark import BaselineActor, PolicyActor, episode_seeds, load_config, outcome_bucket
from training.train import flat_env_cfg

which, model, vecnorm, n, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
cfg = load_config()
stage_cfg = dict(STAGES[0])
stage_cfg["wind_speed"] = 3.0
stage_cfg["gust_intensity"] = STAGES[1]["gust_intensity"]
actor = PolicyActor(model, vecnorm) if which == "policy" else BaselineActor()
if getattr(actor, "obs_version", 1) != 1:
    sys.exit("wind_diag.py reads observation v1 indices (course, yaw); use "
             "analysis/wind_diagnostics.py for v2 models")

rows = []
for seed in episode_seeds(n, 0):
    env = GliderEnv(cfg={**flat_env_cfg(cfg), "obs_version": 1})
    env.set_stage(stage_cfg)
    actor.reset(seed)
    obs, _ = env.reset(seed=seed)
    w = env._wind._mean_ned[:2].copy()          # air mass velocity, NE (m/s)
    ws = float(np.hypot(*w))
    what = w / ws if ws > 1e-6 else np.zeros(2)
    launch = env._fdm.state[0:2].copy()
    path_down = path_up = 0.0
    crab = []
    prev = env._fdm.state[0:2].copy()
    while True:
        obs, r, term, trunc, info = env.step(actor.act(obs))
        pos = env._fdm.state[0:2].copy()
        d = pos - prev
        prev = pos
        step_len = float(np.hypot(*d))
        if step_len > 0:
            if float(d @ what) >= 0:
                path_down += step_len
            else:
                path_up += step_len
        crab.append(abs((float(obs[3]) - float(obs[6]) + np.pi) % (2 * np.pi) - np.pi))
        if term or trunc:
            break
    td = pos                                      # home is the origin
    rows.append({
        "seed": seed, "wind": ws, "outcome": outcome_bucket(info),
        "miss": float(info.get("dist_home", np.nan)),
        "td_along": float(td @ what),             # + = landed downwind of home
        "td_across": float(abs(td[0] * what[1] - td[1] * what[0])),
        "launch_along": float(launch @ what),
        "path_down": path_down, "path_up": path_up,
        "r_path": float(info["episode_reward_components"]["r_path"]),
        "r_terminal": float(info["episode_reward_components"]["r_terminal"]),
        "crab_deg": float(np.degrees(np.mean(crab))),
    })
    env.close()
json.dump(rows, open(out, "w"))
print("done", which, len(rows))
