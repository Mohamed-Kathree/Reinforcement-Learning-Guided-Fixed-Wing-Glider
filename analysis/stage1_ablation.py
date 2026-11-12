"""Which Stage 1 change hurts the policy? Stage 0 with one group of Stage 1
settings switched on at a time; policy and baseline on the same seeds.

Usage:
    python analysis/stage1_ablation.py <group> <model.zip> <vecnormalize.pkl> <episodes> <out.json>
    group: stage0 | wind | meanwind | gusts | sensors | airframe | geometry | grading | stage1
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.curriculum import STAGES
from training.benchmark import (BaselineActor, PolicyActor, episode_seeds, load_config,
                                mcnemar_exact, paired_table, run_episode)
from training.train import flat_env_cfg
from env.glider_env import GliderEnv

GROUPS = {
    "stage0": [],
    "wind": ["wind_speed", "gust_intensity"],
    "meanwind": ["wind_speed"],
    "gusts": ["gust_intensity"],
    "sensors": ["sensor_noise", "dropout_prob"],
    "airframe": ["dr_scale"],
    "geometry": ["R_home_m", "alt0_m", "launch_offset_min_m", "launch_offset_max_m", "sigma_centre_m"],
    "grading": ["sink_bad", "roll_bad_deg"],
    "stage1": list(STAGES[1].keys()),
}

name, model, vecnorm, n, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
cfg = load_config()
stage_cfg = dict(STAGES[0])
for k in GROUPS[name]:
    stage_cfg[k] = STAGES[1][k]

res = {}
for label, actor in (("policy", PolicyActor(model, vecnorm)), ("baseline", BaselineActor())):
    eps = []
    for s in episode_seeds(n, 0):
        env = GliderEnv(cfg=flat_env_cfg(cfg))
        env.set_stage(stage_cfg)
        try:
            eps.append(run_episode(env, actor, s))
        finally:
            env.close()
    res[label] = eps
t = paired_table(res["policy"], res["baseline"])


def rate(eps, b):
    return sum(e["outcome"] == b for e in eps) / len(eps)


summary = {"group": name, "n": n,
           **{f"{l}_{b}": rate(res[l], b) for l in ("policy", "baseline") for b in ("success", "crash")},
           "policy_median_miss": sorted(e["miss_distance_m"] for e in res["policy"])[n // 2],
           "baseline_median_miss": sorted(e["miss_distance_m"] for e in res["baseline"])[n // 2],
           "policy_quality": sum(e["quality"] for e in res["policy"]) / n,
           "p": mcnemar_exact(t)}
json.dump(summary, open(out, "w"))
print(json.dumps(summary))
