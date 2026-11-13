"""Does airframe randomisation make the wind problem worse? V22 best and the
baseline in calm air and steady wind (0-3 m/s) at airframe randomisation
scale 0 (nominal airframe only), 0.25 (Stage 0) and 0.5 (Stage 1); 100
paired seeds per cell. Writes results/dr_wind_check_v22.json."""
import json
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_A = {}


def _one(task):
    from env.curriculum import STAGES
    from env.glider_env import GliderEnv
    from training.benchmark import BaselineActor, PolicyActor, load_config, run_episode
    from training.train import flat_env_cfg
    who, wind, scale, seed = task
    if who not in _A:
        _A[who] = BaselineActor() if who == "baseline" else PolicyActor(
            os.path.join(ROOT, "checkpoints_v22/best_model.zip"),
            os.path.join(ROOT, "checkpoints_v22/vecnormalize_best.pkl"))
    env = GliderEnv(cfg={**flat_env_cfg(load_config()), "obs_version": 1})
    st = dict(STAGES[0])
    st["wind_speed"], st["dr_scale"] = wind, scale
    env.set_stage(st)
    try:
        r = run_episode(env, _A[who], seed)
    finally:
        env.close()
    return who, wind, scale, seed, r["outcome"], r["miss_distance_m"]


if __name__ == "__main__":
    from training.benchmark import episode_seeds, mcnemar_exact, paired_table, wilson
    seeds = episode_seeds(100, 0)
    tasks = [(w, wi, sc, s) for w in ("policy", "baseline") for wi in (0.0, 3.0)
             for sc in (0.0, 0.25, 0.5) for s in seeds]
    with mp.get_context("spawn").Pool(6) as pool:
        res = pool.map(_one, tasks, chunksize=4)
    out = {}
    for wi in (0.0, 3.0):
        for sc in (0.0, 0.25, 0.5):
            cell = {}
            eps = {}
            for w in ("policy", "baseline"):
                e = [{"seed": r[3], "outcome": r[4]} for r in res if r[0] == w and r[1] == wi and r[2] == sc]
                k = sum(x["outcome"] == "success" for x in e)
                cell[w] = {"success": k / len(e), "wilson95": list(wilson(k, len(e)))}
                eps[w] = e
            t = paired_table(eps["policy"], eps["baseline"])
            cell["policy_minus_baseline"] = cell["policy"]["success"] - cell["baseline"]["success"]
            cell["mcnemar_p"] = mcnemar_exact(t)
            out[f"wind{wi:g}_dr{sc:g}"] = cell
            print(f"wind {wi:g} m/s, airframe DR {sc:<4g}: policy {cell['policy']['success']:.2f}  "
                  f"baseline {cell['baseline']['success']:.2f}  gap {cell['policy_minus_baseline']:+.2f} "
                  f"(p {cell['mcnemar_p']:.3f})", flush=True)
    with open(os.path.join(ROOT, "results", "dr_wind_check_v22.json"), "w") as f:
        json.dump(out, f, indent=1)
