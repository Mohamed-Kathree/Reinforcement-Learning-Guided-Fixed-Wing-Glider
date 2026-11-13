"""
build_archive.py
================
SPEC V24 Phase 0: freeze V23 runs 1 and 2 (negative results) and hash-record
the V22 artefacts that the restored training base is checked against.

No training run archived its resolved config (data/runs/*.meta.json holds only
the git hash, seed, step budget and env count), so each resolved config is
rebuilt here from the sources that do survive:

    base.yaml at the commit recorded in the run's meta.json
    + the exact --set overrides the run was launched with
    -> training.train.apply_overrides (the same function train.py uses)

Usage:  python analysis/negative_results/build_archive.py
Writes: analysis/negative_results/{run1,run2,v22}_resolved_config.yaml,
        copies of the benchmark JSONs, and manifest.json (sha256 of everything).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import yaml  # noqa: E402

from training.train import apply_overrides  # noqa: E402

RUNS = {
    "v22": {
        "meta": "data/runs/20261112T013358Z_seed0.meta.json",
        "overrides": [],          # V22 was launched with no --set overrides
        "command": "python -m training.train",
        "checkpoint_dir": "checkpoints_v22",
        "benchmarks": ["results/benchmark_v22_best_15M.json",
                       "results/benchmark_v22_final_20M.json",
                       "results/benchmark_v22_14M.json"],
    },
    "run1": {
        "meta": "data/runs/20261112T194008Z_seed0.meta.json",
        "overrides": ["observation.obs_version=3", "reward.path_reward_frame=air",
                      "reward.cone_shaping=true", "ppo.gae_lambda=0.995",
                      "curriculum.stage0_wind_ms=1.5",
                      "logging.checkpoint_dir=checkpoints_v23_run1/",
                      "logging.tensorboard_log=runs_v23_run1/"],
        "checkpoint_dir": "checkpoints_v23_run1",
        "benchmarks": ["results/benchmark_v23run1_best_6.75M.json",
                       "results/benchmark_v23run1_final_20M.json"],
    },
    "run2": {
        "meta": "data/runs/20261113T083932Z_seed0.meta.json",
        "overrides": ["observation.obs_version=3", "reward.path_reward_frame=air",
                      "ppo.gae_lambda=0.995", "curriculum.stage0_wind_ms=1.5",
                      "logging.checkpoint_dir=checkpoints_v23_run2/",
                      "logging.tensorboard_log=runs_v23_run2/"],
        "checkpoint_dir": "checkpoints_v23_run2",
        "benchmarks": ["results/benchmark_v23run2_best_10M.json",
                       "results/benchmark_v23run2_final_20M.json"],
    },
}
for k in ("run1", "run2"):
    RUNS[k]["command"] = "python -m training.train --set " + " ".join(RUNS[k]["overrides"])

ARTEFACTS = ("best_model.zip", "vecnormalize_best.pkl", "ppo_glider_final.zip",
             "vecnormalize_final.pkl", "best_model.json")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_show(sha: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{sha}:{path}"], cwd=ROOT,
                          check=True, capture_output=True).stdout


def build() -> dict:
    manifest = {"note": __doc__.strip().splitlines()[3], "runs": {}}
    for name, run in RUNS.items():
        with open(os.path.join(ROOT, run["meta"])) as f:
            meta = json.load(f)
        base_bytes = git_show(meta["git_hash"], "training/configs/base.yaml")
        cfg = apply_overrides(yaml.safe_load(base_bytes), list(run["overrides"]))
        cfg_path = os.path.join(HERE, f"{name}_resolved_config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(f"# Resolved config of {name}, rebuilt by build_archive.py from\n"
                    f"# training/configs/base.yaml at {meta['git_hash']} plus:\n"
                    f"#   {run['command']}\n")
            yaml.safe_dump(cfg, f, sort_keys=False)
        rec = {"meta": meta, "command": run["command"], "overrides": run["overrides"],
               "base_yaml_sha256_at_commit": hashlib.sha256(base_bytes).hexdigest(),
               "resolved_config": os.path.basename(cfg_path),
               "resolved_config_sha256": sha256(cfg_path),
               "checkpoints": {}, "benchmarks": {}}
        for a in ARTEFACTS:
            p = os.path.join(ROOT, run["checkpoint_dir"], a)
            if os.path.exists(p):
                rec["checkpoints"][f"{run['checkpoint_dir']}/{a}"] = sha256(p)
        for b in run["benchmarks"]:
            src = os.path.join(ROOT, b)
            dst = os.path.join(HERE, os.path.basename(b))
            shutil.copyfile(src, dst)
            with open(src) as f:
                prov = json.load(f)["provenance"]
            for kind in ("policy", "vecnorm"):     # every file a benchmark scored
                rel = os.path.relpath(prov[kind]["path"], ROOT).replace("\\", "/")
                if os.path.exists(os.path.join(ROOT, rel)):
                    rec["checkpoints"].setdefault(rel, sha256(os.path.join(ROOT, rel)))
            rec["benchmarks"][os.path.basename(b)] = {
                "sha256": sha256(dst), "git_sha": prov["git_sha"],
                "policy_sha256": prov["policy"]["sha256"],
                "vecnorm_sha256": prov["vecnorm"]["sha256"]}
        manifest["runs"][name] = rec
    with open(os.path.join(HERE, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def verify_v22(manifest: dict) -> list[str]:
    """V22's checkpoint and VecNormalize hashes must equal the ones its
    committed benchmark recorded; its base.yaml must equal the benchmark's."""
    problems = []
    v22 = manifest["runs"]["v22"]
    for bname, b in v22["benchmarks"].items():
        with open(os.path.join(HERE, bname)) as f:
            prov = json.load(f)["provenance"]
        for kind in ("policy", "vecnorm"):
            rel = os.path.relpath(prov[kind]["path"], ROOT).replace("\\", "/")
            have = v22["checkpoints"].get(rel)
            if have != prov[kind]["sha256"]:
                problems.append(f"{bname}: {rel} hash {have} != recorded {prov[kind]['sha256']}")
        if bname == "benchmark_v22_best_15M.json" and \
                prov["config"]["sha256"] != v22["base_yaml_sha256_at_commit"]:
            problems.append("base.yaml at V22's commit differs from the benchmark's config hash")
    return problems


if __name__ == "__main__":
    m = build()
    for name, rec in m["runs"].items():
        print(f"{name}: commit {rec['meta']['git_hash'][:7]}, config "
              f"{rec['resolved_config_sha256'][:16]}, {len(rec['checkpoints'])} checkpoints, "
              f"{len(rec['benchmarks'])} benchmarks")
    probs = verify_v22(m)
    print("V22 artefacts verified" if not probs else "V22 PROBLEMS:\n  " + "\n  ".join(probs))
