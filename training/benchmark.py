"""
benchmark.py
============
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

The single evaluator (SPEC V20 Phase 6). Replaces training/evaluate.py,
whose "timeout" bucket counted every soft landing as a timeout and whose
"paired" RL-vs-baseline comparison re-reset the RL env without the seed,
so the two never flew the same episode.

Every controller (trained PPO policy, random, zero, DeterministicRTL) is
stepped through the same loop on a plain GliderEnv. A trained policy sees
its observations through PolicyActor, which applies exactly the training
pipeline (training/train.py build_venv: VecFrameStack, then VecNormalize)
with n_stack read from the model itself, so old 12-observation models and
new frame-stacked ones both load. tests/test_benchmark.py checks the actor
against SB3's own wrappers, observation by observation.

Paired evaluation: each episode seed is flown by the policy under test and
by the baseline, each on a FRESHLY BUILT env reset with that seed, which
re-seeds every random draw (airframe, wind, sensors, launch state, delays).
A fresh env per episode is deliberate: JSBSim carries floating-point state
across run_ic() (back-to-back resets with one seed differ by ~1e-15 in the
launch state), so reusing an env would make an episode depend on whichever
episodes ran before it. Fresh, each result depends on its seed alone,
bit for bit, at ~0.7 s per episode extra.

Outcome buckets come from env/curriculum.py outcome_bucket(): success /
soft_landing / crash / timeout, the same definition training logs.

Statistics: Wilson 95% intervals on every bucket rate; McNemar's exact test
on paired success (policy vs baseline on identical seeds).

Run:
    # trained policy vs the baseline, all stages, 100 paired episodes each
    python -m training.benchmark --policy trained \\
        --model checkpoints_v20/best_model.zip --vecnorm checkpoints_v20/vecnormalize_best.pkl

    # floor controls (no model needed)
    python -m training.benchmark --policy random
    python -m training.benchmark --policy zero

    # baseline alone
    python -m training.benchmark --policy baseline

    # home-information ablation: obs[0], obs[1], obs[3] zeroed for the policy
    python -m training.benchmark --policy trained --model ... --vecnorm ... --ablate-home

Outputs results/benchmark_<policy>_<utc>.json: provenance (policy and
vecnorm SHA-256, config SHA-256, git SHA, UTC timestamp), per-stage summary
and statistics, and every episode (seed, dr_params, outcome, miss distance,
landing quality, steps, return) for both controllers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np
import yaml

PROJ_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from baseline.deterministic_rtl import DeterministicRTL  # noqa: E402
from env.curriculum import OUTCOME_BUCKETS, STAGES, outcome_bucket  # noqa: E402
from env.glider_env import GliderEnv  # noqa: E402
from env.nav_features import OBS_DIMS  # noqa: E402

OBS_DIM = OBS_DIMS[1]     # observation v1 frame size; v2 sizes come from OBS_DIMS
# Home-direction channels of observation v1, per GliderEnv._build_obs()'s
# index contract: [0] dx_home, [1] dy_home, [3] course_angle.
# tests/test_benchmark.py re-derives these indices from that docstring.
HOME_CHANNELS: tuple[int, ...] = (0, 1, 3)

DEFAULT_CONFIG = PROJ_ROOT / "training" / "configs" / "base.yaml"
RESULTS_DIR = PROJ_ROOT / "results"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def file_provenance(path: str | os.PathLike) -> dict:
    """Same pattern as scratch/reward_audit.py's audit artefacts, plus the
    full SHA-256."""
    p = pathlib.Path(path).resolve()
    data = p.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    return {
        "path": str(p),
        "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
        "sha256": digest,
        "sha256_16": digest[:16],
    }


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=PROJ_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def provenance(config_path: str | os.PathLike, model: str | None = None,
               vecnorm: str | None = None) -> dict:
    return {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "HEAD") or None,
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "config": file_provenance(config_path),
        "policy": file_provenance(model) if model else None,
        "vecnorm": file_provenance(vecnorm) if vecnorm else None,
    }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_config(path: str | os.PathLike = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(cfg: dict, stage: int, obs_version: int | None = None) -> GliderEnv:
    """The env exactly as training builds it (training/train.py make_env:
    flat_env_cfg) at curriculum stage `stage` (set_stage).

    obs_version, when given, overrides the config's: a policy is always
    evaluated in the observation version it was trained on (V20-V22 are v1),
    whatever the config's current default. The version changes only what the
    controller sees, never the episode itself (env/nav_features.py), so
    paired episodes stay identical across versions."""
    from training.train import flat_env_cfg
    env_cfg = flat_env_cfg(cfg)
    if obs_version is not None:
        env_cfg['obs_version'] = int(obs_version)
    env = GliderEnv(cfg=env_cfg)
    env.set_stage(dict(STAGES[stage]))
    return env


# ---------------------------------------------------------------------------
# Controllers. Each has reset(seed) and act(raw_obs) -> action in [-1, 1]^2.
# ---------------------------------------------------------------------------

class BaselineActor:
    name = "baseline"

    def __init__(self) -> None:
        self._ctrl = DeterministicRTL()

    def reset(self, seed: int) -> None:
        self._ctrl.reset()

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self._ctrl.act(obs)


class RandomActor:
    """Uniform random actions, seeded from the episode seed (reproducible)."""
    name = "random"

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed + 1)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, 2).astype(np.float32)


class ZeroActor:
    """Zero action: wings level at the middle of the speed-command range."""
    name = "zero"

    def reset(self, seed: int) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)


class _FrozenObsNorm:
    """A copy of a VecNormalize's observation statistics, applied exactly as
    VecNormalize.normalize_obs does (training's live wrapper keeps updating
    its own statistics; this one never changes)."""

    def __init__(self, vecnorm) -> None:
        import copy
        self.obs_rms = copy.deepcopy(vecnorm.obs_rms)
        self.norm_obs = bool(vecnorm.norm_obs)
        self.clip_obs = float(vecnorm.clip_obs)
        self.epsilon = float(vecnorm.epsilon)
        self.training = False

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self.norm_obs:
            return obs
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                       -self.clip_obs, self.clip_obs).astype(np.float32)


class PolicyActor:
    """A trained SB3 policy fed through the training observation pipeline.

    Matches VecNormalize(VecFrameStack(DummyVecEnv([env]), n_stack)) at
    evaluation (frozen statistics): the stack starts as zeros with the reset
    observation in the last slot, each step shifts one frame out and appends
    the newest, and the stacked vector is normalised with the saved running
    statistics. n_stack is the model's observation size / 12, so a model
    trained before frame stacking (12 observations) runs unstacked.
    """
    name = "trained"

    def __init__(self, model_path: str | os.PathLike, vecnorm_path: str | os.PathLike,
                 device: str = "cpu") -> None:
        import pickle
        from stable_baselines3 import PPO
        self.model = PPO.load(str(model_path), device=device)
        self._set_layout(self.model)
        n_obs = self.n_stack * self.obs_dim
        with open(vecnorm_path, "rb") as f:
            self.vecnorm = pickle.load(f)          # VecNormalize, no venv attached
        if self.vecnorm.obs_rms.mean.shape != (n_obs,):
            raise ValueError(f"vecnorm stats {self.vecnorm.obs_rms.mean.shape} do not match "
                             f"model observation size {n_obs}")
        self.vecnorm.training = False
        self._frames: deque[np.ndarray] = deque(maxlen=self.n_stack)

    @classmethod
    def from_model(cls, model, vecnorm) -> "PolicyActor":
        """Wrap a policy already in memory (training's all-stage evaluation).

        `vecnorm` is any VecNormalize; its observation statistics are copied,
        so training can keep updating its own while this actor evaluates the
        frozen snapshot.
        """
        self = cls.__new__(cls)
        self.model = model
        self._set_layout(model)
        n_obs = self.n_stack * self.obs_dim
        self.vecnorm = _FrozenObsNorm(vecnorm)
        if self.vecnorm.obs_rms.mean.shape != (n_obs,):
            raise ValueError(f"vecnorm stats {self.vecnorm.obs_rms.mean.shape} do not match "
                             f"model observation size {n_obs}")
        self.vecnorm.training = False
        self._frames = deque(maxlen=self.n_stack)
        return self

    def _set_layout(self, model) -> None:
        """Observation version and stack depth from the model. Models saved
        before observation v2 carry no obs_version attribute: they are v1."""
        self.obs_version = int(getattr(model, "obs_version", 1))
        if self.obs_version not in OBS_DIMS:
            raise ValueError(f"model has unknown obs_version {self.obs_version!r}")
        self.obs_dim = OBS_DIMS[self.obs_version]
        n_obs = int(model.observation_space.shape[0])
        if n_obs % self.obs_dim:
            raise ValueError(f"model observation size {n_obs} is not a multiple of the "
                             f"obs_version {self.obs_version} frame size {self.obs_dim}")
        self.n_stack = n_obs // self.obs_dim

    def stacked(self) -> np.ndarray:
        return np.concatenate(list(self._frames)).astype(np.float32)

    def observe(self, obs: np.ndarray, first: bool) -> np.ndarray:
        """Push one raw observation; return the normalised model input."""
        if np.shape(obs) != (self.obs_dim,):
            raise ValueError(
                f"observation version mismatch: the policy was trained on obs_version "
                f"{self.obs_version} ({self.obs_dim} values per frame), the env gave "
                f"{np.shape(obs)}; build the env with obs_version={self.obs_version}")
        if first:
            self._frames.clear()
            for _ in range(self.n_stack - 1):
                self._frames.append(np.zeros(self.obs_dim, dtype=np.float32))
        self._frames.append(np.asarray(obs, dtype=np.float32))
        return self.vecnorm.normalize_obs(self.stacked())

    def reset(self, seed: int) -> None:
        self._first = True

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = self.observe(obs, self._first)
        self._first = False
        action, _ = self.model.predict(x, deterministic=True)
        return np.asarray(action, dtype=np.float32)


class AblateHome:
    """Wrap a controller so it never sees the home-direction channels."""

    def __init__(self, actor) -> None:
        self.actor = actor
        self.name = f"{actor.name}+ablate_home"

    @staticmethod
    def ablate(obs: np.ndarray) -> np.ndarray:
        out = np.array(obs, dtype=np.float32, copy=True)
        out[list(HOME_CHANNELS)] = 0.0
        return out

    def reset(self, seed: int) -> None:
        self.actor.reset(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.actor.act(self.ablate(obs))


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

def run_episode(env: GliderEnv, actor, seed: int) -> dict:
    """One episode from env.reset(seed=seed); the per-episode record."""
    actor.reset(seed)
    obs, _ = env.reset(seed=seed)
    launch_state = env._fdm.state.copy()
    total, steps = 0.0, 0
    while True:
        obs, r, terminated, truncated, info = env.step(actor.act(obs))
        total += float(r)
        steps += 1
        if terminated or truncated:
            break
    extra = {"gate": actor.episode_record()} if hasattr(actor, "episode_record") else {}
    return {
        **extra,
        "seed": int(seed),
        "outcome": outcome_bucket(info),
        "miss_distance_m": float(info.get("dist_home", float("nan"))),
        "quality": float(info.get("quality", 0.0)),
        "steps": steps,
        "return": total,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "launch_state": [float(x) for x in launch_state],
        "dr_params": dict(info["dr_params"]),
    }


def episode_seeds(n: int, seed: int) -> list[int]:
    """Episode seeds per stage; the same stream scratch/final_benchmark.py
    and the post-Phase-5 baseline benchmark used (default_rng(seed))."""
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, 2**31)) for _ in range(n)]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def summarise(episodes: list[dict]) -> dict:
    n = len(episodes)
    buckets = {}
    for b in OUTCOME_BUCKETS:
        k = sum(e["outcome"] == b for e in episodes)
        buckets[b] = {"n": k, "rate": k / n if n else float("nan"), "wilson95": list(wilson(k, n))}
    return {
        "n": n,
        "buckets": buckets,
        "mean_quality": float(np.mean([e["quality"] for e in episodes])),
        "mean_miss_distance_m": float(np.mean([e["miss_distance_m"] for e in episodes])),
        "median_miss_distance_m": float(np.median([e["miss_distance_m"] for e in episodes])),
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
    }


def paired_table(policy: list[dict], baseline: list[dict]) -> list[list[int]]:
    """[[both_ok, policy_only], [baseline_only, both_fail]] on success."""
    assert [e["seed"] for e in policy] == [e["seed"] for e in baseline], "episodes not paired"
    p = [e["outcome"] == "success" for e in policy]
    b = [e["outcome"] == "success" for e in baseline]
    return [[sum(x and y for x, y in zip(p, b)), sum(x and not y for x, y in zip(p, b))],
            [sum(y and not x for x, y in zip(p, b)), sum(not x and not y for x, y in zip(p, b))]]


def mcnemar_exact(table: list[list[int]]) -> float:
    from statsmodels.stats.contingency_tables import mcnemar
    return float(mcnemar(np.array(table), exact=True).pvalue)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def make_actor(policy: str, model: str | None = None, vecnorm: str | None = None):
    if policy == "trained":
        if not (model and vecnorm):
            raise ValueError("--policy trained needs --model and --vecnorm")
        return PolicyActor(model, vecnorm)
    return {"random": RandomActor, "zero": ZeroActor, "baseline": BaselineActor}[policy]()


def actor_obs_version(actor) -> int | None:
    """The observation version a controller needs (None: the config's)."""
    return getattr(getattr(actor, "actor", actor), "obs_version", None)


def evaluate_seed(cfg: dict, stage: int, actor, seed: int,
                  obs_version: int | None = None) -> dict:
    """One episode on a freshly built env: depends on (stage, actor, seed) only."""
    env = make_env(cfg, stage, obs_version if obs_version is not None
                   else actor_obs_version(actor))
    try:
        return run_episode(env, actor, seed)
    finally:
        env.close()


def benchmark(actor, cfg: dict, stages: list[int], n: int, seed: int = 0,
              paired_baseline: bool = True) -> dict:
    out = {}
    version = actor_obs_version(actor)
    for stage in stages:
        seeds = episode_seeds(n, seed)
        pol = [evaluate_seed(cfg, stage, actor, s, version) for s in seeds]
        rec: dict[str, Any] = {"stage": stage, "policy": summarise(pol), "episodes": {"policy": pol}}
        if paired_baseline and not isinstance(actor, BaselineActor):
            base_actor = BaselineActor()
            # The baseline always reads observation v1, the exact sensor
            # vector: v2 frames are float32 and scaled, and that rounding is
            # enough to move a JSBSim touchdown by a step. The episode itself
            # is the same under either version, so pairing is unaffected and
            # the baseline's numbers are identical for every policy.
            base = [evaluate_seed(cfg, stage, base_actor, s, 1) for s in seeds]
            table = paired_table(pol, base)
            rec["baseline"] = summarise(base)
            rec["episodes"]["baseline"] = base
            rec["paired"] = {"table": table, "mcnemar_exact_p": mcnemar_exact(table),
                             "table_layout": "[[both_ok, policy_only], [baseline_only, both_fail]]"}
        out[str(stage)] = rec
    return out


def _fmt(summary: dict) -> str:
    return "  ".join(
        f"{b} {v['rate'] * 100:5.1f}% [{v['wilson95'][0] * 100:4.1f}, {v['wilson95'][1] * 100:4.1f}]"
        for b, v in summary["buckets"].items()) + f"  quality {summary['mean_quality']:.2f}"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Single evaluator: policy vs paired baseline")
    ap.add_argument("--policy", choices=("trained", "random", "zero", "baseline"), required=True)
    ap.add_argument("--model", help="trained policy .zip")
    ap.add_argument("--vecnorm", help="VecNormalize .pkl saved with the model")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--stages", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate-home", action="store_true",
                    help=f"zero observation channels {HOME_CHANNELS} for the policy under test")
    ap.add_argument("--no-baseline", action="store_true", help="skip the paired baseline run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    actor = make_actor(args.policy, args.model, args.vecnorm)
    if args.ablate_home:
        actor = AblateHome(actor)

    stages = benchmark(actor, cfg, args.stages, args.episodes, args.seed,
                       paired_baseline=not args.no_baseline)
    record = {
        "description": "SPEC V20 single evaluator (training/benchmark.py)",
        "provenance": provenance(args.config, args.model, args.vecnorm),
        "args": {k: v for k, v in vars(args).items()},
        "policy_name": actor.name,
        "n_stack": getattr(getattr(actor, "actor", actor), "n_stack", None),
        "obs_version": actor_obs_version(actor),
        "stages": stages,
    }
    for s, rec in stages.items():
        print(f"stage {s} {actor.name:>10}: {_fmt(rec['policy'])}")
        if "baseline" in rec:
            print(f"stage {s} {'baseline':>10}: {_fmt(rec['baseline'])}")
            print(f"         paired success {rec['paired']['table']}  "
                  f"McNemar exact p = {rec['paired']['mcnemar_exact_p']:.4f}")

    out = args.out or str(RESULTS_DIR / (
        f"benchmark_{actor.name.replace('+', '_')}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"))
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    print(f"wrote {os.path.relpath(out, PROJ_ROOT)}")
    return record


if __name__ == "__main__":
    main()
