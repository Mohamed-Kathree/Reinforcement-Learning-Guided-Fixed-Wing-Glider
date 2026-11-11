"""
test_curriculum_v21.py
======================
The curriculum changes made after the V20 run's checkpoint sweep, which
showed Stage 0 success peak at 51% (2.0M steps) and fall to 4% (18.6M)
once training moved on, with both forced advances preceding the collapse:

  1. replay      -- a share of episodes runs at an earlier stage, and only
                    current-stage episodes count toward an advance;
  2. no forcing  -- base.yaml disables max_episodes_at_stage;
  3. selection   -- best_model.zip is chosen by deterministic success over
                    all stages, on seeds disjoint from the benchmark's;
  4. thresholds  -- one advance threshold per stage.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.curriculum import STAGES, CurriculumScheduler, episode_stage  # noqa: E402
from env.glider_env import GliderEnv  # noqa: E402

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "training", "configs", "base.yaml")


def _cfg() -> dict:
    with open(CONFIG) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 1. Replay
# ---------------------------------------------------------------------------

def test_episode_stage_without_replay_never_draws():
    rng = np.random.default_rng(7)
    before = rng.bit_generator.state
    assert [episode_stage(2, 0.0, rng) for _ in range(50)] == [2] * 50
    assert rng.bit_generator.state == before


def test_episode_stage_at_stage_zero_is_always_zero():
    rng = np.random.default_rng(7)
    assert {episode_stage(0, 0.5, rng) for _ in range(200)} == {0}


def test_episode_stage_replays_earlier_stages_at_the_configured_rate():
    rng = np.random.default_rng(0)
    draws = np.array([episode_stage(3, 0.25, rng) for _ in range(20_000)])
    assert set(draws) == {0, 1, 2, 3}
    assert abs(np.mean(draws < 3) - 0.25) < 0.015
    replayed = draws[draws < 3]
    for s in (0, 1, 2):                         # uniform over earlier stages
        assert abs(np.mean(replayed == s) - 1 / 3) < 0.03


@pytest.mark.parametrize("frac", [-0.1, 1.0])
def test_episode_stage_rejects_bad_fraction(frac):
    with pytest.raises(ValueError):
        episode_stage(2, frac, np.random.default_rng(0))


def _run(env: GliderEnv, seed: int, n: int = 40):
    obs, _ = env.reset(seed=seed)
    trace = [obs.copy()]
    for _ in range(n):
        obs, _, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
        trace.append(obs.copy())
        if term or trunc:
            break
    return np.array(trace)


def test_curriculum_without_replay_flies_the_identical_episode():
    """set_curriculum(stage, 0.0) must change nothing about the episode."""
    stage = 2
    plain = GliderEnv()
    plain.set_stage(dict(STAGES[stage]))
    cur = GliderEnv()
    cur.set_stage(dict(STAGES[stage]))
    cur.set_curriculum(stage, 0.0)
    assert np.array_equal(_run(plain, 123), _run(cur, 123))


def test_replayed_episode_runs_the_earlier_stage_config():
    env = GliderEnv()
    env.set_stage(dict(STAGES[3]))
    env.set_curriculum(3, 0.9)
    seen = set()
    for seed in range(30):
        env.reset(seed=seed)
        s = env._episode_stage
        seen.add(s)
        for key, value in STAGES[s].items():
            assert env.cfg[key] == value, (s, key)
    assert seen - {3}, "replay_frac 0.9 should have replayed at least one episode"


def test_episode_stage_reported_on_terminal_step():
    env = GliderEnv()
    env.set_stage(dict(STAGES[1]))
    env.set_curriculum(1, 0.5)
    env.reset(seed=5)
    expected = env._episode_stage
    for _ in range(5000):
        _, _, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
        if term or trunc:
            break
    assert info["episode_stage"] == expected


def test_env_outside_the_curriculum_reports_no_stage():
    env = GliderEnv()
    env.set_stage(dict(STAGES[0]))
    env.reset(seed=1)
    assert env._episode_stage is None


class _Logger:
    def __init__(self):
        self.values = {}

    def record(self, key, value):
        self.values[key] = value


def _callback(scheduler):
    from training.train import CurriculumCallback
    venv = SimpleNamespace(env_method=lambda *a, **k: None, set_attr=lambda *a, **k: None)
    cb = CurriculumCallback(scheduler=scheduler, venv=venv, replay_frac=0.25, verbose=0)
    cb.model = SimpleNamespace(logger=_Logger())
    return cb


def _info(stage, success):
    return {"episode": {"r": 0.0, "l": 1}, "success": success, "crash": False,
            "landed": True, "episode_stage": stage}


def test_only_current_stage_episodes_count_toward_an_advance():
    sched = CurriculumScheduler(advance_threshold=[0.5, 0.5, 0.5], rolling_window=4)
    sched._stage = 1
    cb = _callback(sched)
    # A flood of successful replayed Stage 0 episodes must not advance Stage 1.
    cb.locals = {"infos": [_info(0, True) for _ in range(20)]}
    cb._on_step()
    assert sched.current_stage == 1 and sched.episodes_in_window == 0
    assert cb.replay_episodes == 20
    # Current-stage episodes do count.
    cb.locals = {"infos": [_info(1, True) for _ in range(4)]}
    cb._on_step()
    assert sched.current_stage == 2


def test_env_without_episode_stage_is_still_recorded():
    sched = CurriculumScheduler(advance_threshold=0.5, rolling_window=4)
    cb = _callback(sched)
    info = _info(None, False)
    del info["episode_stage"]
    cb.locals = {"infos": [info]}
    cb._on_step()
    assert sched.episodes_in_window == 1


# ---------------------------------------------------------------------------
# 2. Forced advances off
# ---------------------------------------------------------------------------

def test_base_config_disables_forced_advances_and_enables_replay():
    cur = _cfg()["curriculum"]
    assert cur["max_episodes_at_stage"] is None
    assert 0.0 < cur["replay_frac"] < 1.0


# ---------------------------------------------------------------------------
# 4. Per-stage thresholds
# ---------------------------------------------------------------------------

def test_per_stage_thresholds_apply_to_their_stage():
    sched = CurriculumScheduler(advance_threshold=[0.5, 0.25, 0.75], rolling_window=4)
    assert sched.advance_threshold == 0.5
    for ok in (True, True, False, False):       # 50% leaves stage 0
        sched.record_episode(ok)
    assert sched.current_stage == 1 and sched.advance_threshold == 0.25
    for ok in (True, False, False, False):      # 25% leaves stage 1
        sched.record_episode(ok)
    assert sched.current_stage == 2 and sched.advance_threshold == 0.75
    for ok in (True, True, False, False):       # 50% does not leave stage 2
        sched.record_episode(ok)
    assert sched.current_stage == 2
    sched._stage = 3
    assert sched.advance_threshold == 0.75      # final stage: last value


def test_scalar_threshold_still_applies_to_every_stage():
    sched = CurriculumScheduler(advance_threshold=0.6, rolling_window=10)
    assert sched.advance_thresholds == [0.6] * (len(STAGES) - 1)


@pytest.mark.parametrize("bad", [[0.3, 0.3], [0.3, 0.3, 0.3, 0.3], [0.3, 0.0, 0.3], [0.3, 1.2, 0.3]])
def test_bad_threshold_lists_are_rejected(bad):
    with pytest.raises(ValueError):
        CurriculumScheduler(advance_threshold=bad)


def test_base_config_thresholds_sit_above_the_baseline():
    """Each threshold must be above the deterministic baseline's success at
    that stage, or an advance would not mean the policy is ahead of it."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "analysis", "benchmark_baseline_v20.json")) as f:
        base = json.load(f)
    stages = base["stages"]
    thresholds = _cfg()["curriculum"]["advance_threshold"]
    assert len(thresholds) == len(STAGES) - 1
    for k, t in enumerate(thresholds):
        rate = stages[str(k)]["policy"]["buckets"]["success"]["rate"]
        assert t > rate, (k, t, rate)


# ---------------------------------------------------------------------------
# 3. All-stage model selection
# ---------------------------------------------------------------------------

def _tiny_setup(tmp_path):
    from stable_baselines3 import PPO

    from training.train import build_venv
    cfg = _cfg()
    cfg["ppo"]["n_envs"] = 1
    venv = build_venv(cfg, seed=0, force_dummy=True)
    model = PPO("MlpPolicy", venv, n_steps=32, batch_size=32, seed=0, device="cpu")
    venv.reset()
    for _ in range(20):                          # give obs_rms non-trivial statistics
        venv.step(np.array([model.action_space.sample()]))
    return cfg, venv, model


def test_in_memory_actor_matches_the_saved_one(tmp_path):
    from training.benchmark import PolicyActor
    cfg, venv, model = _tiny_setup(tmp_path)
    model.save(tmp_path / "m")
    venv.save(str(tmp_path / "v.pkl"))
    mem = PolicyActor.from_model(model, venv)
    disk = PolicyActor(tmp_path / "m.zip", tmp_path / "v.pkl")
    rng = np.random.default_rng(0)
    mem.reset(0)
    disk.reset(0)
    for _ in range(12):
        o = rng.normal(size=12).astype(np.float32)
        assert np.allclose(mem.act(o), disk.act(o), atol=1e-6)
    # The snapshot is detached: later training updates do not leak into it.
    venv.obs_rms.mean[:] += 100.0
    assert not np.allclose(mem.vecnorm.obs_rms.mean, venv.obs_rms.mean)


def test_multistage_eval_saves_best_model_and_logs(tmp_path):
    from training.train import MultiStageEvalCallback
    from stable_baselines3.common.logger import configure
    cfg, venv, model = _tiny_setup(tmp_path)
    model.set_logger(configure(None, []))       # learn() would set this up
    cb = MultiStageEvalCallback(cfg=cfg, venv=venv, checkpoint_dir=tmp_path,
                                eval_freq=1, n_per_stage=1, stages=[0, 3], seed=1000,
                                verbose=0)
    cb.init_callback(model)
    assert cb.on_step()
    best = json.loads((tmp_path / "best_model.json").read_text())
    assert best["new_best"] and set(best["stages"]) == {"0", "3"}
    assert (tmp_path / "best_model.zip").exists()
    assert (tmp_path / "vecnormalize_best.pkl").exists()
    lines = (tmp_path / "eval_logs" / "multistage.jsonl").read_text().splitlines()
    assert len(lines) == 1

    # Same policy, same seeds -> same score, so no second "best".
    assert cb.on_step()
    lines = (tmp_path / "eval_logs" / "multistage.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["new_best"] is False


def test_selection_seeds_differ_from_the_benchmarks():
    from training.benchmark import episode_seeds
    from training.train import MultiStageEvalCallback
    with pytest.raises(ValueError):
        MultiStageEvalCallback(cfg={}, venv=None, checkpoint_dir=".", eval_freq=1, seed=0)
    sel = set(episode_seeds(10, _cfg()["eval"]["seed"]))
    assert not sel & set(episode_seeds(100, 0))
