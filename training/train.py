"""
train.py
========
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

PPO training entry point using Stable-Baselines3.

Architecture:
    SubprocVecEnv  -- n_envs parallel GliderEnv workers (separate processes)
    VecNormalize   -- running obs/reward normalisation (mandatory for 12-D obs)
    PPO            -- MlpPolicy, hyperparams from training/configs/base.yaml
    CurriculumCallback -- advances the stage when rolling success at the
                          current stage >= that stage's threshold; earlier
                          stages are replayed at curriculum.replay_frac
    CheckpointCallback -- periodic model + VecNormalize snapshots
    MultiStageEvalCallback -- periodic deterministic evaluation at every
                          stage; saves best model by mean success

Usage:
    # Train from scratch with default config:
    python -m training.train

    # Override config values on the command line:
    python -m training.train --config training/configs/base.yaml \\
                              --set ppo.n_envs=4 ppo.total_timesteps=5000000

    # Resume from a checkpoint:
    python -m training.train --resume checkpoints/ppo_glider_1000000_steps

Outputs (all under logging.checkpoint_dir from the config):
    ppo_glider_<step>_steps.zip   -- periodic policy snapshots
    vecnormalize_<step>.pkl       -- matching VecNormalize stats
    best_model.zip                -- best evaluation policy so far
    vecnormalize_best.pkl         -- VecNormalize stats for best model
    best_model.json               -- the evaluation that selected best_model
    eval_logs/multistage.jsonl    -- every all-stage evaluation

TensorBoard:
    tensorboard --logdir runs/

Coordinate frames used:
    NED : North-East-Down inertial frame (world)
    BODY: Forward-Right-Down body frame (FRD, attached to glider)
    WIND: Stability/wind frame (x into relative wind)

Quaternion convention: [q0, q1, q2, q3] where q0 is the scalar component.
Rotation quat_to_rotmat(q) maps BODY -> NED: v_ned = R @ v_body

Units: SI throughout (m, m/s, rad, rad/s, kg, N, N*m)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import time
from typing import Any

import numpy as np
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecEnvWrapper, VecFrameStack, VecNormalize,
)
from stable_baselines3.common.monitor import Monitor

from env.glider_env import GliderEnv
from env.curriculum import CurriculumScheduler, STAGES, outcome_bucket

# best_model.zip is selected by MultiStageEvalCallback: deterministic
# success across EVERY curriculum stage, on a fixed seed set disjoint from
# the benchmark's. It replaced SB3's EvalCallback pinned at the final stage,
# which picked the V20 run's 9.0M checkpoint by Stage 3 reward while the
# 2.0M checkpoint was the strongest policy the run produced (51% vs the
# baseline's 27% at Stage 0; results/benchmark_v20_2.0M.json).

# Optional dashboard hook: streams live metrics to data/train_stream.jsonl
# for the showcase frontend's Training tab (backend/routers/training.py's
# WebSocket tails it). Additive only -- training must run identically with
# or without the dashboard backend present, so this never raises.
try:
    from backend.callbacks.web_stream_callback import WebStreamCallback
except ImportError:
    WebStreamCallback = None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str | pathlib.Path) -> dict:
    """Load and return the YAML config as a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply 'section.key=value' overrides to a nested config dict.

    Example:
        apply_overrides(cfg, ['ppo.n_envs=4', 'reward.w_land=600'])
    """
    for item in overrides:
        key_path, _, raw_value = item.partition('=')
        if not _:
            raise ValueError(f"Override must be 'key=value', got: {item!r}")
        parts = key_path.strip().split('.')
        node = cfg
        for part in parts[:-1]:
            node = node[part]
        leaf = parts[-1]
        # Attempt numeric coercion, then booleans/null (YAML spellings: a bare
        # "false" kept as a string would be truthy); fall back to string
        try:
            value: Any = int(raw_value)
        except ValueError:
            try:
                value = float(raw_value)
            except ValueError:
                lowered = raw_value.strip().lower()
                if lowered in ('true', 'false'):
                    value = lowered == 'true'
                elif lowered in ('null', 'none', '~'):
                    value = None
                else:
                    value = raw_value
        node[leaf] = value
    return cfg


def archive_resolved_config(cfg: dict, checkpoint_dir: str | pathlib.Path,
                            resume: str | None = None, seed: int = 0) -> pathlib.Path:
    """Write the fully resolved config (file + --set overrides) next to the
    checkpoints, with the command line and git commit (SPEC V24 Phase 0: no
    run before V24 archived one, so their configs had to be rebuilt from git).
    One file per launch, so a --resume does not overwrite the original."""
    import datetime
    import subprocess
    import sys as _sys
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain'], capture_output=True,
                                    text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = 'unknown', None
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = pathlib.Path(checkpoint_dir) / f'resolved_config_{stamp}.yaml'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"# argv: {' '.join(_sys.argv)}\n# git: {sha} dirty={dirty}\n"
                f"# seed: {seed}  resume: {resume}\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


_LAUNCH_KEY_MAP = {
    'speed_min_ms':  'launch_speed_min_ms',
    'speed_max_ms':  'launch_speed_max_ms',
    'pitch_min_deg': 'launch_pitch_min_deg',
    'pitch_max_deg': 'launch_pitch_max_deg',
}


def flat_env_cfg(cfg: dict) -> dict:
    """Flatten the reward + launch config sections into one dict for GliderEnv.

    Only launch.speed_{min,max}_ms / launch.pitch_{min,max}_deg are pulled in
    here (renamed to the launch_speed_min_ms/etc. keys GliderEnv reads) --
    they represent a fixed hardware uncertainty (the not-yet-built ESP32
    apex-detection firmware's unknown hand-off energy state), not something
    curriculum-stage-dependent. launch.alt0_m is deliberately NOT included:
    that IS curriculum-stage-dependent and is applied via set_stage() (see
    train(), which applies STAGES[0] immediately after building the
    environments, then CurriculumCallback applies later stages as training
    progresses).
    """
    flat = dict(cfg['reward'])
    # Observation version (SPEC V23 Phase O); absent in pre-V23 configs: v1.
    flat['obs_version'] = int(cfg.get('observation', {}).get('obs_version', 1))
    launch = cfg.get('launch', {})
    for yaml_key, cfg_key in _LAUNCH_KEY_MAP.items():
        if yaml_key in launch:
            flat[cfg_key] = launch[yaml_key]
    return flat


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _extract_step_count(checkpoint_path: str) -> int | None:
    """Parse the timestep count out of a checkpoint path such as
    'checkpoints/ppo_glider_1000000_steps' or '...ppo_glider_1000000_steps.zip'.

    Returns None if the '_<digits>_steps' suffix isn't found, e.g. for
    'ppo_glider_final'  (there is no matching curriculum_final.json).
    """
    stem = pathlib.Path(checkpoint_path).stem
    m = re.search(r'_(\d+)_steps$', stem)
    return int(m.group(1)) if m else None


def _vecnorm_path_for_checkpoint(checkpoint_path: str) -> pathlib.Path:
    """Map a checkpoint path to its matching VecNormalize stats file.

    CheckpointCallback names periodic snapshots 'ppo_glider_<N>_steps.zip'
    while VecNormCheckpointCallback names the matching stats
    'vecnormalize_<N>.pkl' -- no '_steps' suffix -- so a naive
    ppo_glider->vecnormalize substring swap keeps the stray '_steps' and
    never matches. Found while verifying Phase B's resume behaviour: every
    --resume from a periodic checkpoint (the shape this runbook itself
    hands off, e.g. after an interruption) silently reset the running
    normalisation stats instead of restoring them.

        ppo_glider_<N>_steps[.zip]  -> vecnormalize_<N>.pkl
        ppo_glider_final[.zip]      -> vecnormalize_final.pkl
        best_model[.zip]            -> vecnormalize_best.pkl
    """
    p = pathlib.Path(checkpoint_path)
    stem = p.stem if p.suffix == '.zip' else p.name
    step_num = _extract_step_count(checkpoint_path)
    if step_num is not None:
        name = f"vecnormalize_{step_num}.pkl"
    elif stem == 'best_model':
        name = "vecnormalize_best.pkl"
    else:
        name = stem.replace('ppo_glider', 'vecnormalize') + '.pkl'
    return p.parent / name


def make_env(cfg: dict, rank: int, seed: int = 0):
    """Return a callable that creates a single monitored GliderEnv.

    Each worker gets a unique seed derived from the base seed + rank,
    guaranteeing diverse episode starts across parallel environments.
    """
    def _init() -> Monitor:
        env = GliderEnv(cfg=flat_env_cfg(cfg))
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


class VecScaleReward(VecEnvWrapper):
    """Multiplies every reward by a fixed constant, and does nothing else.

    This is the whole of the reward transform between env/reward.py and PPO
    (ppo.reward_scale, with ppo.normalize_reward false). It replaced
    VecNormalize's reward normalisation, which divides by a running return
    standard deviation and then CLIPS at +/-10. In the V20/V21 runs that
    standard deviation was 13-26, so every raw reward beyond +/-128..256 was
    clipped, against a landing reward of up to +650 and a crash at -200:
    every clean landing closer than 14 m (start of training) to 38 m (end)
    paid the same. From 1.6M steps on that flat disc was wider than the
    success radius, so landing on target earned nothing over landing 30 m
    out (docs/SPEC_V20_decisions.md, "Training defects").

    It sits inside VecFrameStack, so Monitor (inside each worker) still
    reports raw episode returns and VecNormalize stays outermost.
    """

    def __init__(self, venv, scale: float) -> None:
        if not scale > 0.0:
            raise ValueError(f"reward scale must be positive; got {scale}")
        super().__init__(venv)
        self.scale = float(scale)

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs, rewards * self.scale, dones, infos


def learner_reward(venv, raw, return_std: float | None = None) -> np.ndarray:
    """What PPO receives for raw env rewards `raw`, through venv's wrappers.

    Applies, innermost first, every reward transform in the wrapper chain:
    VecScaleReward's constant, and VecNormalize's divide-and-clip when its
    norm_reward is on (using `return_std` in place of its running estimate
    when given). Pure: no statistics are updated. Used by the tests that pin
    the landing reward's gradient as seen by the learner.
    """
    chain = []
    v = venv
    while v is not None:
        chain.append(v)
        v = getattr(v, 'venv', None)
    out = np.asarray(raw, dtype=np.float64)
    for w in reversed(chain):
        if isinstance(w, VecScaleReward):
            out = out * w.scale
        elif isinstance(w, VecNormalize) and w.norm_reward:
            std = (float(return_std) if return_std is not None
                   else float(np.sqrt(w.ret_rms.var + w.epsilon)))
            out = np.clip(out / std, -w.clip_reward, w.clip_reward)
    return out


def build_venv(cfg: dict, seed: int = 0, force_dummy: bool = False) -> VecNormalize:
    """Create a frame-stacked, VecNormalize-wrapped vectorised environment.

    Frame stacking (cfg['ppo']['n_stack'] frames, SPEC V20 Phase 4): with
    domain randomisation this wide, a memoryless policy on a single 12-element
    observation cannot tell which airframe it is flying. Eight frames at
    20 Hz is 0.4 s of history, enough to infer control effectiveness from
    the roll-response lag. Training AND eval envs are both built here, so
    their observation shapes cannot diverge.

    Wrapper order is VecNormalize(VecFrameStack(base)): VecNormalize must
    stay outermost so `venv.training = False` / `venv.norm_reward = False`
    and VecNormalize.save()/load() act on it directly (attributes set on an
    outer VecFrameStack would not reach it).

    Uses SubprocVecEnv for multi-process parallelism (default) or
    DummyVecEnv for single-process debugging (force_dummy=True or n_envs=1).

    Args:
        cfg         : full config dict
        seed        : base RNG seed
        force_dummy : use DummyVecEnv even if n_envs > 1 (useful for debugging)

    Returns:
        VecNormalize wrapping the vectorised env
    """
    n_envs = int(cfg['ppo']['n_envs'])
    fns    = [make_env(cfg, rank=i, seed=seed) for i in range(n_envs)]

    if force_dummy or n_envs == 1:
        venv = DummyVecEnv(fns)
    else:
        venv = SubprocVecEnv(fns, start_method='spawn')

    reward_scale = float(cfg['ppo'].get('reward_scale', 1.0))
    if reward_scale != 1.0:
        venv = VecScaleReward(venv, reward_scale)
    venv = VecFrameStack(venv, n_stack=int(cfg['ppo']['n_stack']))
    venv = VecNormalize(
        venv,
        norm_obs    = bool(cfg['ppo']['normalize_obs']),
        norm_reward = bool(cfg['ppo']['normalize_reward']),
        clip_obs    = float(cfg['ppo']['clip_obs']),
        gamma       = float(cfg['ppo']['gamma']),
    )
    return venv


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """Advances the curriculum stage when the rolling success rate crosses
    the threshold, then propagates the new stage config to all VecEnv workers.

    Episode outcomes are read from the Monitor 'is_success' key in the
    episode info dict.  If the key is absent (env didn't set it), the
    callback falls back to checking whether episode reward > success_threshold.

    Args:
        scheduler          : CurriculumScheduler instance (shared state)
        venv               : the VecNormalize-wrapped env (workers are set_stage'd)
        success_reward_min : fallback: episode counts as success if ep_reward >= this
        replay_frac        : share of episodes replayed at an earlier stage
                             (passed to every worker's set_curriculum())
        verbose            : 1 to log stage advances

    Only episodes that ran at the scheduler's current stage
    (info['episode_stage']) are recorded toward an advance. Replayed
    earlier-stage episodes would otherwise inflate the success rate, and
    episodes still in flight from the previous stage when an advance lands
    are no longer credited to the new one. An env that reports no
    episode_stage (not under set_curriculum) is always recorded.
    """

    def __init__(
        self,
        scheduler:           CurriculumScheduler,
        venv:                VecNormalize,
        success_reward_min:  float = 400.0,
        replay_frac:         float = 0.0,
        verbose:             int   = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.scheduler          = scheduler
        self.venv               = venv
        self.success_reward_min = success_reward_min
        self.replay_frac        = float(replay_frac)
        self.replay_episodes    = 0

    @staticmethod
    def _outcome_bucket(info: dict) -> str:
        """4-way outcome bucket; see env/curriculum.py outcome_bucket()."""
        return outcome_bucket(info)

    def _on_step(self) -> bool:
        # SB3 populates self.locals['infos'] with a list of info dicts,
        # one per env.  'episode' key appears when the episode ends
        # (injected by Monitor wrapper).
        for info in self.locals.get('infos', []):
            ep = info.get('episode')
            if ep is None:
                continue

            # Determine success: prefer explicit flag, fall back to reward
            if 'success' in info:
                success = bool(info['success'])
            else:
                success = float(ep['r']) >= self.success_reward_min

            outcome_bucket = self._outcome_bucket(info)

            ep_stage = info.get('episode_stage')
            if ep_stage is not None and ep_stage != self.scheduler.current_stage:
                self.replay_episodes += 1
                self.logger.record('curriculum/replay_episodes', self.replay_episodes)
                self.logger.record(f'replay/success_stage{ep_stage}', float(success))
                continue

            # Task/curriculum/reward-budget scalars -- the only way to see
            # curriculum stage, success rate, touchdown distance, and the
            # reward-component split live in TensorBoard over a multi-day
            # run instead of only as stdout prints that scroll away.
            self.logger.record('curriculum/stage',             self.scheduler.current_stage)
            self.logger.record('curriculum/success_rate',      self.scheduler.success_rate)
            self.logger.record('curriculum/episodes_at_stage', self.scheduler.episodes_at_stage)
            self.logger.record('curriculum/advance_threshold', self.scheduler.advance_threshold)
            self.logger.record('task/dist_home_final', info.get('dist_home', float('nan')))
            self.logger.record('task/quality',          info.get('quality', 0.0))

            # Reward-component breakdown -- GliderEnv now accumulates all 7
            # components itself (env/glider_env.py's _REWARD_COMPONENT_KEYS)
            # and hands back the completed totals in info on the terminal
            # step, so this callback just reads them rather than maintaining
            # its own (previously incomplete, 3-of-7-term) per-env sums.
            for name, value in info.get('episode_reward_components', {}).items():
                self.logger.record(f'reward/{name}', value)

            advanced = self.scheduler.record_episode(success=success, outcome=outcome_bucket)

            # Outcome-count breakdown, logged AFTER record_episode() so it
            # reflects the buffer including this episode.
            for bucket, count in self.scheduler.outcome_counts.items():
                self.logger.record(f'curriculum/outcome_{bucket}', count)

            if advanced:
                stage     = self.scheduler.current_stage
                stage_cfg = self.scheduler.current_cfg
                if self.verbose >= 1:
                    if self.scheduler.last_advance_was_forced:
                        # A forced advance means the policy was promoted
                        # while still failing the success-rate threshold --
                        # this is information the user needs while reading
                        # TensorBoard, not a detail to bury alongside earned
                        # advances (RLGlider_Phase5_Reward_Rescaling_Spec.md
                        # §3.1).
                        print(f"\n[Curriculum] WARNING: FORCED advance to Stage {stage} "
                              f"(success_rate={self.scheduler.success_rate:.1%} did not "
                              f"reach threshold after max_episodes_at_stage episodes; "
                              f"{self.scheduler.episodes_seen} episodes seen)")
                    else:
                        print(f"\n[Curriculum] Advanced to Stage {stage}  "
                              f"({self.scheduler.episodes_seen} episodes seen)")

                # Propagate to all parallel workers via set_attr (works for
                # both DummyVecEnv and SubprocVecEnv).
                self.venv.env_method('set_stage', stage_cfg)
                self.venv.env_method('set_curriculum', stage, self.replay_frac)

                # Also update the VecNormalize's wrapped env reference
                # so any future env-level cfg reads are consistent.
                for attr_name, val in stage_cfg.items():
                    try:
                        self.venv.set_attr(attr_name, val)
                    except Exception:
                        pass  # not all attrs are settable via VecEnv

        return True  # do not stop training


# ---------------------------------------------------------------------------
# Checkpoint + VecNormalize save callback
# ---------------------------------------------------------------------------

def retry_io(action, what: str, attempts: int = 4, wait_s: float = 2.0) -> bool:
    """Run a file write, retrying on OSError; never raise.

    The repository sits under OneDrive, whose sync client can hold a file
    open for a moment. A bare write then raises PermissionError, which
    ended the first V20 run at 18.6M of 20M steps (in the dashboard stream;
    the checkpoint writes had the same exposure). A write that still fails
    after `attempts` is reported and skipped: one missing checkpoint, with
    the next due checkpoint_freq steps later, costs less than the run.
    """
    for attempt in range(1, attempts + 1):
        try:
            action()
            return True
        except OSError as exc:
            if attempt == attempts:
                print(f"[IO] WARNING: {what} failed {attempts} times ({exc!r}); skipped",
                      flush=True)
                return False
            time.sleep(wait_s)
    return False


class RetryingCheckpointCallback(CheckpointCallback):
    """SB3's CheckpointCallback with its save retried (see retry_io)."""

    def _on_step(self) -> bool:
        retry_io(super()._on_step, f"model checkpoint at {self.num_timesteps} steps")
        return True


class VecNormCheckpointCallback(BaseCallback):
    """Saves a VecNormalize stats file alongside each model checkpoint.

    SB3's built-in CheckpointCallback only saves the policy zip; this
    companion callback saves the matching normalisation stats so that
    evaluation with --resume loads both files together.

    Also writes curriculum_<num_timesteps>.json alongside each .pkl, so a
    --resume can restore the curriculum stage instead of silently
    restarting at Stage 0 (training runbook Phase B.1).

    Args:
        checkpoint_dir : directory where stats files are written
        save_freq      : timesteps between saves (should match CheckpointCallback)
        venv           : the VecNormalize instance to save
        scheduler      : the CurriculumScheduler whose state is saved alongside
        name_prefix    : filename prefix (default: 'vecnormalize')
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_freq:      int,
        venv:           VecNormalize,
        scheduler:      CurriculumScheduler,
        name_prefix:    str = 'vecnormalize',
    ) -> None:
        super().__init__()
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.save_freq      = save_freq
        self.venv           = venv
        self.scheduler       = scheduler
        self.name_prefix    = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = self.checkpoint_dir / f"{self.name_prefix}_{self.num_timesteps}.pkl"
            retry_io(lambda: self.venv.save(str(path)), f"VecNormalize save {path.name}")
            curriculum_path = self.checkpoint_dir / f"curriculum_{self.num_timesteps}.json"

            def _write_curriculum() -> None:
                with open(curriculum_path, 'w') as f:
                    json.dump(self.scheduler.state_dict(), f)

            retry_io(_write_curriculum, f"curriculum save {curriculum_path.name}")
            if self.verbose >= 1:
                print(f"[VecNorm] Saved {path}")
                print(f"[Curriculum] Saved {curriculum_path}")
        return True


class ActionStdFloorCallback(BaseCallback):
    """Keeps the policy's exploration noise from collapsing.

    Before each rollout the policy's log-std is clamped to at least
    log(min_std). In the V20/V21 runs the action std fell from 1.0 to 0.25
    by 3.5M steps and 0.08 by the end. Two things followed: nothing was
    left to explore with when the curriculum changed the task, and, because
    a fixed-size step in the action mean costs KL ~ (step / std)^2, the
    PPO update's KL rose from 0.016 to 0.19-0.30 with 55-60% of samples
    clipped. Deterministic evaluation is unaffected (it uses the mean).

    Also logs train/action_std_<k> every rollout.
    """

    def __init__(self, min_std: float, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        if not min_std > 0.0:
            raise ValueError(f"min_std must be positive; got {min_std}")
        self.min_log_std = float(np.log(min_std))

    def _on_rollout_start(self) -> None:
        import torch
        log_std = self.model.policy.log_std
        with torch.no_grad():
            log_std.clamp_(min=self.min_log_std)
        for k, v in enumerate(log_std.detach().exp().cpu().numpy().ravel()):
            self.logger.record(f'train/action_std_{k}', float(v))

    def _on_step(self) -> bool:
        return True


def build_model(cfg: dict, venv: VecNormalize, tb_log_dir: str | None = None,
                seed: int = 0) -> PPO:
    """A fresh PPO from cfg['ppo'] (the only place training constructs one)."""
    ppo_cfg = cfg['ppo']
    target_kl = ppo_cfg.get('target_kl')
    model = PPO(
        policy          = ppo_cfg['policy'],
        env             = venv,
        n_steps         = int(ppo_cfg['n_steps']),
        batch_size      = int(ppo_cfg['batch_size']),
        n_epochs        = int(ppo_cfg['n_epochs']),
        gae_lambda      = float(ppo_cfg['gae_lambda']),
        gamma           = float(ppo_cfg['gamma']),
        ent_coef        = float(ppo_cfg['ent_coef']),
        vf_coef         = float(ppo_cfg['vf_coef']),
        max_grad_norm   = float(ppo_cfg['max_grad_norm']),
        learning_rate   = float(ppo_cfg['learning_rate']),
        clip_range      = float(ppo_cfg['clip_range']),
        target_kl       = float(target_kl) if target_kl is not None else None,
        policy_kwargs   = dict(net_arch=list(ppo_cfg['net_arch'])),
        tensorboard_log = tb_log_dir,
        verbose         = int(cfg['logging']['verbose']),
        seed            = seed,
    )
    # Observation version travels with the model (SB3 saves public
    # attributes): PolicyActor and --resume read it back. A model without it
    # predates observation v2 and is v1.
    model.obs_version = flat_env_cfg(cfg)['obs_version']
    return model


def checkpoint_obs_version(path: str) -> int:
    """obs_version saved in a model .zip (1 when absent: pre-V23 models)."""
    from stable_baselines3.common.save_util import load_from_zip_file
    p = str(path) if str(path).endswith('.zip') else f"{path}.zip"
    data, _, _ = load_from_zip_file(p, load_data=True, device='cpu')
    return int((data or {}).get('obs_version', 1))


class MultiStageEvalCallback(BaseCallback):
    """Selects best_model.zip by deterministic success across all stages.

    Every eval_freq calls, the current policy flies n_per_stage episodes at
    each stage in `stages` through training/benchmark.py's single evaluator
    (fresh env per episode, deterministic actions, the training
    VecNormalize statistics frozen at that instant). The seeds are the same
    at every evaluation, so successive checkpoints are compared on identical
    flights, and are drawn from `seed`, which must differ from the
    benchmark's (default 0) so model selection never sees the test episodes.

    Score: mean success rate over the stages; ties go to the lower mean
    crash rate. A strictly better score saves best_model.zip,
    vecnormalize_best.pkl and best_model.json (timestep, score, per-stage
    rates). Every evaluation is appended to eval_logs/multistage.jsonl and
    logged to TensorBoard as eval/success_stage<k>, eval/crash_stage<k> and
    eval/score.

    The score is a max over noisy estimates (10 episodes per stage by
    default), so best_model.json's figures run high; the 100-episode
    benchmark is still the reported number.
    """

    def __init__(
        self,
        cfg:            dict,
        venv:           VecNormalize,
        checkpoint_dir: str | pathlib.Path,
        eval_freq:      int,
        n_per_stage:    int = 10,
        stages:         list[int] | None = None,
        seed:           int = 1000,
        verbose:        int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        from training.benchmark import episode_seeds
        if seed == 0:
            raise ValueError("eval seed 0 is the benchmark's seed set; pick another")
        self.cfg            = cfg
        self.venv           = venv
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.eval_freq      = int(eval_freq)
        self.stages         = list(range(len(STAGES))) if stages is None else [int(x) for x in stages]
        self.seeds          = episode_seeds(int(n_per_stage), int(seed))
        self.best_key: tuple[float, float] | None = None
        self.log_path       = self.checkpoint_dir / 'eval_logs' / 'multistage.jsonl'
        self._pending: list[str] = []

    def evaluate(self) -> dict:
        from training.benchmark import PolicyActor, evaluate_seed
        actor = PolicyActor.from_model(self.model, self.venv)
        per_stage = {}
        for stage in self.stages:
            eps = [evaluate_seed(self.cfg, stage, actor, s) for s in self.seeds]
            n = len(eps)
            per_stage[stage] = {
                'success': sum(e['outcome'] == 'success' for e in eps) / n,
                'crash':   sum(e['outcome'] == 'crash' for e in eps) / n,
                'median_miss_m': float(np.median([e['miss_distance_m'] for e in eps])),
            }
        score = float(np.mean([v['success'] for v in per_stage.values()]))
        crash = float(np.mean([v['crash'] for v in per_stage.values()]))
        return {'timesteps': int(self.num_timesteps), 'score': score,
                'mean_crash': crash, 'stages': per_stage}

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        t0 = time.time()
        result = self.evaluate()
        result['seconds'] = round(time.time() - t0, 1)

        for stage, v in result['stages'].items():
            self.logger.record(f'eval/success_stage{stage}', v['success'])
            self.logger.record(f'eval/crash_stage{stage}', v['crash'])
        self.logger.record('eval/score', result['score'])
        self.logger.record('eval/mean_crash', result['mean_crash'])

        key = (result['score'], -result['mean_crash'])
        result['new_best'] = False
        if self.best_key is None or key > self.best_key:
            try:
                self.model.save(str(self.checkpoint_dir / 'best_model'))
                self.venv.save(str(self.checkpoint_dir / 'vecnormalize_best.pkl'))
                with open(self.checkpoint_dir / 'best_model.json', 'w') as f:
                    json.dump({**result, 'new_best': True}, f, indent=1)
            except OSError as exc:
                # Not recorded as best, so the next evaluation tries again.
                print(f"[Eval] WARNING: could not save best model ({exc!r})")
            else:
                self.best_key = key
                result['new_best'] = True

        self._pending.append(json.dumps(result))
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, 'a') as f:
                f.write(''.join(line + '\n' for line in self._pending))
            self._pending.clear()
        except OSError:
            pass   # kept in _pending and written with the next evaluation

        if self.verbose >= 1:
            rates = '  '.join(f"s{k} {v['success']:.0%}/{v['crash']:.0%}"
                              for k, v in result['stages'].items())
            print(f"[Eval] {self.num_timesteps:,} steps  success/crash {rates}  "
                  f"score {result['score']:.3f}{'  NEW BEST' if result['new_best'] else ''}"
                  f"  ({result['seconds']:.0f} s)")
        return True


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(cfg: dict, resume: str | None = None, seed: int = 0) -> None:
    """Build and run the PPO training loop.

    Args:
        cfg    : full config dict (from base.yaml + CLI overrides)
        resume : path to a .zip checkpoint to resume from (no extension needed)
        seed   : global RNG seed
    """
    ppo_cfg  = cfg['ppo']
    log_cfg  = cfg['logging']
    cur_cfg  = cfg['curriculum']
    eval_cfg = cfg['eval']

    checkpoint_dir = pathlib.Path(log_cfg['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    archive_resolved_config(cfg, checkpoint_dir, resume=resume, seed=seed)

    tb_log_dir = log_cfg['tensorboard_log']
    pathlib.Path(tb_log_dir).mkdir(parents=True, exist_ok=True)

    # --- Build environments ----------------------------------------------
    print(f"Building {ppo_cfg['n_envs']} parallel environments …")
    venv = build_venv(cfg, seed=seed)

    # --- Curriculum scheduler --------------------------------------------
    threshold = cur_cfg['advance_threshold']
    replay_frac = float(cur_cfg.get('replay_frac', 0.0))
    scheduler = CurriculumScheduler(
        advance_threshold     = ([float(t) for t in threshold]
                                 if isinstance(threshold, (list, tuple)) else float(threshold)),
        rolling_window        = int(cur_cfg['rolling_window']),
        max_episodes_at_stage = (
            int(cur_cfg['max_episodes_at_stage'])
            if cur_cfg.get('max_episodes_at_stage') is not None else None
        ),
    )

    # On resume, restore the curriculum stage BEFORE applying it to venv
    # below -- otherwise a run interrupted at Stage 3 would resume at
    # Stage 0 (no wind/gusts/noise/domain-rand) with no indication why the
    # success rate suddenly looks great (training runbook Phase B.1).
    if resume:
        step_num = _extract_step_count(resume)
        curriculum_resume = (
            pathlib.Path(resume).parent / f"curriculum_{step_num}.json"
            if step_num is not None else None
        )
        if curriculum_resume is not None and curriculum_resume.exists():
            with open(curriculum_resume) as f:
                scheduler.load_state_dict(json.load(f))
            print(f"[Curriculum] RESTORED to Stage {scheduler.current_stage} "
                  f"(episodes_at_stage={scheduler.episodes_at_stage}, "
                  f"episodes_seen={scheduler.episodes_seen}) from {curriculum_resume}")
        else:
            print("[Curriculum] WARNING: no matching curriculum_<steps>.json found "
                  "for this resume checkpoint -- restarting the curriculum at "
                  "Stage 0. This is almost certainly not what you want if "
                  "training had already progressed past Stage 0.")

    # make_env() only seeds each worker with the reward section (flat_reward_cfg);
    # it does not know about STAGES[0]. Without this, training silently starts
    # with none of Stage 0's curriculum settings applied (wind, R_home_m,
    # alt0_m, launch offsets, domain-rand ranges) until/unless the scheduler
    # happens to advance past stage 0, at which point set_stage first fires
    # with Stage 1's config. Apply the initial stage explicitly here, the same
    # way CurriculumCallback applies every later stage advance.
    venv.env_method('set_stage', scheduler.current_cfg)
    # ...and put every worker under the curriculum, so episodes replay
    # earlier stages at replay_frac and report info['episode_stage'].
    venv.env_method('set_curriculum', scheduler.current_stage, replay_frac,
                    float(cur_cfg.get('stage0_wind_ms', 0.0)))

    # --- Policy ----------------------------------------------------------
    if resume:
        print(f"Resuming from checkpoint: {resume}")
        want = flat_env_cfg(cfg)['obs_version']
        have = checkpoint_obs_version(resume)
        if have != want:
            raise ValueError(
                f"observation version mismatch: {resume} was trained on obs_version {have}, "
                f"the config says {want}; resume with --set observation.obs_version={have}")
        model = PPO.load(
            resume,
            env         = venv,
            tensorboard_log = tb_log_dir,
            verbose     = int(log_cfg['verbose']),
        )
        # Restore VecNormalize stats if a matching .pkl exists
        vecnorm_resume = _vecnorm_path_for_checkpoint(resume)
        if vecnorm_resume.exists():
            venv = VecNormalize.load(str(vecnorm_resume), venv.venv)
            # The saved wrapper carries the reward-normalisation flag it was
            # trained with; the config decides, not the checkpoint.
            venv.norm_reward = bool(ppo_cfg['normalize_reward'])
            model.set_env(venv)
            print(f"Loaded VecNormalize stats from {vecnorm_resume}")
        else:
            print(f"Warning: no matching VecNormalize stats found at "
                  f"{vecnorm_resume}; continuing with fresh running stats.")
        model.obs_version = have
    else:
        model = build_model(cfg, venv, tb_log_dir, seed)

    # --- Callbacks -------------------------------------------------------
    checkpoint_freq = int(log_cfg['checkpoint_freq'])

    checkpoint_cb = RetryingCheckpointCallback(
        save_freq   = max(checkpoint_freq // int(ppo_cfg['n_envs']), 1),
        save_path   = str(checkpoint_dir),
        name_prefix = 'ppo_glider',
        verbose     = 1,
    )

    vecnorm_cb = VecNormCheckpointCallback(
        checkpoint_dir = str(checkpoint_dir),
        save_freq      = max(checkpoint_freq // int(ppo_cfg['n_envs']), 1),
        venv           = venv,
        scheduler      = scheduler,
        name_prefix    = 'vecnormalize',
    )

    curriculum_cb = CurriculumCallback(
        scheduler   = scheduler,
        venv        = venv,
        replay_frac = replay_frac,
        verbose     = 1,
    )

    eval_cb = MultiStageEvalCallback(
        cfg            = cfg,
        venv           = venv,
        checkpoint_dir = checkpoint_dir,
        eval_freq      = max(int(eval_cfg['eval_freq']) // int(ppo_cfg['n_envs']), 1),
        n_per_stage    = int(eval_cfg.get('n_episodes_per_stage', 10)),
        stages         = eval_cfg.get('stages'),
        seed           = int(eval_cfg.get('seed', 1000)),
        verbose        = 1,
    )

    callback_list = [curriculum_cb, checkpoint_cb, vecnorm_cb, eval_cb]
    if ppo_cfg.get('min_action_std') is not None:
        callback_list.append(ActionStdFloorCallback(float(ppo_cfg['min_action_std'])))
    if WebStreamCallback is not None:
        callback_list.append(WebStreamCallback(
            scheduler       = scheduler,
            seed            = seed,
            total_timesteps = int(ppo_cfg['total_timesteps']),
        ))
    callbacks = CallbackList(callback_list)

    # --- Train -----------------------------------------------------------
    total_steps = int(ppo_cfg['total_timesteps'])
    print(f"Training for {total_steps:,} timesteps …")
    print(f"  TensorBoard: tensorboard --logdir {tb_log_dir}")
    print(f"  Checkpoints: {checkpoint_dir}/")

    t0 = time.time()
    model.learn(
        total_timesteps = total_steps,
        callback        = callbacks,
        reset_num_timesteps = (resume is None),
        progress_bar    = False,
    )
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/3600:.1f} h  ({elapsed:.0f} s)")

    # --- Save final model + VecNormalize stats ---------------------------
    final_model_path  = str(checkpoint_dir / 'ppo_glider_final')
    final_vecnorm_path = str(checkpoint_dir / 'vecnormalize_final.pkl')
    retry_io(lambda: model.save(final_model_path), "final model save")
    retry_io(lambda: venv.save(final_vecnorm_path), "final VecNormalize save")
    print(f"Saved final model  : {final_model_path}.zip")
    print(f"Saved VecNormalize : {final_vecnorm_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the RL-guided glider RTL policy with PPO"
    )
    parser.add_argument(
        '--config', default='training/configs/base.yaml',
        help='path to YAML config file (default: training/configs/base.yaml)',
    )
    parser.add_argument(
        '--set', nargs='*', default=[],
        metavar='SECTION.KEY=VALUE',
        help='override config values, e.g. --set ppo.n_envs=4 ppo.total_timesteps=1000000',
    )
    parser.add_argument(
        '--resume', default=None,
        metavar='CHECKPOINT',
        help='path to .zip checkpoint to resume training from',
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='global RNG seed (default: 0)',
    )
    parser.add_argument(
        '--dummy-vec', action='store_true',
        help='use DummyVecEnv instead of SubprocVecEnv (useful on Windows/debugging)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    cfg = load_config(args.config)
    if args.set:
        cfg = apply_overrides(cfg, args.set)

    # Windows SubprocVecEnv requires the spawn start method and a
    # __main__ guard — both are satisfied here.
    if args.dummy_vec:
        cfg['ppo']['n_envs'] = 1   # DummyVecEnv handles only 1 env elegantly

    train(cfg=cfg, resume=args.resume, seed=args.seed)
