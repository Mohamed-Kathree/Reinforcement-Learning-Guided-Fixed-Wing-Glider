"""
recorder.py
===========
Runs one episode with a chosen controller and records the TRUE simulator
state at every RL policy step (20 Hz) into the Trajectory data contract
(schemas.py / frontend/src/types.ts).

Reads TRUE state from JSBSimFDM's public accessors (position_ned,
euler_angles(), velocity_body, control_surfaces, altitude) -- NOT
env._sensors, which is the noisy, rate-limited, dropout-prone view the
policy itself sees. See FRONTEND_BUILD_CONTEXT.md Section 2.
"""
from __future__ import annotations

import uuid

import numpy as np

from . import config
from .schemas import (
    ControlSurfaces, EpisodeRecordSummary, FlightConditions, Frame, Trajectory, TrajectoryMeta,
)

# env/glider_env.py has no public `dt`/`env.dt` attribute -- DT_RL is a
# module-level constant there (20 Hz RL policy step). Mirrored here rather
# than imported to avoid coupling the dashboard to glider_env's private names.
DT_RL = 0.050


# Lazy-loaded, cached: the policy loads from disk on first "rl" request and
# is reused after that, instead of re-reading the checkpoint files on every
# /api/episode/record call.
_rl_actor = None
_gated_actor = None

# The selected controller (SPEC V24): run 4's best policy, gated by the
# window wind gate frozen in training/configs/base.yaml gate.run4.window.
GATED_POLICY = ("checkpoints_v24_run4", "best_model.zip", "vecnormalize_best.pkl")


def _load_rl_policy():
    """Load the tracked deployment checkpoint (deployment/best_model/), once.

    Returns training/benchmark.py's PolicyActor, which applies the training
    observation pipeline (frame stacking + VecNormalize, n_stack read from
    the model) to raw observations, so a frame-stacked post-V20 model
    replays as correctly as the 12-observation one. Callers reset() it per
    episode.
    """
    global _rl_actor
    if _rl_actor is not None:
        return _rl_actor

    from training.benchmark import PolicyActor

    model_path = config.GLIDER_REPO_ROOT / "deployment" / "best_model" / "best_model.zip"
    vecnorm_path = config.GLIDER_REPO_ROOT / "deployment" / "best_model" / "vecnormalize_best.pkl"
    if not model_path.is_file() or not vecnorm_path.is_file():
        raise NotImplementedError(
            f"RL replay needs {model_path.name} + {vecnorm_path.name} under "
            f"deployment/best_model/ -- not found at {model_path.parent}."
        )

    _rl_actor = PolicyActor(model_path, vecnorm_path)
    return _rl_actor


def _load_gated_actor():
    """Run 4's best policy behind the window wind gate, once.

    The gate reads only the observation vector (baseline/wind_gate.py), so
    it behaves here exactly as in analysis/gate_tune_v24.py's evaluation.
    """
    global _gated_actor
    if _gated_actor is not None:
        return _gated_actor

    from baseline.wind_gated import WindGatedActor
    from training.benchmark import PolicyActor, load_config

    folder, model_name, vecnorm_name = GATED_POLICY
    model_path = config.GLIDER_REPO_ROOT / folder / model_name
    vecnorm_path = config.GLIDER_REPO_ROOT / folder / vecnorm_name
    if not model_path.is_file() or not vecnorm_path.is_file():
        raise NotImplementedError(
            f"Gated replay needs {model_name} + {vecnorm_name} under {folder}/ -- "
            f"not found at {model_path.parent}.")
    gate_cfg = load_config()["gate"]["run4"]
    _gated_actor = WindGatedActor.from_config(PolicyActor(model_path, vecnorm_path),
                                              gate_cfg, "window")
    return _gated_actor


def _extract_true_state(env) -> dict:
    """Pull the TRUE (noise-free) simulator state directly from the FDM.

    `env._fdm` is the GliderEnv's JSBSimFDM instance (sim/jsbsim_fdm.py).
    GliderEnv itself exposes no public accessor for it; this is read-only
    introspection for recording, not a modification of the RL code path.

    `env._wind` (sim/wind.py's WindModel) is read the same way, for
    `wind_ned` -- `.total_ned` is mean + gust, i.e. the actual wind vector
    affecting the aircraft at this instant, not just the per-episode mean.
    """
    fdm = env._fdm
    pos_ned = fdm.position_ned
    roll, pitch, yaw = fdm.euler_angles()
    v_body = fdm.velocity_body
    ail, elev, rud = fdm.control_surfaces
    wind_ned = env._wind.total_ned
    return dict(
        pos_ned=[float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2])],
        euler=[float(roll), float(pitch), float(yaw)],
        v_body=[float(v_body[0]), float(v_body[1]), float(v_body[2])],
        ctrl=ControlSurfaces(ail=float(ail), elev=float(elev), rud=float(rud)),
        agl=float(fdm.altitude),
        wind_ned=[float(wind_ned[0]), float(wind_ned[1]), float(wind_ned[2])],
    )


def record_episode(
    *,
    controller: str = "baseline",
    stage: int = 0,
    seed: int | None = None,
    wind_speed: float | None = None,
    gust_intensity: float | None = None,
    sensor_noise: float | None = None,
    dropout_prob: float | None = None,
    alt0_m: float | None = None,
    launch_offset_m: float | None = None,
    benchmark: bool = False,
) -> Trajectory:
    """Run one episode and record its full TRUE-state trajectory to disk.

    Args:
        controller : "gated" (the selected controller: run 4's best policy
                     behind the window wind gate, baseline fallback),
                     "baseline" (DeterministicRTL) or "rl" (the tracked
                     deployment/best_model/ checkpoint -- the 20M-step
                     reward-fix retrain's best_model.zip, evaluated in
                     Rl-Glider-Context-File-Code-V14-RewardFix-Retrain-Final-Outcome.md).
        stage      : curriculum stage index (0-3), see env/curriculum.py STAGES.
                     Supplies R_home_m/dr_scale/
                     sigma_centre_m/sink_bad/roll_bad_deg and defaults for
                     anything not overridden below.
        seed       : episode RNG seed; drawn randomly if None.

        wind_speed, gust_intensity, sensor_noise, dropout_prob, alt0_m,
        benchmark  : record the flight exactly as the 100-seed benchmark flew it
                     (stage preset, no overrides) under a deterministic id
                     "<controller>-s<stage>-seed<seed>-bench", and return the
                     stored recording if one already exists. Requires a seed.

        launch_offset_m : optional per-episode overrides of the chosen
                     stage's preset, for flying a custom scenario instead
                     of a fixed curriculum difficulty. wind_speed/
                     gust_intensity/sensor_noise/dropout_prob are each the
                     UPPER BOUND GliderEnv.reset() draws the actual
                     per-episode value from -- same convention STAGES
                     already uses, not a fixed exact value (see
                     env/glider_env.py's reset()). launch_offset_m sets
                     both launch_offset_min_m and launch_offset_max_m to
                     the same value, for a deterministic launch distance
                     instead of a preset's randomised range. None (the
                     default) for any of these means "use the stage's
                     preset value," matching pre-existing behaviour.

    Returns:
        The recorded Trajectory (also written to config.EPISODES_DIR).
    """
    if controller not in ("baseline", "rl", "gated"):
        raise ValueError(f"controller must be 'baseline', 'rl' or 'gated'; got {controller!r}")
    if not 0 <= stage < 4:
        raise ValueError(f"stage must be in [0, 3]; got {stage}")
    for name, value in (("wind_speed", wind_speed), ("gust_intensity", gust_intensity),
                        ("sensor_noise", sensor_noise), ("alt0_m", alt0_m),
                        ("launch_offset_m", launch_offset_m)):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be >= 0; got {value}")
    if dropout_prob is not None and not 0.0 <= dropout_prob <= 1.0:
        raise ValueError(f"dropout_prob must be in [0, 1]; got {dropout_prob}")
    if alt0_m is not None and alt0_m <= 0:
        raise ValueError(f"alt0_m must be > 0; got {alt0_m}")
    if launch_offset_m is not None and launch_offset_m <= 0:
        raise ValueError(f"launch_offset_m must be > 0; got {launch_offset_m}")

    from env.curriculum import STAGES
    from env.glider_env import GliderEnv

    if benchmark:
        if seed is None:
            raise ValueError("a benchmark recording needs the benchmark seed")
        if any(v is not None for v in (wind_speed, gust_intensity, sensor_noise, dropout_prob,
                                       alt0_m, launch_offset_m)):
            raise ValueError("a benchmark recording uses the stage preset; drop the overrides")
        bench_id = f"{controller}-s{stage}-seed{seed}-bench"
        existing = config.EPISODES_DIR / f"{bench_id}.json"
        if existing.is_file():
            return Trajectory.model_validate_json(existing.read_text())

    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))

    stage_cfg = dict(STAGES[stage])
    if wind_speed is not None:
        stage_cfg["wind_speed"] = wind_speed
    if gust_intensity is not None:
        stage_cfg["gust_intensity"] = gust_intensity
    if sensor_noise is not None:
        stage_cfg["sensor_noise"] = sensor_noise
    if dropout_prob is not None:
        stage_cfg["dropout_prob"] = dropout_prob
    if alt0_m is not None:
        stage_cfg["alt0_m"] = alt0_m
    if launch_offset_m is not None:
        stage_cfg["launch_offset_min_m"] = launch_offset_m
        stage_cfg["launch_offset_max_m"] = launch_offset_m

    env = GliderEnv(cfg=stage_cfg, seed=seed)

    if controller == "baseline":
        from baseline.deterministic_rtl import DeterministicRTL
        policy = DeterministicRTL()
        policy.reset()  # fresh instance per episode already, but explicit for clarity

        def act(obs):
            return policy.act(obs)
    else:
        # Stepping the raw GliderEnv here (for TRUE-state recording, see
        # module docstring); the actor stacks and normalises observations
        # the way training did.
        rl_actor = _load_gated_actor() if controller == "gated" else _load_rl_policy()
        rl_actor.reset(seed)
        act = rl_actor.act
        # The policy's own observation version (V20-V22 deployment models are
        # v1); it changes what the policy sees, not the episode.
        if env.obs_version != rl_actor.obs_version:
            env = GliderEnv(cfg={**stage_cfg, "obs_version": rl_actor.obs_version}, seed=seed)

    obs, _info = env.reset(seed=seed)

    # Frame 0 is captured before the first step() -- there is no `info` dict
    # yet at this point (it's produced by step()), so dist_home is derived
    # from true position instead of info['dist_home'] (doc gotcha, Section 6).
    s0 = _extract_true_state(env)
    frames: list[Frame] = [Frame(
        t=0.0,
        pos_ned=s0["pos_ned"],
        euler=s0["euler"],
        v_body=s0["v_body"],
        ctrl=s0["ctrl"],
        dist_home=float(np.hypot(s0["pos_ned"][0], s0["pos_ned"][1])),
        agl=s0["agl"],
        wind_ned=s0["wind_ned"],
    )]

    t = 0.0
    info: dict = {}
    terminated = truncated = False
    while True:
        action = act(obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        t += DT_RL

        s = _extract_true_state(env)
        frames.append(Frame(
            t=t,
            pos_ned=s["pos_ned"],
            euler=s["euler"],
            v_body=s["v_body"],
            ctrl=s["ctrl"],
            dist_home=float(info.get("dist_home", np.hypot(s["pos_ned"][0], s["pos_ned"][1]))),
            agl=s["agl"],
            wind_ned=s["wind_ned"],
            # V16 Phase C2 -- straight from compute_reward()'s info dict (via
            # GliderEnv.step()), ground-truth and already computed; frame 0
            # above has no `info` yet, so it keeps Frame's field defaults.
            stall_violation=bool(info.get("stall_violation", False)),
            bank_violation=bool(info.get("bank_violation", False)),
            unreach_violation=bool(info.get("unreach_violation", False)),
            alpha_deg=float(info.get("alpha_deg", 0.0)),
            roll_deg=float(info.get("roll_deg", 0.0)),
            airspeed=float(info.get("airspeed", 0.0)),
        ))

        if terminated or truncated:
            break

    # Four-way outcome, matching env/reward.py's Phase 4 quality-graded
    # landing model -- NOT the old 3-way success/crash/timeout bucketing.
    # A "timeout" is specifically an episode that was truncated WITHOUT
    # ever touching down (real MAX_STEPS exhaustion); an episode that
    # touched down but missed the strict success bar (quality > 0.5 and
    # centred) is a "soft_landing", not a timeout -- under this task most
    # non-crash episodes land clean, just off-centre, so conflating the two
    # (as this function used to) mislabelled the majority of Stage 0-2
    # episodes as timeouts.
    from env.curriculum import outcome_bucket
    outcome: str = outcome_bucket(info)

    quality = float(info.get("quality", 0.0))
    final_dist_home = float(info.get("dist_home", 0.0))

    gate_record = None
    if controller == "gated":
        from .schemas import GateRecord
        gate_record = GateRecord(**rl_actor.episode_record())
        # trace[k] is the estimate from the observation that produced frame k
        # (frame 0 = reset, frame k = after step k); the terminal frame has none.
        for k, w in enumerate(rl_actor.gate.trace[:len(frames)]):
            frames[k].wind_est = float(w)

    episode_id = (f"{controller}-s{stage}-seed{seed}-bench" if benchmark
                  else f"{controller}-s{stage}-{uuid.uuid4().hex[:8]}")
    trajectory = Trajectory(
        episode_id=episode_id,
        meta=TrajectoryMeta(
            stage=stage,
            controller=controller,
            outcome=outcome,
            R_home=float(stage_cfg.get("R_home_m", 20.0)),
            alt0=float(stage_cfg.get("alt0_m", 22.0)),
            seed=seed,
            n_frames=len(frames),
            duration_s=t,
            quality=quality,
            final_dist_home=final_dist_home,
            conditions=FlightConditions(
                wind_speed=float(stage_cfg.get("wind_speed", 0.0)),
                gust_intensity=float(stage_cfg.get("gust_intensity", 0.0)),
                sensor_noise=float(stage_cfg.get("sensor_noise", 0.0)),
                dropout_prob=float(stage_cfg.get("dropout_prob", 0.0)),
                launch_offset_min_m=float(stage_cfg.get("launch_offset_min_m", 0.0)),
                launch_offset_max_m=float(stage_cfg.get("launch_offset_max_m", 0.0)),
            ),
            gate=gate_record,
        ),
        home_ned=[0.0, 0.0, 0.0],
        frames=frames,
    )

    out_path = config.EPISODES_DIR / f"{episode_id}.json"
    out_path.write_text(trajectory.model_dump_json(indent=2))

    # V16 Phase D §D1/§D3 -- a small sidecar so the analysis views can list/
    # filter every recorded episode without parsing full frame arrays (same
    # pattern as data/runs/<id>.meta.json, Phase A §A5).
    summary = EpisodeRecordSummary(
        episode_id=episode_id,
        stage=stage,
        controller=controller,
        outcome=outcome,
        quality=quality,
        seed=seed,
        R_home=trajectory.meta.R_home,
        touchdown_ned=frames[-1].pos_ned,
        home_ned=trajectory.home_ned,
    )
    summary_path = config.EPISODES_DIR / f"{episode_id}.summary.json"
    summary_path.write_text(summary.model_dump_json())

    return trajectory


def load_episode(episode_id: str) -> Trajectory:
    path = config.EPISODES_DIR / f"{episode_id}.json"
    if not path.is_file():
        raise FileNotFoundError(episode_id)
    return Trajectory.model_validate_json(path.read_text())
