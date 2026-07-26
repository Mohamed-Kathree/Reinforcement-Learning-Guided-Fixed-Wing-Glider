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
from .schemas import ControlSurfaces, Frame, Trajectory, TrajectoryMeta

# env/glider_env.py has no public `dt`/`env.dt` attribute -- DT_RL is a
# module-level constant there (20 Hz RL policy step). Mirrored here rather
# than imported to avoid coupling the dashboard to glider_env's private names.
DT_RL = 0.050


def _extract_true_state(env) -> dict:
    """Pull the TRUE (noise-free) simulator state directly from the FDM.

    `env._fdm` is the GliderEnv's JSBSimFDM instance (sim/jsbsim_fdm.py).
    GliderEnv itself exposes no public accessor for it; this is read-only
    introspection for recording, not a modification of the RL code path.
    """
    fdm = env._fdm
    pos_ned = fdm.position_ned
    roll, pitch, yaw = fdm.euler_angles()
    v_body = fdm.velocity_body
    ail, elev, rud = fdm.control_surfaces
    return dict(
        pos_ned=[float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2])],
        euler=[float(roll), float(pitch), float(yaw)],
        v_body=[float(v_body[0]), float(v_body[1]), float(v_body[2])],
        ctrl=ControlSurfaces(ail=float(ail), elev=float(elev), rud=float(rud)),
        agl=float(fdm.altitude),
    )


def record_episode(
    *,
    controller: str = "baseline",
    stage: int = 0,
    seed: int | None = None,
) -> Trajectory:
    """Run one episode and record its full TRUE-state trajectory to disk.

    Args:
        controller : "baseline" (DeterministicRTL). "rl" is not available
                     until a trained model + VecNormalize stats exist
                     (FRONTEND_BUILD_CONTEXT.md TODO 6 / Milestone 6).
        stage      : curriculum stage index (0-3), see env/curriculum.py STAGES.
        seed       : episode RNG seed; drawn randomly if None.

    Returns:
        The recorded Trajectory (also written to config.EPISODES_DIR).
    """
    if controller != "baseline":
        raise NotImplementedError(
            "RL replay needs a trained PPO model + vecnormalize.pkl "
            "(not available yet) -- baseline replay only for now."
        )
    if not 0 <= stage < 4:
        raise ValueError(f"stage must be in [0, 3]; got {stage}")

    from env.curriculum import STAGES
    from env.glider_env import GliderEnv
    from baseline.deterministic_rtl import DeterministicRTL

    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))

    stage_cfg = dict(STAGES[stage])
    env = GliderEnv(cfg=stage_cfg, seed=seed)
    policy = DeterministicRTL()
    policy.reset()  # fresh instance per episode already, but explicit for clarity

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
    )]

    t = 0.0
    info: dict = {}
    terminated = truncated = False
    while True:
        action = policy.act(obs)
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
    if info.get("success"):
        outcome: str = "success"
    elif truncated and not terminated:
        outcome = "timeout"
    elif info.get("crash"):
        outcome = "crash"
    else:
        outcome = "soft_landing"

    quality = float(info.get("quality", 0.0))
    final_dist_home = float(info.get("dist_home", 0.0))

    episode_id = f"{controller}-s{stage}-{uuid.uuid4().hex[:8]}"
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
        ),
        home_ned=[0.0, 0.0, 0.0],
        frames=frames,
    )

    out_path = config.EPISODES_DIR / f"{episode_id}.json"
    out_path.write_text(trajectory.model_dump_json(indent=2))

    return trajectory


def load_episode(episode_id: str) -> Trajectory:
    path = config.EPISODES_DIR / f"{episode_id}.json"
    if not path.is_file():
        raise FileNotFoundError(episode_id)
    return Trajectory.model_validate_json(path.read_text())
