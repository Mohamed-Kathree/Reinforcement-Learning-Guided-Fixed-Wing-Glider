"""
plot_trajectories.py
====================
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

3-D and 2-D flight-path visualisation using matplotlib.

Produces four figures per call:
    Figure 1 -- 3-D trajectory (North / East / Altitude)
    Figure 2 -- Ground track (North vs East, top-down)
    Figure 3 -- Altitude and airspeed time-series
    Figure 4 -- Safety metric time-series (AGL, roll, alpha, reward)

Usage
-----
Run agains the deterministic baseline (no trained model needed):

    python -m analysis.plot_trajectories

Run against a trained RL policy:

    python -m analysis.plot_trajectories \\
        --model   checkpoints/best_model \\
        --vecnorm checkpoints/vecnormalize.pkl

Compare both on one figure:

    python -m analysis.plot_trajectories \\
        --model   checkpoints/best_model \\
        --vecnorm checkpoints/vecnormalize.pkl \\
        --compare

Additional options:

    --stage   0-3      curriculum stage to use (default 0)
    --seed    INT      episode RNG seed        (default 42)
    --save    DIR      save figures to DIR instead of showing interactively
    --no-show          suppress interactive window (useful with --save)

Coordinate frames used:
    NED : North-East-Down inertial frame (world)
    BODY: Forward-Right-Down body frame (FRD, attached to glider)
    WIND: Stability/wind frame (x into relative wind)

Quaternion convention: [q0, q1, q2, q3] where q0 is the scalar component.
Rotation R maps NED -> BODY: v_body = R @ v_ned

Units: SI throughout (m, m/s, rad, rad/s, kg, N, N*m)
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D projection)

from env.glider_env import GliderEnv
from env.reward import DEFAULT_REWARD_CFG
from baseline.deterministic_rtl import DeterministicRTL
from sim.math_utils import euler_from_quat


# ---------------------------------------------------------------------------
# Trajectory data container
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """Stores per-step data recorded during one episode rollout."""
    label:   str
    color:   str

    # Positional / kinematic time-series (one entry per policy step)
    time:    list[float]       = field(default_factory=list)
    north:   list[float]       = field(default_factory=list)
    east:    list[float]       = field(default_factory=list)
    alt:     list[float]       = field(default_factory=list)
    airspeed: list[float]      = field(default_factory=list)
    roll_deg: list[float]      = field(default_factory=list)
    pitch_deg: list[float]     = field(default_factory=list)
    alpha_deg: list[float]     = field(default_factory=list)
    agl:     list[float]       = field(default_factory=list)
    reward:  list[float]       = field(default_factory=list)
    dist_home: list[float]     = field(default_factory=list)

    # Episode outcome
    success: bool  = False
    crash:   bool  = False
    n_steps: int   = 0


# ---------------------------------------------------------------------------
# Rollout collector
# ---------------------------------------------------------------------------

def collect_trajectory(
    policy:    Any,
    env:       GliderEnv,
    seed:      int,
    label:     str,
    color:     str,
    is_vecenv: bool = False,
) -> Trajectory:
    """Run one episode and record per-step state data.

    Works with both a raw GliderEnv (baseline) and a VecNormalize-wrapped
    env (RL policy).  Ground-truth physics state is read directly from the
    underlying GliderEnv in both cases.

    Args:
        policy    : object with .predict(obs, deterministic=True) interface
        env       : GliderEnv or VecNormalize
        seed      : episode RNG seed
        label     : legend label for the trajectory
        color     : matplotlib colour string
        is_vecenv : True when env is VecNormalize-wrapped

    Returns:
        Trajectory with all per-step fields populated.
    """
    traj = Trajectory(label=label, color=color)
    dt   = 0.05   # policy step size (s)

    if is_vecenv:
        env.env_method('reset', seed=seed)
        obs = env.reset()
        # Reach the underlying GliderEnv through VecNormalize -> VecEnv -> envs[0]
        raw_env = env.venv.envs[0].unwrapped
    else:
        obs, _ = env.reset(seed=seed)
        raw_env = env

    t = 0.0
    while True:
        # Record ground-truth state BEFORE this step
        s     = raw_env._fdm.state
        roll, pitch, _ = euler_from_quat(s[6:10])
        alpha = float(np.arctan2(s[5], s[3]))
        V     = float(np.linalg.norm(s[3:6]))
        alt   = float(-s[2])

        traj.time.append(t)
        traj.north.append(float(s[0]))
        traj.east.append(float(s[1]))
        traj.alt.append(alt)
        traj.airspeed.append(V)
        traj.roll_deg.append(float(np.degrees(roll)))
        traj.pitch_deg.append(float(np.degrees(pitch)))
        traj.alpha_deg.append(float(np.degrees(alpha)))

        # Step
        action, _ = policy.predict(obs, deterministic=True)

        if is_vecenv:
            obs, reward, done, info_arr = env.step(action)
            r    = float(reward[0])
            done = bool(done[0])
            info = info_arr[0]
        else:
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        traj.agl.append(float(info.get('lidar_agl', 0.0)))
        traj.reward.append(r)
        traj.dist_home.append(float(info.get('dist_home', np.nan)))

        t += dt

        if done:
            traj.success = bool(info.get('success', False))
            traj.crash   = bool(info.get('crash',   False))
            break

    traj.n_steps = len(traj.time)
    return traj


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _outcome_marker(traj: Trajectory) -> tuple[str, str]:
    """Return (marker, colour) for the episode end-point marker."""
    if traj.success:
        return '*', 'gold'
    if traj.crash:
        return 'X', 'red'
    return 'o', 'grey'   # timeout


def _add_home_marker(ax, kind: str = '3d') -> None:
    """Plot a green circle at the launch / home position."""
    if kind == '3d':
        ax.scatter([0], [0], [0], s=120, c='lime', marker='H',
                   zorder=5, label='Home')
    else:
        ax.scatter([0], [0], s=120, c='lime', marker='H',
                   zorder=5, label='Home')


def plot_3d(trajectories: list[Trajectory], ax: plt.Axes) -> None:
    """Draw 3-D flight paths on an existing Axes3D."""
    ax.set_xlabel('North (m)')
    ax.set_ylabel('East (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('3-D Flight Path')

    for traj in trajectories:
        ax.plot(traj.north, traj.east, traj.alt,
                color=traj.color, linewidth=1.4, label=traj.label)
        # Launch point
        ax.scatter([traj.north[0]], [traj.east[0]], [traj.alt[0]],
                   c=traj.color, marker='^', s=60, zorder=4)
        # End point
        m, mc = _outcome_marker(traj)
        ax.scatter([traj.north[-1]], [traj.east[-1]], [traj.alt[-1]],
                   c=mc, marker=m, s=100, zorder=5)

    _add_home_marker(ax, kind='3d')
    ax.legend(loc='upper left', fontsize=8)


def plot_ground_track(trajectories: list[Trajectory], ax: plt.Axes) -> None:
    """Draw top-down ground track (North vs East)."""
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('Ground Track')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)

    for traj in trajectories:
        ax.plot(traj.east, traj.north,
                color=traj.color, linewidth=1.4, label=traj.label)
        ax.scatter([traj.east[0]],  [traj.north[0]],
                   c=traj.color, marker='^', s=60, zorder=4)
        m, mc = _outcome_marker(traj)
        ax.scatter([traj.east[-1]], [traj.north[-1]],
                   c=mc, marker=m, s=100, zorder=5)

    _add_home_marker(ax, kind='2d')

    # Home-radius rings for each stage R_home_m
    r_home = DEFAULT_REWARD_CFG['R_home_m']
    theta  = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r_home * np.sin(theta), r_home * np.cos(theta),
            'g--', linewidth=0.8, alpha=0.6, label=f'R_home={r_home:.0f} m')
    ax.legend(loc='upper right', fontsize=8)


def plot_timeseries(trajectories: list[Trajectory], fig: plt.Figure) -> None:
    """Draw altitude/airspeed and safety-metric time-series panels."""
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    ax_alt, ax_spd, ax_angles, ax_reward = axes

    # --- Altitude ---
    ax_alt.set_title('Altitude')
    ax_alt.set_xlabel('Time (s)')
    ax_alt.set_ylabel('Alt AGL (m)')
    ax_alt.axhline(DEFAULT_REWARD_CFG['agl_min'], color='red',
                   linestyle='--', linewidth=0.8, label=f"agl_min={DEFAULT_REWARD_CFG['agl_min']:.0f} m")
    ax_alt.grid(True, alpha=0.3)
    for traj in trajectories:
        ax_alt.plot(traj.time, traj.alt, color=traj.color,
                    linewidth=1.2, label=traj.label)
    ax_alt.legend(fontsize=7)

    # --- Airspeed ---
    ax_spd.set_title('Airspeed')
    ax_spd.set_xlabel('Time (s)')
    ax_spd.set_ylabel('Airspeed (m/s)')
    ax_spd.axhline(10.0, color='green', linestyle=':', linewidth=0.8,
                   label='Best-glide 10 m/s')
    ax_spd.axhline(6.5,  color='red',   linestyle='--', linewidth=0.8,
                   label='~Stall speed 6.5 m/s')
    ax_spd.grid(True, alpha=0.3)
    for traj in trajectories:
        ax_spd.plot(traj.time, traj.airspeed, color=traj.color,
                    linewidth=1.2, label=traj.label)
    ax_spd.legend(fontsize=7)

    # --- Roll / AoA ---
    ax_angles.set_title('Roll & Angle-of-Attack')
    ax_angles.set_xlabel('Time (s)')
    ax_angles.set_ylabel('Angle (deg)')
    ax_angles.axhline( 30.0, color='orange', linestyle='--', linewidth=0.8,
                       label='Bank limit ±30°')
    ax_angles.axhline(-30.0, color='orange', linestyle='--', linewidth=0.8)
    ax_angles.axhline( 12.0, color='red',    linestyle=':',  linewidth=0.8,
                       label='Stall α 12°')
    ax_angles.grid(True, alpha=0.3)
    for traj in trajectories:
        ax_angles.plot(traj.time, traj.roll_deg,  color=traj.color,
                       linewidth=1.2, linestyle='-',  label=f'{traj.label} roll')
        ax_angles.plot(traj.time, traj.alpha_deg, color=traj.color,
                       linewidth=1.0, linestyle='--', label=f'{traj.label} α')
    ax_angles.legend(fontsize=6)

    # --- Reward ---
    ax_reward.set_title('Reward per Step')
    ax_reward.set_xlabel('Time (s)')
    ax_reward.set_ylabel('Reward')
    ax_reward.axhline(0, color='grey', linewidth=0.6)
    ax_reward.grid(True, alpha=0.3)
    for traj in trajectories:
        ax_reward.plot(traj.time, traj.reward, color=traj.color,
                       linewidth=1.0, label=traj.label)
    ax_reward.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def make_figures(
    trajectories: list[Trajectory],
    save_dir:     str | None = None,
    show:         bool       = True,
) -> list[plt.Figure]:
    """Build and optionally save / display the four standard figures.

    Args:
        trajectories : list of Trajectory objects to plot
        save_dir     : if given, save PNGs here (figure_1_3d.png etc.)
        show         : call plt.show() at the end

    Returns:
        List of the four Figure objects.
    """
    # Separate outcome label per trajectory
    labels_with_outcome = []
    for t in trajectories:
        suffix = ' ✓' if t.success else (' ✗' if t.crash else ' …')
        labels_with_outcome.append(
            Trajectory.__new__(Trajectory).__class__(
                label=t.label + suffix,
                color=t.color,
            )
        )
    # Use copies with outcome-annotated labels for legends
    display = []
    for orig, lwo in zip(trajectories, labels_with_outcome):
        d        = Trajectory(label=lwo.label, color=orig.color)
        d.__dict__.update({k: v for k, v in orig.__dict__.items() if k not in ('label', 'color')})
        display.append(d)

    figures: list[plt.Figure] = []

    # Figure 1 — 3-D trajectory
    fig1 = plt.figure(figsize=(9, 7))
    ax3d = fig1.add_subplot(111, projection='3d')
    plot_3d(display, ax3d)
    fig1.tight_layout()
    figures.append(fig1)

    # Figure 2 — Ground track
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    plot_ground_track(display, ax2)
    fig2.tight_layout()
    figures.append(fig2)

    # Figure 3 & 4 — Time-series (altitude/speed + safety)
    fig3 = plt.figure(figsize=(12, 7))
    fig3.suptitle('Episode Time-Series', fontsize=12)
    plot_timeseries(display, fig3)
    figures.append(fig3)

    if save_dir is not None:
        out = pathlib.Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = ['figure_1_3d', 'figure_2_groundtrack', 'figure_3_timeseries']
        for fig, name in zip(figures, names):
            path = out / f'{name}.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f'Saved {path}')

    if show:
        plt.show()

    return figures


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    policy,
    cfg:       dict,
    stage_idx: int,
    seed:      int,
    label:     str,
    color:     str,
    is_vecenv: bool = False,
) -> Trajectory:
    """Build env, run one episode, return a Trajectory."""
    from env.curriculum import STAGES

    stage_cfg  = dict(STAGES[stage_idx])
    reward_cfg = dict(DEFAULT_REWARD_CFG)
    reward_cfg.update(cfg.get('reward', {}))
    reward_cfg.update(stage_cfg)

    if is_vecenv:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.monitor import Monitor

        def _make():
            return Monitor(GliderEnv(cfg=reward_cfg))

        venv = DummyVecEnv([_make])
        venv = VecNormalize.load(cfg['_vecnorm_path'], venv)
        venv.training    = False
        venv.norm_reward = False
        traj = collect_trajectory(policy, venv, seed, label, color, is_vecenv=True)
        venv.close()
    else:
        env  = GliderEnv(cfg=reward_cfg)
        traj = collect_trajectory(policy, env, seed, label, color, is_vecenv=False)
        env.close()

    return traj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Plot 3-D and time-series flight trajectories'
    )
    p.add_argument('--model',   default=None,
                   help='path to trained PPO .zip checkpoint')
    p.add_argument('--vecnorm', default=None,
                   help='path to VecNormalize .pkl stats file')
    p.add_argument('--config',  default='training/configs/base.yaml',
                   help='path to YAML config (default: training/configs/base.yaml)')
    p.add_argument('--stage',   type=int, default=0, choices=[0, 1, 2, 3],
                   help='curriculum stage (default: 0)')
    p.add_argument('--seed',    type=int, default=42,
                   help='episode RNG seed (default: 42)')
    p.add_argument('--compare', action='store_true',
                   help='overlay RL and baseline on the same figures')
    p.add_argument('--save',    default=None,
                   help='directory to save PNG figures (default: show interactively)')
    p.add_argument('--no-show', action='store_true',
                   help='suppress interactive display (useful with --save)')
    return p.parse_args()


if __name__ == '__main__':
    import yaml

    args = _parse_args()
    cfg  = yaml.safe_load(pathlib.Path(args.config).read_text())

    trajectories: list[Trajectory] = []

    # --- Deterministic baseline (always collected) -----------------------
    baseline_ctrl = DeterministicRTL()
    base_traj     = run_episode(
        policy    = baseline_ctrl,
        cfg       = cfg,
        stage_idx = args.stage,
        seed      = args.seed,
        label     = 'Deterministic RTL',
        color     = 'steelblue',
    )
    print(f"Baseline  steps={base_traj.n_steps:4d}  "
          f"success={base_traj.success}  crash={base_traj.crash}")

    if args.compare or args.model is None:
        trajectories.append(base_traj)

    # --- RL policy (optional) --------------------------------------------
    if args.model is not None:
        from stable_baselines3 import PPO
        if args.vecnorm is None:
            raise SystemExit('--vecnorm is required when --model is provided')

        cfg['_vecnorm_path'] = args.vecnorm
        rl_model = PPO.load(args.model)
        rl_traj  = run_episode(
            policy    = rl_model,
            cfg       = cfg,
            stage_idx = args.stage,
            seed      = args.seed,
            label     = 'RL Policy',
            color     = 'tomato',
            is_vecenv = True,
        )
        print(f"RL Policy steps={rl_traj.n_steps:4d}  "
              f"success={rl_traj.success}  crash={rl_traj.crash}")
        trajectories.append(rl_traj)

        if not args.compare:
            # RL-only plot (baseline already printed above but not appended)
            pass
    else:
        # Baseline-only run: trajectories already set above
        pass

    if not trajectories:
        trajectories = [base_traj]

    make_figures(
        trajectories = trajectories,
        save_dir     = args.save,
        show         = not args.no_show,
    )
