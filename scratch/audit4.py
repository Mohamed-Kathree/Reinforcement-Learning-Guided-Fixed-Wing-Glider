"""Audit 4: quantify the impact of the quat_to_rotmat transpose bug.

sim/math_utils.quat_to_rotmat() documents "v_body = R @ v_ned" but actually
returns the body->NED DCM. sim/sensor_models._refresh_gps() then computes
v_ned = R.T @ v_body, which mirrors East velocity and flips the sign of the
vertical speed. That corrupts obs[3] (course_angle) and obs[8]
(vertical_speed) -- the two GPS-derived channels the RTL controller and the
RL policy navigate on.

This script benchmarks the deterministic baseline under both conditions.
"""
import sys
import numpy as np

import sim.math_utils as mu
import sim.sensor_models as sm
from sim.math_utils import quat_to_rotmat

FIX = "--fix" in sys.argv
KP = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--kp=")), 0.3))
KD = float([a.split("=")[1] for a in sys.argv if a.startswith("--kd=")][0]) \
    if any(a.startswith("--kd=") for a in sys.argv) else 5.0

if FIX:
    # Correct body->NED transform: quat_to_rotmat already IS the body->NED DCM
    def _refresh_gps(self, state, rng):
        R = quat_to_rotmat(state[6:10])       # actually C_n/b (body -> NED)
        v_ned = R @ state[3:6]                # <-- no transpose
        self._gps_pos, self._gps_vel = sm._gps_noise(
            state[0:3], v_ned, rng, self._noise_scale
        )
    sm.SensorSuite._refresh_gps = _refresh_gps

from env.glider_env import GliderEnv          # noqa: E402
from env.curriculum import STAGES             # noqa: E402
from baseline.deterministic_rtl import DeterministicRTL   # noqa: E402


def bench(stage_idx, n_ep, kp, kd, seed=0):
    env = GliderEnv(cfg=dict(STAGES[stage_idx]))
    ctrl = DeterministicRTL(kp_bank=kp)   # kd retained on the CLI for historical comparison only; controller is P-only now
    rng = np.random.default_rng(seed)
    succ = crash = timeout = 0
    final_d = []
    for _ in range(n_ep):
        ctrl.reset()
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        while True:
            obs, r, term, trunc, info = env.step(ctrl.act(obs))
            if term or trunc:
                break
        final_d.append(info["dist_home"])
        if info.get("success"):
            succ += 1
        elif info.get("crash"):
            crash += 1
        else:
            timeout += 1
    return succ / n_ep, crash / n_ep, timeout / n_ep, float(np.mean(final_d))


label = "GPS velocity FIXED" if FIX else "current code (buggy)"
print(f"=== {label} ===  kp_bank={KP}  kd_bank={KD}")
print(f"{'stage':>6} {'success':>8} {'crash':>7} {'timeout':>8} {'mean final dist':>16}")
for stage in [0, 3]:
    s, c, t, d = bench(stage, 40, KP, KD)
    print(f"{stage:>6} {s*100:>7.1f}% {c*100:>6.1f}% {t*100:>7.1f}% {d:>15.1f} m")
