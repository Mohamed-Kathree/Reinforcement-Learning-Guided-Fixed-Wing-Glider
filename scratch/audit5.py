"""Audit 5: with the GPS fix applied, where does the simple geometric
controller actually break? This locates the difficulty frontier the RL
policy needs to be evaluated on for the comparison to mean anything."""
import numpy as np
import sim.sensor_models as sm
from sim.math_utils import quat_to_rotmat

def _refresh_gps(self, state, rng):
    R = quat_to_rotmat(state[6:10])
    v_ned = R @ state[3:6]
    self._gps_pos, self._gps_vel = sm._gps_noise(state[0:3], v_ned, rng, self._noise_scale)
sm.SensorSuite._refresh_gps = _refresh_gps

from env.glider_env import GliderEnv
from baseline.deterministic_rtl import DeterministicRTL

def bench(cfg, n_ep=40, seed=0):
    env = GliderEnv(cfg=cfg); ctrl = DeterministicRTL(kp_bank=1.5)
    rng = np.random.default_rng(seed); succ = 0
    for _ in range(n_ep):
        ctrl.reset(); obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        while True:
            obs, r, term, trunc, info = env.step(ctrl.act(obs))
            if term or trunc: break
        succ += bool(info.get("success"))
    return succ / n_ep

base = dict(gust_intensity=2.0, sensor_noise=1.0, dropout_prob=0.25, R_home_m=20.0,
            aero_scale_range=(0.8, 1.2))
print("Difficulty frontier for the P-only geometric controller (GPS fix applied)")
print(f"{'wind':>5} {'alt0':>5} {'offset':>10} {'success':>8}")
for wind, alt, off in [(9, 20, (40, 70)), (9, 20, (70, 110)), (12, 20, (40, 70)),
                       (12, 15, (60, 100)), (9, 15, (80, 130)), (15, 15, (60, 100))]:
    cfg = dict(base, wind_speed=float(wind), alt0_m=float(alt),
               launch_offset_min_m=float(off[0]), launch_offset_max_m=float(off[1]))
    print(f"{wind:>5} {alt:>5} {str(off):>10} {bench(cfg)*100:>7.1f}%")
