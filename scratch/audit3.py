"""Audit 3: what does the RL policy's speed_cmd action actually do?

Runs the real GliderEnv with a held action and reports the resulting trimmed
airspeed, alpha (true, from JSBSim), pitch target and glide ratio. Also checks
whether the safety shield's alpha term fires correctly in wind.
"""
import numpy as np
from env.glider_env import GliderEnv

print("speed_cmd sweep -- bank_cmd = 0, calm air, 200 m start altitude")
print(f"{'action[1]':>9} {'V_cmd':>6} {'V_air':>7} {'V_state':>8} {'alpha_true':>11} "
      f"{'pitch':>7} {'L/D':>6}")
for a1 in (-1.0, -0.5, 0.0, 0.5, 1.0):
    env = GliderEnv(cfg=dict(wind_speed=0.0, gust_intensity=0.0, sensor_noise=0.0,
                             dropout_prob=0.0, alt0_m=250.0,
                             launch_offset_min_m=50.0, launch_offset_max_m=50.0,
                             launch_speed_min_ms=10.0, launch_speed_max_ms=10.0,
                             launch_pitch_min_deg=0.0, launch_pitch_max_deg=0.0,
                             aero_scale_range=(1.0, 1.0)))
    env.reset(seed=1)
    for _ in range(400):          # 20 s to settle
        env.step(np.array([0.0, a1], dtype=np.float32))
    s0 = env._fdm.state.copy()
    for _ in range(200):          # 10 s measurement window
        env.step(np.array([0.0, a1], dtype=np.float32))
    s1 = env._fdm.state.copy()
    dh = -(s1[2] - s0[2])
    dxy = np.linalg.norm(s1[0:2] - s0[0:2])
    ld = dxy / max(-dh, 1e-6) if dh < 0 else float('nan')
    v_air = env._fdm.airspeed
    alpha = np.degrees(env._fdm._fdm["aero/alpha-rad"])
    pitch = np.degrees(env._fdm.euler_angles()[1])
    vcmd = 9.0 + a1 * 4.0
    print(f"{a1:>9.1f} {vcmd:>6.1f} {v_air:>7.2f} {np.linalg.norm(s1[3:6]):>8.2f} "
          f"{alpha:>11.2f} {pitch:>7.2f} {ld:>6.2f}")

print()
print("Same sweep with a 9 m/s wind (Stage 3 max) -- does the shield still work?")
print(f"{'action[1]':>9} {'V_air':>7} {'alpha_true':>11} {'alpha_state':>12} {'stalled?':>9}")
for a1 in (-1.0, 0.0, 1.0):
    env = GliderEnv(cfg=dict(wind_speed=9.0, gust_intensity=0.0, sensor_noise=0.0,
                             dropout_prob=0.0, alt0_m=250.0,
                             launch_offset_min_m=50.0, launch_offset_max_m=50.0,
                             launch_speed_min_ms=10.0, launch_speed_max_ms=10.0,
                             launch_pitch_min_deg=0.0, launch_pitch_max_deg=0.0,
                             aero_scale_range=(1.0, 1.0)))
    env.reset(seed=3)
    for _ in range(500):
        env.step(np.array([0.0, a1], dtype=np.float32))
    s = env._fdm.state
    alpha_true = np.degrees(env._fdm._fdm["aero/alpha-rad"])
    alpha_state = np.degrees(np.arctan2(s[5], s[3]))
    print(f"{a1:>9.1f} {env._fdm.airspeed:>7.2f} {alpha_true:>11.2f} {alpha_state:>12.2f} "
          f"{'YES' if alpha_true > 12.0 else 'no':>9}")

print()
print("Observation magnitudes at reset (Stage 3), 5 seeds -- input scaling check")
env = GliderEnv(cfg=dict(wind_speed=9.0, gust_intensity=2.0, sensor_noise=1.0,
                         dropout_prob=0.25, alt0_m=20.0,
                         launch_offset_min_m=40.0, launch_offset_max_m=70.0))
for sd in range(3):
    obs, _ = env.reset(seed=sd)
    print("  " + np.array2string(obs, precision=2, suppress_small=True))
