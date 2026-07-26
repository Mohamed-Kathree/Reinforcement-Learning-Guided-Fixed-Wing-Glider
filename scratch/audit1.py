"""Audit diagnostics: wind/OU magnitude, alpha-in-wind, rudder sign, contact geometry."""
import numpy as np
from sim.wind import WindModel

print("=" * 70)
print("TEST 1: OU gust stationary std vs configured gust_intensity")
print("=" * 70)
rng = np.random.default_rng(0)
for intensity in (0.5, 1.0, 2.0):
    w = WindModel(tau=2.0, dt=0.005)
    w.reset(np.zeros(3), intensity, rng)
    samples = np.array([w.step() for _ in range(200000)])[20000:]
    print(f"  intensity={intensity:.1f}  ->  realised gust std = {samples.std(axis=0)[0]:.4f} m/s "
          f"(expected ~{intensity:.1f})")

print()
print("=" * 70)
print("TEST 2: alpha from state[3:6] vs JSBSim true aerodynamic alpha, in wind")
print("=" * 70)
from sim.jsbsim_fdm import JSBSimFDM
from sim.math_utils import build_state

fdm = JSBSimFDM(dt_phys=0.005)
for wind_speed in (0.0, 5.0, 9.0):
    st = build_state(p_ned=np.array([0.0, 0.0, -25.0]),
                     v_body=np.array([10.0, 0.0, 0.0]),
                     roll=0.0, pitch=0.0, yaw=0.0, omega=np.zeros(3))
    fdm.reset(st)
    wind = np.array([wind_speed, 0.0, 0.0])  # headwind/tailwind from north
    for _ in range(400):  # 2 s
        fdm.step({'aileron': 0.0, 'elevator': 0.0, 'rudder': 0.0}, wind, 0.005)
    s = fdm.state
    alpha_state = np.degrees(np.arctan2(s[5], s[3]))
    alpha_true = np.degrees(fdm._fdm["aero/alpha-rad"])
    V_state = np.linalg.norm(s[3:6])
    V_true = fdm.airspeed
    print(f"  wind_n={wind_speed:.1f} m/s : alpha_state={alpha_state:+.2f} deg  "
          f"alpha_JSBSim={alpha_true:+.2f} deg  |  V_state={V_state:.2f}  V_air={V_true:.2f}")

print()
print("=" * 70)
print("TEST 3: rudder sign -- does positive rudder yaw nose right or left?")
print("=" * 70)
st = build_state(p_ned=np.array([0.0, 0.0, -200.0]),
                 v_body=np.array([10.0, 0.0, 0.0]),
                 roll=0.0, pitch=0.0, yaw=0.0, omega=np.zeros(3))
fdm.reset(st)
for _ in range(200):
    fdm.step({'aileron': 0.0, 'elevator': 0.0, 'rudder': np.radians(15.0)},
             np.zeros(3), 0.005)
r_rate = fdm.state[12]
_, _, yaw = fdm.euler_angles()
print(f"  +15 deg rudder for 1 s -> yaw rate r = {r_rate:+.4f} rad/s, yaw = {np.degrees(yaw):+.2f} deg")
print("  (positive r / positive yaw = nose RIGHT)")

print()
print("TEST 3b: aileron sign -- does positive aileron roll right?")
fdm.reset(st)
for _ in range(100):
    fdm.step({'aileron': np.radians(15.0), 'elevator': 0.0, 'rudder': 0.0},
             np.zeros(3), 0.005)
roll, _, _ = fdm.euler_angles()
print(f"  +15 deg aileron for 0.5 s -> roll = {np.degrees(roll):+.2f} deg (positive = right wing down)")
print("  => in a RIGHT turn (p>0), controller commands rudder = +KR*p (positive).")
print("     Check above whether positive rudder yaws right (coordinating) or left (aggravating).")

print()
print("=" * 70)
print("TEST 4: ground contact geometry -- WOW altitude vs CG altitude")
print("=" * 70)
st = build_state(p_ned=np.array([0.0, 0.0, -3.0]),
                 v_body=np.array([10.0, 0.0, 0.0]),
                 roll=0.0, pitch=0.0, yaw=0.0, omega=np.zeros(3))
fdm.reset(st)
for i in range(4000):
    fdm.step({'aileron': 0.0, 'elevator': 0.0, 'rudder': 0.0}, np.zeros(3), 0.005)
    if fdm.touched_down:
        print(f"  WOW first True at h_agl = {fdm.altitude:+.4f} m (CG altitude), t={i*0.005:.3f}s")
        break
else:
    print("  never touched down")
