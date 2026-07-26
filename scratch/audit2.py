"""Audit 2: feasibility envelope of the RTL task under the curriculum's own settings.

Uses the aircraft's own aero model (CL table + CD0 + induced drag) to build a
speed polar, then computes maximum achievable ground range as a function of
headwind, and finally Monte-Carlos each curriculum stage's launch/wind
distribution to estimate the fraction of episodes that are physically winnable
by ANY policy.
"""
import numpy as np

RHO = 1.225
S = 0.38
M = 1.1
G = 9.80665
CD0 = 0.025
K = 0.03566          # 1/(pi e AR)
W = M * G


def sink_and_speed(CL):
    """Steady glide at a given CL: airspeed and sink rate."""
    CD = CD0 + K * CL ** 2
    V = np.sqrt(2 * W / (RHO * S * CL))
    sink = V * CD / CL
    return V, sink, CD


CLs = np.linspace(0.15, 1.25, 400)
polar = np.array([sink_and_speed(cl) for cl in CLs])
V_arr, sink_arr, CD_arr = polar[:, 0], polar[:, 1], polar[:, 2]
LD = V_arr / sink_arr

i_best = np.argmax(LD)
i_min_sink = np.argmin(sink_arr)
print("Aircraft polar from rlglider.xml's own coefficients")
print(f"  best L/D      = {LD[i_best]:.1f} at V = {V_arr[i_best]:.2f} m/s (CL={CLs[i_best]:.2f}), "
      f"sink = {sink_arr[i_best]:.2f} m/s")
print(f"  min sink      = {sink_arr[i_min_sink]:.2f} m/s at V = {V_arr[i_min_sink]:.2f} m/s")
print(f"  stall speed   = {np.sqrt(2*W/(RHO*S*1.15)):.2f} m/s (at CL=1.15)")
print()

# Speed command range available to the policy: 5..13 m/s
V_MIN_CMD, V_MAX_CMD = 5.0, 13.0
mask = (V_arr >= V_MIN_CMD) & (V_arr <= V_MAX_CMD)


def max_ground_range(alt, headwind):
    """Best achievable range over the ground, optimising speed-to-fly."""
    vg = V_arr[mask] - headwind
    sk = sink_arr[mask]
    glide = vg / sk
    return alt * max(glide.max(), 0.0)


print("Max ground range vs headwind (alt = 20 m, speed-to-fly optimised):")
for hw in (0, 2, 4, 6, 8, 9):
    print(f"   headwind {hw:>2} m/s -> {max_ground_range(20.0, hw):6.1f} m")
print()

STAGES = [
    dict(name="0", wind=0.0, alt=25.0, off=(50.0, 90.0), R=25.0),
    dict(name="1", wind=3.0, alt=23.0, off=(45.0, 80.0), R=22.0),
    dict(name="2", wind=6.0, alt=21.0, off=(40.0, 70.0), R=20.0),
    dict(name="3", wind=9.0, alt=20.0, off=(40.0, 70.0), R=20.0),
]

rng = np.random.default_rng(0)
N = 100_000
print("Monte-Carlo: fraction of episodes physically winnable by an ideal policy")
print("(optimistic: assumes instant optimal turn, no turn losses, no sensor noise,")
print(" straight-line return, and full credit for R_home)")
print()
for st in STAGES:
    wind_speed = rng.uniform(0.0, st["wind"], N)
    wind_dir = rng.uniform(0, 2 * np.pi, N)
    off = rng.uniform(st["off"][0], st["off"][1], N)
    bearing = rng.uniform(0, 2 * np.pi, N)
    # Vector from aircraft to home, in NED
    to_home = -np.stack([off * np.cos(bearing), off * np.sin(bearing)])
    dist = np.linalg.norm(to_home, axis=0)
    unit = to_home / dist
    wind_vec = np.stack([wind_speed * np.cos(wind_dir), wind_speed * np.sin(wind_dir)])
    # Component of wind along the required track (+ = tailwind)
    tail = np.sum(wind_vec * unit, axis=0)
    # crosswind component costs a little too, ignored here (optimistic)
    needed = dist - st["R"]
    # vectorised: best over available speeds of (V - headwind)/sink
    Vsel = V_arr[mask][None, :]
    Ssel = sink_arr[mask][None, :]
    glide = (Vsel + tail[:, None]) / Ssel
    ok = st["alt"] * glide.max(axis=1) >= needed
    print(f"  Stage {st['name']}: winnable = {ok.mean()*100:5.1f}%   "
          f"(median required range {np.median(needed):.0f} m)")
