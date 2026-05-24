"""
Phase 1 verification: load rlglider.xml, inspect catalog, trim, free-glide 30 s.
Run with: python scratch/verify_aircraft.py
"""

import os
import sys
import jsbsim
import numpy as np

# Project root is one level above this script's directory
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FT2M = 1.0 / 3.280839895
FPS2MS = 1.0 / 3.280839895

def load_model(debug=0, dt=1.0/200.0):
    fdm = jsbsim.FGFDMExec(root_dir=PROJ_ROOT)
    fdm.set_debug_level(debug)
    # set_dt BEFORE load_model: FCS rate limiters cache dt at load time.
    # If set_dt is called after load_model the FCS uses the default 1/120 s dt,
    # making rate limits 5/3 too loose at 200 Hz.
    fdm.set_dt(dt)
    ok = fdm.load_model("rlglider")
    if not ok:
        raise RuntimeError("load_model('rlglider') failed")
    return fdm

# ------------------------------------------------------------------ #
# 1. Property catalog spot-check
# ------------------------------------------------------------------ #
def check_catalog():
    fdm = load_model()
    catalog = fdm.query_property_catalog("")

    required = [
        "aero/qbar-psf",
        "aero/alpha-deg",
        "aero/alpha-rad",
        "aero/beta-rad",
        "aero/bi2vel",
        "aero/ci2vel",
        "metrics/Sw-sqft",
        "metrics/bw-ft",
        "metrics/cbarw-ft",
        "velocities/p-rad_sec",
        "velocities/q-rad_sec",
        "velocities/r-rad_sec",
        "velocities/u-fps",
        "velocities/v-fps",
        "velocities/w-fps",
        "velocities/vt-fps",
        "attitude/phi-rad",
        "attitude/theta-rad",
        "attitude/psi-rad",
        "position/h-agl-ft",
        "fcs/elevator-cmd-norm",
        "fcs/elevator-pos-rad",
        "fcs/aileron-cmd-norm",
        "fcs/aileron-pos-rad",
        "fcs/rudder-cmd-norm",
        "fcs/rudder-pos-rad",
        "aero/cl-mult",
        "aero/cd0-add",
        "aero/ctrl-eff-mult",
    ]

    missing = []
    for prop in required:
        if prop not in catalog:
            missing.append(prop)

    if missing:
        print(f"WARN  missing properties: {missing}")
    else:
        print(f"PASS  catalog: all {len(required)} required properties found")

    # Print wind properties to confirm exact names
    wind_props = [p for p in catalog.split("\n") if "wind" in p.lower() and p.strip()]
    print(f"      wind properties: {wind_props[:6]}")

    # Print distance-from-start properties
    dist_props = [p for p in catalog.split("\n") if "distance" in p.lower() and p.strip()]
    print(f"      distance properties: {dist_props[:6]}")

    return fdm

# ------------------------------------------------------------------ #
# 2. Longitudinal trim + 30 s free glide
# ------------------------------------------------------------------ #
def check_trim_and_glide():
    fdm = load_model()   # dt already set to 1/200 inside load_model()

    # Set neutral domain-rand properties (defaults should already be right,
    # but set explicitly to be sure)
    fdm["aero/cl-mult"]       = 1.0
    fdm["aero/cd0-add"]       = 0.0
    fdm["aero/ctrl-eff-mult"] = 1.0

    # Initial conditions: moderate altitude, plausible trim airspeed
    # a0*alpha_trim = Cm0/(-Cm_alpha) = 0.05/0.60 = 0.0833 rad → 4.77 deg
    ALPHA_TRIM_DEG = 4.77
    fdm["ic/h-agl-ft"]      = 300.0 * 3.280839895   # 300 m AGL in ft
    fdm["ic/vt-fps"]         = 10.0 * 3.280839895    # 10 m/s in fps
    fdm["ic/alpha-deg"]      = ALPHA_TRIM_DEG        # seed the trim near equilibrium
    fdm["ic/gamma-deg"]      = -4.0                  # close to equilibrium glide slope
    fdm["ic/psi-true-deg"]   = 0.0
    fdm.run_ic()

    # Skip do_trim(): JSBSim's solver can't navigate through the <actuator> lag
    # filter (it expects a static cmd->pos map).  ICs are already at the
    # analytical trim: Cm0 + Cm_alpha * alpha_trim = 0.05 - 0.60*0.0833 ≈ 0,
    # so elevator=0 is the trim elevator.  do_trim() is tested in validate_glide.py.
    # Run one step so JSBSim computes the aero state from the ICs.
    fdm["fcs/elevator-cmd-norm"] = 0.0
    fdm["fcs/aileron-cmd-norm"]  = 0.0
    fdm["fcs/rudder-cmd-norm"]   = 0.0
    fdm.run()
    print(f"      near-trim IC: alpha={np.degrees(fdm['aero/alpha-rad']):.2f} deg  "
          f"theta={np.degrees(fdm['attitude/theta-rad']):.2f} deg  "
          f"elev={np.degrees(fdm['fcs/elevator-pos-rad']):.2f} deg  "
          f"V={fdm['velocities/vt-fps']*FPS2MS:.2f} m/s  "
          f"q={fdm['velocities/q-rad_sec']:.4f} rad/s")

    # Free glide: controls already zeroed above; run the timed loop
    pitch_rates = []
    glide_ratios = []
    DT = 1.0 / 200.0
    n_steps = int(30.0 / DT)
    for i in range(n_steps):
        ok = fdm.run()
        if not ok:
            print(f"FAIL  fdm.run() returned False at step {i}")
            return

        q_rate = abs(fdm["velocities/q-rad_sec"])
        pitch_rates.append(q_rate)

        vn  = fdm["velocities/v-north-fps"] * FPS2MS
        ve  = fdm["velocities/v-east-fps"]  * FPS2MS
        vd  = fdm["velocities/v-down-fps"]  * FPS2MS
        v_h = np.sqrt(vn**2 + ve**2)
        if vd > 0.1:
            glide_ratios.append(v_h / vd)

    max_q  = float(np.max(pitch_rates))
    glide  = float(np.median(glide_ratios)) if glide_ratios else 0.0
    alt_f  = fdm["position/h-agl-ft"] * FT2M

    print(f"      after 30 s free glide: alt={alt_f:.1f} m  "
          f"max_q={max_q:.4f} rad/s  glide_ratio={glide:.2f}")

    PASS = True
    if max_q >= 0.1:
        print(f"FAIL  pitch rate {max_q:.4f} rad/s >= 0.1 rad/s limit")
        PASS = False
    else:
        print(f"PASS  pitch rate  max = {max_q:.4f} rad/s  (< 0.1)")

    if not (12.0 < glide < 22.0):
        print(f"FAIL  glide ratio {glide:.2f} outside (12, 22)")
        PASS = False
    else:
        print(f"PASS  glide ratio = {glide:.2f}  (in 12–22)")

    if alt_f <= 0:
        print(f"FAIL  glider hit ground (alt={alt_f:.1f} m)")
        PASS = False

    return PASS

# ------------------------------------------------------------------ #
# 3. Actuator lag check (step command -> rate-limited response)
# ------------------------------------------------------------------ #
def check_actuator_lag():
    fdm = load_model()   # dt already set to 1/200 inside load_model()
    fdm["ic/h-agl-ft"]  = 100.0 * 3.280839895
    fdm["ic/vt-fps"]    = 12.0  * 3.280839895
    fdm["ic/gamma-deg"] = -3.0
    fdm.run_ic()

    # Settle actuator at zero so prev is a known baseline before the step test
    fdm["fcs/elevator-cmd-norm"] = 0.0
    fdm["fcs/aileron-cmd-norm"]  = 0.0
    fdm["fcs/rudder-cmd-norm"]   = 0.0
    for _ in range(40):          # 0.2 s settle at zero command
        fdm.run()

    # Step elevator to full deflection
    fdm["fcs/elevator-cmd-norm"] = 1.0
    prev = fdm["fcs/elevator-pos-rad"]
    max_rate = 0.0
    DT2 = 1.0 / 200.0
    for _ in range(40):  # 0.2 s
        fdm.run()
        curr  = fdm["fcs/elevator-pos-rad"]
        rate  = abs(curr - prev) / DT2
        max_rate = max(max_rate, rate)
        prev = curr

    limit = 3.491  # rad/s (200 deg/s)
    if max_rate <= limit + 1e-3:
        print(f"PASS  actuator rate limit  max rate = {np.degrees(max_rate):.1f} deg/s  (<= 200)")
    else:
        print(f"FAIL  actuator rate {np.degrees(max_rate):.1f} deg/s exceeded limit 200 deg/s")

# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("\n=== Phase 1: aircraft XML verification ===\n")
    check_catalog()
    print()
    passed = check_trim_and_glide()
    print()
    check_actuator_lag()
    print()
    if passed:
        print("Phase 1 verification PASSED — XML is flight-worthy.")
    else:
        print("Phase 1 verification FAILED — check warnings above.")
        sys.exit(1)
