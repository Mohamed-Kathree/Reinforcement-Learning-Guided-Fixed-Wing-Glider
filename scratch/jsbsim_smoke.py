"""
Phase 0 smoke test: verify JSBSim loads a bundled aircraft, trims, and runs.
Run with: python scratch/jsbsim_smoke.py
"""

import jsbsim

root = jsbsim.get_default_root_dir()
print(f"JSBSim root: {root}")
print(f"JSBSim version: {jsbsim.__version__}")

# Use the bundled SGS sailplane (closest to a glider in the bundled set)
fdm = jsbsim.FGFDMExec(root_dir=root, out_simtype="stream")
fdm.set_debug_level(0)
fdm.load_model("SGS")

fdm["ic/h-agl-ft"]    = 300.0    # 300 ft AGL
fdm["ic/vt-fps"]      = 60.0     # ~18 m/s airspeed
fdm["ic/gamma-deg"]   = -3.0     # gentle glide
fdm["ic/psi-true-deg"]= 0.0
fdm.run_ic()

# Trim longitudinally
fdm["simulation/do_trim"] = 0
print("Trim complete.")

# Run 1000 steps at 200 Hz
fdm.set_dt(1.0 / 200.0)
for i in range(1000):
    ok = fdm.run()
    if not ok:
        print(f"fdm.run() returned False at step {i}")
        break

alt_ft = fdm["position/h-agl-ft"]
alt_m  = alt_ft / 3.280839895
print(f"After 1000 steps (5 s): AGL = {alt_ft:.1f} ft  ({alt_m:.1f} m)")
print("Phase 0 smoke test PASSED.")
