"""
test_flight_firmware.py
=======================
Host tests for the flight firmware's pure-C core (firmware/flight/src/core/),
compiled with the host gcc and driven through ctypes:

  - Kalman: C rlg_kalman_update() vs glider_viz.py's Kalman (attitude_ref.py
    carries it verbatim), on random sequences.
  - Attitude pipeline: C rlg_attitude_update() vs attitude_ref.Attitude, both
    kinematics modes, accel gate included.
  - Frame: accel roll/pitch signs for the corrected frame; imu_frame.h signs
    have determinant +1.
  - Inner loop: C rlg_inner_loop() vs the simulator's AttitudeController and
    _safety_shield (env/glider_env.py), gains read from the simulator.
  - Servo mapping: rlg_deflection_to_us().
  - Sources: firmware_sync --check (protocol copies and calibration header
    current); extern "C" guards present.
  - Optional: arduino-cli compile of firmware/flight (set RLG_ARDUINO_COMPILE=1).

Proves the arithmetic and the contract with the simulator. Says nothing about
the ESP32 integration (SPI, interrupts, timing, PWM), which is the Phase 1
hardware test plan in firmware/flight/README.md.
"""

from __future__ import annotations

import ctypes
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "firmware", "tools"))

import attitude_ref as AR  # noqa: E402
from env import glider_env as G  # noqa: E402

CORE = os.path.join(ROOT, "firmware", "flight", "src", "core")
_LIB = None


class Kal(ctypes.Structure):
    _fields_ = [("angle", ctypes.c_float), ("bias", ctypes.c_float), ("P", (ctypes.c_float * 2) * 2)]


class Att(ctypes.Structure):
    _fields_ = [("roll", Kal), ("pitch", Kal), ("yaw_deg", ctypes.c_float), ("seeded", ctypes.c_bool),
                ("euler_kinematics", ctypes.c_bool), ("accel_trusted", ctypes.c_bool),
                ("accel_mag_g", ctypes.c_float), ("samples", ctypes.c_uint32),
                ("gated_out", ctypes.c_uint32), ("dt_rejected", ctypes.c_uint32)]


class Surf(ctypes.Structure):
    _fields_ = [("aileron", ctypes.c_float), ("elevator", ctypes.c_float), ("rudder", ctypes.c_float),
                ("shield_bank", ctypes.c_uint8), ("shield_pitch", ctypes.c_uint8)]


def lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no host C compiler: the C core is NOT tested")
    out = os.path.join(tempfile.mkdtemp(prefix="rlg_core_"), "core" + (".dll" if sys.platform == "win32" else ".so"))
    cmd = [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O1", "-shared",
           os.path.join(CORE, "attitude.c"), os.path.join(CORE, "inner_loop.c"), "-o", out, "-lm"]
    if sys.platform != "win32":
        cmd.insert(5, "-fPIC")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    L = ctypes.CDLL(out)
    f = ctypes.c_float
    L.rlg_kalman_update.argtypes = [ctypes.POINTER(Kal), f, f, f]
    L.rlg_kalman_update.restype = f
    L.rlg_attitude_init.argtypes = [ctypes.POINTER(Att), ctypes.c_bool]
    L.rlg_attitude_update.argtypes = [ctypes.POINTER(Att), f, f, f, f, f, f, f]
    L.rlg_attitude_update.restype = ctypes.c_bool
    L.rlg_accel_roll_deg.argtypes = [f, f]
    L.rlg_accel_roll_deg.restype = f
    L.rlg_accel_pitch_deg.argtypes = [f, f, f]
    L.rlg_accel_pitch_deg.restype = f
    L.rlg_inner_loop.argtypes = [f] * 7
    L.rlg_inner_loop.restype = Surf
    L.rlg_deflection_to_us.argtypes = [f, f, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    L.rlg_deflection_to_us.restype = ctypes.c_uint16
    _LIB = L
    return L


def header_define(path, name):
    m = re.search(rf"#define\s+{name}\s+\(?\s*([-+0-9.]+)", open(path).read())
    return float(m.group(1))


# ---------------------------------------------------------------- Kalman

def test_kalman_matches_glider_viz():
    L = lib()
    rng = random.Random(1)
    for _ in range(20):
        k = Kal()
        py = AR.Kalman()
        a0 = rng.uniform(-60, 60)
        k.angle = py.angle = a0
        worst = 0.0
        for _ in range(2000):
            ang, rate, dt = rng.uniform(-90, 90), rng.uniform(-200, 200), rng.uniform(0.003, 0.006)
            c = L.rlg_kalman_update(ctypes.byref(k), ang, rate, dt)
            p = py.update(ang, rate, dt)
            worst = max(worst, abs(c - p))
        assert worst < 0.05, f"C vs Python Kalman diverged by {worst:.4f} deg (float32 vs float64)"
        assert abs(k.bias - py.bias) < 0.05


def test_kalman_constants_match():
    h = os.path.join(CORE, "attitude.h")
    assert header_define(h, "ATT_Q_ANGLE") == AR.Q_ANGLE == 0.001
    assert header_define(h, "ATT_Q_BIAS") == AR.Q_BIAS == 0.003
    assert header_define(h, "ATT_R_MEASURE") == AR.R_MEASURE == 0.03
    assert header_define(h, "ATT_GATE_LO_G") == 0.85
    assert header_define(h, "ATT_GATE_HI_G") == 1.15


# ---------------------------------------------------------------- frame

@pytest.mark.parametrize("name,acc,roll,pitch", [
    ("level", (0, 0, -1), 0, 0),
    ("nose up 30", (math.sin(math.radians(30)), 0, -math.cos(math.radians(30))), 0, 30),
    ("right wing down 30", (0, -math.sin(math.radians(30)), -math.cos(math.radians(30))), 30, 0),
    ("left wing down 20", (0, math.sin(math.radians(20)), -math.cos(math.radians(20))), -20, 0),
])
def test_accel_angles_corrected_frame(name, acc, roll, pitch):
    """Specific force, x fwd / y right / z down: level az=-1 g, nose up ax>0, right wing down ay<0."""
    L = lib()
    ax, ay, az = acc
    assert L.rlg_accel_roll_deg(ay, az) == pytest.approx(roll, abs=1e-3), name
    assert L.rlg_accel_pitch_deg(ax, ay, az) == pytest.approx(pitch, abs=1e-3), name
    assert AR.accel_roll_deg(ay, az) == pytest.approx(roll, abs=1e-4)
    assert AR.accel_pitch_deg(ax, ay, az) == pytest.approx(pitch, abs=1e-4)


def test_imu_frame_is_a_rotation():
    src = open(os.path.join(CORE, "imu_frame.h")).read()
    s = {n: int(re.search(rf"#define\s+{n}\s+\(([-+]?1)\)", src).group(1)) for n in ("AX_SIGN", "AY_SIGN", "AZ_SIGN")}
    assert s["AX_SIGN"] * s["AY_SIGN"] * s["AZ_SIGN"] == 1, "determinant must be +1"
    for a in ("X", "Y", "Z"):      # gyro uses the same transform as the accel
        assert re.search(rf"#define\s+G{a}_SIGN\s+A{a}_SIGN\s", src), f"G{a}_SIGN must alias A{a}_SIGN"
    # The Nov 2026 decision: raw frame IS the body frame.
    assert all(v == 1 for v in s.values())
    assert "_Static_assert" in src


# ---------------------------------------------------------------- attitude pipeline

def _random_imu(rng, n, gate_some=True):
    """Smooth random motion plus noise; some samples well outside the 0.85-1.15 g gate."""
    roll = pitch = 0.0
    for i in range(n):
        roll += rng.uniform(-0.3, 0.3)
        pitch += rng.uniform(-0.2, 0.2)
        roll, pitch = max(-70, min(70, roll)), max(-40, min(40, pitch))
        r, p = math.radians(roll), math.radians(pitch)
        scale = rng.choice([1.0, 1.0, 1.0, 1.4, 0.6]) if gate_some else 1.0
        ax = math.sin(p) * scale + rng.gauss(0, 0.01)
        ay = -math.sin(r) * math.cos(p) * scale + rng.gauss(0, 0.01)
        az = -math.cos(r) * math.cos(p) * scale + rng.gauss(0, 0.01)
        g = (rng.gauss(0, 20), rng.gauss(0, 20), rng.gauss(0, 20))
        dt = rng.choice([0.00444, 0.00445, 0.0089, 0.0])     # incl. a skipped sample and a dt=0 reject
        yield ax, ay, az, *g, dt


@pytest.mark.parametrize("euler", [False, True])
def test_attitude_pipeline_matches_reference(euler):
    L = lib()
    rng = random.Random(7 if euler else 3)
    a = Att()
    L.rlg_attitude_init(ctypes.byref(a), euler)
    py = AR.Attitude(euler=euler)
    worst = 0.0
    gated = 0
    for s in _random_imu(rng, 3000):
        cu = L.rlg_attitude_update(ctypes.byref(a), *s)
        pu = py.update(*s)
        assert cu == pu
        assert a.accel_trusted == py.trusted
        gated += not py.trusted and pu
        worst = max(worst, abs(a.roll.angle - py.roll.angle), abs(a.pitch.angle - py.pitch.angle))
    assert worst < 0.1, f"C vs Python attitude max difference {worst:.4f} deg"
    assert a.gated_out == gated > 100, "the accel gate must have been exercised"


def test_attitude_rejects_bad_dt_and_seeds():
    L = lib()
    a = Att()
    L.rlg_attitude_init(ctypes.byref(a), True)
    assert not L.rlg_attitude_update(ctypes.byref(a), 0, -0.5, -0.866, 0, 0, 0, 0.005)   # seeds
    assert a.seeded and a.roll.angle == pytest.approx(30, abs=0.01)
    assert not L.rlg_attitude_update(ctypes.byref(a), 0, 0, -1, 0, 0, 0, 0.0)
    assert not L.rlg_attitude_update(ctypes.byref(a), 0, 0, -1, 0, 0, 0, 0.25)
    assert a.dt_rejected == 2


def test_euler_kinematics_holds_bank_in_a_turn():
    """Steady 30 deg right turn, accel gated out (|a| = 1/cos 30 = 1.155 g): body rates
    q, r are non-zero, Euler rates are zero. Euler kinematics holds the bank; body-rate
    integration (glider_viz.py) drifts. This is why the flight default is Euler."""
    L = lib()
    bank = math.radians(30)
    yaw_rate = 20.0                                          # deg/s
    q, r = yaw_rate * math.sin(bank), yaw_rate * math.cos(bank)
    acc = (0.0, 0.0, -1.0 / math.cos(bank))                 # coordinated: along body z, outside the gate
    out = {}
    for euler in (True, False):
        a = Att()
        L.rlg_attitude_init(ctypes.byref(a), euler)
        a.seeded = True
        a.roll.angle = 30.0
        for _ in range(225 * 10):                           # 10 s at 225 Hz
            L.rlg_attitude_update(ctypes.byref(a), *acc, 0.0, q, r, 1 / 225)
        out[euler] = (a.roll.angle, a.pitch.angle, a.gated_out)
    assert out[True][2] == 2250, "1.155 g must be gated out"
    assert out[True][0] == pytest.approx(30.0, abs=0.01)
    assert out[True][1] == pytest.approx(0.0, abs=0.01)
    assert abs(out[False][1]) > 50, "body-rate integration should have drifted (documents why euler is default)"


# ---------------------------------------------------------------- inner loop vs simulator

def test_gains_match_simulator():
    h = os.path.join(CORE, "inner_loop.h")
    assert header_define(h, "IL_KP_ROLL") == G.KP_ROLL
    assert header_define(h, "IL_KD_ROLL") == G.KD_ROLL
    assert header_define(h, "IL_KP_PITCH") == G.KP_PITCH
    assert header_define(h, "IL_KD_PITCH") == G.KD_PITCH
    assert header_define(h, "IL_KR_RUDDER") == G.KR_RUDDER
    assert header_define(h, "IL_SHIELD_BANK") == pytest.approx(G.SHIELD_MAX_BANK_RAD, abs=1e-7)
    assert header_define(h, "IL_SHIELD_PITCH") == pytest.approx(G.SHIELD_MAX_PITCH_RAD, abs=1e-7)


def _sim_pitch_target(roll, V_actual, speed_cmd):
    """Same arithmetic as AttitudeController.update(); the firmware receives this as its setpoint."""
    cos_factor = max(abs(math.cos(roll)), 0.5)
    base = (G.ALPHA_TRIM * (G.V_TRIM / max(speed_cmd, 4.0)) ** 2) / cos_factor
    return base - G.KV_PITCH * (speed_cmd - V_actual)


def test_inner_loop_matches_simulator_controller_and_shield():
    L = lib()
    rng = np.random.default_rng(11)
    ctrl = G.AttitudeController()
    shield_hits = [0, 0]
    for _ in range(5000):
        roll = rng.uniform(-1.2, 1.2)
        pitch = rng.uniform(-0.5, 0.6)
        p, q, r = rng.uniform(-6, 6, size=3)
        bank_cmd = rng.uniform(-math.pi / 4, math.pi / 4)
        speed_cmd = rng.uniform(9.5, 16.0)
        V = rng.uniform(7, 18)
        cmds = ctrl.update(np.array([roll, pitch, 0.0]), np.array([p, q, r]), V, bank_cmd, speed_cmd)
        cmds = G.GliderEnv._safety_shield(None, np.array([roll, pitch, 0.0]), np.array([p, q, r]), cmds)
        pitch_sp = _sim_pitch_target(roll, V, speed_cmd)
        s = L.rlg_inner_loop(roll, pitch, p, q, r, bank_cmd, pitch_sp)
        assert s.aileron == pytest.approx(cmds["aileron"], abs=2e-5)
        assert s.elevator == pytest.approx(cmds["elevator"], abs=2e-5)
        assert s.rudder == pytest.approx(cmds["rudder"], abs=2e-5)
        shield_hits[0] += s.shield_bank
        shield_hits[1] += s.shield_pitch
    assert shield_hits[0] > 100 and shield_hits[1] > 50, f"shields not exercised: {shield_hits}"


def test_inner_loop_signs():
    """Right wing down -> aileron negative (roll left); nose high -> elevator positive (nose down)."""
    L = lib()
    s = L.rlg_inner_loop(math.radians(20), 0.0, 0, 0, 0, 0.0, 0.0)
    assert s.aileron < 0
    s = L.rlg_inner_loop(0.0, math.radians(10), 0, 0, 0, 0.0, 0.0)
    assert s.elevator > 0
    s = L.rlg_inner_loop(0.0, 0.0, 1.0, 0, 0, 0.0, 0.0)     # rolling right
    assert s.rudder < 0                                      # negative rudder = nose right (Cn_dr < 0)


def test_failsafe_setpoint_in_simulator_range():
    cfg = os.path.join(ROOT, "firmware", "flight", "config.h")
    roll = header_define(cfg, "FAILSAFE_ROLL_DEG")
    pitch = header_define(cfg, "FAILSAFE_PITCH_DEG")
    assert roll == 0.0
    assert -8.0 <= pitch <= 12.0, "must lie inside the simulator's pitch-target clip"


# ---------------------------------------------------------------- servo mapping

def test_deflection_to_us():
    L = lib()
    f = L.rlg_deflection_to_us
    assert f(0.0, 0.4363, 1500, 1, 400, 600, 2300) == 1500
    assert f(0.4363, 0.4363, 1500, 1, 400, 600, 2300) == 1900
    assert f(0.4363, 0.4363, 1500, -1, 400, 600, 2300) == 1100
    assert f(0.2, 0.4, 1480, 1, 400, 600, 2300) == 1680
    assert f(9.0, 0.4, 1500, 1, 400, 600, 2300) == 1900          # input clamped to full deflection
    assert f(0.4, 0.4, 2100, 1, 400, 600, 2300) == 2300          # then to the travel limit
    assert f(-0.4, 0.4, 900, 1, 400, 700, 2300) == 700


# ---------------------------------------------------------------- sources

def test_firmware_sources_in_sync():
    if subprocess.run(["git", "diff", "--quiet", "--", "flighttest/config/calibration.json"], cwd=ROOT).returncode:
        pytest.skip("calibration.json has uncommitted edits (saved from the dashboard; the aircraft gets them "
                    "over the radio). Commit, then python -m flighttest.firmware_sync, to update the compiled-in default")
    r = subprocess.run([sys.executable, "-m", "flighttest.firmware_sync", "--check"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_extern_c_guards():
    for path in (os.path.join(ROOT, "firmware", "protocol", "rlg_safety.h"),
                 os.path.join(ROOT, "firmware", "protocol", "rlg_protocol.h"),
                 os.path.join(CORE, "attitude.h"), os.path.join(CORE, "inner_loop.h")):
        src = open(path).read()
        assert '#ifdef __cplusplus\nextern "C" {' in src.replace("\r\n", "\n"), path


def test_radio_never_sent_from_control_loop():
    """Brief: esp_now_send only from the radio task, with a 15 ms settle."""
    d = os.path.join(ROOT, "firmware", "flight")
    ino = re.sub(r"/\*.*?\*/|//[^\n]*", "", open(os.path.join(d, "flight.ino")).read(), flags=re.S)
    radio = re.sub(r"/\*.*?\*/|//[^\n]*", "", open(os.path.join(d, "radio_link.h")).read(), flags=re.S)
    assert "esp_now_send" not in ino
    assert radio.count("esp_now_send(") == 1
    assert re.search(r"#define\s+RADIO_SETTLE_MS\s+15\b", open(os.path.join(d, "config.h")).read())
    assert "SEND_CB_ARG" in radio


@pytest.mark.skipif(os.environ.get("RLG_ARDUINO_COMPILE") != "1", reason="set RLG_ARDUINO_COMPILE=1 (slow)")
def test_arduino_compile():
    cli = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Arduino IDE", "resources", "app",
                       "lib", "backend", "resources", "arduino-cli.exe")
    cli = cli if os.path.exists(cli) else shutil.which("arduino-cli")
    if not cli:
        pytest.skip("arduino-cli not found")
    r = subprocess.run([cli, "compile", "--fqbn", "esp32:esp32:esp32", os.path.join(ROOT, "firmware", "flight")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
