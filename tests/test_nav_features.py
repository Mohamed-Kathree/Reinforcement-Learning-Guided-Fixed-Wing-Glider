"""
test_nav_features.py
====================
SPEC V23 Phase O: the shared navigation geometry (env/nav_features.py).

Sign and frame errors are this project's most expensive bug class (the
transposed rotation matrix, the mirrored course reading), so every quantity
here has a sign-convention test and the steering law built on it has a
closed-loop test in the simulator.

Conventions: NED; angles from North, positive toward East; "right" is
clockwise from the direction of travel.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.nav_features import (OBS_DIMS, V1, V2, features_from_v1, guidance,  # noqa: E402
                              nav_features, obs_v2_from_v1)
from sim.math_utils import wrap_pi  # noqa: E402

KEYS = ("along", "cross", "sin_err", "cos_err", "sin_crab", "cos_crab", "ground_speed",
        "course_valid")


def _random_states(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        dx, dy = rng.uniform(-300, 300, 2)
        speed = rng.uniform(2.5, 20.0)
        course = rng.uniform(-math.pi, math.pi)
        yaw = rng.uniform(-math.pi, math.pi)
        yield dx, dy, speed * math.cos(course), speed * math.sin(course), yaw, course


def test_bearing_error_parity():
    """atan2(sin_err, cos_err) is exactly the baseline's v1 heading error."""
    worst = 0.0
    for dx, dy, vn, ve, yaw, _ in _random_states(10_000):
        if math.hypot(dx, dy) < 1.0:
            continue
        f = nav_features(dx, dy, vn, ve, yaw)
        course = math.atan2(ve, vn)
        want = wrap_pi(math.atan2(-dy, -dx) - course)
        got = math.atan2(f["sin_err"], f["cos_err"])
        worst = max(worst, abs(wrap_pi(got - want)))
    assert worst < 1e-9


def test_sign_home_right():
    """Travelling North with home due East: home is to the right."""
    f = nav_features(dx_home=0.0, dy_home=-50.0, vn=10.0, ve=0.0, yaw=0.0)
    assert f["cross"] == pytest.approx(50.0)
    assert f["along"] == pytest.approx(0.0, abs=1e-12)
    assert f["sin_err"] == pytest.approx(1.0)
    assert f["cos_err"] == pytest.approx(0.0, abs=1e-12)
    # ...and home dead ahead: along > 0, no cross error
    g = nav_features(dx_home=-50.0, dy_home=0.0, vn=10.0, ve=0.0, yaw=0.0)
    assert g["along"] == pytest.approx(50.0) and g["cos_err"] == pytest.approx(1.0)
    assert g["sin_err"] == pytest.approx(0.0, abs=1e-12)


def test_sign_crab():
    """Nose North while drifting East: the track is clockwise of the nose."""
    f = nav_features(dx_home=100.0, dy_home=0.0, vn=10.0, ve=3.0, yaw=0.0)
    assert f["sin_crab"] > 0
    assert f["sin_crab"] == pytest.approx(math.sin(math.atan2(3.0, 10.0)))
    assert f["cos_crab"] == pytest.approx(math.cos(math.atan2(3.0, 10.0)))


def test_rotation_invariance():
    rng = np.random.default_rng(1)
    for dx, dy, vn, ve, yaw, _ in _random_states(2_000, seed=2):
        a = rng.uniform(-math.pi, math.pi)
        c, s = math.cos(a), math.sin(a)
        rot = nav_features(c * dx - s * dy, s * dx + c * dy,
                           c * vn - s * ve, s * vn + c * ve, yaw + a)
        ref = nav_features(dx, dy, vn, ve, yaw)
        for k in KEYS:
            assert rot[k] == pytest.approx(ref[k], abs=1e-9), k


def test_no_seam():
    """Sweeping the course through +/-pi in 0.1 deg steps: no feature jumps."""
    dx, dy, speed = 40.0, 0.0, 10.0          # home due South of the glider
    prev = None
    worst = 0.0
    for deg in np.arange(150.0, 210.0001, 0.1):
        course = math.radians(deg)
        f = nav_features(dx, dy, speed * math.cos(course), speed * math.sin(course),
                         course + math.radians(5.0))
        if prev is not None:
            worst = max(worst, max(abs(f[k] - prev[k]) for k in KEYS))
        prev = f
    # 0.1 deg of rotation moves a unit vector by 0.0017 and a 40 m one by 0.07 m
    assert worst < 0.1


def test_low_speed_fallback():
    f = nav_features(dx_home=30.0, dy_home=-20.0, vn=0.5, ve=1.0, yaw=0.7)
    h = nav_features(dx_home=30.0, dy_home=-20.0, vn=5.0 * math.cos(0.7),
                     ve=5.0 * math.sin(0.7), yaw=0.7)
    assert f["course_valid"] == 0.0 and h["course_valid"] == 1.0
    assert f["sin_crab"] == pytest.approx(0.0, abs=1e-12)
    assert f["cos_crab"] == pytest.approx(1.0)
    for k in ("along", "cross", "sin_err", "cos_err"):   # heading used as the track
        assert f[k] == pytest.approx(h[k], abs=1e-12)


def test_v2_frame_layout_and_scales():
    v1 = np.zeros(OBS_DIMS[1], dtype=np.float32)
    v1[[V1["dx_home"], V1["dy_home"]]] = [-50.0, 0.0]        # home 50 m North
    v1[V1["ground_speed"]], v1[V1["course"]] = 10.0, 0.0     # flying North
    v1[V1["yaw"]] = 0.0
    v1[V1["roll"]], v1[V1["pitch"]] = 0.1, -0.05
    v1[V1["baro_alt"]], v1[V1["vertical_speed"]] = 15.0, -1.0
    v1[V1["lidar_agl"]], v1[V1["lidar_valid"]], v1[V1["req_glide_ratio"]] = 2.0, 1.0, 0.3
    o = obs_v2_from_v1(v1, np.array([0.25, -0.5], dtype=np.float32))
    assert o.shape == (OBS_DIMS[2],) and o.dtype == np.float32
    assert o[V2["along"]] == pytest.approx(0.5) and o[V2["cross"]] == pytest.approx(0.0, abs=1e-7)
    assert o[V2["cos_err"]] == pytest.approx(1.0) and o[V2["ground_speed"]] == pytest.approx(1.0)
    assert o[V2["course_valid"]] == 1.0 and o[V2["baro_alt"]] == pytest.approx(0.5)
    assert o[V2["roll"]] == pytest.approx(0.1) and o[V2["pitch"]] == pytest.approx(-0.05)
    assert (o[V2["prev_bank"]], o[V2["prev_speed"]]) == (pytest.approx(0.25), pytest.approx(-0.5))


def test_guidance_reads_both_versions_alike():
    rng = np.random.default_rng(3)
    for dx, dy, vn, ve, yaw, course in _random_states(500, seed=4):
        v1 = np.zeros(OBS_DIMS[1], dtype=np.float32)
        v1[[V1["dx_home"], V1["dy_home"]]] = [dx, dy]
        v1[V1["ground_speed"]], v1[V1["course"]] = math.hypot(vn, ve), course
        v1[V1["yaw"]], v1[V1["baro_alt"]] = yaw, rng.uniform(1, 30)
        g1 = guidance(v1)
        g2 = guidance(obs_v2_from_v1(v1, np.zeros(2)))
        assert g2["dist_home"] == pytest.approx(g1["dist_home"], rel=1e-5)
        assert wrap_pi(g2["heading_err"] - g1["heading_err"]) == pytest.approx(0.0, abs=1e-5)
        assert g2["baro_alt"] == pytest.approx(g1["baro_alt"], rel=1e-5)
        assert features_from_v1(v1)["ground_speed"] == pytest.approx(math.hypot(vn, ve), rel=1e-5)


# ---------------------------------------------------------------------------
# Closed loop, in the simulator
# ---------------------------------------------------------------------------

def _crosswind_flight(seed: int, steer: str, wind_ms: float = 3.0) -> tuple[float, float]:
    """Fly a proportional law on the bearing error to home, at constant speed
    command, in a steady wind perpendicular to the launch-to-home line.

    steer='track'  : error relative to the GPS ground track (the v2 features)
    steer='heading': error relative to the nose (what a policy steering by
                     yaw computes)
    Returns (closest approach to home, mean |angle between ground track and
    home bearing| on the inbound leg from 40 m to 15 m, degrees).
    """
    from env.curriculum import STAGES
    from env.glider_env import GliderEnv

    env = GliderEnv()
    env.set_stage(dict(STAGES[0]))
    obs, _ = env.reset(seed=seed)
    p0 = env._fdm.state[0:2].copy()
    u = p0 / np.hypot(*p0)
    env._wind.reset(np.array([-u[1] * wind_ms, u[0] * wind_ms, 0.0]), 0.0,
                    np.random.default_rng(seed))
    closest, prev, errs = np.inf, p0, []
    for _ in range(2000):
        if steer == "track":
            f = features_from_v1(obs)
        else:
            f = nav_features(float(obs[V1["dx_home"]]), float(obs[V1["dy_home"]]), 0.0, 0.0,
                             float(obs[V1["yaw"]]))      # zero velocity: heading as track
        bank = float(np.clip(math.atan2(f["sin_err"], f["cos_err"]), -1.0, 1.0))
        obs, _, term, trunc, _ = env.step(np.array([bank, 0.0], dtype=np.float32))
        p = env._fdm.state[0:2].copy()
        d, prev = p - prev, p
        dist = float(np.hypot(*p))
        closest = min(closest, dist)
        if 15.0 < dist < 40.0 and closest > 15.0:
            errs.append(abs(wrap_pi(math.atan2(-p[1], -p[0]) - math.atan2(d[1], d[0]))))
        if term or trunc:
            break
    env.close()
    return closest, math.degrees(float(np.mean(errs)))


def test_closed_loop_turn_sign():
    """A positive bank command turns the ground track clockwise (course
    increases): the whole chain action -> controller -> JSBSim -> GPS."""
    from env.curriculum import STAGES
    from env.glider_env import GliderEnv
    for cmd, sign in ((0.6, 1.0), (-0.6, -1.0)):
        env = GliderEnv()
        env.set_stage(dict(STAGES[0]))
        obs, _ = env.reset(seed=11)
        for _ in range(40):                                   # 2 s wings level
            obs, *_ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        c0 = float(obs[V1["course"]])
        turned = 0.0
        for _ in range(40):                                   # 2 s banked
            obs, *_ = env.step(np.array([cmd, 0.0], dtype=np.float32))
            c1 = float(obs[V1["course"]])
            turned += wrap_pi(c1 - c0)
            c0 = c1
        env.close()
        assert sign * turned > math.radians(20), (cmd, math.degrees(turned))


def test_course_steering_cancels_drift():
    """In a 3 m/s crosswind, steering the ground track at home holds the
    track on the home bearing; steering the nose at home leaves a standing
    track error near the crab angle.

    Both laws reach home in the end (pure pursuit converges whenever
    airspeed exceeds the wind), so closest approach alone does not separate
    them; the inbound track error does, and it is what turns into a
    cross-wind miss when the glide runs out of height.
    """
    seeds = range(12)
    track = [_crosswind_flight(s, "track") for s in seeds]
    head = [_crosswind_flight(s, "heading") for s in seeds]
    assert max(c for c, _ in track) < 10.0
    med_track = float(np.median([e for _, e in track]))
    med_head = float(np.median([e for _, e in head]))
    assert med_track < 6.0, med_track
    assert med_head > 3.0 * med_track and med_head > 12.0, (med_track, med_head)
