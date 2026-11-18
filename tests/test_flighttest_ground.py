"""
test_flighttest_ground.py
=========================
Flight-test phase 2: the ground station against the stub glider, over the
real byte protocol, with a fake clock (deterministic).

  - link statistics: rate, loss from sequence gaps (u16 wrap), duplicates and
    reboots not counted, CRC drops;
  - command sender: explicit arming, <= 20 Hz regardless of how fast the UI
    sends, DISARMED burst on stop, UI-idle expiry;
  - firmware behaviour seen through telemetry: arming needs consecutive frames,
    the 150 ms timeout when the link drops, re-arming after recovery, refusal
    when airborne;
  - honest sensor reporting: rail voltage "not measured" unless a sensor exists;
  - the log format and parser, the stub Pi, and the Pi-side log server over HTTP.
"""

from __future__ import annotations

import json
import os
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flighttest import calibration as C  # noqa: E402
from flighttest import logs  # noqa: E402
from flighttest.ground import COMMAND_PERIOD_MS, UI_IDLE_EXPIRY_MS, GroundStation  # noqa: E402
from flighttest.links import Link, StubLink  # noqa: E402
from flighttest.pi import PiClient, StubPi, safe_log_name  # noqa: E402
from flighttest.protocol import packets as P  # noqa: E402
from flighttest.stub_glider import StubGlider  # noqa: E402

NEUTRAL = [1480, 1510, 1500, 1495]
CAL = C.with_updates(C.load(), **{f"servos.{n}.neutral_us": v for n, v in zip(C.SERVO_NAMES, NEUTRAL)})


def rig(loss=0.0, seed=0, cal=CAL, **kw):
    g = StubGlider(C.neutral_us(cal), now_ms=0, seed=seed, loss=loss, limits=C.limits(cal), **kw)
    gs = GroundStation(StubLink(g), cal=cal)
    return g, gs


def run(gs, t0, t1, dt=10):
    t = t0
    while t <= t1:
        gs.step(t)
        t += dt
    return t1


def glider_report(gs):
    return gs.latest["status"] if gs.latest else None


# ---------------------------------------------------------------- statistics

def test_telemetry_rate_and_zero_loss():
    g, gs = rig(loss=0.0)
    run(gs, 0, 10_000)
    snap = gs.snapshot(10_000)["ground"]
    assert 4.6 <= snap["rate_hz"] <= 5.2
    assert snap["loss_pct"] == 0.0 and snap["crc_drops"] == 0
    assert snap["receiving"] and snap["last_packet_age_ms"] < 250


def test_loss_measured_from_sequence_gaps():
    g, gs = rig(loss=0.2, seed=4)
    run(gs, 0, 120_000)
    lost, rec = gs.lost_total, gs.received_total
    assert 15.0 < 100 * lost / (lost + rec) < 25.0


class FakeLink(Link):
    kind, port = "fake", "fake"

    def __init__(self):
        self.inbox = bytearray()

    def read(self):
        out = bytes(self.inbox)
        self.inbox.clear()
        return out

    def write(self, data):
        pass


def test_sequence_wrap_duplicates_and_reboots():
    link = FakeLink()
    gs = GroundStation(link, cal=CAL)
    for t, seq in enumerate([65534, 65535, 1, 1, 2, 40000, 40001]):   # 0 lost at wrap? 65535->1 skips 0
        link.inbox += P.Telemetry(seq=seq).encode()
        gs.step(t * 200)
    assert gs.lost_total == 1            # only seq 0 was missing; duplicate and the reboot jump not counted
    assert gs.received_total == 7


def test_crc_drops_counted_not_delivered():
    link = FakeLink()
    gs = GroundStation(link, cal=CAL)
    bad = bytearray(P.Telemetry(seq=5).encode())
    bad[10] ^= 0xFF
    link.inbox += bytes(bad) + P.Telemetry(seq=6).encode()
    gs.step(0)
    assert gs.received_total == 1 and gs.parser.dropped >= 1


# ---------------------------------------------------------------- commands

def test_cannot_arm_flight_or_without_link():
    gs = GroundStation(None, cal=CAL)
    with pytest.raises(RuntimeError):
        gs.arm(P.BENCH, 0)
    g, gs = rig()
    with pytest.raises(ValueError):
        gs.arm(P.FLIGHT, 0)


def test_bench_arms_through_firmware_and_moves_surfaces():
    g, gs = rig()
    run(gs, 0, 1000)
    gs.arm(P.BENCH, 1000)
    gs.set_servos([1700, 1300, 1800, 1600], 1000)
    run(gs, 1000, 1150)                               # < 5 frames: glider not armed yet
    assert not g.safety.s.armed and g.safety.s.servo_us == NEUTRAL
    run(gs, 1150, 1600)
    assert g.safety.s.armed and g.safety.s.mode == P.BENCH
    assert g.safety.s.servo_us == [1700, 1300, 1800, 1600]
    st = glider_report(gs)
    assert st["mode"] == P.BENCH and st["armed"] == 1


def test_ui_values_are_clamped_to_calibration_limits():
    g, gs = rig()
    gs.arm(P.BENCH, 0)
    assert gs.set_servos([5000, 0, 2500, 1500], 0) == [2300, 600, 2300, 1500]
    narrow = C.with_updates(CAL, **{"servos.elevator.max_us": 2000, "servos.elevator.sweep_max_us": 1950})
    gs.set_calibration(narrow)                        # e.g. a different servo fitted
    assert gs.session["servo_us"][2] == 2000
    assert gs.set_servos([1500, 1500, 2300, 1500], 1) == [1500, 1500, 2000, 1500]


def test_command_rate_never_exceeds_20hz():
    g, gs = rig()
    gs.arm(P.REPLAY, 0)
    sent0 = gs.commands_sent
    for t in range(0, 2000):                          # UI hammers set_servos every 1 ms (scrubbing)
        gs.set_servos([1500 + (t % 400)] * 4, t)
        gs.step(t)
    sent = gs.commands_sent - sent0
    assert 39 <= sent <= 41                           # 2 s at 20 Hz


def test_stall_does_not_cause_a_catch_up_burst():
    g, gs = rig()
    gs.arm(P.BENCH, 0)
    gs.step(0)
    n = gs.commands_sent
    gs.step(1000)                                     # thread stalled for 1 s
    gs.step(1001)
    assert gs.commands_sent - n == 1


def test_disarm_sends_disarmed_and_centres():
    g, gs = rig()
    gs.arm(P.BENCH, 0)
    gs.set_servos([2000] * 4, 0)
    run(gs, 0, 600)
    assert g.safety.s.armed
    gs.disarm(600)
    run(gs, 600, 700)
    assert not g.safety.s.armed and g.safety.s.servo_us == NEUTRAL
    assert gs.session is None and gs.last_end["reason"] == "requested"


def test_ui_idle_expiry():
    g, gs = rig()
    gs.arm(P.BENCH, 0)
    run(gs, 0, UI_IDLE_EXPIRY_MS - 1000, dt=50)
    assert gs.session is not None
    run(gs, UI_IDLE_EXPIRY_MS - 1000, UI_IDLE_EXPIRY_MS + 200, dt=50)
    assert gs.session is None and gs.last_end["reason"].startswith("expired")
    assert not g.safety.s.armed


def test_link_drop_times_out_in_firmware_then_rearms_after_recovery():
    g, gs = rig()
    gs.arm(P.BENCH, 0)
    gs.set_servos([1800] * 4, 0)
    run(gs, 0, 1000)
    assert g.safety.s.armed
    g.loss = 1.0                                      # link gone, both directions
    run(gs, 1000, 1200)
    assert not g.safety.s.armed and g.safety.s.timeout_disarmed
    assert g.safety.s.servo_us == NEUTRAL             # centred within the timeout
    g.loss = 0.0
    run(gs, 1200, 1210)
    assert not g.safety.s.armed                       # one frame is not enough
    run(gs, 1210, 1600)
    assert g.safety.s.armed                           # 5 consecutive frames re-arm


def test_cold_start_bench_still_works():
    """The ~2 m warm-up transient cannot make a cold bench system read airborne:
    there is no baro reference yet, so the baro does not vote."""
    g, gs = rig()
    run(gs, 0, 5000)                                  # 5 s after boot, baro still warming
    gs.arm(P.BENCH, 5000)
    run(gs, 5000, 5600)
    assert g.safety.s.armed and not g.airborne


def test_moving_glider_with_gps_fix_refuses_bench():
    g, gs = rig()
    run(gs, 0, 40_000, dt=50)                         # GPS 3D fix after 35 s
    g.set_flying(True)                                # 9 m/s ground speed
    gs.arm(P.BENCH, 40_000)
    gs.set_servos([2000] * 4, 40_000)
    run(gs, 40_000, 41_000)
    assert not g.safety.s.armed and g.safety.s.servo_us == NEUTRAL
    assert glider_report(gs)["cmd_rejected_airborne"] == 1 and glider_report(gs)["airborne"] == 1


def test_without_fix_or_reference_only_the_flight_latch_protects():
    """Stated plainly: with no GPS fix and no baro reference, a moving glider
    reads grounded. Real flights are always armed in FLIGHT first, and the
    60 s latch is what makes the firmware refuse BENCH/REPLAY then."""
    g, gs = rig()
    g.set_flying(True)
    gs.arm(P.BENCH, 0)
    run(gs, 0, 600)
    assert g.safety.s.armed                           # no vote: accepted
    gs.disarm(600)
    run(gs, 600, 800)
    g.arm_flight_locally(800)
    gs.arm(P.BENCH, 900)
    run(gs, 900, 1500)
    assert not g.safety.s.armed or g.safety.s.mode == P.FLIGHT
    assert glider_report(gs)["cmd_rejected_airborne"] == 1


# ---------------------------------------------------------------- sensors in telemetry

def test_rail_voltage_scaled_from_calibration_and_marked_indicative():
    g, gs = rig()                                     # divider fitted (GPIO34), scale not calibrated
    run(gs, 0, 2000)
    adc, v = gs.latest["rail_adc_v"], gs.latest["rail_v"]
    assert 2.4 < adc < 2.6 and v == pytest.approx(2.0 * adc)
    rail = gs.latest["rail"]
    assert rail["measured"] and not rail["calibrated"] and "indicative" in rail["note"] and rail["level"] == "ok"
    cal = C.with_updates(CAL, **{"rail.scale": 1.98, "rail.calibrated": True})
    gs.set_calibration(cal)
    run(gs, 2000, 3000)
    assert gs.latest["rail"]["note"] is None and gs.latest["rail_v"] == pytest.approx(1.98 * gs.latest["rail_adc_v"])


def test_rail_not_measured_without_divider():
    g, gs = rig(rail_sensor=False)
    run(gs, 0, 2000)
    assert gs.latest["rail_v"] is None and gs.latest["rail"]["level"] is None
    assert gs.latest["status"]["rail_v_valid"] == 0


def test_rail_brownout_levels():
    g, gs = rig()
    assert gs.rail_state(4.70)["level"] == "brownout"
    assert gs.rail_state(4.85)["level"] == "low"
    assert gs.rail_state(5.05)["level"] == "ok"


def test_pi_failsafe_visible_in_telemetry():
    g, gs = rig()
    run(gs, 0, 1000)
    g.arm_flight_locally(1000)
    run(gs, 1000, 2000)
    assert glider_report(gs)["pi_failsafe"] == 0
    g.pi_alive = False
    run(gs, 2000, 3000)
    assert glider_report(gs)["pi_failsafe"] == 1 and glider_report(gs)["mode"] == P.FLIGHT


def test_baro_warm_and_gps_fix_progress():
    g, gs = rig()
    run(gs, 0, 10_000, dt=50)
    assert gs.latest["status"]["baro_warm"] == 0 and gs.latest["status"]["gps_fix"] == 0
    run(gs, 10_000, 125_000, dt=50)
    assert gs.latest["status"]["baro_warm"] == 1 and gs.latest["status"]["gps_fix"] == 2
    assert gs.latest["status"]["baro_ref_captured"] == 0   # captured on arm, never at boot


def test_history_feeds_charts():
    g, gs = rig()
    run(gs, 0, 3000)
    h = list(gs.history)
    assert len(h) >= 14 and {"t_ms", "roll", "alt_baro", "loss_pct", "gps_sats"} <= set(h[0])


# ---------------------------------------------------------------- logs and Pi

def _sim_trajectory():
    d = os.path.join(ROOT, "data", "episodes")
    names = sorted(n for n in os.listdir(d) if n.endswith(".json") and not n.endswith(".summary.json")) \
        if os.path.isdir(d) else []
    if not names:
        pytest.skip("no recorded simulator flight to synthesize a log from")
    with open(os.path.join(d, names[0]), encoding="utf-8") as f:
        return json.load(f)


def test_log_round_trip_positions_and_bad_lines(tmp_path):
    traj = _sim_trajectory()
    header, records = logs.synthesize_from_trajectory(traj, (-33.93, 18.63))
    p = tmp_path / "x.rlglog"
    logs.write(str(p), header, records)
    text = p.read_text(encoding="utf-8").splitlines()
    text.insert(5, "{not json")
    text.insert(9, json.dumps({"t": 1.0}))
    parsed = logs.parse("\n".join(text))
    assert parsed["summary"]["skipped_lines"] == 2 and parsed["skipped"][0]["line"] == 6
    assert parsed["summary"]["n_frames"] == len(traj["frames"])
    for f, src in zip(parsed["frames"][::50], traj["frames"][::50]):
        assert abs(f["pos_ned"][0] - src["pos_ned"][0]) < 0.5
        assert abs(f["pos_ned"][1] - src["pos_ned"][1]) < 0.5
    assert "STUB" in parsed["header"]["source"]


def test_log_rejects_wrong_format_or_version():
    with pytest.raises(ValueError):
        logs.parse(json.dumps({"format": "csv"}))
    with pytest.raises(ValueError):
        logs.parse(json.dumps({"format": "rlglog", "version": 99}))


def test_log_names_are_whitelisted():
    assert safe_log_name("flight-2026-11-18_1.rlglog")
    for bad in ("../etc/passwd", "a/b.rlglog", ".hidden.rlglog", "x.txt", "..rlglog", "a b.rlglog"):
        with pytest.raises(ValueError):
            safe_log_name(bad)


def test_stub_pi(tmp_path):
    d = os.path.join(ROOT, "data", "episodes")
    pi = StubPi(str(tmp_path), d if os.path.isdir(d) else None)
    st = pi.status()
    assert st["reachable"] and st["disk_free_bytes"] > 0
    for entry in pi.list_logs():
        assert logs.parse(pi.fetch(entry["name"]).decode())["summary"]["n_frames"] > 0


def test_pi_server_over_http(tmp_path):
    from http.server import ThreadingHTTPServer
    from flighttest.pi_server import make_handler
    header, records = logs.synthesize_from_trajectory(
        {"episode_id": "t", "frames": [{"t": 0.05 * i, "pos_ned": [i, 0, -10], "euler": [0, 0, 0]}
                                       for i in range(40)]}, (-33.93, 18.63))
    logs.write(str(tmp_path / "f1.rlglog"), header, records)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(tmp_path)))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        c = PiClient(f"http://127.0.0.1:{srv.server_address[1]}")
        st = c.status()
        assert st["reachable"] and st["disk_total_bytes"] > 0
        assert [x["name"] for x in c.list_logs()] == ["f1.rlglog"]
        assert logs.parse(c.fetch("f1.rlglog").decode())["summary"]["n_frames"] == 40
        import urllib.error
        import urllib.request
        for path in ("/logs/..%2F..%2Fsecret.rlglog", "/logs/missing.rlglog"):
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(c.base_url + path, timeout=2)
    finally:
        srv.shutdown()


def test_unreachable_pi_reports_quickly():
    import time
    t = time.monotonic()
    st = PiClient("http://127.0.0.1:9", timeout_s=0.5).status()
    assert not st["reachable"] and "error" in st and time.monotonic() - t < 3


def test_constants():
    assert COMMAND_PERIOD_MS == 50
