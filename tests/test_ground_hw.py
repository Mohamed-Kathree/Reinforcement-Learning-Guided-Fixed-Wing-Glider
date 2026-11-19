"""
test_ground_hw.py
=================
GroundStation features for the real flight computer (schema v3):
  - the bridge stream mixes console text and binary frames: text becomes
    console lines, frames are parsed, neither corrupts the other;
  - Info packets; calibration push (CalibrationSet bytes, confirmation only
    when the aircraft reports the same cal_id loaded from NVS, resend, give up,
    refused while armed/airborne);
  - console commands are whitelisted;
  - the calibration id matches the firmware header's CAL_HASH.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flighttest import calibration as C  # noqa: E402
from flighttest.ground import GroundStation, calibration_id  # noqa: E402
from flighttest.links import Link  # noqa: E402
from flighttest.protocol import packets as P  # noqa: E402

CAL = C.load()


class FakeLink(Link):
    kind = "fake"
    port = "fake"

    def __init__(self):
        self.inbox = bytearray()
        self.outbox = bytearray()

    def read(self):
        b = bytes(self.inbox)
        self.inbox.clear()
        return b

    def write(self, data):
        self.outbox += data


def info(cal_id, nvs=True, **kw):
    return P.Info(seq=kw.get("seq", 1), cal_id=cal_id, cal_flags=(4 if nvs else 0), schema=3, loop_hz=228,
                  pi_rx_gaps=0, pi_rx_bad=0, pi_tlm_gaps=0, gps_rate_dhz=50, launch_cause=0,
                  pitch_sp=0.0, roll_sp=0.0).encode()


def test_text_and_frames_are_split():
    link = FakeLink()
    gs = GroundStation(link, cal=CAL)
    stream = (b"S t=12 DISARMED rpy=1.1,0.2,0.0 | loop=228Hz\n" + P.Telemetry(seq=1).encode()
              + b"R imu ok=1\nR baro pa=101" + P.Telemetry(seq=2).encode() + b"347.5\n" + info(0x1234))
    link.inbox += stream[:30]
    gs.step(0)
    link.inbox += stream[30:]
    gs.step(10)
    texts = [c["text"] for c in gs.console_lines]
    assert texts == ["S t=12 DISARMED rpy=1.1,0.2,0.0 | loop=228Hz", "R imu ok=1", "R baro pa=101347.5"]
    assert gs.received_total == 2 and gs.crc_drops == 0
    assert gs.info["cal_id_hex"] == "00001234" and gs.info["loop_hz"] == 228
    snap = gs.snapshot(10)
    assert snap["telemetry"]["phase"] == "DISARMED" and snap["console"][-1]["text"] == "R baro pa=101347.5"


def test_calibration_packet_contents():
    cal = C.load()
    cal["servos"]["elevator"]["direction"] = -1
    link = FakeLink()
    gs = GroundStation(link, cal=cal)
    gs.push_calibration(0)
    pkt = P.CalibrationSet.decode(bytes(link.outbox))
    names = C.SERVO_NAMES
    assert pkt.cal_id == calibration_id(cal)
    assert pkt.dir_neg == 1 << names.index("elevator")
    assert pkt.neutral_us == [cal["servos"][n]["neutral_us"] for n in names]
    assert pkt.full_us == [cal["servos"][n]["us_full_deflection"] for n in names]
    assert (pkt.flags & 2) == (2 if cal["imu"]["verified"] else 0)


def test_calibration_push_confirm_resend_and_give_up():
    link = FakeLink()
    gs = GroundStation(link, cal=CAL)
    gs.push_calibration(0)
    assert gs.cal_push["sends"] == 1
    gs.step(1600)                                       # no confirmation -> resend
    assert gs.cal_push["sends"] == 2
    link.inbox += info(calibration_id(CAL), nvs=False)  # applied but not yet in NVS: not confirmed
    gs.step(1700)
    assert gs.cal_push["state"] == "sending"
    link.inbox += info(calibration_id(CAL), nvs=True)
    gs.step(1800)
    assert gs.cal_push["state"] == "confirmed"

    gs.push_calibration(10_000)
    for t in range(10_000, 17_000, 500):
        gs.step(t)
    # the aircraft still reports a stale Info from before -> it counts only if ids match; here they do,
    # so make it a different id to exercise the give-up path
    gs.info = None
    gs.push_calibration(20_000)
    for t in range(20_000, 27_000, 500):
        gs.step(t)
    assert gs.cal_push["state"] == "not confirmed" and gs.cal_push["sends"] == 3


def test_calibration_push_refused_when_armed_or_airborne():
    link = FakeLink()
    gs = GroundStation(link, cal=CAL)
    link.inbox += P.Telemetry(seq=1, status=P.pack_status(mode=P.FLIGHT, armed=1)).encode()
    gs.step(0)
    with pytest.raises(RuntimeError):
        gs.push_calibration(0)
    link.inbox += P.Telemetry(seq=2, status=P.pack_status(airborne=1)).encode()
    gs.step(10)
    with pytest.raises(RuntimeError):
        gs.push_calibration(10)
    assert link.outbox == b""


def test_console_whitelist():
    link = FakeLink()
    gs = GroundStation(link, cal=CAL)
    assert gs.console("  ARM ", 0) == "arm"
    assert bytes(link.outbox) == b"arm\n"
    for bad in ("rm -rf", "arm; disarm", "stream", "8", "srv"):
        with pytest.raises(ValueError):
            gs.console(bad, 0)
    assert bytes(link.outbox) == b"arm\n"
    assert gs.console_lines[-1]["text"] == "> arm"


def test_calibration_id_matches_firmware_header():
    src = open(os.path.join(ROOT, "firmware", "flight", "src", "generated", "calibration_gen.h")).read()
    h = re.search(r'#define CAL_HASH "([0-9a-f]+)"', src).group(1)
    raw = open(C.DEFAULT_PATH, "rb").read().replace(b"\r\n", b"\n")
    import json
    if raw.decode("utf-8") != json.dumps(C.load(), indent=1) + "\n":
        pytest.skip("calibration.json was not written by calibration.save(); ids are computed from that form")
    assert calibration_id(C.load()) == int(h[:8], 16)
