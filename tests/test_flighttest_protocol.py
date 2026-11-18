"""
test_flighttest_protocol.py
===========================
The glider <-> ground packet protocol (flighttest/protocol/schema.json).

  - generated Python, C header and vectors are up to date with the schema;
  - CRC8 (poly 0x07) matches the standard check value and the vectors, in
    Python and in the compiled C;
  - the C structs have exactly the Python layout (sizes, offsets via the
    header's static asserts, and field-by-field decoding of the same bytes);
  - the stream parser hunts sync, drops corrupt frames silently, resyncs, and
    never blocks on a partial packet.
"""

from __future__ import annotations

import json
import os
import random
import struct
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "c"))

from flighttest.protocol import codegen  # noqa: E402
from flighttest.protocol import packets as P  # noqa: E402
import host_lib  # noqa: E402

VECTORS = json.load(open(os.path.join(ROOT, "flighttest", "protocol", "test_vectors.json")))
needs_cc = pytest.mark.skipif(host_lib.compiler() is None, reason="no C compiler: C side not checked")


def test_generated_files_match_schema():
    assert codegen.main(["--check"]) == 0, "run: python -m flighttest.protocol.codegen"


def test_crc8_check_value_and_vectors():
    assert P.crc8(b"123456789") == 0xF4
    for v in VECTORS["crc8"]:
        data = v["ascii"].encode() if "ascii" in v else bytes.fromhex(v["hex"])
        assert P.crc8(data) == v["crc"]


def test_sizes_fit_espnow_and_have_no_padding():
    assert P.Telemetry.SIZE == 61 and P.Command.SIZE == 14
    for cls in (P.Telemetry, P.Command):
        assert struct.calcsize(cls.FMT) == cls.SIZE <= 250


def test_round_trip_and_vectors():
    for v in VECTORS["packets"]:
        cls = getattr(P, v["name"])
        raw = bytes.fromhex(v["hex"])
        pkt = cls.decode(raw)
        assert pkt.encode() == raw
        for k, want in v["fields"].items():
            got = getattr(pkt, k)
            if isinstance(want, float):
                assert got == pytest.approx(want, rel=1e-6)
            else:
                assert got == want
    t = P.Telemetry(seq=1, sonar=-1.0, rail_v=-1.0)
    assert P.Telemetry.decode(t.encode()).sonar == -1.0


def test_decode_rejects_bad_frames():
    raw = bytearray(P.Command(seq=1, mode=P.BENCH, servo_us=[1500] * 4).encode())
    for bad in (raw[:-1], bytes([0x00]) + raw[1:], raw[:1] + bytes([9]) + raw[2:]):
        with pytest.raises(ValueError):
            P.Command.decode(bytes(bad))
    raw[5] ^= 0x01
    with pytest.raises(ValueError):
        P.Command.decode(bytes(raw))


def test_status_bits_round_trip_and_do_not_overlap():
    s = P.pack_status(mode=P.REPLAY, armed=1, airborne=0, sonar_valid=1, gps_fix=2, gps_sats=40,
                      baro_ref_captured=1, baro_warm=1, rail_v_valid=0, timeout_disarmed=1, cmd_rejected_airborne=1)
    u = P.unpack_status(s)
    assert u["mode"] == P.REPLAY and u["gps_fix"] == 2 and u["gps_sats"] == 31  # saturates at 5 bits
    assert u["timeout_disarmed"] == 1 and u["rail_v_valid"] == 0


# ---------------------------------------------------------------- stream parser

def _stream(n=20, seed=0):
    rng = random.Random(seed)
    pkts = []
    for i in range(n):
        if rng.random() < 0.5:
            pkts.append(P.Command(seq=i, mode=rng.choice([0, 1, 2, 3]),
                                  servo_us=[rng.randrange(600, 2300) for _ in range(4)]))
        else:
            pkts.append(P.Telemetry(seq=i, roll=rng.uniform(-1, 1), alt_baro=rng.uniform(0, 30),
                                    status=rng.randrange(0, 1 << 18)))
    return pkts


def test_parser_whole_and_byte_by_byte():
    pkts = _stream()
    data = b"".join(p.encode() for p in pkts)
    sp = P.StreamParser()
    assert [p.encode() for p in sp.feed(data)] == [p.encode() for p in pkts]
    sp = P.StreamParser()
    out = []
    for b in data:
        out += sp.feed(bytes([b]))      # never blocks; partial frames wait
    assert [p.encode() for p in out] == [p.encode() for p in pkts]


def test_parser_drops_garbage_and_corruption_then_resyncs():
    pkts = _stream(30, seed=3)
    rng = random.Random(7)
    data = bytearray()
    good = []
    for i, p in enumerate(pkts):
        data += bytes(rng.randrange(256) for _ in range(rng.randrange(0, 6)))   # line noise, may contain 0xA5
        frame = bytearray(p.encode())
        if i % 7 == 3:
            frame[rng.randrange(1, len(frame))] ^= 1 << rng.randrange(8)       # corrupt this one
        else:
            good.append(p.encode())
        data += frame
    sp = P.StreamParser()
    got = [p.encode() for p in sp.feed(bytes(data))]
    # Every intact frame is recovered; noise can at worst cost frames that a
    # corrupt neighbour overlapped, never produce a false packet.
    assert set(got) <= set(p.encode() for p in pkts)
    assert len(set(got) & set(good)) >= len(good) - 2
    assert sp.dropped > 0


def test_parser_sync_byte_inside_payload():
    p = P.Command(seq=0xA5A5, mode=P.BENCH, servo_us=[0xA5A5 % 2300, 1500, 1500, 1500])
    assert bytes([P.SYNC]) in p.encode()[1:]
    sp = P.StreamParser()
    assert [x.encode() for x in sp.feed(b"\xa5\xa5" + p.encode())] == [p.encode()]


# ---------------------------------------------------------------- C side

@needs_cc
def test_c_crc_matches_python():
    lib = host_lib.load()
    rng = random.Random(1)
    cases = [b"123456789", b"", b"\x00", b"\xff"] + [bytes(rng.randrange(256) for _ in range(rng.randrange(1, 64)))
                                                     for _ in range(200)]
    for data in cases:
        assert lib.rlg_crc8(host_lib.buf(data), len(data)) == P.crc8(data)


@needs_cc
def test_c_struct_layout_matches_python():
    import ctypes
    lib = host_lib.load()
    assert lib.rlg_test_telemetry_size() == P.Telemetry.SIZE
    assert lib.rlg_test_command_size() == P.Command.SIZE
    for v in VECTORS["packets"]:
        raw = bytes.fromhex(v["hex"])
        pkt = getattr(P, v["name"]).decode(raw)
        if v["name"] == "Telemetry":
            out = (ctypes.c_double * 16)()
            lib.rlg_test_decode_telemetry(host_lib.buf(raw), out)
            want = [pkt.seq, pkt.roll, pkt.pitch, pkt.yaw, pkt.p, pkt.q, pkt.r, pkt.alt_baro, pkt.sonar,
                    pkt.lat, pkt.lon, pkt.vn, pkt.ve, pkt.rail_v, pkt.status, raw[-1]]
            assert list(out) == pytest.approx(want, rel=0, abs=0)
        else:
            out = (ctypes.c_uint32 * 7)()
            lib.rlg_test_decode_command(host_lib.buf(raw), out)
            assert list(out) == [pkt.seq, pkt.mode, *pkt.servo_us, raw[-1]]
