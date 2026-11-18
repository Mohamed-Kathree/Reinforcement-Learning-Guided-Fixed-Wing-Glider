# flighttest/

Ground-side code for physical flight testing. Built in phases with an approval gate between
each; no hardware has been available, so everything is built against stubs with the hardware
integration points marked.

```
GLIDER                                    GROUND
  ESP32 (flight)  --ESP-NOW 5 Hz-->  ESP32 (receiver) --USB serial--> FastAPI
  Pi 4B           <------- WiFi, ground only ------------------------> FastAPI
```

In flight only ESP-NOW is used; the full-rate log is written to the Pi's SD card and
retrieved over WiFi on the ground. The stream is for monitoring; the SD log is the record.

## Phases

| Phase | Contents | Status |
|---|---|---|
| FT-1 Protocol foundation | `protocol/schema.json` (single source), generated Python and C (`codegen.py`), CRC8 with test vectors, stream parser, calibration file and validator, safety state machine in Python and C | **Done, host-tested** |
| FT-2 Backend | serial transport and stub glider, connection status, WebSocket telemetry, command sender (20 Hz cap), Pi log listing and retrieval, log parser, calibration endpoints | Awaiting approval |
| FT-3 Panels | Connection, control-surface tester, live telemetry, replay with servo playback (shares the existing replay components) | Awaiting FT-2 |
| FT-4 Hardware bring-up | firmware integration (`firmware/README.md`), measurements, IMU sign derivation | Needs the aircraft |

## Files

| File | Purpose |
|---|---|
| `protocol/schema.json` | Packets, modes, servo channels and limits, status bits, safety constants |
| `protocol/codegen.py` | `python -m flighttest.protocol.codegen [--check]` |
| `protocol/packets.py` | Generated: Telemetry, Command, CRC8, StreamParser |
| `protocol/test_vectors.json` | Generated sample packets and CRCs |
| `safety.py` | Python reference of `firmware/protocol/rlg_safety.c` (used by the stub glider) |
| `calibration.py`, `config/calibration.json` | Stored servo neutrals and directions, IMU transform (det +1 enforced), baro warm-up |

## Tests

```bash
pytest tests/test_flighttest_protocol.py tests/test_flighttest_safety.py tests/test_flighttest_calibration.py
```

The C side is compiled with the host `gcc` (skipped, with a message, if none is installed).
Every safety scenario runs against both implementations, and a randomised differential test
requires them to agree after every step.
