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
| FT-2 Backend | serial transport and stub glider, connection status, WebSocket telemetry, command sender (20 Hz cap), Pi log listing and retrieval, log parser, calibration endpoints | **Done, stub-tested** |
| FT-3 Panels | Connection, control-surface tester, live telemetry, replay with servo playback (shares the existing replay components) | **Done, stub-tested** (dashboard: Ground Station, Flight Logs) |
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
| `firmware_sync.py` | `python -m flighttest.firmware_sync [--check]`: copies `firmware/protocol/` into the flight sketch and generates `firmware/flight/src/generated/calibration_gen.h` from `calibration.json` (servo `direction` is relative to the simulator's positive deflection; `us_full_deflection` = µs at the simulator's maximum deflection). Re-run and re-flash after every calibration change |
| `stub_glider.py` | STUB aircraft + ground radio on the real byte protocol: firmware safety, lossy 5 Hz telemetry, baro warm-up/drift, GPS fix, rail voltage "not measured" |
| `links.py` | `StubLink`; `SerialLink` (HARDWARE SEAM, untested; assumes a transparent ESP-NOW/USB bridge) |
| `ground.py` | `GroundStation`: link stats (rate, loss from sequence gaps, CRC drops, age), telemetry history, command sender (explicit arming, 20 Hz cap, UI-idle expiry, DISARMED burst) |
| `logs.py` | `.rlglog` flight-log format (JSON Lines, defined here; the Pi logger must write it), parser to replay frames, stub log synthesis |
| `pi.py` | `PiClient` (HARDWARE SEAM, untested) and `StubPi` |
| `pi_server.py` | Runs on the Pi (stdlib only, read-only log server). Untested on the Pi |

## API (backend/routers/flighttest.py)

`GET /api/ft/status` (ground link and Pi reported separately), `POST /api/ft/connect?transport=stub|serial`,
`/disconnect`, `/arm?mode=bench|replay`, `/disarm`, `/servos`, `/keepalive`, `GET|PUT /api/ft/calibration`,
`POST /api/ft/calibration/push` (send the saved calibration to the aircraft; confirmed via its Info packet), `POST /api/ft/console` (whitelisted flight-computer commands, e.g. `arm` = FLIGHT),
`POST /api/ft/stub` (stub only: airborne, loss, rail_sensor), `GET /api/ft/pi/logs`,
`POST /api/ft/pi/logs/{name}/fetch`, `GET /api/ft/logs[/{name}]`, `WS /ws/ft`.

## Tests

```bash
pytest tests/test_flighttest_protocol.py tests/test_flighttest_safety.py tests/test_flighttest_calibration.py tests/test_flighttest_ground.py
```

The C side is compiled with the host `gcc` (skipped, with a message, if none is installed).
Every safety scenario runs against both implementations, and a randomised differential test
requires them to agree after every step.
