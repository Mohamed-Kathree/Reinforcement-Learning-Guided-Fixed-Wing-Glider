# firmware/

Shared protocol and safety code for the ESP32 flight controller, plus the sketches.

| Folder | What it is |
|---|---|
| `flight/` | **Production flight firmware.** Phase 1 (inner loop, no Pi): compiles, host-tested, not yet run on hardware. Build, decisions and the bench test plan: [flight/README.md](flight/README.md). |
| `bench/` | The bench test menu (from `Hardware/PCB_test_code/Flight_Controller`), kept as the hardware regression suite. IMU signs and formulas corrected on 19 Nov 2026. |
| `ground_station/` | The ESP-NOW ↔ USB bridge (from `Hardware/PCB_test_code`). It forwards USB **one byte per packet**, which needs fixing before binary commands go over the air. |
| `protocol/` | Shared protocol and safety code (below). Copied into `flight/src/protocol/` by `python -m flighttest.firmware_sync`. |
| `tools/` | `glider_viz.py` (corrected frame), `attitude_ref.py` (Python reference filter), `compare_attitude.py` (firmware vs visualiser measurement). |

## Phase 2 (Pi policy interface): contract mismatches to resolve first

The trained policy's contract is defined by `env/glider_env.py`. These are the
differences between that contract and the Pi link described in the flight brief:

1. **The policy does not output attitude setpoints.** Its action is
   [bank angle ±45°, airspeed command 9.5–16 m/s]. The simulator turns the speed command
   into a pitch target using **true airspeed**: `−KV_PITCH·(V_cmd − V_actual)`. The
   aircraft has no pitot tube. Either the Pi computes the pitch target without that term
   (and the simulator is retrained the same way), or a ground-speed-based estimate is
   modelled in the simulator first.
2. **The policy has no rudder output.** In the simulator, the rudder is the inner loop's
   `−KR·p` turn coordination. A rudder field from the Pi would be unused, or would differ
   from training.
3. **GPS rate.** The simulator's GPS runs at 5 Hz; the NEO-6M defaults to 1 Hz. It can be
   set to 5 Hz with UBX CFG-RATE.
4. **Vertical speed.** obs[8] is GPS vertical velocity. NMEA GGA and RMC do not carry it.
   The options are UBX NAV-VELNED, a value derived from the barometer (which then needs
   modelling in the simulator), or GPS altitude differences.
5. **The telemetry packet** (`rlg_telemetry_t`) has no vertical speed, ground speed or
   course field. The Pi link's telemetry packet must carry everything the 12-element
   observation needs.

## protocol/

> **Status: host-tested only. None of this has run on an ESP32.**
> `tests/test_flighttest_protocol.py` and `tests/test_flighttest_safety.py` compile these
> files with the host `gcc` and check them against the Python side. They prove the logic,
> not the integration (timers, PWM, ESP-NOW callbacks, UART, interrupt timing).

## Files

| File | What it is |
|---|---|
| `protocol/rlg_protocol.h` | **Generated** from `flighttest/protocol/schema.json` (v2) by `python -m flighttest.protocol.codegen`. Packed structs, compile-time size and offset checks, modes, the absolute servo envelope, status bits, safety constants. Do not edit by hand. |
| `protocol/rlg_crc8.c` | CRC8, polynomial 0x07, init 0x00 (check value 0xF4). |
| `protocol/rlg_safety.h/.c` | Safety state machine. Rules, the two separate timeouts and the seams are in the header. Mirrors `flighttest/safety.py`. |

## Two links, two timeouts (do not conflate)

| Link | Applies in | Timeout | Effect |
|---|---|---|---|
| ESP-NOW ground link (commands) | BENCH, REPLAY | 150 ms | surfaces centre, DISARMED |
| ESP-NOW ground link | FLIGHT | **none** | monitoring only; losing it must not affect flight |
| Pi UART (setpoints) | FLIGHT | 150 ms | failsafe glide; 5 fresh setpoints in a row end it |

## Integration seams (to do on hardware, one at a time)

`flight/` now implements seams 1, 2, 4, 5, 7 and 8, all unverified on hardware. Seam 3
(the Pi UART) is Phase 2. Seam 6 uses a header generated from the calibration file at
build time (`flight/src/generated/calibration_gen.h`) instead of NVS. The rail reading
uses `analogReadMilliVolts()`, which applies the eFuse ADC calibration.

1. **Add these files to the ESP32 project** (ESP-IDF or Arduino). Both compile with gcc;
   `_Static_assert` needs C11.
2. **ESP-NOW receive:** pass each frame to `rlg_safety_on_command()` with `millis()` and the
   current `rlg_sensors_t`. Prefer a queue drained in the main loop over calling it inside
   the Wi-Fi callback.
3. **Pi UART:** call `rlg_safety_on_pi_setpoint()` for every valid setpoint. While
   `rlg_safety_pi_failsafe()` is true, the attitude controller must fly the failsafe glide
   setpoints (wings level, trim speed) instead of the Pi's. Pass its servo outputs through
   `rlg_safety_flight_output()`.
4. **Main loop:** call `rlg_safety_tick(millis(), &sensors)` at least every 10 ms, then write
   `st.servo_us[]` to the LEDC PWM channels for GPIO 13, 14, 25, 27.
5. **Sensors (`rlg_sensors_t`):** `baro_alt_m` = altitude above the ground reference, `NAN`
   until the reference is captured (on arming, after the 2-minute warm-up; never at boot);
   `gps_speed_ms` and `gps_fix` (0 none, 1 2D, 2 3D) from the NEO-6M. The safety layer decides
   "airborne" itself: baro > 5 m, OR GPS speed > 2 m/s with a fix, OR armed in FLIGHT within
   the last 60 s.
6. **Calibration in NVS:** neutrals and per-servo travel limits come from
   `flighttest/config/calibration.json`, pushed by the ground app (push not yet written).
   Load them at boot and call `rlg_safety_set_limits()`. Until then: 1500 µs neutrals and the
   absolute envelope 600–2300 µs.
7. **Telemetry:** fill `rlg_telemetry_t`, OR `rlg_safety_status_bits()` into `status`, set
   `crc8 = rlg_crc8(&t.type, RLG_TELEMETRY_SIZE - 2)`, send at 5 Hz.
8. **Rail voltage (GPIO34, ADC1_CH6):** read with the ESP-IDF ADC calibration API
   (`adc_cali_*` / eFuse Vref), 11 dB attenuation, and send the **pin voltage** in
   `rail_adc_v` with `rail_v_valid` set. The ground app applies `rail.scale` and `offset_v`
   from the calibration file; do not hard-code the divider ratio here.

## Verification on hardware (record every value, not just pass or fail)

- Bench, surfaces disconnected first: arm BENCH from the ground app; one frame does not arm,
  five do.
- Unplug the ground ESP32 while armed in BENCH: surfaces centre within 150 ms (measure it).
- In FLIGHT on the bench, stop the Pi's setpoints: failsafe within 150 ms (measure it); restart
  them: failsafe clears after 5 setpoints. Unplug the ground ESP32 in FLIGHT: nothing changes.
- Cold-start the system and immediately arm BENCH: it must work (the baro cannot vote yet).
- Disarm from FLIGHT, then try BENCH: refused for 60 s, then accepted.
- Walk with a GPS fix above 2 m/s: BENCH and REPLAY refused.
- Leave BENCH armed and untouched: it disarms after 120 s.
- Rail: compare `rail_adc_v` × scale with a multimeter at idle and under full servo load.
