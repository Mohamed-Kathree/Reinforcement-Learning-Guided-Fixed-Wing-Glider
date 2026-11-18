# firmware/

Shared protocol and safety code for the ESP32 flight controller.

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
