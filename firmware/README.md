# firmware/

Shared protocol and safety code for the ESP32 flight controller.

> **Status: host-tested only. None of this has run on an ESP32.**
> `tests/test_flighttest_protocol.py` and `tests/test_flighttest_safety.py` compile these
> files with the host `gcc` and check them against the Python side. They prove the logic,
> not the integration (timers, PWM, ESP-NOW callbacks, interrupt timing).

## Files

| File | What it is |
|---|---|
| `protocol/rlg_protocol.h` | **Generated** from `flighttest/protocol/schema.json` by `python -m flighttest.protocol.codegen`. Packed structs, compile-time size and offset checks, modes, servo limits, status bits. Do not edit by hand. |
| `protocol/rlg_crc8.c` | CRC8, polynomial 0x07, init 0x00 (check value 0xF4). |
| `protocol/rlg_safety.h/.c` | Command safety state machine. Rules, open design decisions and hardware seams are in the header. Mirrors `flighttest/safety.py`. |

## Integration seams (to do on hardware, one at a time)

1. **Add these files to the ESP32 project** (ESP-IDF or Arduino). Both compile with gcc;
   `_Static_assert` needs C11.
2. **ESP-NOW receive:** pass each received buffer to `rlg_safety_on_command()` with the
   current `millis()` and the airborne flag. Prefer a queue drained in the main loop over
   calling it inside the Wi-Fi callback.
3. **Main loop:** call `rlg_safety_tick(millis())` at least every 10 ms, then write
   `st.servo_us[]` to the LEDC PWM channels for GPIO 13, 14, 25, 27.
4. **Airborne flag:** barometric altitude above `RLG_AIRBORNE_ALT_M` relative to the ground
   reference, OR GPS ground speed above `RLG_AIRBORNE_GROUND_SPEED_MS`. Both thresholds are
   first guesses: measure baro noise while carried and GPS speed at rest before trusting them.
5. **Neutrals:** load per-servo neutral microseconds from NVS. 1500 µs is a placeholder.
6. **FLIGHT outputs:** the Pi's commands (UART) go through `rlg_safety_flight_output()`.
   That channel needs its own watchdog (no Pi command for N ms → baseline or centre). Not
   written yet.
7. **Telemetry:** fill `rlg_telemetry_t`, OR `rlg_safety_status_bits()` into `status`, set
   `crc8 = rlg_crc8(&t.type, RLG_TELEMETRY_SIZE - 2)`, send at 5 Hz.
8. **Rail voltage:** there is no divider to an ADC pin in the current wiring. Until one is
   fitted on an ADC1 pin (GPIO34, 36 or 39; ADC2 does not work with the radio on), send
   `rail_v = -1` and leave `rail_v_valid` clear.

## Verification on hardware (record every value, not just pass or fail)

- Bench, propeller-free, surfaces disconnected first: arm BENCH from the ground app; confirm
  one frame does not arm; confirm 5 frames do.
- Unplug the ground ESP32 while armed: surfaces centre within 150 ms (measure it).
- Hold the aircraft above the airborne altitude threshold (or walk with GPS fix): BENCH and
  REPLAY are refused.
- Leave BENCH armed and untouched: it disarms after 120 s.
