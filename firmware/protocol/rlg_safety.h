/*
 * rlg_safety.h -- firmware-side command safety for the control surfaces.
 *
 * UNTESTED ON HARDWARE. Host-tested only: tests/test_flighttest_safety.py
 * compiles this file with gcc and runs the same scenarios against it and
 * against the Python reference (flighttest/safety.py); the two must agree.
 *
 * Rules (flight-test brief, SAFETY section):
 *  - Commands carry a mode: DISARMED, FLIGHT, BENCH, REPLAY.
 *  - BENCH and REPLAY are refused while airborne (the caller decides
 *    "airborne" from baro altitude or GPS ground speed; see seams below).
 *  - BENCH/REPLAY: no valid command for RLG_COMMAND_TIMEOUT_MS (150 ms)
 *    -> surfaces centre, mode DISARMED.
 *  - Arming (any mode) needs RLG_REARM_CONSECUTIVE_FRAMES valid frames in a
 *    row requesting the same mode; one stray frame never arms.
 *  - BENCH/REPLAY disarm on their own after RLG_ARM_IDLE_EXPIRY_MS with no
 *    change in the commanded servo values.
 *  - Servo outputs are always clamped to RLG_SERVO_MIN/MAX_US.
 *
 * Design decisions not settled by the brief (review before flight):
 *  - FLIGHT has NO ground-command timeout: ESP-NOW is expected to drop
 *    packets in flight, and losing the ground link must not centre the
 *    surfaces mid-air. FLIGHT outputs come from the Pi via
 *    rlg_safety_flight_output(); that channel needs its own watchdog (seam).
 *  - Arming into FLIGHT is allowed while airborne (recovery after an
 *    in-air reboot); arming into BENCH/REPLAY never is.
 *  - A BENCH/REPLAY command received while airborne also disarms an armed
 *    BENCH/REPLAY session.
 *  - Sequence numbers must increase (u16, wrap-aware). After a gap longer than
 *    the timeout any sequence number is accepted, so a restarted ground
 *    station can resync.
 *
 * Hardware seams (not implemented here):
 *  - Call rlg_safety_on_command() from the ESP-NOW receive callback (or a
 *    queue drained in the main loop), and rlg_safety_tick() every loop
 *    iteration, at least every 10 ms.
 *  - Write st->servo_us[] to the PWM driver (LEDC) after every call.
 *  - Load neutral_us from NVS (calibration pushed by the ground app);
 *    1500 us is only a placeholder.
 *  - Provide `airborne` from the barometer (above RLG_AIRBORNE_ALT_M after
 *    the ground reference) OR GPS ground speed above
 *    RLG_AIRBORNE_GROUND_SPEED_MS. Both thresholds are first guesses.
 */
#ifndef RLG_SAFETY_H
#define RLG_SAFETY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "rlg_protocol.h"

typedef enum {
    RLG_CMD_ACCEPTED = 0,         /* armed in this mode; servo values applied (BENCH/REPLAY) */
    RLG_CMD_ARMING = 1,           /* valid frame counted toward arming; surfaces still centred */
    RLG_CMD_ARMED = 2,            /* this frame completed arming */
    RLG_CMD_DISARMED = 3,         /* DISARMED requested and applied */
    RLG_CMD_REJECTED_FRAME = 4,   /* wrong length, sync, type, CRC or mode value */
    RLG_CMD_REJECTED_STALE = 5,   /* sequence number not newer than the last one */
    RLG_CMD_REJECTED_AIRBORNE = 6 /* BENCH/REPLAY refused while airborne */
} rlg_cmd_result_t;

typedef struct {
    uint8_t mode;              /* rlg_mode_t */
    bool armed;
    uint8_t pending_mode;
    uint16_t consecutive;
    bool have_seq;
    uint16_t last_seq;
    uint32_t last_cmd_ms;
    uint32_t last_activity_ms;
    uint16_t neutral_us[RLG_NUM_SERVOS];
    uint16_t last_cmd_us[RLG_NUM_SERVOS];
    uint16_t servo_us[RLG_NUM_SERVOS];   /* write these to the PWM driver */
    bool timeout_disarmed;
    bool rejected_airborne;
} rlg_safety_t;

void rlg_safety_init(rlg_safety_t *st, const uint16_t neutral_us[RLG_NUM_SERVOS], uint32_t now_ms);
rlg_cmd_result_t rlg_safety_on_command(rlg_safety_t *st, uint32_t now_ms, bool airborne,
                                       const uint8_t *frame, size_t len);
void rlg_safety_tick(rlg_safety_t *st, uint32_t now_ms);
/* FLIGHT outputs from the Pi policy/baseline; ignored unless armed in FLIGHT. */
bool rlg_safety_flight_output(rlg_safety_t *st, const uint16_t us[RLG_NUM_SERVOS]);
/* Status bits owned by the safety layer (mode, armed, timeout, rejection). */
uint32_t rlg_safety_status_bits(const rlg_safety_t *st);

/* Accessors for host tests (ctypes). */
size_t rlg_safety_sizeof(void);
uint8_t rlg_safety_mode(const rlg_safety_t *st);
bool rlg_safety_armed(const rlg_safety_t *st);
uint16_t rlg_safety_servo(const rlg_safety_t *st, int i);

#endif /* RLG_SAFETY_H */
