/*
 * rlg_safety.h -- firmware-side safety for the control surfaces.
 *
 * UNTESTED ON HARDWARE. Host-tested only: tests/test_flighttest_safety.py
 * compiles this file with gcc and runs the same scenarios against it and
 * against the Python reference (flighttest/safety.py); the two must agree.
 *
 * TWO LINKS, TWO SEPARATE TIMEOUTS -- do not conflate them:
 *  - ESP-NOW ground link (commands from the ground app): BENCH and REPLAY
 *    only. No valid command for RLG_COMMAND_TIMEOUT_MS (150 ms) -> surfaces
 *    centre, mode DISARMED. It NEVER applies in FLIGHT: in flight that link is
 *    monitoring only, and losing it must not affect flight.
 *  - Pi UART link (setpoints from the policy/baseline): FLIGHT only. No
 *    setpoint for RLG_PI_TIMEOUT_MS (150 ms) -> failsafe glide
 *    (rlg_safety_pi_failsafe() true; the attitude controller must then fly the
 *    failsafe setpoints). RLG_PI_RESUME_FRAMES fresh setpoints in a row end it.
 *
 * Other rules (flight-test brief and 18 Nov answers):
 *  - BENCH and REPLAY are refused while airborne. Airborne if ANY of:
 *      barometric altitude above launch > RLG_AIRBORNE_ALT_M (5 m), only once
 *        a ground reference exists (NaN = no reference, does not vote);
 *      GPS fix > 0 AND ground speed > RLG_AIRBORNE_GROUND_SPEED_MS (2 m/s);
 *        without a fix speed is meaningless and does not vote;
 *      armed in FLIGHT within the last RLG_FLIGHT_LATCH_MS (60 s); state, not
 *        a sensor reading, so drift cannot fool it.
 *    Errs toward airborne: a false "airborne" on the bench is an annoyance; a
 *    false "grounded" in flight could let a replay packet move a surface.
 *  - Arming (any mode) needs RLG_REARM_CONSECUTIVE_FRAMES (5) valid frames in a
 *    row requesting the same mode (250 ms at 20 Hz; rejects a link flap).
 *  - BENCH/REPLAY disarm on their own after RLG_ARM_IDLE_EXPIRY_MS with no
 *    change in the commanded servo values.
 *  - Outputs are clamped to the per-servo limits set by rlg_safety_set_limits()
 *    (from the calibration file), and always to RLG_ABS_MIN/MAX_US.
 *  - Sequence numbers must increase (u16, wrap-aware); after a gap longer than
 *    the ESP-NOW timeout any sequence number is accepted (ground restart).
 *
 * Hardware seams (not implemented here):
 *  - Call rlg_safety_on_command() for each ESP-NOW frame (prefer a queue drained
 *    in the main loop), rlg_safety_on_pi_setpoint() for each valid Pi UART
 *    setpoint, and rlg_safety_tick() every loop iteration (<= 10 ms).
 *  - Fill rlg_sensors_t from the barometer (altitude above the ground reference,
 *    NAN before it is captured on arming) and the GPS (ground speed, fix).
 *  - Write st->servo_us[] to the LEDC PWM channels after every call.
 *  - Load neutral_us and per-servo limits from NVS (pushed from the calibration
 *    file); 1500 us and the absolute envelope are placeholders until then.
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
    float baro_alt_m;     /* above the ground reference; NAN = no reference yet (does not vote) */
    float gps_speed_ms;   /* ground speed */
    uint8_t gps_fix;      /* 0 none, 1 2D, 2 3D; speed only votes when > 0 */
} rlg_sensors_t;

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
    uint16_t min_us[RLG_NUM_SERVOS];
    uint16_t max_us[RLG_NUM_SERVOS];
    uint16_t last_cmd_us[RLG_NUM_SERVOS];
    uint16_t servo_us[RLG_NUM_SERVOS];   /* write these to the PWM driver */
    bool timeout_disarmed;
    bool rejected_airborne;
    bool airborne;                       /* last computed airborne decision */
    bool have_flight;                    /* has been armed in FLIGHT since boot */
    uint32_t last_flight_ms;             /* last time armed in FLIGHT (for the 60 s latch) */
    bool pi_failsafe;                    /* FLIGHT: Pi setpoints stale -> failsafe glide */
    uint32_t last_pi_ms;
    uint16_t pi_consecutive;
} rlg_safety_t;

#ifdef __cplusplus
extern "C" {
#endif

void rlg_safety_init(rlg_safety_t *st, const uint16_t neutral_us[RLG_NUM_SERVOS], uint32_t now_ms);
/* Per-servo travel limits from the calibration file. Returns false (and changes
 * nothing) if any min > max; values are clamped to the absolute envelope. */
bool rlg_safety_set_limits(rlg_safety_t *st, const uint16_t min_us[RLG_NUM_SERVOS],
                           const uint16_t max_us[RLG_NUM_SERVOS]);
bool rlg_safety_is_airborne(rlg_safety_t *st, uint32_t now_ms, const rlg_sensors_t *sens);
rlg_cmd_result_t rlg_safety_on_command(rlg_safety_t *st, uint32_t now_ms, const rlg_sensors_t *sens,
                                       const uint8_t *frame, size_t len);
void rlg_safety_on_pi_setpoint(rlg_safety_t *st, uint32_t now_ms);
void rlg_safety_tick(rlg_safety_t *st, uint32_t now_ms, const rlg_sensors_t *sens);
bool rlg_safety_pi_failsafe(const rlg_safety_t *st);
/* Servo outputs computed by the attitude controller in FLIGHT (from Pi setpoints,
 * or failsafe setpoints); ignored unless armed in FLIGHT. */
bool rlg_safety_flight_output(rlg_safety_t *st, const uint16_t us[RLG_NUM_SERVOS]);
uint32_t rlg_safety_status_bits(const rlg_safety_t *st);

/* Accessors for host tests (ctypes). */
size_t rlg_safety_sizeof(void);
uint8_t rlg_safety_mode(const rlg_safety_t *st);
bool rlg_safety_armed(const rlg_safety_t *st);
uint16_t rlg_safety_servo(const rlg_safety_t *st, int i);

#ifdef __cplusplus
}
#endif

#endif /* RLG_SAFETY_H */
