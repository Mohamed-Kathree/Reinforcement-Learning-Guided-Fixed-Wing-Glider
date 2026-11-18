/*
 * rlg_safety.c -- see rlg_safety.h for the rules, the two separate timeouts and
 * the hardware seams. UNTESTED ON HARDWARE. Mirrors flighttest/safety.py; both
 * run the same scenarios in tests/test_flighttest_safety.py.
 */
#include <string.h>
#include "rlg_safety.h"

static uint16_t clamp_abs(uint16_t v)
{
    if (v < RLG_ABS_MIN_US) return RLG_ABS_MIN_US;
    if (v > RLG_ABS_MAX_US) return RLG_ABS_MAX_US;
    return v;
}

static uint16_t clamp_us(const rlg_safety_t *st, int i, uint16_t v)
{
    if (v < st->min_us[i]) v = st->min_us[i];
    if (v > st->max_us[i]) v = st->max_us[i];
    return clamp_abs(v);
}

static void centre(rlg_safety_t *st)
{
    for (int i = 0; i < RLG_NUM_SERVOS; i++) st->servo_us[i] = clamp_us(st, i, st->neutral_us[i]);
}

static void disarm(rlg_safety_t *st, uint32_t now_ms)
{
    if (st->armed && st->mode == RLG_MODE_FLIGHT) {
        st->have_flight = true;
        st->last_flight_ms = now_ms;          /* the 60 s latch runs from leaving FLIGHT */
    }
    st->mode = RLG_MODE_DISARMED;
    st->armed = false;
    st->consecutive = 0;
    st->pi_failsafe = false;
    centre(st);
}

static bool ground_mode(uint8_t m) { return m == RLG_MODE_BENCH || m == RLG_MODE_REPLAY; }

void rlg_safety_init(rlg_safety_t *st, const uint16_t neutral_us[RLG_NUM_SERVOS], uint32_t now_ms)
{
    memset(st, 0, sizeof(*st));
    for (int i = 0; i < RLG_NUM_SERVOS; i++) {
        st->min_us[i] = RLG_ABS_MIN_US;
        st->max_us[i] = RLG_ABS_MAX_US;
        st->neutral_us[i] = neutral_us ? neutral_us[i] : 1500u;
        st->last_cmd_us[i] = st->neutral_us[i];
    }
    st->last_cmd_ms = now_ms;
    st->last_activity_ms = now_ms;
    st->last_pi_ms = now_ms;
    disarm(st, now_ms);
}

bool rlg_safety_set_limits(rlg_safety_t *st, const uint16_t min_us[RLG_NUM_SERVOS],
                           const uint16_t max_us[RLG_NUM_SERVOS])
{
    for (int i = 0; i < RLG_NUM_SERVOS; i++)
        if (min_us[i] > max_us[i]) return false;
    for (int i = 0; i < RLG_NUM_SERVOS; i++) {
        st->min_us[i] = clamp_abs(min_us[i]);
        st->max_us[i] = clamp_abs(max_us[i]);
    }
    if (!st->armed) centre(st);
    else for (int i = 0; i < RLG_NUM_SERVOS; i++) st->servo_us[i] = clamp_us(st, i, st->servo_us[i]);
    return true;
}

bool rlg_safety_is_airborne(rlg_safety_t *st, uint32_t now_ms, const rlg_sensors_t *s)
{
    bool baro = s->baro_alt_m == s->baro_alt_m && s->baro_alt_m > RLG_AIRBORNE_ALT_M;   /* NaN never votes */
    bool gps = s->gps_fix > 0 && s->gps_speed_ms > RLG_AIRBORNE_GROUND_SPEED_MS;
    bool latch = (st->armed && st->mode == RLG_MODE_FLIGHT)
              || (st->have_flight && (uint32_t)(now_ms - st->last_flight_ms) <= RLG_FLIGHT_LATCH_MS);
    st->airborne = baro || gps || latch;
    return st->airborne;
}

static void apply_servos(rlg_safety_t *st, const rlg_command_t *c, uint32_t now_ms)
{
    bool changed = false;
    for (int i = 0; i < RLG_NUM_SERVOS; i++) {
        if (c->servo_us[i] != st->last_cmd_us[i]) changed = true;
        st->last_cmd_us[i] = c->servo_us[i];
        st->servo_us[i] = clamp_us(st, i, c->servo_us[i]);
    }
    if (changed) st->last_activity_ms = now_ms;
}

rlg_cmd_result_t rlg_safety_on_command(rlg_safety_t *st, uint32_t now_ms, const rlg_sensors_t *sens,
                                       const uint8_t *frame, size_t len)
{
    rlg_command_t c;
    if (len != RLG_COMMAND_SIZE) return RLG_CMD_REJECTED_FRAME;
    memcpy(&c, frame, sizeof c);
    if (c.sync != RLG_SYNC || c.type != RLG_TYPE_COMMAND) return RLG_CMD_REJECTED_FRAME;
    if (rlg_crc8(frame + 1, RLG_COMMAND_SIZE - 2u) != c.crc8) return RLG_CMD_REJECTED_FRAME;
    if (c.mode > RLG_MODE_REPLAY) return RLG_CMD_REJECTED_FRAME;

    uint32_t gap = now_ms - st->last_cmd_ms;   /* unsigned: wrap-safe */
    if (st->have_seq && gap <= RLG_COMMAND_TIMEOUT_MS && (int16_t)(uint16_t)(c.seq - st->last_seq) <= 0)
        return RLG_CMD_REJECTED_STALE;
    st->have_seq = true;
    st->last_seq = c.seq;

    if (c.mode == RLG_MODE_DISARMED) {
        disarm(st, now_ms);
        st->last_cmd_ms = now_ms;
        return RLG_CMD_DISARMED;
    }

    bool airborne = rlg_safety_is_airborne(st, now_ms, sens);
    if (ground_mode(c.mode) && airborne) {
        st->rejected_airborne = true;
        if (ground_mode(st->mode)) disarm(st, now_ms);
        st->consecutive = 0;
        return RLG_CMD_REJECTED_AIRBORNE;
    }

    if (st->armed && c.mode == st->mode) {
        st->last_cmd_ms = now_ms;
        if (ground_mode(c.mode)) apply_servos(st, &c, now_ms);
        return RLG_CMD_ACCEPTED;
    }

    /* Not armed in this mode: count toward arming. Leaving an armed mode centres first. */
    if (st->armed) disarm(st, now_ms);
    if (c.mode == st->pending_mode && st->consecutive > 0 && gap <= RLG_COMMAND_TIMEOUT_MS)
        st->consecutive++;
    else {
        st->pending_mode = c.mode;
        st->consecutive = 1;
    }
    st->last_cmd_ms = now_ms;
    if (st->consecutive < RLG_REARM_CONSECUTIVE_FRAMES) return RLG_CMD_ARMING;

    st->mode = c.mode;
    st->armed = true;
    st->consecutive = 0;
    st->timeout_disarmed = false;
    st->rejected_airborne = false;
    st->last_activity_ms = now_ms;
    if (c.mode == RLG_MODE_FLIGHT) {
        st->have_flight = true;
        st->last_flight_ms = now_ms;
        st->pi_failsafe = true;            /* until the Pi has sent fresh setpoints */
        st->pi_consecutive = 0;
    }
    if (ground_mode(c.mode)) {
        for (int i = 0; i < RLG_NUM_SERVOS; i++) st->last_cmd_us[i] = c.servo_us[i];
        for (int i = 0; i < RLG_NUM_SERVOS; i++) st->servo_us[i] = clamp_us(st, i, c.servo_us[i]);
    }
    return RLG_CMD_ARMED;
}

void rlg_safety_on_pi_setpoint(rlg_safety_t *st, uint32_t now_ms)
{
    uint32_t gap = now_ms - st->last_pi_ms;
    st->last_pi_ms = now_ms;
    if (!st->pi_failsafe) return;
    st->pi_consecutive = (st->pi_consecutive > 0 && gap <= RLG_PI_TIMEOUT_MS) ? st->pi_consecutive + 1u : 1u;
    if (st->pi_consecutive >= RLG_PI_RESUME_FRAMES) {
        st->pi_failsafe = false;
        st->pi_consecutive = 0;
    }
}

void rlg_safety_tick(rlg_safety_t *st, uint32_t now_ms, const rlg_sensors_t *sens)
{
    if (st->armed && st->mode == RLG_MODE_FLIGHT) {
        st->last_flight_ms = now_ms;
        if (!st->pi_failsafe && (uint32_t)(now_ms - st->last_pi_ms) > RLG_PI_TIMEOUT_MS) {
            st->pi_failsafe = true;      /* Pi link timeout: failsafe glide. ESP-NOW is not checked here. */
            st->pi_consecutive = 0;
        }
    }
    if (sens) rlg_safety_is_airborne(st, now_ms, sens);

    uint32_t gap = now_ms - st->last_cmd_ms;
    if (!st->armed && st->consecutive > 0 && gap > RLG_COMMAND_TIMEOUT_MS) st->consecutive = 0;
    if (!st->armed || !ground_mode(st->mode)) return;
    if (gap > RLG_COMMAND_TIMEOUT_MS) {
        disarm(st, now_ms);
        st->timeout_disarmed = true;
        return;
    }
    if ((uint32_t)(now_ms - st->last_activity_ms) > RLG_ARM_IDLE_EXPIRY_MS) disarm(st, now_ms);
}

bool rlg_safety_pi_failsafe(const rlg_safety_t *st) { return st->pi_failsafe; }

bool rlg_safety_flight_output(rlg_safety_t *st, const uint16_t us[RLG_NUM_SERVOS])
{
    if (!st->armed || st->mode != RLG_MODE_FLIGHT) return false;
    for (int i = 0; i < RLG_NUM_SERVOS; i++) st->servo_us[i] = clamp_us(st, i, us[i]);
    return true;
}

uint32_t rlg_safety_status_bits(const rlg_safety_t *st)
{
    return ((uint32_t)st->mode << RLG_STATUS_MODE_LSB)
         | ((uint32_t)st->armed << RLG_STATUS_ARMED_LSB)
         | ((uint32_t)st->airborne << RLG_STATUS_AIRBORNE_LSB)
         | ((uint32_t)st->timeout_disarmed << RLG_STATUS_TIMEOUT_DISARMED_LSB)
         | ((uint32_t)st->rejected_airborne << RLG_STATUS_CMD_REJECTED_AIRBORNE_LSB)
         | ((uint32_t)st->pi_failsafe << RLG_STATUS_PI_FAILSAFE_LSB);
}

size_t rlg_safety_sizeof(void) { return sizeof(rlg_safety_t); }
uint8_t rlg_safety_mode(const rlg_safety_t *st) { return st->mode; }
bool rlg_safety_armed(const rlg_safety_t *st) { return st->armed; }
uint16_t rlg_safety_servo(const rlg_safety_t *st, int i) { return st->servo_us[i]; }
