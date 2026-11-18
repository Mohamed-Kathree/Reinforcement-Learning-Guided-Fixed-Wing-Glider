/*
 * rlg_safety.c -- see rlg_safety.h for the rules, decisions and hardware seams.
 * UNTESTED ON HARDWARE. Mirrors flighttest/safety.py; both run the same scenarios
 * in tests/test_flighttest_safety.py.
 */
#include <string.h>
#include "rlg_safety.h"

static uint16_t clamp_us(int i, uint16_t v)
{
    if (v < RLG_SERVO_MIN_US[i]) return RLG_SERVO_MIN_US[i];
    if (v > RLG_SERVO_MAX_US[i]) return RLG_SERVO_MAX_US[i];
    return v;
}

static void centre(rlg_safety_t *st)
{
    for (int i = 0; i < RLG_NUM_SERVOS; i++) st->servo_us[i] = clamp_us(i, st->neutral_us[i]);
}

static void disarm(rlg_safety_t *st)
{
    st->mode = RLG_MODE_DISARMED;
    st->armed = false;
    st->consecutive = 0;
    centre(st);
}

static bool ground_mode(uint8_t m) { return m == RLG_MODE_BENCH || m == RLG_MODE_REPLAY; }

void rlg_safety_init(rlg_safety_t *st, const uint16_t neutral_us[RLG_NUM_SERVOS], uint32_t now_ms)
{
    memset(st, 0, sizeof(*st));
    for (int i = 0; i < RLG_NUM_SERVOS; i++) {
        st->neutral_us[i] = neutral_us ? neutral_us[i] : 1500u;
        st->last_cmd_us[i] = st->neutral_us[i];
    }
    st->last_cmd_ms = now_ms;
    st->last_activity_ms = now_ms;
    disarm(st);
}

static void apply_servos(rlg_safety_t *st, const rlg_command_t *c, uint32_t now_ms)
{
    bool changed = false;
    for (int i = 0; i < RLG_NUM_SERVOS; i++) {
        if (c->servo_us[i] != st->last_cmd_us[i]) changed = true;
        st->last_cmd_us[i] = c->servo_us[i];
        st->servo_us[i] = clamp_us(i, c->servo_us[i]);
    }
    if (changed) st->last_activity_ms = now_ms;
}

rlg_cmd_result_t rlg_safety_on_command(rlg_safety_t *st, uint32_t now_ms, bool airborne,
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
        disarm(st);
        st->last_cmd_ms = now_ms;
        return RLG_CMD_DISARMED;
    }

    if (ground_mode(c.mode) && airborne) {
        st->rejected_airborne = true;
        if (ground_mode(st->mode)) disarm(st);
        st->consecutive = 0;
        return RLG_CMD_REJECTED_AIRBORNE;
    }

    if (st->armed && c.mode == st->mode) {
        st->last_cmd_ms = now_ms;
        if (ground_mode(c.mode)) apply_servos(st, &c, now_ms);
        return RLG_CMD_ACCEPTED;
    }

    /* Not armed in this mode: count toward arming. Leaving an armed mode centres first. */
    if (st->armed) disarm(st);
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
    if (ground_mode(c.mode)) {
        for (int i = 0; i < RLG_NUM_SERVOS; i++) st->last_cmd_us[i] = c.servo_us[i];
        for (int i = 0; i < RLG_NUM_SERVOS; i++) st->servo_us[i] = clamp_us(i, c.servo_us[i]);
    }
    return RLG_CMD_ARMED;
}

void rlg_safety_tick(rlg_safety_t *st, uint32_t now_ms)
{
    uint32_t gap = now_ms - st->last_cmd_ms;
    if (!st->armed && st->consecutive > 0 && gap > RLG_COMMAND_TIMEOUT_MS) st->consecutive = 0;
    if (!st->armed || !ground_mode(st->mode)) return;
    if (gap > RLG_COMMAND_TIMEOUT_MS) {
        disarm(st);
        st->timeout_disarmed = true;
        return;
    }
    if ((uint32_t)(now_ms - st->last_activity_ms) > RLG_ARM_IDLE_EXPIRY_MS) disarm(st);
}

bool rlg_safety_flight_output(rlg_safety_t *st, const uint16_t us[RLG_NUM_SERVOS])
{
    if (!st->armed || st->mode != RLG_MODE_FLIGHT) return false;
    for (int i = 0; i < RLG_NUM_SERVOS; i++) st->servo_us[i] = clamp_us(i, us[i]);
    return true;
}

uint32_t rlg_safety_status_bits(const rlg_safety_t *st)
{
    return ((uint32_t)st->mode << RLG_STATUS_MODE_LSB)
         | ((uint32_t)st->armed << RLG_STATUS_ARMED_LSB)
         | ((uint32_t)st->timeout_disarmed << RLG_STATUS_TIMEOUT_DISARMED_LSB)
         | ((uint32_t)st->rejected_airborne << RLG_STATUS_CMD_REJECTED_AIRBORNE_LSB);
}

size_t rlg_safety_sizeof(void) { return sizeof(rlg_safety_t); }
uint8_t rlg_safety_mode(const rlg_safety_t *st) { return st->mode; }
bool rlg_safety_armed(const rlg_safety_t *st) { return st->armed; }
uint16_t rlg_safety_servo(const rlg_safety_t *st, int i) { return st->servo_us[i]; }
