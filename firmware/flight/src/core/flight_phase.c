/* flight_phase.c -- see flight_phase.h. */
#include <math.h>
#include <string.h>
#include "flight_phase.h"
#include "../protocol/rlg_protocol.h"

rlg_phase_cfg_t rlg_phase_default_cfg(void)
{
    rlg_phase_cfg_t c;
    c.launch_accel_g = 2.5f;
    c.launch_hold_ms = 20;
    c.apex_min_ms = 400;
    c.apex_drop_m = 0.3f;
    c.apex_timeout_ms = 3000;
    c.baro_tau_s = 0.15f;
    c.handoff_at_apex = true;
    return c;
}

void rlg_phase_init(rlg_flightphase_t *s, rlg_phase_cfg_t cfg)
{
    memset(s, 0, sizeof(*s));
    s->cfg = cfg;
    s->phase = RLG_PHASE_DISARMED;
}

static void to_autonomous(rlg_flightphase_t *s, uint32_t now_ms, uint8_t cause)
{
    s->phase = RLG_PHASE_AUTONOMOUS;
    s->apex_cause = cause;
    s->t_apex_ms = now_ms;
}

uint8_t rlg_phase_update(rlg_flightphase_t *s, uint32_t now_ms, bool flight_armed,
                         float accel_mag_g, float baro_alt_m)
{
    if (!flight_armed) {
        if (s->phase != RLG_PHASE_DISARMED) rlg_phase_init(s, s->cfg);
        s->last_ms = now_ms;
        return s->phase;
    }
    float dt = (s->last_ms && now_ms > s->last_ms) ? (now_ms - s->last_ms) * 1e-3f : 0.0f;
    s->last_ms = now_ms;

    switch (s->phase) {
    case RLG_PHASE_DISARMED:
        s->phase = RLG_PHASE_ARMED;
        s->accel_hi = false;
        break;

    case RLG_PHASE_ARMED:
        if (accel_mag_g > s->cfg.launch_accel_g) {
            if (accel_mag_g > s->peak_accel_g) s->peak_accel_g = accel_mag_g;
            if (!s->accel_hi) { s->accel_hi = true; s->accel_since_ms = now_ms; }
            else if (now_ms - s->accel_since_ms >= s->cfg.launch_hold_ms) {
                s->phase = RLG_PHASE_LAUNCH;
                s->launch_cause = 1;
                s->t_launch_ms = now_ms;
                s->alt_init = false;
                if (!s->cfg.handoff_at_apex) to_autonomous(s, now_ms, 3);
            }
        } else {
            s->accel_hi = false;
            s->peak_accel_g = 0.0f;
        }
        break;

    case RLG_PHASE_LAUNCH: {
        if (!isnan(baro_alt_m)) {
            if (!s->alt_init) { s->alt_f = s->alt_peak = baro_alt_m; s->alt_init = true; }
            else {
                float a = dt / (s->cfg.baro_tau_s + dt);
                s->alt_f += a * (baro_alt_m - s->alt_f);
                if (s->alt_f > s->alt_peak) s->alt_peak = s->alt_f;
            }
        }
        uint32_t since = now_ms - s->t_launch_ms;
        if (since >= s->cfg.apex_min_ms && s->alt_init && s->alt_f < s->alt_peak - s->cfg.apex_drop_m)
            to_autonomous(s, now_ms, 1);
        else if (since >= s->cfg.apex_timeout_ms)
            to_autonomous(s, now_ms, 2);
        break;
    }

    default:                                   /* AUTONOMOUS: until disarmed */
        break;
    }
    return s->phase;
}
