/*
 * flight_phase.h -- launch and apex detection, pure C (host-tested).
 *
 * The policy was trained from the APEX of the launch zoom-climb (episode start:
 * ~22 m, 8-15 m/s, pitch 0-15 deg; env/glider_env.py _build_launch_state), so
 * the Pi takes over there, not at the throw:
 *
 *   DISARMED --FLIGHT armed--> ARMED --launch--> LAUNCH --apex--> AUTONOMOUS
 *
 *   launch: |a| > launch_accel_g continuously for launch_hold_ms (the throw).
 *   apex:   after apex_min_ms, the low-passed baro altitude has dropped
 *           apex_drop_m below its peak since launch (descent begun), or
 *           apex_timeout_ms after launch, whichever comes first.
 *   handoff_at_apex = false skips LAUNCH: AUTONOMOUS at the throw (the
 *   researcher's literal first choice; outside the training distribution).
 *
 * Disarming returns to DISARMED from any phase. Thresholds are FIRST GUESSES:
 * record a few real throws (the Info packet and the log carry the causes and
 * timings) and tune them before relying on them.
 */
#ifndef RLG_FLIGHT_PHASE_H
#define RLG_FLIGHT_PHASE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float launch_accel_g;      /* 2.5 g */
    uint32_t launch_hold_ms;   /* 20 ms */
    uint32_t apex_min_ms;      /* 400 ms after launch */
    float apex_drop_m;         /* 0.3 m below the filtered peak */
    uint32_t apex_timeout_ms;  /* 3000 ms after launch */
    float baro_tau_s;          /* 0.15 s low-pass on baro altitude */
    bool handoff_at_apex;      /* true */
} rlg_phase_cfg_t;

/* launch_cause: 0 none, 1 acceleration. apex_cause: 0 none, 1 baro descent,
 * 2 timeout, 3 immediate (handoff_at_apex off). */
typedef struct {
    rlg_phase_cfg_t cfg;
    uint8_t phase;              /* rlg_phase_t value */
    uint32_t t_launch_ms, t_apex_ms, accel_since_ms, last_ms;
    bool accel_hi, alt_init;
    float alt_f, alt_peak;
    uint8_t launch_cause, apex_cause;
    float peak_accel_g;
} rlg_flightphase_t;

rlg_phase_cfg_t rlg_phase_default_cfg(void);
void rlg_phase_init(rlg_flightphase_t *s, rlg_phase_cfg_t cfg);
/* Call every control-loop step. baro_alt_m may be NAN (no reference / no
 * reading): then only the timeout can end LAUNCH. Returns the phase. */
uint8_t rlg_phase_update(rlg_flightphase_t *s, uint32_t now_ms, bool flight_armed,
                         float accel_mag_g, float baro_alt_m);

#ifdef __cplusplus
}
#endif

#endif /* RLG_FLIGHT_PHASE_H */
