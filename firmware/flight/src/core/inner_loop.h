/*
 * inner_loop.h -- the 200 Hz stabilisation law and safety shield, ported from
 * the simulator (env/glider_env.py AttitudeController.update() and
 * GliderEnv._safety_shield()). Pure C; tests/test_flight_firmware.py checks it
 * against the Python source, gains included.
 *
 * Input is an attitude SETPOINT (roll, pitch). In the simulator the pitch
 * target is computed from the policy's speed command (a trim schedule plus an
 * airspeed correction that needs TRUE airspeed); producing that target on the
 * real aircraft, which has no pitot, is a Phase 2 decision. See
 * firmware/flight/README.md.
 *
 * Surface sign conventions (simulator aero model, rlglider.xml):
 *   aileron  +  rolls RIGHT     (Cl_da = +0.15)
 *   elevator +  pitches nose DOWN (Cm_de < 0)
 *   rudder   +  yaws nose LEFT   (Cn_dr = -0.05)
 * Servo calibration (direction per servo) must reproduce exactly these.
 */
#ifndef RLG_INNER_LOOP_H
#define RLG_INNER_LOOP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Gains: identical to env/glider_env.py (checked by the host test). */
#define IL_KP_ROLL   2.0f
#define IL_KD_ROLL   0.3f
#define IL_KP_PITCH  3.0f
#define IL_KD_PITCH  0.4f
#define IL_KR_RUDDER 0.05f
/* Limits (radians) */
#define IL_AIL_MAX   0.43633231f   /* 25 deg */
#define IL_ELEV_MAX  0.34906585f   /* 20 deg */
#define IL_RUD_MAX   0.43633231f   /* 25 deg */
#define IL_PITCH_SP_MIN (-0.13962634f)   /* -8 deg: simulator pitch-target clip */
#define IL_PITCH_SP_MAX ( 0.20943951f)   /* +12 deg */
#define IL_SHIELD_BANK  0.78539816f      /* 45 deg */
#define IL_SHIELD_PITCH 0.34906585f      /* 20 deg, derated by 1/max(|cos roll|, 0.5) */

typedef struct {
    float aileron, elevator, rudder;     /* rad, simulator sign convention */
    uint8_t shield_bank, shield_pitch;   /* 1 if that shield acted this step */
} rlg_surfaces_t;

/* roll, pitch in rad; p, q, r in rad/s; setpoints in rad. */
rlg_surfaces_t rlg_inner_loop(float roll, float pitch, float p, float q, float r,
                              float roll_sp, float pitch_sp);

/* Surface deflection (rad) -> servo pulse (us): neutral + direction *
 * (defl / max_rad) * full_us, clamped to [min_us, max_us]. */
uint16_t rlg_deflection_to_us(float defl_rad, float max_rad, int neutral_us, int direction,
                              int full_us, int min_us, int max_us);

#ifdef __cplusplus
}
#endif

#endif /* RLG_INNER_LOOP_H */
