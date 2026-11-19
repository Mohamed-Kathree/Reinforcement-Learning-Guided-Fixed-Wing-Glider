/* inner_loop.c -- see inner_loop.h. */
#include <math.h>
#include "inner_loop.h"

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

static float wrap_pi(float a)
{
    while (a > 3.14159265f) a -= 6.28318531f;
    while (a < -3.14159265f) a += 6.28318531f;
    return a;
}

rlg_surfaces_t rlg_inner_loop(float roll, float pitch, float p, float q, float r,
                              float roll_sp, float pitch_sp)
{
    rlg_surfaces_t s = {0};

    /* AttitudeController.update() */
    float roll_err = wrap_pi(roll_sp - roll);
    float aileron = IL_KP_ROLL * roll_err - IL_KD_ROLL * p;
    float pitch_target = clampf(pitch_sp, IL_PITCH_SP_MIN, IL_PITCH_SP_MAX);
    float pitch_err = pitch_target - pitch;
    float theta_dot = q * cosf(roll) - r * sinf(roll);
    float elevator = -(IL_KP_PITCH * pitch_err - IL_KD_PITCH * theta_dot);
    float rudder = -IL_KR_RUDDER * p;
    s.aileron = clampf(aileron, -IL_AIL_MAX, IL_AIL_MAX);
    s.elevator = clampf(elevator, -IL_ELEV_MAX, IL_ELEV_MAX);
    s.rudder = clampf(rudder, -IL_RUD_MAX, IL_RUD_MAX);

    /* GliderEnv._safety_shield() */
    if (fabsf(roll) > IL_SHIELD_BANK) {
        s.aileron = clampf(IL_KP_ROLL * (0.0f - roll) - IL_KD_ROLL * p, -IL_AIL_MAX, IL_AIL_MAX);
        s.shield_bank = 1;
    }
    float c = fabsf(cosf(roll));
    float ceiling = IL_SHIELD_PITCH / (c > 0.5f ? c : 0.5f);
    if (pitch > ceiling && s.elevator < 0.0f) {      /* block nose-up; nose-down allowed */
        s.elevator = 0.0f;
        s.shield_pitch = 1;
    }
    return s;
}

uint16_t rlg_deflection_to_us(float defl_rad, float max_rad, int neutral_us, int direction,
                              int full_us, int min_us, int max_us)
{
    float frac = clampf(defl_rad / max_rad, -1.0f, 1.0f);
    float us = (float)neutral_us + (float)direction * frac * (float)full_us;
    if (us < (float)min_us) us = (float)min_us;
    if (us > (float)max_us) us = (float)max_us;
    return (uint16_t)lroundf(us);
}
