/*
 * attitude.c -- see attitude.h. The Kalman arithmetic is a line-for-line port
 * of glider_viz.py's Kalman.update(); tests/test_flight_firmware.py checks the
 * two agree to float precision on random sequences.
 */
#include <math.h>
#include <string.h>
#include "attitude.h"

#define RAD2DEG 57.29577951308232f
#define DEG2RAD 0.017453292519943295f

void rlg_kalman_init(rlg_kalman_t *k)
{
    memset(k, 0, sizeof(*k));
}

float rlg_kalman_update(rlg_kalman_t *k, float new_angle, float new_rate, float dt)
{
    /* predict */
    float rate_unbiased = new_rate - k->bias;
    k->angle += dt * rate_unbiased;
    k->P[0][0] += dt * (dt * k->P[1][1] - k->P[0][1] - k->P[1][0] + ATT_Q_ANGLE);
    k->P[0][1] -= dt * k->P[1][1];
    k->P[1][0] -= dt * k->P[1][1];
    k->P[1][1] += ATT_Q_BIAS * dt;

    /* update */
    float S = k->P[0][0] + ATT_R_MEASURE;
    float K0 = k->P[0][0] / S;
    float K1 = k->P[1][0] / S;
    float y = new_angle - k->angle;
    while (y > 180.0f) y -= 360.0f;
    while (y < -180.0f) y += 360.0f;
    k->angle += K0 * y;
    k->bias += K1 * y;
    float P00 = k->P[0][0], P01 = k->P[0][1];
    k->P[0][0] -= K0 * P00;
    k->P[0][1] -= K0 * P01;
    k->P[1][0] -= K1 * P00;
    k->P[1][1] -= K1 * P01;
    return k->angle;
}

float rlg_kalman_predict_only(rlg_kalman_t *k, float rate, float dt)
{
    k->angle += dt * (rate - k->bias);
    return k->angle;
}

float rlg_accel_roll_deg(float ay, float az)
{
    return atan2f(-ay, -az) * RAD2DEG;
}

float rlg_accel_pitch_deg(float ax, float ay, float az)
{
    return atan2f(ax, sqrtf(ay * ay + az * az)) * RAD2DEG;
}

void rlg_attitude_init(rlg_attitude_t *a, bool euler_kinematics)
{
    memset(a, 0, sizeof(*a));
    rlg_kalman_init(&a->roll);
    rlg_kalman_init(&a->pitch);
    a->euler_kinematics = euler_kinematics;
}

static float wrap180(float d)
{
    while (d > 180.0f) d -= 360.0f;
    while (d < -180.0f) d += 360.0f;
    return d;
}

bool rlg_attitude_update(rlg_attitude_t *a, float ax, float ay, float az,
                         float gx, float gy, float gz, float dt)
{
    float roll_a = rlg_accel_roll_deg(ay, az);
    float pitch_a = rlg_accel_pitch_deg(ax, ay, az);
    a->accel_mag_g = sqrtf(ax * ax + ay * ay + az * az);

    if (!a->seeded) {
        a->roll.angle = roll_a;
        a->pitch.angle = pitch_a;
        a->seeded = true;
        return false;
    }
    if (!(dt > 0.0f) || dt > ATT_DT_MAX_S) {
        a->dt_rejected++;
        return false;
    }
    a->samples++;

    /* Rates driving each angle. */
    float roll_rate = gx, pitch_rate = gy, yaw_rate = gz;
    if (a->euler_kinematics) {
        float phi = a->roll.angle * DEG2RAD, th = a->pitch.angle * DEG2RAD;
        float sp = sinf(phi), cp = cosf(phi), ct = cosf(th);
        if (fabsf(ct) < 0.05f) ct = ct < 0 ? -0.05f : 0.05f;      /* near +/-90 deg pitch */
        float tt = sinf(th) / ct;
        roll_rate = gx + (gy * sp + gz * cp) * tt;
        pitch_rate = gy * cp - gz * sp;
        yaw_rate = (gy * sp + gz * cp) / ct;
    }

    a->accel_trusted = a->accel_mag_g > ATT_GATE_LO_G && a->accel_mag_g < ATT_GATE_HI_G;
    if (a->accel_trusted) {
        rlg_kalman_update(&a->roll, roll_a, roll_rate, dt);
        rlg_kalman_update(&a->pitch, pitch_a, pitch_rate, dt);
    } else {
        a->gated_out++;
        rlg_kalman_predict_only(&a->roll, roll_rate, dt);
        rlg_kalman_predict_only(&a->pitch, pitch_rate, dt);
    }
    a->roll.angle = wrap180(a->roll.angle);
    a->yaw_deg = wrap180(a->yaw_deg + dt * yaw_rate);
    return true;
}
