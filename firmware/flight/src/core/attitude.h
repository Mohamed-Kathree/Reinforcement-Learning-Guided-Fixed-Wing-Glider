/*
 * attitude.h -- roll/pitch estimation: one 2-state Kalman filter per axis
 * (state [angle, gyro bias]), ported from Hardware/Code/glider_viz.py.
 * Pure C, no Arduino: host-tested against the Python original in
 * tests/test_flight_firmware.py.
 *
 * Units inside the filter are DEGREES and deg/s, as in the Python version, so
 * the tuning carries over unchanged: Q_ANGLE 0.001, Q_BIAS 0.003, R_MEASURE 0.03.
 *
 * Accelerometer gate: the accel correction is applied only while |a| is within
 * [ATT_GATE_LO_G, ATT_GATE_HI_G] (0.85-1.15 g); outside it the filter propagates
 * on gyro alone. On a desk this rarely fires; in flight it fires constantly.
 *
 * Kinematics (one deliberate difference from the Python version):
 *   euler_kinematics = 0  integrate body rates p, q directly as roll/pitch rates,
 *                         exactly as glider_viz.py does. Use to compare against
 *                         the visualiser.
 *   euler_kinematics = 1  (default in flight) integrate Euler-angle rates
 *                         phi_dot = p + (q sin phi + r cos phi) tan theta,
 *                         theta_dot = q cos phi - r sin phi.
 *                         Body rates equal Euler rates only wings-level; in a
 *                         banked turn the accel gate is closed (|a| ~ 1/cos(bank))
 *                         so the filter runs on gyro alone, and body-rate
 *                         integration would drift for the whole turn.
 * Yaw has no absolute reference (magnetometer not configured): it is integrated
 * from gyro rates and drifts. Telemetry must say so.
 */
#ifndef RLG_ATTITUDE_H
#define RLG_ATTITUDE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ATT_Q_ANGLE   0.001f
#define ATT_Q_BIAS    0.003f
#define ATT_R_MEASURE 0.03f
#define ATT_GATE_LO_G 0.85f
#define ATT_GATE_HI_G 1.15f
#define ATT_DT_MAX_S  0.2f     /* larger gaps are rejected, as in the Python version */

typedef struct {
    float angle;     /* deg */
    float bias;      /* deg/s */
    float P[2][2];
} rlg_kalman_t;

void  rlg_kalman_init(rlg_kalman_t *k);
/* Predict with gyro rate, correct with accel angle (degrees). Returns angle. */
float rlg_kalman_update(rlg_kalman_t *k, float new_angle_deg, float new_rate_dps, float dt);
/* Gyro-only propagation (accel gated out): angle += dt * (rate - bias). */
float rlg_kalman_predict_only(rlg_kalman_t *k, float rate_dps, float dt);

typedef struct {
    rlg_kalman_t roll, pitch;
    float yaw_deg;           /* integrated, unreferenced: drifts */
    bool seeded;
    bool euler_kinematics;
    bool accel_trusted;      /* last sample's gate decision */
    float accel_mag_g;
    /* measurements for the bench record */
    uint32_t samples, gated_out, dt_rejected;
} rlg_attitude_t;

void rlg_attitude_init(rlg_attitude_t *a, bool euler_kinematics);
/* One IMU sample, BODY frame (imu_frame.h already applied): accel in g, gyro
 * in deg/s, dt in seconds from the data-ready timestamps. Returns false if the
 * sample was not used (first sample seeds the filter; dt out of range). */
bool rlg_attitude_update(rlg_attitude_t *a, float ax, float ay, float az,
                         float gx, float gy, float gz, float dt);
/* Accel-only angles (deg) from a specific-force vector, x fwd / y right / z down. */
float rlg_accel_roll_deg(float ay, float az);
float rlg_accel_pitch_deg(float ax, float ay, float az);

#ifdef __cplusplus
}
#endif

#endif /* RLG_ATTITUDE_H */
