/*
 * imu_frame.h -- the ONE place the IMU board-to-body transform lives.
 *
 * Body frame: x forward, y right, z down. An accelerometer at rest reads +1 g
 * along whichever axis points up (specific force), so level -> az = -1 g,
 * nose up -> ax > 0, right wing down -> ay < 0.
 *
 * Measured 2026-11-19 on the soldered mount (raw counts / 16384):
 *   level -> az -1.037   nose up -> ax +0.256   right wing down -> ay -0.281
 * The raw board frame already is the body frame: every sign is +1.
 * (The bench sketch previously used AX -1 / AY -1, i.e. Rz(180 deg): a proper
 * rotation, so a handedness check cannot catch it, but +x pointed at the tail.
 * Corrected with the user's agreement on 2026-11-19.)
 *
 * Any change must keep determinant +1 (an even number of flips), and the gyro
 * signs must equal the accel signs (same chip axes). Both are checked below at
 * compile time.
 */
#ifndef RLG_IMU_FRAME_H
#define RLG_IMU_FRAME_H

#define AX_SIGN (+1)
#define AY_SIGN (+1)
#define AZ_SIGN (+1)
#define GX_SIGN AX_SIGN
#define GY_SIGN AY_SIGN
#define GZ_SIGN AZ_SIGN

_Static_assert(AX_SIGN * AY_SIGN * AZ_SIGN == 1,
               "IMU transform must have determinant +1 (an odd number of sign flips is left-handed)");

#endif /* RLG_IMU_FRAME_H */
