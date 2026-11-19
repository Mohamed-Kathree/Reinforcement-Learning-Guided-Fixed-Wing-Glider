/*
 * ubx.h -- u-blox UBX protocol for the NEO-6M, pure C.
 *
 * Why UBX: the simulator's GPS runs at 5 Hz and the policy observes GPS
 * vertical velocity (obs[8]); the NEO-6M's default NMEA output is 1 Hz and
 * carries no vertical velocity. The firmware configures the receiver at boot
 * (CFG-RATE 200 ms, NAV-POSLLH + NAV-VELNED + NAV-SOL, UBX-only output at
 * 38400 baud; not saved to the receiver's flash, so it is redone every boot)
 * and falls back to NMEA if no UBX arrives.
 *
 * Message layouts (u-blox 6 Receiver Description, GPS.G6-SW-10018):
 *   NAV-POSLLH 0x01 0x02, 28 B: iTOW u4, lon i4 1e-7 deg, lat i4, height i4 mm,
 *                               hMSL i4 mm, hAcc u4 mm, vAcc u4 mm
 *   NAV-VELNED 0x01 0x12, 36 B: iTOW u4, velN i4 cm/s, velE i4, velD i4,
 *                               speed u4, gSpeed u4, heading i4 1e-5 deg, sAcc u4, cAcc u4
 *   NAV-SOL    0x01 0x06, 52 B: gpsFix u1 @10, flags u1 @11 (bit0 gpsFixOk), numSV u1 @47
 */
#ifndef RLG_UBX_H
#define RLG_UBX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UBX_MAX_PAYLOAD 64

typedef struct {
    /* parser */
    uint8_t state, cls, id, ck_a, ck_b;
    uint16_t len, idx;
    uint8_t payload[UBX_MAX_PAYLOAD];
    uint32_t frames, bad_ck, too_long;
    /* decoded */
    int32_t lat_e7, lon_e7, hmsl_mm;
    float vn, ve, vd, gspeed, heading_deg;   /* m/s, deg */
    uint8_t fix;                             /* 0 none, 1 2D, 2 3D (protocol convention) */
    uint8_t sats;
    uint32_t itow_pos, itow_vel;
    bool have_pos, have_vel;
    uint32_t solutions;                      /* POSLLH+VELNED pairs with equal iTOW */
} rlg_ubx_t;

void rlg_ubx_init(rlg_ubx_t *u);
/* Feed one byte. Returns true when this byte completed a NEW solution
 * (POSLLH and VELNED with the same iTOW). */
bool rlg_ubx_feed(rlg_ubx_t *u, uint8_t byte);

/* Build a UBX frame into out (size >= len + 8). Returns the frame length. */
size_t rlg_ubx_build(uint8_t cls, uint8_t id, const uint8_t *payload, uint16_t len, uint8_t *out);

#ifdef __cplusplus
}
#endif

#endif /* RLG_UBX_H */
