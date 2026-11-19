/* ubx.c -- see ubx.h. */
#include <string.h>
#include "ubx.h"

static int32_t rd_i32(const uint8_t *p) { return (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24)); }
static uint32_t rd_u32(const uint8_t *p) { return (uint32_t)rd_i32(p); }

void rlg_ubx_init(rlg_ubx_t *u)
{
    memset(u, 0, sizeof(*u));
}

static bool decode(rlg_ubx_t *u)
{
    const uint8_t *p = u->payload;
    if (u->cls != 0x01) return false;
    if (u->id == 0x02 && u->len == 28) {                 /* NAV-POSLLH */
        u->itow_pos = rd_u32(p);
        u->lon_e7 = rd_i32(p + 4);
        u->lat_e7 = rd_i32(p + 8);
        u->hmsl_mm = rd_i32(p + 16);
        u->have_pos = true;
    } else if (u->id == 0x12 && u->len == 36) {          /* NAV-VELNED */
        u->itow_vel = rd_u32(p);
        u->vn = rd_i32(p + 4) * 0.01f;
        u->ve = rd_i32(p + 8) * 0.01f;
        u->vd = rd_i32(p + 12) * 0.01f;
        u->gspeed = rd_u32(p + 20) * 0.01f;
        u->heading_deg = rd_i32(p + 24) * 1e-5f;
        u->have_vel = true;
    } else if (u->id == 0x06 && u->len == 52) {          /* NAV-SOL */
        uint8_t gps_fix = p[10], ok = p[11] & 0x01;
        u->fix = !ok ? 0 : (gps_fix == 3 || gps_fix == 4) ? 2 : (gps_fix == 2 ? 1 : 0);
        u->sats = p[47];
        return false;
    } else {
        return false;
    }
    if (u->have_pos && u->have_vel && u->itow_pos == u->itow_vel) {
        u->have_pos = u->have_vel = false;
        u->solutions++;
        return true;
    }
    return false;
}

bool rlg_ubx_feed(rlg_ubx_t *u, uint8_t b)
{
    switch (u->state) {
    case 0: if (b == 0xB5) u->state = 1; return false;
    case 1: u->state = (b == 0x62) ? 2 : (b == 0xB5 ? 1 : 0); return false;
    case 2: u->cls = b; u->ck_a = b; u->ck_b = b; u->state = 3; return false;
    case 3: u->id = b; u->ck_a += b; u->ck_b += u->ck_a; u->state = 4; return false;
    case 4: u->len = b; u->ck_a += b; u->ck_b += u->ck_a; u->state = 5; return false;
    case 5:
        u->len |= (uint16_t)b << 8; u->ck_a += b; u->ck_b += u->ck_a; u->idx = 0;
        if (u->len > UBX_MAX_PAYLOAD) { u->too_long++; u->state = 0; return false; }
        u->state = u->len ? 6 : 7;
        return false;
    case 6:
        u->payload[u->idx++] = b; u->ck_a += b; u->ck_b += u->ck_a;
        if (u->idx >= u->len) u->state = 7;
        return false;
    case 7:
        if (b != u->ck_a) { u->bad_ck++; u->state = 0; return false; }
        u->state = 8;
        return false;
    case 8:
        u->state = 0;
        if (b != u->ck_b) { u->bad_ck++; return false; }
        u->frames++;
        return decode(u);
    default:
        u->state = 0;
        return false;
    }
}

size_t rlg_ubx_build(uint8_t cls, uint8_t id, const uint8_t *payload, uint16_t len, uint8_t *out)
{
    uint8_t a = 0, b = 0;
    out[0] = 0xB5; out[1] = 0x62; out[2] = cls; out[3] = id;
    out[4] = (uint8_t)(len & 0xFF); out[5] = (uint8_t)(len >> 8);
    if (len) memcpy(out + 6, payload, len);
    for (size_t i = 2; i < (size_t)len + 6; i++) { a += out[i]; b += a; }
    out[len + 6] = a; out[len + 7] = b;
    return (size_t)len + 8;
}
