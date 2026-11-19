/* Host-test accessors for firmware/flight/src/core (ctypes cannot see struct layouts). */
#include <string.h>
#include "../../firmware/flight/src/core/pilink.h"
#include "../../firmware/flight/src/core/ubx.h"
#include "../../firmware/flight/src/core/flight_phase.h"

/* ---- pilink ---- */
size_t t_pilink_size(void) { return sizeof(rlg_pilink_rx_t); }
void t_pilink_init(void *rx) { rlg_pilink_init((rlg_pilink_rx_t *)rx); }
int t_pilink_feed(void *rx, uint8_t b, uint16_t *seq, float *roll_sp, float *speed_cmd, uint8_t *controller)
{
    rlg_pisetpoint_t f;
    if (!rlg_pilink_feed((rlg_pilink_rx_t *)rx, b, &f)) return 0;
    *seq = f.seq; *roll_sp = f.roll_sp; *speed_cmd = f.speed_cmd; *controller = f.controller;
    return 1;
}
void t_pilink_counts(void *rx, uint32_t out[4])
{
    rlg_pilink_rx_t *r = (rlg_pilink_rx_t *)rx;
    out[0] = r->ok; out[1] = r->bad; out[2] = r->gaps; out[3] = r->dup;
}
size_t t_pitelemetry_size(void) { return sizeof(rlg_pitelemetry_t); }
size_t t_pisetpoint_size(void) { return sizeof(rlg_pisetpoint_t); }
/* Build a PiTelemetry with a few known fields, CRC by the firmware routine. */
void t_pitelemetry_example(uint8_t *out)
{
    rlg_pitelemetry_t t;
    memset(&t, 0, sizeof t);
    t.seq = 513; t.t_ms = 123456; t.roll = 0.25f; t.yaw = -3.0f; t.baro_alt = 21.5f; t.sonar = -1.0f;
    t.lat = -339339590; t.lon = 184546280; t.home_lat = -339339000; t.home_lon = 184546000;
    t.vn = 1.5f; t.ve = -2.5f; t.vd = 0.75f; t.gps_fix = 2; t.gps_sats = 7; t.gps_count = 42;
    t.status = 0x00A40019u; t.pi_rx_gaps = 3; t.pi_rx_bad = 1;
    rlg_pitelemetry_finish(&t);
    memcpy(out, &t, sizeof t);
}

/* ---- ubx ---- */
size_t t_ubx_size(void) { return sizeof(rlg_ubx_t); }
void t_ubx_init(void *u) { rlg_ubx_init((rlg_ubx_t *)u); }
int t_ubx_feed(void *u, uint8_t b) { return rlg_ubx_feed((rlg_ubx_t *)u, b); }
void t_ubx_get(void *p, int32_t ints[3], float flts[5], uint32_t cnt[4])
{
    rlg_ubx_t *u = (rlg_ubx_t *)p;
    ints[0] = u->lat_e7; ints[1] = u->lon_e7; ints[2] = u->hmsl_mm;
    flts[0] = u->vn; flts[1] = u->ve; flts[2] = u->vd; flts[3] = u->gspeed; flts[4] = u->heading_deg;
    cnt[0] = u->fix; cnt[1] = u->sats; cnt[2] = u->solutions; cnt[3] = u->bad_ck;
}
size_t t_ubx_build(uint8_t cls, uint8_t id, const uint8_t *payload, uint16_t len, uint8_t *out)
{
    return rlg_ubx_build(cls, id, payload, len, out);
}

/* ---- flight phase ---- */
size_t t_phase_size(void) { return sizeof(rlg_flightphase_t); }
void t_phase_init(void *s, int handoff_at_apex)
{
    rlg_phase_cfg_t c = rlg_phase_default_cfg();
    c.handoff_at_apex = handoff_at_apex != 0;
    rlg_phase_init((rlg_flightphase_t *)s, c);
}
int t_phase_update(void *s, uint32_t now_ms, int armed, float accel_g, float baro)
{
    return rlg_phase_update((rlg_flightphase_t *)s, now_ms, armed != 0, accel_g, baro);
}
void t_phase_causes(void *p, uint32_t out[4])
{
    rlg_flightphase_t *s = (rlg_flightphase_t *)p;
    out[0] = s->launch_cause; out[1] = s->apex_cause; out[2] = s->t_launch_ms; out[3] = s->t_apex_ms;
}
