/* Test-only helpers: expose struct decoding to Python (ctypes) so the C
 * layout is checked against the Python codec byte for byte. Host builds only. */
#include <string.h>
#include "rlg_protocol.h"

size_t rlg_test_telemetry_size(void) { return sizeof(rlg_telemetry_t); }
size_t rlg_test_command_size(void) { return sizeof(rlg_command_t); }

/* out[]: seq, roll, pitch, yaw, p, q, r, alt_baro, sonar, lat, lon, vn, ve, rail_v, status, crc */
void rlg_test_decode_telemetry(const uint8_t *frame, double out[16])
{
    rlg_telemetry_t t;
    memcpy(&t, frame, sizeof t);
    double v[16] = {t.seq, t.roll, t.pitch, t.yaw, t.p, t.q, t.r, t.alt_baro, t.sonar,
                    t.lat, t.lon, t.vn, t.ve, t.rail_v, (double)t.status, t.crc8};
    memcpy(out, v, sizeof v);
}

/* out[]: seq, mode, servo0..3, crc */
void rlg_test_decode_command(const uint8_t *frame, uint32_t out[7])
{
    rlg_command_t c;
    memcpy(&c, frame, sizeof c);
    uint32_t v[7] = {c.seq, c.mode, c.servo_us[0], c.servo_us[1], c.servo_us[2], c.servo_us[3], c.crc8};
    memcpy(out, v, sizeof v);
}
