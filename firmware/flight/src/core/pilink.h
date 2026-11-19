/*
 * pilink.h -- receive side of the Pi UART link (PiSetpoint frames), pure C.
 *
 * Byte-at-a-time, never blocks: a partial frame just waits for more bytes.
 * Hunts for RLG_SYNC, checks the type byte, collects RLG_PISETPOINT_SIZE
 * bytes, verifies the CRC8; on any failure it resynchronises from the next
 * byte so a corrupted frame cannot hide the start of the next one.
 * Counts dropped frames and sequence gaps (the brief: track gaps on both ends).
 * Host-tested against the Python encoder (tests/test_flight_firmware.py).
 */
#ifndef RLG_PILINK_H
#define RLG_PILINK_H

#include <stdbool.h>
#include <stdint.h>
#include "../protocol/rlg_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t buf[RLG_PISETPOINT_SIZE];
    uint8_t n;
    bool have_seq;
    uint16_t last_seq;
    uint32_t ok, bad, gaps, dup;
} rlg_pilink_rx_t;

void rlg_pilink_init(rlg_pilink_rx_t *rx);
/* Feed one byte. Returns true when *out holds a new valid frame. */
bool rlg_pilink_feed(rlg_pilink_rx_t *rx, uint8_t byte, rlg_pisetpoint_t *out);

/* Fill sync/type and the CRC of a PiTelemetry the caller has populated. */
void rlg_pitelemetry_finish(rlg_pitelemetry_t *t);

#ifdef __cplusplus
}
#endif

#endif /* RLG_PILINK_H */
