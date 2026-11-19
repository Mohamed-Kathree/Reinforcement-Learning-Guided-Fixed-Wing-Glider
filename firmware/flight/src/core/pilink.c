/* pilink.c -- see pilink.h. */
#include <string.h>
#include "pilink.h"

void rlg_pilink_init(rlg_pilink_rx_t *rx)
{
    memset(rx, 0, sizeof(*rx));
}

/* Try to complete a frame from rx->buf; on failure drop the first byte and
 * re-scan what is left for the next SYNC. */
static bool try_frame(rlg_pilink_rx_t *rx, rlg_pisetpoint_t *out)
{
    for (;;) {
        if (rx->n == 0) return false;
        if (rx->buf[0] != RLG_SYNC) {
            memmove(rx->buf, rx->buf + 1, --rx->n);
            continue;
        }
        if (rx->n >= 2 && rx->buf[1] != RLG_TYPE_PISETPOINT) {
            rx->bad++;
            memmove(rx->buf, rx->buf + 1, --rx->n);
            continue;
        }
        if (rx->n < RLG_PISETPOINT_SIZE) return false;
        if (rlg_crc8(rx->buf + 1, RLG_PISETPOINT_SIZE - 2) != rx->buf[RLG_PISETPOINT_SIZE - 1]) {
            rx->bad++;
            memmove(rx->buf, rx->buf + 1, --rx->n);
            continue;
        }
        memcpy(out, rx->buf, sizeof(*out));
        rx->n = 0;
        if (rx->have_seq) {
            uint16_t step = (uint16_t)(out->seq - rx->last_seq);
            if (step == 0) rx->dup++;
            else if (step > 1 && step < 0x8000) rx->gaps += (uint32_t)(step - 1);
        }
        rx->have_seq = true;
        rx->last_seq = out->seq;
        rx->ok++;
        return true;
    }
}

bool rlg_pilink_feed(rlg_pilink_rx_t *rx, uint8_t byte, rlg_pisetpoint_t *out)
{
    if (rx->n == 0 && byte != RLG_SYNC) return false;     /* hunting: ignore noise */
    rx->buf[rx->n++] = byte;
    return try_frame(rx, out);
}

void rlg_pitelemetry_finish(rlg_pitelemetry_t *t)
{
    t->sync = RLG_SYNC;
    t->type = RLG_TYPE_PITELEMETRY;
    t->crc8 = rlg_crc8(&t->type, RLG_PITELEMETRY_SIZE - 2);
}
