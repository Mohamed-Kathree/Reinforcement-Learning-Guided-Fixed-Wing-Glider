/*
 * rlg_crc8.c -- CRC8, polynomial 0x07, init 0x00, no reflection, no final XOR
 * (CRC-8/SMBUS; check value over "123456789" = 0xF4).
 * Covers every byte from `type` to the last payload byte.
 * HOST-TESTED ONLY against flighttest/protocol/test_vectors.json. Not yet run on an ESP32.
 */
#include "rlg_protocol.h"

uint8_t rlg_crc8(const uint8_t *p, size_t len)
{
    uint8_t c = RLG_CRC_INIT;
    for (size_t i = 0; i < len; i++) {
        c ^= p[i];
        for (int b = 0; b < 8; b++)
            c = (c & 0x80u) ? (uint8_t)((c << 1) ^ RLG_CRC_POLY) : (uint8_t)(c << 1);
    }
    return c;
}
