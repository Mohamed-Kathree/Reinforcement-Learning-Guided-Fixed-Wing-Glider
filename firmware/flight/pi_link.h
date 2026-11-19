/*
 * pi_link.h -- UART link to the Raspberry Pi 4B (GPIO 32 TX -> Pi pin 10 RXD,
 * GPIO 33 RX <- Pi pin 8 TXD, 115200 8N1, common ground).
 *
 *   ESP32 -> Pi: PiTelemetry (83 B) at 50 Hz: attitude, rates, baro, sonar,
 *                GPS position/velocity, home, phase/status, link counters.
 *   Pi -> ESP32: PiSetpoint (16 B) at 20 Hz: bank angle and speed command.
 *
 * Serviced by piTask (core 0). Received setpoints go through a queue to the
 * control loop, which alone calls rlg_safety_on_pi_setpoint() and flies them.
 * Parsing is byte-at-a-time and never blocks (src/core/pilink.c).
 * NEW, NOT YET VERIFIED ON HARDWARE.
 */
#pragma once
#include "config.h"
#include "src/core/pilink.h"

#define PI_BAUD 115200
static_assert(sizeof(rlg_pitelemetry_t) == RLG_PITELEMETRY_SIZE, "PiTelemetry size");
static_assert(sizeof(rlg_pisetpoint_t) == RLG_PISETPOINT_SIZE, "PiSetpoint size");

static rlg_pilink_rx_t piRx;

static void piLinkBegin() {
  Serial1.setRxBufferSize(512);
  Serial1.setTxBufferSize(512);
  Serial1.begin(PI_BAUD, SERIAL_8N1, PIN_PI_RX, PIN_PI_TX);
  rlg_pilink_init(&piRx);
}

// Drain the UART; push each valid setpoint to q (non-blocking).
static void piLinkPoll(QueueHandle_t q) {
  rlg_pisetpoint_t f;
  while (Serial1.available()) {
    if (rlg_pilink_feed(&piRx, (uint8_t)Serial1.read(), &f)) xQueueSend(q, &f, 0);
  }
}

static void piLinkSend(rlg_pitelemetry_t& t) {
  rlg_pitelemetry_finish(&t);
  if (Serial1.availableForWrite() >= (int)sizeof t) Serial1.write((const uint8_t*)&t, sizeof t);
}
