/*
 * hw_gps.h -- NEO-6M on UART2 (GPIO 16 RX / 17 TX), serviced by sensorTask.
 *
 * UBX mode (preferred): 5 Hz NAV-POSLLH + NAV-VELNED + NAV-SOL at 38400 baud,
 * UBX-only output. Matches the simulator's 5 Hz GPS and supplies vertical
 * velocity (obs[8]). The configuration is not saved in the receiver, so it is
 * redone every boot. After an EN reset the receiver keeps power and may still
 * be at 38400 from the previous boot, so 38400 is tried first.
 *
 * NMEA fallback (the bench-verified 9600-baud GGA/RMC parser in hw_sensors.h):
 * used if no UBX frame arrives. 1 Hz, no vertical velocity (vd = NaN). The
 * status bit gps_ubx says which mode is running.
 *
 * NEW, NOT YET VERIFIED ON HARDWARE: the UBX configuration. The parser is
 * host-tested (tests/test_flight_core_v3.py).
 */
#pragma once
#include "config.h"
#include "hw_sensors.h"
#include "src/core/ubx.h"

struct GpsSnap {
  bool ubx = false;
  uint8_t fix = 0, sats = 0;
  int32_t lat_e7 = 0, lon_e7 = 0;
  float vn = NAN, ve = NAN, vd = NAN, speed = NAN, course_deg = NAN;
  uint32_t last_ms = 0;            // millis() of the last new solution
  uint16_t count = 0;              // new solutions since boot
};

static rlg_ubx_t ubx;
static GpsSnap gpsNow;
static bool gpsUbxMode = false;

static void ubxSend(uint8_t cls, uint8_t id, const uint8_t* pl, uint16_t len) {
  uint8_t f[48];
  size_t n = rlg_ubx_build(cls, id, pl, len, f);
  Serial2.write(f, n);
  Serial2.flush();
  delay(30);
}

// Listen for ms; true if a valid UBX frame arrived.
static bool ubxSeen(uint32_t ms) {
  uint32_t f0 = ubx.frames, t0 = millis();
  while (millis() - t0 < ms) {
    while (Serial2.available()) rlg_ubx_feed(&ubx, (uint8_t)Serial2.read());
    if (ubx.frames != f0) return true;
    delay(5);
  }
  return false;
}

static void ubxConfigure() {
  // NAV-POSLLH, NAV-VELNED, NAV-SOL at 1 per navigation solution
  const uint8_t msgs[3][2] = {{0x01, 0x02}, {0x01, 0x12}, {0x01, 0x06}};
  for (auto& m : msgs) { uint8_t pl[3] = {m[0], m[1], 1}; ubxSend(0x06, 0x01, pl, 3); }
  // CFG-RATE: 200 ms, 1 cycle, GPS time  (5 Hz)
  const uint8_t rate[6] = {0xC8, 0x00, 0x01, 0x00, 0x01, 0x00};
  ubxSend(0x06, 0x08, rate, 6);
}

static void cfgPrt38400UbxOnly() {
  // CFG-PRT UART1: 8N1, 38400, in UBX+NMEA, out UBX only
  uint8_t pl[20] = {0};
  pl[0] = 1;                                         // portID UART1
  pl[4] = 0xD0; pl[5] = 0x08;                        // mode 0x000008D0 = 8N1
  uint32_t baud = 38400; memcpy(pl + 8, &baud, 4);
  pl[12] = 0x03;                                     // inProtoMask UBX|NMEA
  pl[14] = 0x01;                                     // outProtoMask UBX
  ubxSend(0x06, 0x00, pl, 20);
}

static void gpsBeginAuto(Print* log) {
  rlg_ubx_init(&ubx);
  Serial2.begin(38400, SERIAL_8N1, PIN_GPS_RX, PIN_GPS_TX);       // still configured from a warm reset?
  if (!ubxSeen(1200)) {
    Serial2.updateBaudRate(GPS_BAUD);                              // factory 9600 NMEA
    delay(50);
    ubxConfigure();                                                // at 9600 (UBX accepted on input)
    cfgPrt38400UbxOnly();
    Serial2.updateBaudRate(38400);
    delay(100);
  }
  ubxConfigure();                                                  // (re)apply at 38400
  gpsUbxMode = ubxSeen(2500);
  if (!gpsUbxMode) {
    Serial2.updateBaudRate(GPS_BAUD);                              // fall back to the verified NMEA path
    log->println("gps: NO UBX -> NMEA fallback (1 Hz, no vertical velocity)");
  } else {
    log->println("gps: UBX 5 Hz (POSLLH + VELNED + SOL), 38400 baud");
  }
}

// sensorTask: returns true if a new solution arrived.
static bool gpsService(uint32_t now) {
  bool fresh = false;
  if (gpsUbxMode) {
    while (Serial2.available()) {
      if (rlg_ubx_feed(&ubx, (uint8_t)Serial2.read())) {
        gpsNow.ubx = true;
        gpsNow.fix = ubx.fix; gpsNow.sats = ubx.sats;
        gpsNow.lat_e7 = ubx.lat_e7; gpsNow.lon_e7 = ubx.lon_e7;
        gpsNow.vn = ubx.vn; gpsNow.ve = ubx.ve; gpsNow.vd = ubx.vd;
        gpsNow.speed = ubx.gspeed;
        gpsNow.course_deg = atan2f(ubx.ve, ubx.vn) * RAD_TO_DEG;
        gpsNow.last_ms = now; gpsNow.count++;
        fresh = true;
      }
    }
  } else {
    uint32_t before = gps.last_gga_ms;
    gpsPoll(now);                                                  // NMEA (hw_sensors.h)
    if (gps.last_gga_ms != before) {
      gpsNow.ubx = false;
      gpsNow.fix = gps.fix; gpsNow.sats = gps.sats;
      if (!isnan(gps.lat)) { gpsNow.lat_e7 = (int32_t)lround(gps.lat * 1e7); gpsNow.lon_e7 = (int32_t)lround(gps.lon * 1e7); }
      gpsNow.vn = gps.vn; gpsNow.ve = gps.ve; gpsNow.vd = NAN;
      gpsNow.speed = gps.speed_ms; gpsNow.course_deg = gps.course_deg;
      gpsNow.last_ms = now; gpsNow.count++;
      fresh = true;
    }
  }
  return fresh;
}
