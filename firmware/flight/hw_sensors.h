/*
 * hw_sensors.h -- the slow sensors, serviced by sensorTask on core 0 at 50 Hz.
 * None of them is read inside the 200 Hz control loop.
 *
 * Barometer (BMP280, I2C 21/22): bench settings (NORMAL, x2 temp, x16 press,
 *   IIR x16, 0.5 ms standby). The ground pressure reference is NOT captured at
 *   boot: it is captured when FLIGHT is armed, and FLIGHT arming is refused
 *   until CAL_BARO_WARMUP_MS (calibration.json baro.warmup_s) has passed since
 *   boot. Before capture, altitude is NaN ("no reference": does not vote
 *   airborne, telemetry carries NaN).
 * Sonar (TRIG 26 / ECHO 35): the echo is timed by a pin-change interrupt, not
 *   pulseIn() (which blocks up to its timeout). Valid only for an echo shorter
 *   than SONAR_MAX_ECHO_US (4.5 m, the simulator lidar_agl range); it is a
 *   flare trigger only. Range = echo_us * 343 m/s / 2.
 * GPS (NEO-6M, UART2 16/17, 9600 baud): non-blocking NMEA. GGA gives fix,
 *   satellites, lat/lon; RMC gives ground speed and course -> vn/ve. GPS
 *   altitude is parsed for logging but never used for control.
 * Rail (GPIO 34): pin volts averaged over 8 reads. The divider is NOT fitted
 *   (reads ~0.02 V): below RAIL_FITTED_MIN_V the reading is reported invalid.
 */
#pragma once
#include <Wire.h>
#include <Adafruit_BMP280.h>
#include <math.h>
#include "config.h"
#include "src/generated/calibration_gen.h"

// ---------------- barometer ----------------
static Adafruit_BMP280 bmp;
static bool baroOk = false;

static bool baroBegin(Print* log) {
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  baroOk = bmp.begin(0x76) || bmp.begin(0x77);
  if (baroOk)
    bmp.setSampling(Adafruit_BMP280::MODE_NORMAL, Adafruit_BMP280::SAMPLING_X2,
                    Adafruit_BMP280::SAMPLING_X16, Adafruit_BMP280::FILTER_X16,
                    Adafruit_BMP280::STANDBY_MS_1);
  log->printf("baro: %s\n", baroOk ? "BMP280 ok" : "NOT FOUND");
  return baroOk;
}

// Hypsometric altitude of pressure p relative to reference pressure ref (Pa).
static float baroAltitude(float p, float ref) {
  return 44330.0f * (1.0f - powf(p / ref, 0.1903f));
}

// ---------------- sonar ----------------
static volatile uint32_t sonRise = 0, sonWidth = 0;
static volatile bool sonDone = false;

static void IRAM_ATTR sonarIsr() {
  uint32_t t = micros();
  if (digitalRead(PIN_SON_ECHO)) sonRise = t;
  else if (sonRise) { sonWidth = t - sonRise; sonDone = true; }
}

static void sonarBegin() {
  pinMode(PIN_SON_TRIG, OUTPUT);
  digitalWrite(PIN_SON_TRIG, LOW);
  pinMode(PIN_SON_ECHO, INPUT);
  attachInterrupt(digitalPinToInterrupt(PIN_SON_ECHO), sonarIsr, CHANGE);
}

static void sonarTrigger() {
  sonDone = false; sonRise = 0;
  digitalWrite(PIN_SON_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_SON_TRIG, LOW);
}

// Called just before the next trigger. Returns metres, or -1 if no valid echo.
static float sonarCollect() {
  if (!sonDone) return -1.0f;
  uint32_t w = sonWidth;
  if (w == 0 || w > SONAR_MAX_ECHO_US) return -1.0f;
  return (float)w * 343.0e-6f / 2.0f;
}

// ---------------- GPS ----------------
struct GpsState {
  uint8_t fix = 0, sats = 0;           // fix: 0 none, 1 GPS, 2 DGPS (GGA quality, capped)
  double lat = NAN, lon = NAN;
  float alt_m = NAN;                   // logging only, never control
  float speed_ms = NAN, course_deg = NAN, vn = NAN, ve = NAN;
  uint32_t last_gga_ms = 0, last_rmc_ms = 0;
  uint32_t sentences = 0, bad_checksum = 0;
};
static GpsState gps;
static char gpsLine[100];
static uint8_t gpsLen = 0;

static int nmeaField(const char* s, int idx, char* out, int cap) {
  int f = 0; const char* p = s;
  while (*p && f < idx) { if (*p == ',') f++; p++; }
  int n = 0;
  while (*p && *p != ',' && *p != '*' && n < cap - 1) out[n++] = *p++;
  out[n] = 0;
  return n;
}

static double nmeaDeg(const char* v, const char* hemi) {
  if (!*v) return NAN;
  double raw = atof(v);
  int deg = (int)(raw / 100.0);
  double d = deg + (raw - deg * 100.0) / 60.0;
  return (*hemi == 'S' || *hemi == 'W') ? -d : d;
}

static bool nmeaChecksumOk(const char* s) {
  const char* star = strchr(s, '*');
  if (!star || s[0] != '$') return false;
  uint8_t c = 0;
  for (const char* p = s + 1; p < star; p++) c ^= (uint8_t)*p;
  return (uint8_t)strtol(star + 1, nullptr, 16) == c;
}

static void gpsParse(const char* s, uint32_t now) {
  if (!nmeaChecksumOk(s)) { gps.bad_checksum++; return; }
  gps.sentences++;
  char f[20], h[4];
  if (!strncmp(s + 3, "GGA", 3)) {
    nmeaField(s, 6, f, sizeof f); int q = atoi(f);
    gps.fix = q > 2 ? 2 : (uint8_t)q;
    nmeaField(s, 7, f, sizeof f); gps.sats = (uint8_t)atoi(f);
    if (gps.fix) {
      nmeaField(s, 2, f, sizeof f); nmeaField(s, 3, h, sizeof h); gps.lat = nmeaDeg(f, h);
      nmeaField(s, 4, f, sizeof f); nmeaField(s, 5, h, sizeof h); gps.lon = nmeaDeg(f, h);
      nmeaField(s, 9, f, sizeof f); gps.alt_m = *f ? atof(f) : NAN;
    }
    gps.last_gga_ms = now;
  } else if (!strncmp(s + 3, "RMC", 3)) {
    nmeaField(s, 2, f, sizeof f);
    if (f[0] == 'A') {
      nmeaField(s, 7, f, sizeof f); gps.speed_ms = *f ? atof(f) * 0.514444f : NAN;
      nmeaField(s, 8, f, sizeof f); gps.course_deg = *f ? atof(f) : NAN;
      if (!isnan(gps.speed_ms)) {
        float c = isnan(gps.course_deg) ? 0.0f : gps.course_deg * (float)DEG_TO_RAD;
        gps.vn = gps.speed_ms * cosf(c);
        gps.ve = gps.speed_ms * sinf(c);
      }
    } else {
      gps.speed_ms = gps.vn = gps.ve = NAN;
    }
    gps.last_rmc_ms = now;
  }
}

static void gpsBegin() { Serial2.begin(GPS_BAUD, SERIAL_8N1, PIN_GPS_RX, PIN_GPS_TX); }

static void gpsPoll(uint32_t now) {
  while (Serial2.available()) {
    char c = (char)Serial2.read();
    if (c == '\r') continue;
    if (c == '\n') { gpsLine[gpsLen] = 0; if (gpsLen > 6) gpsParse(gpsLine, now); gpsLen = 0; }
    else if (gpsLen < sizeof(gpsLine) - 1) gpsLine[gpsLen++] = c;
    else gpsLen = 0;   // overlong: drop
  }
}

// ---------------- rail ----------------
static float railPinVolts() {
  uint32_t mv = 0;
  for (int i = 0; i < 8; i++) mv += analogReadMilliVolts(PIN_RAIL_ADC);
  return (float)mv / 8000.0f;
}
