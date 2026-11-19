/*
 * hw_calib.h -- servo calibration in use at run time.
 *
 * Source of truth: flighttest/config/calibration.json, edited on the
 * dashboard. Two ways it reaches the aircraft:
 *   1. compiled in: src/generated/calibration_gen.h (firmware_sync), the
 *      default when nothing has been pushed;
 *   2. pushed over the radio: a CalibrationSet packet from the ground app,
 *      accepted ONLY while DISARMED and not airborne, validated with the same
 *      rules as flighttest/calibration.py, applied immediately and stored in
 *      NVS so it survives a reboot. "calclear" on the console erases it.
 * The Info packet (1 Hz) reports which calibration is in use (cal_id, NVS or
 * compiled), so the dashboard can confirm a push.
 */
#pragma once
#include <Preferences.h>
#include "src/generated/calibration_gen.h"
#include "src/protocol/rlg_protocol.h"

struct RuntimeCal {
  uint16_t neutral[RLG_NUM_SERVOS], full[RLG_NUM_SERVOS], mn[RLG_NUM_SERVOS], mx[RLG_NUM_SERVOS];
  int8_t dir[RLG_NUM_SERVOS];
  bool servosMeasured, imuVerified, fromNvs;
  uint32_t id;
};
static RuntimeCal cal;

static uint32_t hexId(const char* h) { return (uint32_t)strtoul(String(h).substring(0, 8).c_str(), nullptr, 16); }

static void calFromHeader() {
  for (int i = 0; i < RLG_NUM_SERVOS; i++) {
    cal.neutral[i] = CAL_NEUTRAL_US[i]; cal.full[i] = CAL_FULL_US[i];
    cal.mn[i] = CAL_MIN_US[i]; cal.mx[i] = CAL_MAX_US[i]; cal.dir[i] = CAL_DIRECTION[i];
  }
  cal.servosMeasured = CAL_SERVOS_MEASURED; cal.imuVerified = CAL_IMU_VERIFIED;
  cal.fromNvs = false; cal.id = hexId(CAL_HASH);
}

// Same rules as flighttest/calibration.py _servo_problems(). Returns nullptr if valid.
static const char* calProblem(const rlg_calibrationset_t& c) {
  for (int i = 0; i < RLG_NUM_SERVOS; i++) {
    if (c.min_us[i] < RLG_ABS_MIN_US || c.max_us[i] > RLG_ABS_MAX_US || c.min_us[i] >= c.max_us[i]) return "limits";
    if (c.neutral_us[i] < c.min_us[i] || c.neutral_us[i] > c.max_us[i]) return "neutral outside limits";
    if (c.full_us[i] < 50 || c.full_us[i] > 900) return "full deflection";
    if (c.neutral_us[i] - c.full_us[i] < c.min_us[i] || c.neutral_us[i] + c.full_us[i] > c.max_us[i]) return "full deflection beyond limits";
  }
  return nullptr;
}

static void calFromPacket(const rlg_calibrationset_t& c) {
  for (int i = 0; i < RLG_NUM_SERVOS; i++) {
    cal.neutral[i] = c.neutral_us[i]; cal.full[i] = c.full_us[i];
    cal.mn[i] = c.min_us[i]; cal.mx[i] = c.max_us[i];
    cal.dir[i] = (c.dir_neg >> i) & 1 ? -1 : 1;
  }
  cal.servosMeasured = c.flags & 1; cal.imuVerified = (c.flags >> 1) & 1;
  cal.id = c.cal_id;
}

// NVS: the raw packet is stored; it is re-validated on load.
static bool calLoadNvs() {
  Preferences p;
  if (!p.begin("rlgcal", true)) return false;
  rlg_calibrationset_t c;
  size_t n = p.getBytes("set", &c, sizeof c);
  p.end();
  if (n != sizeof c || calProblem(c)) return false;
  calFromPacket(c);
  cal.fromNvs = true;
  return true;
}

static bool calSaveNvs(const rlg_calibrationset_t& c) {
  Preferences p;
  if (!p.begin("rlgcal", false)) return false;
  bool ok = p.putBytes("set", &c, sizeof c) == sizeof c;
  p.end();
  return ok;
}

static void calClearNvs() {
  Preferences p;
  if (p.begin("rlgcal", false)) { p.clear(); p.end(); }
}

static uint8_t calFlags() {
  return (cal.servosMeasured ? 1 : 0) | (cal.imuVerified ? 2 : 0) | (cal.fromNvs ? 4 : 0);
}
