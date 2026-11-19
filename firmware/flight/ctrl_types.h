/*
 * ctrl_types.h -- types used in flight.ino function signatures. They live in
 * a header because the Arduino builder inserts function prototypes near the
 * top of the .ino, before any type defined in the .ino itself.
 */
#pragma once
#include "hw_gps.h"
#include "src/protocol/rlg_safety.h"

// One control-loop snapshot of the slow sensors.
struct CtrlSensors {
  rlg_sensors_t safety;   // what rlg_safety needs (baro, GPS speed/fix)
  float baro_alt;         // m above the arm reference, NaN before capture
  GpsSnap gps;
  bool gps_fresh;
};
