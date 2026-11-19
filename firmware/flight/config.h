/*
 * config.h -- flight firmware build configuration (Phase 1: inner loop, no Pi).
 *
 * Pins are FIXED (soldered). Servo neutrals/directions/limits and rail scaling
 * are NOT here: they come from flighttest/config/calibration.json via
 * src/generated/calibration_gen.h. The IMU frame is in src/core/imu_frame.h.
 */
#pragma once

// ---------------- pin map (fixed, soldered, do not change) ----------------
#define PIN_IMU_SCK   18      // breakout SCL
#define PIN_IMU_MISO  19      // breakout AD0
#define PIN_IMU_MOSI  23      // breakout SDA
#define PIN_IMU_CS     5      // breakout NCS; strapping pin, 10k pull-up fitted
#define PIN_IMU_INT    4      // data-ready
#define PIN_I2C_SDA   21      // BMP280
#define PIN_I2C_SCL   22
#define PIN_GPS_RX    16      // <- GPS TX
#define PIN_GPS_TX    17      // -> GPS RX
#define PIN_SON_TRIG  26
#define PIN_SON_ECHO  35      // input-only, deliberate
#define PIN_SRV_AILL  13
#define PIN_SRV_AILR  14
#define PIN_SRV_ELEV  25
#define PIN_SRV_RUD   27
#define PIN_PI_TX     32      // Phase 2
#define PIN_PI_RX     33      // Phase 2
#define PIN_RAIL_ADC  34      // divider not fitted yet: reads ~0.02 V

// ---------------- radio (ESP-NOW to the ground station) ----------------
static const uint8_t GROUND_MAC[6] = {0xD4, 0x8A, 0xFC, 0xA4, 0xA0, 0x08};
#define RADIO_SETTLE_MS      15     // after each esp_now_send; tuned, do not reduce
#define RADIO_QUEUE_LEN      24
#define TELEMETRY_PERIOD_MS  200    // binary rlg_telemetry_t, 5 Hz, ON by default (console 'tlm' toggles)
#define INFO_PERIOD_MS       1000   // binary rlg_info_t, 1 Hz (calibration in use, link counters)
#define STATUS_PERIOD_MS     1000   // one text status line per second

// ---------------- IMU (ICM-20948 over SPI) ----------------
#define IMU_SPI_HZ       1000000    // as the verified bench sketch
// Full scale. The bench sketch uses the power-on default (+/-2 g, +/-250 dps);
// a hand throw exceeds both (the launch detector needs > 2.5 g), so flight uses
// +/-8 g and +/-1000 dps. FS_SEL and LSB MUST change together.
#define IMU_ACCEL_FS_SEL 2          // 0 +/-2 g, 1 +/-4 g, 2 +/-8 g, 3 +/-16 g
#define IMU_ACC_LSB      4096.0f    // LSB per g at +/-8 g
#define IMU_GYRO_FS_SEL  2          // 0 +/-250, 1 +/-500, 2 +/-1000, 3 +/-2000 dps
#define IMU_GYRO_LSB     32.8f      // LSB per dps at +/-1000 dps
// Output data rate = 1125 Hz / (1 + div). div 4 -> 225 Hz: the nearest rate at
// or above 200 Hz. dt is measured from the data-ready timestamps, so the
// estimator does not assume it.
#define IMU_SMPLRT_DIV   4
// Digital low-pass: cfg 3 = ~51 Hz gyro / ~50 Hz accel. To be re-tuned on the
// airframe for vibration (record the spectrum first).
#define IMU_DLPF_CFG     3
#define IMU_DRDY_TIMEOUT_MS 20      // no data-ready edge -> poll and count it

// ---------------- estimation ----------------
#define ATT_EULER_KINEMATICS 1      // 1 flight default; 'K' toggles 0 to match glider_viz.py

// ---------------- failsafe glide (Pi silent > 150 ms in FLIGHT) ----------------
// Wings level, trimmed slightly nose-down, gentle glide. FIRST GUESS: the
// simulator's trimmed glide pitch is about +0.7 deg (glide path ~ -4 deg plus
// ~4.8 deg alpha); -2 deg trades a little glide ratio for stall margin. Tune on
// hardware/in simulation before relying on it.
#define FAILSAFE_ROLL_DEG    0.0f
#define FAILSAFE_PITCH_DEG  -2.0f

// ---------------- flight phases (src/core/flight_phase.h; FIRST GUESSES, tune on real throws) ----------------
#define HANDOFF_AT_APEX      1      // 1: Pi takes over at the apex (training start state); 0: at the throw
#define LAUNCH_ACCEL_G       2.5f   // |a| above this ...
#define LAUNCH_HOLD_MS       20     // ... for this long = the throw
#define APEX_MIN_MS          400    // earliest apex after the throw
#define APEX_DROP_M          0.3f   // low-passed baro this far below its peak = descending
#define APEX_TIMEOUT_MS      3000   // apex at the latest this long after the throw
#define YAW_ALIGN_SPEED_MS   3.0f   // after launch, set yaw = GPS course once ground speed exceeds this
#define REQUIRE_GPS_FIX_TO_ARM 1    // FLIGHT arming needs a 3D fix (home); the 'uncal' bench override skips it

// ---------------- Pi link (pi_link.h) ----------------
#define PI_TLM_PERIOD_MS     20     // PiTelemetry 50 Hz
#define BANK_SP_MAX_DEG      45.0f  // policy action[0] range
#define SPEED_CMD_MIN_MS     9.5f   // policy action[1] range (env/glider_env.py)
#define SPEED_CMD_MAX_MS     16.0f

// ---------------- slow sensors (core 0) ----------------
#define SENSOR_PERIOD_MS     20     // 50 Hz task
#define SONAR_PERIOD_MS      50     // trigger every 50 ms (20 Hz, as the simulator)
#define SONAR_MAX_ECHO_US    26240  // 4.5 m (= simulator lidar_agl validity range); longer or none = invalid
#define GPS_BAUD             9600   // NEO-6M default: 1 Hz NMEA (simulator assumes 5 Hz, see README)
#define GPS_STALE_MS         3000   // older fix data does not vote "airborne"
#define RAIL_PERIOD_MS       200
#define RAIL_FITTED_MIN_V    0.30f  // pin volts below this = divider not fitted (reads ~0.02 V)
