/*
 * flight.ino -- RL glider flight firmware (Phase 2: Pi policy interface).
 *
 *   core 1, loopTask (raised priority): the ~225 Hz control loop. Woken by the
 *     ICM-20948 data-ready interrupt (GPIO 4); dt from the interrupt
 *     timestamps. IMU -> Kalman attitude -> commands (ground, Pi, calibration)
 *     -> rlg_safety -> flight phase -> setpoints -> inner loop -> servos (MCPWM).
 *     It never prints, never touches I2C/UART/radio, never calls esp_now_send.
 *   core 0, sensorTask (50 Hz): barometer, sonar, GPS (UBX 5 Hz), rail ADC.
 *   core 0, piTask (100 Hz): Pi UART, PiTelemetry at 50 Hz, PiSetpoint parsing.
 *   core 0, radioTask: the only caller of esp_now_send (15 ms settle each).
 *   core 0, consoleTask (100 Hz): console, 1 Hz status line, 5 Hz Telemetry,
 *     1 Hz Info, calibration persistence (NVS), optional USB stream.
 *
 * FLIGHT: DISARMED -> ARMED (home + baro reference captured; failsafe glide)
 *   -> LAUNCH (throw detected; wings level, elevator neutral) -> AUTONOMOUS at
 *   the apex (the Pi's bank + speed setpoints; failsafe glide whenever the Pi
 *   link is stale). See src/core/flight_phase.h and firmware/flight/README.md.
 */
#include <Arduino.h>
#include <esp_task_wdt.h>
#include "config.h"
#include "hw_imu.h"
#include "hw_sensors.h"
#include "hw_gps.h"
#include "hw_servo.h"
#include "hw_calib.h"
#include "ctrl_types.h"
#include "radio_link.h"
#include "pi_link.h"
#include "src/core/attitude.h"
#include "src/core/inner_loop.h"
#include "src/core/flight_phase.h"
#include "src/protocol/rlg_protocol.h"
#include "src/protocol/rlg_safety.h"
#include "src/generated/calibration_gen.h"

#define FW_NAME "rlg-flight phase2"
#define STREAM_DECIMATE 2            // USB stream at ~112 Hz (225 / 2); USB only, never radio

static RadioLink Link;
static const int SERVO_PINS[RLG_NUM_SERVOS] = {PIN_SRV_AILL, PIN_SRV_AILR, PIN_SRV_ELEV, PIN_SRV_RUD};
static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
static QueueHandle_t cmdQueue, streamQueue, piQueue, calQueue;
static TaskHandle_t ctrlTask = nullptr;

// ---------------- shared: sensors (written by sensorTask) ----------------
struct SensorSnap {
  float baro_alt = NAN;          // m above the arm-time reference; NaN before capture
  float baro_pa = NAN, baro_ref_pa = NAN;
  bool baro_ref = false, baro_warm = false;
  float sonar_m = -1.0f;         // -1 = no valid echo
  GpsSnap gps;
  bool gps_fresh = false;        // a solution within GPS_STALE_MS
  float gps_rate_hz = 0;
  float rail_pin_v = 0.0f;
  bool rail_valid = false;
};
static SensorSnap sensShared;
static volatile bool baroCaptureReq = false;

// ---------------- shared: control (written by the control loop) ----------------
struct CtrlSnap {
  float roll_deg = 0, pitch_deg = 0, yaw_deg = 0, p = 0, q = 0, r = 0;   // rates rad/s
  float accel_g = 0;
  float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0;   // body frame, g and deg/s
  bool seeded = false, accel_trusted = false, euler = true;
  uint8_t mode = 0; bool armed = false, failsafe = false, airborne = false;
  uint8_t phase = RLG_PHASE_DISARMED, launch_cause = 0, apex_cause = 0;
  bool yaw_aligned = false, home_set = false, pi_link = false;
  int32_t home_lat = 0, home_lon = 0;
  float roll_sp = 0, pitch_sp = 0, pi_speed_cmd = NAN;
  uint8_t pi_controller = 0;
  uint16_t pi_tlm_gaps = 0;
  uint16_t us[RLG_NUM_SERVOS] = {0};
  float ail = 0, elev = 0, rud = 0;    // rad, last inner-loop output
  uint32_t status = 0;
  uint32_t att_samples = 0, att_gated = 0, att_dt_rej = 0;
  uint32_t t_launch = 0, t_apex = 0;
  float peak_accel_g = 0;
  float loop_hz = 0;
};
static CtrlSnap ctrlShared;

struct LoopStats {
  uint32_t n = 0, dtMin = UINT32_MAX, dtMax = 0, compMax = 0;
  uint64_t dtSum = 0, compSum = 0;
  uint32_t gated = 0, timeouts = 0, overruns = 0, shieldBank = 0, shieldPitch = 0;
  uint32_t secImu = 0, secAtt = 0, secCtl = 0, secSrv = 0;   // max us per stage (status "sec=")
};
static LoopStats statsShared;

static void mergeStats(LoopStats& a, const LoopStats& b) {
  if (b.dtMin < a.dtMin) a.dtMin = b.dtMin;
  if (b.dtMax > a.dtMax) a.dtMax = b.dtMax;
  if (b.compMax > a.compMax) a.compMax = b.compMax;
  a.n += b.n; a.dtSum += b.dtSum; a.compSum += b.compSum; a.gated += b.gated;
  a.timeouts += b.timeouts; a.overruns += b.overruns; a.shieldBank += b.shieldBank; a.shieldPitch += b.shieldPitch;
  if (b.secImu > a.secImu) a.secImu = b.secImu;
  if (b.secAtt > a.secAtt) a.secAtt = b.secAtt;
  if (b.secCtl > a.secCtl) a.secCtl = b.secCtl;
  if (b.secSrv > a.secSrv) a.secSrv = b.secSrv;
}

struct StreamRec { uint32_t t_ms; int16_t raw[6]; float roll, pitch; uint8_t gate; };

// ---------------- console -> control flags ----------------
static volatile bool wantEuler = ATT_EULER_KINEMATICS;
static volatile bool allowUncal = false;      // per boot: bench override (unmeasured calibration, no GPS fix)
static volatile bool streamOn = false, tlmOn = true;
static volatile bool servoWrites = true;      // "srv off" stops PWM updates (diagnostic)
static volatile bool calSaveReq = false;
static rlg_calibrationset_t calToSave;
static volatile int lastCalResult = -1;       // 0 applied, 1 refused armed/airborne, 2 invalid, 3 saved, 4 save failed

// ---------------- control-loop-owned state ----------------
static rlg_attitude_t att;
static rlg_safety_t st;
static rlg_flightphase_t ph;
static bool useDrdyInterrupt = false;
static float drdyRateHz = 0;
static bool imuOk = false;
static volatile uint32_t drdyTs = 0, drdyCount = 0;
static volatile int lastCmdResult = -1;
static volatile uint32_t refusedNotWarm = 0, refusedUncal = 0, refusedNoImu = 0, refusedNoFix = 0;

static void IRAM_ATTR drdyIsr() {
  drdyTs = micros();
  drdyCount = drdyCount + 1;
  BaseType_t hpw = pdFALSE;
  if (ctrlTask) vTaskNotifyGiveFromISR(ctrlTask, &hpw);
  portYIELD_FROM_ISR(hpw);
}

static CtrlSensors readSensors() {
  CtrlSensors c;
  portENTER_CRITICAL(&mux);
  c.baro_alt = sensShared.baro_alt;
  c.gps = sensShared.gps;
  c.gps_fresh = sensShared.gps_fresh;
  portEXIT_CRITICAL(&mux);
  c.safety.baro_alt_m = c.baro_alt;
  c.safety.gps_fix = c.gps_fresh ? c.gps.fix : 0;
  c.safety.gps_speed_ms = (c.gps_fresh && !isnan(c.gps.speed)) ? c.gps.speed : 0.0f;
  return c;
}

static bool flightArmed() { return st.armed && st.mode == RLG_MODE_FLIGHT; }

// Home (GPS at FLIGHT arm), owned by the control loop.
static bool homeSet = false;
static int32_t homeLat = 0, homeLon = 0;

// One command frame (ESP-NOW or console) -> rlg_safety. Control loop only.
static void handleFrame(CmdFrame& f, uint32_t now_ms, const CtrlSensors& cs) {
  rlg_command_t c;
  memcpy(&c, f.data, sizeof c);
  if (f.src == 1) {                     // console frames take the next sequence number
    c.seq = st.have_seq ? (uint16_t)(st.last_seq + 1) : 1;
    c.crc8 = rlg_crc8(&c.type, sizeof c - 2);
    memcpy(f.data, &c, sizeof c);
  }
  if (c.mode == RLG_MODE_FLIGHT && !flightArmed()) {
    // Firmware preconditions for FLIGHT (on top of rlg_safety's own rules).
    if (!imuOk) { refusedNoImu++; lastCmdResult = 100; return; }
    if (millis() < CAL_BARO_WARMUP_MS) { refusedNotWarm++; lastCmdResult = 101; return; }
    if (!(cal.servosMeasured && cal.imuVerified) && !allowUncal) { refusedUncal++; lastCmdResult = 102; return; }
    if (REQUIRE_GPS_FIX_TO_ARM && !(cs.gps_fresh && cs.gps.fix >= 2) && !allowUncal) { refusedNoFix++; lastCmdResult = 103; return; }
  }
  bool airborneBefore = rlg_safety_is_airborne(&st, now_ms, &cs.safety);
  rlg_cmd_result_t res = rlg_safety_on_command(&st, now_ms, &cs.safety, f.data, RLG_COMMAND_SIZE);
  lastCmdResult = (int)res;
  // Ground pressure reference and home: captured on FLIGHT arm (after warm-up),
  // never at boot, and never re-captured while airborne.
  if (res == RLG_CMD_ARMED && st.mode == RLG_MODE_FLIGHT && !airborneBefore) {
    baroCaptureReq = true;
    if (cs.gps_fresh && cs.gps.fix >= 1) { homeLat = cs.gps.lat_e7; homeLon = cs.gps.lon_e7; homeSet = true; }
    else homeSet = false;
  }
}

// A CalibrationSet from the ground: only DISARMED and on the ground.
static void handleCalibration(const rlg_calibrationset_t& c, uint32_t now_ms, const CtrlSensors& cs) {
  if (st.armed || rlg_safety_is_airborne(&st, now_ms, &cs.safety)) { lastCalResult = 1; return; }
  if (calProblem(c)) { lastCalResult = 2; return; }
  calFromPacket(c);
  cal.fromNvs = false;                  // until consoleTask has written it
  uint16_t mn[RLG_NUM_SERVOS], mx[RLG_NUM_SERVOS];
  for (int i = 0; i < RLG_NUM_SERVOS; i++) { st.neutral_us[i] = cal.neutral[i]; mn[i] = cal.mn[i]; mx[i] = cal.mx[i]; }
  rlg_safety_set_limits(&st, mn, mx);
  for (int i = 0; i < RLG_NUM_SERVOS; i++) st.servo_us[i] = st.neutral_us[i];   // disarmed: new neutral now
  calToSave = c;
  calSaveReq = true;
  lastCalResult = 0;
}

// Servo output (hw_servo.h, MCPWM): changed values only; a write costs microseconds.
static void writeServoValues(const uint16_t us[RLG_NUM_SERVOS]) {
  static uint16_t written[RLG_NUM_SERVOS] = {0};
  for (int i = 0; i < RLG_NUM_SERVOS; i++)
    if (us[i] >= RLG_ABS_MIN_US && us[i] != written[i]) { servoWriteUs(i, us[i]); written[i] = us[i]; }
}

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ======================= setup =======================
void setup() {
  Serial.setTxBufferSize(2048);
  Serial.begin(115200);
  delay(400);
  cmdQueue = xQueueCreate(16, sizeof(CmdFrame));
  streamQueue = xQueueCreate(64, sizeof(StreamRec));
  piQueue = xQueueCreate(8, sizeof(rlg_pisetpoint_t));
  calQueue = xQueueCreate(2, sizeof(rlg_calibrationset_t));
  Link.calQueue = calQueue;
  Link.begin(cmdQueue);

  calFromHeader();
  bool nvs = calLoadNvs();
  Link.printf("\n%s  schema v%d  cal %08lX (%s) servos_measured=%d imu_verified=%d\n", FW_NAME, RLG_SCHEMA_VERSION,
              (unsigned long)cal.id, nvs ? "NVS, pushed from the ground" : "compiled-in", cal.servosMeasured, cal.imuVerified);
  Link.printf("imu frame signs a(%d,%d,%d) g(%d,%d,%d)\n", AX_SIGN, AY_SIGN, AZ_SIGN, GX_SIGN, GY_SIGN, GZ_SIGN);
  if (!cal.servosMeasured)
    Link.println("WARNING: servo calibration NOT measured (placeholders). FLIGHT arming refused unless 'uncal' is typed.");

  // Servos first, at neutral, so they hold still through the rest of boot.
  uint16_t neutral[RLG_NUM_SERVOS], mn[RLG_NUM_SERVOS], mx[RLG_NUM_SERVOS];
  for (int i = 0; i < RLG_NUM_SERVOS; i++) { neutral[i] = cal.neutral[i]; mn[i] = cal.mn[i]; mx[i] = cal.mx[i]; }
  rlg_safety_init(&st, neutral, millis());
  if (!rlg_safety_set_limits(&st, mn, mx)) Link.println("ERROR: calibration min > max; using absolute envelope");
  servoBegin(SERVO_PINS, st.servo_us, &Link);
  writeServoValues(st.servo_us);
  for (int i = 0; i < RLG_NUM_SERVOS; i++) ctrlShared.us[i] = st.servo_us[i];

  imuOk = imuBegin(&Link);
  if (imuOk) {
    imuConfigureRateAndInterrupt();
    attachInterrupt(digitalPinToInterrupt(PIN_IMU_INT), drdyIsr, RISING);
    uint32_t c0 = drdyCount, t0 = millis();
    while (millis() - t0 < 500) { imuSampleBody(); delay(2); }   // reading clears the flag
    drdyRateHz = (drdyCount - c0) * 2.0f;
    useDrdyInterrupt = drdyRateHz > 50.0f;
    if (!useDrdyInterrupt) detachInterrupt(digitalPinToInterrupt(PIN_IMU_INT));
    Link.printf("imu: data-ready %.0f Hz measured (expect ~%d) -> %s; +/-%d g, +/-%d dps\n", drdyRateHz,
                (int)(1125 / (1 + IMU_SMPLRT_DIV)),
                useDrdyInterrupt ? "INTERRUPT timing" : "NO EDGES: timed POLLING at 200 Hz",
                2 << IMU_ACCEL_FS_SEL, 250 << IMU_GYRO_FS_SEL);
  } else {
    Link.println("imu: FAILED. Attitude is invalid and FLIGHT cannot be armed.");
  }
  rlg_attitude_init(&att, ATT_EULER_KINEMATICS);

  rlg_phase_cfg_t pc = rlg_phase_default_cfg();
  pc.launch_accel_g = LAUNCH_ACCEL_G; pc.launch_hold_ms = LAUNCH_HOLD_MS; pc.apex_min_ms = APEX_MIN_MS;
  pc.apex_drop_m = APEX_DROP_M; pc.apex_timeout_ms = APEX_TIMEOUT_MS; pc.handoff_at_apex = HANDOFF_AT_APEX;
  rlg_phase_init(&ph, pc);

  baroBegin(&Link);
  sonarBegin();
  gpsBeginAuto(&Link);
  piLinkBegin();
  analogReadResolution(12);
  analogSetPinAttenuation(PIN_RAIL_ADC, ADC_11db);
  Link.printf("baro: reference + home captured on FLIGHT arm; arming allowed after %u s warm-up%s\n",
              (unsigned)(CAL_BARO_WARMUP_MS / 1000), REQUIRE_GPS_FIX_TO_ARM ? " with a 3D GPS fix" : "");
  Link.printf("phases: handoff at %s; launch > %.1f g for %d ms; apex: baro -%.1f m from peak after %d ms, or %d ms\n",
              HANDOFF_AT_APEX ? "APEX" : "THROW", LAUNCH_ACCEL_G, LAUNCH_HOLD_MS, APEX_DROP_M, APEX_MIN_MS, APEX_TIMEOUT_MS);
  Link.println("type 'help'");

  extern void sensorTask(void*);
  extern void radioTask(void*);
  extern void consoleTask(void*);
  extern void piTask(void*);
  xTaskCreatePinnedToCore(sensorTask, "sensors", 4096, nullptr, 3, nullptr, 0);
  xTaskCreatePinnedToCore(piTask, "pi", 4096, nullptr, 4, nullptr, 0);
  xTaskCreatePinnedToCore(radioTask, "radio", 4096, nullptr, 2, nullptr, 0);
  xTaskCreatePinnedToCore(consoleTask, "console", 6144, nullptr, 1, nullptr, 0);

  ctrlTask = xTaskGetCurrentTaskHandle();
  vTaskPrioritySet(nullptr, 10);
  esp_task_wdt_add(nullptr);   // a control loop stalled > 5 s is reported with a backtrace
}

// ======================= ~225 Hz control loop (core 1) =======================
void loop() {
  static uint32_t lastTs = 0, lastCount = 0, n = 0;
  static TickType_t lastWake = xTaskGetTickCount();
  static LoopStats ls;
  static bool yawAligned = false;
  static uint16_t lastGpsCount = 0;
  static rlg_pisetpoint_t piSp = {};
  static bool havePiSp = false;
  uint32_t ts;

  if (useDrdyInterrupt) {
    if (ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(IMU_DRDY_TIMEOUT_MS)) > 0) {
      ts = drdyTs;
      uint32_t c = drdyCount;
      if (lastCount && c - lastCount > 1) ls.overruns += c - lastCount - 1;   // missed samples
      lastCount = c;
    } else { ts = micros(); ls.timeouts++; }
  } else {
    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(5));
    ts = micros();
  }
  esp_task_wdt_reset();
  uint32_t t0 = micros();
  uint32_t now_ms = millis();

  // ---- attitude ----
  ImuRaw raw = imuOk ? imuSampleBody() : ImuRaw{0, 0, 0, 0, 0, 0};
  uint32_t tImu = micros();
  float ax = raw.ax / IMU_ACC_LSB, ay = raw.ay / IMU_ACC_LSB, az = raw.az / IMU_ACC_LSB;
  float gx = raw.gx / IMU_GYRO_LSB, gy = raw.gy / IMU_GYRO_LSB, gz = raw.gz / IMU_GYRO_LSB;
  att.euler_kinematics = wantEuler;
  float dt = lastTs ? (ts - lastTs) * 1e-6f : 0.0f;
  uint32_t dtUs = lastTs ? ts - lastTs : 0;
  lastTs = ts;
  if (imuOk) rlg_attitude_update(&att, ax, ay, az, gx, gy, gz, dt);
  uint32_t tAtt = micros();

  // ---- commands, calibration, Pi setpoints, safety ----
  CtrlSensors cs = readSensors();
  CmdFrame f;
  while (xQueueReceive(cmdQueue, &f, 0) == pdTRUE) handleFrame(f, now_ms, cs);
  rlg_calibrationset_t cset;
  while (xQueueReceive(calQueue, &cset, 0) == pdTRUE) handleCalibration(cset, now_ms, cs);
  rlg_pisetpoint_t sp;
  while (xQueueReceive(piQueue, &sp, 0) == pdTRUE) {
    rlg_safety_on_pi_setpoint(&st, now_ms);
    piSp = sp; havePiSp = true;
  }
  rlg_safety_tick(&st, now_ms, &cs.safety);

  // ---- flight phase ----
  uint8_t phase = rlg_phase_update(&ph, now_ms, flightArmed(), att.accel_mag_g, cs.baro_alt);
  if (phase == RLG_PHASE_DISARMED) yawAligned = false;
  // Yaw: gyro-integrated from 0 at boot; aligned once to the GPS course after
  // the throw (researcher's decision 19 Nov 2026). One-shot per flight.
  if (phase >= RLG_PHASE_LAUNCH && !yawAligned && cs.gps.count != lastGpsCount &&
      cs.gps_fresh && cs.gps.speed > YAW_ALIGN_SPEED_MS && !isnan(cs.gps.course_deg)) {
    att.yaw_deg = cs.gps.course_deg;
    yawAligned = true;
  }
  lastGpsCount = cs.gps.count;

  float roll = att.roll.angle * DEG_TO_RAD, pitch = att.pitch.angle * DEG_TO_RAD;
  float p = gx * DEG_TO_RAD, q = gy * DEG_TO_RAD, r = gz * DEG_TO_RAD;

  // ---- setpoints and inner loop (FLIGHT only) ----
  rlg_surfaces_t s = {0};
  float roll_sp = FAILSAFE_ROLL_DEG * DEG_TO_RAD, pitch_sp = FAILSAFE_PITCH_DEG * DEG_TO_RAD;
  if (flightArmed()) {
    bool elevNeutral = false;
    if (phase == RLG_PHASE_LAUNCH) {
      roll_sp = 0.0f;                                   // zoom climb: wings level, no pitch command
      elevNeutral = true;
    } else if (phase == RLG_PHASE_AUTONOMOUS && !rlg_safety_pi_failsafe(&st) && havePiSp) {
      roll_sp = clampf(piSp.roll_sp, -BANK_SP_MAX_DEG * DEG_TO_RAD, BANK_SP_MAX_DEG * DEG_TO_RAD);
      float v = clampf(piSp.speed_cmd, SPEED_CMD_MIN_MS, SPEED_CMD_MAX_MS);
      pitch_sp = rlg_pitch_target(v, roll);             // trim schedule, no airspeed term
    }                                                   // else (ARMED, or Pi stale): failsafe glide
    s = rlg_inner_loop(roll, pitch, p, q, r, roll_sp, pitch_sp);
    if (elevNeutral) s.elevator = 0.0f;
    uint16_t us[RLG_NUM_SERVOS];
    // Both ailerons take the same signed command (+ = roll right); each servo's
    // calibrated direction makes its own surface move the right way.
    const float defl[RLG_NUM_SERVOS] = {s.aileron, s.aileron, s.elevator, s.rudder};
    const float mx[RLG_NUM_SERVOS] = {IL_AIL_MAX, IL_AIL_MAX, IL_ELEV_MAX, IL_RUD_MAX};
    for (int i = 0; i < RLG_NUM_SERVOS; i++)
      us[i] = rlg_deflection_to_us(defl[i], mx[i], cal.neutral[i], cal.dir[i], cal.full[i], cal.mn[i], cal.mx[i]);
    rlg_safety_flight_output(&st, us);
  }
  uint32_t tCtl = micros();
  if (servoWrites) writeServoValues(st.servo_us);
  uint32_t tSrv = micros();
  uint32_t comp = tSrv - t0;
  if (tImu - t0 > ls.secImu) ls.secImu = tImu - t0;
  if (tAtt - tImu > ls.secAtt) ls.secAtt = tAtt - tImu;
  if (tCtl - tAtt > ls.secCtl) ls.secCtl = tCtl - tAtt;
  if (tSrv - tCtl > ls.secSrv) ls.secSrv = tSrv - tCtl;

  // ---- measurements ----
  if (dtUs) {
    ls.n++; ls.dtSum += dtUs; ls.compSum += comp;
    if (dtUs < ls.dtMin) ls.dtMin = dtUs;
    if (dtUs > ls.dtMax) ls.dtMax = dtUs;
    if (comp > ls.compMax) ls.compMax = comp;
    if (!att.accel_trusted) ls.gated++;
    if (s.shield_bank) ls.shieldBank++;
    if (s.shield_pitch) ls.shieldPitch++;
  }

  bool piLink = havePiSp && !rlg_safety_pi_failsafe(&st) && (uint32_t)(now_ms - st.last_pi_ms) <= RLG_PI_TIMEOUT_MS;
  uint32_t status = rlg_safety_status_bits(&st)
                  | ((uint32_t)phase << RLG_STATUS_PHASE_LSB)
                  | ((uint32_t)yawAligned << RLG_STATUS_YAW_ALIGNED_LSB)
                  | ((uint32_t)(homeSet && flightArmed()) << RLG_STATUS_HOME_SET_LSB)
                  | ((uint32_t)cs.gps.ubx << RLG_STATUS_GPS_UBX_LSB)
                  | ((uint32_t)piLink << RLG_STATUS_PI_LINK_LSB)
                  | ((uint32_t)cal.fromNvs << RLG_STATUS_CAL_NVS_LSB);
  portENTER_CRITICAL(&mux);
  ctrlShared.roll_deg = att.roll.angle; ctrlShared.pitch_deg = att.pitch.angle; ctrlShared.yaw_deg = att.yaw_deg;
  ctrlShared.p = p; ctrlShared.q = q; ctrlShared.r = r;
  ctrlShared.accel_g = att.accel_mag_g; ctrlShared.seeded = att.seeded;
  ctrlShared.ax = ax; ctrlShared.ay = ay; ctrlShared.az = az;
  ctrlShared.gx = gx; ctrlShared.gy = gy; ctrlShared.gz = gz;
  ctrlShared.accel_trusted = att.accel_trusted; ctrlShared.euler = att.euler_kinematics;
  ctrlShared.mode = st.mode; ctrlShared.armed = st.armed;
  ctrlShared.failsafe = rlg_safety_pi_failsafe(&st); ctrlShared.airborne = st.airborne;
  ctrlShared.phase = phase; ctrlShared.launch_cause = ph.launch_cause; ctrlShared.apex_cause = ph.apex_cause;
  ctrlShared.t_launch = ph.t_launch_ms; ctrlShared.t_apex = ph.t_apex_ms; ctrlShared.peak_accel_g = ph.peak_accel_g;
  ctrlShared.yaw_aligned = yawAligned; ctrlShared.home_set = homeSet && flightArmed();
  ctrlShared.home_lat = homeLat; ctrlShared.home_lon = homeLon; ctrlShared.pi_link = piLink;
  ctrlShared.roll_sp = roll_sp; ctrlShared.pitch_sp = pitch_sp;
  ctrlShared.pi_speed_cmd = havePiSp ? piSp.speed_cmd : NAN;
  ctrlShared.pi_controller = havePiSp ? piSp.controller : 0;
  ctrlShared.pi_tlm_gaps = havePiSp ? piSp.tlm_gaps : 0;
  for (int i = 0; i < RLG_NUM_SERVOS; i++) ctrlShared.us[i] = st.servo_us[i];
  ctrlShared.ail = s.aileron; ctrlShared.elev = s.elevator; ctrlShared.rud = s.rudder;
  ctrlShared.status = status;
  ctrlShared.att_samples = att.samples; ctrlShared.att_gated = att.gated_out; ctrlShared.att_dt_rej = att.dt_rejected;
  if (ls.n) { mergeStats(statsShared, ls); ls = LoopStats(); }   // console takes and clears it once a second
  portEXIT_CRITICAL(&mux);

  if (streamOn && (++n % STREAM_DECIMATE) == 0) {
    StreamRec rec = {now_ms, {raw.ax, raw.ay, raw.az, raw.gx, raw.gy, raw.gz},
                     att.roll.angle, att.pitch.angle, (uint8_t)att.accel_trusted};
    xQueueSend(streamQueue, &rec, 0);   // full queue: drop, never block
  }
}

// ======================= sensorTask (core 0, 50 Hz) =======================
void sensorTask(void*) {
  TickType_t wake = xTaskGetTickCount();
  float pBuf[25]; int pN = 0, pI = 0;      // 0.5 s of pressure for the reference
  float refPa = NAN, sonar = -1.0f, railV = 0.0f;
  uint32_t lastSonar = 0, lastRail = 0, rateT0 = millis();
  uint16_t rateC0 = 0;
  float rateHz = 0;
  for (;;) {
    vTaskDelayUntil(&wake, pdMS_TO_TICKS(SENSOR_PERIOD_MS));
    uint32_t now = millis();
    gpsService(now);
    if (now - rateT0 >= 10000) { rateHz = (uint16_t)(gpsNow.count - rateC0) / ((now - rateT0) / 1000.0f); rateC0 = gpsNow.count; rateT0 = now; }

    float pa = baroOk ? bmp.readPressure() : NAN;
    if (!isnan(pa) && pa > 30000.0f) { pBuf[pI] = pa; pI = (pI + 1) % 25; if (pN < 25) pN++; }
    if (baroCaptureReq && pN) {
      float sum = 0; for (int i = 0; i < pN; i++) sum += pBuf[i];
      refPa = sum / pN;
      baroCaptureReq = false;
    }
    float alt = (!isnan(refPa) && !isnan(pa)) ? baroAltitude(pa, refPa) : NAN;

    if (now - lastSonar >= SONAR_PERIOD_MS) { sonar = sonarCollect(); sonarTrigger(); lastSonar = now; }
    if (now - lastRail >= RAIL_PERIOD_MS) { railV = railPinVolts(); lastRail = now; }

    bool fresh = gpsNow.last_ms && (now - gpsNow.last_ms) < GPS_STALE_MS;
    portENTER_CRITICAL(&mux);
    sensShared.baro_pa = pa; sensShared.baro_ref_pa = refPa; sensShared.baro_alt = alt;
    sensShared.baro_ref = !isnan(refPa); sensShared.baro_warm = now >= CAL_BARO_WARMUP_MS;
    sensShared.sonar_m = sonar;
    sensShared.gps = gpsNow; sensShared.gps_fresh = fresh; sensShared.gps_rate_hz = rateHz;
    sensShared.rail_pin_v = railV; sensShared.rail_valid = railV >= RAIL_FITTED_MIN_V;
    portEXIT_CRITICAL(&mux);
  }
}

// ======================= piTask (core 0, 100 Hz) =======================
void piTask(void*) {
  TickType_t wake = xTaskGetTickCount();
  uint32_t lastTlm = 0;
  uint16_t seq = 0;
  for (;;) {
    vTaskDelayUntil(&wake, pdMS_TO_TICKS(10));
    piLinkPoll(piQueue);
    uint32_t now = millis();
    if (now - lastTlm < PI_TLM_PERIOD_MS) continue;
    lastTlm = now;
    CtrlSnap c; SensorSnap s;
    portENTER_CRITICAL(&mux); c = ctrlShared; s = sensShared; portEXIT_CRITICAL(&mux);
    rlg_pitelemetry_t t = {};
    t.seq = ++seq; t.t_ms = now;
    t.roll = c.roll_deg * DEG_TO_RAD; t.pitch = c.pitch_deg * DEG_TO_RAD; t.yaw = c.yaw_deg * DEG_TO_RAD;
    t.p = c.p; t.q = c.q; t.r = c.r;
    t.baro_alt = s.baro_alt; t.sonar = s.sonar_m;
    t.lat = s.gps.lat_e7; t.lon = s.gps.lon_e7; t.home_lat = c.home_lat; t.home_lon = c.home_lon;
    t.vn = s.gps.vn; t.ve = s.gps.ve; t.vd = s.gps.vd;
    t.gps_fix = s.gps_fresh ? s.gps.fix : 0; t.gps_sats = s.gps.sats;
    uint32_t age = s.gps.last_ms ? now - s.gps.last_ms : 65535;
    t.gps_age_ms = age > 65535 ? 65535 : age;
    t.gps_count = s.gps.count;
    uint32_t sats = s.gps.sats > 31 ? 31 : s.gps.sats;
    t.status = c.status
             | ((uint32_t)(s.sonar_m >= 0) << RLG_STATUS_SONAR_VALID_LSB)
             | ((uint32_t)t.gps_fix << RLG_STATUS_GPS_FIX_LSB)
             | (sats << RLG_STATUS_GPS_SATS_LSB)
             | ((uint32_t)s.baro_ref << RLG_STATUS_BARO_REF_CAPTURED_LSB)
             | ((uint32_t)s.baro_warm << RLG_STATUS_BARO_WARM_LSB)
             | ((uint32_t)s.rail_valid << RLG_STATUS_RAIL_V_VALID_LSB);
    t.pi_rx_gaps = piRx.gaps > 65535 ? 65535 : piRx.gaps;
    t.pi_rx_bad = piRx.bad > 65535 ? 65535 : piRx.bad;
    piLinkSend(t);
  }
}

// ======================= radioTask (core 0) =======================
void radioTask(void*) { Link.serviceForever(); }

// ======================= consoleTask (core 0, 100 Hz) =======================
static void queueConsoleFrame(uint8_t mode) {
  rlg_command_t c = {};
  c.sync = RLG_SYNC; c.type = RLG_TYPE_COMMAND; c.mode = mode;
  for (int i = 0; i < RLG_NUM_SERVOS; i++) c.servo_us[i] = cal.neutral[i];
  c.crc8 = rlg_crc8(&c.type, sizeof c - 2);   // seq and CRC are redone by the control loop
  CmdFrame f; memcpy(f.data, &c, sizeof c); f.src = 1;
  xQueueSend(cmdQueue, &f, pdMS_TO_TICKS(50));
}

static const char* modeName(uint8_t m) {
  switch (m) { case RLG_MODE_FLIGHT: return "FLIGHT"; case RLG_MODE_BENCH: return "BENCH";
               case RLG_MODE_REPLAY: return "REPLAY"; default: return "DISARMED"; }
}

static const char* phaseName(uint8_t p) {
  switch (p) { case RLG_PHASE_ARMED: return "ARMED"; case RLG_PHASE_LAUNCH: return "LAUNCH";
               case RLG_PHASE_AUTONOMOUS: return "AUTO"; default: return "-"; }
}

static const char* resultName(int r) {
  switch (r) {
    case -1: return "-"; case 0: return "accepted"; case 1: return "arming"; case 2: return "ARMED";
    case 3: return "disarmed"; case 4: return "rejected:frame"; case 5: return "rejected:stale";
    case 6: return "rejected:airborne"; case 100: return "refused:imu";
    case 101: return "refused:baro-warmup"; case 102: return "refused:uncalibrated";
    case 103: return "refused:no-3D-fix";
    default: return "?";
  }
}

static void printStatus(const CtrlSnap& c, const SensorSnap& s, const LoopStats& l) {
  float secs = l.n ? l.dtSum / 1e6f : 1.0f;
  Link.printf("S t=%lu %s%s%s %s rpy=%.1f,%.1f,%.1f%s |a|=%.2f %s | loop=%.0fHz dt=%lu/%lu/%luus comp=%lu/%luus "
              "sec=%lu/%lu/%lu/%lu gated=%.0f%% tmo=%lu ovr=%lu shield=%lu/%lu | sp=%.1f,%.1f pi=%s gaps=%lu bad=%lu"
              " | baro=%s son=%.2f gps=%u/%u%s%s %.1fHz rail=%s | us=%u,%u,%u,%u cmd=%s\n",
              (unsigned long)(millis() / 1000), modeName(c.mode), c.armed ? "" : "(unarmed)",
              (c.armed && c.mode == RLG_MODE_FLIGHT) ? (c.failsafe ? " FAILSAFE" : " PI") : "", phaseName(c.phase),
              c.roll_deg, c.pitch_deg, c.yaw_deg, c.yaw_aligned ? "" : "(drift)", c.accel_g,
              c.accel_trusted ? "acc" : "GYRO-ONLY",
              l.n / secs, (unsigned long)(l.n ? l.dtMin : 0), (unsigned long)(l.n ? l.dtSum / l.n : 0),
              (unsigned long)l.dtMax, (unsigned long)(l.n ? l.compSum / l.n : 0), (unsigned long)l.compMax,
              (unsigned long)l.secImu, (unsigned long)l.secAtt, (unsigned long)l.secCtl, (unsigned long)l.secSrv,
              l.n ? 100.0f * l.gated / l.n : 0.0f, (unsigned long)l.timeouts, (unsigned long)l.overruns,
              (unsigned long)l.shieldBank, (unsigned long)l.shieldPitch,
              c.roll_sp * RAD_TO_DEG, c.pitch_sp * RAD_TO_DEG, c.pi_link ? "ok" : "none",
              (unsigned long)piRx.gaps, (unsigned long)piRx.bad,
              s.baro_ref ? String(s.baro_alt, 2).c_str() : (s.baro_warm ? "noref" : "warming"),
              s.sonar_m, s.gps.fix, s.gps.sats, s.gps_fresh ? "" : "(stale)", s.gps.ubx ? " ubx" : " nmea",
              s.gps_rate_hz,
              s.rail_valid ? String(s.rail_pin_v * CAL_RAIL_SCALE + CAL_RAIL_OFFSET_V, 2).c_str() : "n/f",
              c.us[0], c.us[1], c.us[2], c.us[3], resultName(lastCmdResult));
}

static void printReport(const CtrlSnap& c, const SensorSnap& s) {
  Link.printf("R fw=%s schema=%d cal=%08lX %s servos_measured=%d imu_verified=%d uncal_allowed=%d last_cal=%d\n",
              FW_NAME, RLG_SCHEMA_VERSION, (unsigned long)cal.id, cal.fromNvs ? "NVS" : "compiled",
              cal.servosMeasured, cal.imuVerified, (int)allowUncal, lastCalResult);
  Link.printf("R cal neutral=%u,%u,%u,%u dir=%d,%d,%d,%d full=%u,%u,%u,%u min=%u,%u,%u,%u max=%u,%u,%u,%u\n",
              cal.neutral[0], cal.neutral[1], cal.neutral[2], cal.neutral[3], cal.dir[0], cal.dir[1], cal.dir[2], cal.dir[3],
              cal.full[0], cal.full[1], cal.full[2], cal.full[3], cal.mn[0], cal.mn[1], cal.mn[2], cal.mn[3],
              cal.mx[0], cal.mx[1], cal.mx[2], cal.mx[3]);
  Link.printf("R imu ok=%d drdy=%.0fHz timing=%s +/-%dg +/-%ddps kin=%s\n", imuOk, drdyRateHz,
              useDrdyInterrupt ? "interrupt" : "poll", 2 << IMU_ACCEL_FS_SEL, 250 << IMU_GYRO_FS_SEL,
              c.euler ? "euler" : "body-rate(viz)");
  Link.printf("R raw body a=(%+.3f,%+.3f,%+.3f)g g=(%+.1f,%+.1f,%+.1f)dps  "
              "[expect level az~-1, nose up ax>0, right wing down ay<0, yaw right gz>0]\n",
              c.ax, c.ay, c.az, c.gx, c.gy, c.gz);
  Link.printf("R kalman samples=%lu gated=%lu dt_rejected=%lu | phase=%s launch_cause=%u apex_cause=%u t_launch=%lu t_apex=%lu peak=%.1fg yaw_aligned=%d\n",
              (unsigned long)c.att_samples, (unsigned long)c.att_gated, (unsigned long)c.att_dt_rej, phaseName(c.phase),
              c.launch_cause, c.apex_cause, (unsigned long)c.t_launch, (unsigned long)c.t_apex, c.peak_accel_g, c.yaw_aligned);
  Link.printf("R baro pa=%.1f ref_pa=%.1f alt=%.2f warm=%d | sonar=%.3f | gps %s fix=%u sats=%u lat=%.7f lon=%.7f "
              "vn=%.2f ve=%.2f vd=%.2f count=%u rate=%.1fHz | home=%d %.7f,%.7f\n",
              s.baro_pa, s.baro_ref_pa, s.baro_alt, s.baro_warm, s.sonar_m, s.gps.ubx ? "UBX" : "NMEA", s.gps.fix,
              s.gps.sats, s.gps.lat_e7 * 1e-7, s.gps.lon_e7 * 1e-7, s.gps.vn, s.gps.ve, s.gps.vd, s.gps.count,
              s.gps_rate_hz, c.home_set, c.home_lat * 1e-7, c.home_lon * 1e-7);
  Link.printf("R pi link=%d ok=%lu bad=%lu gaps=%lu dup=%lu controller=%u speed_cmd=%.2f tlm_gaps(pi)=%u | sp roll=%.1f pitch=%.1f\n",
              c.pi_link, (unsigned long)piRx.ok, (unsigned long)piRx.bad, (unsigned long)piRx.gaps, (unsigned long)piRx.dup,
              c.pi_controller, c.pi_speed_cmd, c.pi_tlm_gaps, c.roll_sp * RAD_TO_DEG, c.pitch_sp * RAD_TO_DEG);
  Link.printf("R rail pin=%.3fV valid=%d | surfaces ail=%.2f elev=%.2f rud=%.2f deg us=%u,%u,%u,%u\n", s.rail_pin_v,
              s.rail_valid, c.ail * RAD_TO_DEG, c.elev * RAD_TO_DEG, c.rud * RAD_TO_DEG, c.us[0], c.us[1], c.us[2], c.us[3]);
  Link.printf("R radio sent=%lu failed=%lu qdrop=%lu cmddrop=%lu | refused warmup=%lu uncal=%lu imu=%lu nofix=%lu status=0x%08lX\n",
              (unsigned long)RadioLink::sent, (unsigned long)RadioLink::failed, (unsigned long)Link.queueDropped,
              (unsigned long)Link.cmdDropped, (unsigned long)refusedNotWarm, (unsigned long)refusedUncal,
              (unsigned long)refusedNoImu, (unsigned long)refusedNoFix, (unsigned long)c.status);
}

static void sendTelemetry(const CtrlSnap& c, const SensorSnap& s) {
  static uint16_t seq = 0;
  rlg_telemetry_t t = {};
  t.sync = RLG_SYNC; t.type = RLG_TYPE_TELEMETRY; t.seq = ++seq;
  t.roll = c.roll_deg * DEG_TO_RAD; t.pitch = c.pitch_deg * DEG_TO_RAD; t.yaw = c.yaw_deg * DEG_TO_RAD;
  t.p = c.p; t.q = c.q; t.r = c.r;
  t.alt_baro = s.baro_alt; t.sonar = s.sonar_m;
  t.lat = s.gps_fresh ? (float)(s.gps.lat_e7 * 1e-7) : NAN; t.lon = s.gps_fresh ? (float)(s.gps.lon_e7 * 1e-7) : NAN;
  t.vn = s.gps.vn; t.ve = s.gps.ve;
  t.rail_adc_v = s.rail_valid ? s.rail_pin_v : -1.0f;
  uint32_t sats = s.gps.sats > 31 ? 31 : s.gps.sats;
  t.status = c.status
           | ((uint32_t)(s.sonar_m >= 0) << RLG_STATUS_SONAR_VALID_LSB)
           | ((uint32_t)(s.gps_fresh ? s.gps.fix : 0) << RLG_STATUS_GPS_FIX_LSB)
           | (sats << RLG_STATUS_GPS_SATS_LSB)
           | ((uint32_t)s.baro_ref << RLG_STATUS_BARO_REF_CAPTURED_LSB)
           | ((uint32_t)s.baro_warm << RLG_STATUS_BARO_WARM_LSB)
           | ((uint32_t)s.rail_valid << RLG_STATUS_RAIL_V_VALID_LSB);
  t.crc8 = rlg_crc8(&t.type, sizeof t - 2);
  Link.sendBinary(&t, sizeof t);
}

static void sendInfo(const CtrlSnap& c, const SensorSnap& s, float loopHz) {
  static uint16_t seq = 0;
  rlg_info_t i = {};
  i.sync = RLG_SYNC; i.type = RLG_TYPE_INFO; i.seq = ++seq;
  i.cal_id = cal.id; i.cal_flags = calFlags(); i.schema = RLG_SCHEMA_VERSION;
  i.loop_hz = (uint16_t)lroundf(loopHz);
  i.pi_rx_gaps = piRx.gaps > 65535 ? 65535 : piRx.gaps;
  i.pi_rx_bad = piRx.bad > 65535 ? 65535 : piRx.bad;
  i.pi_tlm_gaps = c.pi_tlm_gaps;
  i.gps_rate_dhz = (uint8_t)lroundf(clampf(s.gps_rate_hz * 10.0f, 0, 255));
  i.launch_cause = (uint8_t)(c.launch_cause | (c.apex_cause << 4));
  i.pitch_sp = c.pitch_sp; i.roll_sp = c.roll_sp;
  i.crc8 = rlg_crc8(&i.type, sizeof i - 2);
  Link.sendBinary(&i, sizeof i);
}

static void runCommand(const char* cmd, Print& out) {
  if (!strcmp(cmd, "help")) {
    out.println("arm     arm FLIGHT (5 frames; needs warm-up, IMU, calibration and a 3D fix, or 'uncal')");
    out.println("disarm  disarm (surfaces to neutral)");
    out.println("uncal   BENCH OVERRIDE until reboot: allow FLIGHT arming with unmeasured calibration and no GPS fix");
    out.println("tlm     toggle 5 Hz binary telemetry over the radio (default ON)");
    out.println("stream  toggle USB stream for the visualiser (D,t,ax..gz raw body counts + E,t,roll,pitch,gate); '8' also starts it");
    out.println("kin     toggle Euler kinematics (off = integrate body rates exactly like glider_viz.py)");
    out.println("report  full measurement report");
    out.println("srv on|off  enable/disable servo PWM updates (diagnostic)");
    out.println("calclear    erase the calibration pushed over the radio (back to compiled-in; reboot to apply)");
  } else if (!strcmp(cmd, "arm")) {
    for (int i = 0; i < (int)RLG_REARM_CONSECUTIVE_FRAMES; i++) queueConsoleFrame(RLG_MODE_FLIGHT);
    out.println("arm: 5 FLIGHT frames queued");
  } else if (!strcmp(cmd, "disarm")) {
    queueConsoleFrame(RLG_MODE_DISARMED);
    out.println("disarm: queued");
  } else if (!strcmp(cmd, "uncal")) {
    allowUncal = true;
    out.println("uncal: BENCH OVERRIDE until reboot: FLIGHT arming allowed with UNMEASURED calibration and without a GPS fix.");
  } else if (!strcmp(cmd, "tlm")) {
    tlmOn = !tlmOn; out.printf("tlm: %s\n", tlmOn ? "on" : "off");
  } else if (!strcmp(cmd, "stream")) {
    streamOn = !streamOn; out.printf("stream: %s\n", streamOn ? "on" : "off");
  } else if (!strcmp(cmd, "srv off") || !strcmp(cmd, "srv on")) {
    servoWrites = !strcmp(cmd, "srv on"); out.printf("srv: writes %s\n", servoWrites ? "on" : "OFF");
  } else if (!strcmp(cmd, "kin")) {
    wantEuler = !wantEuler; out.printf("kin: %s\n", wantEuler ? "euler" : "body-rate (glider_viz.py)");
  } else if (!strcmp(cmd, "calclear")) {
    calClearNvs(); out.println("calclear: NVS calibration erased; reboot to use the compiled-in calibration");
  } else if (!strcmp(cmd, "report")) {
    CtrlSnap c; SensorSnap s;
    portENTER_CRITICAL(&mux); c = ctrlShared; s = sensShared; portEXIT_CRITICAL(&mux);
    printReport(c, s);
  } else if (*cmd) {
    out.printf("unknown '%s' (type help)\n", cmd);
  }
}

void consoleTask(void*) {
  esp_task_wdt_add(nullptr);
  char line[48]; uint8_t len = 0;
  uint32_t lastStatus = 0, lastTlm = 0, lastInfo = 0;
  float loopHz = 0;
  TickType_t wake = xTaskGetTickCount();
  for (;;) {
    vTaskDelayUntil(&wake, pdMS_TO_TICKS(10));
    esp_task_wdt_reset();
    uint32_t now = millis();

    while (Link.available()) {
      int ch = Link.read();
      if (ch < 0) break;
      if (ch == '8' && len == 0) { streamOn = true; continue; }     // glider_viz.py sends a bare '8'
      if (ch == '\r') continue;
      if (ch == '\n') { line[len] = 0; Link.lockText(); runCommand(line, Link); Link.unlockText(); len = 0; }
      else if (len < sizeof(line) - 1) line[len++] = (char)tolower(ch);
    }

    if (calSaveReq) {                       // flash write: only ever while DISARMED (handleCalibration)
      calSaveReq = false;
      bool ok = calSaveNvs(calToSave);
      if (ok) cal.fromNvs = true;
      lastCalResult = ok ? 3 : 4;
      Link.lockText();
      Link.printf("cal: %08lX %s\n", (unsigned long)calToSave.cal_id, ok ? "applied and saved to NVS" : "applied, NVS SAVE FAILED");
      Link.unlockText();
    }

    StreamRec rec;
    while (xQueueReceive(streamQueue, &rec, 0) == pdTRUE) {
      Serial.printf("D,%lu,%d,%d,%d,%d,%d,%d\n", (unsigned long)rec.t_ms, rec.raw[0], rec.raw[1], rec.raw[2],
                    rec.raw[3], rec.raw[4], rec.raw[5]);
      Serial.printf("E,%lu,%.2f,%.2f,%u\n", (unsigned long)rec.t_ms, rec.roll, rec.pitch, rec.gate);
    }

    CtrlSnap c; SensorSnap s; LoopStats l;
    bool haveStats = false;
    portENTER_CRITICAL(&mux);
    c = ctrlShared; s = sensShared;
    if (now - lastStatus >= STATUS_PERIOD_MS && statsShared.n) { l = statsShared; statsShared = LoopStats(); haveStats = true; }
    portEXIT_CRITICAL(&mux);

    if (haveStats) {
      lastStatus = now;
      loopHz = l.n / (l.dtSum / 1e6f);
      Link.lockText(); printStatus(c, s, l); Link.unlockText();
    }
    if (tlmOn && now - lastTlm >= TELEMETRY_PERIOD_MS) { lastTlm = now; sendTelemetry(c, s); }
    if (tlmOn && now - lastInfo >= INFO_PERIOD_MS) { lastInfo = now; sendInfo(c, s, loopHz); }
  }
}
