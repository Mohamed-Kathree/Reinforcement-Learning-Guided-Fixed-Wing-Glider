/*
 * flight.ino -- RL glider flight firmware, PHASE 1: inner loop, no Pi.
 *
 *   core 1, loopTask (raised priority): the 200 Hz control loop. Woken by the
 *     ICM-20948 data-ready interrupt (GPIO 4, ~225 Hz); dt comes from the
 *     interrupt timestamps. IMU read -> Kalman attitude -> command frames ->
 *     rlg_safety -> inner loop (FLIGHT) -> servos. It never prints, never
 *     touches I2C/UART/radio and never calls esp_now_send.
 *   core 0, sensorTask (50 Hz): barometer, sonar, GPS, rail ADC.
 *   core 0, radioTask: the only caller of esp_now_send (15 ms settle each).
 *   core 0, consoleTask (100 Hz): text console (USB and radio), 1 Hz status
 *     line, optional 5 Hz binary telemetry, optional USB stream for the
 *     visualiser.
 *
 * In FLIGHT with no Pi (this phase), the Pi link is always "timed out", so
 * the aircraft always flies the failsafe glide setpoints (config.h).
 * Console: type "help". See firmware/flight/README.md for the bench test plan.
 */
#include <Arduino.h>
#include <ESP32Servo.h>
#include "config.h"
#include "hw_imu.h"
#include "hw_sensors.h"
#include "radio_link.h"
#include "src/core/attitude.h"
#include "src/core/inner_loop.h"
#include "src/protocol/rlg_protocol.h"
#include "src/protocol/rlg_safety.h"
#include "src/generated/calibration_gen.h"

#define FW_NAME "rlg-flight phase1"
#define STREAM_DECIMATE 2            // USB stream at ~112 Hz (225 / 2); USB only, never radio

static RadioLink Link;
static Servo servos[RLG_NUM_SERVOS];
static const int SERVO_PINS[RLG_NUM_SERVOS] = {PIN_SRV_AILL, PIN_SRV_AILR, PIN_SRV_ELEV, PIN_SRV_RUD};
static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
static QueueHandle_t cmdQueue, streamQueue;
static TaskHandle_t ctrlTask = nullptr;

// ---------------- shared: sensors (written by sensorTask) ----------------
struct SensorSnap {
  float baro_alt = NAN;          // m above the arm-time reference; NaN before capture
  float baro_pa = NAN, baro_ref_pa = NAN;
  bool baro_ref = false, baro_warm = false;
  float sonar_m = -1.0f;         // -1 = no valid echo
  uint8_t fix = 0, sats = 0;
  bool gps_fresh = false;
  double lat = NAN, lon = NAN;
  float vn = NAN, ve = NAN, speed = NAN;
  float rail_pin_v = 0.0f;
  bool rail_valid = false;
};
static SensorSnap sensShared;
static volatile bool baroCaptureReq = false;

// ---------------- shared: control (written by the control loop) ----------------
struct CtrlSnap {
  float roll_deg = 0, pitch_deg = 0, yaw_deg = 0, p = 0, q = 0, r = 0;   // rates rad/s
  float accel_g = 0;
  float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0;   // body frame, g and deg/s (frame check)
  bool seeded = false, accel_trusted = false, euler = true;
  uint8_t mode = 0; bool armed = false, failsafe = false, airborne = false;
  uint16_t us[RLG_NUM_SERVOS] = {0};
  float ail = 0, elev = 0, rud = 0;    // rad, last inner-loop output
  uint32_t status = 0;
  uint32_t att_samples = 0, att_gated = 0, att_dt_rej = 0;
};
static CtrlSnap ctrlShared;

struct LoopStats {
  uint32_t n = 0, dtMin = UINT32_MAX, dtMax = 0, compMax = 0;
  uint64_t dtSum = 0, compSum = 0;
  uint32_t gated = 0, timeouts = 0, overruns = 0, shieldBank = 0, shieldPitch = 0;
};
static LoopStats statsShared;

static void mergeStats(LoopStats& a, const LoopStats& b) {
  if (b.dtMin < a.dtMin) a.dtMin = b.dtMin;
  if (b.dtMax > a.dtMax) a.dtMax = b.dtMax;
  if (b.compMax > a.compMax) a.compMax = b.compMax;
  a.n += b.n; a.dtSum += b.dtSum; a.compSum += b.compSum; a.gated += b.gated;
  a.timeouts += b.timeouts; a.overruns += b.overruns; a.shieldBank += b.shieldBank; a.shieldPitch += b.shieldPitch;
}

struct StreamRec { uint32_t t_ms; int16_t raw[6]; float roll, pitch; uint8_t gate; };

// ---------------- console -> control flags ----------------
static volatile bool wantEuler = ATT_EULER_KINEMATICS;
static volatile bool allowUncal = false;      // per boot, typed "uncal"
static volatile bool streamOn = false, tlmOn = false;

// ---------------- control-loop-owned state ----------------
static rlg_attitude_t att;
static rlg_safety_t st;
static bool useDrdyInterrupt = false;
static float drdyRateHz = 0;
static bool imuOk = false;
static volatile uint32_t drdyTs = 0, drdyCount = 0;
static volatile int lastCmdResult = -1;
static volatile uint32_t refusedNotWarm = 0, refusedUncal = 0, refusedNoImu = 0;

static void IRAM_ATTR drdyIsr() {
  drdyTs = micros();
  drdyCount++;
  BaseType_t hpw = pdFALSE;
  if (ctrlTask) vTaskNotifyGiveFromISR(ctrlTask, &hpw);
  portYIELD_FROM_ISR(hpw);
}

static rlg_sensors_t safetySensors() {
  rlg_sensors_t s;
  portENTER_CRITICAL(&mux);
  s.baro_alt_m = sensShared.baro_alt;
  s.gps_fix = sensShared.gps_fresh ? sensShared.fix : 0;
  s.gps_speed_ms = (sensShared.gps_fresh && !isnan(sensShared.speed)) ? sensShared.speed : 0.0f;
  portEXIT_CRITICAL(&mux);
  return s;
}

static bool flightArmed() { return st.armed && st.mode == RLG_MODE_FLIGHT; }

// One command frame (ESP-NOW or console) -> rlg_safety. Control loop only.
static void handleFrame(CmdFrame& f, uint32_t now_ms, const rlg_sensors_t& sens) {
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
    if (!(CAL_SERVOS_MEASURED && CAL_IMU_VERIFIED) && !allowUncal) { refusedUncal++; lastCmdResult = 102; return; }
  }
  bool airborneBefore = rlg_safety_is_airborne(&st, now_ms, &sens);
  rlg_cmd_result_t res = rlg_safety_on_command(&st, now_ms, &sens, f.data, RLG_COMMAND_SIZE);
  lastCmdResult = (int)res;
  // Ground pressure reference: captured on FLIGHT arm (after warm-up), never at
  // boot, and never re-captured while airborne.
  if (res == RLG_CMD_ARMED && st.mode == RLG_MODE_FLIGHT && !airborneBefore) baroCaptureReq = true;
}

static void writeServos() {
  for (int i = 0; i < RLG_NUM_SERVOS; i++) servos[i].writeMicroseconds(st.servo_us[i]);
}

// ======================= setup =======================
void setup() {
  Serial.setTxBufferSize(2048);
  Serial.begin(115200);
  delay(400);
  cmdQueue = xQueueCreate(16, sizeof(CmdFrame));
  streamQueue = xQueueCreate(64, sizeof(StreamRec));
  Link.begin(cmdQueue);

  Link.printf("\n%s  cal %s  servos_measured=%d imu_verified=%d rail_calibrated=%d\n",
              FW_NAME, CAL_HASH, CAL_SERVOS_MEASURED, CAL_IMU_VERIFIED, CAL_RAIL_CALIBRATED);
  Link.printf("imu frame signs a(%d,%d,%d) g(%d,%d,%d)\n", AX_SIGN, AY_SIGN, AZ_SIGN, GX_SIGN, GY_SIGN, GZ_SIGN);
  if (!CAL_SERVOS_MEASURED)
    Link.println("WARNING: servo calibration NOT measured (placeholders). FLIGHT arming refused unless 'uncal' is typed.");

  // Servos first, at neutral, so they hold still through the rest of boot.
  uint16_t neutral[RLG_NUM_SERVOS], mn[RLG_NUM_SERVOS], mx[RLG_NUM_SERVOS];
  for (int i = 0; i < RLG_NUM_SERVOS; i++) {
    neutral[i] = CAL_NEUTRAL_US[i]; mn[i] = CAL_MIN_US[i]; mx[i] = CAL_MAX_US[i];
  }
  rlg_safety_init(&st, neutral, millis());
  if (!rlg_safety_set_limits(&st, mn, mx)) Link.println("ERROR: calibration min > max; using absolute envelope");
  for (int i = 0; i < RLG_NUM_SERVOS; i++) {
    servos[i].setPeriodHertz(50);
    servos[i].attach(SERVO_PINS[i], RLG_ABS_MIN_US, RLG_ABS_MAX_US);
  }
  writeServos();

  imuOk = imuBegin(&Link);
  if (imuOk) {
    imuConfigureRateAndInterrupt();
    // Measure the data-ready rate. No edges -> timed polling, and say so.
    attachInterrupt(digitalPinToInterrupt(PIN_IMU_INT), drdyIsr, RISING);
    uint32_t c0 = drdyCount, t0 = millis();
    while (millis() - t0 < 500) { imuSampleBody(); delay(2); }   // reading clears the flag
    drdyRateHz = (drdyCount - c0) * 2.0f;
    useDrdyInterrupt = drdyRateHz > 50.0f;
    if (!useDrdyInterrupt) detachInterrupt(digitalPinToInterrupt(PIN_IMU_INT));
    Link.printf("imu: data-ready %.0f Hz measured (expect ~%d) -> %s\n", drdyRateHz,
                (int)(1125 / (1 + IMU_SMPLRT_DIV)),
                useDrdyInterrupt ? "INTERRUPT timing" : "NO EDGES: timed POLLING at 200 Hz, dt from micros()");
  } else {
    Link.println("imu: FAILED. Attitude is invalid and FLIGHT cannot be armed.");
  }
  rlg_attitude_init(&att, ATT_EULER_KINEMATICS);

  baroBegin(&Link);
  sonarBegin();
  gpsBegin();
  analogReadResolution(12);
  analogSetPinAttenuation(PIN_RAIL_ADC, ADC_11db);
  Link.printf("baro: ground reference is captured on FLIGHT arm; arming allowed after %u s warm-up\n",
              (unsigned)(CAL_BARO_WARMUP_MS / 1000));
  Link.println("type 'help'");

  extern void sensorTask(void*);
  extern void radioTask(void*);
  extern void consoleTask(void*);
  xTaskCreatePinnedToCore(sensorTask, "sensors", 4096, nullptr, 3, nullptr, 0);
  xTaskCreatePinnedToCore(radioTask, "radio", 4096, nullptr, 2, nullptr, 0);
  xTaskCreatePinnedToCore(consoleTask, "console", 6144, nullptr, 1, nullptr, 0);

  ctrlTask = xTaskGetCurrentTaskHandle();
  vTaskPrioritySet(nullptr, 10);
}

// ======================= 200 Hz control loop (core 1) =======================
void loop() {
  static uint32_t lastTs = 0, lastCount = 0, n = 0;
  static TickType_t lastWake = xTaskGetTickCount();
  static LoopStats ls;
  uint32_t ts;
  bool timedOut = false;

  if (useDrdyInterrupt) {
    if (ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(IMU_DRDY_TIMEOUT_MS)) > 0) {
      ts = drdyTs;
      uint32_t c = drdyCount;
      if (lastCount && c - lastCount > 1) ls.overruns += c - lastCount - 1;   // missed samples
      lastCount = c;
    } else { ts = micros(); timedOut = true; ls.timeouts++; }
  } else {
    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(5));
    ts = micros();
  }
  uint32_t t0 = micros();
  uint32_t now_ms = millis();

  // ---- attitude ----
  ImuRaw raw = imuOk ? imuSampleBody() : ImuRaw{0, 0, 0, 0, 0, 0};
  float ax = raw.ax / IMU_ACC_LSB, ay = raw.ay / IMU_ACC_LSB, az = raw.az / IMU_ACC_LSB;
  float gx = raw.gx / IMU_GYRO_LSB, gy = raw.gy / IMU_GYRO_LSB, gz = raw.gz / IMU_GYRO_LSB;
  att.euler_kinematics = wantEuler;
  float dt = lastTs ? (ts - lastTs) * 1e-6f : 0.0f;
  uint32_t dtUs = lastTs ? ts - lastTs : 0;
  lastTs = ts;
  if (imuOk) rlg_attitude_update(&att, ax, ay, az, gx, gy, gz, dt);
  float roll = att.roll.angle * DEG_TO_RAD, pitch = att.pitch.angle * DEG_TO_RAD;
  float p = gx * DEG_TO_RAD, q = gy * DEG_TO_RAD, r = gz * DEG_TO_RAD;

  // ---- commands and safety ----
  rlg_sensors_t sens = safetySensors();
  CmdFrame f;
  while (xQueueReceive(cmdQueue, &f, 0) == pdTRUE) handleFrame(f, now_ms, sens);
  rlg_safety_tick(&st, now_ms, &sens);

  // ---- inner loop (FLIGHT only) ----
  rlg_surfaces_t s = {0};
  if (flightArmed()) {
    // Phase 1: no Pi link, so rlg_safety_pi_failsafe() is always true and the
    // setpoints are always the failsafe glide. Phase 2 uses the Pi's setpoints
    // when !rlg_safety_pi_failsafe(&st).
    float roll_sp = FAILSAFE_ROLL_DEG * DEG_TO_RAD, pitch_sp = FAILSAFE_PITCH_DEG * DEG_TO_RAD;
    s = rlg_inner_loop(roll, pitch, p, q, r, roll_sp, pitch_sp);
    uint16_t us[RLG_NUM_SERVOS];
    // Both ailerons take the same signed command (+ = roll right); each servo's
    // calibrated direction makes its own surface move the right way.
    const float defl[RLG_NUM_SERVOS] = {s.aileron, s.aileron, s.elevator, s.rudder};
    const float mx[RLG_NUM_SERVOS] = {IL_AIL_MAX, IL_AIL_MAX, IL_ELEV_MAX, IL_RUD_MAX};
    for (int i = 0; i < RLG_NUM_SERVOS; i++)
      us[i] = rlg_deflection_to_us(defl[i], mx[i], CAL_NEUTRAL_US[i], CAL_DIRECTION[i],
                                   CAL_FULL_US[i], CAL_MIN_US[i], CAL_MAX_US[i]);
    rlg_safety_flight_output(&st, us);
  }
  writeServos();
  uint32_t comp = micros() - t0;

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
  (void)timedOut;

  uint32_t status = rlg_safety_status_bits(&st);
  portENTER_CRITICAL(&mux);
  ctrlShared.roll_deg = att.roll.angle; ctrlShared.pitch_deg = att.pitch.angle; ctrlShared.yaw_deg = att.yaw_deg;
  ctrlShared.p = p; ctrlShared.q = q; ctrlShared.r = r;
  ctrlShared.accel_g = att.accel_mag_g; ctrlShared.seeded = att.seeded;
  ctrlShared.ax = ax; ctrlShared.ay = ay; ctrlShared.az = az;
  ctrlShared.gx = gx; ctrlShared.gy = gy; ctrlShared.gz = gz;
  ctrlShared.accel_trusted = att.accel_trusted; ctrlShared.euler = att.euler_kinematics;
  ctrlShared.mode = st.mode; ctrlShared.armed = st.armed;
  ctrlShared.failsafe = rlg_safety_pi_failsafe(&st); ctrlShared.airborne = st.airborne;
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
  uint32_t lastSonar = 0, lastRail = 0;
  for (;;) {
    vTaskDelayUntil(&wake, pdMS_TO_TICKS(SENSOR_PERIOD_MS));
    uint32_t now = millis();
    gpsPoll(now);

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

    bool fresh = gps.last_gga_ms && (now - gps.last_gga_ms) < GPS_STALE_MS;
    portENTER_CRITICAL(&mux);
    sensShared.baro_pa = pa; sensShared.baro_ref_pa = refPa; sensShared.baro_alt = alt;
    sensShared.baro_ref = !isnan(refPa); sensShared.baro_warm = now >= CAL_BARO_WARMUP_MS;
    sensShared.sonar_m = sonar;
    sensShared.fix = gps.fix; sensShared.sats = gps.sats; sensShared.gps_fresh = fresh;
    sensShared.lat = gps.lat; sensShared.lon = gps.lon;
    sensShared.vn = gps.vn; sensShared.ve = gps.ve; sensShared.speed = gps.speed_ms;
    sensShared.rail_pin_v = railV; sensShared.rail_valid = railV >= RAIL_FITTED_MIN_V;
    portEXIT_CRITICAL(&mux);
  }
}

// ======================= radioTask (core 0) =======================
void radioTask(void*) { Link.serviceForever(); }

// ======================= consoleTask (core 0, 100 Hz) =======================
static void queueConsoleFrame(uint8_t mode) {
  rlg_command_t c = {};
  c.sync = RLG_SYNC; c.type = RLG_TYPE_COMMAND; c.mode = mode;
  for (int i = 0; i < RLG_NUM_SERVOS; i++) c.servo_us[i] = CAL_NEUTRAL_US[i];
  c.crc8 = rlg_crc8(&c.type, sizeof c - 2);   // seq and CRC are redone by the control loop
  CmdFrame f; memcpy(f.data, &c, sizeof c); f.src = 1;
  xQueueSend(cmdQueue, &f, pdMS_TO_TICKS(50));
}

static const char* modeName(uint8_t m) {
  switch (m) { case RLG_MODE_FLIGHT: return "FLIGHT"; case RLG_MODE_BENCH: return "BENCH";
               case RLG_MODE_REPLAY: return "REPLAY"; default: return "DISARMED"; }
}

static const char* resultName(int r) {
  switch (r) {
    case -1: return "-"; case 0: return "accepted"; case 1: return "arming"; case 2: return "ARMED";
    case 3: return "disarmed"; case 4: return "rejected:frame"; case 5: return "rejected:stale";
    case 6: return "rejected:airborne"; case 100: return "refused:imu";
    case 101: return "refused:baro-warmup"; case 102: return "refused:uncalibrated";
    default: return "?";
  }
}

static void printStatus(const CtrlSnap& c, const SensorSnap& s, const LoopStats& l) {
  float secs = l.n ? l.dtSum / 1e6f : 1.0f;
  Link.printf("S t=%lu %s%s%s rpy=%.1f,%.1f,%.1f(drift) |a|=%.2f %s | loop=%.0fHz dt=%lu/%lu/%luus comp=%lu/%luus "
              "gated=%.0f%% tmo=%lu ovr=%lu shield=%lu/%lu | baro=%s son=%.2f gps=%u/%u%s rail=%s | us=%u,%u,%u,%u cmd=%s\n",
              (unsigned long)(millis() / 1000), modeName(c.mode), c.armed ? "" : "(unarmed)",
              (c.armed && c.mode == RLG_MODE_FLIGHT) ? (c.failsafe ? " FAILSAFE" : " PI") : "",
              c.roll_deg, c.pitch_deg, c.yaw_deg, c.accel_g, c.accel_trusted ? "acc" : "GYRO-ONLY",
              l.n / secs, (unsigned long)(l.n ? l.dtMin : 0), (unsigned long)(l.n ? l.dtSum / l.n : 0),
              (unsigned long)l.dtMax, (unsigned long)(l.n ? l.compSum / l.n : 0), (unsigned long)l.compMax,
              l.n ? 100.0f * l.gated / l.n : 0.0f, (unsigned long)l.timeouts, (unsigned long)l.overruns,
              (unsigned long)l.shieldBank, (unsigned long)l.shieldPitch,
              s.baro_ref ? String(s.baro_alt, 2).c_str() : (s.baro_warm ? "noref" : "warming"),
              s.sonar_m, s.fix, s.sats, s.gps_fresh ? "" : "(stale)",
              s.rail_valid ? String(s.rail_pin_v * CAL_RAIL_SCALE + CAL_RAIL_OFFSET_V, 2).c_str() : "n/f",
              c.us[0], c.us[1], c.us[2], c.us[3], resultName(lastCmdResult));
}

static void printReport(const CtrlSnap& c, const SensorSnap& s) {
  Link.printf("R fw=%s cal=%s servos_measured=%d imu_verified=%d uncal_allowed=%d\n", FW_NAME, CAL_HASH,
              CAL_SERVOS_MEASURED, CAL_IMU_VERIFIED, (int)allowUncal);
  Link.printf("R imu ok=%d drdy=%.0fHz timing=%s signs a(%d,%d,%d) g(%d,%d,%d) kin=%s\n", imuOk, drdyRateHz,
              useDrdyInterrupt ? "interrupt" : "poll", AX_SIGN, AY_SIGN, AZ_SIGN, GX_SIGN, GY_SIGN, GZ_SIGN,
              c.euler ? "euler" : "body-rate(viz)");
  Link.printf("R raw body a=(%+.3f,%+.3f,%+.3f)g g=(%+.1f,%+.1f,%+.1f)dps  "
              "[expect level az~-1, nose up ax>0, right wing down ay<0, yaw right gz>0]\n",
              c.ax, c.ay, c.az, c.gx, c.gy, c.gz);
  Link.printf("R kalman Q_ANGLE=%.4f Q_BIAS=%.4f R=%.3f gate=%.2f-%.2fg samples=%lu gated=%lu dt_rejected=%lu\n",
              ATT_Q_ANGLE, ATT_Q_BIAS, ATT_R_MEASURE, ATT_GATE_LO_G, ATT_GATE_HI_G, (unsigned long)c.att_samples,
              (unsigned long)c.att_gated, (unsigned long)c.att_dt_rej);
  Link.printf("R baro pa=%.1f ref_pa=%.1f alt=%.2f warm=%d | sonar=%.3f | gps fix=%u sats=%u lat=%.6f lon=%.6f "
              "vn=%.2f ve=%.2f sentences=%lu badsum=%lu\n", s.baro_pa, s.baro_ref_pa, s.baro_alt, s.baro_warm,
              s.sonar_m, s.fix, s.sats, s.lat, s.lon, s.vn, s.ve, (unsigned long)gps.sentences,
              (unsigned long)gps.bad_checksum);
  Link.printf("R rail pin=%.3fV valid=%d scaled=%.2fV (brownout %.2fV, divider %s)\n", s.rail_pin_v, s.rail_valid,
              s.rail_pin_v * CAL_RAIL_SCALE + CAL_RAIL_OFFSET_V, CAL_RAIL_BROWNOUT_V,
              s.rail_valid ? "reads" : "NOT FITTED");
  Link.printf("R surfaces ail=%.2f elev=%.2f rud=%.2f deg  us=%u,%u,%u,%u  failsafe sp roll=%.1f pitch=%.1f\n",
              c.ail * RAD_TO_DEG, c.elev * RAD_TO_DEG, c.rud * RAD_TO_DEG, c.us[0], c.us[1], c.us[2], c.us[3],
              FAILSAFE_ROLL_DEG, FAILSAFE_PITCH_DEG);
  Link.printf("R radio sent=%lu failed=%lu qdrop=%lu cmddrop=%lu | refused warmup=%lu uncal=%lu imu=%lu status=0x%08lX\n",
              (unsigned long)RadioLink::sent, (unsigned long)RadioLink::failed, (unsigned long)Link.queueDropped,
              (unsigned long)Link.cmdDropped, (unsigned long)refusedNotWarm, (unsigned long)refusedUncal,
              (unsigned long)refusedNoImu, (unsigned long)c.status);
}

static void sendTelemetry(const CtrlSnap& c, const SensorSnap& s) {
  static uint16_t seq = 0;
  rlg_telemetry_t t = {};
  t.sync = RLG_SYNC; t.type = RLG_TYPE_TELEMETRY; t.seq = ++seq;
  t.roll = c.roll_deg * DEG_TO_RAD; t.pitch = c.pitch_deg * DEG_TO_RAD; t.yaw = c.yaw_deg * DEG_TO_RAD;
  t.p = c.p; t.q = c.q; t.r = c.r;
  t.alt_baro = s.baro_alt; t.sonar = s.sonar_m;
  t.lat = s.gps_fresh ? (float)s.lat : NAN; t.lon = s.gps_fresh ? (float)s.lon : NAN;
  t.vn = s.vn; t.ve = s.ve;
  t.rail_adc_v = s.rail_valid ? s.rail_pin_v : -1.0f;
  uint32_t sats = s.sats > 31 ? 31 : s.sats;
  t.status = c.status
           | ((uint32_t)(s.sonar_m >= 0) << RLG_STATUS_SONAR_VALID_LSB)
           | ((uint32_t)(s.gps_fresh ? s.fix : 0) << RLG_STATUS_GPS_FIX_LSB)
           | (sats << RLG_STATUS_GPS_SATS_LSB)
           | ((uint32_t)s.baro_ref << RLG_STATUS_BARO_REF_CAPTURED_LSB)
           | ((uint32_t)s.baro_warm << RLG_STATUS_BARO_WARM_LSB)
           | ((uint32_t)s.rail_valid << RLG_STATUS_RAIL_V_VALID_LSB);
  t.crc8 = rlg_crc8(&t.type, sizeof t - 2);
  Link.sendBinary(&t, sizeof t);
}

static void runCommand(const char* cmd, Print& out) {
  if (!strcmp(cmd, "help")) {
    out.println("arm     arm FLIGHT (5 frames; needs warm-up, IMU, calibration or 'uncal'); flies failsafe glide");
    out.println("disarm  disarm (surfaces to neutral)");
    out.println("uncal   allow FLIGHT arming with UNMEASURED servo calibration, this boot only (bench)");
    out.println("tlm     toggle 5 Hz binary telemetry over the radio (default off)");
    out.println("stream  toggle USB stream for the visualiser (D,t,ax..gz raw body counts + E,t,roll,pitch,gate); '8' also starts it");
    out.println("kin     toggle Euler kinematics (off = integrate body rates exactly like glider_viz.py)");
    out.println("report  full measurement report");
  } else if (!strcmp(cmd, "arm")) {
    for (int i = 0; i < (int)RLG_REARM_CONSECUTIVE_FRAMES; i++) queueConsoleFrame(RLG_MODE_FLIGHT);
    out.println("arm: 5 FLIGHT frames queued");
  } else if (!strcmp(cmd, "disarm")) {
    queueConsoleFrame(RLG_MODE_DISARMED);
    out.println("disarm: queued");
  } else if (!strcmp(cmd, "uncal")) {
    allowUncal = true;
    out.println("uncal: FLIGHT arming allowed with UNMEASURED calibration until reboot. Bench only: surface directions are unverified.");
  } else if (!strcmp(cmd, "tlm")) {
    tlmOn = !tlmOn; out.printf("tlm: %s\n", tlmOn ? "on" : "off");
  } else if (!strcmp(cmd, "stream")) {
    streamOn = !streamOn; out.printf("stream: %s\n", streamOn ? "on" : "off");
  } else if (!strcmp(cmd, "kin")) {
    wantEuler = !wantEuler; out.printf("kin: %s\n", wantEuler ? "euler" : "body-rate (glider_viz.py)");
  } else if (!strcmp(cmd, "report")) {
    CtrlSnap c; SensorSnap s;
    portENTER_CRITICAL(&mux); c = ctrlShared; s = sensShared; portEXIT_CRITICAL(&mux);
    printReport(c, s);
  } else if (*cmd) {
    out.printf("unknown '%s' (type help)\n", cmd);
  }
}

void consoleTask(void*) {
  char line[48]; uint8_t len = 0;
  uint32_t lastStatus = 0, lastTlm = 0;
  TickType_t wake = xTaskGetTickCount();
  for (;;) {
    vTaskDelayUntil(&wake, pdMS_TO_TICKS(10));
    uint32_t now = millis();

    while (Link.available()) {
      int ch = Link.read();
      if (ch < 0) break;
      if (ch == '8' && len == 0) { streamOn = true; continue; }     // glider_viz.py sends a bare '8'
      if (ch == '\r') continue;
      if (ch == '\n') { line[len] = 0; Link.lockText(); runCommand(line, Link); Link.unlockText(); len = 0; }
      else if (len < sizeof(line) - 1) line[len++] = (char)tolower(ch);
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

    if (haveStats) { lastStatus = now; Link.lockText(); printStatus(c, s, l); Link.unlockText(); }
    if (tlmOn && now - lastTlm >= TELEMETRY_PERIOD_MS) { lastTlm = now; sendTelemetry(c, s); }
  }
}
