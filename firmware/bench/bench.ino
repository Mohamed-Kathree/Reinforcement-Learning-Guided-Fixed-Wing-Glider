#include <WiFi.h>
#include <esp_now.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_BMP280.h>
#include <ESP32Servo.h>
#include <esp_system.h>

static const uint8_t GROUND_MAC[6] = {0xD4,0x8A,0xFC,0xA4,0xA0,0x08};

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5,5,0)
  #define SEND_CB_ARG const wifi_tx_info_t*
#else
  #define SEND_CB_ARG const uint8_t*
#endif

struct ImuSample {
  int16_t ax, ay, az;        // body frame, raw counts
  int16_t gx, gy, gz;
};

// ---------- pin map ----------
#define IMU_SCK  18        // breakout: SCL
#define IMU_MISO 19        // breakout: AD0
#define IMU_MOSI 23        // breakout: SDA
#define IMU_CS    5        // breakout: NCS
#define IMU_INT   4

#define I2C_SDA  21
#define I2C_SCL  22

#define GPS_RX   16        // <- GPS TX
#define GPS_TX   17        // -> GPS RX

#define SON_TRIG 26
#define SON_ECHO 35

#define SRV_AILL 13
#define SRV_AILR 14
#define SRV_ELEV 25
#define SRV_RUD  27

#define PI_TX    32
#define PI_RX    33

#define RAIL_ADC 34        // rail voltage divider, ~2:1
#define RAIL_SCALE 2.0f    // calibrate against a multimeter

// ---------- IMU axis convention ----------
// Body frame: x forward, y right, z down (aerospace). An accelerometer at
// rest reads +1 g along whichever axis points UP (specific force), so:
//   level           -> az -1 g   (z points down)
//   nose up         -> ax  > 0   (x tilts up)
//   right wing down -> ay  < 0   (y tilts down)
// Measured 2026-11-19 against the soldered mount at the CG:
//   level -> raw az -1.037, nose up -> raw ax +0.256, right wing down -> raw ay -0.281
// The RAW board frame already is the body frame, so every sign is +1.
//
// CORRECTED 2026-11-19 (repo copy): this sketch previously used
// AX -1, AY -1, AZ +1 (gyro likewise), from a frame-derive target that mixed
// the gravity and specific-force conventions. That is Rz(180 deg): det +1,
// so the handedness check passes, but +x pointed at the tail, a level
// airframe read roll ~ +/-180 deg, and the gyro roll/pitch rates were
// inverted (anti-damping in the inner loop). Decision recorded with the user;
// see firmware/flight/README.md. Signs live here and nowhere else.
#define AX_SIGN (+1)
#define AY_SIGN (+1)
#define AZ_SIGN (+1)
#define GX_SIGN (+1)
#define GY_SIGN (+1)
#define GZ_SIGN (+1)

#define REG_BANK_SEL 0x7F
#define WHO_AM_I     0x00
#define PWR_MGMT_1   0x06
#define PWR_MGMT_2   0x07
#define ACCEL_XOUT_H 0x2D

#define ACC_LSB  16384.0f
#define GYRO_LSB   131.0f

SPIClass spi(VSPI);
SPISettings imuSpi(1000000, MSBFIRST, SPI_MODE0);

Adafruit_BMP280 bmp;
Servo sAilL, sAilR, sElev, sRud;

// ---------- transparent radio link ----------
class RadioLink : public Print {
  char     tx[200];
  size_t   txn = 0;
  volatile char rx[128];
  volatile uint8_t head = 0, tail = 0;
  static RadioLink* self;

  static void onRecv(const esp_now_recv_info_t*, const uint8_t* d, int len) {
    if (!self) return;
    for (int i = 0; i < len; i++) {
      uint8_t n = (self->head + 1) % sizeof(self->rx);
      if (n != self->tail) { self->rx[self->head] = d[i]; self->head = n; }
    }
  }

  static void onSent(SEND_CB_ARG, esp_now_send_status_t s) {
    sent++;
    if (s != ESP_NOW_SEND_SUCCESS) failed++;
  }

public:
  static volatile uint32_t sent, failed;

  void begin() {
    Serial.begin(115200);
    delay(400);
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    if (esp_now_init() != ESP_OK) { Serial.println("ESP-NOW init failed"); return; }

    esp_now_peer_info_t p = {};
    memcpy(p.peer_addr, GROUND_MAC, 6);
    p.channel = 0;
    p.encrypt = false;
    esp_now_add_peer(&p);

    self = this;
    esp_now_register_recv_cb(onRecv);
    esp_now_register_send_cb(onSent);

    Serial.printf("radio up, my MAC %s\n", WiFi.macAddress().c_str());
  }

  size_t write(uint8_t c) override {
    Serial.write(c);
    tx[txn++] = c;
    if (c == '\n' || txn >= sizeof(tx)) flushTx();
    return 1;
  }

  void flushTx() {
    if (!txn) return;
    esp_now_send(GROUND_MAC, (const uint8_t*)tx, txn);
    txn = 0;
    delay(8);
  }

  int available() {
    int n = (int)head - (int)tail;
    if (n < 0) n += sizeof(rx);
    return n + Serial.available();
  }

  int read() {
    if (head != tail) {
      char c = rx[tail];
      tail = (tail + 1) % sizeof(rx);
      return c;
    }
    return Serial.read();
  }
};

RadioLink* RadioLink::self = nullptr;
volatile uint32_t RadioLink::sent = 0;
volatile uint32_t RadioLink::failed = 0;
RadioLink Link;

bool keyPressed() { if (Link.available()) { Link.read(); return true; } return false; }

// ---------- IMU low level ----------
uint8_t imuRead(uint8_t reg) {
  spi.beginTransaction(imuSpi);
  digitalWrite(IMU_CS, LOW);
  spi.transfer(reg | 0x80);
  uint8_t v = spi.transfer(0x00);
  digitalWrite(IMU_CS, HIGH);
  spi.endTransaction();
  return v;
}

void imuWrite(uint8_t reg, uint8_t val) {
  spi.beginTransaction(imuSpi);
  digitalWrite(IMU_CS, LOW);
  spi.transfer(reg & 0x7F);
  spi.transfer(val);
  digitalWrite(IMU_CS, HIGH);
  spi.endTransaction();
}

void imuReadBurst(uint8_t reg, uint8_t* buf, uint8_t n) {
  spi.beginTransaction(imuSpi);
  digitalWrite(IMU_CS, LOW);
  spi.transfer(reg | 0x80);
  for (uint8_t i = 0; i < n; i++) buf[i] = spi.transfer(0x00);
  digitalWrite(IMU_CS, HIGH);
  spi.endTransaction();
}

void imuBank(uint8_t b) { imuWrite(REG_BANK_SEL, b << 4); }

ImuSample imuSample() {
  uint8_t b[12];
  imuBank(0);
  imuReadBurst(ACCEL_XOUT_H, b, 12);
  ImuSample s;
  s.ax = AX_SIGN * (int16_t)((b[0]  << 8) | b[1]);
  s.ay = AY_SIGN * (int16_t)((b[2]  << 8) | b[3]);
  s.az = AZ_SIGN * (int16_t)((b[4]  << 8) | b[5]);
  s.gx = GX_SIGN * (int16_t)((b[6]  << 8) | b[7]);
  s.gy = GY_SIGN * (int16_t)((b[8]  << 8) | b[9]);
  s.gz = GZ_SIGN * (int16_t)((b[10] << 8) | b[11]);
  return s;
}

ImuSample imuSampleRaw() {
  uint8_t b[12];
  imuBank(0);
  imuReadBurst(ACCEL_XOUT_H, b, 12);
  ImuSample s;
  s.ax = (int16_t)((b[0]  << 8) | b[1]);
  s.ay = (int16_t)((b[2]  << 8) | b[3]);
  s.az = (int16_t)((b[4]  << 8) | b[5]);
  s.gx = (int16_t)((b[6]  << 8) | b[7]);
  s.gy = (int16_t)((b[8]  << 8) | b[9]);
  s.gz = (int16_t)((b[10] << 8) | b[11]);
  return s;
}

bool imuBegin() {
  pinMode(IMU_CS, OUTPUT);
  digitalWrite(IMU_CS, HIGH);
  pinMode(IMU_INT, INPUT);
  spi.begin(IMU_SCK, IMU_MISO, IMU_MOSI, IMU_CS);
  delay(50);
  imuBank(0);
  uint8_t id = imuRead(WHO_AM_I);
  Link.printf("  WHO_AM_I = 0x%02X (expect 0xEA)\n", id);
  if (id != 0xEA) return false;
  imuWrite(PWR_MGMT_1, 0x80); delay(100);
  imuBank(0);
  imuWrite(PWR_MGMT_1, 0x01); delay(50);
  imuWrite(PWR_MGMT_2, 0x00); delay(50);
  return true;
}

float railVolts() {
  uint32_t sum = 0;
  for (int i = 0; i < 16; i++) { sum += analogRead(RAIL_ADC); delay(2); }
  return (sum / 16.0f) * (3.3f / 4095.0f) * RAIL_SCALE;
}

// ---------- tests ----------
void testIMU() {
  Link.println("\n[IMU] ICM-20948, body frame (x fwd, y right, z down)");
  if (!imuBegin()) {
    Link.println("  FAIL. Check NCS->5, AD0->19, SDA->23, SCL->18.");
    return;
  }
  Link.println("  Level airframe should read az near -1.00 g.");
  Link.println("  Signs may be stale — run 'f' if anything looks wrong.");
  Link.println("  Any key to stop.\n");
  while (!keyPressed()) {
    ImuSample s = imuSample();
    float axg = s.ax / ACC_LSB, ayg = s.ay / ACC_LSB, azg = s.az / ACC_LSB;
    Link.printf("  a %+6.2f %+6.2f %+6.2f g | g %+7.1f %+7.1f %+7.1f dps | roll %+6.1f pitch %+6.1f\n",
      axg, ayg, azg,
      s.gx / GYRO_LSB, s.gy / GYRO_LSB, s.gz / GYRO_LSB,
      atan2f(-ayg, -azg) * 57.2958f,                       // roll, right wing down +
      atan2f(axg, sqrtf(ayg*ayg + azg*azg)) * 57.2958f);   // pitch, nose up +
    delay(200);
  }
}

void deriveFrame() {
  Link.println("\n[FRAME] raw board-frame derivation, against the real mount.");
  if (!imuBegin()) { Link.println("  IMU not responding."); return; }
  Link.println("  Tilt the WHOLE AIRFRAME, not the board.");
  Link.println("  Hold each position, then press a key to capture.\n");

  const char* steps[3] = { "LEVEL", "NOSE UP ~30 deg", "RIGHT WING DOWN ~30 deg" };
  float cap[3][3];

  for (int i = 0; i < 3; i++) {
    Link.printf("  >>> %s — hold it, then press any key\n", steps[i]);
    while (!Link.available()) {
      ImuSample s = imuSampleRaw();
      Link.printf("      raw a %+7.3f %+7.3f %+7.3f g\n",
                  s.ax / ACC_LSB, s.ay / ACC_LSB, s.az / ACC_LSB);
      delay(400);
    }
    while (Link.available()) Link.read();

    float sx = 0, sy = 0, sz = 0;
    for (int k = 0; k < 20; k++) {
      ImuSample s = imuSampleRaw();
      sx += s.ax; sy += s.ay; sz += s.az;
      delay(10);
    }
    cap[i][0] = sx / 20.0f / ACC_LSB;
    cap[i][1] = sy / 20.0f / ACC_LSB;
    cap[i][2] = sz / 20.0f / ACC_LSB;
    Link.printf("      captured  %+6.3f %+6.3f %+6.3f g\n\n",
                cap[i][0], cap[i][1], cap[i][2]);
  }

  Link.println("  --- deltas from level ---");
  for (int i = 1; i < 3; i++)
    Link.printf("  %-26s dx %+6.3f  dy %+6.3f  dz %+6.3f\n", steps[i],
                cap[i][0] - cap[0][0], cap[i][1] - cap[0][1], cap[i][2] - cap[0][2]);

  Link.println("\n  Largest delta identifies the responding axis.");
  Link.println("  Target (specific force, x fwd / y right / z down):");
  Link.println("    nose up -> ax POS, right wing down -> ay NEG, level -> az NEG.");
  Link.println("  A sign is -1 only if that raw axis responds the opposite way.");
  Link.println("  Product of the three accel signs MUST be +1.");
}

void testI2CScan() {
  Link.println("\n[I2C] scanning");
  Wire.begin(I2C_SDA, I2C_SCL);
  int n = 0;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) { Link.printf("  found 0x%02X\n", a); n++; }
  }
  if (!n) Link.println("  FAIL: nothing found.");
}

void testBaro() {
  Link.println("\n[BARO] BMP280, now inside the static port.");
  Wire.begin(I2C_SDA, I2C_SCL);
  if (!(bmp.begin(0x76) || bmp.begin(0x77))) {
    Link.println("  FAIL: nothing at 0x76 or 0x77.");
    return;
  }
  bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,
                  Adafruit_BMP280::SAMPLING_X2,
                  Adafruit_BMP280::SAMPLING_X16,
                  Adafruit_BMP280::FILTER_X16,
                  Adafruit_BMP280::STANDBY_MS_1);
  delay(300);
  float groundPa = bmp.readPressure();
  Link.println("  Zeroed here. ~2 min warm-up transient from cold.");
  Link.println("  Record fresh noise and drift — the port changes them.");
  Link.println("  Any key to stop.\n");
  float lo = 1e9f, hi = -1e9f;
  while (!keyPressed()) {
    float rel = bmp.readAltitude(groundPa / 100.0f);
    if (rel < lo) lo = rel;
    if (rel > hi) hi = rel;
    Link.printf("  %.2f C | %.1f Pa | rel_alt %+6.2f m | spread %.2f m\n",
                bmp.readTemperature(), bmp.readPressure(), rel, hi - lo);
    delay(400);
  }
}

void testGPS() {
  Link.println("\n[GPS] NEO-6M, antenna in its final position. Needs clear sky.");
  Link.println("  Note: the radio is transmitting nearby. Any key to stop.\n");
  Serial2.begin(9600, SERIAL_8N1, GPS_RX, GPS_TX);

  unsigned long t0 = millis(), firstFix = 0, lastByte = millis(), lastPrint = 0;
  int bestSats = 0;
  String line = "";

  while (!keyPressed()) {
    while (Serial2.available()) {
      char c = Serial2.read();
      lastByte = millis();
      if (c == '\n') {
        if (line.startsWith("$GPGGA") || line.startsWith("$GNGGA")) {
          int f[16], n = 0;
          for (int i = 0; i < (int)line.length() && n < 16; i++)
            if (line[i] == ',') f[n++] = i;
          if (n >= 7) {
            int q = line.substring(f[5]+1, f[6]).toInt();
            int s = line.substring(f[6]+1, f[7]).toInt();
            if (s > bestSats) bestSats = s;
            if (q > 0 && firstFix == 0) {
              firstFix = millis() - t0;
              Link.printf("\n  *** FIX at %.1f s, %d sats ***\n\n", firstFix / 1000.0f, s);
            }
            if (millis() - lastPrint > 2000) {
              lastPrint = millis();
              Link.printf("  t=%5.1f s  fix=%d  sats=%2d  %s\n",
                          (millis() - t0) / 1000.0f, q, s,
                          q == 0 ? "(searching)" : "LOCKED");
            }
          }
        }
        line = "";
      } else if (c != '\r') line += c;
    }
    if (millis() - lastByte > 3000) {
      Link.println("  FAIL: no NMEA. Check GPS TX -> GPIO16.");
      lastByte = millis();
    }
    delay(5);
  }

  Link.printf("\n  Summary: ");
  if (firstFix) Link.printf("cold fix in %.1f s, best sats %d\n", firstFix / 1000.0f, bestSats);
  else          Link.printf("NO FIX after %.1f s, best sats %d\n",
                            (millis() - t0) / 1000.0f, bestSats);
}

void testSonar() {
  Link.println("\n[SONAR] RCWL-1655, through the belly opening.");
  pinMode(SON_TRIG, OUTPUT);
  pinMode(SON_ECHO, INPUT);
  Link.println("  Check 0.5 / 1 / 2 / 4 m, confirm no echo past ~4.5 m.");
  Link.println("  Watch for the airframe blocking the cone. Any key to stop.\n");
  while (!keyPressed()) {
    digitalWrite(SON_TRIG, LOW);  delayMicroseconds(3);
    digitalWrite(SON_TRIG, HIGH); delayMicroseconds(10);
    digitalWrite(SON_TRIG, LOW);
    unsigned long us = pulseIn(SON_ECHO, HIGH, 30000);
    if (!us) Link.println("  no echo");
    else     Link.printf("  %.2f m\n", us * 0.000343f / 2.0f);
    delay(300);
  }
}

void centreServos() {
  Link.println("\n[CENTRE] servo neutral points.");
  Link.println("  a/d  -10/+10 us   z/c  -50/+50 us");
  Link.println("  t    pulse +100 us for 1 s, check direction");
  Link.println("  n    next servo    q  finish\n");

  sAilL.setPeriodHertz(50); sAilL.attach(SRV_AILL, 600, 2300);
  sAilR.setPeriodHertz(50); sAilR.attach(SRV_AILR, 600, 2300);
  sElev.setPeriodHertz(50); sElev.attach(SRV_ELEV, 600, 2300);
  sRud .setPeriodHertz(50); sRud .attach(SRV_RUD,  600, 2300);

  Servo* sv[4]      = { &sAilL, &sAilR, &sElev, &sRud };
  const char* nm[4] = { "Aileron L", "Aileron R", "Elevator", "Rudder" };
  int us[4] = { 1500, 1500, 1500, 1500 };

  for (int i = 0; i < 4; i++) sv[i]->writeMicroseconds(us[i]);
  int cur = 0;
  Link.printf("  %s: %d us\n", nm[cur], us[cur]);

  while (true) {
    if (!Link.available()) { delay(10); continue; }
    char c = Link.read();

    if (c == 'q') break;
    if (c == 'n') {
      cur = (cur + 1) % 4;
      Link.printf("\n  %s: %d us\n", nm[cur], us[cur]);
      continue;
    }
    if (c == 't') {
      int probe = constrain(us[cur] + 100, 600, 2300);
      Link.printf("  -> %d us, note the direction\n", probe);
      sv[cur]->writeMicroseconds(probe);
      delay(1000);
      sv[cur]->writeMicroseconds(us[cur]);
      Link.printf("  back to %d us\n", us[cur]);
      continue;
    }

    int step = 0;
    if      (c == 'a') step = -10;
    else if (c == 'd') step =  10;
    else if (c == 'z') step = -50;
    else if (c == 'c') step =  50;
    else continue;

    us[cur] = constrain(us[cur] + step, 600, 2300);
    sv[cur]->writeMicroseconds(us[cur]);
    Link.printf("  %s: %d us\n", nm[cur], us[cur]);
  }

  Link.println("\n  --- neutral points ---");
  for (int i = 0; i < 4; i++) Link.printf("    %-12s %d us\n", nm[i], us[i]);
}

void sweep(Servo& s, const char* name) {
  Link.printf("\n  %s sweeping. Rail: ", name);
  Link.println("watching for sag below 4.75 V. Any key for next.");
  int a = 700; bool up = true;
  unsigned long lastPrint = 0;
  float minV = 99.0f;
  while (!keyPressed()) {
    s.writeMicroseconds(a);
    a += up ? 20 : -20;
    if (a >= 2200) { a = 2200; up = false; }
    if (a <= 700)  { a = 700;  up = true;  }
    float v = railVolts();
    if (v < minV) minV = v;
    if (millis() - lastPrint > 500) {
      lastPrint = millis();
      Link.printf("    rail %.2f V (min %.2f V)\n", v, minV);
    }
    delay(20);
  }
  s.writeMicroseconds(1500);
  Link.printf("  %s done. Minimum rail %.2f V\n", name, minV);
}

void testServos() {
  Link.println("\n[SERVOS] sweep with rail monitoring. Pi 5 V lead OFF.");
  sAilL.setPeriodHertz(50); sAilL.attach(SRV_AILL, 600, 2300);
  sAilR.setPeriodHertz(50); sAilR.attach(SRV_AILR, 600, 2300);
  sElev.setPeriodHertz(50); sElev.attach(SRV_ELEV, 600, 2300);
  sRud .setPeriodHertz(50); sRud .attach(SRV_RUD,  600, 2300);

  sweep(sAilL, "Aileron L");
  sweep(sAilR, "Aileron R");
  sweep(sElev, "Elevator");
  sweep(sRud,  "Rudder");
  Link.println("\n  All centred. Record the per-servo minimums.");
}

void testPiLink() {
  Link.println("\n[UART] Pi link, GPIO32 TX / GPIO33 RX at 115200.");
  Serial1.begin(115200, SERIAL_8N1, PI_RX, PI_TX);
  Link.println("  Run link_test.py on the Pi. Any key to stop.\n");
  String in = "";
  uint32_t got = 0;
  unsigned long report = millis();
  while (!keyPressed()) {
    while (Serial1.available()) {
      char c = Serial1.read();
      if (c == '\n') {
        got++;
        Serial1.println("ESP_PONG");
        in = "";
      } else if (c != '\r') in += c;
    }
    if (millis() - report > 5000) {
      Link.printf("  -- %lu packets in 5 s --\n", got);
      got = 0; report = millis();
    }
  }
}

void testRail() {
  Link.println("\n[RAIL] 16-sample average on GPIO34. Any key to stop.");
  Link.printf("  Scale factor %.2f — calibrate against a multimeter.\n\n", RAIL_SCALE);
  while (!keyPressed()) {
    Link.printf("  rail %.2f V\n", railVolts());
    delay(500);
  }
}

void streamIMU() {
  Link.println("\n[STREAM] body-frame IMU CSV at 50 Hz. Any key to stop.");
  if (!imuBegin()) { Link.println("  IMU not responding."); return; }
  unsigned long next = micros();
  while (!keyPressed()) {
    next += 20000;
    ImuSample s = imuSample();
    Link.printf("D,%lu,%d,%d,%d,%d,%d,%d\n", millis(),
                s.ax, s.ay, s.az, s.gx, s.gy, s.gz);
    while ((long)(micros() - next) < 0) { }
  }
}

void printStatus() {
  esp_reset_reason_t r = esp_reset_reason();
  const char* why;
  switch (r) {
    case ESP_RST_POWERON:  why = "power-on";           break;
    case ESP_RST_SW:       why = "software reset";     break;
    case ESP_RST_PANIC:    why = "panic / exception";  break;
    case ESP_RST_BROWNOUT: why = "BROWNOUT";           break;
    case ESP_RST_EXT:      why = "external reset pin"; break;
    case ESP_RST_WDT:      why = "watchdog";           break;
    default:               why = "other";              break;
  }
  Link.printf("\nlast reset : %s (%d)\n", why, (int)r);
  Link.printf("rail       : %.2f V\n", railVolts());
  Link.printf("radio      : %lu sent, %lu failed (%.1f%%)\n",
    RadioLink::sent, RadioLink::failed,
    RadioLink::sent ? 100.0 * RadioLink::failed / RadioLink::sent : 0.0);
}

void menu() {
  Link.println("\n===== RL glider — assembled board =====");
  Link.println("  f  Frame derive   tilt the airframe, get axis signs");
  Link.println("  0  Servo centre   neutrals + direction");
  Link.println("  1  IMU            body-frame readings");
  Link.println("  2  I2C scan");
  Link.println("  3  Barometer      in the static port");
  Link.println("  4  GPS            final antenna position");
  Link.println("  5  Sonar          through the belly");
  Link.println("  6  Servo sweep    with rail monitoring");
  Link.println("  7  Pi link        UART1");
  Link.println("  8  IMU stream     CSV for the visualiser");
  Link.println("  9  Status         reset reason, rail, radio");
  Link.print("> ");
}

void setup() {
  Link.begin();
  analogReadResolution(12);
  analogSetPinAttenuation(RAIL_ADC, ADC_11db);
  delay(700);
  printStatus();
  menu();
}

void loop() {
  if (!Link.available()) return;
  char c = Link.read();
  while (Link.available()) Link.read();
  switch (c) {
    case 'f': deriveFrame();  break;
    case '0': centreServos(); break;
    case '1': testIMU();      break;
    case '2': testI2CScan();  break;
    case '3': testBaro();     break;
    case '4': testGPS();      break;
    case '5': testSonar();    break;
    case '6': testServos();   break;
    case '7': testPiLink();   break;
    case '8': streamIMU();    break;
    case '9': printStatus();  break;
    default:  break;
  }
  menu();
}