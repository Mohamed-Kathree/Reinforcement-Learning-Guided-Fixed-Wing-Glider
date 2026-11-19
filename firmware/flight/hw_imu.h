/*
 * hw_imu.h -- ICM-20948 over SPI.
 *
 * imuRead/imuWrite/imuReadBurst/imuBank/imuBegin are the VERIFIED bench code
 * (firmware/bench/bench.ino), unchanged except that log output goes through a
 * Print* argument.
 *
 * NEW, NOT YET VERIFIED ON HARDWARE: imuConfigureRateAndInterrupt() sets the
 * output data rate, the digital low-pass filter and the data-ready interrupt on
 * GPIO 4. Register addresses are from the ICM-20948 datasheet (DS-000189).
 * flight.ino measures the actual data-ready rate at boot and falls back to
 * timed polling if no edges arrive, and reports which it is using.
 */
#pragma once
#include <SPI.h>
#include "config.h"
#include "src/core/imu_frame.h"

#define ICM_REG_BANK_SEL   0x7F
// bank 0
#define ICM_WHO_AM_I       0x00
#define ICM_USER_CTRL      0x03
#define ICM_PWR_MGMT_1     0x06
#define ICM_PWR_MGMT_2     0x07
#define ICM_INT_PIN_CFG    0x0F
#define ICM_INT_ENABLE_1   0x11
#define ICM_ACCEL_XOUT_H   0x2D
// bank 2
#define ICM_GYRO_SMPLRT_DIV    0x00
#define ICM_GYRO_CONFIG_1      0x01
#define ICM_ODR_ALIGN_EN       0x09
#define ICM_ACCEL_SMPLRT_DIV_1 0x10
#define ICM_ACCEL_SMPLRT_DIV_2 0x11
#define ICM_ACCEL_CONFIG       0x14

static SPIClass imuSpiBus(VSPI);
static SPISettings imuSpiCfg(IMU_SPI_HZ, MSBFIRST, SPI_MODE0);

static uint8_t imuRead(uint8_t reg) {
  imuSpiBus.beginTransaction(imuSpiCfg);
  digitalWrite(PIN_IMU_CS, LOW);
  imuSpiBus.transfer(reg | 0x80);
  uint8_t v = imuSpiBus.transfer(0x00);
  digitalWrite(PIN_IMU_CS, HIGH);
  imuSpiBus.endTransaction();
  return v;
}

static void imuWrite(uint8_t reg, uint8_t val) {
  imuSpiBus.beginTransaction(imuSpiCfg);
  digitalWrite(PIN_IMU_CS, LOW);
  imuSpiBus.transfer(reg & 0x7F);
  imuSpiBus.transfer(val);
  digitalWrite(PIN_IMU_CS, HIGH);
  imuSpiBus.endTransaction();
}

static void imuReadBurst(uint8_t reg, uint8_t* buf, uint8_t n) {
  imuSpiBus.beginTransaction(imuSpiCfg);
  digitalWrite(PIN_IMU_CS, LOW);
  imuSpiBus.transfer(reg | 0x80);
  for (uint8_t i = 0; i < n; i++) buf[i] = imuSpiBus.transfer(0x00);
  digitalWrite(PIN_IMU_CS, HIGH);
  imuSpiBus.endTransaction();
}

static void imuBank(uint8_t b) { imuWrite(ICM_REG_BANK_SEL, b << 4); }

static bool imuBegin(Print* log) {
  pinMode(PIN_IMU_CS, OUTPUT);
  digitalWrite(PIN_IMU_CS, HIGH);
  pinMode(PIN_IMU_INT, INPUT);
  imuSpiBus.begin(PIN_IMU_SCK, PIN_IMU_MISO, PIN_IMU_MOSI, PIN_IMU_CS);
  delay(50);
  imuBank(0);
  uint8_t id = imuRead(ICM_WHO_AM_I);
  log->printf("imu: WHO_AM_I = 0x%02X (expect 0xEA)\n", id);
  if (id != 0xEA) return false;
  imuWrite(ICM_PWR_MGMT_1, 0x80); delay(100);
  imuBank(0);
  imuWrite(ICM_PWR_MGMT_1, 0x01); delay(50);
  imuWrite(ICM_PWR_MGMT_2, 0x00); delay(50);
  return true;
}

// NEW (unverified): ODR, DLPF, data-ready interrupt. Full scales stay at the
// power-on defaults (+/-2 g, +/-250 dps) that the bench sketch and its LSB
// constants assume.
static void imuConfigureRateAndInterrupt() {
  imuBank(2);
  imuWrite(ICM_GYRO_SMPLRT_DIV, IMU_SMPLRT_DIV);
  imuWrite(ICM_GYRO_CONFIG_1, (IMU_DLPF_CFG << 3) | (0 << 1) | 0x01);   // DLPF, +/-250 dps, FCHOICE=1
  imuWrite(ICM_ACCEL_SMPLRT_DIV_1, (IMU_SMPLRT_DIV >> 8) & 0x0F);
  imuWrite(ICM_ACCEL_SMPLRT_DIV_2, IMU_SMPLRT_DIV & 0xFF);
  imuWrite(ICM_ACCEL_CONFIG, (IMU_DLPF_CFG << 3) | (0 << 1) | 0x01);    // DLPF, +/-2 g, FCHOICE=1
  imuWrite(ICM_ODR_ALIGN_EN, 0x01);
  imuBank(0);
  imuWrite(ICM_INT_PIN_CFG, 0x00);     // active high, push-pull, 50 us pulse
  imuWrite(ICM_INT_ENABLE_1, 0x01);    // RAW_DATA_0_RDY_EN
}

struct ImuRaw { int16_t ax, ay, az, gx, gy, gz; };

// Burst read with the board-to-body transform applied HERE and nowhere else.
static ImuRaw imuSampleBody() {
  uint8_t b[12];
  imuReadBurst(ICM_ACCEL_XOUT_H, b, 12);       // bank 0 is left selected after config
  ImuRaw s;
  s.ax = AX_SIGN * (int16_t)((b[0] << 8) | b[1]);
  s.ay = AY_SIGN * (int16_t)((b[2] << 8) | b[3]);
  s.az = AZ_SIGN * (int16_t)((b[4] << 8) | b[5]);
  s.gx = GX_SIGN * (int16_t)((b[6] << 8) | b[7]);
  s.gy = GY_SIGN * (int16_t)((b[8] << 8) | b[9]);
  s.gz = GZ_SIGN * (int16_t)((b[10] << 8) | b[11]);
  return s;
}
