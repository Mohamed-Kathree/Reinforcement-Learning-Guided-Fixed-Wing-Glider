// hardware.ts
// ===========
// The as-wired avionics, from the wiring diagram and the 18 Nov 2026 hardware
// answers. One place, so the readiness page and the flight-test panels agree.
// Per-servo limits are NOT here: they live in flighttest/config/calibration.json.

export interface Connection {
  part: string;
  role: string;
  bus: string;
  pins: string;
  note?: string;
}

export const AVIONICS: Connection[] = [
  { part: "ESP32 WROOM-32", role: "Flight controller: sensors 200 Hz, servos, safety", bus: "—", pins: "—" },
  { part: "Raspberry Pi 4B", role: "Policy 20 Hz, full-rate log to SD", bus: "UART 115200",
    pins: "Pi pin 8 GPIO14 TXD → ESP32 GPIO33; Pi pin 10 GPIO15 RXD ← ESP32 GPIO32",
    note: "/dev/serial0 → ttyAMA0, Bluetooth and serial console off. Powered from header pins 2/6, not USB-C. No setpoint for 150 ms in FLIGHT → failsafe glide." },
  { part: "ICM-20948 IMU", role: "Attitude and rates", bus: "SPI, NCS low",
    pins: "SCL/clock GPIO18, SDA/data in GPIO23, AD0/data out GPIO19, NCS GPIO5, INT GPIO4",
    note: "Board frame is not the body frame; axis signs are being re-derived." },
  { part: "BMP280", role: "Barometric altitude (only continuous altitude source)", bus: "I²C 0x76",
    pins: "SDA GPIO21, SCL GPIO22", note: "±0.05 m noise, ~0.3 m drift, ~2 m warm-up transient over 2 min. Reference captured on arm, never at boot." },
  { part: "NEO-6M GPS", role: "Position, ground velocity", bus: "UART2 9600",
    pins: "GPS TX → GPIO16, GPS RX ← GPIO17", note: "Speed only votes 'airborne' with a fix." },
  { part: "RCWL-1655 sonar", role: "Flare trigger below ~4.5 m", bus: "Trigger/echo, 3.3 V only",
    pins: "TRIG GPIO26, ECHO GPIO35" },
  { part: "Aileron L servo", role: "Control surface", bus: "PWM", pins: "GPIO13", note: "MG90S" },
  { part: "Aileron R servo", role: "Control surface", bus: "PWM", pins: "GPIO14", note: "MG90S" },
  { part: "Elevator servo", role: "Control surface", bus: "PWM", pins: "GPIO25", note: "MG90S (MG996R replaced, Nov 2026)" },
  { part: "Rudder servo", role: "Control surface", bus: "PWM", pins: "GPIO27", note: "MG90S" },
  { part: "Rail-voltage divider", role: "5 V rail sensing", bus: "ADC1", pins: "GPIO34, ~2:1 divider",
    note: "5 V lands near 2.5 V, mid-band of the ADC. The ESP32 sends the pin voltage; the scale factor lives in the calibration file." },
  { part: "3S LiPo → switch → Mini 560 Pro", role: "5.1 V rail", bus: "Power",
    pins: "Switch 6 A+, bulk capacitor 470–1000 µF on the 5 V rail",
    note: "Peak budget ~5.6 A against a ~5 A converter (was ~8.2 A with the MG996R). Below 4.75 V the Pi's SD card can corrupt." },
];

export interface Issue {
  severity: "blocking" | "check";
  title: string;
  detail: string;
}

export const HARDWARE_ISSUES: Issue[] = [
  {
    severity: "check",
    title: "Rail voltage is indicative until calibrated",
    detail:
      "The divider on GPIO34 is uncalibrated. Measure the rail with a multimeter at two loads and set rail.scale " +
      "(and offset_v if needed) in the calibration file. The ESP32 ADC is nonlinear below ~0.15 V and above ~3.0 V; " +
      "a ~2:1 divider keeps 5 V mid-band. Until then the dashboard labels every rail reading as indicative.",
  },
  {
    severity: "check",
    title: "Power budget slightly over on paper",
    detail:
      "Four MG90S servos put the peak at ~5.6 A against a ~5 A converter. The bulk capacitor covers transients, " +
      "but the rail minimum under full servo load has not been measured. Measure it before flight.",
  },
  {
    severity: "check",
    title: "IMU axis signs are stale",
    detail:
      "The board-to-body transform is being re-derived. The calibration file marks it unverified, and nothing should " +
      "trust attitude until the four bench checks pass and the transform has determinant +1.",
  },
];
