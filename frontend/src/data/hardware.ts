// hardware.ts
// ===========
// The as-wired avionics, transcribed from the wiring diagram (Nov 2026) and
// the flight-test brief. One place, so the readiness page and the future
// flight-test panels agree. Where the two sources disagree it is recorded in
// HARDWARE_ISSUES rather than silently resolved.

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
    note: "/dev/serial0 → ttyAMA0, Bluetooth and serial console off. Powered from header pins 2/6, not USB-C." },
  { part: "ICM-20948 IMU", role: "Attitude and rates", bus: "SPI, NCS low",
    pins: "SCL/clock GPIO18, SDA/data in GPIO23, AD0/data out GPIO19, NCS GPIO5, INT GPIO4",
    note: "Board frame is not the body frame; axis signs are being re-derived." },
  { part: "BMP280", role: "Barometric altitude (only continuous altitude source)", bus: "I²C 0x76",
    pins: "SDA GPIO21, SCL GPIO22", note: "±0.05 m noise, ~0.3 m drift, ~2 m warm-up transient over 2 min." },
  { part: "NEO-6M GPS", role: "Position, ground velocity", bus: "UART2 9600",
    pins: "GPS TX → GPIO16, GPS RX ← GPIO17" },
  { part: "RCWL-1655 sonar", role: "Flare trigger below ~4.5 m", bus: "Trigger/echo, 3.3 V only",
    pins: "TRIG GPIO26, ECHO GPIO35" },
  { part: "Aileron L servo", role: "Control surface", bus: "PWM", pins: "GPIO13", note: "MG90S, 600–2300 µs" },
  { part: "Aileron R servo", role: "Control surface", bus: "PWM", pins: "GPIO14", note: "MG90S, 600–2300 µs" },
  { part: "Elevator servo", role: "Control surface", bus: "PWM", pins: "GPIO25",
    note: "Brief: MG996R 1000–2000 µs. Wiring diagram: MG90S. See issues." },
  { part: "Rudder servo", role: "Control surface", bus: "PWM", pins: "GPIO27", note: "MG90S, 600–2300 µs" },
  { part: "3S LiPo → switch → Mini 560 Pro", role: "5.1 V rail", bus: "Power",
    pins: "Switch 6 A+, bulk capacitor 470–1000 µF on the 5 V rail",
    note: "Peak budget ~8.2 A against a ~5 A converter. Below 4.75 V the Pi's SD card can corrupt." },
];

export interface Issue {
  severity: "blocking" | "check";
  title: string;
  detail: string;
}

export const HARDWARE_ISSUES: Issue[] = [
  {
    severity: "blocking",
    title: "No rail-voltage sensing in the wiring diagram",
    detail:
      "Telemetry carries rail_v and the brief makes it the brownout warning, but no divider runs from the 5 V rail to an " +
      "ESP32 ADC pin. It must be an ADC1 pin (GPIO32–39): ADC2 cannot be read while the radio (ESP-NOW) is on. GPIO32 is " +
      "taken by the Pi UART and GPIO35 by the sonar echo, which leaves GPIO34, 36 (VP) or 39 (VN). Until this exists, " +
      "rail_v will be reported as unavailable, not as a number.",
  },
  {
    severity: "check",
    title: "Elevator servo type disagrees",
    detail:
      "The brief says the elevator is an MG996R (1000–2000 µs, 2.5 A stall); the wiring diagram labels it MG90S. Travel " +
      "limits and the power budget depend on which is fitted. Confirm and update this file.",
  },
  {
    severity: "check",
    title: "IMU axis signs are stale",
    detail:
      "The board-to-body transform is being re-derived. The calibration file marks it unverified, and nothing should " +
      "trust attitude until the four bench checks pass and the transform has determinant +1.",
  },
];
