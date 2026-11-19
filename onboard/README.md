# onboard/ — the Raspberry Pi flight program

> **Status: tested on the PC against the simulator and the real byte protocol.
> NOT yet run on the Pi or connected to the ESP32.**

The Pi runs the trained controller at 20 Hz. It reads telemetry from the ESP32 over the
UART and sends back a bank-angle and airspeed command. The ESP32 does everything
time-critical:
- 225 Hz attitude estimation and stabilisation;
- servos;
- every safety limit;
- the failsafe glide;
- **when** the Pi's commands are flown at all.

The Pi can't move a surface outside the AUTONOMOUS phase.

```
 ESP32 ── PiTelemetry 50 Hz (95 B) ──▶ Pi   attitude, rates, baro, sonar, GPS pos/vel, home,
                                             phase/status, servo outputs, link counters
 ESP32 ◀── PiSetpoint 20 Hz (16 B) ── Pi    bank angle, speed command, controller, gaps
 UART 115200 8N1: ESP32 GPIO32 TX → Pi pin 10 (RXD), ESP32 GPIO33 RX ← Pi pin 8 (TXD), common GND
```

## What it flies

| `controller` (config.json) | |
|---|---|
| **`gated`** (default) | Run 4 policy (`checkpoints_v24_run4/best_model.zip`) from the apex; one-way handover to the scripted baseline at the first on-board wind estimate > 1.5 m/s within 8–15 s (`training/configs/base.yaml` gate.run4.window). |
| `policy` | The Run 4 policy alone. |
| `baseline` | The scripted return-to-launch controller alone. |

The policy runs in **numpy**. There is no torch on the Pi: `export_policy.py` writes the
weights and normalisation statistics to `model/run4.npz`, and the export stores SHA-256
hashes of the source checkpoint. The gate, the baseline and the packet codec are
**byte-identical copies** in `vendor/`, made by `vendor_sync.py`. Only the imports are
rewritten, so the Pi needs nothing beyond numpy and pyserial. A test fails if a copy
goes stale.

## A flight, phase by phase (the ESP32 decides the phase)

| ESP32 phase | ESP32 flies | Pi |
|---|---|---|
| DISARMED | surfaces at neutral | heartbeat setpoints (`controller` = 0, never flown) |
| ARMED | failsafe glide (wings level, −2°) | opens `logs/flight_<UTC>.rlglog`, heartbeat |
| LAUNCH (throw detected) | wings level, elevator neutral (zoom climb) | heartbeat |
| AUTONOMOUS (apex detected) | **the Pi's setpoints** while they are fresh; failsafe glide otherwise | controller reset (empty frame stack, gate reset, the same as a training episode start, which begins at the apex), then 20 Hz control |
| back to DISARMED | neutral | log closed |

If telemetry stops for more than 0.25 s, the runner stops acting and only sends
heartbeats. The ESP32's own 150 ms Pi timeout then makes it fly the failsafe glide.

## Decisions this relies on (19 Nov 2026)

- **Pitch:** the ESP32 turns the speed command into a pitch target with the simulator's
  trim schedule, **without** the true-airspeed correction (there is no pitot).
  **Measured cost (20 Nov 2026, 200 development seeds, Run 4 gated, paired):** success
  falls from 111 to 72 at stage 0 (p < 0.001), 58 to 32 at stage 1 (p = 0.001), 34 to 14
  at stage 2 (p = 0.002) and 17 to 6 at stage 3 (p = 0.019). Crashes go 0→3, 8→14,
  19→26 and 78→73. See `results/onboard_airspeed_term_dev.json`. **This decision is under
  review:** the policy depends on that speed correction.
- **Yaw:** aligned once to the GPS course after the throw, then gyro-integrated.
- **Home:** the GPS position at FLIGHT arming. The barometric zero is also captured at arming.
- **Hand-off at the apex,** not at the throw, because training starts at the apex
  (`HANDOFF_AT_APEX` in `firmware/flight/config.h`).

## Install on the Pi

The Pi must be powered from the aircraft supply or a proper 5 V supply, **never from a
PC USB port**.

```bash
git clone <this repo>   # or git pull; the model .npz is tracked
cd Reinforcement-Learning-Guided-Fixed-Wing-Glider
sudo bash onboard/deploy/install.sh          # checks the UART; installs python3-numpy, python3-serial;
                                             # self-test; installs + starts rlglider-onboard.service
journalctl -u rlglider-onboard -f            # one status line every 5 s
```

The script only **checks** the UART (`enable_uart=1`, no serial console on
`/dev/serial0`). `--configure-uart` applies raspi-config's serial settings (reboot
afterwards). It never touches the network parameters on the kernel command line.

Flight logs are written to `logs/` in the checkout. `flighttest/pi_server.py` serves them
read-only to the dashboard over Wi-Fi, on the ground only.

## Files

| File | |
|---|---|
| `runner.py` | The 20 Hz program (systemd service). Phases, heartbeats, logging. |
| `link.py` | UART link: PiTelemetry parsing (sync + CRC8, never blocks), sequence gaps, PiSetpoint. |
| `obs.py` | PiTelemetry → observation v1, channel for channel as `env/glider_env.py _build_obs`. |
| `policy.py` | Numpy policy + the evaluation observation pipeline (8-frame stack, frozen normalisation). |
| `controllers.py` | gated / policy / baseline; action → setpoint (bank = a0·45°, speed = 12.75 + a1·3.25 m/s). |
| `flightlog.py` | `.rlglog` v2 writer (the format of `flighttest/logs.py`, replayable in the dashboard). |
| `config.json` | Controller, model, gate, serial device, rates, log folder. |
| `vendor/` | **Generated**: copies of `env/nav_features.py`, `env/wind_estimate.py`, `baseline/wind_gate.py`, `baseline/wind_gated.py`, `baseline/deterministic_rtl.py`, `flighttest/protocol/packets.py`. |
| `vendor_sync.py` | `python -m onboard.vendor_sync [--check]` (dev PC). |
| `export_policy.py` | SB3 checkpoint → `model/run4.npz` (dev PC; needs torch). |
| `sim_check.py` | Dev PC: the runner closed-loop against the simulator through the wire protocol; the airspeed-term measurement. |
| `deploy/` | `install.sh`, `rlglider-onboard.service`. |

## Verified on the PC (tests/test_onboard.py)

- Numpy policy vs SB3 deterministic actions: max difference < 1e-5.
- Observation pipeline vs `training/benchmark.py PolicyActor` on a simulated episode: < 1e-5.
- Through the wire protocol (PiTelemetry bytes in, PiSetpoint bytes out), vs the
  repository's `WindGatedActor` on recorded simulator observations: actions within 1e-3
  (2.4e-4 typical), and the **same handover step** on a stage-3 seed that hands over.
  The residual comes from positions travelling as 1e-7° integers (about 1 cm) and from
  float32 arithmetic. In closed loop, two simulated flights drift apart from that tiny
  difference, as any two runs with different rounding would.
- Runner: heartbeats before the apex, control from AUTONOMOUS, stale telemetry
  → heartbeat, and the log written and readable by `flighttest.logs.parse`.

## Not yet verified (needs hardware)

- The UART link on the Pi (`/dev/serial0` mapping, timing under load).
- 20 Hz timing on the Pi 4B. The forward pass is a 96→128→128→2 MLP in numpy, which
  should be well under 1 ms, but this needs measuring.
- Launch and apex detection thresholds on real throws.
- Yaw alignment and GPS UBX configuration on the real NEO-6M.
