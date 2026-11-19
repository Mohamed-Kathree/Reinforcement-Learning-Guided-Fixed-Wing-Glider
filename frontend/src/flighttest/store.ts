// store.ts
// ========
// One shared client for the flight-test backend (backend/routers/flighttest.py):
// a WebSocket for the ground-station snapshot and telemetry history (5 Hz),
// a 3 s poll of /api/ft/status for the Pi and stub state, and the actions the
// panels call. A module-level store (useSyncExternalStore) so every panel on
// every page sees the same state over one socket.
//
// Servo commands: the UI may call setServos() on every slider or scrub event;
// this file sends at most one request per 50 ms (trailing), and the backend
// independently caps what goes over the radio at 20 Hz.
import { useSyncExternalStore } from "react";
import { API_BASE, WS_BASE } from "../config";

export interface StatusBits {
  mode: number; armed: number; airborne: number; sonar_valid: number; gps_fix: number; gps_sats: number;
  baro_ref_captured: number; baro_warm: number; rail_v_valid: number; timeout_disarmed: number;
  cmd_rejected_airborne: number; pi_failsafe: number;
}

export interface RailState {
  volts: number | null; level: "ok" | "low" | "brownout" | null; brownout_v: number;
  calibrated: boolean; measured: boolean; note: string | null;
}

export interface Telemetry {
  seq: number; t_ms: number; roll: number; pitch: number; yaw: number; p: number; q: number; r: number;
  alt_baro: number; sonar: number | null; lat: number; lon: number; vn: number; ve: number;
  rail_adc_v: number | null; rail_v: number | null; rail: RailState; status: StatusBits; mode: string;
}

export interface GroundState {
  connected: boolean; kind: string | null; port: string | null; rate_hz: number; loss_pct: number | null;
  crc_drops: number; rssi_dbm: number | null; last_packet_age_ms: number | null; receiving: boolean;
  received_total: number; lost_total: number; commands_sent: number;
}

export interface Session { mode: string; servo_us: number[]; expires_in_ms: number }

// Aircraft-side state reported by the flight computer's 1 Hz Info packet (schema v3).
export interface AircraftInfo {
  t_ms: number; cal_id_hex: string; cal_servos_measured: boolean; cal_imu_verified: boolean; cal_from_nvs: boolean;
  schema: number; loop_hz: number; pi_rx_gaps: number; pi_rx_bad: number; pi_tlm_gaps: number; gps_rate_hz: number;
  launch_cause: number; apex_cause: number; pitch_sp: number; roll_sp: number; file_cal_id_hex: string;
}

export interface CalibrationPush {
  cal_id_hex: string; sends: number; first_ms: number; last_ms: number | null;
  state: "sending" | "confirmed" | "not confirmed"; confirmed_ms?: number;
}

export interface ConsoleLine { n: number; t_ms: number; text: string; sent?: boolean }

export interface Snapshot {
  ground: GroundState;
  telemetry: (Telemetry & { phase?: string; status: StatusBits & Record<string, number> }) | null;
  session: Session | null;
  last_session_end: { reason: string; at_ms: number } | null;
  info?: AircraftInfo | null;
  calibration_push?: CalibrationPush | null;
  file_cal_id_hex?: string;
  console?: ConsoleLine[];
}

export interface PiLog { name: string; size_bytes: number; modified_utc: string }

export interface PiStatus {
  reachable: boolean; kind: string; address: string; hostname?: string; ip?: string;
  disk_free_bytes?: number; disk_total_bytes?: number; logs?: PiLog[]; error?: string; logs_error?: string;
}

export interface StubState { flying: boolean; airborne: boolean; loss: number; rail_sensor: boolean; pi_alive: boolean }

export interface HistoryPoint {
  t_ms: number; roll: number; pitch: number; r: number; alt_baro: number; sonar: number | null;
  rail_v: number | null; gps_sats: number; gps_fix: number; loss_pct: number | null;
}

interface State {
  backend: "unknown" | "ok" | "missing" | "down";   // missing = backend without /api/ft (restart it)
  socket: "connecting" | "open" | "closed";
  snapshot: Snapshot | null;
  pi: PiStatus | null;
  stub: StubState | null;
  history: HistoryPoint[];
  error: string | null;
  logsVersion: number;     // bumped after every successful log fetch
}

const HISTORY_MAX = 300;
let state: State = { backend: "unknown", socket: "closed", snapshot: null, pi: null, stub: null, history: [], error: null, logsVersion: 0 };
const listeners = new Set<() => void>();
let started = false;

function set(patch: Partial<State>) {
  state = { ...state, ...patch };
  listeners.forEach((l) => l());
}

async function api<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(API_BASE + path, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      if (j?.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

async function pollStatus() {
  try {
    const st = await api<Snapshot & { pi: PiStatus; stub: StubState | null }>("GET", "/api/ft/status");
    set({ backend: "ok", pi: st.pi, stub: st.stub, ...(state.socket !== "open" ? { snapshot: st } : {}) });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    set({ backend: msg.startsWith("404") ? "missing" : "down" });
  }
}

let ws: WebSocket | null = null;
let backoff = 1000;

function openSocket() {
  set({ socket: "connecting" });
  try {
    ws = new WebSocket(`${WS_BASE}/ws/ft`);
  } catch {
    set({ socket: "closed" });
    setTimeout(openSocket, backoff);
    return;
  }
  ws.onopen = () => {
    backoff = 1000;
    set({ socket: "open" });
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data) as { snapshot: Snapshot; history: HistoryPoint[] };
    const history = msg.history.length
      ? [...state.history, ...msg.history].slice(-HISTORY_MAX)
      : state.history;
    set({ snapshot: msg.snapshot, history });
  };
  ws.onclose = () => {
    ws = null;
    set({ socket: "closed" });
    setTimeout(openSocket, backoff);
    backoff = Math.min(backoff * 2, 15000);
  };
  ws.onerror = () => ws?.close();
}

function start() {
  if (started) return;
  started = true;
  pollStatus();
  setInterval(pollStatus, 3000);
  openSocket();
}

export function useFlightTest(): State {
  start();
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => state,
  );
}

async function act<T>(fn: () => Promise<T>): Promise<T | null> {
  try {
    const out = await fn();
    set({ error: null });
    pollStatus();
    return out;
  } catch (e) {
    set({ error: e instanceof Error ? e.message : String(e) });
    return null;
  }
}

// ---- servo throttle: at most one POST per 50 ms, always sending the latest values
let pendingServos: number[] | null = null;
let pendingEnvelope = false;
let servoTimer: ReturnType<typeof setTimeout> | null = null;
let lastServoSend = 0;

function flushServos() {
  servoTimer = null;
  if (!pendingServos) return;
  const us = pendingServos;
  pendingServos = null;
  lastServoSend = performance.now();
  api("POST", "/api/ft/servos", { servo_us: us, envelope: pendingEnvelope })
    .catch((e) => set({ error: e instanceof Error ? e.message : String(e) }));
}

// ---- surface (replay-on-aircraft) throttle: same 50 ms trailing pattern
let pendingSurf: [number, number, number] | null = null;
let surfTimer: ReturnType<typeof setTimeout> | null = null;
let lastSurfSend = 0;

function flushSurfaces() {
  surfTimer = null;
  if (!pendingSurf) return;
  const [aileron, elevator, rudder] = pendingSurf;
  pendingSurf = null;
  lastSurfSend = performance.now();
  api("POST", "/api/ft/surfaces", { aileron, elevator, rudder })
    .catch((e) => set({ error: e instanceof Error ? e.message : String(e) }));
}

export const ft = {
  connect: (transport: "stub" | "serial", port?: string) =>
    act(() => api("POST", `/api/ft/connect?transport=${transport}${port ? `&port=${encodeURIComponent(port)}` : ""}`)),
  disconnect: () => act(() => api("POST", "/api/ft/disconnect")),
  // Probes every serial port for a ground station carrying valid telemetry (needs the flight computer on).
  connectAuto: () => act(() => api("POST", "/api/ft/connect?transport=serial&port=auto")),
  // Simulator surface deflections (rad) -> servos through the saved calibration (replay on the aircraft).
  surfaces: (aileron: number, elevator: number, rudder: number) => {
    pendingSurf = [aileron, elevator, rudder];
    if (surfTimer) return;
    const wait = Math.max(0, 50 - (performance.now() - lastSurfSend));
    surfTimer = setTimeout(flushSurfaces, wait);
  },
  ports: () => api<{ available: boolean; reason: string | null; ports: { device: string; description: string }[] }>(
    "GET", "/api/ft/ports"),
  arm: (mode: "bench" | "replay") => act(() => api("POST", `/api/ft/arm?mode=${mode}`)),
  disarm: () => {
    pendingServos = null;
    return act(() => api("POST", "/api/ft/disarm"));
  },
  keepalive: () => act(() => api("POST", "/api/ft/keepalive")),
  // envelope: calibration wizard only (BENCH): clamp to the servo type's full range,
  // not the current calibration's limits, so the real end stops can be found.
  setServos: (us: number[], envelope = false) => {
    pendingEnvelope = envelope;
    pendingServos = us.map((v) => Math.round(v));
    if (servoTimer) return;
    const wait = Math.max(0, 50 - (performance.now() - lastServoSend));
    servoTimer = setTimeout(flushServos, wait);
  },
  stub: (params: Partial<{ flying: boolean; loss: number; rail_sensor: boolean; pi_alive: boolean; flight_arm: boolean }>) =>
    act(() => api("POST", "/api/ft/stub?" + new URLSearchParams(
      Object.entries(params).map(([k, v]) => [k, String(v)])))),
  setPi: (url: string | null) => act(() => api("POST", "/api/ft/pi" + (url ? `?url=${encodeURIComponent(url)}` : ""))),
  fetchLog: async (name: string) => {
    const r = await act(() => api("POST", `/api/ft/pi/logs/${encodeURIComponent(name)}/fetch`));
    if (r) set({ logsVersion: state.logsVersion + 1 });
    return r;
  },
  localLogs: () => api<{ name: string; size_bytes: number }[]>("GET", "/api/ft/logs"),
  log: (name: string) => api<ParsedLog>("GET", `/api/ft/logs/${encodeURIComponent(name)}`),
  calibration: () => api<{ calibration: Calibration; problems: string[] }>("GET", "/api/ft/calibration"),
  saveCalibration: (cal: Calibration) => act(() => api("PUT", "/api/ft/calibration", cal)),
  pushCalibration: () => act(() => api("POST", "/api/ft/calibration/push")),
  console: (line: string) => act(() => api("POST", "/api/ft/console", { line })),
  clearError: () => set({ error: null }),
};

export interface ServoCal {
  type: string; neutral_us: number; direction: 1 | -1; min_us: number; max_us: number;
  sweep_min_us: number; sweep_max_us: number; measured: boolean; measured_on: string | null;
  us_full_deflection: number;
}

export interface Calibration {
  version: number;
  servos: Record<string, ServoCal>;
  imu: { board_to_body: number[][]; verified: boolean; verified_on: string | null };
  baro: { warmup_s: number };
  rail: { adc_gpio: number; scale: number; offset_v: number; calibrated: boolean; calibrated_on: string | null; brownout_v: number };
  [k: string]: unknown;
}

export interface LogFrame {
  t: number; pos_ned: [number, number, number] | null; euler: [number, number, number];
  rates: [number, number, number]; alt_baro: number; sonar: number | null; rail_v: number | null;
  rail_adc_v: number | null; vel_ne: [number, number]; servo_us: number[]; action: number[] | null;
  mode: string; armed: boolean; gps_fix: number; gps_sats: number;
}

export interface ParsedLog {
  name: string;
  header: { source?: string; created_utc?: string; note?: string; [k: string]: unknown };
  home: { lat: number; lon: number; derived?: string } | null;
  frames: LogFrame[];
  skipped: { line: number; reason: string }[];
  summary: {
    n_frames: number; duration_s: number; rate_hz: number | null; frames_without_position: number;
    max_alt_m: number | null; min_rail_v: number | null; skipped_lines: number;
    rail_scaled_with: { scale: number; offset_v: number; calibrated: boolean } | null;
  };
}

export const SERVO_NAMES = ["aileron_l", "aileron_r", "elevator", "rudder"] as const;
export const SERVO_LABELS: Record<string, string> = {
  aileron_l: "Aileron L", aileron_r: "Aileron R", elevator: "Elevator", rudder: "Rudder",
};
export const SERVO_GPIO: Record<string, number> = { aileron_l: 13, aileron_r: 14, elevator: 25, rudder: 27 };
