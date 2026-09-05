// useMetricsStream.ts
// ====================
// Fixes D1 (frontend/src/components/TrainingPanel.tsx's old unbounded
// `setMetrics(prev => [...prev, metric])` -- at emit_freq=200 and a 20M-step
// target that's ~100k appends, each one re-rendering and rebuilding a
// multi-megabyte SVG path string; unusable well before a real run finishes).
//
// Incoming TrainingMetric records are pushed into a plain array held in a
// useRef -- O(1) amortised append, triggers NO re-render. A 4 Hz interval
// timer is the only thing that ever calls setState, producing an
// LTTB-downsampled (lttb.ts) snapshot capped at DISPLAY_POINT_BUDGET points
// per metric. Message arrival rate and render rate are now fully decoupled,
// which is the actual fix -- not the charting library, which just needs to
// be able to redraw a few thousand points at 4 Hz without falling over
// (uPlot handles that trivially).
//
// One WebSocket per subscribed run id, keyed the same way B8's run browser
// and multi-run overlay both consume this hook -- there's no separate code
// path for "the live run" vs "a past run" vs "an overlaid run", they're all
// just entries in the same runIds array.
import { useEffect, useRef, useState } from "react";
import { downsample, type Point } from "./lttb";
import type { TrainingMetric } from "../../types";

const WS_BASE = "ws://localhost:8000/ws/training";
const DECIMATE_INTERVAL_MS = 250; // 4 Hz
const DISPLAY_POINT_BUDGET = 1500;

// "" is the sentinel for the live/default run (no ?run= query param --
// backend/routers/training.py resolves that to the newest file in
// data/runs/).
export const LIVE_RUN_ID = "";

export const METRIC_KEYS = [
  "success_rate",
  "ep_rew_mean",
  "ep_len_mean",
  "explained_variance",
  "approx_kl",
  "clip_fraction",
  "entropy_loss",
  "value_loss",
] as const;
export type MetricKey = (typeof METRIC_KEYS)[number];

export interface StageBand {
  startX: number;
  endX: number;
  stage: number;
}

export interface RunSnapshot {
  runId: string;
  connected: boolean;
  points: Record<MetricKey, Point[]>;
  stageBands: StageBand[];
  latest: TrainingMetric | null;
  count: number; // total records received (pre-decimation) -- for the status bar / Gate B checks
}

function emptySnapshot(runId: string): RunSnapshot {
  const points = {} as Record<MetricKey, Point[]>;
  for (const key of METRIC_KEYS) points[key] = [];
  return { runId, connected: false, points, stageBands: [], latest: null, count: 0 };
}

function computeStageBands(raw: TrainingMetric[]): StageBand[] {
  const bands: StageBand[] = [];
  for (const m of raw) {
    const last = bands[bands.length - 1];
    if (last && last.stage === m.stage) {
      last.endX = m.timesteps;
    } else {
      bands.push({ startX: m.timesteps, endX: m.timesteps, stage: m.stage });
    }
  }
  return bands;
}

interface RunBuffers {
  raw: TrainingMetric[];
  ws: WebSocket | null;
  lastProcessedLen: number; // memoization: skip recompute if raw hasn't grown since the last tick
}

export function useMetricsStream(runIds: string[]) {
  const buffersRef = useRef<Map<string, RunBuffers>>(new Map());
  const [snapshots, setSnapshots] = useState<Record<string, RunSnapshot>>({});

  // Open/close WebSockets to track exactly the requested run ids.
  useEffect(() => {
    const buffers = buffersRef.current;
    const wanted = new Set(runIds);

    for (const runId of wanted) {
      if (buffers.has(runId)) continue;
      const url = runId === LIVE_RUN_ID ? WS_BASE : `${WS_BASE}?run=${encodeURIComponent(runId)}`;
      const ws = new WebSocket(url);
      const entry: RunBuffers = { raw: [], ws, lastProcessedLen: -1 };
      buffers.set(runId, entry);

      ws.onmessage = (ev) => {
        try {
          entry.raw.push(JSON.parse(ev.data) as TrainingMetric);
        } catch {
          // ignore malformed lines
        }
      };
      ws.onclose = () => {
        setSnapshots((prev) => {
          const existing = prev[runId];
          if (!existing) return prev;
          return { ...prev, [runId]: { ...existing, connected: false } };
        });
      };
    }

    for (const [runId, entry] of buffers) {
      if (!wanted.has(runId)) {
        entry.ws?.close();
        buffers.delete(runId);
        setSnapshots((prev) => {
          if (!(runId in prev)) return prev;
          const next = { ...prev };
          delete next[runId];
          return next;
        });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runIds.join(",")]);

  // Close everything on unmount. buffersRef.current is always the same Map
  // instance for the life of this hook (only ever mutated in place, never
  // reassigned), but copy it to a local so the cleanup doesn't read
  // `.current` directly.
  useEffect(() => {
    const buffers = buffersRef.current;
    return () => {
      for (const entry of buffers.values()) entry.ws?.close();
      buffers.clear();
    };
  }, []);

  // The one thing allowed to call setState -- 4 Hz, decoupled from message rate.
  useEffect(() => {
    const id = setInterval(() => {
      const buffers = buffersRef.current;
      if (buffers.size === 0) return;

      // Decide per-run what this tick needs to do -- and mutate
      // entry.lastProcessedLen -- *before* calling setSnapshots. The updater
      // passed to setSnapshots must be a pure function of `prev`: React 18
      // (under StrictMode, and potentially under concurrent rendering more
      // generally) can invoke a state updater more than once for a single
      // commit to check for exactly this kind of impurity. Mutating
      // entry.lastProcessedLen *inside* the updater made the second
      // invocation see raw.length already equal to lastProcessedLen (set by
      // the first invocation), so it took the "no new data" branch and
      // returned a stale snapshot -- which is the one that got committed.
      const recomputed = new Map<string, RunSnapshot>();
      for (const [runId, entry] of buffers) {
        const raw = entry.raw;
        if (raw.length === 0 || raw.length === entry.lastProcessedLen) continue;
        entry.lastProcessedLen = raw.length;

        const points = {} as Record<MetricKey, Point[]>;
        for (const key of METRIC_KEYS) {
          const full: Point[] = raw.map((m) => ({
            x: m.timesteps,
            y: (m as unknown as Record<string, number | null>)[key] ?? null,
          }));
          points[key] = downsample(full, DISPLAY_POINT_BUDGET);
        }

        recomputed.set(runId, {
          runId,
          connected: entry.ws?.readyState === WebSocket.OPEN,
          points,
          stageBands: computeStageBands(raw),
          latest: raw[raw.length - 1],
          count: raw.length,
        });
      }

      setSnapshots((prev) => {
        const next: Record<string, RunSnapshot> = { ...prev };
        for (const [runId, entry] of buffers) {
          if (recomputed.has(runId)) {
            next[runId] = recomputed.get(runId)!;
            continue;
          }
          // No new data since the last tick -- most ticks, for most runs
          // (message spacing is typically well over 250ms; a finished/
          // paused run never grows at all). Reuse the existing snapshot,
          // only refreshing the cheap `connected` flag.
          next[runId] = { ...(next[runId] ?? emptySnapshot(runId)), connected: entry.ws?.readyState === WebSocket.OPEN };
        }
        return next;
      });
    }, DECIMATE_INTERVAL_MS);
    return () => clearInterval(id);
  }, []);

  return snapshots;
}
