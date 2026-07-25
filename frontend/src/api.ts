// api.ts
// ======
// Thin fetch wrapper for the FastAPI backend. No caching, no retries --
// this is a showcase dashboard, not a production client.

import type { TestRunResult } from "./types";

const API_BASE = "http://localhost:8000";

async function postJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { method: "POST" });
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export function runTests(): Promise<TestRunResult> {
  return postJSON<TestRunResult>("/api/tests/run");
}
