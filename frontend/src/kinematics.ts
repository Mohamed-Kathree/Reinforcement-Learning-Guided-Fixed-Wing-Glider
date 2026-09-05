// kinematics.ts
// =============
// Shared by EpisodeViewer.tsx and InstrumentCluster.tsx. Ground/vertical
// speed aren't recorded directly (only body-frame velocity is) -- derive
// them from consecutive recorded positions instead of doing a body->NED
// rotation client-side (that would edge toward reimplementing physics,
// which ground rule 2 forbids; a finite difference of already-recorded
// true positions is just kinematics, not a reimplementation of anything
// compute_reward() does).
import type { Frame } from "./types";

const DT_RL = 0.05; // matches env/glider_env.py's policy step (20 Hz)

export function kinematicsAt(frames: Frame[], index: number): { groundSpeed: number; verticalSpeed: number } {
  const a = frames[index > 0 ? index - 1 : index];
  const b = frames[index > 0 ? index : Math.min(index + 1, frames.length - 1)];
  if (!a || !b || a === b) return { groundSpeed: 0, verticalSpeed: 0 };
  const dn = b.pos_ned[0] - a.pos_ned[0];
  const de = b.pos_ned[1] - a.pos_ned[1];
  const dd = b.pos_ned[2] - a.pos_ned[2];
  return {
    groundSpeed: Math.hypot(dn, de) / DT_RL,
    verticalSpeed: -dd / DT_RL, // positive = climbing
  };
}
