// lttb.ts
// =======
// Largest-Triangle-Three-Buckets downsampling (Sveinn Steinarsson, 2013).
// Reduces an ordered series to `threshold` points while preserving visual
// peaks a naive stride-sample would drop -- important here because a
// transient success-rate collapse (exactly the kind of thing a live
// training monitor exists to catch) is a single-point spike that a
// "keep every Nth point" decimation would very likely erase.
//
// x must be non-decreasing (a time/timestep series, as every chart here
// uses). Points with y === null are passed through as-is at their sampled
// positions (a null keeps the series showing "no data yet" rather than
// being silently interpolated across).

export interface Point {
  x: number;
  y: number | null;
}

export function downsample(data: Point[], threshold: number): Point[] {
  const n = data.length;
  if (threshold >= n || threshold <= 2) return data;

  const sampled: Point[] = [data[0]];
  const bucketSize = (n - 2) / (threshold - 2);

  let a = 0; // index of the previously selected point

  for (let i = 0; i < threshold - 2; i++) {
    // Average point of the NEXT bucket, used as the triangle's third vertex.
    const nextStart = Math.floor((i + 1) * bucketSize) + 1;
    const nextEnd = Math.min(Math.floor((i + 2) * bucketSize) + 1, n);
    let avgX = 0;
    let avgY = 0;
    let avgCount = 0;
    for (let j = nextStart; j < nextEnd; j++) {
      const p = data[j];
      if (p.y === null) continue;
      avgX += p.x;
      avgY += p.y;
      avgCount++;
    }
    if (avgCount > 0) {
      avgX /= avgCount;
      avgY /= avgCount;
    } else {
      // No non-null points in the next bucket -- fall back to its midpoint
      // so the triangle-area comparison below still has a valid vertex.
      avgX = (data[nextStart]?.x ?? data[n - 1].x + data[Math.max(nextEnd - 1, 0)].x) / 2;
      avgY = 0;
    }

    const rangeStart = Math.floor(i * bucketSize) + 1;
    const rangeEnd = Math.floor((i + 1) * bucketSize) + 1;

    let maxArea = -1;
    let maxAreaIdx = rangeStart;
    const pointA = data[a];

    for (let j = rangeStart; j < rangeEnd && j < n; j++) {
      const p = data[j];
      const py = p.y ?? 0;
      const ay = pointA.y ?? 0;
      const area = Math.abs(
        (pointA.x - avgX) * (py - ay) - (pointA.x - p.x) * (avgY - ay),
      );
      if (area > maxArea) {
        maxArea = area;
        maxAreaIdx = j;
      }
    }

    sampled.push(data[maxAreaIdx]);
    a = maxAreaIdx;
  }

  sampled.push(data[n - 1]);
  return sampled;
}
