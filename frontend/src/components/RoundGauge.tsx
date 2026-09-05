// RoundGauge.tsx
// ===============
// V16 Phase C2 §C2.2 -- one shared circular-dial gauge, parametrised for
// airspeed/altimeter/vertical-speed (three near-identical instruments, so
// one component, not three copies of the same tick/needle math). Matches
// AttitudeIndicator.tsx's existing viewBox/bezel/amber-needle visual
// language rather than inventing a second style.
//
// Needle rotation is a plain linear function of value -> angle with no CSS
// transition on the rotating element: "needles interpolate between frames
// with no easing beyond linear" (spec) means don't ease the rotation with a
// timing curve -- let it snap directly to each frame's true angle, which
// reads as linear motion during playback and an instant snap when
// scrubbing, both correct.
const START_ANGLE = -120; // degrees; dial sweep start
const END_ANGLE = 120; // degrees; dial sweep end
const SWEEP = END_ANGLE - START_ANGLE;
const TICK_COUNT = 6;

interface RoundGaugeProps {
  label: string;
  value: number;
  min: number;
  max: number;
  unit: string;
  decimals?: number;
}

function angleFor(value: number, min: number, max: number): number {
  const clamped = Math.min(Math.max(value, min), max);
  const frac = (clamped - min) / (max - min);
  return START_ANGLE + frac * SWEEP;
}

function pointAt(angleDeg: number, radius: number): [number, number] {
  const rad = (angleDeg * Math.PI) / 180;
  return [50 + radius * Math.sin(rad), 50 - radius * Math.cos(rad)];
}

export default function RoundGauge({ label, value, min, max, unit, decimals = 1 }: RoundGaugeProps) {
  const needleAngle = angleFor(value, min, max);
  const [needleX, needleY] = pointAt(needleAngle, 34);

  const ticks = Array.from({ length: TICK_COUNT + 1 }, (_, i) => {
    const angle = START_ANGLE + (i / TICK_COUNT) * SWEEP;
    const [x1, y1] = pointAt(angle, 38);
    const [x2, y2] = pointAt(angle, 46);
    return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#888" strokeWidth={1.5} />;
  });

  return (
    <div className="round-gauge">
      <svg viewBox="0 0 100 100" width={84} height={84} className="round-gauge-dial">
        <circle cx="50" cy="50" r="46" fill="none" stroke="#666" strokeWidth="2" />
        {ticks}
        <line x1="50" y1="50" x2={needleX} y2={needleY} stroke="#ffb300" strokeWidth="2.5" />
        <circle cx="50" cy="50" r="3" fill="#ffb300" />
      </svg>
      <div className="round-gauge-label">{label}</div>
      <div className="round-gauge-value">
        {value.toFixed(decimals)}
        <span>{unit}</span>
      </div>
    </div>
  );
}
