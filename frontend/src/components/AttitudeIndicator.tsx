interface AttitudeIndicatorProps {
  roll: number;  // radians
  pitch: number; // radians
}

// Standard artificial-horizon convention: the sky/ground background rotates
// opposite to roll and shifts with pitch, while the aircraft reference
// symbol (the amber wings) stays fixed -- matching a real cockpit AI, where
// the world appears to rotate around the aircraft, not the other way round.
export default function AttitudeIndicator({ roll, pitch }: AttitudeIndicatorProps) {
  const rollDeg = -(roll * 180) / Math.PI;
  const pitchOffset = (pitch * 90) / (Math.PI / 4); // ~90px per 45 deg of pitch

  return (
    <svg viewBox="0 0 100 100" width={84} height={84} className="attitude-indicator">
      <defs>
        <clipPath id="ai-clip">
          <circle cx="50" cy="50" r="46" />
        </clipPath>
      </defs>
      <g clipPath="url(#ai-clip)">
        <g transform={`rotate(${rollDeg} 50 50) translate(0 ${pitchOffset})`}>
          <rect x="-60" y="0" width="220" height="220" fill="#3a6ea5" />
          <rect x="-60" y="-220" width="220" height="220" fill="#7a5a3a" />
          <line x1="-60" y1="0" x2="160" y2="0" stroke="#fff" strokeWidth="1.5" />
        </g>
      </g>
      <circle cx="50" cy="50" r="46" fill="none" stroke="#666" strokeWidth="2" />
      <line x1="12" y1="50" x2="38" y2="50" stroke="#ffb300" strokeWidth="2.5" />
      <line x1="62" y1="50" x2="88" y2="50" stroke="#ffb300" strokeWidth="2.5" />
      <circle cx="50" cy="50" r="2" fill="#ffb300" />
    </svg>
  );
}
