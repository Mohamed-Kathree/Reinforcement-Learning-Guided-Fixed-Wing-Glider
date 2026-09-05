// InstrumentCluster.tsx
// ======================
// V16 Phase C2 §C2.2 -- extends the previously-unused AttitudeIndicator.tsx
// into a 4-gauge cluster (attitude, airspeed, altimeter, vertical speed) for
// the Flight Replay viewer, replacing OrientationSphere there (confirmed:
// showing attitude twice, once as a 3D ball and once as a flat horizon, was
// redundant). Airspeed reads Frame.airspeed directly (ground-truth, added in
// this same phase); altimeter reads the pre-existing Frame.agl; vertical
// speed reuses the shared kinematicsAt() finite-difference helper already
// established for exactly this in EpisodeViewer.tsx.
import AttitudeIndicator from "./AttitudeIndicator";
import RoundGauge from "./RoundGauge";
import { kinematicsAt } from "../kinematics";
import type { Frame } from "../types";

interface InstrumentClusterProps {
  frame: Frame;
  frames: Frame[];
  frameIndex: number;
}

export default function InstrumentCluster({ frame, frames, frameIndex }: InstrumentClusterProps) {
  const { verticalSpeed } = kinematicsAt(frames, frameIndex);

  return (
    <div className="instrument-cluster">
      <div className="round-gauge">
        <AttitudeIndicator roll={frame.euler[0]} pitch={frame.euler[1]} />
        <div className="round-gauge-label">ATTITUDE</div>
      </div>
      <RoundGauge label="AIRSPEED" value={frame.airspeed} min={0} max={15} unit=" m/s" />
      <RoundGauge label="ALTIMETER" value={frame.agl} min={0} max={30} unit=" m" decimals={0} />
      <RoundGauge label="VERT SPEED" value={verticalSpeed} min={-5} max={5} unit=" m/s" />
    </div>
  );
}
