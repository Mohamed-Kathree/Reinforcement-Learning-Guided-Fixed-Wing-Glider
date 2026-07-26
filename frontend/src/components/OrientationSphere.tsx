import { useEffect, useRef } from "react";
import * as THREE from "three";
import { bodyQuaternion } from "../nedThree";

interface OrientationSphereProps {
  roll: number;  // radians
  pitch: number; // radians
  yaw: number;   // radians
}

const SIZE = 176;

// Builds a "attitude ball" texture: a sky/ground sphere painted once, in the
// ball's OWN body-frame axes (same "local axes == FRD body axes" convention
// FlightScene's glider mesh uses), so rotating the ball via bodyQuaternion()
// each frame is all that's needed to show the current attitude -- same idea
// as the flat AttitudeIndicator.tsx's rotated sky/ground rectangle, just as
// an actual 3D sphere instead of a 2D SVG.
function buildBallTexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 1024;
  canvas.height = 512;
  const ctx = canvas.getContext("2d")!;

  const sky = ctx.createLinearGradient(0, 0, 0, canvas.height / 2);
  sky.addColorStop(0, "#2b4a72");
  sky.addColorStop(1, "#3a6ea5");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, canvas.width, canvas.height / 2);

  const ground = ctx.createLinearGradient(0, canvas.height / 2, 0, canvas.height);
  ground.addColorStop(0, "#7a5a3a");
  ground.addColorStop(1, "#4a3520");
  ctx.fillStyle = ground;
  ctx.fillRect(0, canvas.height / 2, canvas.width, canvas.height / 2);

  // Horizon
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(0, canvas.height / 2);
  ctx.lineTo(canvas.width, canvas.height / 2);
  ctx.stroke();

  // Pitch graduation bands every 15 deg (+/- 60), a fixed line of latitude
  // on the ball -- as the ball pitches, the band under the fixed reference
  // marker changes, exactly like a real attitude-ball instrument.
  ctx.strokeStyle = "rgba(255,255,255,0.32)";
  ctx.lineWidth = 1.5;
  ctx.font = "18px sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.55)";
  ctx.textAlign = "left";
  for (let deg = -60; deg <= 60; deg += 15) {
    if (deg === 0) continue;
    const y = canvas.height / 2 - (deg / 90) * canvas.height;
    ctx.beginPath();
    ctx.moveTo(canvas.width * 0.32, y);
    ctx.lineTo(canvas.width * 0.68, y);
    ctx.stroke();
    ctx.fillText(`${deg > 0 ? "+" : ""}${deg}`, canvas.width * 0.7, y + 6);
  }

  // Longitude stripes every 30 deg so yaw rotation reads visually, same as
  // heading lines on a real navigation ball.
  ctx.strokeStyle = "rgba(255,255,255,0.22)";
  ctx.lineWidth = 1;
  for (let lon = 0; lon < 360; lon += 30) {
    const x = (lon / 360) * canvas.width;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

export default function OrientationSphere({ roll, pitch, yaw }: OrientationSphereProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const ballRef = useRef<THREE.Mesh | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0d10);

    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 50);
    camera.position.set(0, 0.4, 3.4);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(SIZE, SIZE);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const sun = new THREE.DirectionalLight(0xffffff, 0.6);
    sun.position.set(2, 3, 2);
    scene.add(sun);

    // Bake a one-time local rotation so the texture's pole axis (default:
    // Three.js sphere UV poles run along local Y) lines up with local -Z
    // (body "up", FRD convention) instead -- see OrientationSphere's design
    // notes for the derivation. This is baked into the geometry once, not
    // per frame; bodyQuaternion() below then rotates the whole already-
    // textured rigid body into view, exactly like FlightScene's glider mesh.
    const geometry = new THREE.SphereGeometry(1, 48, 32);
    geometry.rotateX(-Math.PI / 2);
    const material = new THREE.MeshStandardMaterial({ map: buildBallTexture(), roughness: 0.9 });
    const ball = new THREE.Mesh(geometry, material);
    scene.add(ball);
    ballRef.current = ball;

    const ring = new THREE.Mesh(
      new THREE.RingGeometry(1.02, 1.06, 48),
      new THREE.MeshBasicMaterial({ color: 0x333844, side: THREE.DoubleSide }),
    );
    scene.add(ring);

    let raf = 0;
    function animate() {
      raf = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    }
    animate();

    return () => {
      cancelAnimationFrame(raf);
      renderer.dispose();
      geometry.dispose();
      material.map?.dispose();
      material.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  useEffect(() => {
    const ball = ballRef.current;
    if (!ball) return;
    ball.quaternion.copy(bodyQuaternion(roll, pitch, yaw));
  }, [roll, pitch, yaw]);

  return (
    <div className="orientation-sphere-wrap" style={{ width: SIZE, height: SIZE }}>
      <div ref={mountRef} className="orientation-sphere" />
      <div className="orientation-sphere-marker" aria-hidden="true">
        <span className="marker-wing marker-wing-left" />
        <span className="marker-dot" />
        <span className="marker-wing marker-wing-right" />
      </div>
    </div>
  );
}
