import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { bodyQuaternion, nedToThreePosition } from "../nedThree";
import type { Frame } from "../types";

interface FlightSceneProps {
  frames: Frame[];
  currentIndex: number;
  rHome: number;
}

function buildGliderMesh(): THREE.Group {
  // Built directly in body FRD local axes (+X forward, +Y right, +Z down) --
  // see nedThree.ts's bodyQuaternion, which assumes local == body coords.
  const group = new THREE.Group();

  const fuselageGeom = new THREE.ConeGeometry(0.12, 0.9, 8);
  fuselageGeom.rotateZ(-Math.PI / 2); // default apex is +Y; rotate to point along +X (forward)
  const fuselage = new THREE.Mesh(
    fuselageGeom,
    new THREE.MeshStandardMaterial({ color: 0xf5f5f5 }),
  );
  group.add(fuselage);

  const wingMat = new THREE.MeshStandardMaterial({ color: 0x4caf50 });

  const wing = new THREE.Mesh(new THREE.BoxGeometry(0.18, 1.6, 0.03), wingMat);
  wing.position.set(0.05, 0, 0);
  group.add(wing);

  const tailWing = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.5, 0.02), wingMat);
  tailWing.position.set(-0.45, 0, 0);
  group.add(tailWing);

  const fin = new THREE.Mesh(
    new THREE.BoxGeometry(0.05, 0.05, 0.35),
    new THREE.MeshStandardMaterial({ color: 0xf44336 }),
  );
  fin.position.set(-0.45, 0, -0.17); // -Z_local = up relative to body
  group.add(fin);

  return group;
}

export default function FlightScene({ frames, currentIndex, rHome }: FlightSceneProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const gliderRef = useRef<THREE.Group | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const width = mount.clientWidth || 800;
    const height = mount.clientHeight || 500;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x11151c);

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 5000);
    camera.position.set(60, 45, 60);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 10, 0);
    // Respect prefers-reduced-motion: keep the view fully interactive, just
    // drop the lingering drag inertia for users who've opted out of motion.
    controls.enableDamping = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    controls.dampingFactor = 0.08;
    controls.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const sun = new THREE.DirectionalLight(0xffffff, 0.8);
    sun.position.set(50, 80, 30);
    scene.add(sun);

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(400, 400),
      new THREE.MeshStandardMaterial({ color: 0x1b2430, side: THREE.DoubleSide }),
    );
    ground.rotation.x = -Math.PI / 2;
    scene.add(ground);

    const grid = new THREE.GridHelper(400, 40, 0x334155, 0x223040);
    scene.add(grid);

    const ring = new THREE.Mesh(
      new THREE.RingGeometry(Math.max(rHome - 0.4, 0.1), rHome, 64),
      new THREE.MeshBasicMaterial({
        color: 0x4caf50,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.5,
      }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.05;
    scene.add(ring);

    const trailPoints = frames.map((f) =>
      nedToThreePosition(f.pos_ned[0], f.pos_ned[1], f.pos_ned[2]),
    );
    const trailGeom = new THREE.BufferGeometry().setFromPoints(trailPoints);
    const trail = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ color: 0x88a0c0 }));
    scene.add(trail);

    const glider = buildGliderMesh();
    glider.scale.setScalar(3);
    scene.add(glider);
    gliderRef.current = glider;

    let raf = 0;
    function animate() {
      raf = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    function handleResize() {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (w === 0 || h === 0) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", handleResize);
      controls.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, [frames, rHome]);

  useEffect(() => {
    const glider = gliderRef.current;
    const frame = frames[currentIndex];
    if (!glider || !frame) return;
    glider.position.copy(nedToThreePosition(frame.pos_ned[0], frame.pos_ned[1], frame.pos_ned[2]));
    glider.quaternion.copy(bodyQuaternion(frame.euler[0], frame.euler[1], frame.euler[2]));
  }, [frames, currentIndex]);

  return <div ref={mountRef} className="flight-scene" />;
}
