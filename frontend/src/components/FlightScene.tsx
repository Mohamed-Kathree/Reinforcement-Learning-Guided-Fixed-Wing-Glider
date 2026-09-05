import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { bodyQuaternion, nedToThreePosition } from "../nedThree";
import type { Frame } from "../types";

interface FlightSceneProps {
  frames: Frame[];
  currentIndex: number;
  rHome: number;
  // When true, the scene drives its own looping playback on the
  // trajectory's own clock instead of being controlled by `currentIndex`
  // (used by the Overview hero, which has no scrubber). Default false
  // leaves every existing controlled usage (EpisodeViewer) unaffected.
  autoplay?: boolean;
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

export default function FlightScene({ frames, currentIndex, rHome, autoplay = false }: FlightSceneProps) {
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

    // ResizeObserver, not just a window "resize" listener: this mount div
    // is 0x0 while its tab-pane is hidden (display: none -- see App.tsx's
    // always-mounted-tabs pattern), and the hero on the Overview tab in
    // particular can finish its own data fetch and mount this scene for
    // the first time *after* the user has already navigated to a different
    // tab. A plain resize listener would never fire in that case, silently
    // freezing the renderer at its fallback 800x500 size forever. Same
    // pattern already used by training/UplotChart.tsx for the same reason.
    const resizeObserver = new ResizeObserver(() => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (w === 0 || h === 0) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    resizeObserver.observe(mount);

    return () => {
      cancelAnimationFrame(raf);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, [frames, rHome]);

  useEffect(() => {
    if (autoplay) return; // owned by the autoplay effect below instead
    const glider = gliderRef.current;
    const frame = frames[currentIndex];
    if (!glider || !frame) return;
    glider.position.copy(nedToThreePosition(frame.pos_ned[0], frame.pos_ned[1], frame.pos_ned[2]));
    glider.quaternion.copy(bodyQuaternion(frame.euler[0], frame.euler[1], frame.euler[2]));
  }, [frames, currentIndex, autoplay]);

  // Autoplay: loop through the trajectory on its own clock instead of being
  // driven by an external scrubber. Mutates the glider mesh directly inside
  // a requestAnimationFrame loop -- no React state, no re-renders, matching
  // the "never animate through React state" discipline the Phase B chart
  // rewrite established (see useMetricsStream.ts's module docstring).
  useEffect(() => {
    if (!autoplay || frames.length === 0) return;
    const glider = gliderRef.current;
    if (!glider) return;

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      // A static representative frame instead of continuous looping.
      const frame = frames[Math.floor(frames.length / 2)];
      glider.position.copy(nedToThreePosition(frame.pos_ned[0], frame.pos_ned[1], frame.pos_ned[2]));
      glider.quaternion.copy(bodyQuaternion(frame.euler[0], frame.euler[1], frame.euler[2]));
      return;
    }

    const t0Frame = frames[0].t;
    const duration = frames[frames.length - 1].t - t0Frame || 1;
    const startTime = performance.now();
    let raf = 0;
    let idx = 0;
    function tick() {
      raf = requestAnimationFrame(tick);
      const elapsed = ((performance.now() - startTime) / 1000) % duration;
      // Playback is monotonic except at the loop wrap, which is the one
      // place idx needs to reset rather than just advance.
      if (elapsed < frames[idx].t - t0Frame) idx = 0;
      while (idx < frames.length - 1 && frames[idx + 1].t - t0Frame <= elapsed) idx++;
      const frame = frames[idx];
      glider!.position.copy(nedToThreePosition(frame.pos_ned[0], frame.pos_ned[1], frame.pos_ned[2]));
      glider!.quaternion.copy(bodyQuaternion(frame.euler[0], frame.euler[1], frame.euler[2]));
    }
    tick();
    return () => cancelAnimationFrame(raf);
  }, [frames, autoplay]);

  return <div ref={mountRef} className="flight-scene" />;
}
