// nedThree.ts
// ===========
// NED (North-East-Down, right-handed) -> Three.js (right-handed, Y-up)
// coordinate and attitude conversion.
//
// Axis mapping:  X_three = East,  Y_three = -Down (altitude, up),
//                Z_three = -North
// Note the North axis is NEGATED, not just relabeled: East x Up = -North in
// this basis (NED itself is right-handed via North x East = Down), so
// mapping straight to (East, Up, North) would be LEFT-handed and silently
// mirror every rotation. Negating Z keeps X x Y = Z and avoids that mirror.
//
// Sign convention verified empirically against a real recorded baseline
// episode (not just derived from the docstrings): positive roll correlates
// with increasing yaw (a right turn) throughout a sustained turn, confirming
// sim/math_utils.py's "standard NED aerospace ZYX" convention is the
// textbook one (positive roll = right wing down = right turn).
import * as THREE from "three";

export function nedToThreePosition(n: number, e: number, d: number): THREE.Vector3 {
  return new THREE.Vector3(e, -d, -n);
}

// R_ned_from_body : v_ned = R @ v_body -- standard ZYX aerospace DCM
// (Rz(yaw) * Ry(pitch) * Rx(roll)), matching sim/math_utils.py's convention.
function dcmNedFromBody(roll: number, pitch: number, yaw: number): number[][] {
  const cr = Math.cos(roll), sr = Math.sin(roll);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  return [
    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
    [-sp, cp * sr, cp * cr],
  ];
}

// Maps a vector given in NED coordinates to this module's Three.js basis:
// [E, -D, -N] -- same mapping as nedToThreePosition, as a matrix.
const M: number[][] = [
  [0, 1, 0],
  [0, 0, -1],
  [-1, 0, 0],
];

function matMul3(a: number[][], b: number[][]): number[][] {
  const out = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  return out;
}

/**
 * World-space quaternion for a body oriented by (roll, pitch, yaw), expressed
 * in this module's Three.js NED->Y-up basis. Assumes the mesh's own local
 * axes are built as FRD (local +X = forward, +Y = right, +Z = down) -- i.e.
 * body and local-mesh coordinates are identical, so this quaternion can be
 * applied directly to the mesh.
 */
export function bodyQuaternion(roll: number, pitch: number, yaw: number): THREE.Quaternion {
  const rNedFromBody = dcmNedFromBody(roll, pitch, yaw);
  const rThreeFromBody = matMul3(M, rNedFromBody);
  const m4 = new THREE.Matrix4().set(
    rThreeFromBody[0][0], rThreeFromBody[0][1], rThreeFromBody[0][2], 0,
    rThreeFromBody[1][0], rThreeFromBody[1][1], rThreeFromBody[1][2], 0,
    rThreeFromBody[2][0], rThreeFromBody[2][1], rThreeFromBody[2][2], 0,
    0, 0, 0, 1,
  );
  return new THREE.Quaternion().setFromRotationMatrix(m4);
}
