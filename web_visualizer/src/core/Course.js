// Seeded course layout: obstacles, boost pads and knock-over cones.
// Obstacle/pad placement mirrors python_training/envs/custom_car_env.py::_reset_course (evaluation mode):
// one candidate per evenly spaced slot, map-specific density, >= lateral_gap of free road, clear of spawn,
// and no boost pad within 30 points before an obstacle.
import { SIM } from './SimConfig.js';

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function generateCourse(geo, mapName, seed, spawnIdx = 0) {
  const cc = SIM.course;
  const m = cc.maps[mapName];
  const rnd = mulberry32(seed);
  const N = geo.N;
  const hw = geo.halfWidth;
  const M = cc.obstacle_slots;
  const P = cc.boost_slots;

  // --- obstacles
  const obstacles = [];
  const slotIdx = [];
  for (let k = 0; k < M; k++) slotIdx.push(Math.floor(((k + rnd() * 0.6 + 0.2) * N) / M) % N);
  // choose exactly m.obstacles slots (shuffled), skipping ones too close to the spawn point
  const order = [...Array(M).keys()].sort(() => rnd() - 0.5);
  for (const k of order) {
    if (obstacles.length >= m.obstacles) break;
    const idx = slotIdx[k];
    const rel = (((idx - spawnIdx) % N) + N) % N;
    if (rel < cc.spawn_clear_ahead_pts || rel > N - cc.spawn_clear_behind_pts) continue;
    const r = m.radius[0] + (m.radius[1] - m.radius[0]) * rnd();
    const latMax = Math.max(0, hw - r - cc.lateral_gap);
    const lat = (rnd() * 2 - 1) * latMax;
    obstacles.push({ idx, lat, r, x: geo.cx[idx] + geo.nx[idx] * lat, y: geo.cy[idx] + geo.ny[idx] * lat, h: geo.elev[idx] });
  }

  // --- boost pads
  const pads = [];
  const seg = geo.length / N;
  const padPts = Math.ceil(cc.boost_length / seg);
  const padSlots = [...Array(P).keys()].sort(() => rnd() - 0.5);
  for (const k of padSlots) {
    if (pads.length >= m.boost_pads) break;
    // a few tries per slot to find a spot that isn't just before an obstacle
    for (let tries = 0; tries < 8; tries++) {
      const idx = Math.floor(((k + rnd() * 0.8 + 0.1) * N) / P) % N;
      const rel = (((idx - spawnIdx) % N) + N) % N;
      const conflict = rel < 6 || obstacles.some((o) => {
        const gap = (((o.idx - idx) % N) + N) % N;
        return gap < 30 || gap > N - 4;
      });
      if (conflict) continue;
      const lat = (rnd() * 2 - 1) * Math.max(0, hw - cc.boost_half_width - 0.5);
      pads.push({ idx, lat, pts: padPts });
      break;
    }
  }

  // --- cones: little rows on the shoulders, away from obstacles/pads (browser-only physics props)
  const cones = [];
  let guard = 0;
  while (cones.length < m.cones && guard++ < 400) {
    const idx = Math.floor(rnd() * N);
    const near = obstacles.some((o) => Math.abs(((o.idx - idx + N / 2) % N) - N / 2) < 8) ||
      pads.some((p) => Math.abs(((p.idx - idx + N / 2) % N) - N / 2) < 8);
    const rel = (((idx - spawnIdx) % N) + N) % N;
    if (near || rel < 10 || rel > N - 4) continue;
    const side = rnd() < 0.5 ? 1 : -1;
    const run = 3 + Math.floor(rnd() * 3);
    for (let j = 0; j < run && cones.length < m.cones; j++) {
      const i = geo.wrap(idx + j * 2);
      const lat = side * (hw - 0.55);
      cones.push({ x: geo.cx[i] + geo.nx[i] * lat, y: geo.cy[i] + geo.ny[i] * lat, h: geo.elev[i] });
    }
  }
  return { seed, obstacles, pads, cones };
}

/** Boost force (N) if the car centre (track index + lateral offset) is over a pad. Mirrors VecCarEnv._boost. */
export function boostForceAt(course, geo, mapName, idx, offset) {
  const cc = SIM.course;
  const seg = geo.length / geo.N;
  for (const p of course.pads) {
    const rel = (((idx - p.idx) % geo.N) + geo.N) % geo.N;
    if (rel * seg < cc.boost_length && Math.abs(offset - p.lat) < cc.boost_half_width) return cc.maps[mapName].boost_force;
  }
  return 0;
}

/** Same two-circle car footprint test as VecCarEnv._obstacle_hit. */
export function obstacleHit(course, x, y, yaw) {
  const cc = SIM.course;
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  for (const k of [cc.car_collision_offset, -cc.car_collision_offset]) {
    const px = x + c * k;
    const py = y + s * k;
    for (const o of course.obstacles) {
      const rr = o.r + cc.car_collision_radius;
      if ((px - o.x) ** 2 + (py - o.y) ** 2 < rr * rr) return true;
    }
  }
  return false;
}
