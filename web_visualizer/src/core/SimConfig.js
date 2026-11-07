// Shared physics / sensor contract with python_training (see /shared/sim_config.json).
import cfg from '../../../shared/sim_config.json';

export const SIM = cfg;
export const MAPS = ['grass', 'desert', 'windy', 'ice'];
export const CONTROL_DT = 1 / cfg.control_hz;
export const SUBSTEPS = cfg.physics_substeps;
export const G = 9.81;

const s = cfg.sensors;
export const RAY_COUNT = s.ray_count;
export const RAY_MAX = s.ray_max_dist;
// Obstacles look this much bigger to the sensor rays (body-width clearance), matching the trainer.
export const OBSTACLE_INFLATE = s.obstacle_inflate ?? 0;
// Denser straight ahead (±10°, ±20°) so obstacles can't hide between rays; falls back to uniform spacing.
export const RAY_ANGLES = (s.ray_angles_deg ?? Array.from({ length: s.ray_count }, (_, i) =>
  s.ray_min_deg + ((s.ray_max_deg - s.ray_min_deg) * i) / (s.ray_count - 1),
)).map((d) => (d * Math.PI) / 180);

// 2D sim plane (x, y) <-> three.js world (x, height, -y)
export const to3 = (x, y, h = 0) => ({ x, y: h, z: -y });
export const to2 = (v) => ({ x: v.x, y: -v.z });

/** Deterministic gusting wind, identical to python_training/envs/custom_car_env.py::wind_lateral. */
export function windAt(t, wmax, phi0, phase) {
  const phi = phi0 + 0.15 * Math.sin(0.05 * t);
  const raw = 0.6 + 0.3 * Math.sin(0.4 * t + phase) + 0.1 * Math.sin(1.7 * t + 2 * phase);
  return { phi, strength: wmax * Math.min(Math.max(raw, 0), 1) };
}
