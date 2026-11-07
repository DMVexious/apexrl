// onnxruntime-web policy session + the 19-dim observation builder.
// Falls back to a hand-written ray-following controller if car_policy.onnx is missing.
import * as ort from 'onnxruntime-web/wasm';
import { RAY_ANGLES, RAY_COUNT, RAY_MAX, G } from './SimConfig.js';

export const OBS_DIM = 19;

export class InferenceEngine {
  constructor(modelPath) {
    this.modelPath = modelPath;
    this.session = null;
    this.isReady = false;
    this.backend = 'heuristic';
    this.lastLatencyMs = 0;
    this.input = new Float32Array(OBS_DIM);
    this.inputTensor = null;
  }

  async init() {
    // A 19x256x256x3 MLP is faster single-threaded; ?threads=N enables WASM multi-threading
    // (needs cross-origin isolation, configured in vite.config.js).
    const requested = Number(new URLSearchParams(location.search).get('threads') || 1);
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, requested) : 1;
    try {
      const res = await fetch(this.modelPath, { method: 'HEAD' });
      if (!res.ok) throw new Error(`${this.modelPath} -> HTTP ${res.status}`);
      this.session = await ort.InferenceSession.create(this.modelPath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      this.inputTensor = new ort.Tensor('float32', this.input, [1, OBS_DIM]);
      await this.session.run({ input_states: this.inputTensor }); // warm-up
      this.isReady = true;
      this.backend = `onnx-wasm x${ort.env.wasm.numThreads}`;
      console.log('[APEX-RL] ONNX policy loaded:', this.modelPath, this.session.inputNames, this.session.outputNames);
    } catch (e) {
      console.warn('[APEX-RL] ONNX policy unavailable, using heuristic controller.', e);
      this.isReady = false;
      this.backend = 'heuristic';
    }
    return this.isReady;
  }

  /** @param {Float32Array|number[]} stateArray19 raw (unnormalized) observation */
  async predict(stateArray19) {
    const t0 = performance.now();
    let action;
    if (this.isReady) {
      this.input.set(stateArray19);
      const results = await this.session.run({ input_states: this.inputTensor });
      const d = results.output_actions.data;
      action = [d[0], d[1], d[2]];
    } else {
      action = heuristicPolicy(stateArray19);
    }
    this.lastLatencyMs = performance.now() - t0;
    return action;
  }
}

/**
 * Builds the observation vector exactly like python_training/envs/custom_car_env.py::_observe.
 * [0..10] ray distances (m) | vx, vy, yaw rate | cos/sin heading err | offset/halfwidth | mu | wind
 */
export function buildObservation(out, rays, st, frame, halfWidth, mu, windLat) {
  for (let i = 0; i < RAY_COUNT; i++) out[i] = rays[i];
  out[11] = st.vx;
  out[12] = st.vy;
  out[13] = st.r;
  out[14] = Math.cos(frame.headingErr);
  out[15] = Math.sin(frame.headingErr);
  out[16] = Math.max(-1, Math.min(1, frame.offset / halfWidth));
  out[17] = mu;
  out[18] = Math.max(-1, Math.min(1, windLat));
  return out;
}

/** Simple reactive driver: steer toward open space + hold the centerline, speed from free distance. */
export function heuristicPolicy(o) {
  let sx = 0;
  let sw = 0;
  for (let i = 0; i < RAY_COUNT; i++) {
    const w = (o[i] / RAY_MAX) ** 4;
    sx += w * RAY_ANGLES[i];
    sw += w;
  }
  const gap = sw > 0 ? sx / sw : 0;
  const steer = Math.max(-1, Math.min(1, 1.6 * gap - 0.6 * o[16] - 1.2 * o[15] - 0.15 * o[13]));
  const front = Math.min(o[4], o[5], o[6]);
  const mu = o[17];
  const vTarget = Math.max(4, Math.min(38, Math.sqrt(2 * mu * G * 0.6 * Math.max(front - 6, 0)) + 3));
  const throttle = Math.max(-1, Math.min(1, 0.35 * (vTarget - o[11])));
  return [steer, throttle, 0];
}
