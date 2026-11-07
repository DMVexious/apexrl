# APEX-RL — 3D Reinforcement-Learning Car Simulator

An end-to-end autonomous racing stack:

1. **Training** (`python_training/`): a batched PyTorch simulator (thousands of cars in lock-step,
   CUDA or CPU) trains a PPO policy. The policy is exported to `car_policy.onnx`.
2. **Visualizer** (`web_visualizer/`): Three.js + Rapier3D + onnxruntime-web. The browser
   reads the same 19 sensor values as the trainer, runs the network every 60 Hz control tick and
   drives a Rapier rigid body across four stylized maps.

```
shared/sim_config.json  ← single source of truth: car params, terrains, tracks, sensors, obs scaling
        │
        ├── python_training/envs/raycast_car.py   (torch bicycle model + raycasts + tracks)
        │   python_training/envs/custom_car_env.py (VecCarEnv + Gymnasium "ApexRL-Car-v0")
        │   python_training/train_ppo.py → checkpoints/best.pt
        │   python_training/export_onnx.py → web_visualizer/public/weights/car_policy.onnx
        │
        └── web_visualizer/src/core/PhysicsWorld.js (Rapier: same tyre-force equations)
            web_visualizer/src/core/InferenceEngine.js (ORT session + identical observation builder)
```

## Quick start

```bash
# Visualizer (ships with a trained policy)
cd web_visualizer
npm install
npm run dev            # http://localhost:5173   (?map=ice, ?threads=4)

# Training (optional: retrain the policy)
cd python_training
pip install -r requirements.txt     # see the CUDA note inside for RTX 50-series
python train_ppo.py                 # auto-detects CUDA; exports ONNX when done
python export_onnx.py               # re-export / validate best.pt at any time
```

## Controls

| Key | Action |
| --- | --- |
| `M` | Toggle AI pilot / manual driving |
| `W A S D` / arrows, `Space` | Drive / handbrake (manual mode) |
| `C` | Cycle camera: chase → cinematic orbit → hood → top-down |
| `V` | Toggle LIDAR ray visualization |
| `O` | Generate a new random course (obstacles, boost pads, cones) |
| `H` | Hide / show the whole HUD (each panel also has its own minimize button) |
| `R` | Reset race · `1–4` switch maps · `P` pause |

## Observation & action contract

| Index | Meaning | Units |
| --- | --- | --- |
| 0–10 | Distance to walls *or obstacles* on 11 rays at −90, −60, −35, −20, −10, 0, 10, 20, 35, 60, 90° (− = right). Denser ahead so obstacles can't hide between rays | m, max 50 |
| 11, 12, 13 | Forward velocity, lateral velocity (+left), yaw rate (+CCW) | m/s, rad/s |
| 14, 15 | cos / sin of heading error versus the centerline tangent | – |
| 16 | Centerline offset ÷ half track width (+left), clipped to [−1, 1] | – |
| 17 | Surface friction μ (grass 0.85 · desert 0.50 · windy 0.75 · ice 0.12) | – |
| 18 | Lateral wind component (+ pushes left) | [−1, 1] |

Actions: `steer ∈ [−1, 1]` (±35°, +left, slew-limited), `throttle/brake ∈ [−1, 1]`,
`handbrake ∈ [0, 1]` (cuts rear lateral grip by up to 80%).

The ONNX graph takes **raw** physical values (`input_states [batch, 19]`). Normalization and the
handbrake clamp are baked into the graph, so the browser never has to replicate them.

## Physics

Both simulators use a dynamic bicycle model on a 3D road:
- Saturating `tanh` tyre curves and a rear-axle friction circle, so throttle and handbrake break traction.
- Rear-wheel drive, aero drag and rolling resistance.
- Desert "torque sinking": 72% drive force and 3× rolling resistance.
- Ice: 0.8× rear grip.
- A deterministic gusting crosswind, used in the windy map.
- **Hills on every track.** Elevation harmonics in `sim_config.json` give up to 12% grades (3% on ice,
  where μ = 0.12 cannot climb more). In the trainer, gravity acts along and across the slope
  (`slope_terms()`), and a normal-load factor `cos(pitch) + v²·h''/g` makes the car go light over crests
  and dig in through compressions, which scales tyre grip. The browser measures the same load factor from
  its suspension forces.

Training integrates it directly. In the browser, Rapier provides the rigid body, 4-ray spring/damper
suspension, slopes (the canyon track has elevation) and wall collisions. The tyre forces are computed
by `computeTireForces()`, a line-for-line port of `car_forces()`, and applied at the axle positions.
Wall colliders are 16 m tall and invisible, so LIDAR rays still hit them when the road ahead rises or falls.
Control runs at 60 Hz with 2 physics substeps, via a fixed-step accumulator in both simulators.

During training the simulator also randomizes the domain: μ, mass and drive force vary by ±10%,
light random wind is added on every map, and sensor noise is injected. Episodes run on the four map
tracks plus 64 random procedurally generated circuits, from flat to very hilly.

## Courses: obstacles, boost pads, cones

Every map gets a random course, defined in `shared/sim_config.json` under `"course"`. Press `O` for a new one.
- **Obstacles**, themed per map: tyre stacks, boulders, basalt pillars and pink ice crystals. They sit one per evenly
  spaced slot, so there is always room to react. Each one leaves at least 3.2 m of road free on its narrower side and
  stays clear of the spawn point. They are vertical cylinders: the LIDAR sees them, and hitting one ends a training
  episode. The trainer re-randomizes their layout, density and size every episode, and the web course generator
  (`src/core/Course.js`) follows the same rules.
- **Boost pads**: animated chevron strips that push the car forward like a jet (5 kN, or 1.2 kN on ice). The push
  doesn't depend on tyre grip, so the car needs to brake afterwards. They are never placed just before an obstacle.
  The policy can't see them, but it trains with them, so it handles the sudden extra speed.
- **Traffic cones** line the shoulders as light Rapier rigid bodies that go flying when hit. They exist in the
  browser only, are invisible to the sensors, and are just for fun.

## Maps & effects

| Map | Physics | Rendering |
| --- | --- | --- |
| Grassy Circuit | μ 0.85 | 70k instanced grass blades sampled with `MeshSurfaceSampler`; they sway and bend away from the car. Also: DoF (`BokehPass`) focused on the car, speed-driven radial motion blur, low-poly pines |
| Desert Dunes | μ 0.50, torque sinking | Heat-shimmer full-screen pass (`HeatDistortionShader`), velocity/slip-driven sand emitters at the rear tyres, procedural warm 3D LUT (`LUTPass`) |
| Windy Canyons | μ 0.75, gusting crosswind | Wind-streak particles blowing along the live wind vector, `FogExp2` whose density follows camera elevation, 26–56 m canyon walls, elevated track |
| Synthwave Glacier | μ 0.12, 0.8× rear grip | `RFL_Glacier` Fresnel ice: parallax sub-surface cracks, sky reflection, neon lane edges. Bloom at threshold 0.2 / strength 1.5 / radius 0.8. Neon light trails behind the rear wheels when rear slip > 0.4 |

## Shipped policy & results

The shipped `car_policy.onnx` (= `checkpoints_course_scratch/best.pt`, 123M steps) was trained **from scratch on the
obstacle / boost-pad courses** in one GPU run (RTX 5060, 4096 envs, about 1.5 h). It uses a separate, larger penalty for hitting
an obstacle and picks the checkpoint by a crash-weighted score:

```bash
python train_ppo.py --num-envs 4096 --minibatch 16384 --total-steps 150e6 --eval-every 10     --terrain-weights 0.2 0.2 0.35 0.25 --obstacle-crash-penalty 15 --eval-crash-weight 1.5
```

- `--obstacle-crash-penalty`: terminal penalty for an obstacle hit. Wall crashes keep `--crash-penalty` (5).
- `--eval-crash-weight`: `score = mean(speed/ref) − w · crashes/min`. The old default of 0.5 let faster but crashier checkpoints win.
- The terrain weights oversample windy and ice.

The previous policy (`course_v2_best.pt`: staged CPU→GPU→hills training, then a course fine-tune) is kept as
`python_training/checkpoints/car_policy_course_v2.onnx`.

### Obstacle courses

Python eval on the four map tracks with obstacles (deterministic policy, 256 cars × 120 s per map; speed · crashes per minute):

| Checkpoint | Grass | Desert | Windy | Ice | Total crashes/min |
| --- | --- | --- | --- | --- | --- |
| **`course_scratch` (shipped)** | **22.4 m/s · 1.2** | **15.6 m/s · 0.4** | **21.2 m/s · 1.7** | **5.9 m/s · 1.2** | **4.4** |
| `course_v2_best.pt` (previous) | 19.4 m/s · 4.0 | 13.0 m/s · 2.3 | 17.0 m/s · 3.8 | 4.6 m/s · 0.6 | 10.8 |
| `hills_best.pt` (never saw obstacles) | 19.0 m/s · 3.7 | 12.1 m/s · 1.8 | 16.3 m/s · 3.5 | 3.9 m/s · 0.5 | 9.5 |
| `obstacle_inflate` 1.0 fine-tune | 20.4 m/s · 4.1 | 14.6 m/s · 2.3 | 18.6 m/s · 4.2 | 5.0 m/s · 2.6 | 13.1 |

Without obstacles the new policy is also faster: grass 25.6, desert 18.5, windy 23.6 and ice 5.8 m/s, against 21.7 / 13.4 / 18.4 / 4.7.
Windy wall contacts are down from 1.0 to 0.4/min. **Ice is the one regression:** about 1 wall contact/min on the course
(0.5/min without obstacles), against ≤0.5 before, because it now drives ice faster.

Why the earlier course fine-tunes failed (diagnostics on `course_v2`/`best.pt`):
- **Perception was not the bottleneck.** The obstacle that ends up being hit was first seen at a median of 50 m and was
  on a ray 84% of the time.
- **Hits were rare late dodges in sharp turns.** Most obstacles were passed with a wide margin (median clearance 3.9 m).
  The ~6% that were hit were front-corner clips, concentrated in tight corners (77% of hits at radius < 150 m vs 50% of
  obstacles), where cornering already uses the tyre grip. The fix is to brake earlier.
- **The −5 crash penalty and the speed-heavy `eval_score` rewarded speed over safety.** Every fine-tune got faster and
  crashier. A stiffer obstacle penalty plus a crash-weighted checkpoint score fixed this.
- `sensors.obstacle_inflate` (grow obstacles by ~1 m in the ray test only, so a ray reads free distance for the whole
  car body) is implemented in both simulators. At 1.0 it gave no net gain, so it is off (0), which the shipped policy expects.

Closed-loop results **in the browser** on the hilly tracks, from the older stage-3 policy *before* obstacles were added
(Rapier physics, ONNX WASM, 120 s per map):

| Map | Avg speed | Best lap | Wall contacts | Load range (crest…dip) | Inference |
| --- | --- | --- | --- | --- | --- |
| Grass | 23.4 m/s (84 km/h) | 36.2 s | 0 | 0.70…1.25 g | 0.06 ms |
| Desert | 16.9 m/s | 48.4 s | 0 | 0.70…1.14 g | 0.05 ms |
| Windy | 21.2 m/s | 44.2 s | 3 | 0.69…1.21 g | 0.05 ms |
| Ice | 7.3 m/s | 111.1 s | 0 | 0.76…1.12 g | 0.05 ms |

That older policy was weakest on Windy Canyons, where gusts occasionally brushed it into the inside wall. A run with a
−15 penalty on *all* crashes plus a wall-proximity penalty made every terrain worse. The shipped policy instead uses the
stiffer penalty only for obstacle hits (see above). These browser numbers have not been re-measured for the new policy yet.

Training on the GPU runs about 60k env-steps/s end-to-end (versus about 10k on CPU). For RTX 50-series
cards, install the CUDA 12.8 PyTorch wheel (see `requirements.txt`).

## Notes and deviations from the original PRD

- **Versions**: three r187, Rapier 0.21, onnxruntime-web 1.31 and Vite 8 (the newest releases at build time).
  - three removed `PCFSoftShadowMap`, so the visualizer uses `PCFShadowMap` with a blur radius.
  - The shadow frustum follows the car, which replaces cascaded shadow maps.
- **No PyBullet**: training uses a custom vectorized torch simulator.
  - It runs thousands of envs per step on GPU/CPU.
  - Its equations are mirrored exactly in the browser, which keeps the policy transferable.
- **ORT execution provider**: WASM only (the WebGL EP is deprecated upstream). Inference defaults to 1 thread,
  because a 19→256→256→3 MLP is faster without thread sync. Use `?threads=4` for multi-threaded WASM;
  the dev server sends COOP/COEP headers for it.
- **Car and track models are procedural** instead of GLTF (see `public/models/README.md`).
- **A heuristic fallback controller** drives the car if `car_policy.onnx` is missing.
