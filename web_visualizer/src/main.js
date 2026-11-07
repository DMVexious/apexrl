// APEX-RL bootstrap: fixed 60 Hz control loop (sense -> ONNX policy -> Rapier physics) decoupled
// from rendering through a step accumulator, plus camera rig, map switching, input and HUD.
import * as THREE from 'three';
import { createIcons, Trees, Sun, Wind, Snowflake, Brain, Camera, Radar, RotateCcw, Shuffle, Minus, Plus } from 'lucide';
import { Engine } from './core/Engine.js';
import { PhysicsWorld, initRapier } from './core/PhysicsWorld.js';
import { InferenceEngine, buildObservation, OBS_DIM } from './core/InferenceEngine.js';
import { CONTROL_DT, MAPS, RAY_MAX, RAY_COUNT } from './core/SimConfig.js';
import { generateCourse, boostForceAt, obstacleHit } from './core/Course.js';
import { CourseVisuals } from './entities/CourseVisuals.js';
import { Track, MAP_PROFILES } from './entities/Tracks.js';
import { Car } from './entities/Car.js';
import { VisualRaycaster } from './entities/Raycaster.js';
import { DustSystem, WindStreaks } from './entities/Effects.js';
import { TelemetryHUD } from './ui/TelemetryHUD.js';
import { SensorOverlay } from './ui/SensorOverlay.js';

const CAMERA_MODES = ['chase', 'cinematic', 'hood', 'top'];
const MAX_TICKS_PER_FRAME = 4;

class App {
  async init() {
    createIcons({ icons: { Trees, Sun, Wind, Snowflake, Brain, Camera, Radar, RotateCcw, Shuffle, Minus, Plus } });
    const status = (t) => (document.getElementById('loading-text').textContent = t);

    await initRapier();
    status('Building render pipeline…');
    this.engine = new Engine(document.getElementById('app-container'));
    this.scene = this.engine.scene;
    this.camera = this.engine.camera;

    this.physics = new PhysicsWorld();
    this.car = new Car(this.scene);
    this.rays = new VisualRaycaster(this.scene, RAY_COUNT);
    this.dust = new DustSystem(this.scene);
    this.windFx = new WindStreaks(this.scene);
    this.courseFx = new CourseVisuals(this.scene);
    this.hud = new TelemetryHUD({ onPanelToggle: (name, open) => name === 'radar' && open && this.radar.resize() });
    this.radar = new SensorOverlay(document.getElementById('radar-canvas'));

    status('Loading car_policy.onnx…');
    this.inference = new InferenceEngine(`${import.meta.env.BASE_URL}weights/car_policy.onnx`);
    await this.inference.init();

    this.obs = new Float32Array(OBS_DIM);
    this.rayBuf = { dist: new Float32Array(RAY_COUNT), points: Array.from({ length: RAY_COUNT }, () => new THREE.Vector3()) };
    this.action = [0, 0, 0];
    this.mode = 'ai';
    this.cameraMode = 0;
    this.paused = false;
    this.keys = new Set();
    this.manual = { steer: 0 };
    this.fps = 60;
    this.camPos = new THREE.Vector3();
    this.camLook = new THREE.Vector3();

    this.bindUI();
    status('Generating terrain…');
    await this.loadMap(new URLSearchParams(location.search).get('map') || 'grass');

    document.getElementById('loading').classList.add('hidden');
    this.last = performance.now();
    this.acc = 0;
    requestAnimationFrame(() => this.frame());
  }

  // ------------------------------------------------------------------ map lifecycle
  async loadMap(name) {
    if (!MAPS.includes(name)) name = 'grass';
    this.loadingMap = true;
    document.querySelectorAll('.map-btn').forEach((b) => {
      b.classList.toggle('active', b.dataset.map === name);
      b.disabled = true;
    });
    await new Promise((r) => setTimeout(r, 0)); // let the UI paint before heavy generation
    this.track?.dispose();
    this.map = name;
    this.track = new Track(this.scene, name);
    this.engine.applyProfile(this.track.profile);
    this.physics.build(this.track.geo, name);
    this.newCourse(false);
    this.car.setMap(name);
    this.dust.setMap(name);
    this.windFx.setMap(name);
    this.resetRace();
    this.snapCamera();
    document.querySelectorAll('.map-btn').forEach((b) => (b.disabled = false));
    this.loadingMap = false;
  }

  /** New random obstacle / boost-pad / cone layout for the current map. */
  newCourse(restart = true) {
    const seed = (Math.random() * 2 ** 32) >>> 0;
    this.course = generateCourse(this.track.geo, this.map, seed, 0);
    this.physics.setCourse(this.course);
    this.courseFx.build(this.course, this.track.geo, this.map);
    if (restart) this.resetRace();
  }

  resetRace() {
    this.race = {
      idx: 0, sPrev: 0, progress: 0, lap: 0, lapStart: 0, lastLap: NaN, bestLap: NaN,
      crashes: 0, obstacleHits: 0, inContact: false, inObstacle: false, boosting: false, stuck: 0, wrongWay: 0,
    };
    this.respawn(0, 0);
    this.race.lapStart = this.physics.time;
  }

  respawn(idx, speed = 0) {
    const geo = this.track.geo;
    // never respawn on top of / right in front of an obstacle
    const blocked = (i) => this.course?.obstacles.some((o) => ((o.idx - i) % geo.N + geo.N) % geo.N < 14 || ((i - o.idx) % geo.N + geo.N) % geo.N < 3);
    for (let tries = 0; tries < 40 && blocked(geo.wrap(idx)); tries++) idx += 3;
    this.physics.vehicle.reset(geo, idx, 0, speed);
    const st = this.physics.vehicle.state();
    this.race.idx = geo.wrap(idx);
    this.race.sPrev = geo.frame(this.race.idx, st.x2, st.y2, st.yaw).s;
    this.race.stuck = 0;
    this.race.wrongWay = 0;
    this.car.trails.forEach((t) => t.clear());
  }

  // ------------------------------------------------------------------ control tick (60 Hz)
  async tick() {
    const phys = this.physics;
    const geo = this.track.geo;
    const veh = phys.vehicle;
    const race = this.race;

    // 1. sense
    let st = veh.state();
    race.idx = geo.nearestIndex(st.x2, st.y2, race.idx);
    const fr = geo.frame(race.idx, st.x2, st.y2, st.yaw);
    phys.castSensorRays(this.rayBuf);
    const wind = phys.windLateral(st.yaw);
    buildObservation(this.obs, this.rayBuf.dist, st, fr, geo.halfWidth, phys.terrain.mu, wind.lateral);

    // 2. act
    if (this.mode === 'ai') {
      this.action = await this.inference.predict(this.obs);
    } else {
      this.action = this.manualAction();
    }

    // 3. physics (2 Rapier substeps with re-evaluated tyre forces); boost decided from the pre-step pose
    const boost = boostForceAt(this.course, geo, this.map, race.idx, fr.offset);
    race.boosting = boost > 0;
    phys.step(this.action, boost);

    // 4. race bookkeeping
    st = veh.state();
    race.idx = geo.nearestIndex(st.x2, st.y2, race.idx);
    const f2 = geo.frame(race.idx, st.x2, st.y2, st.yaw);
    const L = geo.length;
    const ds = ((((f2.s - race.sPrev + L / 2) % L) + L) % L) - L / 2;
    race.sPrev = f2.s;
    race.progress += ds;
    if (race.progress >= (race.lap + 1) * L) {
      const t = phys.time - race.lapStart;
      race.lastLap = t;
      if (!(t >= race.bestLap)) race.bestLap = t; // also true when bestLap is NaN
      race.lap++;
      race.lapStart = phys.time;
    }
    const touching = Math.abs(f2.offset) > geo.halfWidth - 0.95 - 0.05;
    if (touching && !race.inContact) race.crashes++;
    race.inContact = touching;
    const hitObs = obstacleHit(this.course, st.x2, st.y2, st.yaw);
    if (hitObs && !race.inObstacle) race.obstacleHits++;
    race.inObstacle = hitObs;

    const speed = Math.hypot(st.vx, st.vy);
    race.stuck = speed < 0.4 ? race.stuck + CONTROL_DT : 0; // low threshold: ice (mu 0.12) accelerates at ~0.4 m/s²
    race.wrongWay = Math.cos(f2.headingErr) < -0.3 && speed > 2 ? race.wrongWay + CONTROL_DT : 0;
    const flipped = st.up.y < 0.3;
    const lost = Math.abs(f2.offset) > geo.halfWidth + 3 || st.position.y < geo.elev[race.idx] - 2;
    if ((this.mode === 'ai' && (race.stuck > 4 || race.wrongWay > 2)) || flipped || lost) {
      this.respawn(race.idx + 4, 5);
    }
    this.lastState = st;
    this.lastFrame = f2;
    this.lastWind = wind;
  }

  manualAction() {
    const k = this.keys;
    const left = k.has('a') || k.has('arrowleft');
    const right = k.has('d') || k.has('arrowright');
    const target = (left ? 1 : 0) - (right ? 1 : 0);
    this.manual.steer += Math.max(-0.08, Math.min(0.08, target - this.manual.steer));
    const throttle = (k.has('w') || k.has('arrowup') ? 1 : 0) - (k.has('s') || k.has('arrowdown') ? 1 : 0);
    return [this.manual.steer, throttle, k.has(' ') ? 1 : 0];
  }

  // ------------------------------------------------------------------ render frame
  async frame() {
    const now = performance.now();
    const dt = Math.min((now - this.last) / 1000, 0.1);
    this.last = now;
    this.fps = THREE.MathUtils.lerp(this.fps, 1 / Math.max(dt, 1e-3), 0.05);

    if (!this.paused && !this.loadingMap) {
      this.acc += dt;
      let n = 0;
      while (this.acc >= CONTROL_DT && n < MAX_TICKS_PER_FRAME) {
        await this.tick();
        this.acc -= CONTROL_DT;
        n++;
      }
      if (n === MAX_TICKS_PER_FRAME) this.acc = 0; // drop time instead of spiralling
    }

    if (this.lastState && !this.loadingMap) this.updateVisuals(dt);
    this.engine.render();
    requestAnimationFrame(() => this.frame());
  }

  updateVisuals(dt) {
    const veh = this.physics.vehicle;
    const st = veh.state();
    const time = this.engine.clock.getElapsed();
    const speed = st.velocity.length();

    this.car.update(st, veh, dt, time);
    this.track.update(time, st.position);
    const origin = st.position;
    this.rays.update(origin, this.rayBuf.points, this.rayBuf.dist, RAY_MAX);
    this.dust.update(dt, [veh.wheels[2], veh.wheels[3]], st.velocity, veh.slip.rearRatio);
    this.windFx.update(dt, this.lastWind, st.position, st.position.y - 1, time);
    this.courseFx.update(time, this.physics.coneBodies, this.race.boosting);
    this.updateCamera(dt, st);
    this.engine.update(dt, st.position, speed);

    this.hud.update({
      speed: st.vx,
      slip: veh.slip.rearRatio,
      throttle: veh.throttle,
      steer: veh.steer / veh.maxSteer,
      handbrake: veh.handbrake,
      latency: this.inference.lastLatencyMs,
      fps: this.fps,
      backend: this.mode === 'ai' ? this.inference.backend : 'keyboard',
      mode: this.mode === 'ai' ? 'AI PILOT' : 'MANUAL',
      lap: this.race.lap,
      lapTime: this.physics.time - this.race.lapStart,
      lastLap: this.race.lastLap,
      bestLap: this.race.bestLap,
      crashes: this.race.crashes,
      obstacleHits: this.race.obstacleHits,
      boosting: this.race.boosting,
      load: veh.load,
      mu: this.physics.terrain.mu,
      mapLabel: MAP_PROFILES[this.map].label.toUpperCase(),
    });
    this.radar.draw(this.rayBuf.dist, this.lastWind?.lateral ?? 0, this.obs);
  }

  // ------------------------------------------------------------------ camera rig
  cameraTarget(st, time) {
    const fwd = new THREE.Vector3(st.forward.x, 0, st.forward.z).normalize();
    const p = st.position;
    switch (CAMERA_MODES[this.cameraMode]) {
      case 'cinematic': {
        const a = time * 0.25;
        return {
          pos: p.clone().add(new THREE.Vector3(Math.cos(a) * 14, 4.5 + Math.sin(time * 0.4) * 1.5, Math.sin(a) * 14)),
          look: p.clone().add(new THREE.Vector3(0, 0.6, 0)),
          k: 3,
        };
      }
      case 'hood':
        return { pos: p.clone().addScaledVector(fwd, 0.6).add(new THREE.Vector3(0, 1.0, 0)), look: p.clone().addScaledVector(fwd, 20).add(new THREE.Vector3(0, 0.6, 0)), k: 30 };
      case 'top':
        return { pos: p.clone().addScaledVector(fwd, -6).add(new THREE.Vector3(0, 55, 0)), look: p.clone().addScaledVector(fwd, 8), k: 4 };
      default: // chase
        return { pos: p.clone().addScaledVector(fwd, -9.5).add(new THREE.Vector3(0, 3.6, 0)), look: p.clone().addScaledVector(fwd, 4).add(new THREE.Vector3(0, 1.0, 0)), k: 6 };
    }
  }

  updateCamera(dt, st) {
    const t = this.cameraTarget(st, this.engine.clock.getElapsed());
    this.camPos.x = THREE.MathUtils.damp(this.camPos.x, t.pos.x, t.k, dt);
    this.camPos.y = THREE.MathUtils.damp(this.camPos.y, t.pos.y, t.k, dt);
    this.camPos.z = THREE.MathUtils.damp(this.camPos.z, t.pos.z, t.k, dt);
    this.camLook.lerp(t.look, 1 - Math.exp(-t.k * 1.5 * dt));
    this.camera.position.copy(this.camPos);
    this.camera.lookAt(this.camLook);
    const fov = 60 + Math.min(Math.max(st.vx - 10, 0), 30) * 0.35;
    if (Math.abs(this.camera.fov - fov) > 0.05) {
      this.camera.fov = THREE.MathUtils.damp(this.camera.fov, fov, 3, dt);
      this.camera.updateProjectionMatrix();
    }
  }

  snapCamera() {
    const st = this.physics.vehicle.state();
    const t = this.cameraTarget(st, 0);
    this.camPos.copy(t.pos);
    this.camLook.copy(t.look);
  }

  // ------------------------------------------------------------------ UI / input
  bindUI() {
    document.querySelectorAll('.map-btn').forEach((b) =>
      b.addEventListener('click', () => !this.loadingMap && b.dataset.map !== this.map && this.loadMap(b.dataset.map)),
    );
    const modeBtn = document.getElementById('btn-mode');
    const raysBtn = document.getElementById('btn-rays');
    const toggleMode = () => {
      this.mode = this.mode === 'ai' ? 'manual' : 'ai';
      this.manual.steer = 0;
      modeBtn.classList.toggle('on', this.mode === 'ai');
    };
    const toggleRays = () => {
      this.rays.visible = !this.rays.visible;
      raysBtn.classList.toggle('on', this.rays.visible);
    };
    const cycleCam = () => (this.cameraMode = (this.cameraMode + 1) % CAMERA_MODES.length);
    modeBtn.addEventListener('click', toggleMode);
    raysBtn.addEventListener('click', toggleRays);
    document.getElementById('btn-camera').addEventListener('click', cycleCam);
    document.getElementById('btn-reset').addEventListener('click', () => this.resetRace());
    document.getElementById('btn-shuffle').addEventListener('click', () => !this.loadingMap && this.newCourse());

    window.addEventListener('keydown', (e) => {
      const k = e.key.toLowerCase();
      if ([' ', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(k)) e.preventDefault();
      if (e.repeat) return;
      this.keys.add(k);
      if (k === 'm') toggleMode();
      else if (k === 'c') cycleCam();
      else if (k === 'v') toggleRays();
      else if (k === 'r') this.resetRace();
      else if (k === 'p') this.paused = !this.paused;
      else if (k === 'h') this.hud.toggleHidden();
      else if (k === 'o' && !this.loadingMap) this.newCourse();
      else if (['1', '2', '3', '4'].includes(k) && !this.loadingMap) {
        const m = MAPS[Number(k) - 1];
        if (m !== this.map) this.loadMap(m);
      }
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.key.toLowerCase()));
    window.addEventListener('blur', () => this.keys.clear());
    window.addEventListener('resize', () => this.radar.resize());
  }
}

const app = new App();
app.init().catch((e) => {
  console.error(e);
  document.getElementById('loading-text').textContent = `Startup failed: ${e.message}`;
});
window.apex = app; // handy for debugging in the console
