// Rapier3D world: road trimesh, wall colliders, and a raycast-suspension vehicle whose tyre forces
// use exactly the bicycle-model equations the policy was trained on (python_training/envs/raycast_car.py).
import RAPIER from '@dimforge/rapier3d-compat';
import * as THREE from 'three';
import { SIM, G, CONTROL_DT, SUBSTEPS, RAY_ANGLES, RAY_MAX, OBSTACLE_INFLATE, windAt } from './SimConfig.js';

const GROUP_ROAD = 0x0001;
const GROUP_WALL = 0x0002;
const GROUP_CAR = 0x0004;
const GROUP_PROP = 0x0008; // knock-over cones: collide with everything, invisible to the LIDAR
// Obstacles collide physically but are sensed analytically (inflated circles, like the trainer), not by Rapier rays.
const GROUP_OBSTACLE = 0x0010;
const groups = (member, filter) => ((member << 16) | filter) >>> 0;
const QUERY_ROAD = groups(0xffff, GROUP_ROAD);
const QUERY_WALL = groups(0xffff, GROUP_WALL);

let rapierReady = null;
export function initRapier() {
  rapierReady ??= RAPIER.init();
  return rapierReady;
}

const SUSP = { rest: 0.45, wheelRadius: 0.36, k: 30000, c: 3000, mountY: -0.2, trackHalf: 0.8 };

export class PhysicsWorld {
  constructor() {
    this.world = null;
    this.vehicle = null;
    this.time = 0;
  }

  /** (Re)builds the world for a track + terrain. */
  build(track, terrainName) {
    if (this.world) this.world.free();
    this.track = track;
    this.terrainName = terrainName;
    this.terrain = { ...SIM.terrains[terrainName] };
    this.world = new RAPIER.World({ x: 0, y: -G, z: 0 });
    this.world.timestep = CONTROL_DT / SUBSTEPS;
    this.wind = { max: this.terrain.wind, phi0: Math.random() * Math.PI * 2, phase: Math.random() * Math.PI * 2 };

    const fixed = this.world.createRigidBody(RAPIER.RigidBodyDesc.fixed());
    this._buildRoad(fixed, track);
    this._buildWalls(fixed, track);
    this.fixedBody = fixed;
    this.course = null;
    this.courseColliders = [];
    this.coneBodies = [];
    this.vehicle = new RaycastVehicle(this);
  }

  /** Replace obstacles (fixed cylinders, visible to the sensor rays) and cones (light dynamic props). */
  setCourse(course) {
    for (const c of this.courseColliders) this.world.removeCollider(c, false);
    for (const b of this.coneBodies) this.world.removeRigidBody(b);
    this.courseColliders = [];
    this.coneBodies = [];
    this.course = course;
    for (const o of course.obstacles) {
      // Tall cylinder footprint = exactly the circle the trainer uses for collisions.
      const desc = RAPIER.ColliderDesc.cylinder(4, o.r)
        .setTranslation(o.x, o.h + 1, -o.y)
        .setFriction(0.3)
        .setRestitution(0.2)
        .setCollisionGroups(groups(GROUP_OBSTACLE, GROUP_CAR | GROUP_PROP));
      this.courseColliders.push(this.world.createCollider(desc, this.fixedBody));
    }
    for (const c of course.cones) {
      const body = this.world.createRigidBody(
        RAPIER.RigidBodyDesc.dynamic().setTranslation(c.x, c.h + 0.37, -c.y).setLinearDamping(0.3).setAngularDamping(0.6),
      );
      const col = RAPIER.ColliderDesc.cone(0.35, 0.24)
        .setMass(5)
        .setFriction(0.8)
        .setRestitution(0.3)
        .setCollisionGroups(groups(GROUP_PROP, GROUP_ROAD | GROUP_WALL | GROUP_OBSTACLE | GROUP_CAR | GROUP_PROP));
      this.world.createCollider(col, body);
      this.coneBodies.push(body);
    }
  }

  _buildRoad(body, track) {
    const N = track.N;
    const w = track.halfWidth + 2.0; // extends under the walls so suspension never falls through
    const verts = new Float32Array(N * 2 * 3);
    const idx = new Uint32Array(N * 6);
    for (let i = 0; i < N; i++) {
      const l = track.edge(i, +1, w);
      const r = track.edge(i, -1, w);
      const h = track.elev[i];
      verts.set([l.x, h, -l.y, r.x, h, -r.y], i * 6);
      const j = (i + 1) % N;
      idx.set([2 * i, 2 * j, 2 * i + 1, 2 * i + 1, 2 * j, 2 * j + 1], i * 6);
    }
    const desc = RAPIER.ColliderDesc.trimesh(verts, idx).setCollisionGroups(groups(GROUP_ROAD, GROUP_CAR | GROUP_PROP)).setFriction(0.0);
    this.world.createCollider(desc, body);
    // Safety floor far below the road (catches a car that jumps a wall).
    let minH = Infinity;
    for (const h of track.elev) minH = Math.min(minH, h);
    const floor = RAPIER.ColliderDesc.cuboid(2000, 0.5, 2000)
      .setTranslation(0, minH - 3, 0)
      .setCollisionGroups(groups(GROUP_ROAD, GROUP_CAR | GROUP_PROP));
    this.world.createCollider(floor, body);
  }

  _buildWalls(body, track) {
    const N = track.N;
    const thick = 0.3;
    for (const side of [+1, -1]) {
      for (let i = 0; i < N; i++) {
        const a = track.edge(i, side);
        const b = track.edge(i + 1, side);
        const j = track.wrap(i + 1);
        const len = Math.hypot(b.x - a.x, b.y - a.y);
        const yaw = Math.atan2(b.y - a.y, b.x - a.x);
        // push the box outward so its inner face sits exactly on the boundary
        const ox = track.nx[i] * side * thick;
        const oy = track.ny[i] * side * thick;
        const mx = (a.x + b.x) / 2 + ox;
        const my = (a.y + b.y) / 2 + oy;
        const h = (track.elev[i] + track.elev[j]) / 2;
        const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
        // Tall (16 m) invisible walls: sensor rays run in the car's pitched plane, so on hills they must
        // still hit the wall ahead even where the road rises or falls (the 2D trainer has infinite walls).
        const desc = RAPIER.ColliderDesc.cuboid(len / 2 + 0.08, 8, thick)
          .setTranslation(mx, h + 2, -my)
          .setRotation({ x: q.x, y: q.y, z: q.z, w: q.w })
          .setFriction(0.15)
          .setRestitution(0.1)
          .setCollisionGroups(groups(GROUP_WALL, GROUP_CAR | GROUP_PROP));
        this.world.createCollider(desc, body);
      }
    }
  }

  windLateral(yaw) {
    const w = windAt(this.time, this.wind.max, this.wind.phi0, this.wind.phase);
    return { lateral: Math.max(-1, Math.min(1, w.strength * Math.sin(w.phi - yaw))), ...w };
  }

  /** One control tick: steering slew, SUBSTEPS physics steps with freshly computed tyre forces. */
  /** @param boost extra forward force (N) from a boost pad, decided from the pre-step pose like the trainer */
  step(action, boost = 0) {
    const v = this.vehicle;
    v.setControls(action);
    const wind = this.windLateral(v.state().yaw).lateral;
    for (let i = 0; i < SUBSTEPS; i++) {
      v.applyForces(wind, this.world.timestep, boost);
      this.world.step();
    }
    this.time += CONTROL_DT;
  }

  /** Wall / obstacle distances along the 11 sensor rays (car's local plane). Also returns hit points.
   * Obstacles are tested as 2D circles grown by sensors.obstacle_inflate, exactly like
   * python_training/envs/custom_car_env.py::_observe, so a reading is the free distance for the car's whole body. */
  castSensorRays(out = { dist: new Float32Array(RAY_ANGLES.length), points: RAY_ANGLES.map(() => new THREE.Vector3()) }) {
    const v = this.vehicle;
    const st = v.state();
    const origin = st.position;
    for (let i = 0; i < RAY_ANGLES.length; i++) {
      const a = RAY_ANGLES[i];
      const dir = st.forward.clone().multiplyScalar(Math.cos(a)).addScaledVector(st.left, Math.sin(a)).normalize();
      const ray = new RAPIER.Ray(origin, dir);
      const hit = this.world.castRay(ray, RAY_MAX, true, undefined, QUERY_WALL, undefined, v.body);
      let d = hit ? Math.min(hit.timeOfImpact ?? hit.toi, RAY_MAX) : RAY_MAX;
      const obs = this.course?.obstacles;
      if (obs?.length) {
        // horizontal component of the (possibly pitched) ray; circle distances are converted back to ray length
        const hx = dir.x, hy = -dir.z, hl = Math.hypot(hx, hy);
        for (const o of obs) {
          const rr = o.r + OBSTACLE_INFLATE;
          const fx = st.x2 - o.x, fy = st.y2 - o.y;
          const b = (fx * hx + fy * hy) / hl;
          const disc = b * b - (fx * fx + fy * fy - rr * rr);
          if (disc < 0) continue;
          const t = (-b - Math.sqrt(disc)) / hl;
          if (t > 0 && t < d) d = t;
        }
      }
      out.dist[i] = d;
      out.points[i].copy(origin).addScaledVector(dir, d);
    }
    return out;
  }
}

const _q = new THREE.Quaternion();
const _v = new THREE.Vector3();

export class RaycastVehicle {
  constructor(phys) {
    this.phys = phys;
    const car = SIM.car;
    this.p = car;
    this.maxSteer = (car.max_steer_deg * Math.PI) / 180;
    this.steer = 0;
    this.throttle = 0;
    this.handbrake = 0;
    this.wheels = [
      { x: car.lf, z: -SUSP.trackHalf, front: true }, // front-left (left = -Z)
      { x: car.lf, z: SUSP.trackHalf, front: true },
      { x: -car.lr, z: -SUSP.trackHalf, front: false },
      { x: -car.lr, z: SUSP.trackHalf, front: false },
    ].map((w) => ({ ...w, contact: false, compression: 0, hitPoint: new THREE.Vector3(), spin: 0 }));
    this.slip = { front: 0, rear: 0, rearRatio: 0 };
    this.load = 1; // normal force / (m g), measured from the suspension (crests < 1, compressions > 1)

    const desc = RAPIER.RigidBodyDesc.dynamic().setCanSleep(false).setCcdEnabled(true).setAngularDamping(0.05);
    this.body = phys.world.createRigidBody(desc);
    const col = RAPIER.ColliderDesc.cuboid(car.half_length, 0.35, car.half_width)
      .setMassProperties(car.mass, { x: 0, y: -0.25, z: 0 }, { x: 650, y: car.yaw_inertia, z: 2100 }, { x: 0, y: 0, z: 0, w: 1 })
      .setFriction(0.2)
      .setRestitution(0.05)
      .setCollisionGroups(groups(GROUP_CAR, GROUP_ROAD | GROUP_WALL | GROUP_OBSTACLE | GROUP_PROP));
    this.collider = phys.world.createCollider(col, this.body);
  }

  /** Teleport to track index with heading along the tangent. */
  reset(track, idx, lateral = 0, speed = 0) {
    const i = track.wrap(idx);
    const x = track.cx[i] + track.nx[i] * lateral;
    const y = track.cy[i] + track.ny[i] * lateral;
    const yaw = Math.atan2(track.ty[i], track.tx[i]);
    _q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
    this.body.setTranslation({ x, y: track.elev[i] + 0.95, z: -y }, true);
    this.body.setRotation({ x: _q.x, y: _q.y, z: _q.z, w: _q.w }, true);
    this.body.setLinvel({ x: Math.cos(yaw) * speed, y: 0, z: -Math.sin(yaw) * speed }, true);
    this.body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    this.steer = 0;
  }

  setControls([steer, throttle, handbrake]) {
    const clamp = (v, a, b) => Math.min(b, Math.max(a, Number.isFinite(v) ? v : 0));
    const cmd = clamp(steer, -1, 1) * this.maxSteer;
    const maxDelta = this.p.steer_rate * CONTROL_DT;
    this.steer += clamp(cmd - this.steer, -maxDelta, maxDelta);
    this.throttle = clamp(throttle, -1, 1);
    this.handbrake = clamp(handbrake, 0, 1);
  }

  /** Body-frame kinematics. */
  state() {
    const t = this.body.translation();
    const r = this.body.rotation();
    _q.set(r.x, r.y, r.z, r.w);
    const forward = new THREE.Vector3(1, 0, 0).applyQuaternion(_q);
    const left = new THREE.Vector3(0, 0, -1).applyQuaternion(_q);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(_q);
    const lv = this.body.linvel();
    const av = this.body.angvel();
    const vel = new THREE.Vector3(lv.x, lv.y, lv.z);
    return {
      position: new THREE.Vector3(t.x, t.y, t.z),
      quaternion: _q.clone(),
      forward,
      left,
      up,
      velocity: vel,
      vx: vel.dot(forward),
      vy: vel.dot(left),
      r: av.x * up.x + av.y * up.y + av.z * up.z,
      yaw: Math.atan2(-forward.z, forward.x),
      x2: t.x,
      y2: -t.z,
    };
  }

  applyForces(windLat, dt, boost = 0) {
    this.boosting = boost > 0;
    const body = this.body;
    const st = this.state();
    const world = this.phys.world;
    body.resetForces(true);
    body.resetTorques(true);

    // --- raycast suspension (4 wheels)
    let frontContact = 0;
    let rearContact = 0;
    let suspTotal = 0;
    const down = st.up.clone().negate();
    const lv = body.linvel();
    const av = body.angvel();
    const com = body.worldCom();
    for (const w of this.wheels) {
      const mount = new THREE.Vector3(w.x, SUSP.mountY, w.z).applyQuaternion(st.quaternion).add(st.position);
      const ray = new RAPIER.Ray(mount, down);
      const maxLen = SUSP.rest + SUSP.wheelRadius;
      const hit = world.castRay(ray, maxLen, true, undefined, QUERY_ROAD, undefined, body);
      w.contact = !!hit;
      if (hit) {
        const toi = hit.timeOfImpact ?? hit.toi;
        const comp = maxLen - toi;
        // point velocity along up
        const rx = mount.x - com.x, ry = mount.y - com.y, rz = mount.z - com.z;
        const pvx = lv.x + (av.y * rz - av.z * ry);
        const pvy = lv.y + (av.z * rx - av.x * rz);
        const pvz = lv.z + (av.x * ry - av.y * rx);
        const vUp = pvx * st.up.x + pvy * st.up.y + pvz * st.up.z;
        const f = Math.max(0, SUSP.k * comp - SUSP.c * vUp);
        body.addForceAtPoint({ x: st.up.x * f, y: st.up.y * f, z: st.up.z * f }, mount, true);
        suspTotal += f;
        w.compression = comp;
        w.hitPoint.copy(mount).addScaledVector(down, toi);
        if (w.front) frontContact += 0.5;
        else rearContact += 0.5;
      } else {
        w.compression = 0;
      }
      w.spin += (st.vx / SUSP.wheelRadius) * dt;
    }

    // --- normal-load factor: low-passed total suspension force (filters spring chatter, keeps crest unloading)
    const rawLoad = Math.min(2, suspTotal / (this.p.mass * G));
    this.load += (rawLoad - this.load) * Math.min(1, dt / 0.04);

    // --- bicycle-model tyre forces (identical equations to training)
    const F = computeTireForces(this.p, st.vx, st.vy, st.r, this.steer, this.throttle, this.handbrake, this.phys.terrain, windLat, this.load);
    const cAt = _v.set(0, -0.25, 0).applyQuaternion(st.quaternion).add(st.position); // CoM height
    const front = cAt.clone().addScaledVector(st.forward, this.p.lf);
    const rear = cAt.clone().addScaledVector(st.forward, -this.p.lr);
    const toWorld = (fx, fy) => ({
      x: st.forward.x * fx + st.left.x * fy,
      y: st.forward.y * fx + st.left.y * fy,
      z: st.forward.z * fx + st.left.z * fy,
    });
    // load already falls to 0 when airborne; the per-axle contact factor handles one axle off the ground
    const fC = frontContact > 0 ? 1 : 0;
    const rC = rearContact > 0 ? 1 : 0;
    body.addForceAtPoint(toWorld(F.frontFx * fC, F.frontFy * fC), front, true);
    body.addForceAtPoint(toWorld(F.rearFx * rC, F.rearFy * rC), rear, true);
    const grounded = (frontContact + rearContact) / 2;
    body.addForceAtPoint(toWorld(F.extraFx * grounded + (1 - grounded) * F.dragOnly + boost, F.extraFy), cAt, true);

    this.slip.front = F.alphaF;
    this.slip.rear = F.alphaR;
    const vRearLat = st.vy - this.p.lr * st.r;
    this.slip.rearRatio = Math.min(1, Math.abs(vRearLat) / Math.max(Math.hypot(st.vx, vRearLat), 2.0));
  }
}

/** Port of python_training/envs/raycast_car.py::car_forces (single env, mass_scale = 1).
 *  Slope gravity needs no port: Rapier applies gravity to the pitched chassis directly. */
export function computeTireForces(p, vx, vy, r, steer, throttle, hb, terrain, windLat, load = 1) {
  const m = p.mass;
  const L = p.lf + p.lr;
  const fzF = ((m * G * p.lr) / L) * load;
  const fzR = ((m * G * p.lf) / L) * load;
  const vRef = Math.max(Math.abs(vx), p.min_slip_speed);
  const alphaF = steer - Math.atan2(vy + p.lf * r, vRef);
  const alphaR = -Math.atan2(vy - p.lr * r, vRef);
  const mu = terrain.mu;
  const muRLong = mu * terrain.rear_grip;
  const muRLat = muRLong * (1 - p.handbrake_grip_cut * hb);
  const rollSign = Math.tanh(vx / 0.5);
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const fDrive = p.drive_force * terrain.drive * clamp(throttle, 0, 1);
  const fBrake = -p.brake_force * clamp(-throttle, 0, 1) * rollSign;
  const fHb = -p.handbrake_force * hb * rollSign;
  const fxRCap = muRLong * fzR;
  const fxR = clamp(fDrive + 0.4 * fBrake + fHb, -fxRCap, fxRCap);
  const fxFCap = mu * fzF;
  const fxF = clamp(0.6 * fBrake, -fxFCap, fxFCap);
  const fyF = mu * fzF * Math.tanh(p.tire_stiffness * alphaF);
  const fyRMax = Math.sqrt(Math.max((muRLat * fzR) ** 2 - fxR ** 2, 0) + (0.1 * muRLat * fzR) ** 2);
  const fyR = fyRMax * Math.tanh(p.tire_stiffness * alphaR);
  const cs = Math.cos(steer);
  const sn = Math.sin(steer);
  const dragOnly = -p.drag * vx * Math.abs(vx);
  return {
    frontFx: fxF * cs - fyF * sn,
    frontFy: fyF * cs + fxF * sn,
    rearFx: fxR,
    rearFy: fyR,
    extraFx: -terrain.roll * m * G * load * rollSign + dragOnly,
    extraFy: p.wind_force * windLat,
    dragOnly,
    alphaF,
    alphaR,
  };
}
