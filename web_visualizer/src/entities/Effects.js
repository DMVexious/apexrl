// Particle systems: velocity-dependent sand/dust emitters at the rear tyres (desert) and
// horizontally blowing volumetric wind streaks (windy canyons).
import * as THREE from 'three';
import { ParticlePointsShader, StreakShader, makeShaderMaterial } from '../shaders/ParticleTrailShader.js';

export class DustSystem {
  constructor(scene, max = 2400) {
    this.max = max;
    this.geo = new THREE.BufferGeometry();
    this.pos = new Float32Array(max * 3);
    this.vel = new Float32Array(max * 3);
    this.life = new Float32Array(max);
    this.maxLife = new Float32Array(max);
    this.size = new Float32Array(max);
    this.alpha = new Float32Array(max);
    this.color = new Float32Array(max * 3);
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute('aSize', new THREE.BufferAttribute(this.size, 1).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute('aAlpha', new THREE.BufferAttribute(this.alpha, 1).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute('aColor', new THREE.BufferAttribute(this.color, 3));
    this.mat = makeShaderMaterial(ParticlePointsShader, { fog: true });
    this.points = new THREE.Points(this.geo, this.mat);
    this.points.frustumCulled = false;
    scene.add(this.points);
    this.cursor = 0;
    this.carry = 0;
    this.enabled = false;
    this.tint = new THREE.Color(0xe0a36a);
  }

  setMap(map) {
    this.enabled = map === 'desert' || map === 'grass';
    this.tint.set(map === 'desert' ? 0xe0a36a : 0x8a7a5a);
    this.rate = map === 'desert' ? 1 : 0.25; // grass only kicks up a little dirt when sliding
    this.life.fill(0);
    this.alpha.fill(0);
  }

  emit(p, v, speed) {
    const i = this.cursor;
    this.cursor = (this.cursor + 1) % this.max;
    this.pos.set([p.x + (Math.random() - 0.5) * 0.4, p.y + 0.1, p.z + (Math.random() - 0.5) * 0.4], i * 3);
    const spread = 1.2 + speed * 0.08;
    this.vel.set([v.x * 0.25 + (Math.random() - 0.5) * spread, 0.6 + Math.random() * 1.6, v.z * 0.25 + (Math.random() - 0.5) * spread], i * 3);
    this.maxLife[i] = this.life[i] = 1.2 + Math.random() * 1.6;
    const c = this.tint.clone().offsetHSL(0, 0, (Math.random() - 0.5) * 0.12);
    this.color.set([c.r, c.g, c.b], i * 3);
  }

  /** contacts: [{point, contact}] rear wheels; vel: car world velocity; slip 0..1 */
  update(dt, contacts, vel, slip) {
    if (this.enabled) {
      const speed = vel.length();
      const perSec = this.rate === 1 ? (speed * 9 + slip * 220) * (speed > 2 ? 1 : 0) : Math.max(0, slip - 0.3) * 400 * (speed > 4 ? 1 : 0);
      this.carry += perSec * dt;
      while (this.carry >= 1) {
        this.carry -= 1;
        const w = contacts[(Math.random() * contacts.length) | 0];
        if (w.contact) this.emit(w.hitPoint, vel, speed);
      }
    }
    let anyAlive = false;
    for (let i = 0; i < this.max; i++) {
      if (this.life[i] <= 0) {
        this.alpha[i] = 0;
        continue;
      }
      anyAlive = true;
      this.life[i] -= dt;
      const k = i * 3;
      this.vel[k] *= 1 - 0.9 * dt;
      this.vel[k + 2] *= 1 - 0.9 * dt;
      this.vel[k + 1] += (0.15 - this.vel[k + 1] * 0.8) * dt;
      this.pos[k] += this.vel[k] * dt;
      this.pos[k + 1] += this.vel[k + 1] * dt;
      this.pos[k + 2] += this.vel[k + 2] * dt;
      const t = 1 - this.life[i] / this.maxLife[i];
      this.size[i] = 0.6 + t * 3.2;
      this.alpha[i] = Math.sin(Math.PI * Math.min(t * 1.4, 1)) * 0.38 * (1 - t);
    }
    if (anyAlive || this.enabled) {
      this.geo.attributes.position.needsUpdate = true;
      this.geo.attributes.aSize.needsUpdate = true;
      this.geo.attributes.aAlpha.needsUpdate = true;
      this.geo.attributes.aColor.needsUpdate = true;
    }
  }
}

export class WindStreaks {
  constructor(scene, count = 520, box = 90) {
    this.count = count;
    this.box = box;
    this.geo = new THREE.BufferGeometry();
    this.pos = new Float32Array(count * 2 * 3);
    this.head = new Float32Array(count * 2);
    this.fade = new Float32Array(count * 2);
    this.seeds = [];
    for (let i = 0; i < count; i++) {
      this.head.set([0, 1], i * 2);
      this.seeds.push({
        x: (Math.random() - 0.5) * box * 2,
        y: 0.5 + Math.random() * 14,
        z: (Math.random() - 0.5) * box * 2,
        len: 2 + Math.random() * 5,
        speed: 0.7 + Math.random() * 0.6,
        phase: Math.random() * Math.PI * 2,
      });
    }
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute('aHead', new THREE.BufferAttribute(this.head, 1));
    this.geo.setAttribute('aFade', new THREE.BufferAttribute(this.fade, 1).setUsage(THREE.DynamicDrawUsage));
    this.mat = makeShaderMaterial(StreakShader, { fog: true, blending: THREE.AdditiveBlending });
    this.lines = new THREE.LineSegments(this.geo, this.mat);
    this.lines.frustumCulled = false;
    scene.add(this.lines);
    this.enabled = false;
  }

  setMap(map) {
    this.enabled = map === 'windy';
    this.lines.visible = this.enabled;
  }

  /** wind: {phi, strength} in the 2D sim plane; center: THREE.Vector3 around which streaks live. */
  update(dt, wind, center, groundY, time) {
    if (!this.enabled) return;
    const dir = new THREE.Vector3(Math.cos(wind.phi), 0, -Math.sin(wind.phi)); // 2D -> three
    const v = 12 + wind.strength * 38;
    const B = this.box;
    for (let i = 0; i < this.count; i++) {
      const s = this.seeds[i];
      s.x += dir.x * v * s.speed * dt;
      s.z += dir.z * v * s.speed * dt;
      // wrap inside a box that follows the car
      let rx = s.x - center.x, rz = s.z - center.z;
      if (rx > B) s.x -= 2 * B; else if (rx < -B) s.x += 2 * B;
      if (rz > B) s.z -= 2 * B; else if (rz < -B) s.z += 2 * B;
      const len = s.len * (0.4 + wind.strength * 1.6);
      const y = groundY + s.y + Math.sin(time * 2 + s.phase) * 0.3;
      this.pos.set([s.x - dir.x * len, y, s.z - dir.z * len, s.x, y, s.z], i * 6);
      const edge = Math.max(0, 1 - Math.max(Math.abs(rx), Math.abs(rz)) / B);
      const f = Math.min(1, edge * 3) * (0.25 + 0.75 * wind.strength);
      this.fade[i * 2] = f;
      this.fade[i * 2 + 1] = f;
    }
    this.geo.attributes.position.needsUpdate = true;
    this.geo.attributes.aFade.needsUpdate = true;
  }
}
