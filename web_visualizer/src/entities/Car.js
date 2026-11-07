// Procedural low-poly race car (local +X forward, -Z left) with suspension-driven wheels,
// visual body roll/pitch, emissive lights and neon light trails from the rear tyres.
import * as THREE from 'three';
import { RibbonTrailShader, makeShaderMaterial } from '../shaders/ParticleTrailShader.js';

const LIVERY = { grass: 0xe8262f, desert: 0x00a6e0, windy: 0xffa400, ice: 0xff2d95 };

export class Car {
  constructor(scene) {
    this.scene = scene;
    this.root = new THREE.Group();
    this.body = new THREE.Group(); // rolls/pitches relative to the chassis
    this.root.add(this.body);
    this.paint = new THREE.MeshStandardMaterial({ color: LIVERY.grass, metalness: 0.55, roughness: 0.3, flatShading: true });
    this.buildBody();
    this.buildWheels();
    scene.add(this.root);

    this.trails = [new LightTrail(scene, 0xff2d95), new LightTrail(scene, 0x00f3ff)];
    this.trailsEnabled = false;
    this.prevVel = new THREE.Vector3();
    this.roll = 0;
    this.pitch = 0;
  }

  buildBody() {
    const dark = new THREE.MeshStandardMaterial({ color: 0x15181f, metalness: 0.3, roughness: 0.6, flatShading: true });
    const glass = new THREE.MeshStandardMaterial({ color: 0x0b1a2a, metalness: 0.9, roughness: 0.08 });
    const add = (geo, mat, x, y, z) => {
      const m = new THREE.Mesh(geo, mat);
      m.position.set(x, y, z);
      m.castShadow = true;
      m.receiveShadow = true;
      this.body.add(m);
      return m;
    };
    // hull: tapered nose via scaled box segments
    add(new THREE.BoxGeometry(2.6, 0.5, 1.9), this.paint, -0.35, -0.05, 0);
    const nose = add(new THREE.CylinderGeometry(0.62, 0.95, 1.7, 4, 1), this.paint, 1.6, -0.12, 0);
    nose.rotation.set(Math.PI / 4, 0, -Math.PI / 2);
    nose.scale.set(1, 1, 0.55);
    add(new THREE.BoxGeometry(0.5, 0.18, 2.05), dark, 2.25, -0.3, 0); // splitter
    // cabin
    const cabin = new THREE.CylinderGeometry(0.5, 0.82, 1.9, 4, 1);
    cabin.rotateY(Math.PI / 4);
    cabin.rotateZ(Math.PI / 2);
    const c = add(cabin, glass, -0.35, 0.42, 0);
    c.scale.set(0.42, 1, 0.95);
    add(new THREE.BoxGeometry(1.2, 0.06, 1.3), this.paint, -0.55, 0.66, 0); // roof
    // rear wing
    add(new THREE.BoxGeometry(0.45, 0.06, 2.0), dark, -1.85, 0.62, 0);
    add(new THREE.BoxGeometry(0.1, 0.42, 0.08), dark, -1.8, 0.38, 0.65);
    add(new THREE.BoxGeometry(0.1, 0.42, 0.08), dark, -1.8, 0.38, -0.65);
    // side pods / skirts
    add(new THREE.BoxGeometry(2.4, 0.22, 0.12), dark, -0.2, -0.28, 0.98);
    add(new THREE.BoxGeometry(2.4, 0.22, 0.12), dark, -0.2, -0.28, -0.98);
    // lights
    this.headMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0xdff6ff).multiplyScalar(3), toneMapped: false });
    this.tailMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0xff1030).multiplyScalar(2.5), toneMapped: false });
    for (const z of [0.62, -0.62]) {
      add(new THREE.BoxGeometry(0.06, 0.1, 0.42), this.headMat, 2.1, -0.02, z);
      add(new THREE.BoxGeometry(0.06, 0.12, 0.5), this.tailMat, -1.66, 0.05, z);
    }
    // underglow (visible mostly on ice / at night)
    this.glow = new THREE.Mesh(
      new THREE.PlaneGeometry(4.6, 2.6),
      new THREE.MeshBasicMaterial({ color: 0x00f3ff, map: radialTexture(), transparent: true, opacity: 0.0, blending: THREE.AdditiveBlending, depthWrite: false }),
    );
    this.glow.rotation.x = -Math.PI / 2;
    this.glow.position.y = -0.62;
    this.root.add(this.glow);
  }

  buildWheels() {
    const tire = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.9, flatShading: true });
    const rim = new THREE.MeshStandardMaterial({ color: 0xc8ccd4, metalness: 0.9, roughness: 0.25, flatShading: true });
    this.wheelMeshes = [];
    for (let i = 0; i < 4; i++) {
      const pivot = new THREE.Group(); // steering
      const spin = new THREE.Group(); // rolling
      const t = new THREE.Mesh(new THREE.CylinderGeometry(0.36, 0.36, 0.32, 14), tire);
      t.rotation.x = Math.PI / 2;
      t.castShadow = true;
      const r = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.34, 6), rim);
      r.rotation.x = Math.PI / 2;
      spin.add(t, r);
      pivot.add(spin);
      this.root.add(pivot);
      this.wheelMeshes.push({ pivot, spin });
    }
  }

  setMap(map) {
    this.paint.color.set(LIVERY[map]);
    if (map === 'ice') {
      this.paint.emissive.set(0x400020);
    } else {
      this.paint.emissive.set(0x000000);
    }
    this.glow.material.opacity = map === 'ice' ? 0.45 : map === 'windy' ? 0.3 : 0.0;
    this.glow.material.color.set(map === 'windy' ? 0x22fff0 : 0x00f3ff);
    this.trailsEnabled = map === 'ice';
    for (const t of this.trails) t.clear();
  }

  /** @param st vehicle.state()  @param vehicle RaycastVehicle */
  update(st, vehicle, dt, time) {
    this.root.position.copy(st.position);
    this.root.quaternion.copy(st.quaternion);

    // visual roll / pitch from body-frame acceleration
    const acc = st.velocity.clone().sub(this.prevVel).divideScalar(Math.max(dt, 1e-3));
    this.prevVel.copy(st.velocity);
    const aLong = acc.dot(st.forward);
    const aLat = acc.dot(st.left);
    this.roll = THREE.MathUtils.damp(this.roll, THREE.MathUtils.clamp(-aLat * 0.008, -0.08, 0.08), 8, dt);
    this.pitch = THREE.MathUtils.damp(this.pitch, THREE.MathUtils.clamp(aLong * 0.006, -0.06, 0.06), 8, dt);
    this.body.rotation.set(this.roll, 0, this.pitch);

    vehicle.wheels.forEach((w, i) => {
      const { pivot, spin } = this.wheelMeshes[i];
      const travel = w.contact ? 0.81 - w.compression : 0.81;
      pivot.position.set(w.x, -0.2 - travel + 0.36, w.z);
      pivot.rotation.y = w.front ? vehicle.steer : 0;
      spin.rotation.z = -w.spin;
    });

    // brake lights brighten when braking / handbraking
    const braking = vehicle.throttle < -0.05 || vehicle.handbrake > 0.1;
    this.tailMat.color.setRGB(1, 0.06, 0.18).multiplyScalar(braking ? 6 : 2.2);

    if (this.trailsEnabled) {
      const slip = vehicle.slip.rearRatio;
      const intensity = THREE.MathUtils.smoothstep(slip, 0.4, 0.55);
      [2, 3].forEach((wi, k) => {
        const w = vehicle.wheels[wi];
        const p = w.contact ? w.hitPoint.clone().addScaledVector(st.up, 0.04) : null;
        this.trails[k].push(p, st.left, intensity, time);
      });
    }
    for (const t of this.trails) t.update(time);
  }
}

function radialTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.5, 'rgba(255,255,255,0.35)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

/** Ribbon of the last N tyre contact points; alpha = slip intensity x age fade. */
class LightTrail {
  constructor(scene, color, maxPoints = 240, life = 1.8) {
    this.max = maxPoints;
    this.life = life;
    this.points = [];
    this.geo = new THREE.BufferGeometry();
    this.pos = new Float32Array(maxPoints * 2 * 3);
    this.alpha = new Float32Array(maxPoints * 2);
    this.side = new Float32Array(maxPoints * 2);
    for (let i = 0; i < maxPoints; i++) this.side.set([-1, 1], i * 2);
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute('aAlpha', new THREE.BufferAttribute(this.alpha, 1).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute('aSide', new THREE.BufferAttribute(this.side, 1));
    const idx = [];
    for (let i = 0; i < maxPoints - 1; i++) {
      const a = i * 2;
      idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
    this.geo.setIndex(idx);
    const mat = makeShaderMaterial(RibbonTrailShader, { blending: THREE.AdditiveBlending, side: THREE.DoubleSide });
    mat.uniforms.uColor.value.set(color);
    this.mesh = new THREE.Mesh(this.geo, mat);
    this.mesh.frustumCulled = false;
    scene.add(this.mesh);
  }

  clear() {
    this.points.length = 0;
    this.geo.setDrawRange(0, 0);
  }

  push(p, left, intensity, time) {
    const last = this.points[this.points.length - 1];
    if (!p) {
      if (last) last.intensity = 0;
      return;
    }
    if (last && last.p.distanceToSquared(p) < 0.09) {
      last.intensity = Math.max(last.intensity, intensity);
      return;
    }
    this.points.push({ p, l: left.clone(), intensity, t: time });
    if (this.points.length > this.max) this.points.shift();
  }

  update(time) {
    const n = this.points.length;
    for (let i = 0; i < n; i++) {
      const { p, l, intensity, t } = this.points[i];
      const fade = Math.max(0, 1 - (time - t) / this.life);
      const w = 0.16 + 0.12 * intensity;
      this.pos.set([p.x - l.x * w, p.y, p.z - l.z * w, p.x + l.x * w, p.y, p.z + l.z * w], i * 6);
      const a = intensity * fade;
      this.alpha[i * 2] = a;
      this.alpha[i * 2 + 1] = a;
    }
    this.geo.attributes.position.needsUpdate = true;
    this.geo.attributes.aAlpha.needsUpdate = true;
    this.geo.setDrawRange(0, Math.max(0, (n - 1) * 6));
  }

  dispose() {
    this.mesh.parent?.remove(this.mesh);
    this.geo.dispose();
  }
}
