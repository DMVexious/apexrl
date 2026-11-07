// Themed obstacle meshes, animated boost-pad chevrons and instanced knock-over cones.
import * as THREE from 'three';
import { SIM } from '../core/SimConfig.js';

const PAD_COLORS = { grass: 0x00f3ff, desert: 0xffaa00, windy: 0x22fff0, ice: 0xff2d95 };

const BoostPadShader = {
  uniforms: { uTime: { value: 0 }, uColor: { value: new THREE.Color(0x00f3ff) }, uActive: { value: 0 } },
  vertexShader: /* glsl */ `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */ `
    uniform float uTime, uActive; uniform vec3 uColor; varying vec2 vUv;
    void main(){
      // vUv.x: along the track (0..1), vUv.y: across (0..1). Chevrons point forward and scroll.
      float across = abs(vUv.y - 0.5) * 2.0;
      float v = fract(vUv.x * 4.0 - across * 0.45 - uTime * 1.6);
      float chevron = smoothstep(0.0, 0.08, v) * (1.0 - smoothstep(0.32, 0.42, v));
      float border = smoothstep(0.86, 0.95, across);
      float a = max(chevron * 0.9, border * 0.8) + 0.12;
      vec3 col = uColor * (1.4 + chevron * 1.6 + uActive * 2.0);
      gl_FragColor = vec4(col, a);
    }`,
};

export class CourseVisuals {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    scene.add(this.group);
    this.padMats = [];
    this.cones = null;
  }

  build(course, geo, map) {
    this.clear();
    this.map = map;
    for (const o of course.obstacles) this.group.add(this.obstacleMesh(o, map));
    for (const p of course.pads) this.group.add(this.padMesh(p, geo, map));
    this.buildCones(course.cones.length, map);
  }

  obstacleMesh(o, map) {
    const g = new THREE.Group();
    const r = o.r;
    const std = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, flatShading: true, roughness: 0.8, ...extra });
    if (map === 'grass') {
      // stack of tyres with a red/white band on top
      for (let k = 0; k < 3; k++) {
        const t = new THREE.Mesh(new THREE.TorusGeometry(r * 0.72, r * 0.28, 8, 16), std(0x1a1a1a, { roughness: 0.95 }));
        t.rotation.x = Math.PI / 2;
        t.position.y = 0.28 * r + k * 0.5 * r;
        g.add(t);
      }
      const band = new THREE.Mesh(new THREE.TorusGeometry(r * 0.72, r * 0.12, 6, 16), std(0xe3262f));
      band.rotation.x = Math.PI / 2;
      band.position.y = 1.55 * r;
      g.add(band);
    } else if (map === 'desert') {
      const rock = new THREE.Mesh(new THREE.DodecahedronGeometry(r * 1.05, 0), std(0xa4532d, { roughness: 1 }));
      rock.scale.set(1, 0.85, 1);
      rock.rotation.set(Math.random(), Math.random() * 6, Math.random());
      rock.position.y = r * 0.6;
      g.add(rock);
      const cap = new THREE.Mesh(new THREE.DodecahedronGeometry(r * 0.55, 0), std(0xc9743f));
      cap.position.set(r * 0.2, r * 1.25, -r * 0.1);
      g.add(cap);
    } else if (map === 'windy') {
      const pillar = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.85, r, 4.2, 6), std(0x1b222c, { roughness: 0.75 }));
      pillar.position.y = 2.1;
      g.add(pillar);
      const ring = new THREE.Mesh(
        new THREE.CylinderGeometry(r * 0.98, r * 0.98, 0.16, 6, 1, true),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(0x22fff0).multiplyScalar(2.2), toneMapped: false }),
      );
      ring.position.y = 1.4;
      g.add(ring);
    } else {
      const crystal = new THREE.Mesh(
        new THREE.OctahedronGeometry(r, 0),
        new THREE.MeshStandardMaterial({ color: 0x2a0820, emissive: 0xff2d95, emissiveIntensity: 0.9, metalness: 0.6, roughness: 0.15, flatShading: true }),
      );
      crystal.scale.set(1, 2.2, 1);
      crystal.position.y = r * 2.1;
      g.add(crystal);
      const halo = new THREE.Mesh(
        new THREE.RingGeometry(r * 1.05, r * 1.35, 24),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(0xff2d95).multiplyScalar(1.4), side: THREE.DoubleSide, toneMapped: false }),
      );
      halo.rotation.x = -Math.PI / 2;
      halo.position.y = 0.05;
      g.add(halo);
    }
    g.traverse((m) => {
      if (m.isMesh) {
        m.castShadow = true;
        m.receiveShadow = true;
      }
    });
    g.position.set(o.x, o.h, -o.y);
    return g;
  }

  padMesh(p, geo, map) {
    // ribbon following the road surface over the pad length
    const hw = SIM.course.boost_half_width;
    const pos = [];
    const uv = [];
    const idx = [];
    const n = p.pts;
    for (let k = 0; k <= n; k++) {
      const i = geo.wrap(p.idx + k);
      for (const side of [-1, 1]) {
        const lat = p.lat + side * hw;
        pos.push(geo.cx[i] + geo.nx[i] * lat, geo.elev[i] + 0.05, -(geo.cy[i] + geo.ny[i] * lat));
        uv.push(k / n, side < 0 ? 0 : 1);
      }
      if (k < n) idx.push(2 * k, 2 * k + 2, 2 * k + 1, 2 * k + 1, 2 * k + 2, 2 * k + 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    const mat = new THREE.ShaderMaterial({
      uniforms: THREE.UniformsUtils.clone(BoostPadShader.uniforms),
      vertexShader: BoostPadShader.vertexShader,
      fragmentShader: BoostPadShader.fragmentShader,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    });
    mat.uniforms.uColor.value.set(PAD_COLORS[map]);
    this.padMats.push(mat);
    const mesh = new THREE.Mesh(g, mat);
    mesh.renderOrder = 2;
    return mesh;
  }

  buildCones(count, map) {
    if (!count) return;
    const neon = map === 'ice';
    const body = new THREE.ConeGeometry(0.24, 0.7, 10);
    const mat = new THREE.MeshStandardMaterial({
      color: neon ? 0xff2d95 : 0xff6a00,
      emissive: neon ? 0xff2d95 : 0x000000,
      emissiveIntensity: neon ? 0.8 : 0,
      roughness: 0.55,
      flatShading: true,
    });
    this.cones = new THREE.InstancedMesh(body, mat, count);
    this.cones.castShadow = true;
    this.cones.frustumCulled = false;
    this.group.add(this.cones);
    // white reflective band
    this.coneBands = new THREE.InstancedMesh(
      new THREE.CylinderGeometry(0.135, 0.165, 0.12, 10, 1, true),
      new THREE.MeshStandardMaterial({ color: 0xf5f5f5, emissive: neon ? 0x00f3ff : 0x222222, emissiveIntensity: neon ? 1.2 : 0.3 }),
      count,
    );
    this.coneBands.frustumCulled = false;
    this.group.add(this.coneBands);
  }

  /** Sync cone transforms from Rapier and animate pads. */
  update(time, coneBodies, boosting) {
    for (const m of this.padMats) {
      m.uniforms.uTime.value = time;
      m.uniforms.uActive.value = THREE.MathUtils.damp(m.uniforms.uActive.value, boosting ? 1 : 0, 6, 1 / 60);
    }
    if (!this.cones) return;
    const m4 = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const p = new THREE.Vector3();
    const one = new THREE.Vector3(1, 1, 1);
    const bandOff = new THREE.Vector3();
    coneBodies.forEach((b, i) => {
      const t = b.translation();
      const r = b.rotation();
      q.set(r.x, r.y, r.z, r.w);
      // Rapier's cone collider and THREE.ConeGeometry are both centred at half height.
      p.set(t.x, t.y, t.z);
      m4.compose(p, q, one);
      this.cones.setMatrixAt(i, m4);
      bandOff.set(0, 0.02, 0).applyQuaternion(q).add(p);
      m4.compose(bandOff, q, one);
      this.coneBands.setMatrixAt(i, m4);
    });
    this.cones.instanceMatrix.needsUpdate = true;
    this.coneBands.instanceMatrix.needsUpdate = true;
  }

  clear() {
    this.group.traverse((o) => {
      o.geometry?.dispose();
      const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
      mats.forEach((m) => m.dispose());
    });
    this.group.clear();
    this.padMats = [];
    this.cones = null;
    this.coneBands = null;
  }
}
