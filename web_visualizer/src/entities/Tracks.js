// Track + environment builder: road ribbon, barriers, low-poly terrain, sky, instanced foliage/props,
// and the per-map art direction profile (lights, fog, post FX) + surface physics.
import * as THREE from 'three';
import { MeshSurfaceSampler } from 'three/examples/jsm/math/MeshSurfaceSampler.js';
import { SIM } from '../core/SimConfig.js';
import { TrackGeometry } from '../core/TrackGeometry.js';
import { createFresnelIceMaterial } from '../shaders/FresnelIceShader.js';
import { createSkyMaterial, createGrassMaterial, createNeonGridMaterial } from '../shaders/EnvironmentShaders.js';

export const MAP_PROFILES = {
  grass: {
    label: 'Grassy Circuit',
    exposure: 1.05,
    hemi: [0xcfe8ff, 0x2f5d2a, 0.9],
    sun: { color: 0xffe2a8, intensity: 3.0, dir: new THREE.Vector3(0.6, 0.75, 0.35) },
    fog: { color: 0xb5d6ea, density: 0.0016 },
    bloom: { strength: 0.3, radius: 0.4, threshold: 0.92 },
    sky: { top: 0x2a72cf, horizon: 0xd4ebf7, bottom: 0x6a8f5a, sunColor: 0xffd27a, sunSize: 0.035 },
    dof: true,
    motionBlur: true,
  },
  desert: {
    label: 'Desert Dunes',
    exposure: 1.1,
    hemi: [0xfff0d6, 0x8a4a22, 0.85],
    sun: { color: 0xffc27a, intensity: 3.4, dir: new THREE.Vector3(-0.5, 0.5, -0.6) },
    fog: { color: 0xf0b98a, density: 0.0021 },
    bloom: { strength: 0.4, radius: 0.5, threshold: 0.85 },
    sky: { top: 0x13b5dc, horizon: 0xffd7a8, bottom: 0xc9743f, sunColor: 0xfff0c0, sunSize: 0.04 },
    heat: true,
    lut: true,
  },
  windy: {
    label: 'Windy Canyons',
    exposure: 1.0,
    hemi: [0x6fa3bd, 0x0a0f1a, 0.75],
    sun: { color: 0x9fd8ff, intensity: 1.7, dir: new THREE.Vector3(-0.3, 0.7, 0.5) },
    fog: { color: 0x1d3346, density: 0.0062, elevationK: 0.035, refHeight: 12 },
    bloom: { strength: 0.6, radius: 0.5, threshold: 0.72 },
    sky: { top: 0x040a1c, horizon: 0x2b5a72, bottom: 0x0b1420, sunColor: 0x8fe9ff, sunSize: 0.02 },
  },
  ice: {
    label: 'Synthwave Glacier',
    exposure: 1.0,
    hemi: [0x6a3cff, 0x050010, 0.35],
    sun: { color: 0xff6ad5, intensity: 0.9, dir: new THREE.Vector3(0.0, 0.35, -1) },
    fog: { color: 0x14062c, density: 0.0021 },
    bloom: { strength: 1.5, radius: 0.8, threshold: 0.2 },
    sky: { top: 0x030008, horizon: 0x3a0a3a, bottom: 0x05010f, sunColor: 0x5a0a3a, sunSize: 0.13, synth: true, sunDir: new THREE.Vector3(0, 0.1, -1) },
  },
};

// ------------------------------------------------------------------ helpers
const smooth = (a, b, x) => {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
  return t * t * (3 - 2 * t);
};
function hash(x, y) {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
}
function vnoise(x, y) {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy);
  const a = hash(ix, iy), b = hash(ix + 1, iy), c = hash(ix, iy + 1), d = hash(ix + 1, iy + 1);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}
function fbm(x, y, oct = 4) {
  let s = 0, a = 0.5;
  for (let i = 0; i < oct; i++) {
    s += a * vnoise(x, y);
    x *= 2.03; y *= 2.03; a *= 0.5;
  }
  return s;
}

function roadTexture(map) {
  const c = document.createElement('canvas');
  c.width = 512;
  c.height = 256;
  const g = c.getContext('2d');
  const base = { grass: '#34373d', desert: '#8a6a4f', windy: '#22262c' }[map];
  g.fillStyle = base;
  g.fillRect(0, 0, 512, 256);
  for (let i = 0; i < 9000; i++) {
    const v = Math.random();
    g.fillStyle = `rgba(${v > 0.5 ? 255 : 0},${v > 0.5 ? 255 : 0},${v > 0.5 ? 255 : 0},${Math.random() * 0.06})`;
    g.fillRect(Math.random() * 512, Math.random() * 256, 2, 2);
  }
  if (map === 'desert') {
    // drifting sand streaks
    for (let i = 0; i < 40; i++) {
      g.fillStyle = `rgba(230,170,110,${0.08 + Math.random() * 0.12})`;
      g.beginPath();
      g.ellipse(Math.random() * 512, Math.random() * 256, 30 + Math.random() * 90, 4 + Math.random() * 12, 0, 0, Math.PI * 2);
      g.fill();
    }
  }
  const edge = { grass: '#ffffff', desert: '#f3e2c4', windy: '#2ff3e0' }[map];
  g.fillStyle = edge;
  g.fillRect(0, 14, 512, 6);
  g.fillRect(0, 236, 512, 6);
  if (map === 'grass') {
    for (let x = 0; x < 512; x += 64) {
      g.fillStyle = (x / 64) % 2 ? '#ffffff' : '#e3262f';
      g.fillRect(x, 0, 64, 12);
      g.fillRect(x, 244, 64, 12);
    }
  }
  g.fillStyle = map === 'windy' ? '#e8f6ff' : '#f6d55c';
  for (let x = 0; x < 512; x += 128) g.fillRect(x, 125, 64, 6);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.anisotropy = 8;
  return tex;
}

// ------------------------------------------------------------------ Track
export class Track {
  constructor(scene, mapName) {
    this.scene = scene;
    this.map = mapName;
    this.profile = MAP_PROFILES[mapName];
    this.surface = SIM.terrains[mapName]; // surface physics: mu, drive (sinking), rear grip, wind
    this.geo = new TrackGeometry(SIM.tracks[mapName], SIM.track_points);
    this.group = new THREE.Group();
    this.animated = []; // materials with uTime
    this.carUniforms = []; // uniforms taking the car position
    this.build();
    scene.add(this.group);
  }

  build() {
    this.buildSky();
    this.buildRoad();
    this.buildWalls();
    this.buildGround();
    this.buildProps();
    this.buildStartLine();
  }

  buildSky() {
    const s = this.profile.sky;
    const mat = createSkyMaterial({
      top: s.top,
      horizon: s.horizon,
      bottom: s.bottom,
      sunDir: s.sunDir || this.profile.sun.dir,
      sunColor: s.sunColor,
      sunSize: s.sunSize,
      synthSun: !!s.synth,
    });
    this.sky = new THREE.Mesh(new THREE.SphereGeometry(1400, 48, 24), mat);
    this.sky.renderOrder = -1;
    this.sky.frustumCulled = false;
    this.group.add(this.sky);
  }

  buildRoad() {
    const t = this.geo;
    const N = t.N;
    const w = t.halfWidth + 0.6;
    const pos = new Float32Array((N + 1) * 2 * 3);
    const uv = new Float32Array((N + 1) * 2 * 2);
    const nrm = new Float32Array((N + 1) * 2 * 3);
    const index = [];
    for (let k = 0; k <= N; k++) {
      const i = k % N;
      const l = t.edge(i, +1, w);
      const r = t.edge(i, -1, w);
      const h = t.elev[i] + 0.02;
      pos.set([l.x, h, -l.y, r.x, h, -r.y], k * 6);
      const u = (k === N ? t.length : t.s[i]) / 16;
      uv.set([u, 1, u, 0], k * 4);
      nrm.set([0, 1, 0, 0, 1, 0], k * 6);
      if (k < N) {
        const a = 2 * k;
        index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
    g.setIndex(index);
    g.computeVertexNormals();

    let mat;
    if (this.map === 'ice') {
      mat = createFresnelIceMaterial();
      this.animated.push(mat);
      this.carUniforms.push(mat.uniforms.uCarPos);
    } else {
      mat = new THREE.MeshStandardMaterial({ map: roadTexture(this.map), roughness: this.map === 'desert' ? 0.95 : 0.8, metalness: 0.0 });
    }
    this.road = new THREE.Mesh(g, mat);
    this.road.receiveShadow = true;
    this.group.add(this.road);
  }

  wallRibbon(side, height, inset = 0) {
    const t = this.geo;
    const N = t.N;
    const pos = [];
    const col = [];
    const index = [];
    const thick = 0.5;
    for (let k = 0; k <= N; k++) {
      const i = k % N;
      const a = t.edge(i, side, t.halfWidth + inset);
      const b = t.edge(i, side, t.halfWidth + inset + thick);
      const h = t.elev[i];
      pos.push(a.x, h - 0.5, -a.y, a.x, h + height, -a.y, b.x, h + height, -b.y);
      const stripe = Math.floor(k / 3) % 2;
      for (let j = 0; j < 3; j++) col.push(stripe, stripe, stripe);
      if (k < N) {
        const o = 3 * k;
        // inner face + top face (winding so faces point toward the road / up)
        if (side > 0) index.push(o, o + 3, o + 1, o + 1, o + 3, o + 4, o + 1, o + 4, o + 2, o + 2, o + 4, o + 5);
        else index.push(o, o + 1, o + 3, o + 1, o + 4, o + 3, o + 1, o + 2, o + 4, o + 2, o + 5, o + 4);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('stripe', new THREE.Float32BufferAttribute(col, 3));
    g.setIndex(index);
    g.computeVertexNormals();
    return g;
  }

  stripRibbon(side, y, width = 0.12) {
    const t = this.geo;
    const pos = [];
    const index = [];
    for (let k = 0; k <= t.N; k++) {
      const i = k % t.N;
      const a = t.edge(i, side, t.halfWidth - 0.01);
      const h = t.elev[i] + y;
      pos.push(a.x, h - width, -a.y, a.x, h + width, -a.y);
      if (k < t.N) index.push(2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 1, 2 * k + 3, 2 * k + 2);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setIndex(index);
    return g;
  }

  buildWalls() {
    const cfg = {
      grass: { h: 0.9, a: 0xe8ecf0, b: 0xd7262f, rough: 0.6 },
      desert: { h: 1.0, a: 0xc98a55, b: 0xb06f3e, rough: 1.0 },
      windy: { h: 1.1, a: 0x2b3440, b: 0x232a33, rough: 0.9, strip: [0x22fff0, 0x22fff0] },
      ice: { h: 0.7, a: 0x0d0b24, b: 0x14102e, rough: 0.2, strip: [0xff2d95, 0x00f3ff] },
    }[this.map];
    for (const side of [+1, -1]) {
      const g = this.wallRibbon(side, cfg.h);
      // map the 0/1 stripe attribute to two colours
      const st = g.getAttribute('stripe');
      const ca = new THREE.Color(cfg.a), cb = new THREE.Color(cfg.b);
      const colors = new Float32Array(st.count * 3);
      for (let i = 0; i < st.count; i++) (st.getX(i) ? cb : ca).toArray(colors, i * 3);
      g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      const m = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: cfg.rough, metalness: this.map === 'ice' ? 0.6 : 0.05, flatShading: true, side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(g, m);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.group.add(mesh);
      if (cfg.strip) {
        const color = new THREE.Color(cfg.strip[side > 0 ? 0 : 1]).multiplyScalar(this.map === 'ice' ? 1.3 : 2.2);
        const strip = new THREE.Mesh(this.stripRibbon(side, cfg.h * 0.75), new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, toneMapped: false }));
        this.group.add(strip);
      }
    }
  }

  heightAt(x2, y2, info) {
    const t = this.geo;
    const d = info.dist;
    const hw = t.halfWidth;
    const base = info.elev - 0.15;
    switch (this.map) {
      case 'grass': {
        const near = smooth(hw + 4, hw + 22, d);
        return base + near * (fbm(x2 * 0.012, y2 * 0.012) * 14 - 2 + smooth(hw + 60, hw + 200, d) * 18);
      }
      case 'desert': {
        const near = smooth(hw + 4, hw + 26, d);
        const dune = Math.pow(Math.abs(Math.sin(x2 * 0.018 + Math.sin(y2 * 0.011) * 2.2 + fbm(x2 * 0.01, y2 * 0.01) * 3)), 1.6);
        return base + near * (dune * 9 + fbm(x2 * 0.02, y2 * 0.02) * 6);
      }
      case 'windy': {
        const cliff = smooth(hw + 8, hw + 34, d);
        const rock = fbm(x2 * 0.03, y2 * 0.03, 5);
        return base + smooth(hw + 3, hw + 10, d) * 1.5 + cliff * (26 + rock * 30) - (1 - cliff) * 0.0;
      }
      default: {
        // ice: the neon floor follows the road's gentle dips/rises near the track, flat far away
        return info.elev * (1 - smooth(hw + 6, hw + 70, d)) - 0.15;
      }
    }
  }

  buildGround() {
    const t = this.geo;
    const size = 1100;
    const seg = this.map === 'ice' ? 160 : 230;
    const g = new THREE.PlaneGeometry(size, size, seg, seg);
    g.rotateX(-Math.PI / 2);
    const pos = g.getAttribute('position');
    const colors = new Float32Array(pos.count * 3);
    const c = new THREE.Color();
    this.groundInfo = [];
    for (let i = 0; i < pos.count; i++) {
      const x2 = pos.getX(i);
      const y2 = -pos.getZ(i);
      const info = t.distanceInfo(x2, y2);
      const h = this.heightAt(x2, y2, info);
      pos.setY(i, h);
      if (this.map === 'ice') continue; // grid shader, no vertex colours
      const n = fbm(x2 * 0.05, y2 * 0.05);
      const rel = h - info.elev;
      if (this.map === 'grass') {
        c.setHSL(0.31 + n * 0.06, 0.55, 0.22 + n * 0.12 + rel * 0.006);
        if (rel > 16) c.lerp(new THREE.Color(0x6f7a80), smooth(16, 26, rel)); // slate peaks
      } else if (this.map === 'desert') {
        c.setHSL(0.065 + n * 0.02, 0.62, 0.42 + n * 0.12 + Math.min(rel, 12) * 0.008);
        if (n > 0.62) c.lerp(new THREE.Color(0xa8502c), 0.5); // terracotta patches
      } else {
        c.setHSL(0.58 + n * 0.04, 0.22, 0.06 + n * 0.08 + Math.min(rel, 50) * 0.002);
        if (rel > 18 && n > 0.55) c.lerp(new THREE.Color(0x1b6b6b), 0.35); // teal mineral veins
      }
      c.toArray(colors, i * 3);
    }
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    g.computeVertexNormals();
    let mat;
    if (this.map === 'ice') {
      mat = createNeonGridMaterial();
      this.animated.push(mat);
    } else {
      mat = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.95 });
    }
    this.ground = new THREE.Mesh(g, mat);
    this.ground.receiveShadow = true;
    this.group.add(this.ground);
  }

  scatter(count, minD, maxD, fn, maxTries = count * 8) {
    // Samples ground surface points (MeshSurfaceSampler) within a distance band from the centerline.
    const sampler = new MeshSurfaceSampler(this.ground).build();
    const p = new THREE.Vector3();
    const nrm = new THREE.Vector3();
    let n = 0;
    for (let tries = 0; tries < maxTries && n < count; tries++) {
      sampler.sample(p, nrm);
      const info = this.geo.distanceInfo(p.x, -p.z, 8);
      if (info.dist < minD || info.dist > maxD) continue;
      fn(n++, p, nrm, info);
    }
    return n;
  }

  buildProps() {
    const t = this.geo;
    const hw = t.halfWidth;
    const m4 = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    const place = (mesh, count, minD, maxD, scale, yOff = 0, tilt = 0) => {
      const placed = this.scatter(count, minD, maxD, (i, p) => {
        q.setFromAxisAngle(up, Math.random() * Math.PI * 2);
        if (tilt) q.multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler((Math.random() - 0.5) * tilt, 0, (Math.random() - 0.5) * tilt)));
        const k = scale[0] + Math.random() * (scale[1] - scale[0]);
        s.set(k, k * (scale[2] ? 0.7 + Math.random() * 0.8 : 1), k);
        m4.compose(p.clone().setY(p.y + yOff * k), q, s);
        mesh.setMatrixAt(i, m4);
      });
      mesh.count = placed;
      mesh.instanceMatrix.needsUpdate = true;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.group.add(mesh);
      return mesh;
    };

    if (this.map === 'grass') {
      // Interactive instanced grass blades
      const blade = new THREE.BufferGeometry();
      blade.setAttribute('position', new THREE.Float32BufferAttribute([-0.06, 0, 0, 0.06, 0, 0, -0.04, 0.45, 0, 0.04, 0.45, 0, 0, 0.9, 0], 3));
      blade.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 0, 0.5, 1, 0.5, 0.5, 1], 2));
      blade.setIndex([0, 1, 2, 1, 3, 2, 2, 3, 4]);
      const grassMat = createGrassMaterial();
      this.animated.push(grassMat);
      this.carUniforms.push(grassMat.uniforms.uCarPos);
      const grass = new THREE.InstancedMesh(blade, grassMat, 70000);
      const placed = this.scatter(70000, hw + 1.2, hw + 38, (i, p) => {
        q.setFromAxisAngle(up, Math.random() * Math.PI * 2);
        const k = 0.6 + Math.random() * 0.8;
        m4.compose(p, q, s.set(k, k * (0.7 + Math.random() * 0.7), k));
        grass.setMatrixAt(i, m4);
      }, 260000);
      grass.count = placed;
      grass.frustumCulled = false;
      this.group.add(grass);

      // Low-poly pines
      const pine = new THREE.InstancedMesh(
        mergeColored([
          [new THREE.CylinderGeometry(0.25, 0.35, 2, 6).translate(0, 1, 0), 0x5b3a22],
          [new THREE.ConeGeometry(2.2, 4.5, 7).translate(0, 3.6, 0), 0x1f6b3a],
          [new THREE.ConeGeometry(1.6, 3.4, 7).translate(0, 5.6, 0), 0x2a8a4a],
        ]),
        new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.9 }),
        420,
      );
      place(pine, 420, hw + 16, hw + 140, [0.8, 1.6, 1]);
    } else if (this.map === 'desert') {
      const rock = new THREE.InstancedMesh(
        new THREE.DodecahedronGeometry(1.6, 0),
        new THREE.MeshStandardMaterial({ color: 0xa4532d, flatShading: true, roughness: 1 }),
        260,
      );
      place(rock, 260, hw + 8, hw + 160, [0.6, 2.6, 1], 0.3, 0.6);
      const cactus = new THREE.InstancedMesh(
        mergeColored([
          [new THREE.CylinderGeometry(0.35, 0.4, 4, 7).translate(0, 2, 0), 0x3f7d3a],
          [new THREE.CylinderGeometry(0.22, 0.25, 1.6, 6).rotateZ(Math.PI / 2).translate(0.8, 2.2, 0), 0x3f7d3a],
          [new THREE.CylinderGeometry(0.22, 0.25, 1.4, 6).translate(1.5, 2.9, 0), 0x46893f],
        ]),
        new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true }),
        140,
      );
      place(cactus, 140, hw + 6, hw + 90, [0.7, 1.4, 1]);
    } else if (this.map === 'windy') {
      const spire = new THREE.InstancedMesh(
        new THREE.CylinderGeometry(0.6, 3.2, 26, 6, 3),
        new THREE.MeshStandardMaterial({ color: 0x161c25, flatShading: true, roughness: 0.85 }),
        90,
      );
      place(spire, 90, hw + 14, hw + 70, [0.5, 1.4, 1], 10, 0.25);
      const boulder = new THREE.InstancedMesh(
        new THREE.IcosahedronGeometry(1.2, 0),
        new THREE.MeshStandardMaterial({ color: 0x252e3a, flatShading: true }),
        300,
      );
      place(boulder, 300, hw + 3, hw + 40, [0.5, 2.0, 1], 0.2, 1);
    } else if (this.map === 'ice') {
      const shard = new THREE.InstancedMesh(
        new THREE.OctahedronGeometry(2, 0),
        new THREE.MeshStandardMaterial({ color: 0x1a1240, emissive: 0x00c8ff, emissiveIntensity: 0.08, metalness: 0.8, roughness: 0.15, flatShading: true }),
        160,
      );
      // ice: no sampler-worthy terrain — place shards on a ring band directly
      let n = 0;
      for (let tries = 0; tries < 4000 && n < 160; tries++) {
        const x = (Math.random() - 0.5) * 700;
        const y = (Math.random() - 0.5) * 600;
        const di = t.distanceInfo(x, y, 8);
        const d = di.dist;
        if (d < hw + 10 || d > hw + 120) continue;
        const k = 0.6 + Math.random() * 1.8;
        q.setFromEuler(new THREE.Euler(0, Math.random() * 3, (Math.random() - 0.5) * 0.4));
        const floor = this.heightAt(x, y, di);
        m4.compose(new THREE.Vector3(x, floor + k * 1.6, -y), q, s.set(k, k * (1.5 + Math.random() * 2), k));
        shard.setMatrixAt(n++, m4);
      }
      shard.count = n;
      this.group.add(shard);
      // distant wireframe mountain ring
      const mount = new THREE.Group();
      for (let i = 0; i < 26; i++) {
        const a = (i / 26) * Math.PI * 2;
        const r = 520 + Math.random() * 120;
        const h = 60 + Math.random() * 110;
        const cone = new THREE.ConeGeometry(70 + Math.random() * 50, h, 4 + Math.floor(Math.random() * 3), 3);
        const solid = new THREE.Mesh(cone, new THREE.MeshBasicMaterial({ color: 0x07011a }));
        const wire = new THREE.LineSegments(new THREE.WireframeGeometry(cone), new THREE.LineBasicMaterial({ color: new THREE.Color(0xff2d95), transparent: true, opacity: 0.45 }));
        for (const o of [solid, wire]) {
          o.position.set(Math.cos(a) * r, h / 2 - 2, Math.sin(a) * r);
          mount.add(o);
        }
      }
      this.group.add(mount);
    }
  }

  buildStartLine() {
    const t = this.geo;
    const c = document.createElement('canvas');
    c.width = 64;
    c.height = 16;
    const g = c.getContext('2d');
    for (let x = 0; x < 16; x++)
      for (let y = 0; y < 4; y++) {
        g.fillStyle = (x + y) % 2 ? '#111' : '#f5f5f5';
        g.fillRect(x * 4, y * 4, 4, 4);
      }
    const tex = new THREE.CanvasTexture(c);
    tex.magFilter = THREE.NearestFilter;
    tex.colorSpace = THREE.SRGBColorSpace;
    const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(2.5, t.halfWidth * 2),
      new THREE.MeshStandardMaterial({ map: tex, roughness: 0.7, emissive: this.map === 'ice' ? 0x331144 : 0x000000 }),
    );
    mesh.rotation.x = -Math.PI / 2;
    mesh.rotation.z = Math.atan2(t.ty[0], t.tx[0]);
    mesh.position.set(t.cx[0], t.elev[0] + 0.04, -t.cy[0]);
    mesh.receiveShadow = true;
    this.group.add(mesh);
  }

  update(time, carPos) {
    for (const m of this.animated) m.uniforms.uTime.value = time;
    for (const u of this.carUniforms) u.value.copy(carPos);
  }

  dispose() {
    this.scene.remove(this.group);
    this.group.traverse((o) => {
      o.geometry?.dispose();
      const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
      for (const m of mats) {
        m.map?.dispose();
        m.dispose();
      }
    });
  }
}

/** Merge [geometry, color] pairs into one non-indexed BufferGeometry with vertex colours. */
function mergeColored(parts) {
  const pos = [];
  const nrm = [];
  const col = [];
  const c = new THREE.Color();
  for (const [geom, color] of parts) {
    const g = geom.index ? geom.toNonIndexed() : geom;
    g.computeVertexNormals();
    c.set(color);
    const p = g.getAttribute('position');
    const n = g.getAttribute('normal');
    for (let i = 0; i < p.count; i++) {
      pos.push(p.getX(i), p.getY(i), p.getZ(i));
      nrm.push(n.getX(i), n.getY(i), n.getZ(i));
      col.push(c.r, c.g, c.b);
    }
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  out.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
  out.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  return out;
}
