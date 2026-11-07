// Supporting shaders: gradient sky dome (with optional synthwave sun), interactive instanced grass,
// neon grid ground, radial speed blur (motion blur) and a procedural colour-grading LUT.
import * as THREE from 'three';

export function createSkyMaterial({ top, horizon, bottom, sunDir, sunColor, sunSize = 0.03, synthSun = false }) {
  return new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
    uniforms: {
      uTop: { value: new THREE.Color(top) },
      uHorizon: { value: new THREE.Color(horizon) },
      uBottom: { value: new THREE.Color(bottom) },
      uSunDir: { value: sunDir.clone().normalize() },
      uSunColor: { value: new THREE.Color(sunColor) },
      uSunSize: { value: sunSize },
      uSynth: { value: synthSun ? 1 : 0 },
    },
    vertexShader: /* glsl */ `varying vec3 vDir; void main(){ vDir = normalize(position); vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_Position = p.xyww; }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uTop, uHorizon, uBottom, uSunDir, uSunColor; uniform float uSunSize, uSynth; varying vec3 vDir;
      void main(){
        vec3 d = normalize(vDir);
        float h = d.y;
        vec3 col = h > 0.0 ? mix(uHorizon, uTop, pow(h, 0.55)) : mix(uHorizon, uBottom, pow(-h, 0.4));
        float sd = distance(d, uSunDir);
        if (uSynth > 0.5) {
          float disc = smoothstep(uSunSize, uSunSize - 0.004, sd);
          float yy = (d.y - uSunDir.y) / uSunSize;              // -1..1 across the sun
          float stripes = step(0.5, fract(yy * 5.0 + 0.5)) + step(0.25, yy);
          vec3 sun = mix(vec3(1.0, 0.15, 0.55), vec3(1.0, 0.85, 0.2), clamp(yy * 0.5 + 0.5, 0.0, 1.0));
          col = mix(col, sun * 1.0, disc * clamp(stripes, 0.0, 1.0));
          col += uSunColor * exp(-sd * 9.0) * 0.6;
        } else {
          col += uSunColor * (smoothstep(uSunSize, uSunSize * 0.6, sd) * 4.0 + exp(-sd * 6.0) * 0.45);
        }
        gl_FragColor = vec4(col, 1.0);
      }`,
  });
}

/** Instanced grass blades: wind sway + bend away from the car (uCarPos), base->tip gradient, fog. */
export function createGrassMaterial() {
  return new THREE.ShaderMaterial({
    side: THREE.DoubleSide,
    fog: true,
    uniforms: THREE.UniformsUtils.merge([
      THREE.UniformsLib.fog,
      {
        uTime: { value: 0 },
        uCarPos: { value: new THREE.Vector3(1e5, 0, 1e5) },
        uBase: { value: new THREE.Color(0x0b4d2a) },
        uTip: { value: new THREE.Color(0x9be15d) },
        uSun: { value: new THREE.Color(0xffd27a) },
      },
    ]),
    vertexShader: /* glsl */ `
      uniform float uTime; uniform vec3 uCarPos; varying float vH; varying float vShade;
      #include <fog_pars_vertex>
      void main(){
        vec3 pos = position;
        vH = uv.y;
        vec4 world = modelMatrix * instanceMatrix * vec4(pos, 1.0);
        vec3 root = (modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
        float k = vH * vH;
        float sway = sin(uTime * 1.8 + root.x * 0.15 + root.z * 0.11) * 0.25 + sin(uTime * 3.7 + root.x * 0.6) * 0.07;
        world.x += sway * k; world.z += sway * 0.6 * k;
        vec2 away = root.xz - uCarPos.xz; float d = length(away);
        float push = smoothstep(5.0, 0.0, d);
        world.xz += normalize(away + 1e-4) * push * k * 0.9;
        world.y -= push * k * 0.35;
        vShade = 0.75 + 0.25 * fract(sin(dot(root.xz, vec2(12.9898, 78.233))) * 43758.5453);
        vec4 mvPosition = viewMatrix * world;
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uBase, uTip, uSun; varying float vH; varying float vShade;
      #include <fog_pars_fragment>
      void main(){
        vec3 col = mix(uBase, uTip, vH) * vShade;
        col += uSun * pow(vH, 4.0) * 0.25;
        gl_FragColor = vec4(col, 1.0);
        #include <fog_fragment>
      }`,
  });
}

/** Dark synthwave floor with glowing grid lines that fade with distance. */
export function createNeonGridMaterial() {
  return new THREE.ShaderMaterial({
    fog: true,
    uniforms: THREE.UniformsUtils.merge([
      THREE.UniformsLib.fog,
      { uTime: { value: 0 }, uColorA: { value: new THREE.Color(0xff2d95) }, uColorB: { value: new THREE.Color(0x00f3ff) } },
    ]),
    vertexShader: /* glsl */ `
      varying vec3 vWorld;
      #include <fog_pars_vertex>
      void main(){ vec4 wp = modelMatrix * vec4(position, 1.0); vWorld = wp.xyz; vec4 mvPosition = viewMatrix * wp; gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }`,
    fragmentShader: /* glsl */ `
      uniform float uTime; uniform vec3 uColorA, uColorB; varying vec3 vWorld;
      #include <fog_pars_fragment>
      void main(){
        vec2 g = vWorld.xz / 12.0;
        vec2 w = fwidth(g);
        vec2 l = smoothstep(w * 1.5, vec2(0.0), abs(fract(g - 0.5) - 0.5));
        float line = max(l.x, l.y);
        float pulse = 0.6 + 0.4 * sin(uTime * 1.5 - length(vWorld.xz) * 0.02);
        vec3 col = vec3(0.004, 0.001, 0.01) + mix(uColorA, uColorB, 0.5 + 0.5 * sin(vWorld.x * 0.004)) * line * 0.32 * pulse;
        gl_FragColor = vec4(col, 1.0);
        #include <fog_fragment>
      }`,
  });
}

/** Radial speed blur ("motion blur") around screen centre, strength driven by vehicle speed. */
export const SpeedBlurShader = {
  name: 'SpeedBlurShader',
  uniforms: { tDiffuse: { value: null }, uAmount: { value: 0 }, uCenter: { value: new THREE.Vector2(0.5, 0.45) } },
  vertexShader: /* glsl */ `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse; uniform float uAmount; uniform vec2 uCenter; varying vec2 vUv;
    void main(){
      vec2 dir = vUv - uCenter; float dist = length(dir);
      float amt = uAmount * smoothstep(0.15, 0.7, dist);
      vec4 acc = vec4(0.0);
      for (int i = 0; i < 8; i++) acc += texture2D(tDiffuse, vUv - dir * amt * float(i) / 8.0);
      gl_FragColor = acc / 8.0;
    }`,
};

/** Procedural 3D LUT (Data3DTexture) for LUTPass: warm desert grade = lifted, orange shadows + teal-free highlights. */
export function createWarmLUT(size = 32) {
  const data = new Uint8Array(size * size * size * 4);
  let i = 0;
  for (let b = 0; b < size; b++)
    for (let g = 0; g < size; g++)
      for (let r = 0; r < size; r++) {
        let R = r / (size - 1), G = g / (size - 1), B = b / (size - 1);
        const lum = 0.299 * R + 0.587 * G + 0.114 * B;
        // split-tone: warm shadows / golden highlights, gentle S-curve, slight desaturation of blues
        R = R + 0.08 * (1 - lum) + 0.05 * lum;
        G = G + 0.03 * (1 - lum) + 0.02 * lum;
        B = B * 0.86 - 0.02 * (1 - lum);
        const s = (x) => x * x * (3 - 2 * x);
        R = 0.25 * R + 0.75 * s(Math.min(Math.max(R, 0), 1));
        G = 0.3 * G + 0.7 * s(Math.min(Math.max(G, 0), 1));
        B = 0.3 * B + 0.7 * s(Math.min(Math.max(B, 0), 1));
        data[i++] = Math.round(Math.min(Math.max(R, 0), 1) * 255);
        data[i++] = Math.round(Math.min(Math.max(G, 0), 1) * 255);
        data[i++] = Math.round(Math.min(Math.max(B, 0), 1) * 255);
        data[i++] = 255;
      }
  const tex = new THREE.Data3DTexture(data, size, size, size);
  tex.format = THREE.RGBAFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}
