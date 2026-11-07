// Particle & trail shaders: soft point sprites (dust), additive neon ribbons (ice light trails)
// and fading line streaks (wind).
import * as THREE from 'three';

/** Soft, size-attenuated point sprites with per-particle alpha/size/color. */
export const ParticlePointsShader = {
  uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, { uScale: { value: 400 } }]),
  vertexShader: /* glsl */ `
    attribute float aSize; attribute float aAlpha; attribute vec3 aColor;
    uniform float uScale; varying float vAlpha; varying vec3 vColor;
    #include <fog_pars_vertex>
    void main(){
      vAlpha = aAlpha; vColor = aColor;
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = aSize * uScale / max(-mvPosition.z, 0.1);
      gl_Position = projectionMatrix * mvPosition;
      #include <fog_vertex>
    }`,
  fragmentShader: /* glsl */ `
    varying float vAlpha; varying vec3 vColor;
    #include <fog_pars_fragment>
    void main(){
      vec2 c = gl_PointCoord - 0.5; float r = dot(c, c) * 4.0;
      if (r > 1.0) discard;
      float a = vAlpha * (1.0 - r) * (1.0 - r);
      gl_FragColor = vec4(vColor, a);
      #include <fog_fragment>
    }`,
};

/** Additive ribbon: per-vertex alpha fades with age, `vSide` gives a glowing core across the width. */
export const RibbonTrailShader = {
  uniforms: { uColor: { value: new THREE.Color(0xff2d95) }, uIntensity: { value: 4.0 } },
  vertexShader: /* glsl */ `
    attribute float aAlpha; attribute float aSide; varying float vAlpha; varying float vSide;
    void main(){ vAlpha = aAlpha; vSide = aSide; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */ `
    uniform vec3 uColor; uniform float uIntensity; varying float vAlpha; varying float vSide;
    void main(){
      float core = 1.0 - abs(vSide);
      vec3 col = uColor * uIntensity * (0.35 + core * core) + vec3(1.0) * pow(core, 6.0) * 1.5;
      gl_FragColor = vec4(col * vAlpha, vAlpha * core);
    }`,
};

/** Wind streak lines: alpha ramps from tail (0) to head (1). */
export const StreakShader = {
  uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, { uColor: { value: new THREE.Color(0xbff6ff) }, uOpacity: { value: 0.5 } }]),
  vertexShader: /* glsl */ `
    attribute float aHead; attribute float aFade; varying float vA;
    #include <fog_pars_vertex>
    void main(){ vA = aHead * aFade; vec4 mvPosition = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mvPosition;
      #include <fog_vertex>
    }`,
  fragmentShader: /* glsl */ `
    uniform vec3 uColor; uniform float uOpacity; varying float vA;
    #include <fog_pars_fragment>
    void main(){ gl_FragColor = vec4(uColor, vA * uOpacity);
      #include <fog_fragment>
    }`,
};

export function makeShaderMaterial(shader, extra = {}) {
  return new THREE.ShaderMaterial({
    uniforms: THREE.UniformsUtils.clone(shader.uniforms),
    vertexShader: shader.vertexShader,
    fragmentShader: shader.fragmentShader,
    transparent: true,
    depthWrite: false,
    ...extra,
  });
}
