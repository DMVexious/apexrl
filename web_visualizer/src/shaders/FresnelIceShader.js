// RFL_Glacier — synthwave ice road: Fresnel-weighted sky reflection over a parallax "refraction" of
// layered sub-surface cracks, neon lane edges and a cyan glow pool under the car. Outputs HDR for bloom.
import * as THREE from 'three';

const NOISE = /* glsl */ `
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float vnoise(vec2 p){ vec2 i=floor(p), f=fract(p); vec2 u=f*f*(3.-2.*f);
  return mix(mix(hash12(i),hash12(i+vec2(1,0)),u.x), mix(hash12(i+vec2(0,1)),hash12(i+vec2(1,1)),u.x), u.y); }
float fbm(vec2 p){ float a=.5, s=0.; for(int i=0;i<5;i++){ s+=a*vnoise(p); p*=2.03; a*=.5; } return s; }
// Voronoi edge distance -> crack lines
float cracks(vec2 p){ vec2 n=floor(p), f=fract(p); float md=8., md2=8.;
  for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){ vec2 g=vec2(i,j); vec2 o=vec2(hash12(n+g), hash12(n+g+17.3));
    vec2 r=g+o-f; float d=dot(r,r); if(d<md){ md2=md; md=d; } else if(d<md2){ md2=d; } }
  return smoothstep(0.0, 0.06, sqrt(md2)-sqrt(md)); }
`;

export const FresnelIceShader = {
  name: 'RFL_Glacier',
  uniforms: THREE.UniformsUtils.merge([
    THREE.UniformsLib.fog,
    {
      uTime: { value: 0 },
      uCarPos: { value: new THREE.Vector3() },
      uSunDir: { value: new THREE.Vector3(0, 0.15, -1).normalize() },
      uDeep: { value: new THREE.Color(0x020a1c) },
      uShallow: { value: new THREE.Color(0x2fb7d6) },
      uSkyLow: { value: new THREE.Color(0xff3fa4) },
      uSkyHigh: { value: new THREE.Color(0x1a0638) },
      uNeonA: { value: new THREE.Color(0x00f3ff) },
      uNeonB: { value: new THREE.Color(0xff2d95) },
    },
  ]),
  vertexShader: /* glsl */ `
    varying vec3 vWorld; varying vec2 vUv; varying vec3 vNormalW;
    #include <fog_pars_vertex>
    void main(){
      vUv = uv;
      vec4 wp = modelMatrix * vec4(position, 1.0);
      vWorld = wp.xyz;
      vNormalW = normalize(mat3(modelMatrix) * normal);
      vec4 mvPosition = viewMatrix * wp;
      gl_Position = projectionMatrix * mvPosition;
      #include <fog_vertex>
    }`,
  fragmentShader: /* glsl */ `
    uniform float uTime; uniform vec3 uCarPos; uniform vec3 uSunDir;
    uniform vec3 uDeep, uShallow, uSkyLow, uSkyHigh, uNeonA, uNeonB;
    varying vec3 vWorld; varying vec2 vUv; varying vec3 vNormalW;
    #include <fog_pars_fragment>
    ${NOISE}
    void main(){
      vec3 V = normalize(cameraPosition - vWorld);
      vec2 p = vWorld.xz;
      // micro-normal perturbation from fbm gradient (frozen ripples)
      float e = 0.35;
      float h0 = fbm(p*0.35), hx = fbm((p+vec2(e,0.))*0.35), hz = fbm((p+vec2(0.,e))*0.35);
      vec3 N = normalize(vNormalW + vec3(h0-hx, 0.0, h0-hz) * 0.9);

      float NdV = clamp(dot(N, V), 0.0, 1.0);
      float F = 0.04 + 0.96 * pow(1.0 - NdV, 5.0);

      // Fake refraction: parallax-sampled crack layers at three depths below the surface
      vec3 Rf = refract(-V, N, 1.0/1.31);
      vec3 sub = uDeep;
      for(int i=1;i<=3;i++){
        float depth = float(i) * 0.9;
        vec2 q = p + Rf.xz / max(-Rf.y, 0.2) * depth;
        float c = 1.0 - cracks(q * (0.18 + 0.07*float(i)));
        float frost = fbm(q*0.6 + float(i)*7.1);
        sub += (uShallow * c * 0.16 + uShallow * frost * 0.03) / float(i);
      }

      // Reflection of a synthwave sky + sun glint
      vec3 R = reflect(-V, N);
      float sky = clamp(R.y*1.6, 0.0, 1.0);
      vec3 refl = mix(uSkyLow*0.22, uSkyHigh*0.6, sky);
      refl += uSkyLow * pow(max(dot(R, uSunDir), 0.0), 160.0) * 3.0;

      vec3 col = mix(sub, refl, F * 0.85);

      // Neon lane edges + scrolling grid across the road
      float edge = smoothstep(0.035, 0.0, vUv.y) + smoothstep(0.965, 1.0, vUv.y);
      col += mix(uNeonB, uNeonA, step(0.5, vUv.y)) * edge * 1.1;
      float grid = smoothstep(0.03, 0.0, abs(fract(vUv.x*0.5) - 0.5) - 0.47);
      col += uNeonA * grid * 0.08;
      float center = smoothstep(0.012, 0.0, abs(vUv.y-0.5)) * step(0.5, fract(vUv.x*0.6));
      col += vec3(0.8, 0.9, 1.0) * center * 0.45;

      // Underglow pool following the car
      float d = length(vWorld.xz - uCarPos.xz);
      col += uNeonA * exp(-d*d*0.18) * 0.35;

      gl_FragColor = vec4(col, 1.0);
      #include <fog_fragment>
    }`,
};

export function createFresnelIceMaterial() {
  return new THREE.ShaderMaterial({
    name: FresnelIceShader.name,
    uniforms: THREE.UniformsUtils.clone(FresnelIceShader.uniforms),
    vertexShader: FresnelIceShader.vertexShader,
    fragmentShader: FresnelIceShader.fragmentShader,
    fog: true,
  });
}
