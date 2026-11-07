// Screen-space heat shimmer (desert). Applied as a full-screen quad ShaderPass: UVs are displaced by
// rising, animated noise that is strongest in a band around the horizon and over hot ground.
export const HeatDistortionShader = {
  name: 'HeatDistortionShader',
  uniforms: {
    tDiffuse: { value: null },
    uTime: { value: 0 },
    uStrength: { value: 0.0035 },
    uHorizon: { value: 0.55 },
    uBand: { value: 0.22 },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse; uniform float uTime, uStrength, uHorizon, uBand;
    varying vec2 vUv;
    float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
    float noise(vec2 p){ vec2 i=floor(p), f=fract(p); vec2 u=f*f*(3.-2.*f);
      return mix(mix(hash(i),hash(i+vec2(1,0)),u.x), mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),u.x), u.y); }
    void main(){
      float band = 1.0 - smoothstep(0.0, uBand, abs(vUv.y - uHorizon));
      float ground = smoothstep(uHorizon, 0.0, vUv.y) * 0.45;
      float mask = max(band, ground);
      vec2 q = vec2(vUv.x * 9.0, vUv.y * 22.0 - uTime * 2.2);
      vec2 off = vec2(noise(q) - 0.5, noise(q + 19.7) - 0.5);
      off += 0.5 * vec2(sin(vUv.y * 140.0 - uTime * 7.0), 0.0) * 0.4;
      vec2 uv = vUv + off * uStrength * mask * 2.0;
      // slight chromatic split in the hottest band
      float ca = uStrength * band * 0.6;
      vec3 col = vec3(texture2D(tDiffuse, uv + vec2(ca, 0.)).r, texture2D(tDiffuse, uv).g, texture2D(tDiffuse, uv - vec2(ca, 0.)).b);
      gl_FragColor = vec4(col, 1.0);
    }`,
};
