import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { BokehPass } from 'three/examples/jsm/postprocessing/BokehPass.js';
import { LUTPass } from 'three/examples/jsm/postprocessing/LUTPass.js';
import { HeatDistortionShader } from '../shaders/HeatDistortionShader.js';
import { SpeedBlurShader, createWarmLUT } from '../shaders/EnvironmentShaders.js';

export class Engine {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 3000);
    this.clock = new THREE.Timer();

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    container.appendChild(this.renderer.domElement);

    this.setupLights();
    this.setupPostProcessing();
    window.addEventListener('resize', () => this.onWindowResize());
  }

  setupPostProcessing() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.composer = new EffectComposer(this.renderer);
    this.renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(this.renderPass);

    // Grass: depth of field focused on the car + speed (motion) blur
    this.bokehPass = new BokehPass(this.scene, this.camera, { focus: 12, aperture: 0.00002, maxblur: 0.004 });
    this.composer.addPass(this.bokehPass);
    this.speedBlurPass = new ShaderPass(SpeedBlurShader);
    this.composer.addPass(this.speedBlurPass);

    // Desert: heat shimmer full-screen quad
    this.heatPass = new ShaderPass(HeatDistortionShader);
    this.composer.addPass(this.heatPass);

    this.bloomPass = new UnrealBloomPass(new THREE.Vector2(w, h), 0.6, 0.4, 0.85);
    this.composer.addPass(this.bloomPass);

    this.composer.addPass(new OutputPass());

    // Desert: warm colour-grading LUT (applied on display-referred colour, after tone mapping)
    this.lutPass = new LUTPass({ lut: createWarmLUT(), intensity: 0.85 });
    this.composer.addPass(this.lutPass);
  }

  setupLights() {
    this.hemiLight = new THREE.HemisphereLight(0xbfdcff, 0x3a2f20, 0.8);
    this.scene.add(this.hemiLight);

    this.dirLight = new THREE.DirectionalLight(0xfffaed, 2.5);
    this.dirLight.position.set(50, 80, 30);
    this.dirLight.castShadow = true;
    this.dirLight.shadow.mapSize.set(2048, 2048);
    this.dirLight.shadow.camera.near = 0.5;
    this.dirLight.shadow.camera.far = 260;
    const d = 45;
    Object.assign(this.dirLight.shadow.camera, { left: -d, right: d, top: d, bottom: -d });
    this.dirLight.shadow.bias = -0.0004;
    this.dirLight.shadow.normalBias = 0.03;
    this.dirLight.shadow.radius = 4;
    this.scene.add(this.dirLight, this.dirLight.target);
    this.sunOffset = new THREE.Vector3(60, 90, 40);
  }

  /** Configure lights, fog, exposure and the post-processing chain for a map. */
  applyProfile(p) {
    this.profile = p;
    this.renderer.toneMappingExposure = p.exposure;
    this.hemiLight.color.set(p.hemi[0]);
    this.hemiLight.groundColor.set(p.hemi[1]);
    this.hemiLight.intensity = p.hemi[2];
    this.dirLight.color.set(p.sun.color);
    this.dirLight.intensity = p.sun.intensity;
    this.sunOffset.copy(p.sun.dir).normalize().multiplyScalar(120);
    this.scene.fog = new THREE.FogExp2(p.fog.color, p.fog.density);
    this.scene.background = new THREE.Color(p.fog.color);

    this.bloomPass.strength = p.bloom.strength;
    this.bloomPass.radius = p.bloom.radius;
    this.bloomPass.threshold = p.bloom.threshold;
    this.bokehPass.enabled = !!p.dof;
    this.speedBlurPass.enabled = !!p.motionBlur;
    this.heatPass.enabled = !!p.heat;
    this.lutPass.enabled = !!p.lut;
  }

  /** Per-frame effect uniforms: shadows follow the car, DoF focus, blur by speed, fog by elevation. */
  update(dt, carPos, speed) {
    const t = this.clock.getElapsed();
    this.dirLight.position.copy(carPos).add(this.sunOffset);
    this.dirLight.target.position.copy(carPos);
    if (this.bokehPass.enabled) this.bokehPass.uniforms.focus.value = this.camera.position.distanceTo(carPos);
    if (this.speedBlurPass.enabled) this.speedBlurPass.uniforms.uAmount.value = THREE.MathUtils.clamp((speed - 15) / 35, 0, 1) * 0.06;
    if (this.heatPass.enabled) this.heatPass.uniforms.uTime.value = t;
    const p = this.profile;
    if (p?.fog.elevationK && this.scene.fog) {
      // Thicker fog in low canyon floors, thinner on the ridges.
      const k = THREE.MathUtils.clamp(1 + (p.fog.refHeight - this.camera.position.y) * p.fog.elevationK, 0.35, 2.2);
      this.scene.fog.density = p.fog.density * k;
    }
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.composer.setSize(window.innerWidth, window.innerHeight);
  }

  render() {
    this.clock.update();
    this.composer.render();
  }
}
