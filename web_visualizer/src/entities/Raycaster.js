import * as THREE from 'three';

/** Visualized 3D LIDAR sensor rays (laser lines + hit markers), coloured by proximity. */
export class VisualRaycaster {
  constructor(scene, rayCount = 11) {
    this.scene = scene;
    this.rayCount = rayCount;
    this.lines = [];
    this.group = new THREE.Group();
    const dotGeo = new THREE.SphereGeometry(0.16, 8, 6);
    for (let i = 0; i < rayCount; i++) {
      const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 0, -1)]);
      const material = new THREE.LineBasicMaterial({ color: 0x00ffcc, transparent: true, opacity: 0.8, toneMapped: false });
      const line = new THREE.Line(geometry, material);
      line.frustumCulled = false;
      const dot = new THREE.Mesh(dotGeo, new THREE.MeshBasicMaterial({ color: 0x00ffcc, toneMapped: false }));
      this.group.add(line, dot);
      this.lines.push({ line, dot });
    }
    scene.add(this.group);
  }

  set visible(v) {
    this.group.visible = v;
  }

  get visible() {
    return this.group.visible;
  }

  /** origin: THREE.Vector3; hitPoints: THREE.Vector3[]; rayDistances: number[] */
  update(origin, hitPoints, rayDistances, maxDistance = 50.0) {
    for (let i = 0; i < this.rayCount; i++) {
      const { line, dot } = this.lines[i];
      const end = hitPoints[i];
      const positions = line.geometry.attributes.position.array;
      positions[0] = origin.x; positions[1] = origin.y; positions[2] = origin.z;
      positions[3] = end.x; positions[4] = end.y; positions[5] = end.z;
      line.geometry.attributes.position.needsUpdate = true;
      dot.position.copy(end);

      // Red = obstacle close, Amber = warning, Cyan = clear. Boosted >1 so the bloom pass catches it.
      const normalizedDist = rayDistances[i] / maxDistance;
      const hex = normalizedDist < 0.2 ? 0xff0055 : normalizedDist < 0.5 ? 0xffaa00 : 0x00ffcc;
      line.material.color.setHex(hex).multiplyScalar(1.6);
      dot.material.color.setHex(hex).multiplyScalar(2.5);
      dot.visible = normalizedDist < 0.999;
    }
  }
}
