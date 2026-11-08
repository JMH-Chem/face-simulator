// Procedural photo-studio lighting. Each preset builds a small emissive scene
// (dome + softboxes) that is prefiltered into the image based lighting, plus a
// shadow casting key light aligned with the main source. The softboxes show
// up as rectangular catch-lights in the eyes, as in real portrait photography.
import * as THREE from 'three';

const deg = THREE.MathUtils.degToRad;

function dirFrom(az, el) {
  // az: 0 = in front of the face (+Z), positive towards the subject's left (+X)
  return new THREE.Vector3(Math.sin(deg(az)) * Math.cos(deg(el)), Math.sin(deg(el)), Math.cos(deg(az)) * Math.cos(deg(el)));
}

export const PRESETS = {
  studio: {
    label: 'Studio softbox',
    dome: { top: [0.05, 0.05, 0.055], horizon: [0.045, 0.045, 0.05], bottom: [0.02, 0.019, 0.018] },
    backdrop: { center: [0.032, 0.031, 0.033], edge: [0.006, 0.006, 0.007] },
    boxes: [
      { az: 42, el: 28, dist: 1.6, w: 1.0, h: 1.3, color: [1, 0.97, 0.93], power: 2.6 },
      { az: -55, el: 8, dist: 2.0, w: 1.8, h: 1.8, color: [0.95, 0.97, 1.0], power: 0.7 },
      { az: -150, el: 32, dist: 1.4, w: 0.3, h: 1.4, color: [1, 1, 1], power: 7 },
      { az: 150, el: 20, dist: 1.5, w: 0.3, h: 1.4, color: [1, 0.98, 0.95], power: 3 },
    ],
    key: { az: 42, el: 28, intensity: 4.2, color: [1, 0.97, 0.93], softness: 22 },
    exposure: 1.0,
  },
  window: {
    label: 'Window light',
    dome: { top: [0.06, 0.055, 0.05], horizon: [0.05, 0.045, 0.04], bottom: [0.025, 0.02, 0.016] },
    backdrop: { center: [0.035, 0.032, 0.029], edge: [0.008, 0.007, 0.006] },
    boxes: [
      { az: 78, el: 12, dist: 1.8, w: 1.6, h: 2.2, color: [0.92, 0.96, 1.05], power: 2.0 },
      { az: -100, el: 5, dist: 2.5, w: 2.5, h: 2.0, color: [1.0, 0.9, 0.8], power: 0.25 },
    ],
    key: { az: 78, el: 12, intensity: 4.6, color: [0.94, 0.97, 1.03], softness: 32 },
    exposure: 1.05,
  },
  overcast: {
    label: 'Overcast outdoor',
    dome: { top: [1.15, 1.2, 1.3], horizon: [0.75, 0.78, 0.82], bottom: [0.1, 0.095, 0.085] },
    backdrop: { center: [0.36, 0.38, 0.36], edge: [0.2, 0.22, 0.2] },
    boxes: [],
    key: { az: 20, el: 70, intensity: 0.5, color: [0.95, 0.97, 1.0], softness: 45 },
    exposure: 0.95,
  },
  dramatic: {
    label: 'Dramatic side',
    dome: { top: [0.004, 0.004, 0.005], horizon: [0.004, 0.004, 0.005], bottom: [0.002, 0.002, 0.002] },
    backdrop: { center: [0.012, 0.012, 0.014], edge: [0.002, 0.002, 0.003] },
    boxes: [
      { az: 88, el: 18, dist: 1.4, w: 0.5, h: 0.7, color: [1, 0.95, 0.88], power: 8 },
      { az: -140, el: 25, dist: 1.3, w: 0.25, h: 1.2, color: [0.85, 0.92, 1.0], power: 9 },
    ],
    key: { az: 88, el: 18, intensity: 6.0, color: [1, 0.95, 0.88], softness: 10 },
    exposure: 1.1,
  },
  golden: {
    label: 'Golden hour',
    dome: { top: [0.25, 0.38, 0.7], horizon: [0.75, 0.6, 0.45], bottom: [0.12, 0.08, 0.05] },
    backdrop: { center: [0.25, 0.2, 0.17], edge: [0.12, 0.1, 0.09] },
    boxes: [
      { az: 62, el: 9, dist: 3.0, w: 0.25, h: 0.25, color: [1.0, 0.62, 0.32], power: 160, round: true },
    ],
    key: { az: 62, el: 9, intensity: 6.0, color: [1.0, 0.66, 0.38], softness: 1.5 },
    exposure: 0.9,
  },
};

const domeMaterial = (d) => new THREE.ShaderMaterial({
  side: THREE.BackSide,
  depthWrite: false,
  uniforms: { top: { value: new THREE.Color(...d.top) }, horizon: { value: new THREE.Color(...d.horizon) }, bottom: { value: new THREE.Color(...d.bottom) } },
  vertexShader: 'varying vec3 vDir; void main(){ vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
  fragmentShader: `uniform vec3 top, horizon, bottom; varying vec3 vDir;
    void main(){ float y = vDir.y; vec3 c = y > 0.0 ? mix(horizon, top, pow(y, 0.6)) : mix(horizon, bottom, pow(-y, 0.4));
    gl_FragColor = vec4(c, 1.0); }`,
});

function softboxMaterial(color, power, round) {
  return new THREE.ShaderMaterial({
    side: THREE.DoubleSide,
    uniforms: { col: { value: new THREE.Color(...color).multiplyScalar(power) }, round: { value: round ? 1 : 0 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
    fragmentShader: `uniform vec3 col; uniform float round; varying vec2 vUv;
      void main(){ vec2 d = abs(vUv - 0.5) * 2.0;
        float e = round > 0.5 ? smoothstep(1.0, 0.6, length(d)) : (1.0 - smoothstep(0.82, 1.0, max(d.x, d.y)));
        float diff = mix(0.82, 1.0, 1.0 - dot(d, d) * 0.5); // diffusion panel hot spot
        gl_FragColor = vec4(col * e * diff, 1.0); }`,
  });
}

export function backdropMaterial(b) {
  return new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: true,
    uniforms: { center: { value: new THREE.Color(...b.center) }, edge: { value: new THREE.Color(...b.edge) } },
    vertexShader: 'varying vec3 vDir; void main(){ vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
    fragmentShader: `uniform vec3 center, edge; varying vec3 vDir;
      void main(){
        // lit seamless paper behind the subject (-Z), falling off to the sides
        float spot = smoothstep(1.25, 0.0, length(vec2(vDir.x * 1.3, vDir.y + 0.05)) ) * step(vDir.z, 0.2);
        vec3 c = mix(edge, center, spot);
        gl_FragColor = vec4(c, 1.0);
      }`,
  });
}

export class StudioLighting {
  constructor(renderer, scene, head) {
    this.renderer = renderer;
    this.scene = scene;
    this.pmrem = new THREE.PMREMGenerator(renderer);
    this.key = new THREE.DirectionalLight(0xffffff, 2);
    this.key.castShadow = true;
    const s = this.key.shadow;
    s.mapSize.set(2048, 2048);
    s.camera.left = -0.185; s.camera.right = 0.185; s.camera.top = 0.2; s.camera.bottom = -0.2;
    s.camera.near = 0.2; s.camera.far = 3.0;
    s.bias = -0.00025;
    s.normalBias = 0.0008;
    s.radius = 22; // PCSS: apparent light size in degrees
    this.key.target = head;
    scene.add(this.key);
    scene.add(this.key.target);
    this.backdrop = new THREE.Mesh(new THREE.SphereGeometry(4, 64, 32), backdropMaterial(PRESETS.studio.backdrop));
    this.backdrop.userData.backdrop = true;
    scene.add(this.backdrop);
    this.rotation = 0;
    this.intensity = 1;
    this.envTarget = null;
  }

  apply(name) {
    const p = PRESETS[name];
    this.preset = name;
    // Environment scene for prefiltered IBL.
    const env = new THREE.Scene();
    env.add(new THREE.Mesh(new THREE.SphereGeometry(20, 64, 32), domeMaterial(p.dome)));
    const back = new THREE.Mesh(new THREE.SphereGeometry(4, 64, 32), backdropMaterial(p.backdrop));
    env.add(back);
    for (const b of p.boxes) {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(b.w, b.h), softboxMaterial(b.color, b.power, b.round));
      m.position.copy(dirFrom(b.az, b.el).multiplyScalar(b.dist));
      m.lookAt(0, 0, 0);
      env.add(m);
    }
    if (this.envTarget) this.envTarget.dispose();
    this.envTarget = this.pmrem.fromScene(env, 0.0, 0.05, 50);
    env.traverse((o) => { if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); } });
    this.scene.environment = this.envTarget.texture;
    this.backdrop.material.dispose();
    this.backdrop.material = backdropMaterial(p.backdrop);
    this.key.color.setRGB(...p.key.color);
    this.key.shadow.radius = p.key.softness;
    this.exposure = p.exposure;
    this.update();
  }

  update() {
    const p = PRESETS[this.preset];
    const rot = deg(this.rotation);
    this.scene.environmentRotation.set(0, rot, 0);
    this.backdrop.rotation.set(0, 0, 0);
    const d = dirFrom(p.key.az, p.key.el).applyAxisAngle(new THREE.Vector3(0, 1, 0), rot);
    this.key.position.copy(d.multiplyScalar(1.5));
    this.key.intensity = p.key.intensity * this.intensity;
    this.scene.environmentIntensity = this.intensity;
  }
}
