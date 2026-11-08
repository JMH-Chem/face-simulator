// Application: scene graph, camera, render loop and the public API the UI uses.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FaceModel } from './model.js';
import {
  loadTextures, createSkinMaterial, createMouthSocketMaterial, createGumsTeethMaterials, createEyeGeometry,
  createEyeMaterial, createLacrimalMaterial, createEyeOcclusionMaterial, createLashMaterial, skinParams, eyeParams,
  IRIS_PRESETS,
} from './materials.js';
import { createPipeline } from './sss.js';
import { StudioLighting } from './lighting.js';
import { Animator } from './animator.js';
import { installPCSS } from './shadows.js';
import { loadBrowStrands, Brows } from './brows.js';

installPCSS();

const SENSOR_H = 24; // mm, full frame

export class FaceApp {
  constructor(canvas, assetBase = './assets/') {
    this.canvas = canvas;
    this.assetBase = assetBase;
    this.renderScale = 1;
    this.focalLength = 85;
    this.fNumber = 4;
    this.followCursor = true;
    this.pointer = null;
    this.clock = new THREE.Clock();
    this.listeners = new Set();
  }

  async init(onProgress = () => {}) {
    const renderer = (this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: false, powerPreference: 'high-performance', preserveDrawingBuffer: true }));
    renderer.toneMapping = THREE.NeutralToneMapping;
    renderer.toneMappingExposure = 1;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap; // patched to PCSS in shadows.js

    const scene = (this.scene = new THREE.Scene());
    const camera = (this.camera = new THREE.PerspectiveCamera(this.fovFor(this.focalLength), 1, 0.05, 20));
    camera.position.set(0.0, 0.035, 1.05);

    onProgress('Loading face model…');
    const [model, tex, browData] = await Promise.all([
      FaceModel.load(this.assetBase),
      loadTextures(this.assetBase + 'tex/'),
      loadBrowStrands(this.assetBase + 'brows.bin'),
    ]);
    this.browData = browData;
    this.model = model;
    this.tex = tex;
    onProgress('Building materials…');

    const head = (this.head = new THREE.Group());
    scene.add(head);
    this.buildMeshes();

    this.lighting = new StudioLighting(renderer, scene, head);
    this.lighting.base = this.assetBase;
    await this.lighting.apply('loft');
    renderer.toneMappingExposure = this.lighting.exposure;

    this.controls = new OrbitControls(camera, this.canvas);
    this.controls.target.set(0, 0.018, 0.06);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 0.3;
    this.controls.maxDistance = 2.2;
    this.controls.minPolarAngle = 0.35;
    this.controls.maxPolarAngle = 2.4;
    this.controls.enablePan = true;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    this.pipeline = createPipeline(renderer, scene, camera, skinParams.uSkinPass);
    this.animator = new Animator(model);

    this.canvas.addEventListener('pointermove', (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.pointer = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    });
    this.canvas.addEventListener('pointerleave', () => { this.pointer = null; });
    this.resize();
    window.addEventListener('resize', () => this.resize());
    onProgress('Compiling shaders…');
    await renderer.compileAsync(scene, camera);
    this.loop = this.loop.bind(this);
    renderer.setAnimationLoop(this.loop);
  }

  fovFor(mm) { return THREE.MathUtils.radToDeg(2 * Math.atan(SENSOR_H / 2 / mm)); }

  buildMeshes() {
    const { model, tex, head } = this;
    const P = model.parts;
    const skinMat = (this.skinMaterial = createSkinMaterial(tex));
    const socket = createMouthSocketMaterial();
    const eyeSocket = new THREE.MeshStandardMaterial({ color: 0x0a0505, roughness: 0.6 });
    const skinMats = P.skin.groups.map((g) => ({ face: skinMat, head: skinMat, mouthSocket: socket, eyeSocket }[g]));
    const skin = (this.skinMesh = new THREE.Mesh(P.skin.geo, skinMats));
    skin.castShadow = skin.receiveShadow = true;
    skin.frustumCulled = false;
    head.add(skin);

    const { gums, teeth } = createGumsTeethMaterials();
    const mouth = new THREE.Mesh(P.mouth.geo, P.mouth.groups.map((g) => (g === 'teeth' ? teeth : gums)));
    mouth.castShadow = mouth.receiveShadow = true;
    mouth.frustumCulled = false;
    head.add(mouth);

    const lac = new THREE.Mesh(P.lacrimal.geo, createLacrimalMaterial());
    lac.userData.auxHide = true;
    lac.renderOrder = 2;
    lac.frustumCulled = false;
    head.add(lac);

    const occ = new THREE.Mesh(P.occlusion.geo, createEyeOcclusionMaterial());
    occ.userData.auxHide = true;
    occ.renderOrder = 3;
    occ.frustumCulled = false;
    head.add(occ);

    const lashMat = createLashMaterial(tex);
    const lashes = new THREE.Mesh(P.lashes.geo, lashMat);
    lashes.userData.lashBlack = new THREE.MeshBasicMaterial({ color: 0x000000, alphaMap: tex.lashes, transparent: true, depthWrite: false, side: THREE.DoubleSide });
    lashes.renderOrder = 4;
    lashes.frustumCulled = false;
    lashes.castShadow = true;
    lashes.customDepthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, alphaMap: tex.lashes, alphaTest: 0.35, side: THREE.DoubleSide });
    head.add(lashes);

    const eyeGeo = createEyeGeometry();
    this.eyes = {};
    for (const side of ['L', 'R']) {
      const pivot = new THREE.Group();
      const mat = createEyeMaterial(tex);
      const ball = new THREE.Mesh(eyeGeo, mat);
      ball.receiveShadow = true;
      ball.onBeforeRender = (r, s, cam) => {
        // camera position in the eyeball's unit-sphere space (for refraction)
        const inv = new THREE.Matrix4().copy(ball.matrixWorld).invert();
        mat.userData.camLocal.value.copy(cam.position).applyMatrix4(inv);
      };
      pivot.add(ball);
      head.add(pivot);
      this.eyes[side] = { pivot, ball };
    }
    this.syncEyes();

    this.brows = new Brows(model, this.browData, skinParams.uBrowTint);
    head.add(this.brows.mesh);
  }

  syncEyes() {
    for (const side of ['L', 'R']) {
      const e = this.model.eyes[side];
      const { pivot, ball } = this.eyes[side];
      pivot.position.copy(e.center);
      ball.scale.setScalar(e.radius * 0.985);
    }
  }

  resize() {
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    const cap = Math.min(w, h) < 700 ? 1.5 : 2; // phones: keep the SSS passes affordable
    const pr = Math.min(window.devicePixelRatio || 1, cap) * this.renderScale;
    this.renderer.setPixelRatio(pr);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.pipeline.setSize(w, h, pr);
    this.brows?.setResolution(this.camera, h * pr);
  }

  // ------------------------------------------------------------- per frame
  loop() {
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const { animator, model, camera, head } = this;
    this.controls.update();

    // Gaze target: the viewer (camera) or the cursor, in head-local space.
    head.updateMatrixWorld();
    const eyeMid = new THREE.Vector3().addVectors(model.eyes.L.center, model.eyes.R.center).multiplyScalar(0.5);
    const eyeMidW = head.localToWorld(eyeMid.clone());
    let target = camera.position.clone();
    if (this.followCursor && this.pointer) {
      const d = camera.position.distanceTo(eyeMidW);
      target = new THREE.Vector3(this.pointer.x, this.pointer.y, 0.5).unproject(camera).sub(camera.position).normalize().multiplyScalar(d).add(camera.position);
    }
    const local = head.worldToLocal(target.clone()).sub(eyeMid);
    const yaw = THREE.MathUtils.clamp(Math.atan2(local.x, local.z), -0.55, 0.55);
    const pitch = THREE.MathUtils.clamp(Math.atan2(local.y, Math.hypot(local.x, local.z)), -0.4, 0.4);
    animator.gazeYaw = yaw;
    animator.gazePitch = pitch;
    animator.update(dt);

    // Head: idle motion + a little head turn towards the gaze target.
    const follow = this.followCursor && this.pointer ? 0.25 : 0.0;
    head.rotation.set(animator.headRot.x - pitch * follow, animator.headRot.y + yaw * follow, animator.headRot.z);
    head.position.copy(animator.headPos);

    // Eyes converge on the target point.
    for (const side of ['L', 'R']) {
      const { pivot } = this.eyes[side];
      const c = model.eyes[side].center;
      const tl = head.worldToLocal(target.clone());
      const dir = tl.sub(c);
      const ey = THREE.MathUtils.clamp(Math.atan2(dir.x, dir.z), -0.6, 0.6) + animator.saccade.x;
      const ep = THREE.MathUtils.clamp(Math.atan2(dir.y, Math.hypot(dir.x, dir.z)), -0.45, 0.45) + animator.saccade.y;
      pivot.rotation.set(-ep, ey, 0, 'YXZ');
    }

    animator.wrinkles(skinParams.uWrinkleW.value);
    skinParams.uMouthOpen.value = model.getExpression('jawOpen');
    if (model.update()) this.brows.update();

    const focus = camera.position.distanceTo(head.localToWorld(eyeMid.clone()));
    this.pipeline.render(dt, focus);
    this.frames = (this.frames || 0) + 1;
    for (const fn of this.listeners) fn(dt);
  }

  // ----------------------------------------------------------------- API
  setIdentity(weights) {
    this.model.setIdentity(weights);
    this.syncEyes();
  }

  randomIdentity(variation = 1, seed = Math.random() * 1e9) {
    let s = seed >>> 0;
    const rand = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
    const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-9)) * Math.cos(2 * Math.PI * rand());
    const w = Array.from({ length: this.model.identityModes.length }, () => gauss() * variation);
    this.setIdentity(w);
    return w;
  }

  setFocalLength(mm) {
    // Keep the head the same size in frame: move the camera with the lens.
    const old = this.focalLength;
    this.focalLength = mm;
    this.camera.fov = this.fovFor(mm);
    this.camera.updateProjectionMatrix();
    const t = this.controls.target;
    const off = this.camera.position.clone().sub(t).multiplyScalar(mm / old);
    this.camera.position.copy(t).add(off);
  }

  setAperture(n) { this.fNumber = n; this.pipeline.dof.uniforms.uFNumber.value = n; this.pipeline.dof.uniforms.uEnabled.value = n < 22 ? 1 : 0; }

  setIris(name) {
    const [o, i] = IRIS_PRESETS[name];
    eyeParams.uIrisOuter.value.setRGB(...o);
    eyeParams.uIrisInner.value.setRGB(...i);
  }

  async setLighting(name) {
    await this.lighting.apply(name);
    this.renderer.toneMappingExposure = this.lighting.exposure * (this.exposureBias || 1);
  }

  setExposure(ev) {
    this.exposureBias = Math.pow(2, ev);
    this.renderer.toneMappingExposure = this.lighting.exposure * this.exposureBias;
  }

  setRenderScale(s) { this.renderScale = s; this.resize(); }

  resetView() {
    this.camera.position.set(0.0, 0.035, 1.05 * this.focalLength / 85);
    this.controls.target.set(0, 0.018, 0.06);
  }
}
