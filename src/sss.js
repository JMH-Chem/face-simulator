// Screen-space subsurface scattering pipeline + photographic post-processing.
//
// Pass 1  scene, skin outputs diffuse lighting (alpha 0 marks skin pixels)
// Pass 2  skin specular only (other opaque geometry depth-only)
// Pass 3  skin albedo only
// Blur    separable sum-of-Gaussians skin profile on irradiance (diffuse /
//         albedo), depth aware, then re-multiplied by albedo so pigment detail
//         (stubble, freckles, brows) stays sharp.
// Composite -> depth of field -> bloom -> tone mapping -> film finish.
import * as THREE from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

// d'Eon & Luebke 2007 skin diffusion profile: variance (mm^2) and RGB weight.
const PROFILE = [
  [0.0064, [0.233, 0.455, 0.649]],
  [0.0484, [0.100, 0.336, 0.344]],
  [0.187, [0.118, 0.198, 0.0]],
  [0.567, [0.113, 0.007, 0.007]],
  [1.99, [0.358, 0.004, 0.0]],
  [7.41, [0.078, 0.0, 0.0]],
];
const TAPS = 25;

function buildKernel(rangeMM = 6.0) {
  const offs = [];
  for (let i = 0; i < TAPS; i++) {
    const t = (i / (TAPS - 1)) * 2 - 1;
    offs.push(rangeMM * Math.sign(t) * t * t);
  }
  const k = [];
  const sum = [0, 0, 0];
  for (let i = 0; i < TAPS; i++) {
    const area = ((i < TAPS - 1 ? offs[i + 1] : offs[i]) - (i > 0 ? offs[i - 1] : offs[i])) / 2;
    const w = [0, 0, 0];
    for (const [v, rgb] of PROFILE) {
      const g = Math.exp(-(offs[i] * offs[i]) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
      for (let c = 0; c < 3; c++) w[c] += rgb[c] * g;
    }
    // Very narrow lobes are sub-pixel: give the centre tap their mass.
    for (let c = 0; c < 3; c++) w[c] *= Math.max(area, 1e-3);
    k.push(new THREE.Vector4(w[0], w[1], w[2], offs[i]));
  }
  const centre = (TAPS - 1) / 2;
  // Add the energy of lobes narrower than the centre tap spacing to the centre tap.
  for (let c = 0; c < 3; c++) {
    let total = 0;
    for (const [, rgb] of PROFILE) total += rgb[c];
    let cur = 0;
    for (const v of k) cur += v.getComponent(c);
    if (cur < total) k[centre].setComponent(c, k[centre].getComponent(c) + (total - cur));
    sum[c] = 0;
    for (const v of k) sum[c] += v.getComponent(c);
    for (const v of k) v.setComponent(c, v.getComponent(c) / sum[c]);
  }
  return k;
}

const BLUR_SHADER = {
  uniforms: {
    tColor: { value: null }, tAlbedo: { value: null }, tDepth: { value: null },
    uDir: { value: new THREE.Vector2(1, 0) }, uTexel: { value: new THREE.Vector2() },
    uProj: { value: 1 }, uNear: { value: 0.1 }, uFar: { value: 10 }, uWidth: { value: 1 },
    uFirst: { value: 1 }, uKernel: { value: buildKernel() },
  },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
  `,
  fragmentShader: /* glsl */`
    #include <packing>
    uniform sampler2D tColor, tAlbedo, tDepth;
    uniform vec2 uDir, uTexel;
    uniform float uProj, uNear, uFar, uWidth;
    uniform int uFirst;
    uniform vec4 uKernel[${TAPS}];
    varying vec2 vUv;
    float viewDepth(vec2 uv) { return -perspectiveDepthToViewZ(texture2D(tDepth, uv).x, uNear, uFar); }
    float maskAt(vec4 c) { return uFirst == 1 ? 1.0 - c.a : c.a; }
    vec3 irr(vec4 c, vec2 uv) {
      if (uFirst == 1) return c.rgb / max(texture2D(tAlbedo, uv).rgb, vec3(0.02));
      return c.rgb;
    }
    void main() {
      vec4 c = texture2D(tColor, vUv);
      float m = maskAt(c);
      if (m < 0.02) { gl_FragColor = vec4(uFirst == 1 ? c.rgb : vec3(0.0), 0.0); return; }
      float z = viewDepth(vUv);
      vec3 ic = irr(c, vUv);
      // kernel offsets are in mm -> pixels at this depth
      vec2 stp = uDir * uTexel * (uProj * 0.001 * uWidth / z);
      vec3 acc = vec3(0.0);
      for (int i = 0; i < ${TAPS}; i++) {
        vec2 uv = vUv + stp * uKernel[i].w;
        vec4 s = texture2D(tColor, uv);
        vec3 si = irr(s, uv);
        float zs = viewDepth(uv);
        float keep = step(0.5, maskAt(s)) * (1.0 - clamp(abs(zs - z) / (0.004 * uWidth), 0.0, 1.0));
        acc += uKernel[i].rgb * mix(ic, si, keep);
      }
      if (uFirst == 1) gl_FragColor = vec4(acc, m);
      else gl_FragColor = vec4(acc * texture2D(tAlbedo, vUv).rgb, m);
    }
  `,
};

const COMPOSITE_SHADER = {
  uniforms: { tMain: { value: null }, tBlur: { value: null }, tSpec: { value: null }, uSSS: { value: 1 } },
  vertexShader: BLUR_SHADER.vertexShader,
  fragmentShader: /* glsl */`
    uniform sampler2D tMain, tBlur, tSpec;
    uniform float uSSS;
    varying vec2 vUv;
    void main() {
      vec4 m = texture2D(tMain, vUv);
      vec4 b = texture2D(tBlur, vUv);
      vec3 spec = texture2D(tSpec, vUv).rgb;
      float mask = clamp(1.0 - m.a, 0.0, 1.0);
      vec3 diff = mix(m.rgb, b.rgb, uSSS * step(0.02, mask));
      gl_FragColor = vec4(m.rgb + mask * (diff - m.rgb) + mask * spec, 1.0);
    }
  `,
};

export class SkinPass extends Pass {
  constructor(scene, camera, skinUniform, { samples = 4 } = {}) {
    super();
    this.skinUniform = skinUniform;
    this.scene = scene;
    this.camera = camera;
    this.needsSwap = false;
    this.enabled = true;
    this.sssEnabled = true;
    this.sssWidth = 1.0;
    const opts = { type: THREE.HalfFloatType, samples };
    this.rtMain = new THREE.WebGLRenderTarget(1, 1, { ...opts, depthTexture: new THREE.DepthTexture(1, 1) });
    this.rtSpec = new THREE.WebGLRenderTarget(1, 1, { ...opts, depthBuffer: true });
    this.rtAlb = new THREE.WebGLRenderTarget(1, 1, { ...opts, depthBuffer: true });
    this.rtBlurA = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType });
    this.rtBlurB = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType });
    this.blur = new THREE.ShaderMaterial({ ...BLUR_SHADER, uniforms: THREE.UniformsUtils.clone(BLUR_SHADER.uniforms) });
    this.blur.uniforms.uKernel.value = buildKernel();
    this.comp = new THREE.ShaderMaterial({ ...COMPOSITE_SHADER, uniforms: THREE.UniformsUtils.clone(COMPOSITE_SHADER.uniforms) });
    this.quad = new FullScreenQuad(null);
    this.depthOnly = new THREE.MeshBasicMaterial({ colorWrite: false });
    this.black = new THREE.MeshBasicMaterial({ color: 0x000000 });
    this.swap = new Map();
  }

  get depthTexture() { return this.rtMain.depthTexture; }

  setSize(w, h) {
    for (const rt of [this.rtMain, this.rtSpec, this.rtAlb, this.rtBlurA, this.rtBlurB]) rt.setSize(w, h);
    this.blur.uniforms.uTexel.value.set(1 / w, 1 / h);
    this.height = h;
  }

  // Swap materials for the specular / albedo passes: skin keeps its own
  // material, opaque geometry becomes a depth-only occluder, films are hidden
  // and lashes render black in the specular pass so they still mask it.
  prepareAux(mode) {
    if (mode === 'restore') {
      for (const [o, s] of this.swap) { o.material = s.material; o.visible = s.visible; }
      this.swap.clear();
      return;
    }
    this.scene.traverse((o) => {
      if (!o.isMesh) return;
      if (!this.swap.has(o)) this.swap.set(o, { material: o.material, visible: o.visible });
      const orig = this.swap.get(o);
      o.material = orig.material;
      o.visible = orig.visible;
      if (!o.visible) return;
      if (o.userData.auxHide) { o.visible = false; return; }
      if (o.userData.lashBlack) {
        if (mode === 'spec') o.material = o.userData.lashBlack;
        else o.visible = false;
        return;
      }
      if (Array.isArray(orig.material)) {
        o.material = orig.material.map((m) => (m.userData.isSkin ? m : this.depthOnly));
      } else if (!orig.material.userData.isSkin) {
        o.material = this.depthOnly;
      }
    });
  }

  render(renderer, writeBuffer, readBuffer) {
    const { scene, camera } = this;
    const skinPass = this.skinUniform;
    const clearColor = renderer.getClearColor(new THREE.Color());
    const clearAlpha = renderer.getClearAlpha();
    renderer.shadowMap.autoUpdate = false;
    renderer.shadowMap.needsUpdate = true;

    // 1: lit scene, skin = diffuse only.
    skinPass.value = this.debugPass ?? (this.sssEnabled ? 1 : 0);
    renderer.setRenderTarget(this.rtMain);
    renderer.setClearColor(0x000000, 1);
    renderer.clear();
    renderer.render(scene, camera);

    if (this.sssEnabled && this.debugPass === undefined) {
      const bg = scene.background;
      scene.background = null;
      // 2: specular
      this.prepareAux('spec');
      skinPass.value = 2;
      renderer.setRenderTarget(this.rtSpec);
      renderer.setClearColor(0x000000, 0);
      renderer.clear();
      renderer.render(scene, camera);
      // 3: albedo
      this.prepareAux('albedo');
      skinPass.value = 3;
      renderer.setRenderTarget(this.rtAlb);
      renderer.clear();
      renderer.render(scene, camera);
      this.prepareAux('restore');
      scene.background = bg;
      skinPass.value = 0;

      // Blur H then V.
      const u = this.blur.uniforms;
      u.tDepth.value = this.rtMain.depthTexture;
      u.tAlbedo.value = this.rtAlb.texture;
      u.uNear.value = camera.near;
      u.uFar.value = camera.far;
      u.uProj.value = camera.projectionMatrix.elements[5] * this.height * 0.5;
      u.uWidth.value = this.sssWidth;
      this.quad.material = this.blur;
      u.tColor.value = this.rtMain.texture; u.uDir.value.set(1, 0); u.uFirst.value = 1;
      renderer.setRenderTarget(this.rtBlurA); this.quad.render(renderer);
      u.tColor.value = this.rtBlurA.texture; u.uDir.value.set(0, 1); u.uFirst.value = 0;
      renderer.setRenderTarget(this.rtBlurB); this.quad.render(renderer);
    }

    const c = this.comp.uniforms;
    c.tMain.value = this.rtMain.texture;
    c.tBlur.value = this.rtBlurB.texture;
    c.tSpec.value = this.rtSpec.texture;
    c.uSSS.value = this.sssEnabled && this.debugPass === undefined ? 1 : 0;
    this.quad.material = this.comp;
    renderer.setRenderTarget(this.renderToScreen ? null : readBuffer);
    this.quad.render(renderer);
    renderer.setClearColor(clearColor, clearAlpha);
  }
}

// Thin-lens depth of field (gather, golden-angle spiral).
const DOF_SHADER = {
  uniforms: {
    tDiffuse: { value: null }, tDepth: { value: null }, uNear: { value: 0.1 }, uFar: { value: 10 },
    uFocus: { value: 1 }, uFNumber: { value: 4 }, uFocalM: { value: 0.085 }, uFocalPx: { value: 1000 },
    uTexel: { value: new THREE.Vector2() }, uMaxCoC: { value: 24 }, uEnabled: { value: 1 },
  },
  vertexShader: /* glsl */`varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */`
    #include <packing>
    uniform sampler2D tDiffuse, tDepth;
    uniform float uNear, uFar, uFocus, uFNumber, uFocalM, uFocalPx, uMaxCoC, uEnabled;
    uniform vec2 uTexel;
    varying vec2 vUv;
    float coc(vec2 uv) {
      float z = -perspectiveDepthToViewZ(texture2D(tDepth, uv).x, uNear, uFar);
      float c = (uFocalM / uFNumber) * uFocalPx * abs(z - uFocus) / (z * max(uFocus - uFocalM, 1e-3));
      return min(c * 0.5, uMaxCoC);
    }
    void main() {
      vec4 base = texture2D(tDiffuse, vUv);
      if (uEnabled < 0.5) { gl_FragColor = base; return; }
      float r = coc(vUv);
      if (r < 0.6) { gl_FragColor = base; return; }
      vec3 acc = base.rgb; float wsum = 1.0;
      const int N = 56;
      for (int i = 1; i < N; i++) {
        float t = float(i) / float(N);
        float a = float(i) * 2.39996323;
        vec2 o = vec2(cos(a), sin(a)) * sqrt(t) * r;
        vec2 uv = vUv + o * uTexel;
        float rs = coc(uv);
        float w = smoothstep(length(o) - 1.5, length(o) + 0.5, rs);
        acc += texture2D(tDiffuse, uv).rgb * w; wsum += w;
      }
      gl_FragColor = vec4(acc / wsum, 1.0);
    }
  `,
};

// Display-referred film finish: vignette, faint lateral CA, grain.
const FINISH_SHADER = {
  uniforms: { tDiffuse: { value: null }, uTime: { value: 0 }, uGrain: { value: 0.035 }, uVignette: { value: 0.28 }, uCA: { value: 0.0004 }, uRes: { value: new THREE.Vector2(1, 1) } },
  vertexShader: DOF_SHADER.vertexShader,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform float uTime, uGrain, uVignette, uCA;
    uniform vec2 uRes;
    varying vec2 vUv;
    float hash(vec2 p) { p = fract(p * vec2(443.897, 441.423)); p += dot(p, p.yx + 19.19); return fract((p.x + p.y) * p.x); }
    void main() {
      vec2 d = vUv - 0.5;
      float r2 = dot(d, d);
      vec3 col;
      col.r = texture2D(tDiffuse, vUv - d * uCA * 4.0).r;
      col.g = texture2D(tDiffuse, vUv).g;
      col.b = texture2D(tDiffuse, vUv + d * uCA * 4.0).b;
      col *= 1.0 - uVignette * smoothstep(0.08, 0.6, r2);
      float lum = dot(col, vec3(0.299, 0.587, 0.114));
      float n = hash(vUv * uRes + fract(uTime) * 917.0) + hash(vUv * uRes * 1.37 - fract(uTime * 1.3) * 431.0) - 1.0;
      col += n * uGrain * (0.35 + 0.65 * sqrt(max(1.0 - lum, 0.0))) * (0.4 + lum);
      gl_FragColor = vec4(col, 1.0);
    }
  `,
};

export function createPipeline(renderer, scene, camera, skinUniform) {
  const size = renderer.getDrawingBufferSize(new THREE.Vector2());
  const target = new THREE.WebGLRenderTarget(size.x, size.y, { type: THREE.HalfFloatType });
  const composer = new EffectComposer(renderer, target);
  const skin = new SkinPass(scene, camera, skinUniform);
  const dof = new ShaderPass(DOF_SHADER);
  const bloom = new UnrealBloomPass(new THREE.Vector2(size.x, size.y), 0.12, 0.5, 1.4);
  const output = new OutputPass();
  const finish = new ShaderPass(FINISH_SHADER);
  composer.addPass(skin);
  composer.addPass(dof);
  composer.addPass(bloom);
  composer.addPass(output);
  composer.addPass(finish);
  dof.uniforms.tDepth.value = skin.depthTexture;

  const api = {
    composer, skin, dof, bloom, output, finish,
    setSize(w, h, pr) {
      composer.setPixelRatio(pr);
      composer.setSize(w, h);
      const W = Math.round(w * pr), H = Math.round(h * pr);
      dof.uniforms.uTexel.value.set(1 / W, 1 / H);
      finish.uniforms.uRes.value.set(W, H);
      api.H = H;
    },
    render(dt, focusDistance) {
      const u = dof.uniforms;
      u.uNear.value = camera.near; u.uFar.value = camera.far;
      u.uFocus.value = focusDistance;
      const fov = THREE.MathUtils.degToRad(camera.fov);
      u.uFocalPx.value = api.H / (2 * Math.tan(fov / 2));
      u.uFocalM.value = 0.012 / Math.tan(fov / 2); // 24 mm sensor height
      finish.uniforms.uTime.value += dt;
      composer.render(dt);
    },
  };
  return api;
}
