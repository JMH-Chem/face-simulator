// Materials: physically based skin (dual-lobe specular, melanin/hemoglobin
// albedo, dynamic wrinkles, micro pores), refractive eyes, teeth/gums, lashes,
// tear film and eye occlusion. The skin shader exposes three outputs for the
// screen-space subsurface pipeline in sss.js (uSkinPass: 0 full, 1 diffuse,
// 2 specular, 3 albedo).
import * as THREE from 'three';

const loader = new THREE.TextureLoader();

export function loadTexture(url, { srgb = false, repeat = false, aniso = 8 } = {}) {
  return new Promise((resolve, reject) => {
    loader.load(url, (t) => {
      t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
      t.anisotropy = aniso;
      if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
      resolve(t);
    }, undefined, reject);
  });
}

export async function loadTextures(base) {
  const names = {
    color: 'skin_color.jpg', normal: 'skin_normal.jpg', detail: 'skin_detail.jpg', hair: 'skin_hair.jpg',
    wrinkles: 'wrinkles.jpg', pores: 'pores.png', iris: 'iris.png', sclera: 'sclera.jpg', lashes: 'lashes.png',
  };
  const out = {};
  await Promise.all(Object.entries(names).map(async ([k, f]) => {
    out[k] = await loadTexture(base + f, { repeat: k === 'pores' });
  }));
  out.wrinkles.generateMipmaps = true;
  return out;
}

// Shared uniforms the UI / animator can drive directly.
export const skinParams = {
  uSkinPass: { value: 0 },
  uMelanin: { value: 0.24 },
  uHemoglobin: { value: 0.46 },
  uFlush: { value: 0.0 },
  uBeard: { value: 0.5 },
  uScalp: { value: 1.0 },
  uHairTint: { value: new THREE.Color(0.05, 0.035, 0.025) },
  uBrowTint: { value: new THREE.Color(0.07, 0.045, 0.03) },
  uAge: { value: 0.35 },
  uWrinkleW: { value: new THREE.Vector3(0, 0, 0) },
  uPoreStrength: { value: 1.0 },
  uPoreScale: { value: 26.0 },
  uRoughness: { value: 1.0 },
  uOil: { value: 0.7 },
  uMouthOpen: { value: 0.0 },
  uTranslucency: { value: 1.0 },
};

const SKIN_PARS_FRAG = /* glsl */`
uniform int uSkinPass;
uniform sampler2D tSkinColor, tSkinNormal, tSkinDetail, tSkinHair, tWrinkles, tPores;
uniform float uMelanin, uHemoglobin, uFlush, uBeard, uScalp, uAge, uPoreStrength, uPoreScale;
uniform float uRoughness, uOil, uMouthOpen, uTranslucency;
uniform vec3 uHairTint, uBrowTint, uWrinkleW;
varying vec2 vSkinUv;
varying vec2 vAOT;
varying vec3 vObjPos;

// Beer-Lambert style two-chromophore skin model (linear RGB).
vec3 skinAlbedo(float mel, float hb) {
  const vec3 dermis = vec3(0.92, 0.84, 0.70);
  const vec3 sigMel = vec3(1.66, 2.03, 2.48);
  const vec3 sigHb  = vec3(0.06, 1.00, 0.72);
  return dermis * exp(-(mel * sigMel + hb * sigHb));
}

vec3 unpackXY(vec2 t) {
  vec2 xy = t * 2.0 - 1.0;
  return vec3(xy, sqrt(max(1.0 - dot(xy, xy), 0.0)));
}

// Whiteout blend of two tangent-space normals.
vec3 blendN(vec3 a, vec3 b) { return normalize(vec3(a.xy + b.xy, a.z * b.z)); }

// Lagarde & de Rousiers specular occlusion from ambient occlusion.
float skinSpecOcc(float dotNV, float ao, float rough) {
  return clamp(pow(dotNV + ao, exp2(-16.0 * rough - 1.0)) - 1.0 + ao, 0.0, 1.0);
}
`;

export function createSkinMaterial(tex) {
  const m = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    roughness: 0.5,
    metalness: 0,
    ior: 1.4,
    specularIntensity: 1.0,
    clearcoat: 0.35,
    clearcoatRoughness: 0.36,
    sheen: 0.25,
    sheenRoughness: 0.6,
    sheenColor: new THREE.Color(0.35, 0.3, 0.28),
    normalMap: tex.normal, // enables the tangent frame; sampled manually below
  });
  m.userData.isSkin = true;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, skinParams, {
      tSkinColor: { value: tex.color }, tSkinNormal: { value: tex.normal }, tSkinDetail: { value: tex.detail },
      tSkinHair: { value: tex.hair }, tWrinkles: { value: tex.wrinkles }, tPores: { value: tex.pores },
    });
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nattribute vec2 aoThick;\nvarying vec2 vSkinUv;\nvarying vec2 vAOT;\nvarying vec3 vObjPos;')
      .replace('#include <uv_vertex>', '#include <uv_vertex>\nvSkinUv = uv;\nvAOT = aoThick;\nvObjPos = position;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\n' + SKIN_PARS_FRAG)
      .replace('#include <map_fragment>', /* glsl */`
        vec4 sc = texture2D(tSkinColor, vSkinUv);
        vec4 sh = texture2D(tSkinHair, vSkinUv);
        float mel = sc.r * 4.0 * uMelanin;
        float hb = sc.g * 2.5 * uHemoglobin * (1.0 + uFlush * (0.6 + 0.8 * smoothstep(0.25, 0.6, sc.g)));
        vec3 albedo = skinAlbedo(mel, hb);
        // Brows: tint the pigment towards the chosen hair colour.
        albedo = mix(albedo, uBrowTint * 1.4, clamp(sh.b * 0.6, 0.0, 1.0));
        // Short hair: shafts cover part of the skin (beard ~ shadow, buzz cut ~ dense).
        float stub = clamp(sh.r * uBeard * 0.9 + sh.g * uScalp * 1.45, 0.0, 0.92);
        albedo = mix(albedo, uHairTint, stub);
        diffuseColor.rgb *= albedo;
        float skinCavity;
      `)
      .replace('#include <roughnessmap_fragment>', /* glsl */`
        vec4 sd = texture2D(tSkinDetail, vSkinUv);
        // + pore-scale roughness that the mip chain averages away at distance
        float roughnessFactor = clamp(sd.r * uRoughness + 0.06 * sd.g * uPoreStrength + 0.1 * sh.g * uScalp + 0.06 * sh.r * uBeard, 0.08, 1.0);
        float skinOil = sc.b;
      `)
      .replace('#include <normal_fragment_maps>', /* glsl */`
        vec4 sn = texture2D(tSkinNormal, vSkinUv);
        skinCavity = sn.b;
        vec3 macroN = unpackXY(sn.xy);
        macroN.xy *= 0.6 + uAge;
        // Dynamic wrinkles: height maps -> gradient in UV space.
        vec2 wsz = vec2(textureSize(tWrinkles, 0));
        vec2 du = vec2(1.0 / wsz.x, 0.0), dv = vec2(0.0, 1.0 / wsz.y);
        vec3 w0 = texture2D(tWrinkles, vSkinUv).rgb;
        vec3 wx = texture2D(tWrinkles, vSkinUv + du).rgb;
        vec3 wy = texture2D(tWrinkles, vSkinUv + dv).rgb;
        vec3 ww = uWrinkleW + uAge * vec3(0.18, 0.12, 0.08);
        float h0 = dot(w0, ww), hx = dot(wx, ww), hy = dot(wy, ww);
        vec3 wrN = normalize(vec3(-(hx - h0) * 2.2, -(hy - h0) * 2.2, 1.0));
        skinCavity *= 1.0 - 0.45 * clamp(h0, 0.0, 1.0);
        // Micro relief: two scales of the tileable pore map, blended per region.
        vec2 puv = vSkinUv * vec2(1.5, 1.0) * uPoreScale;
        vec4 p1 = texture2D(tPores, puv);
        vec4 p2 = texture2D(tPores, puv * 0.55 + 0.37);
        vec4 pm = mix(p1, p2, sd.b);
        vec3 poreN = unpackXY(pm.xy);
        float poreAmt = sd.g * uPoreStrength;
        poreN.xy *= poreAmt * (0.75 + 0.35 * sd.b);
        skinCavity *= mix(1.0, pm.b, poreAmt * 0.8);
        vec3 mapN = blendN(blendN(macroN, wrN), poreN);
        normal = normalize(tbn * mapN);
        vec3 skinCoatN = normalize(tbn * blendN(macroN, wrN));
      `)
      .replace('#include <clearcoat_normal_fragment_begin>', /* glsl */`
        #ifdef USE_CLEARCOAT
          vec3 clearcoatNormal = skinCoatN;
        #endif
      `)
      .replace('#include <lights_physical_fragment>', /* glsl */`
        #include <lights_physical_fragment>
        material.clearcoat *= clamp(skinOil * uOil * 1.2, 0.0, 1.0) * (1.0 - 0.7 * stub);
        material.sheenColor *= 1.0 - stub;
      `)
      .replace('#include <lights_fragment_end>', /* glsl */`
        #include <lights_fragment_end>
        // Transmission through thin parts (ears, nostrils) for back light.
        #if NUM_DIR_LIGHTS > 0
        {
          float thick = max(vAOT.y * 50.0, 0.4); // mm
          vec3 tr = exp(-thick * vec3(0.42, 0.95, 1.45));
          // Light must still reach the far side: shadow lookup biased past the
          // thin part itself so only thicker occluders (the head) block it.
          float tShadow = 1.0;
          #if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
            tShadow = getShadow( directionalShadowMap[ 0 ], directionalLightShadows[ 0 ].shadowMapSize, 1.0,
              directionalLightShadows[ 0 ].shadowBias - 0.0035, directionalLightShadows[ 0 ].shadowRadius, vDirectionalShadowCoord[ 0 ] );
          #endif
          float back = clamp(dot(-geometryNormal, directionalLights[0].direction) + 0.2, 0.0, 1.0);
          reflectedLight.directDiffuse += tr * back * tShadow * directionalLights[0].color * diffuseColor.rgb * uTranslucency * 0.4;
        }
        #endif
      `)
      .replace('#include <aomap_fragment>', /* glsl */`
        {
          float mouthLift = uMouthOpen * smoothstep(0.07, 0.035, length((vObjPos.xy - vec2(0.0, -0.033)) * vec2(0.8, 1.4)));
          float ambientOcclusion = mix(vAOT.x, 1.0, mouthLift * 0.7) * skinCavity;
          reflectedLight.indirectDiffuse *= ambientOcclusion;
          #if defined( USE_CLEARCOAT )
            clearcoatSpecularIndirect *= ambientOcclusion;
          #endif
          #if defined( USE_SHEEN )
            sheenSpecularIndirect *= ambientOcclusion;
          #endif
          float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
          reflectedLight.indirectSpecular *= skinSpecOcc( dotNV, ambientOcclusion, material.roughness );
          reflectedLight.directSpecular *= mix(1.0, skinCavity, 0.6);
        }
      `)
      .replace('#include <opaque_fragment>', /* glsl */`
        vec3 skinDiffuseOut = totalDiffuse;
        #ifdef USE_CLEARCOAT
          skinDiffuseOut *= 1.0 - material.clearcoat * Fcc;
        #endif
        #ifdef USE_SHEEN
          skinDiffuseOut *= sheenEnergyComp;
        #endif
        if (uSkinPass == 1) {
          gl_FragColor = vec4(skinDiffuseOut, 0.0);
        } else if (uSkinPass == 2) {
          gl_FragColor = vec4(max(outgoingLight - skinDiffuseOut, 0.0), 1.0);
        } else if (uSkinPass == 3) {
          gl_FragColor = vec4(diffuseColor.rgb, 1.0);
        } else {
          gl_FragColor = vec4(outgoingLight, 1.0);
        }
      `);
  };
  m.customProgramCacheKey = () => 'skin-v1';
  return m;
}

// Inner mouth socket: dark, wet, falls off with depth into the mouth.
export function createMouthSocketMaterial() {
  const m = new THREE.MeshPhysicalMaterial({ color: new THREE.Color(0.32, 0.07, 0.07), roughness: 0.28, clearcoat: 0.6, clearcoatRoughness: 0.15 });
  addDepthDarkening(m, 0.105, 0.075, 0.05);
  return m;
}

export function createGumsTeethMaterials() {
  const gums = new THREE.MeshPhysicalMaterial({ color: new THREE.Color(0.55, 0.16, 0.16), roughness: 0.3, clearcoat: 0.7, clearcoatRoughness: 0.12 });
  addDepthDarkening(gums, 0.106, 0.085, 0.03);
  const teeth = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(0.58, 0.54, 0.45), roughness: 0.3, clearcoat: 0.35, clearcoatRoughness: 0.12,
    sheen: 0.2, sheenColor: new THREE.Color(0.6, 0.6, 0.65),
  });
  addDepthDarkening(teeth, 0.108, 0.085, 0.05);
  return { gums, teeth };
}

function addDepthDarkening(m, zFront, zBack, floor) {
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uMouthOpen = skinParams.uMouthOpen;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vObjPos;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvObjPos = position;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vObjPos;\nuniform float uMouthOpen;')
      .replace('#include <aomap_fragment>', `
        float depthAO = mix(${floor.toFixed(3)}, 1.0, smoothstep(${zBack.toFixed(4)}, ${zFront.toFixed(4)}, vObjPos.z));
        depthAO *= mix(0.45, 1.0, clamp(uMouthOpen * 1.5, 0.0, 1.0));
        reflectedLight.indirectDiffuse *= depthAO;
        reflectedLight.indirectSpecular *= depthAO * depthAO;
        reflectedLight.directDiffuse *= mix(depthAO, 1.0, 0.2);
        reflectedLight.directSpecular *= depthAO;
        #ifdef USE_CLEARCOAT
          clearcoatSpecularIndirect *= depthAO * depthAO;
          clearcoatSpecularDirect *= depthAO;
        #endif
      `);
  };
  m.customProgramCacheKey = () => `mouth-${zFront}-${zBack}-${floor}`;
}

// ------------------------------------------------------------------ eyes
export const eyeParams = {
  uIrisOuter: { value: new THREE.Color(0.18, 0.11, 0.05) },
  uIrisInner: { value: new THREE.Color(0.30, 0.17, 0.06) },
  uPupil: { value: 0.32 },
  uScleraTint: { value: new THREE.Color(0.75, 0.71, 0.67) },
  uVeins: { value: 0.45 },
};

export const IRIS_PRESETS = {
  'dark brown': [[0.075, 0.042, 0.022], [0.16, 0.085, 0.035]],
  brown: [[0.16, 0.09, 0.04], [0.30, 0.16, 0.06]],
  hazel: [[0.22, 0.20, 0.09], [0.36, 0.20, 0.07]],
  green: [[0.17, 0.26, 0.15], [0.33, 0.25, 0.08]],
  blue: [[0.20, 0.33, 0.48], [0.30, 0.34, 0.33]],
  gray: [[0.30, 0.34, 0.37], [0.34, 0.31, 0.26]],
};

// Eyeball geometry in unit-radius local space, +Z = gaze, with corneal bulge.
export const EYE = { irisR: 0.37, corneaApex: 1.075, irisZ: 0.905 };

export function createEyeGeometry() {
  const g = new THREE.SphereGeometry(1, 96, 72);
  g.rotateX(Math.PI / 2); // pole -> +Z
  const limbZ = Math.sqrt(1 - EYE.irisR * EYE.irisR);
  // cornea sphere through the limbus circle and the apex
  const zc = (EYE.corneaApex ** 2 - EYE.irisR ** 2 - limbZ ** 2) / (2 * (EYE.corneaApex - limbZ));
  const Rc = EYE.corneaApex - zc;
  EYE.corneaZc = zc;
  EYE.corneaR = Rc;
  const p = g.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i).normalize();
    if (v.z > limbZ - 0.02) {
      const b = v.z * zc;
      const t = b + Math.sqrt(Math.max(b * b - zc * zc + Rc * Rc, 0));
      const blend = THREE.MathUtils.smoothstep(v.z, limbZ - 0.02, limbZ + 0.02);
      v.multiplyScalar(THREE.MathUtils.lerp(1, Math.max(t, 1), blend));
    }
    p.setXYZ(i, v.x, v.y, v.z);
  }
  g.computeVertexNormals();
  return g;
}

export function createEyeMaterial(tex) {
  const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 0.1, ior: 1.376, specularIntensity: 1.0, clearcoat: 1.0, clearcoatRoughness: 0.02, envMapIntensity: 1.6 });
  const camLocal = { value: new THREE.Vector3() };
  m.userData.camLocal = camLocal;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, eyeParams, {
      tIris: { value: tex.iris }, tSclera: { value: tex.sclera }, uCamLocal: camLocal,
      uIrisR: { value: EYE.irisR }, uIrisZ: { value: EYE.irisZ }, uCorneaZc: { value: EYE.corneaZc },
    });
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vLocal;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvLocal = position;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', /* glsl */`#include <common>
        varying vec3 vLocal;
        uniform sampler2D tIris, tSclera;
        uniform vec3 uIrisOuter, uIrisInner, uScleraTint, uCamLocal;
        uniform float uPupil, uIrisR, uIrisZ, uCorneaZc, uVeins;
      `)
      .replace('#include <map_fragment>', /* glsl */`
        vec3 lp = vLocal;
        float rxy = length(lp.xy);
        float corneaW = smoothstep(uIrisR + 0.025, uIrisR - 0.01, rxy) * step(0.0, lp.z);
        // Refract the view ray at the cornea and hit the (flat) iris plane.
        vec3 vd = normalize(lp - uCamLocal);
        vec3 cn = normalize(lp - vec3(0.0, 0.0, uCorneaZc));
        vec3 rd = refract(vd, cn, 1.0 / 1.376);
        float s = (uIrisZ - lp.z) / min(rd.z, -1e-3);
        vec3 hit = lp + rd * max(s, 0.0);
        vec2 iuv = hit.xy / uIrisR;
        float ir = length(iuv);
        float pr = uPupil;
        float texR = mix(0.17, 1.0, clamp((ir - pr) / (1.0 - pr), 0.0, 1.0));
        vec2 idir = ir > 1e-4 ? iuv / ir : vec2(0.0);
        vec4 it = texture2D(tIris, 0.5 + 0.5 * idir * texR);
        vec3 irisCol = mix(uIrisOuter, uIrisInner, it.g) * (0.18 + 1.25 * it.r);
        irisCol *= mix(0.55, 1.0, smoothstep(1.02, 0.86, ir)); // limbal ring
        float pupil = smoothstep(pr + 0.012, pr - 0.012, ir);
        irisCol = mix(irisCol, vec3(0.004, 0.003, 0.003), pupil);
        // Sclera (equirect around the gaze axis, v = 1 - theta/pi).
        vec3 nl = normalize(lp);
        float theta = acos(clamp(nl.z, -1.0, 1.0));
        float phi = atan(nl.y, nl.x);
        vec4 st = texture2D(tSclera, vec2(phi / 6.2831853 + 0.5, 1.0 - theta / 3.14159265));
        vec3 scl = uScleraTint * mix(vec3(1.0), vec3(1.0, 0.92, 0.78), st.g * 0.45);
        scl = mix(scl, vec3(0.55, 0.10, 0.08), clamp(st.r * uVeins, 0.0, 1.0));
        scl *= mix(0.8, 1.0, smoothstep(uIrisR, uIrisR + 0.12, rxy));
        float irisMix = corneaW * smoothstep(1.04, 0.97, ir);
        diffuseColor.rgb *= mix(scl, irisCol, irisMix);
      `)
      .replace('#include <roughnessmap_fragment>', /* glsl */`
        float roughnessFactor = mix(0.16, 0.03, corneaW);
      `)
      .replace('#include <opaque_fragment>', 'gl_FragColor = vec4(outgoingLight, 1.0);');
  };
  m.customProgramCacheKey = () => 'eye-v1';
  return m;
}

// Tear film along the lid margin: additive specular only.
export function createLacrimalMaterial() {
  return new THREE.MeshPhysicalMaterial({
    color: 0x000000, roughness: 0.06, metalness: 0, ior: 1.336, specularIntensity: 1.0,
    transparent: true, depthWrite: false, polygonOffset: true,
    blending: THREE.CustomBlending, blendEquation: THREE.AddEquation,
    blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
    blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor,
    polygonOffsetFactor: -1, polygonOffsetUnits: -1,
  });
}

// Soft occlusion / shadow over the eyeball near the lids (multiplicative).
export function createEyeOcclusionMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: { uStrength: { value: 1.0 } },
    vertexShader: /* glsl */`
      varying vec2 vUv;
      void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
    `,
    fragmentShader: /* glsl */`
      varying vec2 vUv;
      uniform float uStrength;
      void main() {
        float e = abs(vUv.y - 0.5) * 2.0;           // 0 centre of the opening .. 1 lid edge
        float top = step(0.5, vUv.y);
        float occ = pow(smoothstep(0.15, 1.0, e), 1.5) * mix(0.35, 0.9, top);
        float corner = smoothstep(0.12, 0.0, vUv.x) + smoothstep(0.88, 1.0, vUv.x);
        occ = clamp(occ + 0.5 * corner, 0.0, 1.0) * uStrength;
        gl_FragColor = vec4(vec3(1.0 - occ), 1.0);
      }
    `,
    transparent: true, depthWrite: false,
    blending: THREE.CustomBlending,
    blendEquation: THREE.AddEquation,
    blendSrc: THREE.ZeroFactor, blendDst: THREE.SrcColorFactor,
    blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor,
  });
}

export function createLashMaterial(tex) {
  return new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(0.012, 0.009, 0.008), roughness: 0.4, sheen: 0.4, sheenColor: new THREE.Color(0.2, 0.18, 0.16),
    alphaMap: tex.lashes, transparent: true, depthWrite: false, side: THREE.DoubleSide,
  });
}

export function createDepthOnlyMaterial() {
  return new THREE.MeshBasicMaterial({ colorWrite: false });
}
