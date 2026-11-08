// Percentage-closer soft shadows (PCSS) for the key light. A softbox casts a
// shadow that is crisp where the occluder touches the skin (nostril, lid
// crease) and widens with distance (nose shadow on the cheek). We patch the
// PCF path of three's shadow chunk; `shadow.radius` is reinterpreted as the
// light's apparent angular size in degrees.
import * as THREE from 'three';

export const SHADOW_DEPTH_RANGE = 2.8; // key light shadow camera far - near (m)
export const SHADOW_FRUSTUM = 0.37;    // shadow camera width (m)

const PCSS = /* glsl */`
float pcssIGN(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }

float pcssShadow(sampler2D sm, vec2 size, float angleDeg, vec4 sc) {
  float ang = radians(max(angleDeg, 0.3));
  float rot = pcssIGN(gl_FragCoord.xy) * 6.2831853;
  mat2 R = mat2(cos(rot), sin(rot), -sin(rot), cos(rot));
  float search = clamp(ang * 0.05 / ${SHADOW_FRUSTUM.toFixed(3)}, 2.0 / size.x, 0.07);
  float bsum = 0.0, bcnt = 0.0;
  for (int i = 0; i < 16; i++) {
    float r = sqrt((float(i) + 0.5) / 16.0);
    float a = float(i) * 2.39996323;
    vec2 o = R * vec2(cos(a), sin(a)) * r * search;
    float d = unpackRGBAToDepth(texture2D(sm, sc.xy + o));
    if (d < sc.z) { bsum += d; bcnt += 1.0; }
  }
  if (bcnt < 0.5) return 1.0;
  float zb = bsum / bcnt;
  float pen = (sc.z - zb) * ${SHADOW_DEPTH_RANGE.toFixed(3)} * ang / ${SHADOW_FRUSTUM.toFixed(3)};
  pen = clamp(pen, 1.25 / size.x, search);
  float s = 0.0;
  for (int i = 0; i < 32; i++) {
    float r = sqrt((float(i) + 0.5) / 32.0);
    float a = float(i) * 2.39996323;
    vec2 o = R * vec2(cos(a), sin(a)) * r * pen;
    s += texture2DCompare(sm, sc.xy + o, sc.z);
  }
  return s / 32.0;
}
`;

let patched = false;
export function installPCSS() {
  if (patched) return;
  patched = true;
  const src = THREE.ShaderChunk.shadowmap_pars_fragment;
  const fn = src.indexOf('float getShadow(');
  const pcfTag = '#if defined( SHADOWMAP_TYPE_PCF )';
  const pcf = fn < 0 ? -1 : src.indexOf(pcfTag, fn);
  if (fn < 0 || pcf < 0 || !src.includes('texture2DCompare')) {
    console.warn('PCSS: shadow chunk layout changed; keeping default PCF');
    return;
  }
  THREE.ShaderChunk.shadowmap_pars_fragment = src.slice(0, fn) + PCSS + '\n' + src.slice(fn, pcf)
    + pcfTag + '\n\t\t\tshadow = pcssShadow( shadowMap, shadowMapSize, shadowRadius, shadowCoord );\n\t\t#elif 0'
    + src.slice(pcf + pcfTag.length);
}
