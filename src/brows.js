// 3D eyebrow hairs. The strand paths come from the same strokes that were
// painted into the skin texture (tools/build_textures.py -> assets/brows.bin),
// so the geometry sits exactly on its own root shading. Every strand point is
// pinned to a skin triangle with barycentric weights, so brows follow identity
// and expression changes; hairs rise a little off the skin towards the tip and
// are drawn as camera-facing ribbons with a minimum on-screen width.
import * as THREE from 'three';
import { fetchBinary } from './model.js';

export async function loadBrowStrands(url) {
  const buf = await fetchBinary(url);
  const head = new Uint16Array(buf, 0, 2);
  const count = head[0], npts = head[1];
  const uv = new Uint16Array(buf, 4, count * npts * 2);
  const shade = new Uint8Array(buf, 4 + count * npts * 4, count);
  return { count, npts, uv, shade };
}

export class Brows {
  constructor(model, data, tint) {
    this.model = model;
    const { count, npts } = data;
    this.count = count;
    this.npts = npts;
    const skin = model.parts.skin;
    const geoUV = skin.geo.attributes.uv.array;
    const index = skin.geo.index.array;
    const faceGroup = skin.geo.groups[skin.groups.indexOf('face')];

    // Point UVs (atlas space) and their bounding box.
    const P = new Float32Array(count * npts * 2);
    let u0 = 1, u1 = 0, v0 = 1, v1 = 0;
    for (let i = 0; i < count * npts; i++) {
      const u = data.uv[2 * i] / 65535, v = data.uv[2 * i + 1] / 65535;
      P[2 * i] = u; P[2 * i + 1] = v;
      u0 = Math.min(u0, u); u1 = Math.max(u1, u); v0 = Math.min(v0, v); v1 = Math.max(v1, v);
    }
    const pad = 0.01;
    u0 -= pad; v0 -= pad; u1 += pad; v1 += pad;

    // Grid of candidate triangles over that box.
    const G = 96;
    const cells = Array.from({ length: G * G }, () => []);
    const cell = (u, v) => [Math.min(G - 1, Math.max(0, Math.floor((u - u0) / (u1 - u0) * G))), Math.min(G - 1, Math.max(0, Math.floor((v - v0) / (v1 - v0) * G)))];
    for (let t = faceGroup.start / 3; t < (faceGroup.start + faceGroup.count) / 3; t++) {
      const a = index[3 * t], b = index[3 * t + 1], c = index[3 * t + 2];
      const us = [geoUV[2 * a], geoUV[2 * b], geoUV[2 * c]], vs = [geoUV[2 * a + 1], geoUV[2 * b + 1], geoUV[2 * c + 1]];
      if (Math.max(...us) < u0 || Math.min(...us) > u1 || Math.max(...vs) < v0 || Math.min(...vs) > v1) continue;
      const [x0, y0] = cell(Math.min(...us), Math.min(...vs));
      const [x1, y1] = cell(Math.max(...us), Math.max(...vs));
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) cells[y * G + x].push(t);
    }

    // Pin every point: original vertex ids + barycentric weights.
    this.pinIdx = new Uint32Array(count * npts * 3);
    this.pinW = new Float32Array(count * npts * 3);
    const orig = skin.orig;
    for (let i = 0; i < count * npts; i++) {
      const u = P[2 * i], v = P[2 * i + 1];
      const [cx, cy] = cell(u, v);
      let best = null, bestErr = Infinity;
      for (const t of cells[cy * G + cx]) {
        const a = index[3 * t], b = index[3 * t + 1], c = index[3 * t + 2];
        const ax = geoUV[2 * a], ay = geoUV[2 * a + 1], bx = geoUV[2 * b], by = geoUV[2 * b + 1], qx = geoUV[2 * c], qy = geoUV[2 * c + 1];
        const den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy);
        if (Math.abs(den) < 1e-14) continue;
        const w0 = ((by - qy) * (u - qx) + (qx - bx) * (v - qy)) / den;
        const w1 = ((qy - ay) * (u - qx) + (ax - qx) * (v - qy)) / den;
        const w2 = 1 - w0 - w1;
        const err = Math.max(0, -w0) + Math.max(0, -w1) + Math.max(0, -w2);
        if (err < bestErr) { bestErr = err; best = [orig[a], orig[b], orig[c], w0, w1, w2]; }
        if (err === 0) break;
      }
      if (!best) best = [0, 0, 0, 1, 0, 0];
      const s = Math.max(best[3], 0) + Math.max(best[4], 0) + Math.max(best[5], 0) || 1;
      for (let k = 0; k < 3; k++) {
        this.pinIdx[3 * i + k] = best[k];
        this.pinW[3 * i + k] = Math.max(best[3 + k], 0) / s;
      }
    }

    // Ribbon geometry: 2 vertices per strand point.
    const nv = count * npts * 2;
    const geo = new THREE.BufferGeometry();
    this.position = new THREE.BufferAttribute(new Float32Array(nv * 3), 3).setUsage(THREE.DynamicDrawUsage);
    this.normal = new THREE.BufferAttribute(new Float32Array(nv * 3), 3).setUsage(THREE.DynamicDrawUsage);
    this.tangent = new THREE.BufferAttribute(new Float32Array(nv * 3), 3).setUsage(THREE.DynamicDrawUsage);
    const side = new Float32Array(nv), along = new Float32Array(nv), shade = new Float32Array(nv);
    const idx = [];
    for (let s = 0; s < count; s++) {
      for (let k = 0; k < npts; k++) {
        const base = (s * npts + k) * 2;
        side[base] = -1; side[base + 1] = 1;
        along[base] = along[base + 1] = k / (npts - 1);
        shade[base] = shade[base + 1] = data.shade[s] / 255;
        if (k < npts - 1) idx.push(base, base + 1, base + 2, base + 1, base + 3, base + 2);
      }
    }
    geo.setAttribute('position', this.position);
    geo.setAttribute('normal', this.normal);
    geo.setAttribute('aTangent', this.tangent);
    geo.setAttribute('aSide', new THREE.BufferAttribute(side, 1));
    geo.setAttribute('aAlong', new THREE.BufferAttribute(along, 1));
    geo.setAttribute('aShade', new THREE.BufferAttribute(shade, 1));
    geo.setIndex(idx);
    this.geometry = geo;

    this.uniforms = { uPixel: { value: 0.001 }, uRootW: { value: 0.00009 }, uTipW: { value: 0.00002 } };
    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.62, metalness: 0, side: THREE.DoubleSide, envMapIntensity: 0.35 });
    mat.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, this.uniforms, { uBrowTint: tint });
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', `#include <common>
          attribute vec3 aTangent; attribute float aSide, aAlong, aShade;
          uniform float uPixel, uRootW, uTipW;
          varying float vShade; varying float vAlong;`)
        .replace('#include <begin_vertex>', `#include <begin_vertex>
          {
            vec4 wp = modelMatrix * vec4(position, 1.0);
            vec3 tw = normalize(mat3(modelMatrix) * aTangent);
            vec3 sw = normalize(cross(tw, cameraPosition - wp.xyz));
            float z = -(viewMatrix * wp).z;
            float w = max(mix(uRootW, uTipW, aAlong), uPixel * z * 0.55);
            transformed += transpose(mat3(modelMatrix)) * sw * (0.5 * w * aSide);
            vShade = aShade; vAlong = aAlong;
          }`);
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', '#include <common>\nuniform vec3 uBrowTint; varying float vShade; varying float vAlong;')
        .replace('#include <map_fragment>', 'diffuseColor.rgb = uBrowTint * mix(0.75, 1.25, vShade) * mix(0.8, 1.15, vAlong);');
    };
    mat.customProgramCacheKey = () => 'brows-v1';
    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.frustumCulled = false;
    this.update();
  }

  setResolution(camera, heightPx) {
    this.uniforms.uPixel.value = 2 / (camera.projectionMatrix.elements[5] * heightPx);
  }

  update() {
    const { pos, nrm } = this.model;
    const { count, npts, pinIdx, pinW } = this;
    const P = this.position.array, N = this.normal.array, T = this.tangent.array;
    const pts = new Float32Array(npts * 3), nn = new Float32Array(npts * 3);
    for (let s = 0; s < count; s++) {
      for (let k = 0; k < npts; k++) {
        const i = s * npts + k;
        let x = 0, y = 0, z = 0, nx = 0, ny = 0, nz = 0;
        for (let j = 0; j < 3; j++) {
          const o = 3 * pinIdx[3 * i + j], w = pinW[3 * i + j];
          x += w * pos[o]; y += w * pos[o + 1]; z += w * pos[o + 2];
          nx += w * nrm[o]; ny += w * nrm[o + 1]; nz += w * nrm[o + 2];
        }
        const l = Math.hypot(nx, ny, nz) || 1;
        nx /= l; ny /= l; nz /= l;
        // rise off the skin towards the tip (hairs leave the follicle at an angle)
        const t = k / (npts - 1);
        const lift = 0.00012 + 0.0009 * Math.pow(t, 0.8);
        pts[3 * k] = x + nx * lift; pts[3 * k + 1] = y + ny * lift; pts[3 * k + 2] = z + nz * lift;
        nn[3 * k] = nx; nn[3 * k + 1] = ny; nn[3 * k + 2] = nz;
      }
      for (let k = 0; k < npts; k++) {
        const a = Math.max(0, k - 1), b = Math.min(npts - 1, k + 1);
        let tx = pts[3 * b] - pts[3 * a], ty = pts[3 * b + 1] - pts[3 * a + 1], tz = pts[3 * b + 2] - pts[3 * a + 2];
        const tl = Math.hypot(tx, ty, tz) || 1;
        tx /= tl; ty /= tl; tz /= tl;
        for (let e = 0; e < 2; e++) {
          const v = 3 * ((s * npts + k) * 2 + e);
          P[v] = pts[3 * k]; P[v + 1] = pts[3 * k + 1]; P[v + 2] = pts[3 * k + 2];
          N[v] = nn[3 * k]; N[v + 1] = nn[3 * k + 1]; N[v + 2] = nn[3 * k + 2];
          T[v] = tx; T[v + 1] = ty; T[v + 2] = tz;
        }
      }
    }
    this.position.needsUpdate = this.normal.needsUpdate = this.tangent.needsUpdate = true;
  }
}
