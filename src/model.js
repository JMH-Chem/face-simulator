// Face model: decodes the ICT-FaceKit derived data (see tools/build_model.py),
// blends identity + expression shapes on the CPU, recomputes welded normals and
// streams the result into one BufferGeometry per part.
import * as THREE from 'three';

// Hosts that only serve text can set window.FACE_B64 = true and ship each
// binary next to itself as "<name>.b64.txt".
export async function fetchBinary(url) {
  let buf;
  if (globalThis.FACE_B64) {
    const res = await fetch(url + '.b64.txt');
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    const bin = atob((await res.text()).trim());
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    buf = bytes.buffer;
  } else {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    buf = await res.arrayBuffer();
  }
  const head = new Uint8Array(buf, 0, 2);
  if (head[0] === 0x1f && head[1] === 0x8b) {
    // Still gzip'd (server did not apply Content-Encoding): inflate here.
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
    return await new Response(stream).arrayBuffer();
  }
  return buf;
}

const TYPED = { float32: Float32Array, int16: Int16Array, uint16: Uint16Array, uint8: Uint8Array };

// Skin atlas remap (must match tools/texlib.py): face tile -> left 2/3,
// back-of-head tile -> lower-right 1/3 x 1/2.
function atlasUV(u, v) {
  if (u < 1) return [u * (2 / 3), v];
  return [2 / 3 + (u - 1) / 3, v * 0.5];
}

export class FaceModel {
  static async load(base) {
    const [meta, bin, extra] = await Promise.all([
      fetch(base + 'face.json').then((r) => r.json()),
      fetchBinary(base + 'face.bin.gz'),
      fetchBinary(base + 'face_extra.bin'),
    ]);
    return new FaceModel(meta, bin, new Uint8Array(extra));
  }

  constructor(meta, bin, extra) {
    this.meta = meta;
    this.bin = bin;
    const n = (this.n = meta.vertexCount);
    this.neutral = this.arr(meta.neutral);
    this.extra = extra;

    // Identity modes (dense) and expressions (sparse), decoded to float metres.
    const Q = meta.quantum;
    this.identityModes = meta.identity.map((d) => this.decodeDeltas(d.data, n, Q));
    this.expressions = meta.expressions.map((e) => {
      const didx = this.arr(e.index);
      const idx = new Uint32Array(e.count);
      let acc = 0;
      for (let i = 0; i < e.count; i++) { acc += didx[i]; idx[i] = acc; }
      return { name: e.name, index: idx, delta: this.decodeDeltas(e.delta, e.count, Q) };
    });
    this.exprIndex = Object.fromEntries(this.expressions.map((e, i) => [e.name, i]));

    this.identityWeights = new Float32Array(this.identityModes.length);
    this.exprWeights = new Float32Array(this.expressions.length);
    this.base = new Float32Array(this.neutral);
    this.pos = new Float32Array(this.neutral);
    this.nrm = new Float32Array(3 * n);

    this.parts = {};
    const allTris = [];
    for (const [name, p] of Object.entries(meta.parts)) {
      const orig = this.arr(p.origIndex);
      const uvRaw = this.arr(p.uv);
      const index = this.arr(p.index);
      const count = p.vertexCount;
      const geo = new THREE.BufferGeometry();
      const position = new THREE.BufferAttribute(new Float32Array(3 * count), 3);
      const normal = new THREE.BufferAttribute(new Float32Array(3 * count), 3);
      position.setUsage(THREE.DynamicDrawUsage);
      normal.setUsage(THREE.DynamicDrawUsage);
      const uv = new Float32Array(2 * count);
      const aoThick = new Float32Array(2 * count);
      for (let i = 0; i < count; i++) {
        let u = uvRaw[2 * i], v = uvRaw[2 * i + 1];
        if (name === 'skin') [u, v] = atlasUV(u, v);
        else if (name === 'lashes') u -= 6;
        uv[2 * i] = u; uv[2 * i + 1] = v;
        const o = orig[i];
        aoThick[2 * i] = extra[2 * o] / 255;
        aoThick[2 * i + 1] = extra[2 * o + 1] / 255;
      }
      geo.setAttribute('position', position);
      geo.setAttribute('normal', normal);
      geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
      geo.setAttribute('aoThick', new THREE.BufferAttribute(aoThick, 2));
      geo.setIndex(new THREE.BufferAttribute(new Uint16Array(index), 1));
      p.groups.forEach((g, gi) => geo.addGroup(g.start, g.count, gi));
      const tris = new Uint32Array(index.length);
      for (let i = 0; i < index.length; i++) tris[i] = orig[index[i]];
      allTris.push(tris);
      this.parts[name] = { geo, orig, groups: p.groups.map((g) => g.name), tris };
    }
    this.tris = concatU32(allTris);

    this.eyes = { L: { center: new THREE.Vector3(), radius: 0 }, R: { center: new THREE.Vector3(), radius: 0 } };
    this.landmarks = meta.landmarks68;
    this.applyIdentity();
    this.update();
  }

  arr(d) {
    const T = TYPED[d.type];
    return new T(this.bin, d.offset, d.length);
  }

  decodeDeltas(d, count, Q) {
    const q = this.arr(d);
    const out = new Float32Array(3 * count);
    let x = 0, y = 0, z = 0;
    for (let i = 0; i < count; i++) {
      x += q[3 * i]; y += q[3 * i + 1]; z += q[3 * i + 2];
      out[3 * i] = x * Q; out[3 * i + 1] = y * Q; out[3 * i + 2] = z * Q;
    }
    return out;
  }

  setIdentity(weights) {
    this.identityWeights.fill(0);
    for (let i = 0; i < Math.min(weights.length, this.identityWeights.length); i++) this.identityWeights[i] = weights[i];
    this.applyIdentity();
    this.dirty = true;
  }

  applyIdentity() {
    const b = this.base;
    b.set(this.neutral);
    this.identityModes.forEach((mode, k) => {
      const w = this.identityWeights[k];
      if (Math.abs(w) < 1e-4) return;
      for (let i = 0; i < b.length; i++) b[i] += w * mode[i];
    });
    // Eye centres from a sphere fit on the (hidden) ICT sclera vertices.
    for (const side of ['L', 'R']) {
      const [a, z] = this.meta.ranges['sclera' + side];
      const fit = fitSphere(b, a, z);
      this.eyes[side].center.copy(fit.center);
      this.eyes[side].radius = fit.radius;
    }
    this.dirty = true;
  }

  setExpression(name, w) {
    const i = this.exprIndex[name];
    if (i === undefined) return;
    if (this.exprWeights[i] !== w) { this.exprWeights[i] = w; this.dirty = true; }
  }

  getExpression(name) { return this.exprWeights[this.exprIndex[name]] || 0; }

  update() {
    if (this.dirty === false) return false;
    const p = this.pos;
    p.set(this.base);
    this.expressions.forEach((e, k) => {
      const w = this.exprWeights[k];
      if (Math.abs(w) < 1e-4) return;
      const idx = e.index, d = e.delta;
      for (let i = 0; i < idx.length; i++) {
        const o = 3 * idx[i];
        p[o] += w * d[3 * i]; p[o + 1] += w * d[3 * i + 1]; p[o + 2] += w * d[3 * i + 2];
      }
    });
    computeNormals(p, this.tris, this.nrm);
    for (const part of Object.values(this.parts)) {
      const P = part.geo.attributes.position.array;
      const N = part.geo.attributes.normal.array;
      const orig = part.orig;
      for (let i = 0; i < orig.length; i++) {
        const o = 3 * orig[i];
        P[3 * i] = p[o]; P[3 * i + 1] = p[o + 1]; P[3 * i + 2] = p[o + 2];
        N[3 * i] = this.nrm[o]; N[3 * i + 1] = this.nrm[o + 1]; N[3 * i + 2] = this.nrm[o + 2];
      }
      part.geo.attributes.position.needsUpdate = true;
      part.geo.attributes.normal.needsUpdate = true;
      part.geo.computeBoundingSphere();
    }
    this.dirty = false;
    return true;
  }

  // Current position of an ICT vertex (after identity + expression).
  vertex(i, target = new THREE.Vector3()) {
    return target.set(this.pos[3 * i], this.pos[3 * i + 1], this.pos[3 * i + 2]);
  }
}

function concatU32(list) {
  const out = new Uint32Array(list.reduce((s, a) => s + a.length, 0));
  let o = 0;
  for (const a of list) { out.set(a, o); o += a.length; }
  return out;
}

function computeNormals(p, tris, out) {
  out.fill(0);
  for (let t = 0; t < tris.length; t += 3) {
    const a = 3 * tris[t], b = 3 * tris[t + 1], c = 3 * tris[t + 2];
    const e1x = p[b] - p[a], e1y = p[b + 1] - p[a + 1], e1z = p[b + 2] - p[a + 2];
    const e2x = p[c] - p[a], e2y = p[c + 1] - p[a + 1], e2z = p[c + 2] - p[a + 2];
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    out[a] += nx; out[a + 1] += ny; out[a + 2] += nz;
    out[b] += nx; out[b + 1] += ny; out[b + 2] += nz;
    out[c] += nx; out[c + 1] += ny; out[c + 2] += nz;
  }
  for (let i = 0; i < out.length; i += 3) {
    const l = Math.hypot(out[i], out[i + 1], out[i + 2]) || 1;
    out[i] /= l; out[i + 1] /= l; out[i + 2] /= l;
  }
}

// Algebraic least-squares sphere fit over vertices [a, z].
function fitSphere(p, a, z) {
  // Solve [2x 2y 2z 1] [cx cy cz k]^T = x^2+y^2+z^2 via normal equations.
  const A = new Float64Array(16), B = new Float64Array(4);
  for (let i = a; i <= z; i++) {
    const x = p[3 * i], y = p[3 * i + 1], w = p[3 * i + 2];
    const row = [2 * x, 2 * y, 2 * w, 1];
    const rhs = x * x + y * y + w * w;
    for (let r = 0; r < 4; r++) {
      B[r] += row[r] * rhs;
      for (let c = 0; c < 4; c++) A[4 * r + c] += row[r] * row[c];
    }
  }
  const s = solve4(A, B);
  const center = new THREE.Vector3(s[0], s[1], s[2]);
  return { center, radius: Math.sqrt(s[3] + center.lengthSq()) };
}

function solve4(A, b) {
  const M = [0, 1, 2, 3].map((r) => [A[4 * r], A[4 * r + 1], A[4 * r + 2], A[4 * r + 3], b[r]]);
  for (let c = 0; c < 4; c++) {
    let piv = c;
    for (let r = c + 1; r < 4; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = 0; r < 4; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k < 5; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, i) => row[4] / row[i]);
}
