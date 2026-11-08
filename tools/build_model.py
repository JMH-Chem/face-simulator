"""Convert ICT-FaceKit (MIT) OBJ data into a compact binary for the web viewer.

Output:
  assets/face.json    - metadata (parts, groups, shapes, offsets)
  assets/face.bin.gz  - gzip'd little-endian blob referenced by face.json

Shape deltas are quantised to QUANTUM_M and stored as int16 differences along
the vertex order ("delta of deltas"), which gzip compresses ~4x better than raw
values. The viewer reverses this with a running prefix sum per component.

Usage:
  python tools/build_model.py --ict /path/to/ICT-FaceKit --out assets
"""
import argparse
import gzip
import json
import os

import numpy as np

from ict_obj import read_full, read_positions

CM_TO_M = 0.01
NUM_IDENTITY = 40          # leading PCA modes kept (rest are sub-millimetre)
SPARSE_EPS_CM = 2e-4       # deltas below this are dropped from expressions
QUANTUM_M = 2e-5           # 0.02 mm fixed quantisation step for all deltas

# Vertex ranges from the ICT-FaceKit README (inclusive).
RANGE_FACE = (0, 9408)
RANGE_HEADNECK = (9409, 11247)
RANGE_MOUTH_SOCKET = (11248, 13293)
RANGE_EYE_SOCKET = (13294, 14061)

# part name -> list of (ICT material, group name)
PARTS = {
    "skin": [("M_Face", "face"), ("M_BackHead", "head")],
    "mouth": [("M_GumsTongue", "gums"), ("M_Teeth", "teeth")],
    "lacrimal": [("M_LacrimalFluid", "lacrimal")],
    "occlusion": [("M_EyeOcclusion", "occlusion")],
    "lashes": [("M_EyeLashes", "lashes")],
}


def in_range(a, r):
    return (a >= r[0]) & (a <= r[1])


class Blob:
    def __init__(self):
        self.chunks = []
        self.size = 0

    def add(self, arr):
        arr = np.ascontiguousarray(arr)
        pad = (-self.size) % 4
        if pad:
            self.chunks.append(b"\0" * pad)
            self.size += pad
        off = self.size
        b = arr.tobytes()
        self.chunks.append(b)
        self.size += len(b)
        return {"offset": off, "length": int(arr.size), "type": arr.dtype.name}

    def bytes(self):
        return b"".join(self.chunks)


def triangulate(corners, V):
    """Split a quad along its shorter diagonal; pass triangles through."""
    if len(corners) == 3:
        return [corners]
    a, b, c, d = corners
    d1 = np.linalg.norm(V[a[0]] - V[c[0]])
    d2 = np.linalg.norm(V[b[0]] - V[d[0]])
    if d1 <= d2:
        return [[a, b, c], [a, c, d]]
    return [[a, b, d], [b, c, d]]


def build_part(name, spec, V, T, F, M, blob):
    groups_out = []
    key_to_gpu = {}
    orig, uvs, tris = [], [], []
    for mat, gname in spec:
        fidx = [k for k, m in enumerate(M) if m == mat]
        # Skin: split the BackHead material into head / mouth socket / eye socket
        buckets = {}
        for k in fidx:
            vids = np.array([c[0] for c in F[k]])
            g = gname
            if mat == "M_BackHead":
                if in_range(vids, RANGE_MOUTH_SOCKET).mean() > 0.5:
                    g = "mouthSocket"
                elif in_range(vids, RANGE_EYE_SOCKET).mean() > 0.5:
                    g = "eyeSocket"
            buckets.setdefault(g, []).append(k)
        for g, flist in buckets.items():
            start = len(tris)
            for k in flist:
                for tri in triangulate(F[k], V):
                    out = []
                    for vi, ti in tri:
                        key = (vi, ti)
                        if key not in key_to_gpu:
                            key_to_gpu[key] = len(orig)
                            orig.append(vi)
                            uvs.append(T[ti] if ti >= 0 else (0.0, 0.0))
                        out.append(key_to_gpu[key])
                    tris.append(out)
            groups_out.append({"name": g, "start": start * 3, "count": (len(tris) - start) * 3})
    orig = np.array(orig, dtype=np.uint16)
    uvs = np.array(uvs, dtype=np.float32)
    tris = np.array(tris, dtype=np.uint16)
    return {
        "vertexCount": int(len(orig)),
        "origIndex": blob.add(orig),
        "uv": blob.add(uvs.reshape(-1)),
        "index": blob.add(tris.reshape(-1)),
        "groups": groups_out,
    }


def encode_deltas(d):
    """(N,3) metres -> int16 running differences along the vertex order."""
    q = np.round(d / QUANTUM_M).astype(np.int64)
    dq = np.diff(q, axis=0, prepend=np.zeros((1, 3), np.int64))
    assert np.abs(dq).max() < 32767
    return dq.astype(np.int16).reshape(-1)


def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    c = x[:3]
    r = float(np.sqrt(x[3] + c @ c))
    return c, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ict", required=True, help="path to ICT-FaceKit checkout")
    ap.add_argument("--out", default="assets")
    args = ap.parse_args()

    D = os.path.join(args.ict, "FaceXModel")
    meta_src = json.load(open(os.path.join(D, "vertex_indices.json")))
    V, T, F, M = read_full(os.path.join(D, "generic_neutral_mesh.obj"))
    n = len(V)
    blob = Blob()
    meta = {
        "source": "ICT-FaceKit (USC Institute for Creative Technologies, MIT License)",
        "units": "meters",
        "vertexCount": n,
        "landmarks68": meta_src["idx_to_landmark_verts"],
        "ranges": {
            "face": RANGE_FACE, "headNeck": RANGE_HEADNECK,
            "mouthSocket": RANGE_MOUTH_SOCKET, "eyeSocket": RANGE_EYE_SOCKET,
            "scleraL": (21451, 22220), "irisL": (22221, 23020),
            "scleraR": (23021, 23790), "irisR": (23791, 24590),
        },
    }
    meta["neutral"] = blob.add((V * CM_TO_M).astype(np.float32).reshape(-1))

    # Topology used for CPU normal computation (original vertex ids).
    meta["parts"] = {}
    for name, spec in PARTS.items():
        meta["parts"][name] = build_part(name, spec, V, T, F, M, blob)

    # Eyes: sphere fit on the sclera (ICT "L" is the +x side).
    eyes = {}
    for side, (a, b) in {"L": (21451, 22220), "R": (23021, 23790)}.items():
        c, r = fit_sphere(V[a:b + 1])
        eyes[side] = {"center": (c * CM_TO_M).tolist(), "radius": r * CM_TO_M}
    meta["eyes"] = eyes

    meta["quantum"] = QUANTUM_M

    # Identity PCA modes (dense). Weights are ~N(0, 1).
    ids = []
    for k in range(NUM_IDENTITY):
        d = (read_positions(os.path.join(D, "identity%03d.obj" % k)) - V) * CM_TO_M
        ids.append({"data": blob.add(encode_deltas(d))})
    meta["identity"] = ids

    # Expressions (sparse: sorted vertex ids stored as differences too).
    exprs = []
    for name in meta_src["expressions"]:
        d = read_positions(os.path.join(D, name + ".obj")) - V
        idx = np.nonzero(np.abs(d).max(1) > SPARSE_EPS_CM)[0]
        didx = np.diff(idx, prepend=0).astype(np.uint16)
        exprs.append({
            "name": name, "count": int(len(idx)),
            "index": blob.add(didx), "delta": blob.add(encode_deltas(d[idx] * CM_TO_M)),
        })
    meta["expressions"] = exprs

    os.makedirs(args.out, exist_ok=True)
    with gzip.open(os.path.join(args.out, "face.bin.gz"), "wb", compresslevel=9) as fh:
        fh.write(blob.bytes())
    with open(os.path.join(args.out, "face.json"), "w") as fh:
        json.dump(meta, fh, separators=(",", ":"))
    print("wrote", blob.size, "bytes")


if __name__ == "__main__":
    main()
