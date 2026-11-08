"""Bake per-vertex ambient occlusion / thickness and per-texel geometry maps.

Writes:
  assets/face_extra.bin   uint8 [AO, thickness] per original ICT vertex
  <cache>/texel_geo.npz   per-texel position/normal/scale for build_textures.py
"""
import argparse
import os

import numpy as np
import trimesh

from facedata import FaceData
from texlib import ATLAS_W, ATLAS_H, atlas_px, rasterize


def vertex_normals(V, tris):
    n = np.zeros_like(V)
    fn = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-12)


def hemisphere_dirs(k, rng):
    """k cosine-weighted directions around +Z."""
    u1, u2 = rng.random(k), rng.random(k)
    r = np.sqrt(u1)
    phi = 2 * np.pi * u2
    return np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)], 1)


def frame(n):
    a = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    t = np.cross(a, n)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(n, t)
    return t, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rays", type=int, default=160)
    args = ap.parse_args()

    fd = FaceData(args.assets)
    V = fd.V
    so, suv, sidx, sgroups = fd.part("skin")
    mo, muv, midx, _ = fd.part("mouth")
    skin_t = so[sidx]
    mouth_t = mo[midx]

    # Occluders: skin + mouth + eyeball spheres.
    occ_V = [V]
    occ_T = [skin_t, mouth_t]
    base = len(V)
    for side in ("L", "R"):
        e = fd.meta["eyes"][side]
        s = trimesh.creation.icosphere(4, radius=e["radius"] * 0.995)
        occ_V.append(s.vertices + np.array(e["center"]))
        occ_T.append(s.faces + base)
        base += len(s.vertices)
    occ = trimesh.Trimesh(np.concatenate(occ_V), np.concatenate(occ_T), process=False)
    ray = occ.ray

    all_t = np.concatenate([skin_t, mouth_t])
    N = vertex_normals(V, all_t)
    used = np.unique(all_t)

    rng = np.random.default_rng(7)
    dirs = hemisphere_dirs(args.rays, rng)
    t, b = frame(N[used])
    ao = np.ones(len(V))
    max_d = 0.05
    hits = np.zeros(len(used))
    for d in dirs:
        D = t * d[0] + b * d[1] + N[used] * d[2]
        O = V[used] + N[used] * 2e-4
        loc, ridx, _ = ray.intersects_location(O, D, multiple_hits=False)
        if len(ridx):
            dist = np.linalg.norm(loc - O[ridx], axis=1)
            close = dist < max_d
            # Soft falloff with distance so far geometry occludes less.
            w = 1.0 - (dist[close] / max_d) ** 2
            np.add.at(hits, ridx[close], w)
    ao[used] = 1.0 - hits / len(dirs)
    print("AO range", ao[used].min(), ao[used].max())

    # Thickness: distance to the opposite surface along a cone around -N.
    cone = hemisphere_dirs(24, rng)
    cone[:, :2] *= 0.5
    cone /= np.linalg.norm(cone, axis=1, keepdims=True)
    thick = np.zeros(len(V))
    acc = np.zeros(len(used))
    for d in cone:
        D = -(t * d[0] + b * d[1] + N[used] * d[2])
        O = V[used] - N[used] * 2e-4
        loc, ridx, _ = ray.intersects_location(O, D, multiple_hits=False)
        dd = np.full(len(used), 0.05)
        if len(ridx):
            dd[ridx] = np.minimum(np.linalg.norm(loc - O[ridx], axis=1), 0.05)
        acc += dd
    thick[used] = acc / len(cone)
    print("thickness mm p5/p50", np.percentile(thick[used], 5) * 1000, np.median(thick[used]) * 1000)

    extra = np.stack([np.clip(ao, 0, 1) * 255, np.clip(thick / 0.05, 0, 1) * 255], 1)
    extra = np.round(extra).astype(np.uint8)
    extra.tofile(os.path.join(args.assets, "face_extra.bin"))

    # Per-texel geometry over the skin atlas (face + head groups only).
    sel = np.concatenate([sgroups["face"], sgroups["head"]])
    tri_uv = suv[sel]
    tri_px = atlas_px(tri_uv)
    P = V[so[sel]]
    Nn = N[so[sel]]
    # metres per pixel for each triangle (sqrt of area ratio)
    a3 = np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    e1, e2 = tri_px[:, 1] - tri_px[:, 0], tri_px[:, 2] - tri_px[:, 0]
    a2 = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    mpp = np.sqrt(a3 / np.maximum(a2, 1e-9))
    attrs = {
        "P": P,
        "N": Nn,
        "ao": ao[so[sel]][..., None],
        "thick": thick[so[sel]][..., None],
        "mpp": np.repeat(mpp[:, None, None], 3, 1),
    }
    maps, mask, tid = rasterize(tri_px, attrs, ATLAS_W, ATLAS_H)
    os.makedirs(args.cache, exist_ok=True)
    np.savez_compressed(os.path.join(args.cache, "texel_geo.npz"),
                        P=maps["P"].astype(np.float32), N=maps["N"].astype(np.float32),
                        ao=maps["ao"][..., 0].astype(np.float32),
                        thick=maps["thick"][..., 0].astype(np.float32),
                        mpp=maps["mpp"][..., 0].astype(np.float32), mask=mask)
    print("coverage", mask.mean())


if __name__ == "__main__":
    main()
