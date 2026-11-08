"""Shared helpers for the procedural texture tools: rasteriser, noise, I/O."""
import numpy as np
from PIL import Image
from scipy import ndimage

# Skin atlas: face UDIM tile (u in [0,1]) at 2048^2 on the left, back-of-head
# tile (u in [1,2]) at 1024^2 in the lower right. The viewer applies the same
# remap to the skin UVs (see src/model.js: atlasUV).
ATLAS_W, ATLAS_H = 3072, 2048
FACE_RES, HEAD_RES = 2048, 1024


def atlas_px(uv, scale=1.0):
    u, v = uv[..., 0], uv[..., 1]
    face = u < 1.0
    x = np.where(face, u * FACE_RES, FACE_RES + (u - 1.0) * HEAD_RES)
    y = np.where(face, (1.0 - v) * FACE_RES, HEAD_RES + (1.0 - v) * HEAD_RES)
    return np.stack([x, y], -1) * scale


def rasterize(tri_px, attrs, W, H):
    """Scan-convert triangles given in pixel coords.

    tri_px: (T,3,2); attrs: dict name -> (T,3,C) per-corner values.
    Returns (dict name -> (H,W,C) interpolated maps, coverage mask (H,W),
    triangle id map (H,W)).
    """
    out = {k: np.zeros((H, W, a.shape[2]), np.float64) for k, a in attrs.items()}
    tid = np.full((H, W), -1, np.int64)
    for t in range(tri_px.shape[0]):
        a, b, c = tri_px[t]
        x0 = max(int(np.floor(min(a[0], b[0], c[0]))), 0)
        x1 = min(int(np.ceil(max(a[0], b[0], c[0]))), W - 1)
        y0 = max(int(np.floor(min(a[1], b[1], c[1]))), 0)
        y1 = min(int(np.ceil(max(a[1], b[1], c[1]))), H - 1)
        if x1 < x0 or y1 < y0:
            continue
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w0 = ((b[1] - c[1]) * (X - c[0]) + (c[0] - b[0]) * (Y - c[1])) / den
        w1 = ((c[1] - a[1]) * (X - c[0]) + (a[0] - c[0]) * (Y - c[1])) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        ys, xs = np.nonzero(inside)
        ys = ys + y0
        xs = xs + x0
        W0, W1, W2 = w0[inside], w1[inside], w2[inside]
        tid[ys, xs] = t
        for k, A in attrs.items():
            out[k][ys, xs] = W0[:, None] * A[t, 0] + W1[:, None] * A[t, 1] + W2[:, None] * A[t, 2]
    return out, tid >= 0, tid


def fill_nearest(img, mask):
    """Copy the nearest covered texel into uncovered ones (seam padding)."""
    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return img[idx[0], idx[1]]


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Perlin3:
    """Vectorised improved Perlin noise in 3D (range roughly [-1, 1])."""

    _G = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
                   [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
                   [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
                   [1, 1, 0], [-1, 1, 0], [0, -1, 1], [0, -1, -1]], np.float64)

    def __init__(self, seed=0):
        p = np.random.default_rng(seed).permutation(256)
        self.p = np.concatenate([p, p]).astype(np.int64)

    def __call__(self, P):
        P = np.asarray(P, np.float64)
        Pi = np.floor(P).astype(np.int64)
        f = P - Pi
        Pi &= 255
        u = f * f * f * (f * (f * 6 - 15) + 10)
        p = self.p
        X, Y, Z = Pi[..., 0], Pi[..., 1], Pi[..., 2]
        res = 0.0
        acc = []
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    h = p[p[p[X + dx] + Y + dy] + Z + dz] & 15
                    g = self._G[h]
                    d = f - np.array([dx, dy, dz], np.float64)
                    acc.append((g * d).sum(-1))
        ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
        lerp = lambda a, b, t: a + t * (b - a)
        x00 = lerp(acc[0], acc[4], ux)
        x01 = lerp(acc[1], acc[5], ux)
        x10 = lerp(acc[2], acc[6], ux)
        x11 = lerp(acc[3], acc[7], ux)
        y0 = lerp(x00, x10, uy)
        y1 = lerp(x01, x11, uy)
        res = lerp(y0, y1, uz)
        return res


def fbm(noise, P, freq, octaves=4, lac=2.0, gain=0.5):
    tot, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        tot = tot + amp * noise(P * freq * (lac ** o) + o * 17.13)
        norm += amp
        amp *= gain
    return tot / norm


def height_to_normal(h, mpp, strength=1.0, wrap=False):
    """Tangent-space normal (OpenGL / +Y up convention) from a height map.

    h: (H,W) height in metres. mpp: metres per pixel (scalar or (H,W)).
    Image rows go down while +v goes up, hence the sign on dy.
    """
    if wrap:
        dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
        dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    else:
        dx = np.gradient(h, axis=1)
        dy = np.gradient(h, axis=0)
    dhdu = dx / mpp * strength
    dhdv = -dy / mpp * strength
    n = np.stack([-dhdu, -dhdv, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n


def encode_normal(n):
    return np.clip(np.round((n * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)


def to_u8(x):
    return np.clip(np.round(np.asarray(x) * 255.0), 0, 255).astype(np.uint8)


def save_png(path, arr):
    Image.fromarray(arr).save(path, optimize=True)


def save_jpg(path, arr, q=92):
    Image.fromarray(arr).save(path, quality=q, subsampling=0)
