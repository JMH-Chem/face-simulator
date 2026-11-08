"""Procedural skin texture synthesis in the ICT face UV atlas.

Inputs : assets/face.*, <cache>/texel_geo.npz (from bake_geometry.py)
Outputs (assets/tex/):
  skin_color.jpg   R melanin multiplier/4, G hemoglobin/2.5, B oil (clear coat)
  skin_normal.jpg  RG tangent normal xy (static micro relief + age lines), B cavity
  skin_detail.jpg  R roughness, G pore strength, B pore size (0 fine .. 1 coarse)
  skin_hair.jpg    R beard stubble, G scalp stubble, B brow density
  wrinkles.jpg     dynamic wrinkle depth: R forehead, G smile/squint, B frown/sneer

Skin colour itself is computed in the shader from melanin + hemoglobin with a
Beer-Lambert style absorption model, so tone and flushing stay adjustable.
All geometric features are placed in metric face coordinates (x right, y up,
in cm) taken from the rasterised 3D position of each texel.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from facedata import FaceData
from texlib import (ATLAS_H, ATLAS_W, FACE_RES, Perlin3, atlas_px, fbm, fill_nearest,
                    height_to_normal, save_jpg, smoothstep, to_u8)

RNG = np.random.default_rng(1234)


def lowres(fn, Pm, factor):
    """Evaluate a smooth (low-frequency) field on a decimated grid and upsample."""
    n = fn(Pm[::factor, ::factor])
    out = ndimage.zoom(n, factor, order=1)
    return out[:Pm.shape[0], :Pm.shape[1]]


def blob(P, c, sigma, mirror=True):
    """Gaussian blob in cm space; mirrored across x=0 by default."""
    x = np.abs(P[..., 0]) if mirror else P[..., 0]
    d2 = (x - c[0]) ** 2 + (P[..., 1] - c[1]) ** 2 + (P[..., 2] - c[2]) ** 2
    return np.exp(-d2 / (2 * sigma * sigma))


def seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy + 1e-12), 0, 1)
    return np.hypot(px - ax - t * vx, py - ay - t * vy), t


def groove(d, w):
    """Wrinkle cross-section: a narrow valley with soft shoulders (d, w in cm)."""
    return np.exp(-(d / w) ** 2)


class Region:
    """Bounding-box index over the (X, Y) cm maps so features only touch nearby texels."""

    def __init__(self, X, Y, cell=0.5):
        self.X, self.Y = X, Y
        self.shape = X.shape
        self.cell = cell
        flat = np.arange(X.size)
        kx = np.floor(X.ravel() / cell).astype(np.int64)
        ky = np.floor(Y.ravel() / cell).astype(np.int64)
        order = np.lexsort((kx, ky))
        self.order = flat[order]
        self.kx, self.ky = kx[order], ky[order]
        self.keys = self.ky * 100000 + self.kx
        self.xf, self.yf = X.ravel(), Y.ravel()

    def select(self, x0, x1, y0, y1):
        out = []
        for ky in range(int(np.floor(y0 / self.cell)), int(np.floor(y1 / self.cell)) + 1):
            a = np.searchsorted(self.keys, ky * 100000 + int(np.floor(x0 / self.cell)))
            b = np.searchsorted(self.keys, ky * 100000 + int(np.floor(x1 / self.cell)) + 1)
            if b > a:
                out.append(self.order[a:b])
        return np.concatenate(out) if out else np.zeros(0, np.int64)


REG = None


def polyline_groove(X, Y, pts, w, depth, taper=True):
    """Accumulate a groove along a polyline of (x, y) points (cm)."""
    h = np.zeros(X.size)
    pts = np.asarray(pts, np.float64)
    m = 4 * w
    idx = REG.select(pts[:, 0].min() - m, pts[:, 0].max() + m, pts[:, 1].min() - m, pts[:, 1].max() + m)
    if len(idx) == 0:
        return h.reshape(X.shape)
    px, py = REG.xf[idx], REG.yf[idx]
    best = np.zeros(len(idx))
    n = len(pts) - 1
    for i in range(n):
        d, t = seg_dist(px, py, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
        s = (i + t) / n
        amp = np.sin(np.pi * s) ** 0.6 if taper else 1.0
        best = np.maximum(best, depth * amp * groove(d, w))
    h[idx] = best
    return h.reshape(X.shape)


def curve(fn, a, b, n=24):
    xs = np.linspace(a, b, n)
    return [(x, fn(x)) for x in xs]


def landmarks_uv(fd):
    o, uv, _, _ = fd.part("skin")
    out = []
    for li in fd.meta["landmarks68"]:
        w = [i for i in np.nonzero(o == li)[0] if uv[i, 0] < 1]
        out.append(uv[w[0]])
    return np.array(out)


def spline_closed(pts, n=40):
    """Closed Catmull-Rom spline through pts (K,2) -> (K*n, 2)."""
    pts = np.asarray(pts, np.float64)
    K = len(pts)
    out = []
    for i in range(K):
        p0, p1, p2, p3 = pts[(i - 1) % K], pts[i], pts[(i + 1) % K], pts[(i + 2) % K]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)


def poly_mask(points_px, W, H, blur):
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in points_px], fill=255)
    m = np.asarray(im, np.float64) / 255.0
    return ndimage.gaussian_filter(m, blur) if blur > 0 else m


def follicles(mask, density_per_px, radius_px, seed):
    """Random dark dots (hair follicles / stubble) inside mask."""
    rng = np.random.default_rng(seed)
    H, W = mask.shape
    n = int(mask.sum() * density_per_px)
    ys = rng.random(n) * H
    xs = rng.random(n) * W
    keep = rng.random(n) < mask[ys.astype(int), xs.astype(int)]
    ys, xs = ys[keep], xs[keep]
    img = np.zeros((H, W))
    np.add.at(img, (ys.astype(int), xs.astype(int)), 0.6 + 0.4 * rng.random(len(ys)))
    img = ndimage.gaussian_filter(img, radius_px) * (2 * np.pi * radius_px ** 2)
    return np.clip(img, 0, 1)


def brow_strokes(lm_px, W, H, side, mpp_cm, rng):
    """Paint individual eyebrow hairs as anti-aliased strokes in UV space.

    lm_px: 5 brow landmarks in pixel coords ordered lateral -> medial.
    Returns (density map (0..1), list of strand polylines in pixel coords).
    """
    scale = 4  # supersample
    im = Image.new("L", (W * scale, H * scale), 0)
    dr = ImageDraw.Draw(im)
    pts = np.array(lm_px, np.float64)
    # resample the brow arc (lateral -> medial), shift slightly down onto the brow body
    t = np.linspace(0, 1, 200)
    seg = np.linspace(0, 1, len(pts))
    arc = np.stack([np.interp(t, seg, pts[:, 0]), np.interp(t, seg, pts[:, 1])], 1)
    px_per_cm = 1.0 / mpp_cm
    arc[:, 1] += 0.25 * px_per_cm
    tang = np.gradient(arc, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    nrm = np.stack([tang[:, 1], -tang[:, 0]], 1)
    if nrm[100, 1] > 0:  # make normal point up (towards forehead => -y in image)
        nrm = -nrm
    nhair = 1150
    strands = []
    for _ in range(nhair):
        s = rng.random() ** 0.85  # 0 lateral tail, 1 medial head
        i = int(s * 199)
        thick = (0.25 + 0.75 * smoothstep(0.0, 0.75, s)) * 0.55 * px_per_cm  # brow height
        off = (rng.random() * 2 - 1)
        root = arc[i] + nrm[i] * off * thick * 0.5
        # hair direction: medial head points up, body runs laterally (towards s=0)
        lat = -tang[i]  # towards tail
        up = nrm[i]
        head = smoothstep(0.78, 0.98, s)
        conv = -off * 0.35  # upper hairs lean down, lower hairs lean up
        d = lat * (1 - head) + up * (head * 1.2) + up * conv * (1 - head) + lat * 0.25 * head * 0
        d += rng.normal(0, 0.12, 2)
        d /= np.linalg.norm(d)
        length = (0.55 + 0.35 * rng.random()) * px_per_cm * (0.7 + 0.3 * (1 - head))
        bend = rng.normal(0, 0.15)
        seg_pts = []
        p = root.copy()
        dd = d.copy()
        # curl towards the brow's centre line (mirror-safe: bend towards -/+ normal)
        toward = -up * np.sign(off) * (1 - head) + up * 0.15 * head
        for k in range(7):
            seg_pts.append(p.copy())
            dd = dd + toward * 0.11 + lat * bend * 0.04
            dd /= np.linalg.norm(dd)
            p = p + dd * length / 6
        strands.append((np.array(seg_pts), val_f := 0.6 + 0.4 * rng.random()))
        seg_pts = [(q[0] * scale, q[1] * scale) for q in seg_pts]
        val = int(255 * val_f)
        for k in range(len(seg_pts) - 1):
            w = max(1, int(round(scale * 0.75 * (1 - k / 7.0))))
            dr.line([seg_pts[k], seg_pts[k + 1]], fill=val, width=w)
    big = np.asarray(im, np.float64) / 255.0
    small = big.reshape(H, scale, W, scale).mean((1, 3))
    return np.clip(small * 1.35, 0, 1), strands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()
    out_dir = os.path.join(args.assets, "tex")
    os.makedirs(out_dir, exist_ok=True)

    fd = FaceData(args.assets)
    g = np.load(os.path.join(args.cache, "texel_geo.npz"))
    mask = g["mask"]
    P = fill_nearest(g["P"].astype(np.float64), mask) * 100.0  # cm
    mpp = fill_nearest(g["mpp"].astype(np.float64), mask)       # m / px
    thick = fill_nearest(g["thick"].astype(np.float64), mask)
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    AX = np.abs(X)
    global REG
    REG = Region(np.where(mask, X, 1e3), np.where(mask, Y, 1e3))
    H, W = mask.shape
    face_tile = np.zeros((H, W), bool)
    face_tile[:, :FACE_RES] = True

    noise = Perlin3(11)
    noise2 = Perlin3(23)
    Pm = P / 100.0  # metres for noise frequencies

    lm_uv = landmarks_uv(fd)
    lm_px = atlas_px(lm_uv)
    LM = fd.V[fd.meta["landmarks68"]] * 100.0

    # ------------------------------------------------------------------ regions
    lips = poly_mask(spline_closed(lm_px[48:60]), W, H, blur=1.0)
    lips_soft = ndimage.gaussian_filter(lips, 6)
    # extend lips into the mouth slit (inner lip mucosa)
    inner = poly_mask(spline_closed(lm_px[60:68]), W, H, blur=2.0)
    lips = np.clip(lips + inner, 0, 1)
    eye_mask = np.zeros((H, W))
    for a in (36, 42):
        eye_mask = np.maximum(eye_mask, poly_mask(lm_px[a:a + 6], W, H, blur=2.0))

    nose_tip = blob(P, (0, 0.2, 12.8), 1.1)
    alar = blob(P, (1.75, -0.4, 10.4), 0.75)
    nose = np.clip(blob(P, (0, 1.5, 12.0), 1.6) + nose_tip + alar, 0, 1)
    cheeks = blob(P, (4.0, 0.3, 8.9), 1.9)
    under_eye = blob(P, (3.0, 2.35, 9.4), 0.75) * (1 - eye_mask)
    lids = blob(P, (3.1, 4.05, 9.3), 0.7) * (1 - eye_mask)
    forehead = smoothstep(6.0, 8.0, Y) * smoothstep(-6.5, -3.0, -AX) * (Z > 3)
    chin = blob(P, (0, -6.3, 10.3), 1.5, mirror=False)
    ears = smoothstep(6.6, 7.6, AX) * smoothstep(-3.5, -1.5, Y) * smoothstep(-8.5, -6.8, -Y) \
        * smoothstep(-4.0, -1.0, Z) * (thick < 0.02)
    ears = ndimage.gaussian_filter(ears.astype(np.float64), 4)
    neck = smoothstep(-8.0, -11.0, Y) * (1 - chin)

    # Scalp: hairline as a function of azimuth around the vertical head axis.
    theta = np.degrees(np.abs(np.arctan2(X, Z - 0.5)))
    tab_t = [0, 18, 35, 52, 62, 70, 80, 95, 115, 135, 180]
    tab_y = [11.3, 11.1, 10.4, 9.3, 8.1, 4.2, 6.6, 6.8, 2.5, -1.5, -2.8]
    hairline = np.interp(theta, tab_t, tab_y)
    hairline += 0.4 * lowres(lambda q: fbm(noise2, q, 120, 3), Pm, 2) + 0.18 * noise2(Pm * 900)
    scalp = smoothstep(-0.55, 0.45, Y - hairline) ** 1.5
    # keep the ear itself clear of hair
    scalp *= 1 - np.clip(ears * 1.5, 0, 1)

    # Beard: below a line through the sideburn, cheek and upper lip.
    beard_top = np.interp(AX, [0, 1.2, 2.6, 4.2, 5.8, 6.8, 7.4], [-1.25, -1.4, -2.3, -2.7, -1.2, 0.4, 2.2])
    beard_top += 0.25 * lowres(lambda q: fbm(noise2, q + 3.1, 90, 3), Pm, 4)
    beard = smoothstep(0.0, 0.6, beard_top - Y)
    beard_bottom = -10.5 + 0.6 * lowres(lambda q: fbm(noise2, q + 7.7, 60, 2), Pm, 4)
    beard *= smoothstep(-0.8, 0.8, Y - beard_bottom) * (Z > -1.5)
    beard *= 1 - np.clip(lips_soft * 1.4 + lips, 0, 1)
    beard *= 1 - smoothstep(6.8, 7.6, AX) * smoothstep(-0.5, 2.0, -Z + 1.5)  # not behind the jaw
    beard = np.clip(beard - ears, 0, 1)

    # ---------------------------------------------------------- melanin / hb
    low = lowres(lambda q: fbm(noise, q, 18, 3), Pm, 8)
    mid = lowres(lambda q: fbm(noise, q + 5.0, 140, 3), Pm, 2)
    fine = fbm(noise2, Pm + 9.0, 700, 2)
    M = 1.0 + 0.09 * low + 0.07 * mid + 0.035 * fine
    M += 0.14 * under_eye + 0.10 * lids
    M *= 1 + 0.12 * lips
    M -= 0.05 * forehead * smoothstep(0.2, 0.8, low)
    # Freckles / small pigment spots (sparse, mostly nose and cheeks).
    spots = smoothstep(0.55, 0.75, noise(Pm * 900 + 3.3)) * smoothstep(0.0, 0.6, lowres(lambda q: noise2(q * 60), Pm, 4))
    M += 0.35 * spots * np.clip(nose + cheeks * 0.8 + forehead * 0.3, 0, 1)
    # A few moles.
    for c, r in [((-3.9, -1.2, 8.6), 0.07), ((5.3, -5.6, 6.0), 0.06), ((-1.6, 8.6, 9.6), 0.05),
                 ((2.2, -9.4, 6.6), 0.08)]:
        M += 2.4 * np.exp(-((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) / (2 * r * r))

    Hb = 0.52 + 0.08 * low + 0.10 * mid + 0.05 * fine
    Hb += 0.2 * blob(P, (4.0, 0.0, 8.7), 2.5) + 0.3 * nose_tip + 0.3 * alar + 0.1 * nose
    Hb += 0.30 * lids + 0.10 * under_eye + 0.18 * chin + 0.40 * ears
    Hb -= 0.08 * forehead + 0.10 * neck + 0.12 * scalp
    # Fine capillary mottling on cheeks / nose.
    cap = smoothstep(0.35, 0.8, np.abs(fbm(noise2, Pm + 1.7, 380, 3)) * 2.2)
    Hb += 0.12 * cap * np.clip(cheeks + nose, 0, 1)
    Hb = Hb * (1 - lips) + lips * (1.5 + 0.2 * mid + 0.12 * fine)

    # Eyebrows (density is used as pigment and also drives the 3D brow cards).
    brows = np.zeros((H, W))
    px_cm = (np.median(mpp[:, :FACE_RES][mask[:, :FACE_RES]]) * 100.0)
    bR, sR = brow_strokes(lm_px[17:22], W, H, "R", px_cm, RNG)
    bL, sL = brow_strokes(lm_px[26:21:-1], W, H, "L", px_cm, RNG)
    brows = np.maximum(bR, bL)
    # Export the same strands for the 3D brow hairs (atlas UV, 7 points each).
    pts = np.stack([s_[0] for s_ in sR + sL])            # (N, 7, 2) pixels
    uv = np.stack([pts[..., 0] / W, 1.0 - pts[..., 1] / H], -1)
    q = np.clip(np.round(uv * 65535), 0, 65535).astype(np.uint16)
    shade = np.array([s_[1] for s_ in sR + sL])
    with open(os.path.join(args.assets, "brows.bin"), "wb") as fh:
        np.array([len(q), q.shape[1]], np.uint16).tofile(fh)
        q.tofile(fh)
        np.round(shade * 255).astype(np.uint8).tofile(fh)
    brows *= face_tile
    brow_shadow = ndimage.gaussian_filter(brows, 5)
    M += 0.45 * brow_shadow + 3.0 * brows

    # --------------------------------------------------------------- stubble
    beard_dots = follicles(beard * face_tile + beard * (~face_tile), 0.015, 0.7, 5)
    beard_map = np.clip(0.35 * beard + 0.75 * beard_dots, 0, 1)
    scalp_dots = follicles(scalp, 0.05, 0.6, 6)
    scalp_map = np.clip(0.45 * scalp + 0.8 * scalp_dots, 0, 1)

    # ---------------------------------------------------------- height maps
    # Static relief (metres): micro bumps + age lines.
    h = 80e-6 * lowres(lambda q: fbm(noise, q + 2.0, 260, 3), Pm, 2) + 28e-6 * fbm(noise, Pm + 6.0, 750, 2) + 12e-6 * fbm(noise2, Pm + 4.0, 1300, 2)
    h *= 1 - 0.85 * lips

    # Forehead lines follow the brow arc.
    def brow_y(x):
        return 6.15 - 0.032 * x * x

    fore = np.zeros((H, W))
    for k, off in enumerate([1.45, 2.25, 3.05, 3.85]):
        wav = 0.08 * lowres(lambda q: noise(np.stack([q[..., 0] * 90, q[..., 0] * 0 + k * 3.7, q[..., 0] * 0], -1)), Pm, 4)
        d = np.abs(Y - (brow_y(X) + off + wav))
        breaks = smoothstep(-0.15, 0.25, lowres(lambda q: noise2(np.stack([q[..., 0] * 60, q[..., 0] * 0 + k * 5.1, q[..., 0] * 0], -1)), Pm, 4))
        span = smoothstep(-(5.0 - 0.4 * k), -(3.2 - 0.4 * k), -AX)
        fore = np.maximum(fore, groove(d, 0.06) * breaks * span)
    fore *= smoothstep(6.4, 7.0, Y) * (Z > 4)

    # Crow's feet: rays from the outer canthus.
    crow = np.zeros((H, W))
    for side in (-1, 1):
        cx, cy = 4.5 * side, 3.4
        for k, ang in enumerate(np.linspace(-38, 26, 6)):
            a = np.radians(ang + RNG.normal(0, 4))
            r0, r1 = 0.55 + 0.1 * RNG.random(), 1.6 + 0.6 * RNG.random()
            bend = RNG.normal(0, 0.05)
            pts = []
            for s in np.linspace(0, 1, 10):
                r = r0 + (r1 - r0) * s
                aa = a + bend * s
                pts.append((cx + side * r * np.cos(aa), cy + r * np.sin(aa)))
            crow = np.maximum(crow, polyline_groove(X, Y, pts, 0.045, 1.0 - 0.1 * k))
    # Fine lines under the lower lid, concentric with the eye.
    for side in (-1, 1):
        cx, cy = 3.05 * side, 3.65
        for k, r in enumerate([0.95, 1.2, 1.5]):
            pts = [(cx + r * 1.15 * np.cos(t), cy + r * 0.75 * np.sin(t))
                   for t in np.linspace(np.radians(205), np.radians(335), 16)]
            crow = np.maximum(crow, polyline_groove(X, Y, pts, 0.04, 0.55 - 0.12 * k))
    # Nasolabial deepening (smile).
    naso = np.zeros((H, W))
    for side in (-1, 1):
        pts = [(side * (1.85 + 0.6 * s + 0.35 * s * s), -0.5 - 3.6 * s) for s in np.linspace(0, 1, 12)]
        naso = np.maximum(naso, polyline_groove(X, Y, pts, 0.11, 1.0))
    smile_set = np.maximum(crow, 0.8 * naso) * (Z > 4)

    # Frown: glabella verticals, procerus horizontal, bunny lines; chin dimpling.
    frown = np.zeros((H, W))
    for side in (-1, 1):
        pts = [(side * (0.55 + 0.15 * s), 4.9 + 1.9 * s) for s in np.linspace(0, 1, 8)]
        frown = np.maximum(frown, polyline_groove(X, Y, pts, 0.06, 1.0))
        pts = [(side * (0.75 + 0.9 * s), 4.2 - 1.2 * s) for s in np.linspace(0, 1, 8)]
        frown = np.maximum(frown, polyline_groove(X, Y, pts, 0.05, 0.75))
        pts = [(side * (1.0 + 0.8 * s), 3.7 - 0.9 * s) for s in np.linspace(0, 1, 8)]
        frown = np.maximum(frown, polyline_groove(X, Y, pts, 0.045, 0.6))
    frown = np.maximum(frown, 0.7 * polyline_groove(X, Y, [(-0.9, 4.55), (0, 4.45), (0.9, 4.55)], 0.05, 1.0))
    dimple = smoothstep(0.1, 0.6, lowres(lambda q: fbm(noise, q + 8.0, 260, 2), Pm, 2) + 0.2) * chin * (Y < -5.0)
    frown = np.maximum(frown * (Z > 4), 0.6 * dimple)

    # Static age lines: faint version of the dynamic sets, plus lip and neck lines.
    h -= 28e-6 * fore + 18e-6 * crow + 10e-6 * frown
    mc = LM[62] * 0.5 + LM[66] * 0.5
    phi = np.arctan2(Y - mc[1], X - mc[0])
    rr = np.hypot(X - mc[0], (Y - mc[1]) * 1.6)
    lip_lines = np.abs(np.sin(phi / 0.055 * np.pi + 1.5 * noise(Pm * 400)))
    lip_lines = (1 - lip_lines) ** 6 * smoothstep(0.3, 0.9, rr)
    h -= 45e-6 * lip_lines * lips
    neck_lines = np.zeros((H, W))
    for k, yy in enumerate([-12.0, -13.4]):
        d = np.abs(Y - (yy + 0.012 * X * X + 0.2 * lowres(lambda q: noise(np.stack([q[..., 0] * 50, q[..., 0] * 0 + k, q[..., 0] * 0], -1)), Pm, 4)))
        neck_lines = np.maximum(neck_lines, groove(d, 0.12) * smoothstep(-6.5, -3.0, -AX))
    h -= 60e-6 * neck_lines * (Z > -2)
    # Upper lid crease.
    for side in (-1, 1):
        pts = [(side * (1.9 + 2.5 * s), 4.55 + 0.25 * np.sin(np.pi * s) - 0.35 * s * s) for s in np.linspace(0, 1, 12)]
        h -= 45e-6 * polyline_groove(X, Y, pts, 0.07, 1.0)

    nrm = height_to_normal(h, mpp)
    cavity = np.clip(1 - 2500 * np.maximum(-(h - ndimage.gaussian_filter(h, 3)), 0) * 10, 0.3, 1)

    # -------------------------------------------------------- spec / pores
    rough = 0.52 + 0.03 * mid
    rough -= 0.10 * np.clip(forehead + nose, 0, 1) + 0.04 * chin
    rough += 0.04 * cheeks * (1 - nose)
    rough = rough * (1 - lips) + lips * (0.42 + 0.12 * lip_lines)
    rough += 0.05 * scalp + 0.05 * neck
    oil = 0.25 + 0.45 * forehead + 0.6 * nose + 0.25 * chin - 0.1 * cheeks
    oil = np.clip(oil * (1 - lips) + lips * 0.55 - 0.15 * neck, 0, 1)
    pore = 0.75 + 0.25 * nose + 0.15 * cheeks - 0.25 * forehead * 0
    pore *= 1 - 0.85 * np.clip(lids * 1.4 + under_eye * 0.6 + eye_mask, 0, 1)
    pore *= 1 - lips
    pore *= 1 - 0.5 * ears
    pore = np.clip(pore * (1 - 0.35 * neck) * (1 - 0.3 * scalp), 0, 1)
    pore_size = np.clip(nose * 1.1 + 0.5 * cheeks * smoothstep(-5.0, -2.0, -AX), 0, 1)

    # ---------------------------------------------------------------- write
    pad = lambda a: fill_nearest(a, mask)
    color = np.stack([pad(np.clip(M / 4.0, 0, 1)), pad(np.clip(Hb / 2.5, 0, 1)), pad(oil)], -1)
    save_jpg(os.path.join(out_dir, "skin_color.jpg"), to_u8(color), 93)
    nrm_img = np.stack([pad(nrm[..., 0] * 0.5 + 0.5), pad(nrm[..., 1] * 0.5 + 0.5), pad(cavity)], -1)
    save_jpg(os.path.join(out_dir, "skin_normal.jpg"), to_u8(nrm_img), 95)
    det = np.stack([pad(np.clip(rough, 0, 1)), pad(pore), pad(pore_size)], -1)
    save_jpg(os.path.join(out_dir, "skin_detail.jpg"), to_u8(det), 92)
    hair = np.stack([pad(beard_map), pad(scalp_map), pad(brows)], -1)
    save_jpg(os.path.join(out_dir, "skin_hair.jpg"), to_u8(hair), 92)
    wr = np.stack([pad(fore), pad(smile_set), pad(frown)], -1)
    save_jpg(os.path.join(out_dir, "wrinkles.jpg"), to_u8(np.clip(wr, 0, 1)), 92)

    # Debug preview (albedo approximation) for quick inspection.
    if os.environ.get("PREVIEW"):
        m_, hb_ = 0.18 * M, 0.55 * Hb
        sig_m = np.array([1.85, 2.26, 2.75]) * 0.9
        sig_h = np.array([0.04, 0.96, 1.20])
        D = np.array([0.93, 0.90, 0.86])
        alb = D * np.exp(-(m_[..., None] * sig_m + hb_[..., None] * sig_h))
        alb *= 1 - 0.55 * beard_map[..., None] * 0.6 - 0.6 * scalp_map[..., None] * 0.8
        alb = np.where(mask[..., None], alb, 0.1)
        srgb = np.where(alb <= 0.0031308, alb * 12.92, 1.055 * alb ** (1 / 2.4) - 0.055)
        Image.fromarray(to_u8(srgb)).save(os.environ["PREVIEW"])
    print("done")


if __name__ == "__main__":
    main()
