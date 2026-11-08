"""Small procedural textures: tileable skin micro-relief, iris, sclera, eyelashes.

Outputs (assets/tex/):
  pores.png    tileable 1024^2, ~12 mm square of skin: RG normal xy, B cavity
  iris.png     512^2, centre = pupil, radius 0.5 = limbus:
               R fibre brightness, G inner-ring pigment mask, B relief height
  sclera.jpg   1024x512 equirect around the eye axis (v=0 at the cornea apex):
               R blood vessels, G yellowing / mottling
  lashes.png   2048^2 alpha for the ICT eyelash strips (UDIM tile 6)
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from texlib import encode_normal, height_to_normal, save_jpg, save_png, smoothstep, to_u8

RNG = np.random.default_rng(99)


# ------------------------------------------------------------------ pores
def worley_wrap(N, n_pts, rng):
    pts = rng.random((n_pts, 2)) * N
    # tile 3x3 for wrap-around
    offs = np.array([[dx, dy] for dx in (-N, 0, N) for dy in (-N, 0, N)], np.float64)
    allp = (pts[None, :, :] + offs[:, None, :]).reshape(-1, 2)
    from scipy.spatial import cKDTree
    tree = cKDTree(allp)
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    q = np.stack([xx.ravel(), yy.ravel()], 1)
    d, i = tree.query(q, k=2)
    F1 = d[:, 0].reshape(N, N)
    F2 = d[:, 1].reshape(N, N)
    return F1, F2, pts


def periodic_noise(N, scale, rng, octaves=3):
    out = np.zeros((N, N))
    amp = 1.0
    for o in range(octaves):
        k = int(scale * 2 ** o)
        g = rng.normal(size=(k, k))
        up = ndimage.zoom(np.tile(g, (3, 3)), N / k, order=3)[N:2 * N, N:2 * N]
        out += amp * up
        amp *= 0.5
    return out / np.abs(out).max()


def make_pores(out, N=1024, size_m=0.012):
    mpp = size_m / N
    rng = np.random.default_rng(5)
    # Skin surface micro-relief: polygonal furrow network.
    F1, F2, pts = worley_wrap(N, 1300, rng)
    furrow = np.exp(-((F2 - F1) / 2.2) ** 2)
    F1b, F2b, _ = worley_wrap(N, 300, rng)
    furrow2 = np.exp(-((F2b - F1b) / 3.5) ** 2)
    h = -5e-6 * furrow - 7e-6 * furrow2
    # Pores: pits placed at a subset of furrow-network nodes.
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    pits = np.zeros((N, N))
    sel = pts[rng.random(len(pts)) < 0.45]
    for (px, py) in sel:
        r = rng.uniform(2.2, 5.0)
        depth = rng.uniform(0.6, 1.0)
        x0, x1 = int(px - 4 * r), int(px + 4 * r) + 1
        y0, y1 = int(py - 4 * r), int(py + 4 * r) + 1
        ys = np.arange(y0, y1) % N
        xs = np.arange(x0, x1) % N
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5 - px, np.arange(y0, y1) + 0.5 - py)
        bump = depth * np.exp(-(gx * gx + gy * gy) / (r * r))
        pits[np.ix_(ys, xs)] = np.maximum(pits[np.ix_(ys, xs)], bump)
    h -= 22e-6 * pits
    h += 3e-6 * periodic_noise(N, 16, rng)
    n = height_to_normal(h, mpp, wrap=True)
    cav = 1.0 - 0.75 * np.clip(pits, 0, 1) - 0.25 * furrow
    img = np.stack([n[..., 0] * 0.5 + 0.5, n[..., 1] * 0.5 + 0.5, np.clip(cav, 0, 1)], -1)
    save_png(os.path.join(out, "pores.png"), to_u8(img))


# ------------------------------------------------------------------- iris
def make_iris(out, N=512):
    rng = np.random.default_rng(17)
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    x = (xx - N / 2) / N
    y = (yy - N / 2) / N
    r = np.hypot(x, y) * 2.0  # 1.0 at limbus
    th = np.arctan2(y, x)

    def polar_noise(ang_freq, rad_freq, seed, octaves=3):
        g = np.random.default_rng(seed)
        acc = np.zeros_like(r)
        amp = 1.0
        for o in range(octaves):
            # sum of random-phase sinusoids in (theta, r) = streaky fibres
            for _ in range(10):
                k = int(ang_freq * (2 ** o) * g.uniform(0.7, 1.3))
                ph = g.uniform(0, 2 * np.pi)
                kr = rad_freq * (2 ** o) * g.uniform(0.5, 1.5)
                phr = g.uniform(0, 2 * np.pi)
                acc += amp * np.sin(k * th + ph + 0.8 * np.sin(kr * r + phr))
            amp *= 0.55
        return acc / np.abs(acc).max()

    fib = polar_noise(70, 9, 1, 4)
    fib2 = polar_noise(160, 20, 2, 3)
    fibres = 0.55 + 0.25 * fib + 0.2 * fib2
    # Collarette: zigzag ring separating pupillary and ciliary zones.
    col_r = 0.42 + 0.03 * np.sin(13 * th + 1.3) + 0.015 * np.sin(31 * th)
    collarette = np.exp(-((r - col_r) / 0.025) ** 2)
    inner = smoothstep(col_r + 0.02, col_r - 0.06, r)
    # Crypts: dark elongated lacunae in the ciliary zone.
    crypts = np.zeros_like(r)
    for _ in range(38):
        a = rng.uniform(-np.pi, np.pi)
        rc = rng.uniform(0.47, 0.8)
        w = rng.uniform(0.04, 0.12)
        ln = rng.uniform(0.04, 0.12)
        da = np.angle(np.exp(1j * (th - a)))
        crypts = np.maximum(crypts, np.exp(-(da * rc / (w * 0.5)) ** 2 - ((r - rc) / ln) ** 2))
    # Contraction furrows: faint concentric rings in the outer ciliary zone.
    furrows = sum(np.exp(-((r - rr - 0.01 * np.sin(7 * th + k)) / 0.008) ** 2) * smoothstep(-0.2, 0.4, np.sin(5 * th + k * 2))
                  for k, rr in enumerate([0.70, 0.79, 0.87]))
    limbal = smoothstep(0.80, 1.0, r)
    ruff = np.exp(-((r - 0.18) / 0.035) ** 2)
    R = fibres * (1 - 0.55 * crypts) * (1 - 0.35 * furrows) * (1 - 0.65 * limbal) + 0.25 * collarette
    R = np.clip(R * (1 - 0.5 * ruff), 0, 1)
    G = np.clip(inner + 0.5 * collarette, 0, 1)
    B = np.clip(0.5 + 0.25 * fib - 0.35 * crypts + 0.4 * collarette - 0.2 * furrows, 0, 1)
    img = np.stack([R, G, B], -1)
    save_png(os.path.join(out, "iris.png"), to_u8(img))


# ----------------------------------------------------------------- sclera
def make_sclera(out, W=1024, H=512):
    rng = np.random.default_rng(31)
    scale = 3
    im = Image.new("L", (W * scale, H * scale), 0)
    dr = ImageDraw.Draw(im)
    # v = theta / pi (0 at cornea apex); u = phi / 2pi. Visible band ~ theta 25..85 deg.
    for _ in range(70):
        # more vessels near the canthi (phi ~ 0 and pi)
        base = rng.choice([0.0, 0.5]) + rng.normal(0, 0.09)
        u = base % 1.0
        v = rng.uniform(0.42, 0.5)
        width = rng.uniform(1.2, 3.0)
        ang = np.pi + rng.normal(0, 0.35)  # heading towards the cornea (decreasing v)
        steps = rng.integers(25, 60)
        for s in range(steps):
            du = np.sin(ang) * 0.004
            dv = np.cos(ang) * 0.004
            nu, nv = u + du, v + dv
            if nv < 0.14:
                break
            w = max(1, int(width * scale * (1 - s / steps)))
            dr.line([(u * W * scale, v * H * scale), (nu * W * scale, nv * H * scale)], fill=int(140 + 100 * rng.random()), width=w)
            u, v = nu, nv
            ang += rng.normal(0, 0.18)
            if rng.random() < 0.06:  # branch
                bu, bv, ba = u, v, ang + rng.choice([-1, 1]) * rng.uniform(0.4, 0.9)
                for t in range(int(steps * 0.4)):
                    nbu, nbv = bu + np.sin(ba) * 0.003, bv + np.cos(ba) * 0.003
                    dr.line([(bu * W * scale, bv * H * scale), (nbu * W * scale, nbv * H * scale)], fill=110, width=max(1, int(w * 0.6)))
                    bu, bv = nbu, nbv
                    ba += rng.normal(0, 0.2)
    ves = np.asarray(im, np.float64).reshape(H, scale, W, scale).mean((1, 3)) / 255.0
    ves = ndimage.gaussian_filter(ves, 0.6)
    yy, xx = np.mgrid[0:H, 0:W] / np.array([H, W])[:, None, None]
    mott = ndimage.gaussian_filter(rng.random((H, W)), 6)
    mott = (mott - mott.min()) / (mott.max() - mott.min())
    yellow = np.clip(smoothstep(0.25, 0.5, yy) * 0.6 + 0.4 * mott, 0, 1)
    img = np.stack([np.clip(ves * 1.5, 0, 1), yellow, np.zeros_like(yellow)], -1)
    save_jpg(os.path.join(out, "sclera.jpg"), to_u8(img), 92)


# ----------------------------------------------------------------- lashes
def make_lashes(out, N=2048):
    rng = np.random.default_rng(77)
    scale = 2
    im = Image.new("L", (N * scale, N * scale), 0)
    dr = ImageDraw.Draw(im)
    rows = [(0.848, 0.988), (0.682, 0.822), (0.516, 0.656), (0.346, 0.486), (0.18, 0.32), (0.014, 0.154)]
    cols = [(0.012, 0.49), (0.51, 0.988)]
    for r, (v0, v1) in enumerate(rows):
        upper = r < 3
        for c, (u0, u1) in enumerate(cols):
            count = (72 if upper else 34) - (r % 3) * 8
            for _ in range(count):
                s = rng.random()
                t = 1 - s if c == 0 else s  # 0 inner canthus .. 1 outer canthus
                if rng.random() > smoothstep(0.0, 0.25, t) * (1.0 if upper else smoothstep(0.1, 0.45, t)):
                    continue
                u = u0 + (u1 - u0) * (0.02 + 0.96 * s)
                length = (0.55 + 0.45 * np.sin(np.pi * np.clip(t * 0.85 + 0.12, 0, 1))) * rng.uniform(0.75, 1.0)
                lean = (0.45 * t ** 1.5 + rng.normal(0, 0.06)) * (1 if c == 1 else -1)
                # root at the top edge (v1), tip towards v0
                x = u * N
                y = (1 - v1) * N + 2
                span = (v1 - v0) * N * length * 0.97
                pts = []
                bend = rng.normal(0, 0.15)
                for k in range(12):
                    f = k / 11
                    dx = (lean + bend * f) * span * f * 0.5
                    pts.append(((x + dx) * scale, (y + span * f) * scale))
                root_w = rng.uniform(4.2, 5.6) if upper else rng.uniform(2.6, 3.6)
                shade = int(rng.uniform(200, 255))
                for k in range(11):
                    w = max(1, int(round(root_w * scale * (1 - k / 11) ** 0.8)))
                    dr.line([pts[k], pts[k + 1]], fill=shade, width=w)
    a = np.asarray(im, np.float64).reshape(N, scale, N, scale).mean((1, 3)) / 255.0
    Image.fromarray(to_u8(np.clip(a, 0, 1))).save(os.path.join(out, "lashes.png"), optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets")
    args = ap.parse_args()
    out = os.path.join(args.assets, "tex")
    os.makedirs(out, exist_ok=True)
    make_pores(out)
    make_iris(out)
    make_sclera(out)
    make_lashes(out)
    print("done")


if __name__ == "__main__":
    main()
