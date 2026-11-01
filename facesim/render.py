"""Render a :class:`FaceParams` to an image, optionally with a capture variation.

The renderer is deliberately procedural and schematic (ellipses, arcs). It
produces consistent-per-identity, clearly-synthetic faces -- ideal as test
fixtures for a recognition pipeline, and unsuitable for impersonating a real
person.

A :class:`Variation` models the kind of nuisance factors a real capture adds:
head pose, lighting, expression, sensor noise and blur. Feeding several
variations of one identity into a matcher is how you probe its robustness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .params import FaceParams, RGB


@dataclass(frozen=True)
class Variation:
    """Capture-time nuisance factors applied on top of an identity."""

    yaw: float = 0.0            # degrees, left/right head turn (approx)
    pitch: float = 0.0          # degrees, up/down head tilt (approx)
    roll: float = 0.0           # degrees, in-plane rotation
    light_angle: float = 90.0   # degrees, 0=right, 90=top, 180=left
    light_intensity: float = 0.35  # 0 flat .. 1 strong directional shading
    expression: str = "neutral"    # neutral | smile | surprise
    noise_std: float = 0.0      # gaussian sensor noise (0..~25)
    blur_radius: float = 0.0    # gaussian blur radius in px
    scale: float = 1.0          # distance-to-camera proxy
    bg_color: RGB = (235, 235, 235)


def _shade(color: RGB, factor: float) -> Tuple[int, int, int]:
    return tuple(int(np.clip(c * factor, 0, 255)) for c in color)


def render(
    params: FaceParams,
    size: int = 256,
    variation: Optional[Variation] = None,
) -> Image.Image:
    """Render ``params`` to an ``size`` x ``size`` RGB image."""
    if variation is None:
        variation = Variation()

    W = H = size
    img = Image.new("RGB", (W, H), variation.bg_color)
    draw = ImageDraw.Draw(img)

    cx, cy = params.face_cx * W, params.face_cy * H
    fw, fh = params.face_width * W, params.face_height * H

    # Expression adjustments.
    smile = 1.0 if variation.expression == "smile" else 0.0
    surprise = 1.0 if variation.expression == "surprise" else 0.0

    # --- Hair (drawn behind the face for long styles) ---
    if params.hair_style != 3:  # not bald
        hair_top = params.hairline * H - fh * 0.12
        if params.hair_style >= 1:
            back_h = fh * (0.9 if params.hair_style == 1 else 1.15)
            draw.ellipse(
                [cx - fw * 0.60, hair_top, cx + fw * 0.60, hair_top + back_h],
                fill=params.hair_color,
            )

    # --- Face oval ---
    jaw = params.jaw
    face_box = [cx - fw / 2, cy - fh / 2, cx + fw / 2, cy + fh / 2]
    draw.ellipse(face_box, fill=params.skin_tone)
    # Narrow the chin by overdrawing background triangles when jaw < 1.
    if jaw < 1.0:
        narrow = (1.0 - jaw) * fw * 0.5
        chin_y = cy + fh * 0.5
        draw.polygon(
            [(cx - fw / 2, cy + fh * 0.15),
             (cx - fw / 2 + narrow, chin_y),
             (cx - fw / 2, chin_y)],
            fill=variation.bg_color,
        )
        draw.polygon(
            [(cx + fw / 2, cy + fh * 0.15),
             (cx + fw / 2 - narrow, chin_y),
             (cx + fw / 2, chin_y)],
            fill=variation.bg_color,
        )

    # --- Hair front (fringe) ---
    if params.hair_style != 3:
        top = cy - fh / 2
        fringe_h = fh * (0.30 if params.hair_style == 0 else 0.22)
        draw.chord(
            [cx - fw * 0.52, top - fh * 0.05,
             cx + fw * 0.52, top + fringe_h * 2],
            180, 360, fill=params.hair_color,
        )

    # --- Eyes ---
    eye_dx = params.eye_distance * W / 2
    eye_y = params.eye_y * H
    ew = params.eye_size * W
    eh = ew * params.eye_openness * (1.6 if surprise else 1.0)
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        # sclera
        draw.ellipse([ex - ew, eye_y - eh, ex + ew, eye_y + eh],
                     fill=(250, 250, 248))
        # iris
        ir = eh * 0.95
        draw.ellipse([ex - ir, eye_y - ir, ex + ir, eye_y + ir],
                     fill=params.eye_color)
        # pupil
        pr = ir * 0.45
        draw.ellipse([ex - pr, eye_y - pr, ex + pr, eye_y + pr],
                     fill=(20, 20, 20))
        # catchlight
        draw.ellipse([ex - pr * 0.3 - ir * 0.3, eye_y - ir * 0.3 - pr * 0.3,
                      ex - ir * 0.3 + pr * 0.3, eye_y - ir * 0.3 + pr * 0.3],
                     fill=(255, 255, 255))

        # eyebrow
        bw = ew * 1.3
        by = eye_y - eh - params.brow_gap * H
        ang = params.brow_angle
        bt = params.brow_thickness * H
        x0, x1 = ex - bw, ex + bw
        y0 = by + sign * 0  # base
        # slant: outer end raised by brow_angle
        dy = math.tan(ang) * bw
        draw.line([(x0, by + (dy if sign < 0 else -dy)),
                   (x1, by + (-dy if sign < 0 else dy))],
                  fill=params.brow_color, width=max(2, int(bt)))

    # --- Nose ---
    nl = params.nose_length * H
    nw = params.nose_width * W
    nose_tip_y = eye_y + nl
    draw.line([(cx, eye_y + eh * 0.5), (cx - nw * 0.2, nose_tip_y)],
              fill=_shade(params.skin_tone, 0.82), width=2)
    draw.arc([cx - nw, nose_tip_y - nw * 0.6, cx + nw, nose_tip_y + nw * 0.6],
             200, 340, fill=_shade(params.skin_tone, 0.7), width=2)

    # --- Mouth ---
    mw = params.mouth_width * W
    my = params.mouth_y * H
    curve = (smile * 0.5 - 0.0) * mw * 0.5  # upward for smile
    full = params.mouth_fullness
    if surprise:
        draw.ellipse([cx - mw * 0.35, my - mw * 0.25,
                      cx + mw * 0.35, my + mw * 0.35],
                     fill=_shade(params.lip_color, 0.7))
    else:
        top_y = my - full * mw * 0.12
        draw.polygon(
            [(cx - mw / 2, my),
             (cx, top_y - curve * 0.2),
             (cx + mw / 2, my),
             (cx, my + full * mw * 0.18 + curve)],
            fill=params.lip_color,
        )
        # lip separation line
        draw.line([(cx - mw / 2, my), (cx, my - curve * 0.3),
                   (cx + mw / 2, my)],
                  fill=_shade(params.lip_color, 0.6), width=2)

    # --- Distinguishing mark ---
    if params.has_mark:
        mx = params.mark_pos[0] * W
        mmy = params.mark_pos[1] * H
        draw.ellipse([mx - 3, mmy - 3, mx + 3, mmy + 3],
                     fill=_shade(params.skin_tone, 0.45))

    # Convert to array for lighting / pose / noise.
    arr = np.asarray(img).astype(np.float32)

    arr = _apply_lighting(arr, variation)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    img = _apply_pose(img, variation, bg=variation.bg_color)

    if variation.blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(variation.blur_radius))

    if variation.noise_std > 0:
        a = np.asarray(img).astype(np.float32)
        a += np.random.default_rng().normal(0, variation.noise_std, a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    return img


def _apply_lighting(arr: np.ndarray, v: Variation) -> np.ndarray:
    """Add a directional brightness gradient across the image."""
    if v.light_intensity <= 0:
        return arr
    H, W = arr.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xx = (xx / W) - 0.5
    yy = (yy / H) - 0.5
    rad = math.radians(v.light_angle)
    # Direction the light comes from.
    gx, gy = math.cos(rad), -math.sin(rad)
    grad = xx * gx + yy * gy  # -0.5 .. 0.5
    factor = 1.0 + v.light_intensity * grad * 2.0
    return arr * factor[..., None]


def _apply_pose(img: Image.Image, v: Variation, bg: RGB) -> Image.Image:
    """Approximate yaw/pitch/roll/scale with an affine transform.

    This is a cheap 2D proxy for 3D head pose -- enough to perturb a matcher,
    not a true 3D reprojection.
    """
    if v.yaw == 0 and v.pitch == 0 and v.roll == 0 and v.scale == 1.0:
        return img

    W, H = img.size
    cx, cy = W / 2, H / 2

    # yaw -> horizontal squeeze + shift; pitch -> vertical squeeze + shift.
    sx = v.scale * (1.0 - min(abs(v.yaw) / 90.0, 0.6) * 0.5)
    sy = v.scale * (1.0 - min(abs(v.pitch) / 90.0, 0.6) * 0.5)
    shx = math.sin(math.radians(v.yaw)) * W * 0.12
    shy = math.sin(math.radians(v.pitch)) * H * 0.12
    roll = math.radians(v.roll)
    cos_r, sin_r = math.cos(roll), math.sin(roll)

    # Build inverse affine for PIL (output -> input).
    # Compose: rotate(roll) * scale(sx,sy), then translate by (shx,shy).
    a = cos_r / sx
    b = sin_r / sx
    d = -sin_r / sy
    e = cos_r / sy
    c = cx - (a * cx + b * cy) - shx
    f = cy - (d * cx + e * cy) - shy

    return img.transform(
        (W, H), Image.AFFINE, (a, b, c, d, e, f),
        resample=Image.BILINEAR, fillcolor=bg,
    )
