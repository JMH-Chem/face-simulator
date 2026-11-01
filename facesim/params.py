"""Identity parameterization for synthetic test faces.

A :class:`FaceParams` is a compact, human-readable description of one synthetic
"identity": face geometry plus colors. Identities are sampled deterministically
from an integer seed so that a test dataset is fully reproducible -- the same
seed always yields the same face.

These faces are intentionally *illustrative / non-photorealistic*. They exist to
exercise a face-authentication pipeline functionally (detect -> enroll -> verify
-> score), not to impersonate any real person.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Tuple
import numpy as np

RGB = Tuple[int, int, int]


# Plausible palettes to keep sampled identities visually distinct.
_SKIN_TONES = [
    (255, 224, 196),
    (241, 194, 165),
    (224, 172, 138),
    (198, 134, 96),
    (161, 102, 70),
    (122, 76, 50),
    (90, 56, 38),
]
_HAIR_COLORS = [
    (30, 24, 20),
    (60, 40, 28),
    (101, 67, 33),
    (150, 110, 60),
    (200, 170, 100),
    (120, 120, 120),
    (210, 210, 210),
    (170, 40, 40),
]
_EYE_COLORS = [
    (72, 52, 38),   # brown
    (40, 30, 24),   # dark brown
    (90, 110, 130),  # blue-grey
    (70, 110, 90),   # green
    (110, 90, 60),   # hazel
]


@dataclass(frozen=True)
class FaceParams:
    """All attributes that define one synthetic identity.

    Geometry fields are expressed as fractions of the image size so a face can
    be rendered at any resolution. Colors are RGB 0-255 tuples.
    """

    # Overall head geometry (fractions of image dimension).
    face_width: float = 0.60
    face_height: float = 0.78
    face_cx: float = 0.50
    face_cy: float = 0.52
    jaw: float = 0.85  # 1.0 = round jaw, lower = narrower chin

    # Skin / complexion.
    skin_tone: RGB = (241, 194, 165)

    # Eyes.
    eye_distance: float = 0.26   # center-to-center, fraction of width
    eye_y: float = 0.46          # vertical position (fraction of image)
    eye_size: float = 0.085      # eye half-width, fraction of width
    eye_openness: float = 0.55   # height/width ratio of the eye
    eye_color: RGB = (72, 52, 38)

    # Eyebrows.
    brow_thickness: float = 0.022
    brow_gap: float = 0.055      # vertical gap above eye
    brow_angle: float = 0.10     # radians, positive = raised outer end
    brow_color: RGB = (60, 40, 28)

    # Nose.
    nose_length: float = 0.14
    nose_width: float = 0.07

    # Mouth.
    mouth_width: float = 0.20
    mouth_y: float = 0.72
    mouth_fullness: float = 0.5
    lip_color: RGB = (190, 110, 100)

    # Hair.
    hair_color: RGB = (60, 40, 28)
    hair_style: int = 1          # 0 short, 1 medium, 2 long, 3 bald
    hairline: float = 0.30       # vertical start of hair (fraction of image)

    # Distinguishing mark (small mole), optional.
    has_mark: bool = False
    mark_pos: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict:
        return asdict(self)


def random_identity(seed: int) -> FaceParams:
    """Deterministically sample a distinct identity from ``seed``."""
    rng = np.random.default_rng(seed)

    def pick(seq):
        return tuple(int(c) for c in seq[rng.integers(len(seq))])

    skin = pick(_SKIN_TONES)
    hair_style = int(rng.integers(0, 4))
    has_mark = bool(rng.random() < 0.3)

    return FaceParams(
        face_width=float(rng.uniform(0.52, 0.66)),
        face_height=float(rng.uniform(0.72, 0.84)),
        face_cx=0.5,
        face_cy=float(rng.uniform(0.50, 0.54)),
        jaw=float(rng.uniform(0.70, 0.98)),
        skin_tone=skin,
        eye_distance=float(rng.uniform(0.22, 0.30)),
        eye_y=float(rng.uniform(0.43, 0.49)),
        eye_size=float(rng.uniform(0.070, 0.100)),
        eye_openness=float(rng.uniform(0.42, 0.68)),
        eye_color=pick(_EYE_COLORS),
        brow_thickness=float(rng.uniform(0.016, 0.030)),
        brow_gap=float(rng.uniform(0.040, 0.075)),
        brow_angle=float(rng.uniform(-0.05, 0.22)),
        brow_color=pick(_HAIR_COLORS),
        nose_length=float(rng.uniform(0.11, 0.17)),
        nose_width=float(rng.uniform(0.055, 0.090)),
        mouth_width=float(rng.uniform(0.16, 0.24)),
        mouth_y=float(rng.uniform(0.69, 0.75)),
        mouth_fullness=float(rng.uniform(0.3, 0.8)),
        lip_color=(
            int(np.clip(skin[0] - rng.integers(10, 60), 0, 255)),
            int(np.clip(skin[1] - rng.integers(40, 90), 0, 255)),
            int(np.clip(skin[2] - rng.integers(20, 70), 0, 255)),
        ),
        hair_color=pick(_HAIR_COLORS),
        hair_style=hair_style,
        hairline=float(rng.uniform(0.26, 0.34)),
        has_mark=has_mark,
        mark_pos=(
            float(rng.uniform(0.35, 0.65)),
            float(rng.uniform(0.58, 0.70)),
        ),
    )
