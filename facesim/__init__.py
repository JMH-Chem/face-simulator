"""face-simulator: synthetic faces for testing face-authentication pipelines.

Generate reproducible, privacy-safe synthetic identities and capture
variations, then measure your own authentication system against them.

Intended use: functional / regression / performance testing of a face-auth
pipeline you are authorized to test. These schematic faces are not designed to
impersonate real individuals or to defeat liveness / anti-spoofing defenses.
"""

from .params import FaceParams, random_identity
from .render import render, Variation
from .dataset import (
    Sample,
    generate_dataset,
    load_manifest,
    enroll_probe_split,
    make_pairs,
    sample_variations,
)
from .harness import (
    AuthBackend,
    EmbeddingBackend,
    Metrics,
    run_verification,
    compute_metrics,
)

__version__ = "0.1.0"

__all__ = [
    "FaceParams",
    "random_identity",
    "render",
    "Variation",
    "Sample",
    "generate_dataset",
    "load_manifest",
    "enroll_probe_split",
    "make_pairs",
    "sample_variations",
    "AuthBackend",
    "EmbeddingBackend",
    "Metrics",
    "run_verification",
    "compute_metrics",
]
