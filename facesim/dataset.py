"""Build reproducible, labeled datasets of synthetic faces for testing.

A dataset is ``n_identities`` distinct synthetic people, each captured under
``variations_per_identity`` nuisance conditions (pose, lighting, expression,
noise). Images and a JSON manifest are written to disk, and helpers derive the
standard evaluation structures:

* enroll/probe split   -- one reference per identity, the rest are probes;
* verification pairs    -- genuine (same identity) and impostor (different)
  pairs with ground-truth labels, for FAR/FRR / EER measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple
import json
import itertools

import numpy as np

from .params import random_identity
from .render import render, Variation


@dataclass
class Sample:
    identity: int
    variation_index: int
    path: str
    variation: dict


def sample_variations(seed: int, k: int) -> List[Variation]:
    """Return ``k`` capture variations; index 0 is a clean near-frontal shot."""
    rng = np.random.default_rng(seed)
    variations = [
        Variation(  # index 0: clean enrollment reference
            yaw=0.0, pitch=0.0, roll=0.0,
            light_angle=90.0, light_intensity=0.15,
            expression="neutral", noise_std=2.0, blur_radius=0.0,
            scale=1.0,
        )
    ]
    expressions = ["neutral", "neutral", "smile", "surprise"]
    for _ in range(max(0, k - 1)):
        variations.append(Variation(
            yaw=float(rng.uniform(-25, 25)),
            pitch=float(rng.uniform(-15, 15)),
            roll=float(rng.uniform(-12, 12)),
            light_angle=float(rng.uniform(0, 360)),
            light_intensity=float(rng.uniform(0.1, 0.5)),
            expression=expressions[rng.integers(len(expressions))],
            noise_std=float(rng.uniform(0, 8)),
            blur_radius=float(rng.uniform(0, 1.5)),
            scale=float(rng.uniform(0.9, 1.08)),
        ))
    return variations


def generate_dataset(
    out_dir: str | Path,
    n_identities: int = 20,
    variations_per_identity: int = 6,
    size: int = 256,
    seed: int = 0,
) -> List[Sample]:
    """Render a full dataset to ``out_dir`` and write ``manifest.json``."""
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    samples: List[Sample] = []
    for ident in range(n_identities):
        params = random_identity(seed * 100003 + ident)
        variations = sample_variations(seed * 911 + ident, variations_per_identity)
        for vi, var in enumerate(variations):
            img = render(params, size=size, variation=var)
            rel = f"images/id{ident:03d}_v{vi:02d}.png"
            img.save(out / rel)
            samples.append(Sample(ident, vi, rel, asdict(var)))

    manifest = {
        "seed": seed,
        "n_identities": n_identities,
        "variations_per_identity": variations_per_identity,
        "size": size,
        "samples": [asdict(s) for s in samples],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return samples


def load_manifest(out_dir: str | Path) -> List[Sample]:
    out = Path(out_dir)
    data = json.loads((out / "manifest.json").read_text())
    return [Sample(**s) for s in data["samples"]]


def enroll_probe_split(samples: List[Sample]) -> Tuple[List[Sample], List[Sample]]:
    """Variation 0 of each identity is the enrollment reference; rest are probes."""
    enroll = [s for s in samples if s.variation_index == 0]
    probe = [s for s in samples if s.variation_index != 0]
    return enroll, probe


def make_pairs(
    samples: List[Sample],
    max_impostor_per_genuine: int = 1,
    seed: int = 0,
) -> List[Tuple[Sample, Sample, bool]]:
    """Build genuine and impostor verification pairs with ground-truth labels.

    Returns a list of ``(a, b, is_same_identity)``. Genuine pairs are all
    within-identity combinations; impostor pairs are sampled across identities
    to keep the list balanced.
    """
    rng = np.random.default_rng(seed)
    by_id = {}
    for s in samples:
        by_id.setdefault(s.identity, []).append(s)

    genuine: List[Tuple[Sample, Sample, bool]] = []
    for sid, group in by_id.items():
        for a, b in itertools.combinations(group, 2):
            genuine.append((a, b, True))

    ids = list(by_id.keys())
    impostor: List[Tuple[Sample, Sample, bool]] = []
    n_imp = len(genuine) * max_impostor_per_genuine
    for _ in range(n_imp):
        i, j = rng.choice(len(ids), size=2, replace=False)
        a = by_id[ids[i]][rng.integers(len(by_id[ids[i]]))]
        b = by_id[ids[j]][rng.integers(len(by_id[ids[j]]))]
        impostor.append((a, b, False))

    pairs = genuine + impostor
    rng.shuffle(pairs)
    return pairs
