"""Test harness: plug your face-authentication system in and measure it.

The harness never decides whether two faces match -- your system does. You
implement :class:`AuthBackend` by calling your enrollment / verification API,
the harness feeds it the labeled synthetic pairs, and :func:`compute_metrics`
turns the returned scores into the standard biometric error rates (FAR, FRR,
EER) plus ROC AUC and accuracy.

A reference :class:`EmbeddingBackend` is included so the whole pipeline runs
end to end out of the box. It is a trivial image-similarity baseline for
demonstration only -- replace it with your real matcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Sequence
import abc

import numpy as np
from PIL import Image

from .dataset import Sample


class AuthBackend(abc.ABC):
    """Adapter around the face-auth system under test.

    Override :meth:`verify` to return a similarity score (higher == more likely
    the same identity). If your system is embedding-based, override
    :meth:`embed` instead and the default :meth:`verify` will cosine-compare.
    """

    def embed(self, image: Image.Image) -> np.ndarray:  # optional
        raise NotImplementedError

    def verify(self, image_a: Image.Image, image_b: Image.Image) -> float:
        va = self.embed(image_a)
        vb = self.embed(image_b)
        denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
        return float(np.dot(va, vb) / denom)


class EmbeddingBackend(AuthBackend):
    """Reference baseline matcher (demo only, NOT a real face recognizer)."""

    def __init__(self, grid: int = 28):
        self.grid = grid

    def embed(self, image: Image.Image) -> np.ndarray:
        # Chromaticity (color / brightness) is largely invariant to the global
        # intensity scaling our lighting variation applies, and a histogram of
        # it is pose-invariant -- so an identity's skin/hair/eye palette drives
        # the match. A low-weight coarse grayscale term adds crude geometry.
        rgb = np.asarray(image.convert("RGB").resize((self.grid, self.grid)))
        rgb = rgb.astype(np.float32) + 1.0
        chroma = rgb / rgb.sum(axis=2, keepdims=True)
        rc, gc = chroma[..., 0].ravel(), chroma[..., 1].ravel()
        hr, _ = np.histogram(rc, bins=16, range=(0.25, 0.50), density=True)
        hg, _ = np.histogram(gc, bins=16, range=(0.25, 0.42), density=True)
        gray = np.asarray(image.convert("L").resize((10, 10))).astype(np.float32)
        gray = (gray - gray.mean()) / (gray.std() + 1e-6)
        return np.concatenate([
            2.0 * hr,
            2.0 * hg,
            0.4 * gray.ravel(),
            6.0 * chroma.reshape(-1, 3).mean(axis=0),
        ])


def _load(sample: Sample, root: str | Path) -> Image.Image:
    return Image.open(Path(root) / sample.path).convert("RGB")


@dataclass
class Metrics:
    eer: float
    eer_threshold: float
    auc: float
    n_genuine: int
    n_impostor: int
    scores: np.ndarray
    labels: np.ndarray

    def far_frr_at(self, threshold: float) -> Tuple[float, float]:
        """False-accept and false-reject rate at a decision ``threshold``."""
        accept = self.scores >= threshold
        gen = self.labels == 1
        imp = self.labels == 0
        frr = float(np.mean(~accept[gen])) if gen.any() else 0.0
        far = float(np.mean(accept[imp])) if imp.any() else 0.0
        return far, frr

    def accuracy_at(self, threshold: float) -> float:
        pred = (self.scores >= threshold).astype(int)
        return float(np.mean(pred == self.labels))

    def summary(self) -> str:
        far_e, frr_e = self.far_frr_at(self.eer_threshold)
        return (
            f"pairs: {self.n_genuine} genuine / {self.n_impostor} impostor\n"
            f"EER:   {self.eer*100:.2f}%  (threshold={self.eer_threshold:.4f})\n"
            f"AUC:   {self.auc:.4f}\n"
            f"@EER threshold -> FAR={far_e*100:.2f}%  FRR={frr_e*100:.2f}%  "
            f"acc={self.accuracy_at(self.eer_threshold)*100:.2f}%"
        )


def run_verification(
    backend: AuthBackend,
    pairs: Sequence[Tuple[Sample, Sample, bool]],
    root: str | Path,
) -> Metrics:
    """Score every pair with ``backend`` and compute error metrics."""
    scores = np.empty(len(pairs), dtype=np.float64)
    labels = np.empty(len(pairs), dtype=np.int64)
    for i, (a, b, same) in enumerate(pairs):
        scores[i] = backend.verify(_load(a, root), _load(b, root))
        labels[i] = 1 if same else 0
    return compute_metrics(scores, labels)


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> Metrics:
    """Compute EER, ROC AUC and pair counts from scores + {0,1} labels."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    thresholds = np.unique(scores)
    thresholds = np.concatenate([[thresholds.min() - 1e-6], thresholds,
                                 [thresholds.max() + 1e-6]])
    gen = labels == 1
    imp = labels == 0

    best_gap = np.inf
    eer = 0.0
    eer_t = thresholds[0]
    for t in thresholds:
        accept = scores >= t
        frr = np.mean(~accept[gen]) if gen.any() else 0.0
        far = np.mean(accept[imp]) if imp.any() else 0.0
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            eer = (far + frr) / 2.0
            eer_t = float(t)

    auc = _roc_auc(scores, gen, imp)

    return Metrics(
        eer=float(eer),
        eer_threshold=eer_t,
        auc=auc,
        n_genuine=int(gen.sum()),
        n_impostor=int(imp.sum()),
        scores=scores,
        labels=labels,
    )


def _roc_auc(scores: np.ndarray, gen: np.ndarray, imp: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties count as 0.5).

    Equivalent to P(score(genuine) > score(impostor)); robust to tied scores,
    unlike a naive trapezoid over an unsorted ROC.
    """
    g = scores[gen]
    i = scores[imp]
    if g.size == 0 or i.size == 0:
        return 0.0
    diff = g[:, None] - i[None, :]
    wins = np.count_nonzero(diff > 0)
    ties = np.count_nonzero(diff == 0)
    return float((wins + 0.5 * ties) / (g.size * i.size))
