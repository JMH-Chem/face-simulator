"""Run a verification benchmark against a face-auth backend.

By default this uses the built-in reference ``EmbeddingBackend`` (a demo
baseline, not a real recognizer) so you can see the full pipeline work. To test
YOUR system, subclass ``AuthBackend`` (see ``MyBackend`` below), call your
enroll/verify API inside it, and pass it to ``run_verification``.

Usage:
    python examples/run_benchmark.py --out out
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from facesim import (
    load_manifest,
    make_pairs,
    run_verification,
    AuthBackend,
    EmbeddingBackend,
)


class MyBackend(AuthBackend):
    """Template for wiring in your own face-authentication system.

    Replace the body of ``verify`` with a call to your matcher and return a
    similarity score (higher == more likely the same person). If your system is
    embedding-based, override ``embed`` instead.
    """

    def verify(self, image_a: Image.Image, image_b: Image.Image) -> float:
        # score = my_face_api.compare(image_a, image_b)
        # return float(score)
        raise NotImplementedError("Wire in your face-auth API here.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--backend", choices=["demo", "mine"], default="demo")
    ap.add_argument("--impostors-per-genuine", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = load_manifest(args.out)
    pairs = make_pairs(samples, args.impostors_per_genuine, seed=args.seed)

    backend = EmbeddingBackend() if args.backend == "demo" else MyBackend()
    metrics = run_verification(backend, pairs, root=args.out)

    print(f"Backend: {args.backend}")
    print(metrics.summary())

    # A few operating points along the curve.
    print("\nOperating points:")
    lo, hi = metrics.scores.min(), metrics.scores.max()
    for frac in (0.2, 0.4, 0.5, 0.6, 0.8):
        t = lo + (hi - lo) * frac
        far, frr = metrics.far_frr_at(t)
        print(f"  threshold={t:+.3f}  FAR={far*100:6.2f}%  FRR={frr*100:6.2f}%")


if __name__ == "__main__":
    main()
