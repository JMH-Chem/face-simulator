"""Generate a labeled synthetic-face dataset and a contact sheet preview.

Usage:
    python examples/generate_dataset.py --out out --identities 20 --variations 6
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from facesim import generate_dataset, load_manifest


def contact_sheet(out_dir, cols=6, thumb=96):
    samples = load_manifest(out_dir)
    rows = (len(samples) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (255, 255, 255))
    for i, s in enumerate(samples):
        im = Image.open(Path(out_dir) / s.path).resize((thumb, thumb))
        sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    path = Path(out_dir) / "contact_sheet.png"
    sheet.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--identities", type=int, default=20)
    ap.add_argument("--variations", type=int, default=6)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = generate_dataset(
        args.out,
        n_identities=args.identities,
        variations_per_identity=args.variations,
        size=args.size,
        seed=args.seed,
    )
    sheet = contact_sheet(args.out)
    print(f"Generated {len(samples)} images for {args.identities} identities "
          f"in '{args.out}/'.")
    print(f"Manifest:      {args.out}/manifest.json")
    print(f"Contact sheet: {sheet}")


if __name__ == "__main__":
    main()
