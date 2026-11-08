"""Minimal, fast reader for ICT-FaceKit OBJ files (quads, shared topology)."""
import numpy as np


def read_positions(path):
    """Return only the 'v' rows of an OBJ as float64 (N,3)."""
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                rows.append(line[2:])
    return np.loadtxt(rows, dtype=np.float64)


def read_full(path):
    """Parse positions, uvs, and polygon faces with per-face material name."""
    v, vt, faces, mats = [], [], [], []
    cur = None
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                v.append(line[2:])
            elif line.startswith("vt "):
                vt.append(line[3:])
            elif line.startswith("usemtl "):
                cur = line.split()[1]
            elif line.startswith("f "):
                corners = []
                for tok in line.split()[1:]:
                    parts = tok.split("/")
                    vi = int(parts[0]) - 1
                    ti = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1
                    corners.append((vi, ti))
                faces.append(corners)
                mats.append(cur)
    V = np.loadtxt(v, dtype=np.float64)
    T = np.loadtxt(vt, dtype=np.float64)[:, :2]
    return V, T, faces, mats
