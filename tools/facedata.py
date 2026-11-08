"""Load assets/face.json + face.bin.gz back into numpy (used by texture tools)."""
import gzip
import json
import os

import numpy as np


class FaceData:
    def __init__(self, assets="assets"):
        self.meta = json.load(open(os.path.join(assets, "face.json")))
        self.blob = gzip.open(os.path.join(assets, "face.bin.gz")).read()
        m = self.meta
        self.V = self.arr(m["neutral"]).reshape(-1, 3).astype(np.float64)
        self.Q = m["quantum"]

    def arr(self, d):
        return np.frombuffer(self.blob, dtype=d["type"], count=d["length"], offset=d["offset"])

    def part(self, name):
        p = self.meta["parts"][name]
        orig = self.arr(p["origIndex"]).astype(np.int64)
        uv = self.arr(p["uv"]).reshape(-1, 2).astype(np.float64)
        idx = self.arr(p["index"]).reshape(-1, 3).astype(np.int64)
        groups = {g["name"]: idx[g["start"] // 3:(g["start"] + g["count"]) // 3] for g in p["groups"]}
        return orig, uv, idx, groups

    def decode(self, d):
        q = np.cumsum(self.arr(d).reshape(-1, 3).astype(np.int64), axis=0)
        return q * self.Q

    def identity(self, k):
        return self.decode(self.meta["identity"][k]["data"])

    def expression(self, name):
        e = next(e for e in self.meta["expressions"] if e["name"] == name)
        idx = np.cumsum(self.arr(e["index"]).astype(np.int64))
        out = np.zeros_like(self.V)
        out[idx] = self.decode(e["delta"])
        return out
