"""Training data plumbing for model v2.

A *sampler* is any object with `.sample() -> (PIL.Image, meta)` where meta
has "fen" (White-view placement), "orientation" ("white"/"black") and
"board_box" ([x0, y0, x1, y1] of the 8x8 grid in image pixels). Samplers can
be generators that render boards on the fly, or `ManifestSampler` over a
directory with manifest.jsonl (the benchmark suite format).
"""
import json
import os
import random

import numpy as np
import torch
from PIL import Image

from .fen import flip, to_indices
from .preprocess import crop_board, letterbox, to_array  # noqa: F401

def to_tensor(img):
    return torch.from_numpy(to_array(img))


class ManifestSampler:
    def __init__(self, root, seed=0):
        self.root = root
        with open(os.path.join(root, "manifest.jsonl")) as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        self.rng = random.Random(seed)

    def sample(self):
        r = self.rng.choice(self.rows)
        img = Image.open(os.path.join(self.root, r["image"])).convert("RGB")
        box = r.get("board_box") or [0, 0, img.size[0], img.size[1]]
        return img, {"fen": r["fen"], "orientation": r.get("orientation", "white"), "board_box": box}


def jitter_box(box, rng, amount):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return [x0 + rng.uniform(-amount, amount) * w, y0 + rng.uniform(-amount, amount) * h,
            x1 + rng.uniform(-amount, amount) * w, y1 + rng.uniform(-amount, amount) * h]


def make_example(img, meta, rng, train=True):
    """-> board tensor, square labels (64,), orientation label, locator
    tensor, normalised box (4,)."""
    box = meta["board_box"]
    if train:
        amt = rng.choice([0.0, 0.01, 0.02, 0.035])
        cbox = jitter_box(box, rng, amt)
        resample = rng.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, Image.BOX])
    else:
        cbox, resample = box, Image.BILINEAR
    board = crop_board(img, cbox, resample=resample)
    if train and rng.random() < 0.08:
        board = board.convert("L").convert("RGB")
    drawn = meta["fen"] if meta["orientation"] == "white" else flip(meta["fen"])
    sq = torch.tensor(to_indices(drawn), dtype=torch.long)
    ori = torch.tensor(0 if meta["orientation"] == "white" else 1, dtype=torch.long)

    # locator: random crop around the board sometimes, so it also sees
    # second-pass views (board filling most of the frame)
    limg, lbox = img, list(box)
    if train and rng.random() < 0.3:
        W, H = img.size
        bw, bh = box[2] - box[0], box[3] - box[1]
        m = rng.uniform(0.0, 0.15)
        cx0 = max(0, box[0] - m * bw * rng.uniform(0.3, 1.7))
        cy0 = max(0, box[1] - m * bh * rng.uniform(0.3, 1.7))
        cx1 = min(W, box[2] + m * bw * rng.uniform(0.3, 1.7))
        cy1 = min(H, box[3] + m * bh * rng.uniform(0.3, 1.7))
        if cx1 - cx0 > 32 and cy1 - cy0 > 32:
            limg = img.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
            lbox = [box[0] - int(cx0), box[1] - int(cy0), box[2] - int(cx0), box[3] - int(cy0)]
    lb, side = letterbox(limg)
    nbox = torch.tensor([v / side for v in lbox], dtype=torch.float32)
    return to_tensor(board), sq, ori, to_tensor(lb), nbox


class StreamDataset(torch.utils.data.IterableDataset):
    """Wraps a sampler factory: factory(seed, worker_id) -> sampler."""

    def __init__(self, factory, seed=0):
        self.factory, self.seed = factory, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        sampler = self.factory(self.seed, wid)
        rng = random.Random(self.seed * 7919 + wid)
        while True:
            try:
                img, meta = sampler.sample()
                yield make_example(img, meta, rng)
            except Exception as e:  # never kill a long run on one bad sample
                print("sample error:", repr(e), flush=True)
