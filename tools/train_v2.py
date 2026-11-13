"""Train model v2 (BoardNet + LocatorNet together, from one sample stream).

    python tools/train_v2.py --sampler manifest:/path/to/dataset [--sampler ...]
        --eval /path/to/suite [...] --steps 60000 --out runs/v2

--sampler accepts:
    manifest:DIR              a directory with manifest.jsonl (suite format)
    module.path:Factory       Factory(seed, worker_id) -> object with .sample()
                              (e.g. a private on-the-fly renderer)
"""
import argparse
import importlib
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from chessimg2pos.v2.data import ManifestSampler, StreamDataset  # noqa: E402
from chessimg2pos.v2.fen import flip  # noqa: E402
from chessimg2pos.v2.model import BoardNet, LocatorNet  # noqa: E402
from chessimg2pos.v2.predictor import PredictorV2  # noqa: E402


class Mixed:
    def __init__(self, samplers, weights, seed):
        self.samplers, self.weights, self.rng = samplers, weights, random.Random(seed)

    def sample(self):
        return self.rng.choices(self.samplers, weights=self.weights)[0].sample()


def factory_from(specs):
    def factory(seed, wid):
        out, w = [], []
        for spec in specs:
            spec, _, weight = spec.partition("@")
            if spec.startswith("manifest:"):
                out.append(ManifestSampler(spec[len("manifest:"):], seed * 31 + wid))
            else:
                mod, _, cls = spec.rpartition(":")
                out.append(getattr(importlib.import_module(mod), cls)(seed, wid))
            w.append(float(weight or 1))
        return out[0] if len(out) == 1 else Mixed(out, w, seed * 17 + wid)
    return factory


def evaluate(ck_path, suites, limit=None, locate=True):
    pred = PredictorV2(ck_path, locate=locate)
    res = {}
    for s in suites:
        rows = [json.loads(l) for l in open(os.path.join(s, "manifest.jsonl")) if l.strip()]
        rows.sort(key=lambda r: r["id"])
        rows = rows[:limit] if limit else rows
        ok = ori = 0
        for i in range(0, len(rows), 64):
            chunk = rows[i:i + 64]
            outs = pred.predict_batch([os.path.join(s, r["image"]) for r in chunk])
            for r, o in zip(chunk, outs):
                as_drawn = r["fen"] if r.get("orientation", "white") == "white" else flip(r["fen"])
                ok += o["fen"] in (r["fen"], as_drawn)
                ori += o["fen"] == r["fen"]
        res[os.path.basename(os.path.normpath(s))] = (round(100 * ok / len(rows), 2), round(100 * ori / len(rows), 2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", action="append", required=True)
    ap.add_argument("--eval", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--out", default="runs/v2")
    ap.add_argument("--resume")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = "cuda"

    board = BoardNet(args.backbone).to(dev)
    loc = LocatorNet().to(dev)
    params = [
        {"params": board.features.parameters(), "lr": args.lr * 0.5},
        {"params": [p for n, p in board.named_parameters() if not n.startswith("features.")], "lr": args.lr},
        {"params": loc.parameters(), "lr": args.lr},
    ]
    opt = torch.optim.AdamW(params, weight_decay=0.05)
    base = [g["lr"] for g in opt.param_groups]
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        board.load_state_dict(ck["board"]); loc.load_state_dict(ck["locator"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        step = ck.get("step", 0)

    dl = torch.utils.data.DataLoader(
        StreamDataset(factory_from(args.sampler), seed=args.seed + step), batch_size=args.batch,
        num_workers=args.workers, persistent_workers=True, prefetch_factor=4, pin_memory=True)
    warm = 1500
    t0, seen = time.time(), 0
    stats = {"sq": 0.0, "ori": 0.0, "loc": 0.0, "acc": 0.0, "n": 0}
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    board.train(); loc.train()
    for xb, ysq, yori, xl, ybox in dl:
        if step >= args.steps:
            break
        f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (args.steps - warm)))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * max(f, 0.01)
        xb, ysq, yori = xb.to(dev, non_blocking=True), ysq.to(dev), yori.to(dev)
        xl, ybox = xl.to(dev, non_blocking=True), ybox.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sq, ori = board(xb)
            pbox = loc(xl)
        l_sq = F.cross_entropy(sq.float().reshape(-1, 13), ysq.reshape(-1), label_smoothing=0.02)
        l_ori = F.cross_entropy(ori.float(), yori)
        l_loc = F.smooth_l1_loss(pbox.float(), ybox, beta=0.01)
        loss = l_sq + 0.3 * l_ori + 5.0 * l_loc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(board.parameters()) + list(loc.parameters()), 2.0)
        opt.step()
        step += 1
        seen += len(xb)
        stats["sq"] += l_sq.item(); stats["ori"] += l_ori.item(); stats["loc"] += l_loc.item()
        stats["acc"] += (sq.argmax(-1) == ysq).all(1).float().mean().item(); stats["n"] += 1
        if step % 200 == 0:
            n = stats["n"]
            rec = {"step": step, "sq": stats["sq"] / n, "ori": stats["ori"] / n, "loc": stats["loc"] / n,
                   "board_acc": stats["acc"] / n, "img_s": seen / (time.time() - t0), "lr": opt.param_groups[1]["lr"]}
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            stats = {k: 0.0 for k in stats}; stats["n"] = 0
        if step % args.eval_every == 0 or step == args.steps:
            path = os.path.join(args.out, "last.pt")
            torch.save({"board": board.state_dict(), "locator": loc.state_dict(),
                        "board_config": board.config, "locator_config": loc.config,
                        "opt": opt.state_dict(), "step": step}, path)
            if args.eval:
                board.eval(); loc.eval()
                r = evaluate(path, args.eval, limit=500)
                board.train(); loc.train()
                print("EVAL", step, json.dumps(r), flush=True)
                log.write(json.dumps({"step": step, "eval": r}) + "\n"); log.flush()
    path = os.path.join(args.out, "final.pt")
    torch.save({"board": board.state_dict(), "locator": loc.state_dict(),
                "board_config": board.config, "locator_config": loc.config, "step": step}, path)
    print("saved", path)


if __name__ == "__main__":
    main()
