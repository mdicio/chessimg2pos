"""Benchmark chess-image-to-FEN tools on labelled suites.

A suite is a directory with manifest.jsonl, one JSON object per line:

    {"id": "...", "image": "images/ab/abc.png", "fen": "<placement, White's view>",
     "orientation": "white" | "black", ...}

Usage:
    python tools/benchmark/run.py --suites DIR [DIR ...] --tools chessimg2pos tsoj fenshot
        [--limit N] [--out results/]

Writes <out>/<tool>__<suite>.jsonl (every prediction) and prints a table of
exact-board and square accuracy per tool and suite.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapters import ADAPTERS  # noqa: E402
from scoring import score  # noqa: E402


def load_suite(path, limit=None):
    rows = []
    with open(os.path.join(path, "manifest.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                r["image_path"] = os.path.join(path, r["image"])
                rows.append(r)
    rows.sort(key=lambda r: r["id"])  # deterministic subset with --limit
    return rows[:limit] if limit else rows


def run_tool(adapter, rows):
    paths = [r["image_path"] for r in rows]
    t0 = time.time()
    if getattr(adapter, "batch", False):
        preds = adapter.predict_batch(paths)
    else:
        preds = []
        for p in paths:
            try:
                preds.append(adapter.predict(p))
            except Exception as e:  # a crash scores as an empty board
                print(f"  {adapter.name}: error on {p}: {e}", file=sys.stderr)
                preds.append(None)
    return preds, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", required=True)
    ap.add_argument("--tools", nargs="+", default=["chessimg2pos"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="bench_results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    table = {}
    for tool in args.tools:
        adapter = ADAPTERS[tool]()
        adapter.setup()
        for suite in args.suites:
            name = os.path.basename(os.path.normpath(suite))
            rows = load_suite(suite, args.limit)
            preds, secs = run_tool(adapter, rows)
            exact = sq = ori = 0
            with open(os.path.join(args.out, f"{tool}__{name}.jsonl"), "w") as f:
                for r, p in zip(rows, preds):
                    s = score(p, r["fen"], r.get("orientation", "white"))
                    exact += s["exact"]
                    ori += s["orientation_ok"]
                    sq += s["squares"]
                    f.write(json.dumps({"id": r["id"], "truth": r["fen"],
                                        "orientation": r.get("orientation", "white"),
                                        "raw": p, **s}) + "\n")
            n = len(rows)
            table[(tool, name)] = {
                "n": n, "exact": 100 * exact / n, "squares": 100 * sq / (64 * n),
                "orientation": 100 * ori / n, "ms": 1000 * secs / n,
            }
            print(f"{tool:14s} {name:16s} n={n:5d} exact={table[(tool, name)]['exact']:5.1f}% "
                  f"squares={table[(tool, name)]['squares']:5.1f}% "
                  f"oriented={table[(tool, name)]['orientation']:5.1f}% "
                  f"{table[(tool, name)]['ms']:.0f} ms/img", flush=True)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({f"{t}__{s}": v for (t, s), v in table.items()}, f, indent=1)

    suites = sorted({s for _, s in table})
    print("\nExact boards (%)\n")
    print("| tool | " + " | ".join(suites) + " |")
    print("|---" * (len(suites) + 1) + "|")
    for tool in args.tools:
        cells = [f"{table[(tool, s)]['exact']:.1f}" if (tool, s) in table else "-" for s in suites]
        print(f"| {tool} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
