"""fenshot (MIT), end to end: finds the board itself. Runs under Node in batch.

Setup (once):
    git clone https://github.com/scoriiu/fenshot $BENCH_TOOLS/fenshot
    cd $BENCH_TOOLS/fenshot && npm install
"""
import json
import os
import shutil
import subprocess


class FenshotAdapter:
    name = "fenshot"
    batch = True

    def __init__(self, root=None):
        tools = os.environ.get("BENCH_TOOLS", os.path.expanduser("~/.cache/chess-bench-tools"))
        self.root = root or os.path.join(tools, "fenshot")

    def setup(self):
        if not os.path.isdir(os.path.join(self.root, "node_modules")):
            raise RuntimeError(f"fenshot not set up at {self.root}; see adapter docstring")
        # The runner must live inside the fenshot workspace to resolve its packages.
        dest = os.path.join(self.root, ".tmp", "bench_runner.ts")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(os.path.join(os.path.dirname(__file__), "fenshot_runner.ts"), dest)
        self.runner = dest

    def predict_batch(self, paths):
        inp = "".join(json.dumps(os.path.abspath(p)) + "\n" for p in paths)
        out = subprocess.run(
            ["npx", "tsx", self.runner], input=inp, capture_output=True,
            text=True, cwd=self.root, check=True,
        ).stdout
        preds = [json.loads(l[6:]) for l in out.splitlines() if l.startswith("@@FEN ")]
        if len(preds) != len(paths):
            raise RuntimeError(f"fenshot returned {len(preds)} results for {len(paths)} images")
        return preds
