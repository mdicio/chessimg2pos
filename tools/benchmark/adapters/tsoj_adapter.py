"""tsoj/Chess_diagram_to_FEN (MIT). Runs in its own uv environment, in batch.

Setup (once):
    git clone https://github.com/tsoj/Chess_diagram_to_FEN $BENCH_TOOLS/tsoj
    cd $BENCH_TOOLS/tsoj && uv sync --extra cuda && bash download_models.sh
"""
import json
import os
import subprocess
import tempfile

RUNNER = r'''
import json, sys
from PIL import Image
from chess_diagram_to_fen import get_fen
for line in sys.stdin:
    path = json.loads(line)
    try:
        r = get_fen(img=Image.open(path), game="chess",
                    auto_rotate_image=True, auto_rotate_board=True)
        fen = r.fen if r is not None else None
    except Exception as e:
        fen = None
        print("ERR", path, e, file=sys.stderr)
    print("@@FEN " + json.dumps(fen), flush=True)
'''


class TsojAdapter:
    name = "tsoj"
    batch = True

    def __init__(self, root=None):
        tools = os.environ.get("BENCH_TOOLS", os.path.expanduser("~/.cache/chess-bench-tools"))
        self.root = root or os.path.join(tools, "tsoj")

    def setup(self):
        if not os.path.isdir(os.path.join(self.root, "models")):
            raise RuntimeError(f"tsoj not set up at {self.root}; see adapter docstring")

    def predict_batch(self, paths):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(RUNNER)
            runner = f.name
        try:
            inp = "".join(json.dumps(os.path.abspath(p)) + "\n" for p in paths)
            out = subprocess.run(
                ["uv", "run", "--no-sync", "python", runner],
                input=inp, capture_output=True, text=True, cwd=self.root, check=True,
            ).stdout
        finally:
            os.unlink(runner)
        preds = [json.loads(l[6:]) for l in out.splitlines() if l.startswith("@@FEN ")]
        if len(preds) != len(paths):
            raise RuntimeError(f"tsoj returned {len(preds)} results for {len(paths)} images")
        return preds
