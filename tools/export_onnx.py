"""Export a v2 checkpoint to ONNX (board.onnx + locator.onnx), check it
against PyTorch, and optionally write int8-quantised copies.

    python tools/export_onnx.py runs/v2/final.pt --out export/ --check suite_dir/ [--quantize]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from chessimg2pos.v2.predictor import load_checkpoint  # noqa: E402
from chessimg2pos.v2.preprocess import BOARD_SIZE, LOC_SIZE  # noqa: E402


class _Board(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, board):
        return self.m(board)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default="export")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check", help="directory of board images for a PyTorch/ONNX parity check")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    board, loc = load_checkpoint(args.checkpoint)
    board.eval(); loc.eval()
    # the fused eval fast path of nn.TransformerEncoder does not trace faithfully
    torch.backends.mha.set_fastpath_enabled(False)

    bpath, lpath = os.path.join(args.out, "board.onnx"), os.path.join(args.out, "locator.onnx")
    xb = torch.zeros(2, 3, BOARD_SIZE, BOARD_SIZE)
    xl = torch.zeros(2, 3, LOC_SIZE, LOC_SIZE)
    torch.onnx.export(_Board(board), (xb,), bpath, input_names=["board"], output_names=["squares", "orientation"],
                      dynamic_axes={"board": {0: "n"}, "squares": {0: "n"}, "orientation": {0: "n"}},
                      opset_version=args.opset, dynamo=False)
    torch.onnx.export(loc, (xl,), lpath, input_names=["image"], output_names=["box"],
                      dynamic_axes={"image": {0: "n"}, "box": {0: "n"}}, opset_version=args.opset, dynamo=False)

    # parity on real boards (synthetic tensors are a poor test: ConvNeXt on
    # noise reaches ranges where fp32 itself is unstable)
    if args.check:
        import glob
        import onnxruntime as ort
        from chessimg2pos.v2.preprocess import crop_board, letterbox, load, to_array
        paths = sorted(glob.glob(os.path.join(args.check, "**", "*.*"), recursive=True))
        paths = [p for p in paths if p.lower().endswith((".png", ".jpg", ".jpeg"))][:32]
        imgs = [load(p) for p in paths]
        xb = np.stack([to_array(crop_board(i, [0, 0, *i.size])) for i in imgs])
        xl = np.stack([to_array(letterbox(i)[0]) for i in imgs])
        # a fresh copy: the traced module is not a reliable reference
        board, loc = load_checkpoint(args.checkpoint)
        board.eval(); loc.eval()
        with torch.no_grad():
            ts, to = board(torch.from_numpy(xb))
            tl = loc(torch.from_numpy(xl))
        s, o = ort.InferenceSession(bpath).run(None, {"board": xb})
        (l,) = ort.InferenceSession(lpath).run(None, {"image": xl})
        print("%d images: max abs diff squares %.2e orientation %.2e locator %.2e; argmax agree %.4f" % (
            len(paths), np.abs(s - ts.numpy()).max(), np.abs(o - to.numpy()).max(), np.abs(l - tl.numpy()).max(),
            (s.argmax(-1) == ts.numpy().argmax(-1)).mean()))

    if args.quantize:
        # NOTE: int8 flips ~0.1% of squares (~5% of boards); not for release
        from onnxruntime.quantization import QuantType, quantize_dynamic
        for p in (bpath, lpath):
            q = p.replace(".onnx", ".int8.onnx")
            quantize_dynamic(p, q, weight_type=QuantType.QInt8)
    for f in sorted(os.listdir(args.out)):
        print(f"{f:24s} {os.path.getsize(os.path.join(args.out, f)) / 1e6:7.1f} MB")


if __name__ == "__main__":
    main()
