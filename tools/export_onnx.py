"""Export a v2 checkpoint to ONNX (board.onnx + locator.onnx), check it
against PyTorch, and optionally write int8-quantised copies.

    python tools/export_onnx.py runs/v2/final.pt --out export/ [--quantize]
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
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    board, loc = load_checkpoint(args.checkpoint)
    board.eval(); loc.eval()
    # the fused eval fast path of nn.TransformerEncoder does not trace faithfully
    torch.backends.mha.set_fastpath_enabled(False)

    bpath, lpath = os.path.join(args.out, "board.onnx"), os.path.join(args.out, "locator.onnx")
    # smooth, image-like inputs: white noise drives ConvNeXt into ranges where
    # fp32 itself is unstable, which makes the parity check meaningless
    up = torch.nn.functional.interpolate
    xb = up(torch.randn(2, 3, 16, 16), size=BOARD_SIZE, mode="bilinear")
    xl = up(torch.randn(2, 3, 20, 20), size=LOC_SIZE, mode="bilinear")
    torch.onnx.export(_Board(board), (xb,), bpath, input_names=["board"], output_names=["squares", "orientation"],
                      dynamic_axes={"board": {0: "n"}, "squares": {0: "n"}, "orientation": {0: "n"}},
                      opset_version=args.opset, dynamo=False)
    torch.onnx.export(loc, (xl,), lpath, input_names=["image"], output_names=["box"],
                      dynamic_axes={"image": {0: "n"}, "box": {0: "n"}}, opset_version=args.opset, dynamo=False)

    import onnxruntime as ort
    with torch.no_grad():
        ts, to = board(xb)
        tl = loc(xl)
    s, o = ort.InferenceSession(bpath).run(None, {"board": xb.numpy()})
    (l,) = ort.InferenceSession(lpath).run(None, {"image": xl.numpy()})
    print("max abs diff  squares %.2e  orientation %.2e  box %.2e" % (
        np.abs(s - ts.numpy()).max(), np.abs(o - to.numpy()).max(), np.abs(l - tl.numpy()).max()))

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
