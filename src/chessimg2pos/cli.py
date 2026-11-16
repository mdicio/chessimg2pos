"""Command line: chessimg2pos IMAGE [IMAGE ...] [--json] [--weights PATH]

Prints one FEN per image (standard placement from White's view).
"""
import argparse
import json
import sys

from ._version import __version__


def main(argv=None):
    ap = argparse.ArgumentParser(prog="chessimg2pos", description="Chess image to FEN")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--json", action="store_true", help="print full results as JSON lines")
    ap.add_argument("--weights", help="v2 checkpoint (.pt) or directory with board.onnx/locator.onnx")
    ap.add_argument("--no-legality", action="store_true")
    ap.add_argument("--lichess", action="store_true", help="also print a lichess analysis URL")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)

    from .v2.weights import load_predictor
    try:
        pred = load_predictor(args.weights, legality=not args.no_legality)
    except Exception as e:  # missing onnxruntime, failed download, bad path
        print(f"chessimg2pos: error: {e}", file=sys.stderr)
        return 1
    failed = 0
    for path in args.images:
        try:
            r = pred.predict(path)
        except Exception as e:  # keep going over a list of files
            print(f"{path}: error: {e}", file=sys.stderr)
            failed += 1
            continue
        if args.json:
            print(json.dumps({"image": path, **r}))
        else:
            line = r["fen"] if len(args.images) == 1 else f"{path}\t{r['fen']}"
            if args.lichess:
                line += "\thttps://lichess.org/analysis/" + r["fen"].replace(" ", "_")
            print(line)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
