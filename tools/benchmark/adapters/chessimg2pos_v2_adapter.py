import os


class Chessimg2posV2Adapter:
    """chessimg2pos (locator + whole-board classifier + legality). Weights:
    a .pt checkpoint or an ONNX folder from $CHESSIMG2POS_V2_WEIGHTS,
    otherwise the release ONNX weights."""

    name = "chessimg2pos"
    batch = True

    def __init__(self, weights=None):
        self.weights = weights or os.environ.get("CHESSIMG2POS_V2_WEIGHTS")
        self.pred = None

    def setup(self):
        from chessimg2pos import load_predictor

        self.pred = load_predictor(self.weights)

    def predict_batch(self, paths, chunk=64):
        out = []
        for i in range(0, len(paths), chunk):
            out += [r["fen"] for r in self.pred.predict_batch(paths[i:i + chunk])]
        return out

    def predict(self, path):
        return self.pred.predict(path)["fen"]
