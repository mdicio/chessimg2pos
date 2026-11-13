import os


class Chessimg2posV2Adapter:
    """chessimg2pos model v2 (locator + whole-board classifier + legality).
    Weights from $CHESSIMG2POS_V2_WEIGHTS."""

    name = "chessimg2pos_v2"
    batch = True

    def __init__(self, weights=None):
        self.weights = weights or os.environ.get("CHESSIMG2POS_V2_WEIGHTS")
        self.pred = None

    def setup(self):
        from chessimg2pos.v2 import PredictorV2

        if not self.weights:
            raise RuntimeError("set CHESSIMG2POS_V2_WEIGHTS to a v2 checkpoint")
        self.pred = PredictorV2(self.weights)

    def predict_batch(self, paths, chunk=64):
        out = []
        for i in range(0, len(paths), chunk):
            out += [r["fen"] for r in self.pred.predict_batch(paths[i:i + chunk])]
        return out

    def predict(self, path):
        return self.pred.predict(path)["fen"]
