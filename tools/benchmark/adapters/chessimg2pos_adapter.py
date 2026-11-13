import os


class Chessimg2posAdapter:
    """This library. Uses the model at $CHESSIMG2POS_MODEL if set, otherwise
    the pretrained model for the installed version (or v0.1.6 as fallback)."""

    name = "chessimg2pos"

    def __init__(self, model_path=None, classifier=None):
        self.model_path = model_path or os.environ.get("CHESSIMG2POS_MODEL")
        self.classifier = classifier
        self.predictor = None

    def setup(self):
        from chessimg2pos import ChessPositionPredictor
        from chessimg2pos.constants import DEFAULT_CLASSIFIER

        if not self.model_path:
            from chessimg2pos.model_loader import download_pretrained_model

            self.model_path = download_pretrained_model(verbose=False)
        self.predictor = ChessPositionPredictor(
            self.model_path, classifier=self.classifier or DEFAULT_CLASSIFIER
        )

    def predict(self, image_path):
        return self.predictor.predict_chessboard(image_path)["fen"]
