from .trainer import ChessRecognitionTrainer
from .predictor import ChessPositionPredictor
from .constants import DEFAULT_CLASSIFIER
from .model_loader import download_pretrained_model
from ._version import __version__


def predict_fen(image_path, output_type="simple", fen_type="compressed"):
    """Predict the FEN piece placement of a chessboard image.

    Args:
        image_path: Path to the image, or a PIL Image.
        output_type: "simple" returns the FEN string; any other value returns
            the full result dict (fen, confidence, per-square predictions).
        fen_type: "compressed" (default) returns standard FEN such as
            "8/8/8/4p1K1/2k1P3/8/8/8". "expanded" returns one character per
            square with "1" for empty squares ("11111111/...").

    Returns:
        str or dict, depending on ``output_type``.
    """
    model_path = download_pretrained_model()
    predictor = ChessPositionPredictor(
        model_path=model_path, classifier=DEFAULT_CLASSIFIER
    )
    result = predictor.predict_chessboard(image_path, fen_type=fen_type)
    if output_type == "simple":
        return result["fen"]
    else:
        return result
