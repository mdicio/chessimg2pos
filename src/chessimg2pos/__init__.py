"""chessimg2pos: chess image to FEN.

    from chessimg2pos import predict_fen
    predict_fen("screenshot.png")        # 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
"""
import warnings

from ._version import __version__
from .v2.weights import load_predictor

__all__ = ["predict_fen", "load_predictor", "__version__"]

_PREDICTORS = {}


def _predictor(weights=None, legality=True):
    key = (weights, legality)
    if key not in _PREDICTORS:  # loading the models once makes repeated calls fast
        _PREDICTORS[key] = load_predictor(weights, legality=legality)
    return _PREDICTORS[key]


def predict_fen(image, output_type="simple", fen_type="compressed", weights=None, legality=True):
    """Recognise the position in a chessboard image.

    Finds the board (also inside a larger screenshot or a phone photo), reads
    all 64 squares and returns the piece placement from White's point of view,
    even when the image shows the board from Black's side.

    Args:
        image: Path (str or os.PathLike) or PIL Image.
        output_type: "simple" returns the FEN string; anything else returns
            the full result dict (fen, fen_as_drawn, orientation, confidence,
            board_corners, ...).
        fen_type: "compressed" (default) gives standard FEN such as
            "8/8/8/4p1K1/2k1P3/8/8/8". "expanded" gives one character per
            square with "1" for empty squares ("11111111/...").
        weights: None to download and cache the release model on first use,
            or a folder with board.onnx/locator.onnx, or a .pt checkpoint.
        legality: Fix impossible boards (e.g. two white kings) on squares the
            model was unsure about.

    Returns:
        str or dict, depending on ``output_type``.
    """
    if fen_type == "standard":
        warnings.warn('fen_type="standard" is deprecated; use "expanded"', DeprecationWarning, stacklevel=2)
        fen_type = "expanded"
    if fen_type not in ("compressed", "expanded"):
        raise ValueError(f'fen_type must be "compressed" or "expanded", not {fen_type!r}')

    result = _predictor(weights, legality).predict(image)
    if fen_type == "expanded":
        from .v2.fen import expand

        result = {**result, "fen": expand(result["fen"]), "fen_as_drawn": expand(result["fen_as_drawn"])}
    return result["fen"] if output_type == "simple" else result
