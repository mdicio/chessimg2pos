"""Turn network outputs into a result dict (numpy only)."""
import numpy as np

from .fen import flip, from_indices
from .legality import fix


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def decode(square_probs, orient_probs, corners, legality=True):
    if legality:
        idx, changed = fix(square_probs)
    else:
        idx, changed = square_probs.argmax(1), 0
    drawn = from_indices(idx)
    black = orient_probs[1] > 0.5
    return {
        "fen": flip(drawn) if black else drawn,
        "fen_as_drawn": drawn,
        "orientation": "black" if black else "white",
        "orientation_confidence": float(max(orient_probs)),
        # the least certain square: a board is only as right as its worst square
        "confidence": float(square_probs[np.arange(64), idx].min()),
        "board_corners": [[float(x), float(y)] for x, y in corners],
        "legality_changes": changed,
    }
