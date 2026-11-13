"""Board-level consistency step.

The network classifies all 64 squares jointly, but it can still output
boards no position can have. When it does, and only for squares it was not
already sure about, swap in the next most likely piece:

- at most one white king and one black king
- no pawns on the first or last rank (true for either orientation)
- at most 8 pawns and 16 pieces per side

Squares predicted with probability >= `lock` are never changed, so unusual
but genuine diagrams (e.g. studies with extra queens) survive.
"""
import numpy as np

from .fen import CLASSES

K, k = CLASSES.index("K"), CLASSES.index("k")
P, p = CLASSES.index("P"), CLASSES.index("p")
WHITE = [CLASSES.index(c) for c in "KQRBNP"]
BLACK = [CLASSES.index(c) for c in "kqrbnp"]


def _demote(probs, idx, sq, banned, lock):
    """Move square sq to its best class not in `banned`. Returns False if locked."""
    if probs[sq, idx[sq]] >= lock:
        return False
    order = np.argsort(-probs[sq])
    for c in order:
        if c not in banned:
            idx[sq] = c
            return True
    return False


def fix(probs, lock=0.97):
    """probs: (64, 13) as drawn. Returns (indices, n_changed)."""
    probs = np.asarray(probs, dtype=np.float64)
    idx = probs.argmax(1).copy()
    orig = idx.copy()

    # pawns on the back ranks
    for sq in list(range(8)) + list(range(56, 64)):
        if idx[sq] in (P, p):
            _demote(probs, idx, sq, {P, p}, lock)

    # kings: exactly one each
    for king in (K, k):
        squares = [s for s in range(64) if idx[s] == king]
        if len(squares) > 1:
            squares.sort(key=lambda s: -probs[s, king])
            for s in squares[1:]:
                _demote(probs, idx, s, {king}, lock)
        # a missing king is never invented: teaching diagrams often omit kings

    # counts per side
    for side, pawn in ((WHITE, P), (BLACK, p)):
        pawns = [s for s in range(64) if idx[s] == pawn]
        if len(pawns) > 8:
            pawns.sort(key=lambda s: probs[s, pawn])
            for s in pawns[:len(pawns) - 8]:
                _demote(probs, idx, s, {pawn}, lock)
        pieces = [s for s in range(64) if idx[s] in side]
        if len(pieces) > 16:
            pieces.sort(key=lambda s: probs[s, idx[s]])
            for s in pieces[:len(pieces) - 16]:
                _demote(probs, idx, s, set(side), lock)
    return idx, int((idx != orig).sum())
