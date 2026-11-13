"""Model v2 pieces that need no weights: FEN helpers, legality, pre/post-processing."""
import numpy as np
from PIL import Image

from chessimg2pos.v2.fen import CLASSES, compress, expand, flip, from_indices, to_indices
from chessimg2pos.v2.legality import fix
from chessimg2pos.v2.postprocess import decode
from chessimg2pos.v2.preprocess import crop_board, letterbox, refine_crop, to_array

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def onehot(placement, p=0.99):
    idx = to_indices(placement)
    probs = np.full((64, 13), (1 - p) / 12)
    probs[np.arange(64), idx] = p
    return probs


def test_fen_roundtrip():
    assert from_indices(to_indices(START)) == START
    assert flip(flip(START)) == START
    assert flip("8/8/8/8/8/8/8/K7") == "7K/8/8/8/8/8/8/8"
    assert compress(expand("3k4/8/8/8/8/8/8/4K3")) == "3k4/8/8/8/8/8/8/4K3"


def test_legality_keeps_confident_boards():
    idx, changed = fix(onehot(START))
    assert from_indices(idx) == START and changed == 0
    # a confident illegal board (extra kings) is left alone
    odd = "KKKKk3/8/8/8/8/8/8/8"
    idx, changed = fix(onehot(odd, 0.995))
    assert from_indices(idx) == odd and changed == 0


def test_legality_fixes_uncertain_squares():
    probs = onehot("4k3/8/8/8/8/8/8/4K3")
    # square a1 (index 56): unsure between pawn and rook -> pawn on the back rank
    probs[56] = 0.0
    probs[56, CLASSES.index("P")] = 0.55
    probs[56, CLASSES.index("R")] = 0.45
    # square h8 (index 7): unsure second white king vs queen
    probs[7] = 0.0
    probs[7, CLASSES.index("K")] = 0.6
    probs[7, CLASSES.index("Q")] = 0.4
    idx, changed = fix(probs)
    assert from_indices(idx) == "4k2Q/8/8/8/8/8/8/R3K3"
    assert changed == 2


def test_decode_orientation():
    drawn = flip(START)
    r = decode(onehot(drawn), np.array([0.1, 0.9]), [0, 0, 8, 8])
    assert r["fen"] == START and r["orientation"] == "black" and r["fen_as_drawn"] == drawn


def test_preprocess_shapes():
    img = Image.new("RGB", (300, 200), (200, 100, 50))
    lb, side = letterbox(img)
    assert lb.size == (320, 320) and side == 300
    assert to_array(lb).shape == (3, 320, 320)
    b = crop_board(img, [-10, -10, 150, 150])
    assert b.size == (256, 256) and b.getpixel((0, 0)) == (0, 0, 0)
    crop, (ox, oy) = refine_crop(img, [50, 50, 150, 150])
    assert (ox, oy) == (40, 40) and crop.size == (120, 120)
