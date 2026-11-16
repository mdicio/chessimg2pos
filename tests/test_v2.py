"""Model v2 pieces that need no weights: FEN helpers, legality, pre/post-processing."""
import os

import numpy as np
import pytest
from PIL import Image

from chessimg2pos.v2.fen import CLASSES, compress, expand, flip, from_indices, to_indices
from chessimg2pos.v2.legality import fix
from chessimg2pos.v2.postprocess import decode
from chessimg2pos.v2.preprocess import (board_input, box_to_corners, crop_board, is_axis_aligned, letterbox,
                                        refine_crop, to_array, warp_board)

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
    r = decode(onehot(drawn), np.array([0.1, 0.9]), [[0, 0], [8, 0], [8, 8], [0, 8]])
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


def test_perspective_warp_matches_crop():
    # a 2x2 checker image: warping its exact corners must reproduce it
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    img.paste((0, 0, 0), (100, 0, 200, 100))
    img.paste((0, 0, 0), (0, 100, 100, 200))
    q = box_to_corners([0, 0, 200, 200])
    assert is_axis_aligned(q)
    w = warp_board(img, q, size=64)
    assert w.getpixel((10, 10)) == (255, 255, 255) and w.getpixel((50, 10)) == (0, 0, 0)
    skew = [[0, 0], [200, 10], [200, 200], [0, 190]]
    assert not is_axis_aligned(skew)
    assert board_input(img, skew).size == (256, 256)


def test_load_accepts_pathlike_and_applies_exif(tmp_path):
    from pathlib import Path

    from chessimg2pos.v2.preprocess import load

    img = Image.new("RGB", (40, 20), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 CW on display
    p = tmp_path / "phone.jpg"
    img.save(p, exif=exif)
    for src in (p, str(p)):
        out = load(src)
        assert out.mode == "RGB" and out.size == (20, 40)
    assert load(Image.new("L", (5, 7))).size == (5, 7)
    assert isinstance(p, Path)


def test_cli_reports_errors(tmp_path):
    from chessimg2pos.cli import main

    assert main(["x.png", "--weights", str(tmp_path / "missing.pt")]) == 1


ROOT = os.path.join(os.path.dirname(__file__), "..")
WEIGHTS = os.environ.get("CHESSIMG2POS_V2_WEIGHTS") or os.path.join(ROOT, "web", "models")


@pytest.mark.skipif(not os.path.exists(os.path.join(WEIGHTS, "board.onnx")), reason="no local v2 ONNX weights")
def test_predict_fen_with_real_weights():
    from chessimg2pos import predict_fen

    img = os.path.join(ROOT, "images", "chess_image.png")
    assert predict_fen(img, weights=WEIGHTS) == "8/8/8/4p1K1/2k1P3/8/8/8"
    r = predict_fen(img, output_type="full", weights=WEIGHTS)
    assert r["orientation"] == "white" and r["confidence"] > 0.9
