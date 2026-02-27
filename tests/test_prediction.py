"""
tests/test_prediction.py
-------------------------
Test ChessPositionPredictor using a freshly-trained tiny model and a
synthetic chessboard image — no internet access or real images needed.
"""

import os
import tempfile
import unittest

import torch
from PIL import Image


FEN_CHARS = "1RNBQKPrnbqkp"


def _save_dummy_model(path: str, classifier: str, use_grayscale: bool = True):
    """Save a randomly-initialised model so the predictor can load it."""
    from chessimg2pos.chessclassifier import (
        ChessPieceClassifier,
        EnhancedChessPieceClassifier,
        UltraEnhancedChessPieceClassifier,
    )

    cls = {
        "standard": ChessPieceClassifier,
        "enhanced": EnhancedChessPieceClassifier,
        "ultra": UltraEnhancedChessPieceClassifier,
    }[classifier]
    model = cls(num_classes=len(FEN_CHARS), use_grayscale=use_grayscale)
    torch.save(model.state_dict(), path)


def _make_synthetic_chessboard(path: str, size: int = 256):
    """Save a plain-colour square image that the tile extractor can handle."""
    img = Image.new("RGB", (size, size), color=(200, 180, 160))
    img.save(path)


class TestPrediction(unittest.TestCase):
    def _test_predict(self, classifier: str):
        from chessimg2pos.predictor import ChessPositionPredictor

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pt")
            img_path = os.path.join(tmpdir, "board.png")
            _save_dummy_model(model_path, classifier)
            _make_synthetic_chessboard(img_path)

            predictor = ChessPositionPredictor(
                model_path=model_path,
                classifier=classifier,
                fen_chars=FEN_CHARS,
                use_grayscale=True,
            )
            result = predictor.predict_chessboard(img_path, return_tiles=True)

            self.assertIn("fen", result)
            self.assertIn("confidence", result)
            self.assertIn("predictions", result)
            self.assertIn("tiles", result)

            fen = result["fen"]
            # FEN must have 8 ranks separated by "/"
            ranks = fen.split("/")
            self.assertEqual(len(ranks), 8, f"FEN has wrong number of ranks: {fen}")

            # Confidence must be a float in [0, 1]
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)

    def test_predict_standard(self):
        self._test_predict("standard")

    def test_predict_enhanced(self):
        self._test_predict("enhanced")

    def test_predict_ultra(self):
        self._test_predict("ultra")

    def test_predict_pil_image_input(self):
        """Predictor must also accept a PIL Image directly (not just a path)."""
        from chessimg2pos.predictor import ChessPositionPredictor

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pt")
            _save_dummy_model(model_path, "enhanced")

            img = Image.new("RGB", (256, 256), color=(180, 160, 140))

            predictor = ChessPositionPredictor(
                model_path=model_path, classifier="enhanced", fen_chars=FEN_CHARS
            )
            result = predictor.predict_chessboard(img)
            self.assertIn("fen", result)

    def test_predict_missing_model_raises(self):
        from chessimg2pos.predictor import ChessPositionPredictor

        with self.assertRaises(FileNotFoundError):
            ChessPositionPredictor("/nonexistent/path/model.pt", classifier="enhanced")

    def test_compressed_fen_roundtrip(self):
        """compressed_fen must produce standard FEN numerics."""
        from chessimg2pos.utils import compressed_fen

        expanded = (
            "11111111/11111111/11111111/11111111/11111111/11111111/PPPPPPPP/RNBQKBNR"
        )
        compressed = compressed_fen(expanded)
        self.assertIn("8", compressed)
        self.assertNotIn("11111111", compressed)


if __name__ == "__main__":
    unittest.main()
