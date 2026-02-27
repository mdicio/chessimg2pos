"""
tests/test_architectures.py
----------------------------
Two test suites:

1. TestModelArchitectures
   Forward-pass smoke tests and validation-rejection checks.  No real data
   needed.

2. TestRealDataPerformance
   For each architecture (standard / enhanced / ultra):
     - Uses ChessRecognitionTrainer (board_level_split=True) to train from
       scratch on training_images/tiles/, ensuring no test-board tiles leak
       into the training set.
     - Saves the trained model, loads it via ChessPositionPredictor.
     - Calls predictor.evaluate_board_accuracy() on the held-out test boards
       to measure both tile-level and board-level FEN accuracy.
     - Asserts board-level FEN accuracy >= MIN_FEN_ACCURACY.

   FEN accuracy definition
   -----------------------
   A board is "correct" if the predicted FEN string (expanded, ranks 8->1)
   exactly matches the ground-truth FEN encoded in the directory name
   (rank separator '-' -> '/').  One wrong tile on any board => that board
   is incorrect.

   The suite is auto-skipped when the tiles directory is absent so CI
   environments without the full dataset are unaffected.
"""

import os
import tempfile
import unittest

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths & hyper-parameters
# ---------------------------------------------------------------------------
_TILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "training_images", "tiles")
)
_FEN_CHARS = "1RNBQKPrnbqkp"
_USE_GRAYSCALE = True
_TRAIN_RATIO = 0.8
_EPOCHS = 3               # kept small for a fast test run
_BATCH_SIZE = 64
_SEED = 42
_MIN_FEN_ACCURACY = 0.75  # board-level FEN accuracy threshold


# ---------------------------------------------------------------------------
# Suite 1 - Shape smoke tests + rejection checks (no real data)
# ---------------------------------------------------------------------------


class TestModelArchitectures(unittest.TestCase):
    """Forward-pass smoke tests for all three classifier variants."""

    NUM_CLASSES = 13
    BATCH = 4
    INPUT_GRAYSCALE = torch.randn(BATCH, 1, 32, 32)
    INPUT_RGB = torch.randn(BATCH, 3, 32, 32)

    def _check_output(self, model, x):
        model.eval()
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (self.BATCH, self.NUM_CLASSES))

    def test_standard_grayscale(self):
        from chessimg2pos.chessclassifier import ChessPieceClassifier
        m = ChessPieceClassifier(num_classes=self.NUM_CLASSES, use_grayscale=True)
        self._check_output(m, self.INPUT_GRAYSCALE)

    def test_standard_rgb(self):
        from chessimg2pos.chessclassifier import ChessPieceClassifier
        m = ChessPieceClassifier(num_classes=self.NUM_CLASSES, use_grayscale=False)
        self._check_output(m, self.INPUT_RGB)

    def test_enhanced_grayscale(self):
        from chessimg2pos.chessclassifier import EnhancedChessPieceClassifier
        m = EnhancedChessPieceClassifier(num_classes=self.NUM_CLASSES, use_grayscale=True)
        self._check_output(m, self.INPUT_GRAYSCALE)

    def test_enhanced_rgb(self):
        from chessimg2pos.chessclassifier import EnhancedChessPieceClassifier
        m = EnhancedChessPieceClassifier(num_classes=self.NUM_CLASSES, use_grayscale=False)
        self._check_output(m, self.INPUT_RGB)

    def test_ultra_grayscale(self):
        from chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier
        m = UltraEnhancedChessPieceClassifier(num_classes=self.NUM_CLASSES, use_grayscale=True)
        self._check_output(m, self.INPUT_GRAYSCALE)

    def test_ultra_rgb(self):
        from chessimg2pos.chessclassifier import UltraEnhancedChessPieceClassifier
        m = UltraEnhancedChessPieceClassifier(num_classes=self.NUM_CLASSES, use_grayscale=False)
        self._check_output(m, self.INPUT_RGB)

    def test_predictor_rejects_unknown_classifier(self):
        """ChessPositionPredictor must raise ValueError for unknown classifier names."""
        from chessimg2pos.chessclassifier import ChessPieceClassifier
        from chessimg2pos.predictor import ChessPositionPredictor

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = f.name
        try:
            m = ChessPieceClassifier(num_classes=13, use_grayscale=True)
            torch.save(m.state_dict(), tmp)
            with self.assertRaises(ValueError):
                ChessPositionPredictor(tmp, classifier="nonexistent")
        finally:
            os.unlink(tmp)

    def test_trainer_rejects_unknown_classifier(self):
        """ChessRecognitionTrainer.train() must raise ValueError for unknown classifier names."""
        from chessimg2pos.trainer import ChessRecognitionTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            tile_dir = os.path.join(tmpdir, "tiles")
            os.makedirs(tile_dir)
            for ch in _FEN_CHARS:
                img = Image.new("L", (32, 32), color=128)
                img.save(os.path.join(tile_dir, f"a1_{ch}.png"))
            trainer = ChessRecognitionTrainer(
                images_dir=tmpdir,
                tiles_dir=tile_dir,
                model_path=os.path.join(tmpdir, "m.pt"),
                epochs=1,
                generate_tiles=False,
            )
            with self.assertRaises(ValueError):
                trainer.train(classifier="nonexistent")


# ---------------------------------------------------------------------------
# Suite 2 - Real-data performance: ChessRecognitionTrainer + ChessPositionPredictor
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    os.path.isdir(_TILES_DIR),
    f"Real tile data not found at {_TILES_DIR!r} - skipping performance tests",
)
class TestRealDataPerformance(unittest.TestCase):
    """
    Train each architecture with ChessRecognitionTrainer (board_level_split=True)
    and evaluate with ChessPositionPredictor.evaluate_board_accuracy().

    setUpClass does all training once and caches results;
    individual test_ methods just assert the thresholds.
    """

    _results: dict = {}
    _tmpdir: tempfile.TemporaryDirectory = None

    @classmethod
    def setUpClass(cls):
        from chessimg2pos.trainer import ChessRecognitionTrainer
        from chessimg2pos.predictor import ChessPositionPredictor

        cls._tmpdir = tempfile.TemporaryDirectory()
        tmpdir = cls._tmpdir.name

        cls._results = {}
        for classifier in ("standard", "enhanced", "ultra"):
            model_path = os.path.join(tmpdir, f"model_{classifier}.pt")

            print(f"\n[TestRealDataPerformance] Training {classifier} ...", flush=True)

            trainer = ChessRecognitionTrainer(
                images_dir=tmpdir,        # not used (generate_tiles=False)
                tiles_dir=_TILES_DIR,
                model_path=model_path,
                fen_chars=_FEN_CHARS,
                use_grayscale=_USE_GRAYSCALE,
                train_test_ratio=_TRAIN_RATIO,
                batch_size=_BATCH_SIZE,
                epochs=_EPOCHS,
                seed=_SEED,
                generate_tiles=False,
                board_level_split=True,
                verbose=False,
            )
            model, device, tile_acc = trainer.train(classifier=classifier)

            print(
                f"[TestRealDataPerformance] {classifier}: "
                f"trainer tile_acc={tile_acc:.4f}",
                flush=True,
            )

            # Load the saved model via ChessPositionPredictor — exercises the
            # full public API: model loading, predict_tile, board reconstruction.
            predictor = ChessPositionPredictor(
                model_path=model_path,
                classifier=classifier,
                fen_chars=_FEN_CHARS,
                use_grayscale=_USE_GRAYSCALE,
            )

            # trainer.test_board_dirs: list of absolute paths to test board dirs
            results = predictor.evaluate_board_accuracy(trainer.test_board_dirs)
            cls._results[classifier] = results

            print(
                f"[TestRealDataPerformance] {classifier}: "
                f"board_accuracy={results['board_accuracy']:.4f}  "
                f"tile_accuracy={results['tile_accuracy']:.4f}  "
                f"boards={results['correct_boards']}/{results['total_boards']}  "
                f"(min={_MIN_FEN_ACCURACY})",
                flush=True,
            )

    @classmethod
    def tearDownClass(cls):
        if cls._tmpdir is not None:
            cls._tmpdir.cleanup()

    # --- individual assertions ---

    def test_standard_fen_accuracy(self):
        """Standard classifier must reach _MIN_FEN_ACCURACY on held-out boards."""
        r = self._results["standard"]
        self.assertGreaterEqual(
            r["board_accuracy"], _MIN_FEN_ACCURACY,
            f"standard board FEN accuracy {r['board_accuracy']:.4f} < {_MIN_FEN_ACCURACY}",
        )

    def test_enhanced_fen_accuracy(self):
        """Enhanced classifier must reach _MIN_FEN_ACCURACY on held-out boards."""
        r = self._results["enhanced"]
        self.assertGreaterEqual(
            r["board_accuracy"], _MIN_FEN_ACCURACY,
            f"enhanced board FEN accuracy {r['board_accuracy']:.4f} < {_MIN_FEN_ACCURACY}",
        )

    def test_ultra_fen_accuracy(self):
        """Ultra classifier must reach _MIN_FEN_ACCURACY on held-out boards."""
        r = self._results["ultra"]
        self.assertGreaterEqual(
            r["board_accuracy"], _MIN_FEN_ACCURACY,
            f"ultra board FEN accuracy {r['board_accuracy']:.4f} < {_MIN_FEN_ACCURACY}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
