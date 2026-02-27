"""
tests/test_training.py
-----------------------
End-to-end training test using synthetic 32x32 tile images.
Runs 1 epoch on a tiny dataset so it's fast (< 10 s on CPU).
"""

import os
import tempfile
import unittest

from PIL import Image


FEN_CHARS = "1RNBQKPrnbqkp"
NUM_TILES_PER_CLASS = 4  # very small dataset


def _make_synthetic_tiles(tiles_dir: str):
    """Create minimal PNG tiles that match the expected filename convention."""
    files_letters = "abcdefgh"
    pieces = list(FEN_CHARS)
    idx = 0
    for piece in pieces:
        for i in range(NUM_TILES_PER_CLASS):
            file_letter = files_letters[idx % 8]
            rank = (idx % 8) + 1
            img = Image.new("L", (32, 32), color=(idx * 17) % 256)
            fname = f"{file_letter}{rank}_{piece}.png"
            img.save(os.path.join(tiles_dir, fname))
            idx += 1


class TestTraining(unittest.TestCase):
    def _run_training(self, classifier: str):
        from chessimg2pos.trainer import ChessRecognitionTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            tiles_dir = os.path.join(tmpdir, "tiles")
            os.makedirs(tiles_dir)
            _make_synthetic_tiles(tiles_dir)

            model_path = os.path.join(tmpdir, "model.pt")
            trainer = ChessRecognitionTrainer(
                images_dir=tmpdir,
                tiles_dir=tiles_dir,
                model_path=model_path,
                fen_chars=FEN_CHARS,
                use_grayscale=True,
                epochs=1,
                batch_size=8,
                generate_tiles=False,
                verbose=False,
            )
            model, device, accuracy = trainer.train(classifier=classifier)
            self.assertTrue(os.path.exists(model_path), "Model file was not saved")
            self.assertIsInstance(accuracy, float)
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)

    def test_train_standard(self):
        self._run_training("standard")

    def test_train_enhanced(self):
        self._run_training("enhanced")

    def test_train_ultra(self):
        self._run_training("ultra")

    def test_train_saves_best_weights(self):
        """Saved model must be load-able back into the matching architecture."""
        import torch
        from chessimg2pos.chessclassifier import EnhancedChessPieceClassifier
        from chessimg2pos.trainer import ChessRecognitionTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            tiles_dir = os.path.join(tmpdir, "tiles")
            os.makedirs(tiles_dir)
            _make_synthetic_tiles(tiles_dir)

            model_path = os.path.join(tmpdir, "model.pt")
            trainer = ChessRecognitionTrainer(
                images_dir=tmpdir,
                tiles_dir=tiles_dir,
                model_path=model_path,
                fen_chars=FEN_CHARS,
                use_grayscale=True,
                epochs=1,
                batch_size=8,
                generate_tiles=False,
                verbose=False,
            )
            trainer.train(classifier="enhanced")

            reloaded = EnhancedChessPieceClassifier(
                num_classes=len(FEN_CHARS), use_grayscale=True
            )
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            reloaded.load_state_dict(state)  # must not raise


if __name__ == "__main__":
    unittest.main()
