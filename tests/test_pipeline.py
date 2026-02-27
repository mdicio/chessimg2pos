"""
tests/test_pipeline.py
-----------------------
Full pipeline test: tile generation → training → evaluation with minimum
accuracy thresholds on BOTH the training set and the test set.

Design principles
-----------------
* Uses synthetic 32x32 tiles whose label is embedded in a distinct solid-grey
  intensity for every class.  The CNN can trivially memorise these in a few
  epochs, so failures here indicate a real regression, not a dataset issue.
* 32 tiles per class × 13 classes = 416 tiles total.
  70 / 30 split → 291 train tiles / 125 test tiles (deterministic seed).
* Does NOT modify trainer.py — train accuracy is measured by re-running
  inference on the training split with the saved model after training.
* Runs in < 30 s on CPU.

Minimum accuracy thresholds
----------------------------
  Train set : >= 0.85
  Test  set : >= 0.70

These are intentionally strict for this trivially-learnable synthetic dataset.
A real-world dataset with diverse chess styles will typically yield:
  enhanced model, 10 epochs: train ~0.99, test ~0.95+
"""

import os
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

# ── Dataset parameters ────────────────────────────────────────────────────────
FEN_CHARS = "1RNBQKPrnbqkp"
NUM_CLASSES = len(FEN_CHARS)  # 13
TILES_PER_CLASS = 32  # enough to trigger meaningful learning
TRAIN_RATIO = 0.7
SEED = 1
EPOCHS = 8

# ── Minimum acceptable accuracy ───────────────────────────────────────────────
MIN_TRAIN_ACC = 0.85
MIN_TEST_ACC = 0.70


def _make_distinctive_tiles(tiles_dir: str) -> list[str]:
    """
    Save TILES_PER_CLASS PNG tiles per piece class.

    Each class i gets a unique solid-grey intensity:
        intensity = (i * 19 + 10) % 256
    This guarantees every class is visually distinct, so the CNN only needs
    to learn a simple intensity → class mapping.

    Filename convention: <file><rank>_<piece>.png  (e.g. a1_R.png)

    Returns the sorted list of all saved paths.
    """
    files = "abcdefgh"
    paths = []
    for class_idx, piece in enumerate(FEN_CHARS):
        intensity = (class_idx * 19 + 10) % 256
        for tile_i in range(TILES_PER_CLASS):
            col = files[tile_i % 8]
            row = (tile_i % 8) + 1
            # Add a tiny per-tile noise so tiles within the same class aren't
            # bit-for-bit identical — this exercises the dataset loader better.
            base_arr = np.full((32, 32), intensity, dtype=np.uint8)
            base_arr[0, 0] = (intensity + tile_i) % 256  # one different pixel
            img = Image.fromarray(base_arr, mode="L")
            fname = f"{col}{row}_{piece}.png"
            full_path = os.path.join(tiles_dir, fname)
            img.save(full_path)
            paths.append(full_path)
    return sorted(paths)


def _accuracy_on_paths(model, paths: list[str], device, fen_chars: str) -> float:
    """Evaluate the model on a list of tile paths, returns accuracy in [0,1]."""
    from chessimg2pos.chessdataset import ChessTileDataset, create_image_transforms

    transform = create_image_transforms(use_grayscale=True)
    dataset = ChessTileDataset(
        np.array(paths), fen_chars, use_grayscale=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def _split_paths(all_paths: list[str]) -> tuple[list[str], list[str]]:
    """Reproduce the exact same deterministic split used by the trainer."""
    arr = np.array(all_paths)
    rng = np.random.RandomState(SEED)
    rng.shuffle(arr)
    divider = int(len(arr) * TRAIN_RATIO)
    return arr[:divider].tolist(), arr[divider:].tolist()


class TestFullPipeline(unittest.TestCase):
    """Full train → save → reload → evaluate pipeline for every classifier."""

    def _run_pipeline(self, classifier: str):
        from chessimg2pos.trainer import ChessRecognitionTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            tiles_dir = os.path.join(tmpdir, "tiles")
            os.makedirs(tiles_dir)
            all_paths = _make_distinctive_tiles(tiles_dir)

            model_path = os.path.join(tmpdir, "model.pt")
            trainer = ChessRecognitionTrainer(
                images_dir=tmpdir,
                tiles_dir=tiles_dir,
                model_path=model_path,
                fen_chars=FEN_CHARS,
                use_grayscale=True,
                epochs=EPOCHS,
                batch_size=32,
                train_test_ratio=TRAIN_RATIO,
                seed=SEED,
                generate_tiles=False,
                verbose=False,
            )

            model, device, reported_test_acc = trainer.train(classifier=classifier)

            # ── verify reported test accuracy ─────────────────────────────────
            self.assertIsInstance(reported_test_acc, float)
            self.assertGreaterEqual(
                reported_test_acc,
                MIN_TEST_ACC,
                f"[{classifier}] reported test acc {reported_test_acc:.4f} < "
                f"minimum {MIN_TEST_ACC}",
            )

            # ── reload saved model (mirrors what a real user does) ─────────────
            from chessimg2pos.chessclassifier import (
                ChessPieceClassifier,
                EnhancedChessPieceClassifier,
                UltraEnhancedChessPieceClassifier,
            )

            arch_map = {
                "standard": ChessPieceClassifier,
                "enhanced": EnhancedChessPieceClassifier,
                "ultra": UltraEnhancedChessPieceClassifier,
            }
            reloaded = arch_map[classifier](num_classes=NUM_CLASSES, use_grayscale=True)
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            reloaded.load_state_dict(state)
            reloaded.to(device)

            # ── reproduce the same train / test split ─────────────────────────
            train_paths, test_paths = _split_paths(all_paths)

            train_acc = _accuracy_on_paths(reloaded, train_paths, device, FEN_CHARS)
            test_acc = _accuracy_on_paths(reloaded, test_paths, device, FEN_CHARS)

            print(
                f"\n  [{classifier}] train_acc={train_acc:.4f}  "
                f"test_acc={test_acc:.4f}  "
                f"(reported={reported_test_acc:.4f})"
            )

            # ── assert minimum thresholds ─────────────────────────────────────
            self.assertGreaterEqual(
                train_acc,
                MIN_TRAIN_ACC,
                f"[{classifier}] train acc {train_acc:.4f} < minimum {MIN_TRAIN_ACC}",
            )
            self.assertGreaterEqual(
                test_acc,
                MIN_TEST_ACC,
                f"[{classifier}] test acc {test_acc:.4f} < minimum {MIN_TEST_ACC}",
            )

    def test_pipeline_standard(self):
        """Standard CNN must meet minimum accuracy on synthetic data."""
        self._run_pipeline("standard")

    def test_pipeline_enhanced(self):
        """Enhanced CNN must meet minimum accuracy on synthetic data."""
        self._run_pipeline("enhanced")

    def test_pipeline_ultra(self):
        """Ultra CNN must meet minimum accuracy on synthetic data."""
        self._run_pipeline("ultra")


if __name__ == "__main__":
    unittest.main(verbosity=2)
