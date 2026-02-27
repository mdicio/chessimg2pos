"""
tests/test_imports.py
---------------------
Verify that all public API symbols import without error, using both
the installed-package path and the editable/src path.
"""

import unittest


class TestImports(unittest.TestCase):
    def test_top_level_api(self):
        from chessimg2pos import (
            predict_fen,
            ChessRecognitionTrainer,
            ChessPositionPredictor,
        )  # noqa

        self.assertTrue(callable(predict_fen))

    def test_constants(self):
        from chessimg2pos.constants import (  # noqa
            DEFAULT_CLASSIFIER,
            DEFAULT_USE_GRAYSCALE,
            DEFAULT_FEN_CHARS,
        )
        from chessimg2pos.constants import DEFAULT_FEN_CHARS

        self.assertEqual(len(DEFAULT_FEN_CHARS), 13)

    def test_version_string(self):
        import chessimg2pos

        self.assertRegex(chessimg2pos.__version__, r"^\d+\.\d+\.\d+$")

    def test_model_loader_url_tracks_version(self):
        """The download URL must always contain the current package version."""
        import chessimg2pos
        from chessimg2pos.model_loader import PRETRAINED_MODEL_URL

        self.assertIn(chessimg2pos.__version__, PRETRAINED_MODEL_URL)


if __name__ == "__main__":
    unittest.main()
