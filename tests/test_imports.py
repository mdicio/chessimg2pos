"""Public API, packaging and predict_fen behaviour (no weights needed)."""
import subprocess
import sys
import warnings

import pytest

import chessimg2pos


def test_public_api():
    from chessimg2pos import load_predictor, predict_fen  # noqa: F401

    assert callable(predict_fen) and callable(load_predictor)
    assert set(chessimg2pos.__all__) == {"predict_fen", "load_predictor", "__version__"}


def test_version_string():
    assert chessimg2pos.__version__.count(".") == 2


def test_import_does_not_load_torch():
    code = "import sys, chessimg2pos, chessimg2pos.cli; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


class _Fake:
    def predict(self, image):
        return {"fen": "8/8/8/4p1K1/2k1P3/8/8/8", "fen_as_drawn": "8/8/8/3P1k2/1K1p4/8/8/8",
                "orientation": "white", "confidence": 0.99}


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(chessimg2pos, "load_predictor", lambda w=None, legality=True: _Fake())
    chessimg2pos._PREDICTORS.clear()
    yield
    chessimg2pos._PREDICTORS.clear()


def test_predict_fen_formats(fake):
    assert chessimg2pos.predict_fen("x.png") == "8/8/8/4p1K1/2k1P3/8/8/8"
    assert chessimg2pos.predict_fen("x.png", fen_type="expanded").startswith("11111111/11111111/11111111/1111p1K1")
    full = chessimg2pos.predict_fen("x.png", output_type="full")
    assert full["orientation"] == "white" and full["confidence"] == 0.99
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert chessimg2pos.predict_fen("x.png", fen_type="standard").startswith("11111111")
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    with pytest.raises(ValueError):
        chessimg2pos.predict_fen("x.png", fen_type="bogus")


def test_predictor_is_cached(fake):
    chessimg2pos.predict_fen("a.png")
    chessimg2pos.predict_fen("b.png")
    assert len(chessimg2pos._PREDICTORS) == 1
