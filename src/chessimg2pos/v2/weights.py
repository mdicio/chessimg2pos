"""Find or download v2 weights and build the best available predictor."""
import os
import zipfile

from .._version import __version__

_RELEASE_BASE = "https://github.com/mdicio/chessimg2pos/releases/download"
V2_ONNX_URL = f"{_RELEASE_BASE}/v{__version__}/chessimg2pos-v2-onnx.zip"
CACHE = os.path.expanduser(os.path.join("~/.cache/chessimg2pos", __version__, "v2"))


def download_v2_onnx(verbose=True):
    """Download board.onnx + locator.onnx for this package version."""
    board = os.path.join(CACHE, "board.onnx")
    if os.path.exists(board):
        return CACHE
    import requests

    os.makedirs(CACHE, exist_ok=True)
    zpath = os.path.join(CACHE, "v2.zip")
    if verbose:
        print(f"Downloading model v2 from {V2_ONNX_URL} ...")
    r = requests.get(V2_ONNX_URL, stream=True, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download v2 weights (HTTP {r.status_code}): {V2_ONNX_URL}")
    with open(zpath, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(CACHE)
    os.remove(zpath)
    return CACHE


def load_predictor(weights=None, legality=True):
    """weights: a .pt checkpoint (PyTorch), a directory with board.onnx and
    locator.onnx (ONNX Runtime), or None to download the release ONNX."""
    weights = weights or os.environ.get("CHESSIMG2POS_V2_WEIGHTS") or download_v2_onnx()
    if os.path.isdir(weights):
        from .onnx_predictor import OnnxPredictorV2
        loc = os.path.join(weights, "locator.onnx")
        return OnnxPredictorV2(os.path.join(weights, "board.onnx"), loc if os.path.exists(loc) else None,
                               legality=legality)
    from .predictor import PredictorV2
    return PredictorV2(weights, legality=legality)
