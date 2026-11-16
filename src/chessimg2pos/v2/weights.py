"""Find or download v2 weights and build the best available predictor."""
import os
import shutil
import tempfile
import zipfile

# The GitHub release that hosts the v2 ONNX weights. Kept apart from the
# package version so a code-only release (e.g. 0.2.1) reuses the same
# download and cache; bump it only when you publish new weights.
V2_WEIGHTS_VERSION = "0.2.0"

_RELEASE_BASE = "https://github.com/mdicio/chessimg2pos/releases/download"
V2_ONNX_URL = f"{_RELEASE_BASE}/v{V2_WEIGHTS_VERSION}/chessimg2pos-v2-onnx.zip"
CACHE = os.path.expanduser(os.path.join("~/.cache/chessimg2pos", V2_WEIGHTS_VERSION, "v2"))
_FILES = ("board.onnx", "locator.onnx")


def download_v2_onnx(verbose=True):
    """Download board.onnx + locator.onnx (once; cached under ~/.cache)."""
    if all(os.path.exists(os.path.join(CACHE, f)) for f in _FILES):
        return CACHE
    import requests

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    if verbose:
        print(f"Downloading model v2 (~170 MB, first use only) from {V2_ONNX_URL} ...")
    # download + extract into a temp dir, then move into place: an interrupted
    # download never leaves a half-written model in the cache
    tmp = tempfile.mkdtemp(dir=os.path.dirname(CACHE))
    try:
        zpath = os.path.join(tmp, "v2.zip")
        with requests.get(V2_ONNX_URL, stream=True, timeout=60) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Failed to download v2 weights (HTTP {r.status_code}): {V2_ONNX_URL}")
            with open(zpath, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        os.remove(zpath)
        missing = [f for f in _FILES if not os.path.exists(os.path.join(tmp, f))]
        if missing:
            raise RuntimeError(f"{', '.join(missing)} not found at the root of {V2_ONNX_URL}")
        os.makedirs(CACHE, exist_ok=True)
        for f in os.listdir(tmp):
            os.replace(os.path.join(tmp, f), os.path.join(CACHE, f))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return CACHE


def load_predictor(weights=None, legality=True):
    """weights: a .pt checkpoint (PyTorch), a directory with board.onnx and
    locator.onnx (ONNX Runtime), or None to use $CHESSIMG2POS_V2_WEIGHTS or
    download the release ONNX."""
    weights = os.fspath(weights or os.environ.get("CHESSIMG2POS_V2_WEIGHTS") or download_v2_onnx())
    if os.path.isdir(weights):
        from .onnx_predictor import OnnxPredictorV2
        loc = os.path.join(weights, "locator.onnx")
        return OnnxPredictorV2(os.path.join(weights, "board.onnx"), loc if os.path.exists(loc) else None,
                               legality=legality)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"v2 weights not found: {weights}")
    try:
        from .predictor import PredictorV2
    except ImportError as e:
        raise ImportError(
            '.pt checkpoints need PyTorch: pip install "chessimg2pos[torch]". '
            "The default ONNX weights do not."
        ) from e
    return PredictorV2(weights, legality=legality)
