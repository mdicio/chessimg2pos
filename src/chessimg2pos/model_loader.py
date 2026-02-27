# model_loader.py
import os
import zipfile
import requests
from ._version import __version__

# The URL is derived from the package version so it automatically points
# to the matching GitHub release whenever __version__ is bumped.
_RELEASE_BASE = "https://github.com/mdicio/chessimg2pos/releases/download"
PRETRAINED_MODEL_URL = f"{_RELEASE_BASE}/v{__version__}/model.zip"


def download_pretrained_model(cache_dir="~/.cache/chessimg2pos", verbose=True):
    """Download the pre-trained model for the current package version.

    The model is cached under ~/.cache/chessimg2pos/<version>/ so that
    upgrading the package automatically fetches the matching model file
    instead of reusing a stale one from a previous version.

    Args:
        cache_dir: Root cache directory (~ is expanded automatically).
        verbose: Print progress messages.

    Returns:
        str: Absolute path to the downloaded model file.
    """
    versioned_dir = os.path.join(os.path.expanduser(cache_dir), __version__)
    os.makedirs(versioned_dir, exist_ok=True)

    model_path = os.path.join(versioned_dir, "model_enhanced.pt")
    if os.path.exists(model_path):
        return model_path

    zip_path = os.path.join(versioned_dir, "model.zip")
    if verbose:
        print(f"Downloading model v{__version__} from {PRETRAINED_MODEL_URL} ...")

    response = requests.get(PRETRAINED_MODEL_URL, stream=True, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download model (HTTP {response.status_code}). "
            f"URL: {PRETRAINED_MODEL_URL}\n"
            f"Make sure a GitHub release v{__version__} exists with model.zip attached."
        )

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(versioned_dir)

    os.remove(zip_path)  # keep cache tidy

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"model_enhanced.pt not found in the downloaded zip. "
            f"Expected it at: {model_path}"
        )

    if verbose:
        print(f"Model saved to {model_path}")
    return model_path
