# utils/model_loader.py
import os
import torch
import hashlib
import requests
from pathlib import Path
PRETRAINED_MODEL_URL = "https://github.com/mdicio/chessimg2pos/releases/download/v0.1-alpha/model.zip"

import zipfile

def download_pretrained_model(cache_dir="~/.cache/chessimg2pos", verbose=True):
    os.makedirs(os.path.expanduser(cache_dir), exist_ok=True)
    zip_path = os.path.join(os.path.expanduser(cache_dir), "model.zip")
    model_path = os.path.join(os.path.expanduser(cache_dir), "model.pth")

    if os.path.exists(model_path):
        return model_path

    # Download zip file
    if verbose:
        print(f"Downloading model from {PRETRAINED_MODEL_URL}...")
    r = requests.get(PRETRAINED_MODEL_URL)
    with open(zip_path, "wb") as f:
        f.write(r.content)

    # Unzip it
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.expanduser(cache_dir))

    return model_path
