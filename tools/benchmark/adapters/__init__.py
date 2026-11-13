"""Tool adapters. Each adapter exposes:

    name: str
    setup() -> None            install/download whatever the tool needs (idempotent)
    predict(image_path) -> str FEN placement, or None if the tool found no board

Register new adapters in ADAPTERS.
"""
from .chessimg2pos_adapter import Chessimg2posAdapter
from .tsoj_adapter import TsojAdapter
from .fenshot_adapter import FenshotAdapter

ADAPTERS = {
    "chessimg2pos": Chessimg2posAdapter,
    "tsoj": TsojAdapter,
    "fenshot": FenshotAdapter,
}
