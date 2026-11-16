"""Tool adapters. Each adapter exposes:

    name: str
    setup() -> None            install/download whatever the tool needs (idempotent)
    predict(image_path) -> str FEN placement, or None if the tool found no board

Register new adapters in ADAPTERS.
"""
from .chessimg2pos_v1_adapter import Chessimg2posV1Adapter
from .chessimg2pos_v2_adapter import Chessimg2posV2Adapter
from .fenshot_adapter import FenshotAdapter
from .tsoj_adapter import TsojAdapter

ADAPTERS = {
    "chessimg2pos": Chessimg2posV2Adapter,
    "chessimg2pos_v1": Chessimg2posV1Adapter,
    "tsoj": TsojAdapter,
    "fenshot": FenshotAdapter,
}
