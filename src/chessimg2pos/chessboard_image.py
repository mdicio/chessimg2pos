import numpy as np
import PIL.Image
from .constants import DEFAULT_USE_GRAYSCALE


def _get_resized_chessboard(chessboard_img_path):
    """chessboard_img_path = path to a chessboard image, or a PIL Image object.
    Returns a 256x256 image of a chessboard (32x32 per tile)
    """
    if isinstance(chessboard_img_path, PIL.Image.Image):
        img_data = chessboard_img_path.convert("RGB")
    else:
        img_data = PIL.Image.open(chessboard_img_path).convert("RGB")
    return img_data.resize([256, 256], PIL.Image.BILINEAR)


def get_chessboard_tiles(chessboard_img_path, use_grayscale=DEFAULT_USE_GRAYSCALE):
    """Extract 64 32x32 tile images from a chessboard.

    Args:
        chessboard_img_path: Path to a chessboard image file, or a PIL Image object.
        use_grayscale: Return tiles in grayscale when True.

    Returns:
        list[PIL.Image]: 64 tiles in order from top-left (A8) to bottom-right (H1).
    """
    img_data = _get_resized_chessboard(chessboard_img_path)
    if use_grayscale:
        img_data = img_data.convert("L", (0.2989, 0.5870, 0.1140, 0))
    chessboard_256x256_img = np.asarray(img_data, dtype=np.uint8)
    # 64 tiles in order from top-left to bottom-right (A8, B8, ..., G1, H1)
    tiles = [None] * 64
    for rank in range(8):  # rows/ranks (numbers)
        for file in range(8):  # columns/files (letters)
            sq_i = rank * 8 + file
            tile = np.zeros([32, 32, 3], dtype=np.uint8)
            for i in range(32):
                for j in range(32):
                    if use_grayscale:
                        tile[i, j] = chessboard_256x256_img[
                            rank * 32 + i,
                            file * 32 + j,
                        ]
                    else:
                        tile[i, j] = chessboard_256x256_img[
                            rank * 32 + i,
                            file * 32 + j,
                            :,
                        ]
            tiles[sq_i] = PIL.Image.fromarray(tile, "RGB")
    return tiles
