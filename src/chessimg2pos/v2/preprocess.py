"""Image pre-processing shared by every backend (numpy + Pillow only)."""
import numpy as np
from PIL import Image

BOARD_SIZE = 256
LOC_SIZE = 320
MEAN = np.array((0.485, 0.456, 0.406), np.float32)
STD = np.array((0.229, 0.224, 0.225), np.float32)


def to_array(img):
    """PIL RGB -> float32 CHW, ImageNet-normalised."""
    a = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(a.transpose(2, 0, 1))


def crop_board(img, box, size=BOARD_SIZE, resample=Image.BILINEAR):
    """Crop the grid box (may extend past the image; filled with black) and
    resize to size x size."""
    x0, y0, x1, y1 = box
    return img.crop((round(x0), round(y0), round(x1), round(y1))).resize((size, size), resample)


def letterbox(img, size=LOC_SIZE):
    """Pad to a square (top-left anchored) and resize. Returns the image and
    the padded side length (normalised coords * side = pixels)."""
    W, H = img.size
    side = max(W, H)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, (0, 0))
    return canvas.resize((size, size), Image.BILINEAR), side


def refine_crop(img, box, margin=0.1):
    """Region around a rough box for the locator's second pass."""
    W, H = img.size
    bw, bh = box[2] - box[0], box[3] - box[1]
    x0, y0 = max(0, int(box[0] - margin * bw)), max(0, int(box[1] - margin * bh))
    x1, y1 = min(W, int(box[2] + margin * bw)), min(H, int(box[3] + margin * bh))
    if x1 - x0 < 32 or y1 - y0 < 32:
        x0, y0, x1, y1 = 0, 0, W, H
    return img.crop((x0, y0, x1, y1)), (x0, y0)


def load(image):
    if isinstance(image, str):
        image = Image.open(image)
    return image.convert("RGB")
