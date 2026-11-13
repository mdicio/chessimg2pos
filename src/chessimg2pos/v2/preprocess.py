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


def box_to_corners(box):
    x0, y0, x1, y1 = box
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def corners_to_box(corners):
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


def is_axis_aligned(corners, tol=0.006):
    """True if the quad is (nearly) the rectangle of its bounding box."""
    x0, y0, x1, y1 = corners_to_box(corners)
    side = max(x1 - x0, y1 - y0, 1e-6)
    return all(abs(a - b) / side <= tol for c, r in zip(corners, box_to_corners([x0, y0, x1, y1]))
               for a, b in zip(c, r))


def _perspective_coeffs(src_corners, size):
    """PIL PERSPECTIVE coefficients mapping the output square (size x size)
    to the quad src_corners (TL, TR, BR, BL) in the input image."""
    dst = [(0, 0), (size, 0), (size, size), (0, size)]
    A, b = [], []
    for (u, v), (x, y) in zip(dst, src_corners):
        A.append([u, v, 1, 0, 0, 0, -u * x, -v * x]); b.append(x)
        A.append([0, 0, 0, u, v, 1, -u * y, -v * y]); b.append(y)
    return np.linalg.solve(np.array(A, np.float64), np.array(b, np.float64)).tolist()


def warp_board(img, corners, size=BOARD_SIZE):
    """Perspective-correct the board quad into a size x size square.
    Downscales first when the board is large, so the warp does not alias."""
    corners = [list(map(float, c)) for c in corners]
    sides = [np.hypot(corners[i][0] - corners[(i + 1) % 4][0], corners[i][1] - corners[(i + 1) % 4][1])
             for i in range(4)]
    f = min(sides) / (1.5 * size)
    if f > 1:
        W, H = img.size
        img = img.resize((max(1, round(W / f)), max(1, round(H / f))), Image.BILINEAR)
        corners = [[x / f, y / f] for x, y in corners]
    return img.transform((size, size), Image.PERSPECTIVE, _perspective_coeffs(corners, size), Image.BILINEAR)


def board_input(img, corners, size=BOARD_SIZE):
    """The classifier input for a located board: plain crop+resize for
    screenshots (axis-aligned), perspective warp for photos."""
    if is_axis_aligned(corners):
        return crop_board(img, corners_to_box(corners), size)
    return warp_board(img, corners, size)
