"""Model v2 inference with ONNX Runtime: no PyTorch needed.

    pip install chessimg2pos      (onnxruntime, numpy, Pillow; no PyTorch)
    OnnxPredictorV2("board.onnx", "locator.onnx").predict("screenshot.png")
"""
import numpy as np

from .postprocess import decode, softmax
from .predictor_common import as_corners
from .preprocess import board_input, box_to_corners, corners_to_box, letterbox, load, refine_crop, to_array


class OnnxPredictorV2:
    def __init__(self, board_path, locator_path=None, legality=True, providers=None):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "chessimg2pos runs on ONNX Runtime, which is not installed: pip install onnxruntime"
            ) from e

        providers = providers or ort.get_available_providers()
        self.board = ort.InferenceSession(board_path, providers=providers)
        self.loc = ort.InferenceSession(locator_path, providers=providers) if locator_path else None
        self.legality = legality

    def _loc(self, imgs):
        xs, sides = zip(*(letterbox(i) for i in imgs))
        out = self.loc.run(None, {"image": np.stack([to_array(v) for v in xs])})[0]
        return [as_corners(o * s) for o, s in zip(out, sides)]

    def locate(self, imgs):
        crops, offs = zip(*(refine_crop(i, corners_to_box(c)) for i, c in zip(imgs, self._loc(imgs))))
        refined = self._loc(list(crops))
        return [[[x + ox, y + oy] for x, y in c] for c, (ox, oy) in zip(refined, offs)]

    def predict_batch(self, images, boxes=None):
        imgs = [load(i) for i in images]
        if boxes is None:
            quads = self.locate(imgs) if self.loc is not None else [box_to_corners([0, 0, *i.size]) for i in imgs]
        else:
            quads = [box_to_corners(b) if len(b) == 4 and not hasattr(b[0], "__len__") else b for b in boxes]
        x = np.stack([to_array(board_input(i, q)) for i, q in zip(imgs, quads)])
        sq, ori = self.board.run(None, {"board": x})
        return [decode(softmax(s), softmax(o), q, self.legality) for s, o, q in zip(sq, ori, quads)]

    def predict(self, image, box=None):
        return self.predict_batch([image], None if box is None else [box])[0]
