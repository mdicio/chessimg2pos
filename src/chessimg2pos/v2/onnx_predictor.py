"""Model v2 inference with ONNX Runtime: no PyTorch needed.

    pip install onnxruntime pillow numpy
    OnnxPredictorV2("board.onnx", "locator.onnx").predict("screenshot.png")
"""
import numpy as np

from .postprocess import decode, softmax
from .preprocess import crop_board, letterbox, load, refine_crop, to_array


class OnnxPredictorV2:
    def __init__(self, board_path, locator_path=None, legality=True, providers=None):
        import onnxruntime as ort

        providers = providers or ort.get_available_providers()
        self.board = ort.InferenceSession(board_path, providers=providers)
        self.loc = ort.InferenceSession(locator_path, providers=providers) if locator_path else None
        self.legality = legality

    def _loc(self, imgs):
        xs, sides = zip(*(letterbox(i) for i in imgs))
        out = self.loc.run(None, {"image": np.stack([to_array(v) for v in xs])})[0]
        return [list(o * s) for o, s in zip(out, sides)]

    def locate(self, imgs):
        crops, offs = zip(*(refine_crop(i, b) for i, b in zip(imgs, self._loc(imgs))))
        refined = self._loc(list(crops))
        return [[b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy] for b, (ox, oy) in zip(refined, offs)]

    def predict_batch(self, images, boxes=None):
        imgs = [load(i) for i in images]
        if boxes is None:
            boxes = self.locate(imgs) if self.loc is not None else [[0, 0, *i.size] for i in imgs]
        x = np.stack([to_array(crop_board(i, b)) for i, b in zip(imgs, boxes)])
        sq, ori = self.board.run(None, {"board": x})
        return [decode(softmax(s), softmax(o), b, self.legality) for s, o, b in zip(sq, ori, boxes)]

    def predict(self, image, box=None):
        return self.predict_batch([image], None if box is None else [box])[0]
