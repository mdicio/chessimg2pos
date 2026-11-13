"""Model v2 inference (PyTorch backend): locate the board, read all 64
squares, apply the legality step, return standard FEN from White's view."""
import numpy as np
import torch

from .model import BoardNet, LocatorNet
from .postprocess import decode
from .predictor_common import as_corners
from .preprocess import (board_input, box_to_corners, corners_to_box, letterbox, load, refine_crop,
                         to_array)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    board = BoardNet(pretrained=False, **ck["board_config"])
    board.load_state_dict(ck["board"])
    loc = None
    if ck.get("locator"):
        loc = LocatorNet(pretrained=False, **ck["locator_config"])
        loc.load_state_dict(ck["locator"])
    return board, loc


class PredictorV2:
    def __init__(self, weights, device=None, legality=True, locate=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.board, self.loc = load_checkpoint(weights, self.device)
        self.board.to(self.device).eval()
        if self.loc is not None:
            self.loc.to(self.device).eval()
        self.legality = legality
        self.locate_boards = locate and self.loc is not None

    @torch.no_grad()
    def _loc(self, imgs):
        xs, sides = zip(*(letterbox(i) for i in imgs))
        x = torch.from_numpy(np.stack([to_array(v) for v in xs])).to(self.device)
        out = self.loc(x).float().cpu().numpy()
        return [as_corners(o * s) for o, s in zip(out, sides)]

    def locate(self, imgs):
        """Board corners (TL, TR, BR, BL). Two passes: rough on the whole
        image, then refined on a crop around the first guess."""
        crops, offs = zip(*(refine_crop(i, corners_to_box(c)) for i, c in zip(imgs, self._loc(imgs))))
        refined = self._loc(list(crops))
        return [[[x + ox, y + oy] for x, y in c] for c, (ox, oy) in zip(refined, offs)]

    @torch.no_grad()
    def predict_batch(self, images, boxes=None):
        imgs = [load(i) for i in images]
        if boxes is None:
            quads = self.locate(imgs) if self.locate_boards else [box_to_corners([0, 0, *i.size]) for i in imgs]
        else:
            quads = [box_to_corners(b) if len(b) == 4 and not hasattr(b[0], "__len__") else b for b in boxes]
        x = torch.from_numpy(np.stack([to_array(board_input(i, q)) for i, q in zip(imgs, quads)])).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            sq, ori = self.board(x)
        sq = torch.softmax(sq.float(), -1).cpu().numpy()
        ori = torch.softmax(ori.float(), -1).cpu().numpy()
        return [decode(s, o, b, self.legality) for s, o, b in zip(sq, ori, quads)]

    def predict(self, image, box=None):
        return self.predict_batch([image], None if box is None else [box])[0]
