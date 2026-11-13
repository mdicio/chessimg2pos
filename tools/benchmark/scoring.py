"""Scoring for chess-image-to-FEN tools.

A board counts as correct only when all 64 squares match.

Tools differ in what they return for a board drawn from Black's side: some
return the board *as drawn* (top row of the image first), others detect the
orientation and return standard FEN from White's view. Both readings are
legitimate, so a prediction is accepted if it matches either the true
position or the true position as drawn. `orientation_ok` records whether the
tool also got the orientation right (i.e. matched the true White-view FEN).
"""
import re


def expand(placement: str) -> str:
    return re.sub(r"[2-8]", lambda m: "1" * int(m.group()), placement)


def compress(expanded: str) -> str:
    return re.sub(r"1{2,8}", lambda m: str(len(m.group())), expanded)


def normalize(fen):
    """Accept full FEN, compressed or expanded placement, '-' or '/' separators.
    Returns compressed placement, or None if not a valid 8x8 placement."""
    if not fen:
        return None
    p = fen.strip().split()[0].replace("-", "/")
    rows = expand(p).split("/")
    if len(rows) != 8 or any(len(r) != 8 or not re.fullmatch(r"[KQRBNPkqrbnp1]{8}", r) for r in rows):
        return None
    return compress("/".join(rows))


def flip(placement: str) -> str:
    rows = expand(placement).split("/")
    return compress("/".join(r[::-1] for r in reversed(rows)))


def squares(a: str, b: str) -> int:
    a, b = expand(a).replace("/", ""), expand(b).replace("/", "")
    return sum(x == y for x, y in zip(a, b))


def score(pred_raw, truth: str, orientation: str) -> dict:
    pred = normalize(pred_raw)
    as_drawn = truth if orientation == "white" else flip(truth)
    if pred is None:
        return {"exact": False, "orientation_ok": False, "squares": 0, "pred": None}
    best = max(squares(pred, truth), squares(pred, as_drawn))
    return {
        "exact": pred in (truth, as_drawn),
        "orientation_ok": pred == truth,
        "squares": best,
        "pred": pred,
    }
