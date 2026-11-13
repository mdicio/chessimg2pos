"""Placement helpers. Labels are standard FEN placement from White's view;
"as drawn" means the board read top row first as it appears in the image."""
import re

CLASSES = "1KQRBNPkqrbnp"  # index 0 = empty square
IDX = {c: i for i, c in enumerate(CLASSES)}


def expand(p):
    return re.sub(r"[2-8]", lambda m: "1" * int(m.group()), p)


def compress(p):
    return re.sub(r"1{2,8}", lambda m: str(len(m.group())), p)


def flip(p):
    rows = expand(p).split("/")
    return compress("/".join(r[::-1] for r in reversed(rows)))


def to_indices(p):
    """placement -> 64 class indices, row-major from the top-left."""
    return [IDX[c] for c in expand(p).replace("/", "")]


def from_indices(idx):
    s = "".join(CLASSES[i] for i in idx)
    return compress("/".join(s[r * 8:(r + 1) * 8] for r in range(8)))
