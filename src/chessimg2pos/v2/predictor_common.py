from .preprocess import box_to_corners


def as_corners(v):
    """Locator output (4 = box, 8 = corners) -> corners TL, TR, BR, BL."""
    v = [float(a) for a in v]
    return box_to_corners(v) if len(v) == 4 else [v[0:2], v[2:4], v[4:6], v[6:8]]
