import re


def compressed_fen(fen):
    """From: 11111q1k/1111r111/111p1pQP/111P1P11/11prn1R1/11111111/111111P1/R11111K1
    To: 5q1k/4r3/3p1pQP/3P1P2/2prn1R1/8/6P1/R5K1
    """
    for length in reversed(range(2, 9)):
        fen = fen.replace(length * "1", str(length))
    return fen


def uncompressed_fen(fen):
    for digit in re.findall(r"[2-8]", fen):
        fen = fen.replace(digit, int(digit) * "1")
    return fen


FEN_TYPES = ("compressed", "expanded")


def normalize_fen_type(fen_type):
    """Validate a ``fen_type`` argument and return "compressed" or "expanded".

    "standard" is accepted for backward compatibility: before 0.2.0 it was the
    default and produced the expanded ("11111111/...") format.
    """
    if fen_type == "standard":
        import warnings

        warnings.warn(
            'fen_type="standard" is deprecated and returns the expanded '
            '"11111111/..." format. Use fen_type="expanded" for that format, or '
            'the default fen_type="compressed" for standard FEN.',
            DeprecationWarning,
            stacklevel=3,
        )
        return "expanded"
    if fen_type not in FEN_TYPES:
        raise ValueError(
            f"fen_type must be one of {FEN_TYPES}, got {fen_type!r}"
        )
    return fen_type
