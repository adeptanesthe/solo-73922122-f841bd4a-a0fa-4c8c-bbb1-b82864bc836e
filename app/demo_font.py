"""Deterministic bitmap font and demo image generator.

A small fixed 5x7 pixel font (plus a 2-wide dot) renders clean binary
8-bit grayscale PNG data for the interactive ``/demo/*`` endpoints and the
standalone demo scripts.  The generator can also drop the blank column
between two glyphs ("stuck" digits) and flip chosen pixels to ink or paper,
which exercises the segmentation and mismatch logic end to end.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

GLYPH_HEIGHT = 7
DIGIT_WIDTH = 5
DOT_WIDTH = 2
GAP_WIDTH = 1

# Classic 5x7 column font: rows top to bottom, each a 5-character string.
_FONT: dict[str, tuple[str, ...]] = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
}


def glyph_bitmap(label: str) -> np.ndarray:
    """Return one glyph as a boolean ink array of shape (7, width)."""
    if label == ".":
        bmp = np.zeros((GLYPH_HEIGHT, DOT_WIDTH), dtype=bool)
        bmp[5:7, :] = True
        return bmp
    rows = _FONT[label]
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def all_glyph_labels() -> list[str]:
    return [str(d) for d in range(10)] + ["."]


def _white(width: int) -> np.ndarray:
    return np.zeros((GLYPH_HEIGHT, width), dtype=bool)


def render_line(
    text: str,
    *,
    stuck_after: set[int] | None = None,
    flips: list[tuple[int, int]] | None = None,
    margin: int = 2,
) -> np.ndarray:
    """Render a reading into a binary grayscale array (255 paper, 0 ink).

    Parameters
    ----------
    text:
        Characters to render, e.g. ``"12.50"``.
    stuck_after:
        Character indices after which the one-column blank gap is removed,
        fusing adjacent ink columns to mimic digits stuck together.
    flips:
        ``(row, col)`` pixel coordinates (in the final rendered array) whose
        binary value is inverted, simulating scan defects.
    margin:
        All-white columns added at both ends (legal to skip).
    """
    stuck_after = stuck_after or set()
    parts: list[np.ndarray] = [_white(margin)]
    for idx, ch in enumerate(text):
        parts.append(glyph_bitmap(ch))
        if idx != len(text) - 1 and idx not in stuck_after:
            parts.append(_white(GAP_WIDTH))
    parts.append(_white(margin))
    ink = np.concatenate(parts, axis=1)

    flips = flips or []
    for row, col in flips:
        ink[row, col] = not ink[row, col]

    return np.where(ink, 0, 255).astype(np.uint8)


def glyph_png(label: str) -> bytes:
    """PNG bytes for one clean labelled sample."""
    arr = np.where(glyph_bitmap(label), 0, 255).astype(np.uint8)
    return to_png(arr)


def to_png(arr: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


# A demo line that exercises segmentation, a decimal point and a defect.
DEMO_TEXT = "12.50"


def demo_line_png() -> bytes:
    # Fuse "1"-"2" and "5"-"0"; flip one ink pixel of the "2" to paper and add
    # a noise ink pixel, yielding a small, predictable mismatch count.
    line = render_line(
        DEMO_TEXT,
        stuck_after={0, 3},
        flips=[(2, 4), (3, 9)],
    )
    return to_png(line)


def demo_spec() -> dict:
    return {"glyph_labels": all_glyph_labels(), "max_mismatch": 12, "min_margin": 3}
