"""Pairwise translation search for declared overlapping tiles.

Coordinate convention: offset ``(dx, dy)`` for pair ``(a, b)`` means that image
b's top-left corner sits at pixel ``(dx, dy)`` of image a's coordinate frame.
The overlap rectangle in a's frame is

    x0 = max(0, dx)        x1 = min(W_a, dx + W_b)
    y0 = max(0, dy)        y1 = min(H_a, dy + H_b)

and the score is the mean absolute difference (MAD) of the two images inside
that rectangle.  ``np.abs`` is applied to int16 differences so it cannot wrap.
Sums are taken as int64; with at most 1024x1024 pixels and a maximum per-pixel
difference of 255 the sum fits comfortably.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .imaging import DecodedImage, overlap_rect

__all__ = ["PairScan", "overlap_rect", "scan_pair"]


@dataclass
class PairScan:
    """Result of scanning every search-window offset for one pair."""

    pair_id: int
    a: int
    b: int
    accepted: list[tuple[int, int]] = field(default_factory=list)
    best_mad: float | None = None
    best_offset: tuple[int, int] | None = None
    best_overlap: int = 0
    scored_positions: int = 0


def scan_pair(
    pair_id: int,
    a: DecodedImage,
    b: DecodedImage,
    offsets: list[tuple[int, int]],
    min_overlap: int,
    max_mad: float,
    *,
    a_i16: np.ndarray | None = None,
    b_i16: np.ndarray | None = None,
) -> PairScan:
    """Score all offsets of one pair, keeping those that satisfy both gates.

    An offset is accepted when the overlap area is at least ``min_overlap``
    pixels *and* its MAD is at most ``max_mad``.  Offsets with too little
    overlap are ignored entirely (they are not evidence of anything).  The
    diagnostics (``best_mad`` etc.) describe the best-scoring offset that had
    enough overlap, so a rejection can explain why.
    """
    scan = PairScan(pair_id=pair_id, a=a.image_id, b=b.image_id)
    a16 = a_i16 if a_i16 is not None else a.array.astype(np.int16)
    b16 = b_i16 if b_i16 is not None else b.array.astype(np.int16)
    a_w, a_h = a.width, a.height
    b_w, b_h = b.width, b.height

    for dx, dy in offsets:
        rect = overlap_rect(a_w, a_h, b_w, b_h, dx, dy)
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        n = (x1 - x0) * (y1 - y0)
        if n < min_overlap:
            continue
        scan.scored_positions += 1
        sub_a = a16[y0:y1, x0:x1]
        sub_b = b16[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
        if sub_b.shape != sub_a.shape:  # pragma: no cover - geometry guarantees it
            continue
        diff = np.abs(sub_a - sub_b)
        mad = float(diff.sum(dtype=np.int64)) / n
        if scan.best_mad is None or mad < scan.best_mad:
            scan.best_mad = mad
            scan.best_offset = (dx, dy)
            scan.best_overlap = n
        if mad <= max_mad + 1e-12:
            scan.accepted.append((dx, dy))
    return scan
