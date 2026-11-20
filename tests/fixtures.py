"""Generate synthetic overlapping tiles for tests and the demo script."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

# Three 600x600 crops of one 1400x1400 drawing:
#   tile 0 at drawing (0, 0), tile 1 at (450, 0), tile 2 at (0, 450).
# Pair overlaps: (0,1) 150x600, (0,2) 600x150, (1,2) 150x150.
DEFAULT_PLACES = [
    (0, 0, 600, 600),
    (450, 0, 600, 600),
    (0, 450, 600, 600),
]
# True offsets: for pair (a, b), b's origin is at a's origin + (dx, dy).
TRUE_OFFSETS = {(0, 1): (450, 0), (0, 2): (0, 450), (1, 2): (-450, 450)}


def drawing(seed: int = 0, size: tuple[int, int] = (1400, 1400)) -> np.ndarray:
    """A deterministic, texture-rich 8-bit 'drawing' with lines and blobs."""
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]

    base = 218 + 18 * np.sin(xx / 37.0) * np.cos(yy / 53.0)
    img = base + rng.normal(0, 2.0, size=(h, w))

    # Long straight "drafting lines" plus some thicker strokes.
    for _ in range(90):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        length = int(rng.integers(60, 500))
        thick = int(rng.integers(1, 4))
        dark = int(rng.integers(20, 90))
        if rng.random() < 0.5:
            img[max(0, y - thick) : y + thick, x : min(w, x + length)] = dark
        else:
            img[y : min(h, y + length), max(0, x - thick) : x + thick] = dark

    # Round "detail callouts" so the texture is not translation-degenerate.
    for _ in range(160):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        r = int(rng.integers(3, 26))
        y0, y1 = max(0, cy - r), min(h, cy + r)
        x0, x1 = max(0, cx - r), min(w, cx + r)
        if y0 >= y1 or x0 >= x1:
            continue
        dy = np.arange(y0, y1)[:, None] - cy
        dx = np.arange(x0, x1)[None, :] - cx
        mask = dx * dx + dy * dy <= r * r
        val = int(rng.integers(30, 200))
        img[y0:y1, x0:x1][mask] = val

    # A border, as a real drawing sheet would have.
    img[0:3, :] = 40
    img[-3:, :] = 40
    img[:, 0:3] = 40
    img[:, -3:] = 40
    return np.clip(img, 0, 255).astype(np.uint8)


def tile_set(
    seed: int = 0,
    places: list[tuple[int, int, int, int]] | None = None,
    noise_sigma: float = 0.0,
    gain: float = 1.0,
    offset: float = 0.0,
    per_tile_gain: list[float] | None = None,
) -> list[np.ndarray]:
    """Crop a drawing at the given (x, y, w, h) boxes, optionally degrading.

    ``noise_sigma`` adds Gaussian noise; ``gain``/``offset`` simulate scanner
    exposure differences.  ``per_tile_gain`` overrides ``gain`` per tile, which
    is what actually produces an *inter-tile* brightness mismatch.
    """
    canvas = drawing(seed)
    places = DEFAULT_PLACES if places is None else places
    rng = np.random.default_rng(seed + 1000)
    tiles = []
    for idx, (x, y, w, h) in enumerate(places):
        t = canvas[y : y + h, x : x + w].astype(np.float64)
        if noise_sigma:
            t = t + rng.normal(0, noise_sigma, size=t.shape)
        g = gain if per_tile_gain is None else per_tile_gain[idx]
        t = t * g + offset
        tiles.append(np.clip(np.rint(t), 0, 255).astype(np.uint8))
    return tiles


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def png_bytes_rgb(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.stack([arr] * 3, axis=-1), mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def png_bytes_palette(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").convert("P").save(buf, format="PNG")
    return buf.getvalue()


def png_bytes_gray_alpha(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    la = np.stack([arr, np.full_like(arr, 255)], axis=-1)
    Image.fromarray(la, mode="LA").save(buf, format="PNG")
    return buf.getvalue()


def png_bytes_16bit(arr: np.ndarray) -> bytes:
    """Save the same image as a 16-bit grayscale PNG."""
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint16) * 257).save(buf, format="PNG")
    return buf.getvalue()


def window(center: tuple[int, int], radius: int = 4) -> dict[str, int]:
    cx, cy = center
    return {
        "dx_min": cx - radius,
        "dx_max": cx + radius,
        "dy_min": cy - radius,
        "dy_max": cy + radius,
    }


def spec_for(
    pairs: list[tuple[int, int]],
    *,
    radius: int = 4,
    min_overlap: int = 20000,
    max_mad: float = 2.0,
    offsets: dict[tuple[int, int], tuple[int, int]] | None = None,
) -> dict:
    offsets = TRUE_OFFSETS if offsets is None else offsets
    return {
        "pairs": [
            {"a": a, "b": b, "window": window(offsets[(a, b)], radius)}
            for a, b in pairs
        ],
        "min_overlap_pixels": min_overlap,
        "max_mean_abs_diff": max_mad,
    }
