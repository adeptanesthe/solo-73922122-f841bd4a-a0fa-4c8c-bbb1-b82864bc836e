"""Self-contained synthetic drawing/tile generator for the demo scripts.

Kept independent of the test fixtures so the demo works from a plain
``pip install -r requirements.txt`` checkout, and so a mistake here cannot hide
behind the test data.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

# 2x2 grid of 600x600 crops of one drawing, 150 px overlap along each seam.
PLACES = [(0, 0, 600, 600), (450, 0, 600, 600), (0, 450, 600, 600), (450, 450, 600, 600)]

# For pair (a, b): b's top-left corner in a's pixel frame.
OFFSETS = {
    (0, 1): (450, 0),
    (0, 2): (0, 450),
    (1, 3): (0, 450),
    (2, 3): (450, 0),
    (0, 3): (450, 450),
}


def drawing(seed: int = 7, size: tuple[int, int] = (1400, 1400)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    img = 218 + 18 * np.sin(xx / 37.0) * np.cos(yy / 53.0) + rng.normal(0, 2.0, (h, w))
    for _ in range(90):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        length, thick = int(rng.integers(60, 500)), int(rng.integers(1, 4))
        dark = int(rng.integers(20, 90))
        if rng.random() < 0.5:
            img[max(0, y - thick) : y + thick, x : min(w, x + length)] = dark
        else:
            img[y : min(h, y + length), max(0, x - thick) : x + thick] = dark
    for _ in range(160):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(3, 26))
        y0, y1 = max(0, cy - r), min(h, cy + r)
        x0, x1 = max(0, cx - r), min(w, cx + r)
        if y0 < y1 and x0 < x1:
            dy = np.arange(y0, y1)[:, None] - cy
            dx = np.arange(x0, x1)[None, :] - cx
            img[y0:y1, x0:x1][dx * dx + dy * dy <= r * r] = int(rng.integers(30, 200))
    img[0:3, :] = img[-3:, :] = 40
    img[:, 0:3] = img[:, -3:] = 40
    return np.clip(img, 0, 255).astype(np.uint8)


def demo_tiles(noise_sigma: float = 0.6) -> list[np.ndarray]:
    canvas = drawing()
    rng = np.random.default_rng(11)
    gains = [1.0, 1.01, 0.99, 1.0]
    tiles = []
    for idx, (x, y, w, h) in enumerate(PLACES):
        t = canvas[y : y + h, x : x + w].astype(np.float64)
        t = t + rng.normal(0, noise_sigma, t.shape)
        tiles.append(np.clip(np.rint(t * gains[idx] - 2.0), 0, 255).astype(np.uint8))
    return tiles


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def window(center: tuple[int, int], radius: int = 2) -> dict[str, int]:
    cx, cy = center
    return {
        "dx_min": cx - radius,
        "dx_max": cx + radius,
        "dy_min": cy - radius,
        "dy_max": cy + radius,
    }
