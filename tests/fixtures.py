"""Synthetic binary line/glyph fixtures for recognition tests."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.demo_font import GAP_WIDTH, glyph_bitmap

HEIGHT = 7
DIGIT_WIDTH = 5
DOT_WIDTH = 2


def glyph_samples(labels: list[str] | None = None) -> dict[str, np.ndarray]:
    """Clean sample arrays for the requested labels (all 11 by default)."""
    labels = labels if labels is not None else [str(d) for d in range(10)] + ["."]
    return {label: np.where(glyph_bitmap(label), 0, 255).astype(np.uint8) for label in labels}


def glyph_samples_arrays(labels: list[str] | None = None) -> list[np.ndarray]:
    samples = glyph_samples(labels)
    labels = labels or list(samples)
    return [samples[label] for label in labels]


def render_text(
    text: str,
    *,
    stuck_after: set[int] | None = None,
    flips: list[tuple[int, int]] | None = None,
    margin: int = 2,
    samples: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Render a reading as a 0/255 grayscale line array."""
    samples = samples or glyph_samples()
    stuck_after = stuck_after or set()
    white = np.full((HEIGHT, GAP_WIDTH), 255, dtype=np.uint8)
    margin_col = np.full((HEIGHT, margin), 255, dtype=np.uint8)
    parts = [margin_col]
    for idx, ch in enumerate(text):
        parts.append(samples[ch].copy())
        if idx != len(text) - 1 and idx not in stuck_after:
            parts.append(white.copy())
    parts.append(margin_col)
    line = np.concatenate(parts, axis=1)
    for row, col in flips or []:
        line[row, col] = 255 - line[row, col]
    return line


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
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint16) * 257).save(buf, format="PNG")
    return buf.getvalue()


def png_bytes_nonbinary(arr: np.ndarray) -> bytes:
    """An 8-bit gray PNG containing a value other than 0/255."""
    graded = arr.copy()
    graded[0, 0] = 128
    return png_bytes(graded)


def default_spec(max_mismatch: int = 20, min_margin: int = 3) -> dict:
    return {
        "glyph_labels": [str(d) for d in range(10)] + ["."],
        "max_mismatch": max_mismatch,
        "min_margin": min_margin,
    }
