"""Strict PNG decoding and image metadata."""

from __future__ import annotations

import hashlib
import io
import struct
from dataclasses import dataclass

import numpy as np
from PIL import Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_SIDE = 1024


class ImageDecodeError(ValueError):
    """Raised when an uploaded image is not an acceptable 8-bit grayscale PNG."""


@dataclass(frozen=True)
class DecodedImage:
    image_id: int
    array: np.ndarray  # uint8, shape (height, width)
    sha256: str

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def height(self) -> int:
        return int(self.array.shape[0])


def _png_header(data: bytes) -> tuple[int, int, int, int, int]:
    """Return (width, height, bit_depth, colour_type, interlace) from IHDR."""
    if len(data) < 33 or not data.startswith(PNG_SIGNATURE):
        raise ImageDecodeError("文件不是 PNG")
    length, chunk_type = struct.unpack(">I4s", data[8:16])
    if chunk_type != b"IHDR" or length != 13:
        raise ImageDecodeError("PNG 缺少有效的 IHDR")
    width, height, bit_depth, colour_type, _comp, _filt, interlace = struct.unpack(
        ">IIBBBBB", data[16:29]
    )
    return width, height, bit_depth, colour_type, interlace


def decode_grayscale_png(image_id: int, data: bytes) -> DecodedImage:
    """Decode and validate one uploaded tile.

    Only non-interlaced, colour-type-0 (grayscale), 8-bit PNGs up to 1024 px per
    side are accepted.  Dimensions and format are checked both from the raw PNG
    header and from Pillow's decode result.
    """
    if not data:
        raise ImageDecodeError(f"图 {image_id} 是空文件")
    width, height, bit_depth, colour_type, interlace = _png_header(data)
    if bit_depth != 8:
        raise ImageDecodeError(
            f"图 {image_id} 的位深为 {bit_depth}，只接受 8 位灰度 PNG"
        )
    if colour_type != 0:
        raise ImageDecodeError(
            f"图 {image_id} 的颜色类型为 {colour_type}，只接受单通道灰度 PNG（颜色类型 0），"
            "不接受调色板、RGB 或带 Alpha 的 PNG"
        )
    if interlace != 0:
        raise ImageDecodeError(f"图 {image_id} 使用了隔行扫描，请重新导出为非隔行 PNG")
    if width < 1 or height < 1:
        raise ImageDecodeError(f"图 {image_id} 尺寸为 0")
    if width > MAX_SIDE or height > MAX_SIDE:
        raise ImageDecodeError(
            f"图 {image_id} 尺寸 {width}x{height} 超过每边 {MAX_SIDE} 像素的上限"
        )
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            if im.format != "PNG":
                raise ImageDecodeError(f"图 {image_id} 不是 PNG")
            array = np.asarray(im, dtype=np.uint8)
    except ImageDecodeError:
        raise
    except Exception as exc:  # noqa: BLE001 - report as a validation error
        raise ImageDecodeError(f"图 {image_id} 解码失败：{exc}") from exc
    if array.ndim != 2 or array.dtype != np.uint8:
        raise ImageDecodeError(f"图 {image_id} 不是单通道 8 位灰度图")
    if array.shape != (height, width):
        raise ImageDecodeError(f"图 {image_id} 的实际尺寸与 PNG 文件头不符")
    return DecodedImage(
        image_id=image_id,
        array=array,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def overlap_rect(
    a_w: int, a_h: int, b_w: int, b_h: int, dx: int, dy: int
) -> tuple[int, int, int, int] | None:
    """Overlap rectangle in a's frame for "b's origin at (dx, dy) of a".

    Returns (x0, y0, x1, y1) as a half-open rectangle, or None if disjoint.
    """
    x0 = max(0, dx)
    y0 = max(0, dy)
    x1 = min(a_w, dx + b_w)
    y1 = min(a_h, dy + b_h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1
