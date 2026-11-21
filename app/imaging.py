"""Strict PNG decoding for binary (0/255) 8-bit grayscale images.

Both the dimension line and the glyph samples are 8-bit grayscale,
non-interlaced PNGs whose pixels take only the values 0 (ink) and 255
(paper).  The dimension line is at most 32 rows by 192 columns; every image
in one request must share the same height, which is checked by the service
layer.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import numpy as np
from PIL import Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

LINE_MAX_ROWS = 32
LINE_MAX_COLS = 192
ALLOWED_PIXEL_VALUES = frozenset({0, 255})


class ImageDecodeError(ValueError):
    """Raised when an uploaded image is not an acceptable binary gray PNG."""


@dataclass(frozen=True)
class DecodedImage:
    image_id: str
    array: np.ndarray  # uint8, shape (height, width), values in {0, 255}

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def ink_mask(self) -> np.ndarray:
        """Boolean mask, True where the pixel is ink (0)."""
        return self.array == 0


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


def decode_grayscale_png(image_id: str, data: bytes) -> DecodedImage:
    """Decode and validate one uploaded binary grayscale PNG."""
    if not data:
        raise ImageDecodeError(f"{image_id} 是空文件")
    width, height, bit_depth, colour_type, interlace = _png_header(data)
    if bit_depth != 8:
        raise ImageDecodeError(f"{image_id} 的位深为 {bit_depth}，只接受 8 位灰度 PNG")
    if colour_type != 0:
        raise ImageDecodeError(
            f"{image_id} 的颜色类型为 {colour_type}，只接受单通道灰度 PNG（颜色类型 0），"
            "不接受调色板、RGB 或带 Alpha 的 PNG"
        )
    if interlace != 0:
        raise ImageDecodeError(f"{image_id} 使用了隔行扫描，请重新导出为非隔行 PNG")
    if width < 1 or height < 1:
        raise ImageDecodeError(f"{image_id} 尺寸为 0")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            if im.format != "PNG":
                raise ImageDecodeError(f"{image_id} 不是 PNG")
            array = np.asarray(im, dtype=np.uint8)
    except ImageDecodeError:
        raise
    except Exception as exc:  # noqa: BLE001 - report as a validation error
        raise ImageDecodeError(f"{image_id} 解码失败：{exc}") from exc
    if array.ndim != 2 or array.dtype != np.uint8:
        raise ImageDecodeError(f"{image_id} 不是单通道 8 位灰度图")
    if array.shape != (height, width):
        raise ImageDecodeError(f"{image_id} 的实际尺寸与 PNG 文件头不符")
    unique = np.unique(array)
    if not set(unique.tolist()).issubset(ALLOWED_PIXEL_VALUES):
        raise ImageDecodeError(
            f"{image_id} 含非 0/255 的像素值（发现 {sorted(set(unique.tolist()) - set(ALLOWED_PIXEL_VALUES))[:5]}），"
            "只接受纯黑白二值图"
        )
    return DecodedImage(image_id=image_id, array=array)
