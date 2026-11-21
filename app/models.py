"""Pydantic request/response models for the dimension recognition API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

MAX_GLYPHS = 33
MAX_DIGITS = 8
MAX_MISMATCH_LIMIT = 100_000


class RecognizeSpec(BaseModel):
    """Recognition metadata, sent as a JSON string in the multipart form.

    ``glyph_labels[i]`` is the label (``"0"``..``"9"`` or ``"."``) of the i-th
    glyph upload, so labels follow upload order instead of file names.
    """

    model_config = ConfigDict(extra="forbid")

    glyph_labels: list[str] = Field(
        min_length=1,
        description="按上传顺序给出每张字样的标注，取值为 0~9 或 \".\"；长度须与字样图张数一致（至多 33）",
    )
    max_mismatch: StrictInt = Field(
        ge=0,
        le=MAX_MISMATCH_LIMIT,
        description="允许录入的整行错点（像素不一致）总数上限",
    )
    min_margin: StrictInt = Field(
        ge=1,
        description="前两名不同数值错点数之差低于该正整数时转人工复核",
    )


class CharSpan(BaseModel):
    """One recognized character and the columns it occupies on the line."""

    char: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]
    start_col: StrictInt = Field(ge=0, description="起始列（含，0 起）")
    end_col: StrictInt = Field(ge=1, description="结束列（不含），宽度 = end_col - start_col")
    mismatches: StrictInt = Field(ge=0, description="该字符与等宽字样比对的错点数")


class Reading(BaseModel):
    """One distinct numeric reading, with its cheapest spelling/segmentation."""

    value: str = Field(description="规范化数值（去前导零、保留小数点与末尾零），如 12.50 -> 12.5")
    raw_text: str = Field(description="识别原文，含小数点与前导/末尾零，如 \"012.50\"")
    mismatches: StrictInt = Field(ge=0, description="整行错点总数（各字符错点之和）")
    chars: list[CharSpan] = Field(description="按原文顺序给出各字符的列范围与错点")


class RecognizeResponse(BaseModel):
    status: Literal["entry_ready", "needs_review"]
    message: str
    review_reasons: list[
        Literal["no_reading", "mismatch_limit_exceeded", "margin_too_small", "search_truncated"]
    ] = Field(default_factory=list, description="status=needs_review 时的原因（可多个）")
    line_width: StrictInt = Field(ge=1)
    line_height: StrictInt = Field(ge=1)
    max_mismatch: StrictInt
    min_margin: StrictInt
    candidates: list[Reading] = Field(
        default_factory=list,
        description="错点最少的至多两个不同数值，按错点数升序、平局按原文确定性排序",
    )
    search_truncated: bool = False
