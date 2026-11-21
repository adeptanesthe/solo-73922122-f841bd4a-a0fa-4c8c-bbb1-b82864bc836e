"""Recognition orchestration: cross-image validation, solving, verdict."""

from __future__ import annotations

from .imaging import DecodedImage
from .models import (
    CharSpan,
    RecognizeResponse,
    RecognizeSpec,
    Reading,
)
from .solver import GlyphSample, RecognitionResult, canonical_value, recognize


class RequestValidationError(ValueError):
    """A semantically invalid request (maps to HTTP 422)."""


def _validate_images(
    line: DecodedImage, glyphs: list[DecodedImage], spec: RecognizeSpec
) -> None:
    if line.height > 32 or line.width > 192:
        raise RequestValidationError(
            f"尺寸图尺寸为 {line.width}x{line.height}，上限为 192 列 x 32 行"
        )
    for idx, glyph in enumerate(glyphs):
        label = spec.glyph_labels[idx]
        if glyph.height != line.height:
            raise RequestValidationError(
                f"字样 {idx}（标注 {label!r}）高 {glyph.height} 像素，"
                f"与尺寸图高度 {line.height} 不一致；所有图像必须同高"
            )
        if glyph.width > line.width:
            raise RequestValidationError(
                f"字样 {idx}（标注 {label!r}）宽 {glyph.width} 像素，"
                f"超过尺寸图宽度 {line.width}，无法参与比对"
            )
        if not glyph.ink_mask.any():
            raise RequestValidationError(
                f"字样 {idx}（标注 {label!r}）是全白图，不含任何墨点，无法作为字样"
            )


def _to_response(
    result: RecognitionResult,
    spec: RecognizeSpec,
    line: DecodedImage,
) -> RecognizeResponse:
    candidates = [
        Reading(
            value=canonical_value(r.raw_text),
            raw_text=r.raw_text,
            mismatches=r.mismatches,
            chars=[
                CharSpan(
                    char=s.char,
                    start_col=s.start_col,
                    end_col=s.end_col,
                    mismatches=s.mismatches,
                )
                for s in r.chars
            ],
        )
        for r in result.readings
    ]

    if not candidates:
        reasons = ["no_reading"]
        message = "不存在满足字样与切分规则的合法读数，转人工复核。"
        if result.truncated:
            reasons.append("search_truncated")
            message += " 同时搜索预算耗尽，无解结论亦未获完整证明。"
        return RecognizeResponse(
            status="needs_review",
            message=message,
            review_reasons=reasons,
            line_width=line.width,
            line_height=line.height,
            max_mismatch=spec.max_mismatch,
            min_margin=spec.min_margin,
            candidates=[],
            search_truncated=result.truncated,
        )

    best = candidates[0]
    reasons: list[str] = []
    if best.mismatches > spec.max_mismatch:
        reasons.append("mismatch_limit_exceeded")
    if len(candidates) >= 2:
        margin = candidates[1].mismatches - best.mismatches
        if margin < spec.min_margin:
            reasons.append("margin_too_small")
    if result.truncated:
        reasons.append("search_truncated")

    if reasons:
        detail = {
            "mismatch_limit_exceeded": (
                f"最佳错点 {best.mismatches} 超过上限 {spec.max_mismatch}"
            ),
            "margin_too_small": (
                f"前两名错点差 {candidates[1].mismatches - best.mismatches} "
                f"低于阈值 {spec.min_margin}"
            ),
            "search_truncated": "候选切分过多，搜索预算耗尽，最优性未获证明",
        }
        message = "转人工复核：" + "；".join(detail[r] for r in reasons) + "。"
        return RecognizeResponse(
            status="needs_review",
            message=message,
            review_reasons=reasons,
            line_width=line.width,
            line_height=line.height,
            max_mismatch=spec.max_mismatch,
            min_margin=spec.min_margin,
            candidates=candidates,
            search_truncated=result.truncated,
        )

    message = f"最佳读数 {best.raw_text}（数值 {best.value}），错点 {best.mismatches}，可直接录入。"
    if len(candidates) == 1:
        message += " 不存在第二个不同的合法数值。"
    else:
        message += (
            f" 次选数值 {candidates[1].value}，错点 {candidates[1].mismatches}，"
            f"错点差 {candidates[1].mismatches - best.mismatches}。"
        )
    return RecognizeResponse(
        status="entry_ready",
        message=message,
        review_reasons=[],
        line_width=line.width,
        line_height=line.height,
        max_mismatch=spec.max_mismatch,
        min_margin=spec.min_margin,
        candidates=candidates,
        search_truncated=False,
    )


def recognize_line(
    line: DecodedImage, glyphs: list[DecodedImage], spec: RecognizeSpec
) -> RecognizeResponse:
    _validate_images(line, glyphs, spec)
    samples = [
        GlyphSample(
            label=spec.glyph_labels[idx],
            width=glyph.width,
            ink=glyph.ink_mask,
        )
        for idx, glyph in enumerate(glyphs)
    ]
    result = recognize(line.ink_mask, samples)
    return _to_response(result, spec, line)
