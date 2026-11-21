"""FastAPI application for recognizing dimension values on scanned drawings."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from pydantic import ValidationError

from . import __version__
from .demo_font import (
    all_glyph_labels,
    demo_line_png,
    demo_spec,
    glyph_png,
)
from .imaging import ImageDecodeError, decode_grayscale_png
from .models import MAX_GLYPHS, RecognizeResponse, RecognizeSpec
from .service import RequestValidationError, recognize_line

DESCRIPTION = """
把裁成**单行**的尺寸图与标有 0～9 或小数点的字样图逐像素比对，给出可直接录入的数值。

### 输入（`multipart/form-data`）
* `line`：1 张裁成单行的尺寸图，**8 位灰度 PNG**（颜色类型 0、非隔行），
  像素只取 0（墨）或 255（纸），至多 32 行、192 列。
* `glyphs`：1～33 张同高的字样 PNG（同样为 0/255 二值灰度图），每张含一个
  0～9 或小数点字样；其标注由 `spec.glyph_labels` 按上传顺序给出。
* `spec`：JSON 字符串：
  - `glyph_labels`：与字样上传顺序一一对应的标注数组；
  - `max_mismatch`：错点总数上限（非负整数）；
  - `min_margin`：前两名错点差的复核阈值（正整数）。

### 识别规则
可在任意列间切分；段间与首尾的全白列允许跳过，每个含墨列段与**等宽**字样
逐像素比对，整行错点为各字符错点之和。合法读数含 1～8 个数字，至多一个
小数点且小数点两侧都必须有数字；允许前导零和小数末尾的零。同一数值只保留
错点最少的写法与切分。

### 判定
返回错点最少的至多两个**不同数值**。出现以下任一情况即 `needs_review`：
无合法读数；最佳错点超过 `max_mismatch`；前两名错点差低于 `min_margin`；
搜索预算耗尽。否则为 `entry_ready`，最佳数值可录入。

### 试用
先调用 `GET /demo/line`、`GET /demo/glyphs/{label}` 下载样图，
`GET /demo/spec` 取配套 spec，再在本页 `/recognize` 的 **Try it out** 中上传。
"""


app = FastAPI(
    title="图纸尺寸识别 API",
    version=__version__,
    description=DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={"name": "档案数字化"},
)

PNG_HEADERS = {"Cache-Control": "no-store"}


@app.get("/health", tags=["meta"], summary="存活检查")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get(
    "/demo/line",
    tags=["demo"],
    summary="演示尺寸图（粘连的 12.50，含两个像素缺陷）",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def demo_line() -> Response:
    return Response(content=demo_line_png(), media_type="image/png", headers=PNG_HEADERS)


@app.get(
    "/demo/glyphs/{label}",
    tags=["demo"],
    summary="演示字样图：label 取 0~9 或小数点（用 dot）",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}, 404: {"description": "label 不合法"}},
)
def demo_glyph(label: str) -> Response:
    if label == "dot":
        label = "."
    if label not in all_glyph_labels():
        raise HTTPException(status_code=404, detail="label 只允许 0~9 或 dot（小数点）")
    return Response(content=glyph_png(label), media_type="image/png", headers=PNG_HEADERS)


@app.get("/demo/spec", tags=["demo"], summary="演示请求的 spec JSON")
def demo_spec_endpoint() -> dict:
    return demo_spec()


def _parse_spec(spec_text: str) -> RecognizeSpec:
    try:
        raw = json.loads(spec_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"spec 不是合法 JSON：{exc}") from exc
    try:
        return RecognizeSpec.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=json.loads(exc.json(include_url=False, include_context=False)),
        ) from exc


MAX_UPLOAD_BYTES = 1 * 1024 * 1024


def _read_upload(upload: UploadFile, name: str) -> bytes:
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400, detail=f"{name} 超过 {MAX_UPLOAD_BYTES // 1024} KiB 上限"
        )
    return data


def _check_png_content_type(upload: UploadFile, name: str) -> None:
    ctype = (upload.content_type or "").split(";")[0].strip().lower()
    if ctype not in ("image/png", "application/octet-stream", ""):
        raise HTTPException(
            status_code=400,
            detail=f"{name} 的 Content-Type 为 {upload.content_type!r}，只接受 image/png",
        )


@app.post(
    "/recognize",
    response_model=RecognizeResponse,
    tags=["recognition"],
    summary="识别单行尺寸图并给出录入/复核结论",
    response_description="识别结论；HTTP 200 时业务结论见 `status` 字段",
    responses={
        400: {"description": "请求格式或图像不符合要求（非 PNG、非 8 位灰度、非 0/255 等）"},
        422: {"description": "spec 结构不合法或图像间约束不满足（高度不一致等）"},
    },
)
def recognize_endpoint(
    line: Annotated[UploadFile, File(description="裁成单行的尺寸图 PNG")],
    glyphs: Annotated[list[UploadFile], File(description="1～33 张字样 PNG，顺序与 glyph_labels 对应")],
    spec: Annotated[str, Form(description="RecognizeSpec JSON 字符串（见接口说明）")],
) -> RecognizeResponse:
    parsed = _parse_spec(spec)

    if not glyphs:
        raise HTTPException(status_code=400, detail="至少需要 1 张字样图")
    if len(glyphs) > MAX_GLYPHS:
        raise HTTPException(status_code=400, detail=f"字样至多 {MAX_GLYPHS} 张，收到 {len(glyphs)} 张")
    if len(parsed.glyph_labels) > MAX_GLYPHS:
        raise HTTPException(
            status_code=422,
            detail=f"glyph_labels 至多 {MAX_GLYPHS} 个，收到 {len(parsed.glyph_labels)} 个",
        )
    if len(glyphs) != len(parsed.glyph_labels):
        raise HTTPException(
            status_code=422,
            detail=(
                f"字样图 {len(glyphs)} 张，但 glyph_labels 有 {len(parsed.glyph_labels)} 个标注；"
                "两者必须按上传顺序一一对应"
            ),
        )
    invalid = [label for label in parsed.glyph_labels if label not in set(all_glyph_labels())]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"glyph_labels 只允许 0~9 或 \".\"，发现非法标注：{invalid[:5]}",
        )
    # A dot-only label set is structurally legal; the solver simply finds no
    # legal reading for it because a dot needs digits on both sides.

    _check_png_content_type(line, "尺寸图")
    line_data = _read_upload(line, "尺寸图")
    try:
        line_img = decode_grayscale_png("尺寸图", line_data)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    glyph_imgs = []
    for idx, upload in enumerate(glyphs):
        _check_png_content_type(upload, f"字样 {idx}")
        data = _read_upload(upload, f"字样 {idx}")
        try:
            glyph_imgs.append(decode_grayscale_png(f"字样 {idx}", data))
        except ImageDecodeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        return recognize_line(line_img, glyph_imgs, parsed)
    except RequestValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
