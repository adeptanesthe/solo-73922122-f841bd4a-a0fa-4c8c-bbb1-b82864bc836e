"""FastAPI application for locating scan tiles of a large-format drawing."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from . import __version__
from .imaging import ImageDecodeError, decode_grayscale_png
from .models import MAX_IMAGES, MAX_PAIRS, MIN_IMAGES, AlignResponse, AlignSpec
from .service import align

DESCRIPTION = """
把一张大幅图纸切块扫描后，用**已声明的重叠图对**与各自的整数平移搜索窗，
确定各块在整张图上的位置。

### 输入
* `images`：3～5 张 **8 位灰度 PNG**（颜色类型 0、非隔行、每边 ≤ 1024 像素）。
  上传顺序即图号 0..n-1，图 0 为定位基准。
* `spec`：JSON 字符串，见下方 schema。包含
  - `pairs`：哪些图对重叠。每对给出 `a`、`b` 以及 `window`
    （`dx_min/dx_max/dy_min/dy_max`，闭区间，至多 81 个整数位置）或
    `candidates`（至多 81 个显式整数平移）；
  - `min_overlap_pixels`：整组共用的最少重叠像素数；
  - `max_mean_abs_diff`：整组共用的平均绝对灰度差上限。

### 坐标约定
第 i 张图左上角在图面上的坐标为 `(x_i, y_i)`；图对 `(a, b)` 的平移 `(dx, dy)`
表示 **b 的左上角位于 a 的坐标 `(dx, dy)`** 处，即
`(x_b, y_b) = (x_a + dx, y_a + dy)`。返回坐标以首图左上角为 `(0, 0)`。

### 结果
* `aligned`：所有图对同时达标且相对布局唯一，返回各图位置与拼接外框。
* `disconnected`：有图块无法经指定图对连到首图；`rescan_image_id` 指出建议补扫的一块。
* `no_solution`：无法同时对齐；`failing_pairs` 给出未达标图对及原因。
* `ambiguous`：多组位置方案都达标；`uncertain_image_ids` 指出位置不确定的图，
  `alternative_layouts` 给出至少两组示例。

### 示例 spec
```json
{
  "pairs": [
    {"a": 0, "b": 1, "window": {"dx_min": 446, "dx_max": 454, "dy_min": -2, "dy_max": 2}},
    {"a": 1, "b": 2, "window": {"dx_min": -454, "dx_max": -446, "dy_min": 448, "dy_max": 452}}
  ],
  "min_overlap_pixels": 20000,
  "max_mean_abs_diff": 3.0
}
```
在上方 `/docs` 页面点击 **Try it out**，先添加 3 张图，再把该 JSON（或 `scripts/make_demo.py`
生成的 `demo/spec.json`）粘贴到 `spec` 字段即可试用。窗口面积两类写法都受 81 个位置的上限约束：
矩形窗口按 `(dx_max-dx_min+1)*(dy_max-dy_min+1) ≤ 81` 计，`candidates` 按条数计。
"""


app = FastAPI(
    title="图纸分块定位 API",
    version=__version__,
    description=DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={"name": "档案数字化"},
)


@app.get("/health", tags=["meta"], summary="存活检查")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


def _parse_spec(spec_text: str, n_images: int) -> AlignSpec:
    try:
        raw = json.loads(spec_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"spec 不是合法 JSON：{exc}") from exc
    try:
        parsed = AlignSpec.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=json.loads(exc.json(include_url=False, include_context=False)),
        ) from exc
    if len(parsed.pairs) > MAX_PAIRS:
        raise HTTPException(status_code=422, detail=f"图对数量至多 {MAX_PAIRS}")
    for pid, pair in enumerate(parsed.pairs):
        if pair.a >= n_images or pair.b >= n_images:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"图对 {pid} 引用了不存在的图号 {max(pair.a, pair.b)}；"
                    f"本次上传了 {n_images} 张图（图号 0..{n_images - 1}）"
                ),
            )
    return parsed


MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # per tile; a 1024x1024 gray PNG is far below this


def _read_upload(upload: UploadFile, idx: int) -> bytes:
    """Read one upload with a hard size cap so a huge file cannot exhaust RAM."""
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"图 {idx} 超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB 上限",
        )
    return data


@app.post(
    "/align",
    response_model=AlignResponse,
    tags=["alignment"],
    summary="对齐分块扫描图并返回拼接坐标",
    response_description="对齐结果；HTTP 200 时结果的业务状态见 `status` 字段",
    responses={
        400: {"description": "请求格式或图像不符合要求"},
        422: {"description": "spec 结构或图号引用不合法"},
    },
)
def align_endpoint(
    images: Annotated[list[UploadFile], File(description="3～5 张 8 位灰度 PNG，按图号顺序上传")],
    spec: Annotated[str, Form(description="AlignSpec JSON 字符串（见接口说明）")],
) -> AlignResponse:
    if not (MIN_IMAGES <= len(images) <= MAX_IMAGES):
        raise HTTPException(
            status_code=400,
            detail=f"需要 {MIN_IMAGES}～{MAX_IMAGES} 张图，收到 {len(images)} 张",
        )

    parsed = _parse_spec(spec, len(images))

    decoded = []
    for idx, upload in enumerate(images):
        ctype = (upload.content_type or "").split(";")[0].strip().lower()
        if ctype not in ("image/png", "application/octet-stream", ""):
            raise HTTPException(
                status_code=400,
                detail=f"图 {idx} 的 Content-Type 为 {upload.content_type!r}，只接受 image/png",
            )
        data = _read_upload(upload, idx)
        try:
            decoded.append(decode_grayscale_png(idx, data))
        except ImageDecodeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return align(decoded, parsed)
