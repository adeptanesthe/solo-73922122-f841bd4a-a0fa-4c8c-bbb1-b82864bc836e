# 图纸尺寸识别 API

旧图纸上的数字常常粘连在一起。本服务接收一张**裁成单行的尺寸图**和至多 33 张
标有 `0～9` 或小数点的字样图（均为同高的 8 位灰度、纯黑白二值 PNG，像素只取
`0` 墨 / `255` 纸），通过任意列切分 + 与等宽字样逐像素比对，给出错点最少的至多
两个不同数值，并按给定阈值判定**可录入**还是**待复核**。

技术栈：FastAPI + NumPy（Pillow 仅用于 PNG 解码）。提供 Docker Compose 与可交互的
`/docs`，并内置一套演示样图，可直接在 Swagger 页面试读。

## 快速启动

```bash
docker compose up --build
# 打开 http://localhost:8000/docs
```

在 `/docs` 页面：

1. 先调 `GET /demo/line` 下载演示尺寸图（粘连的 `12.50`，带两个像素缺陷）；
2. 调 `GET /demo/glyphs/{label}` 下载 11 张字样（label 取 `0`..`9` 或 `dot`）；
3. 调 `GET /demo/spec` 取得配套 JSON；
4. 在 `POST /recognize` 点 **Try it out**，上传尺寸图与全部字样，把 spec 粘进
   `spec` 字段，预期返回 `status = "entry_ready"`、最佳数值 `12.5`。

不用 Docker 时：

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
python scripts/make_demo.py demo          # 生成样图与 curl 请求脚本
bash demo/request.sh
python scripts/demo_client.py             # 或直接走 HTTP 的演示客户端
pytest
```

## 接口

`POST /recognize`，`multipart/form-data`：

| 字段 | 说明 |
| --- | --- |
| `line` | 1 张裁成单行的尺寸图：8 位灰度 PNG（颜色类型 0、非隔行），像素只取 0/255，至多 32 行、192 列。 |
| `glyphs` | 1～33 张与尺寸图**同高**的字样 PNG（同为 0/255 二值灰度），每张含一个 0～9 或小数点字样，不能全白、宽度不超过尺寸图。 |
| `spec` | JSON 字符串，结构见下。 |

辅助端点：`GET /health`、`GET /docs`、`GET /redoc`、`GET /openapi.json`，
以及演示用的 `GET /demo/line`、`GET /demo/glyphs/{label}`、`GET /demo/spec`。

### spec 结构

```json
{
  "glyph_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."],
  "max_mismatch": 12,
  "min_margin": 3
}
```

* `glyph_labels`：按字样**上传顺序**给出的标注，元素只能是 `"0"`..`"9"` 或 `"."`，
  长度必须与字样图张数一致（1～33）。同一标注允许出现多张字样，比对时取错点最少者。
* `max_mismatch`：非负整数。最佳读数的整行错点数超过它即待复核。
* `min_margin`：**正整数**。前两名不同数值的错点差低于它即待复核。

## 识别规则

1. **切分**：可在任意列间切分；段间与首尾的全白列允许跳过，但**含墨列必须被某个
   字符段覆盖**（跳过不能用于绕过墨点），每个字符段本身必须含墨。
2. **比对**：字符段只与**等宽**字样逐像素比较，段代价为 XOR 错点数；同标注多样本取
   最小者。整行代价为各字符代价之和。
3. **合法读数**：含 1～8 个数字；至多一个小数点，且小数点两侧都必须有数字；允许
   前导零和小数末尾的零。
4. **数值归并**：同一数值（如 `1.5`、`01.50`、`1.500`）只保留错点最少的写法与切分；
   再按错点数升序返回至多两个**不同数值**，各附识别原文、总错点和逐字符列范围
   （`start_col` 含、`end_col` 不含）。
5. **判定**：

   | 结论 | 条件 |
   | --- | --- |
   | `needs_review`（待复核） | 无合法读数；或最佳错点 > `max_mismatch`；或前两名错点差 < `min_margin`；或搜索预算耗尽 |
   | `entry_ready`（可录入） | 其余情况，最佳数值可直接录入 |

   注意只有一个合法数值时不存在"前两名"，只要错点不超限即可录入。同代价并列时
   错点差为 0，必然低于任何正整数 `min_margin`，因此并列天然转复核。

### 响应示例

```json
{
  "status": "entry_ready",
  "message": "最佳读数 12.50（数值 12.5），错点 2，可直接录入。……",
  "review_reasons": [],
  "line_width": 30,
  "line_height": 7,
  "max_mismatch": 12,
  "min_margin": 3,
  "candidates": [
    {
      "value": "12.5",
      "raw_text": "12.50",
      "mismatches": 2,
      "chars": [
        {"char": "1", "start_col": 2, "end_col": 7, "mismatches": 0},
        {"char": "2", "start_col": 7, "end_col": 12, "mismatches": 2},
        {"char": ".", "start_col": 12, "end_col": 14, "mismatches": 0},
        {"char": "5", "start_col": 15, "end_col": 20, "mismatches": 0},
        {"char": "0", "start_col": 21, "end_col": 26, "mismatches": 0}
      ]
    }
  ],
  "search_truncated": false
}
```

错误返回：`400` 图像非法（非 PNG、非 8 位灰度、非颜色类型 0、含非 0/255 像素、
字样超 33 张等）或 `spec` 不是 JSON；`422` spec 结构不合法、标注数量与字样不匹配、
图像高度不一致、字样全白或尺寸图超 32×192。

## 实现要点

* **严格解码**：先读 PNG `IHDR` 校验位深=8、颜色类型=0、非隔行，再用 Pillow 解码并
  核对实际尺寸与像素集合 `⊆ {0, 255}`，拒绝 16 位、调色板、RGB、Alpha 与灰度中间值。
* **段代价表**：按字样宽度分组，用 NumPy 滑动窗口一次性取出所有等宽段，XOR 求和
  得到 `(起始列 → 候选标签/代价)`；段内无墨的段直接丢弃。
* **可行自动机 + 反向 DP**：状态为 `(列, 已用数字数, 模式)`，模式区分纯整数、
  小数点后待数字、小数部分；反向 DP 给出每个状态到行尾的最便宜合法完成代价
  （即可采纳启发值 h），同时直接证明"无合法读数"。
* **A\* 最优枚举**：按 `g + h` 堆式展开，保证最便宜的读数先完成；对同一状态只保留
  9 条最便宜路径做支配剪枝（同一数值在固定数字数下至多 8 种点位置写法，9 条足以
  保住前两个不同数值的精确代价），堆推送有硬预算，超限即如实标记 `search_truncated`
  并转复核——宁可说不确定，也不谎报最优。
* **确定性**：并列按 `(错点数, 规范数值, 原文, 列范围)` 排序，重复请求结果稳定。

## 目录

```
app/
  main.py       FastAPI 路由、spec 解析、错误映射、/demo/* 样图端点
  models.py     Pydantic 请求/响应模型与上限常量
  imaging.py    严格 PNG 解码（8 位灰度、0/255 校验）
  solver.py     段代价表、反向 DP、A* 枚举与支配剪枝、数值规范化
  service.py    跨图校验、求解编排、录入/复核判定
  demo_font.py  内置 5x7 点阵字体与演示图生成
tests/          44 个测试（求解器单测含暴力参照与随机性质测试、HTTP 端到端）
scripts/        make_demo.py 生成样图与请求脚本；demo_client.py HTTP 演示客户端
Dockerfile      python:3.11-slim、非 root、HEALTHCHECK
docker-compose.yml  端口 8000、只读根文件系统、健康检查、资源限额
```

## 已知限制

* 字样与尺寸图必须同高，且只支持逐像素等宽比对，不做缩放、旋转或粗细归一化；
  扫描变形需在裁切/采样端消除。
* 只识别单行、至多 8 个数字的读数；超过该长度或无法被字符段覆盖的墨列会导致
  无合法读数，按设计转人工复核。
* 同代价并列的不同数值不做主观取舍，一律通过 `min_margin` 规则转复核。
