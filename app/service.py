"""Alignment orchestration: validation, pairwise scoring, layout solving."""

from __future__ import annotations

import numpy as np

from .alignment import PairScan, overlap_rect, scan_pair
from .imaging import DecodedImage
from .models import (
    AlignResponse,
    AlignSpec,
    ImageInfo,
    ImagePosition,
    MosaicBounds,
    PairFailure,
    PairResult,
)
from .solver import solve


def _connected_from_first(n_images: int, pairs) -> list[int]:
    """Return image ids unreachable from image 0 via the declared pairs."""
    adjacency: dict[int, list[int]] = {i: [] for i in range(n_images)}
    for p in pairs:
        adjacency[p.a].append(p.b)
        adjacency[p.b].append(p.a)
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adjacency[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return [i for i in range(n_images) if i not in seen]


def _pick_rescan_image(unreachable: list[int], pairs) -> int:
    """Name one tile to rescan: the least-connected unreachable tile.

    A tile with no declared pair at all is the clearest candidate; ties are
    broken by lowest upload index so the answer is deterministic.
    """
    degree = {i: 0 for i in unreachable}
    for p in pairs:
        for idx in (p.a, p.b):
            if idx in degree:
                degree[idx] += 1
    return min(unreachable, key=lambda i: (degree[i], i))


def _mosaic_bounds(images: list[DecodedImage], positions) -> MosaicBounds:
    x_min = min(x for x, _ in positions)
    y_min = min(y for _, y in positions)
    x_max = max(x + img.width for (x, _), img in zip(positions, images))
    y_max = max(y + img.height for (_, y), img in zip(positions, images))
    return MosaicBounds(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        width=x_max - x_min,
        height=y_max - y_min,
    )


def _pair_mad(
    a: DecodedImage, b: DecodedImage, dx: int, dy: int
) -> tuple[int, float]:
    """Overlap pixel count and MAD at a chosen offset (used to report results)."""
    rect = overlap_rect(a.width, a.height, b.width, b.height, dx, dy)
    if rect is None:  # pragma: no cover - a solved layout always overlaps
        return 0, float("nan")
    x0, y0, x1, y1 = rect
    sub_a = a.array[y0:y1, x0:x1].astype(np.int16)
    sub_b = b.array[y0 - dy : y1 - dy, x0 - dx : x1 - dx].astype(np.int16)
    n = (x1 - x0) * (y1 - y0)
    return n, float(np.abs(sub_a - sub_b).sum(dtype=np.int64)) / n


def align(images: list[DecodedImage], spec: AlignSpec) -> AlignResponse:
    n = len(images)
    infos = [
        ImageInfo(image_id=im.image_id, width=im.width, height=im.height) for im in images
    ]

    # 1) Link graph over the declared pairs.  If a tile cannot reach image 0,
    #    no scoring can produce a global layout, so ask for a rescan first.
    unreachable = _connected_from_first(n, spec.pairs)
    if unreachable:
        rescan = _pick_rescan_image(unreachable, spec.pairs)
        return AlignResponse(
            status="disconnected",
            layout_unique=False,
            message=(
                "有图块无法经指定图对连到首图，无法确定整张图的位置关系。"
                f" 未能连通的图：{unreachable}。请补扫图 {rescan}，"
                "使其与已连通的图块有重叠区域。"
            ),
            images=infos,
            unreachable_image_ids=unreachable,
            rescan_image_id=rescan,
        )

    # 2) Score every declared pair independently against the shared thresholds.
    i16 = [im.array.astype(np.int16) for im in images]
    scans: list[PairScan] = [
        scan_pair(
            pid,
            images[pair.a],
            images[pair.b],
            pair.offsets(),
            spec.min_overlap_pixels,
            spec.max_mean_abs_diff,
            a_i16=i16[pair.a],
            b_i16=i16[pair.b],
        )
        for pid, pair in enumerate(spec.pairs)
    ]
    candidate_counts = [len(scan.accepted) for scan in scans]
    failing = [scan for scan in scans if not scan.accepted]

    # 3a) Any pair with no acceptable offset means the request has no solution.
    if failing:
        failures = [
            PairFailure(
                pair_id=scan.pair_id,
                a=scan.a,
                b=scan.b,
                reason=(
                    "insufficient_overlap"
                    if scan.scored_positions == 0
                    else "mean_abs_diff_too_high"
                ),
                max_overlap_pixels=scan.best_overlap,
                best_mean_abs_diff=scan.best_mad,
            )
            for scan in failing
        ]
        return AlignResponse(
            status="no_solution",
            layout_unique=False,
            message=(
                "无法同时对齐：至少一个指定图对在搜索窗内不存在同时满足"
                "最小重叠像素数与平均绝对灰度差上限的位置。"
            ),
            images=infos,
            pair_candidate_counts=candidate_counts,
            failing_pairs=failures,
        )

    # 3b) Enumerate globally consistent layouts.
    result = solve(n, spec.pairs, [scan.accepted for scan in scans])

    if result.status == "aligned":
        layout = result.layouts[0]
        positions = [
            ImagePosition(image_id=i, x=x, y=y) for i, (x, y) in enumerate(layout.positions)
        ]
        pair_results = []
        for scan, (dx, dy) in zip(scans, layout.pair_offsets):
            overlap_n, mad = _pair_mad(images[scan.a], images[scan.b], dx, dy)
            pair_results.append(
                PairResult(
                    pair_id=scan.pair_id,
                    a=scan.a,
                    b=scan.b,
                    dx=dx,
                    dy=dy,
                    overlap_pixels=overlap_n,
                    mean_abs_diff=mad,
                )
            )
        return AlignResponse(
            status="aligned",
            layout_unique=True,
            message="所有指定图对同时达标，相对布局唯一。坐标原点为首图左上角。",
            images=infos,
            positions=positions,
            pairs=pair_results,
            mosaic_bounds=_mosaic_bounds(images, layout.positions),
            pair_candidate_counts=candidate_counts,
        )

    if result.status == "ambiguous":
        alternatives = [
            [
                ImagePosition(image_id=i, x=x, y=y)
                for i, (x, y) in enumerate(layout.positions)
            ]
            for layout in result.layouts
        ]
        uncertain = result.uncertain_image_ids or list(range(1, n))
        message = (
            "无法唯一确定相对布局：存在多组都满足全部图对阈值的位置方案。"
            f" 位置不确定的图：{uncertain}。"
        )
        if result.truncated:
            message += " 候选组合数超过搜索上限，已停止枚举；上述图的位置无法唯一确定。"
        return AlignResponse(
            status="ambiguous",
            layout_unique=False,
            message=message,
            images=infos,
            pair_candidate_counts=candidate_counts,
            uncertain_image_ids=uncertain,
            alternative_layouts=alternatives,
            search_truncated=result.truncated,
        )

    # result.status == "no_solution": every pair has candidates individually,
    # but no combination of them is mutually consistent.
    return AlignResponse(
        status="no_solution",
        layout_unique=False,
        message=(
            "无法同时对齐：各图对单独均有达标位置，但不存在一组位置能同时满足"
            "全部指定图对（位置约束互相矛盾）。"
        ),
        images=infos,
        pair_candidate_counts=candidate_counts,
    )
