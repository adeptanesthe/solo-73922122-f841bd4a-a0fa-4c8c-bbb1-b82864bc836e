"""End-to-end tests through the HTTP API."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.fixtures import (
    DEFAULT_PLACES,
    TRUE_OFFSETS,
    png_bytes,
    png_bytes_16bit,
    png_bytes_gray_alpha,
    png_bytes_palette,
    png_bytes_rgb,
    spec_for,
    tile_set,
    window,
)

client = TestClient(app)


def post(tiles, spec, filenames=None):
    files = []
    for idx, tile in enumerate(tiles):
        names = filenames or [f"tile{idx}.png"] * len(tiles)
        files.append(("images", (names[idx], png_bytes(tile), "image/png")))
    return client.post("/align", files=files, data={"spec": json.dumps(spec)})


def test_health_and_docs():
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/align" in schema["paths"]


def test_aligned_three_tiles_chain():
    tiles = tile_set()
    spec = spec_for([(0, 1), (1, 2)])
    r = post(tiles, spec)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "aligned"
    assert body["layout_unique"] is True
    assert [p["image_id"] for p in body["positions"]] == [0, 1, 2]
    assert (body["positions"][0]["x"], body["positions"][0]["y"]) == (0, 0)
    assert (body["positions"][1]["x"], body["positions"][1]["y"]) == TRUE_OFFSETS[(0, 1)]
    assert (body["positions"][2]["x"], body["positions"][2]["y"]) == (0, 450)
    by_pair = {p["pair_id"]: p for p in body["pairs"]}
    assert by_pair[0]["dx"] == 450 and by_pair[0]["dy"] == 0
    assert by_pair[1]["dx"] == -450 and by_pair[1]["dy"] == 450
    assert all(p["overlap_pixels"] >= spec["min_overlap_pixels"] for p in body["pairs"])
    assert all(p["mean_abs_diff"] <= spec["max_mean_abs_diff"] for p in body["pairs"])
    b = body["mosaic_bounds"]
    assert (b["x_min"], b["y_min"], b["width"], b["height"]) == (0, 0, 1050, 1050)


def test_aligned_with_cycle_and_noise():
    tiles = tile_set(seed=3, noise_sigma=1.5)
    spec = spec_for(
        [(0, 1), (1, 2), (0, 2)],
        radius=3,
        min_overlap=20000,
        max_mad=4.0,
    )
    r = post(tiles, spec)
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["status"] == "aligned", body
    assert [(p["x"], p["y"]) for p in body["positions"]] == [(0, 0), (450, 0), (0, 450)]


def test_disconnected_asks_for_rescan():
    tiles = tile_set()
    spec = spec_for([(0, 1)])  # tile 2 is not linked at all
    r = post(tiles, spec)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "disconnected"
    assert body["unreachable_image_ids"] == [2]
    assert body["rescan_image_id"] == 2
    assert body["positions"] is None


def test_disconnected_picks_least_connected_tile():
    # Tile 2 is placed as before; tile 3 is an extra crop that no pair mentions.
    tiles = tile_set(places=DEFAULT_PLACES + [(700, 700, 300, 300)])
    spec = {
        "pairs": [
            {"a": 0, "b": 1, "window": window((550, 0))},
            {"a": 1, "b": 2, "window": window((-550, 550))},
        ],
        "min_overlap_pixels": 5000,
        "max_mean_abs_diff": 4.0,
    }
    body = post(tiles, spec).json()
    assert body["status"] == "disconnected"
    assert body["unreachable_image_ids"] == [3]
    assert body["rescan_image_id"] == 3


def test_no_solution_when_threshold_impossible():
    # Scanner noise makes even the true offset score > 0; demanding a perfect
    # match therefore leaves no acceptable candidate for any pair.
    tiles = tile_set(noise_sigma=1.5)
    spec = {
        "pairs": [
            {"a": 0, "b": 1, "window": window((450, 0))},
            {"a": 1, "b": 2, "window": window((-450, 450))},
        ],
        "min_overlap_pixels": 20000,
        "max_mean_abs_diff": 0.0,
    }
    body = post(tiles, spec).json()
    assert body["status"] == "no_solution"
    assert {f["pair_id"] for f in body["failing_pairs"]} == {0, 1}
    assert all(f["reason"] == "mean_abs_diff_too_high" for f in body["failing_pairs"])


def test_no_solution_when_overlap_too_small():
    tiles = tile_set()
    spec = {
        "pairs": [
            {"a": 0, "b": 1, "window": window((450, 0))},
            {"a": 1, "b": 2, "window": window((-450, 450))},
        ],
        # (0,1) overlaps by 150x600 and (1,2) by 150x150; require more than both.
        "min_overlap_pixels": 100000,
        "max_mean_abs_diff": 4.0,
    }
    body = post(tiles, spec).json()
    assert body["status"] == "no_solution"
    assert body["failing_pairs"][0]["reason"] == "insufficient_overlap"


def test_no_solution_when_pair_constraints_contradict():
    # Tiles 0 and 2 truly overlap at (0, 450).  Declaring (0,2) at (200, 450)
    # instead still leaves a sizeable overlap; the pixels cannot match there,
    # so the pair has no accepted candidate and the request is unsatisfiable.
    tiles = tile_set()
    spec = {
        "pairs": [
            {"a": 0, "b": 1, "window": window((450, 0), radius=0)},
            {"a": 1, "b": 2, "window": window((-450, 450), radius=0)},
            {"a": 0, "b": 2, "window": window((200, 450), radius=0)},
        ],
        "min_overlap_pixels": 20000,
        "max_mean_abs_diff": 6.0,
    }
    body = post(tiles, spec).json()
    assert body["status"] == "no_solution"
    assert [f["pair_id"] for f in body["failing_pairs"]] == [2]
    assert body["failing_pairs"][0]["reason"] == "mean_abs_diff_too_high"


def test_ambiguous_blank_overlap():
    # Constant tiles: on a flat field every offset in the window has MAD 0, so
    # the relative position is not determined.  Tile 0 and 1 are equal-valued,
    # tile 2 differs, so pair (1,2) pins the shift of 2 relative to 1.
    blank_a = np.full((200, 200), 255, dtype=np.uint8)
    blank_b = np.full((200, 200), 255, dtype=np.uint8)
    blank_c = np.full((200, 200), 255, dtype=np.uint8)
    tiles = [blank_a, blank_b, blank_c]
    spec = {
        "pairs": [
            {
                "a": 0,
                "b": 1,
                "window": {"dx_min": 100, "dx_max": 108, "dy_min": 0, "dy_max": 8},
            },
            {
                "a": 1,
                "b": 2,
                "candidates": [[100, 0]],
            },
        ],
        "min_overlap_pixels": 10000,
        "max_mean_abs_diff": 0.0,
    }
    body = post(tiles, spec).json()
    assert body["status"] == "ambiguous"
    assert body["layout_unique"] is False
    assert set(body["uncertain_image_ids"]) == {1, 2}
    assert body["alternative_layouts"] and len(body["alternative_layouts"]) >= 2
    xs = {tuple((p["x"], p["y"]) for p in layout) for layout in body["alternative_layouts"]}
    assert len(xs) >= 2


def test_no_solution_when_candidates_are_mutually_inconsistent():
    # Constant tiles make every offset in every window pass, so each pair has
    # candidates.  The three windows cannot, however, be satisfied at once:
    # (0,1) forces dx_02 = dx_01 + dx_12 >= 200, while (0,2) only allows 0..8.
    blank = np.full((300, 300), 128, dtype=np.uint8)
    tiles = [blank, blank.copy(), blank.copy()]
    spec = {
        "pairs": [
            {"a": 0, "b": 1, "window": {"dx_min": 100, "dx_max": 108, "dy_min": 0, "dy_max": 0}},
            {"a": 1, "b": 2, "window": {"dx_min": 100, "dx_max": 108, "dy_min": 0, "dy_max": 0}},
            {"a": 0, "b": 2, "window": {"dx_min": 0, "dx_max": 8, "dy_min": 0, "dy_max": 0}},
        ],
        "min_overlap_pixels": 1000,
        "max_mean_abs_diff": 0.0,
    }
    body = post(tiles, spec).json()
    assert body["status"] == "no_solution"
    assert body["failing_pairs"] in (None, [])
    assert body["pair_candidate_counts"] == [9, 9, 9]


def test_ambiguous_reports_each_free_tile():
    blank = np.full((300, 300), 128, dtype=np.uint8)
    other = np.full((300, 300), 128, dtype=np.uint8)
    tiles = [blank, other, np.full((300, 300), 128, dtype=np.uint8)]
    spec = {
        "pairs": [
            {"a": 0, "b": 1, "candidates": [[150, 0], [160, 0]]},
            {"a": 1, "b": 2, "candidates": [[0, 150]]},
        ],
        "min_overlap_pixels": 1000,
        "max_mean_abs_diff": 0.0,
    }
    body = post(tiles, spec).json()
    assert body["status"] == "ambiguous"
    # Tile 1's x is uncertain by 10 px, so tile 2 (rigidly attached) is too.
    assert set(body["uncertain_image_ids"]) == {1, 2}


def test_rejects_wrong_image_count():
    tiles = tile_set()[:2]
    spec = spec_for([(0, 1)])
    r = post(tiles, spec)
    assert r.status_code == 400
    assert "3" in r.json()["detail"] or "张" in r.json()["detail"]


def test_rejects_rgb_png():
    tiles = tile_set()
    files = [("images", (f"t{i}.png", png_bytes_rgb(t), "image/png")) for i, t in enumerate(tiles)]
    r = client.post("/align", files=files, data={"spec": json.dumps(spec_for([(0, 1), (1, 2)]))})
    assert r.status_code == 400
    assert "灰度" in r.json()["detail"] or "颜色类型" in r.json()["detail"]


def test_rejects_16bit_png():
    tiles = tile_set()
    files = [("images", (f"t{i}.png", png_bytes_16bit(t), "image/png")) for i, t in enumerate(tiles)]
    r = client.post("/align", files=files, data={"spec": json.dumps(spec_for([(0, 1), (1, 2)]))})
    assert r.status_code == 400
    assert "8 位" in r.json()["detail"]


def test_rejects_palette_png():
    tiles = tile_set()
    files = [("images", (f"t{i}.png", png_bytes_palette(t), "image/png")) for i, t in enumerate(tiles)]
    r = client.post("/align", files=files, data={"spec": json.dumps(spec_for([(0, 1), (1, 2)]))})
    assert r.status_code == 400
    assert "颜色类型" in r.json()["detail"]


def test_rejects_gray_alpha_png():
    tiles = tile_set()
    files = [("images", (f"t{i}.png", png_bytes_gray_alpha(t), "image/png")) for i, t in enumerate(tiles)]
    r = client.post("/align", files=files, data={"spec": json.dumps(spec_for([(0, 1), (1, 2)]))})
    assert r.status_code == 400
    assert "颜色类型" in r.json()["detail"]


def test_rejects_non_png():
    tiles = tile_set()
    files = [("images", (f"t{i}.png", b"not a png", "image/png")) for i, _ in enumerate(tiles)]
    r = client.post("/align", files=files, data={"spec": json.dumps(spec_for([(0, 1), (1, 2)]))})
    assert r.status_code == 400


def test_rejects_oversize_image():
    big = np.zeros((1025, 300), dtype=np.uint8)
    tiles = [big, np.zeros((300, 300), dtype=np.uint8), np.zeros((300, 300), dtype=np.uint8)]
    files = [("images", (f"t{i}.png", png_bytes(t), "image/png")) for i, t in enumerate(tiles)]
    spec = {"pairs": [{"a": 0, "b": 1, "window": window((0, 0))}], "min_overlap_pixels": 10,
            "max_mean_abs_diff": 1.0}
    r = client.post("/align", files=files, data={"spec": json.dumps(spec)})
    assert r.status_code == 400
    assert "1024" in r.json()["detail"]


def test_rejects_window_too_large():
    tiles = tile_set()
    spec = {
        "pairs": [
            {"a": 0, "b": 1, "window": {"dx_min": 0, "dx_max": 9, "dy_min": 0, "dy_max": 9}},
            {"a": 1, "b": 2, "candidates": [[0, 0]]},
        ],
        "min_overlap_pixels": 1000,
        "max_mean_abs_diff": 1.0,
    }
    r = post(tiles, spec)
    assert r.status_code == 422


def test_rejects_too_many_candidates():
    tiles = tile_set()
    spec = {
        "pairs": [
            {"a": 0, "b": 1, "candidates": [[i, 0] for i in range(82)]},
            {"a": 1, "b": 2, "candidates": [[0, 0]]},
        ],
        "min_overlap_pixels": 1000,
        "max_mean_abs_diff": 1.0,
    }
    assert post(tiles, spec).status_code == 422


def test_rejects_unknown_image_index():
    tiles = tile_set()
    spec = {
        "pairs": [{"a": 0, "b": 7, "candidates": [[0, 0]]}],
        "min_overlap_pixels": 1000,
        "max_mean_abs_diff": 1.0,
    }
    r = post(tiles, spec)
    assert r.status_code == 422
    assert "图号" in r.json()["detail"]


def test_rejects_bad_spec_json():
    tiles = tile_set()
    r = client.post(
        "/align",
        files=[("images", (f"t{i}.png", png_bytes(t), "image/png")) for i, t in enumerate(tiles)],
        data={"spec": "{not json"},
    )
    assert r.status_code == 400


def test_rejects_duplicate_pair():
    tiles = tile_set()
    spec = {
        "pairs": [
            {"a": 0, "b": 1, "candidates": [[550, 0]]},
            {"a": 0, "b": 1, "candidates": [[550, 0]]},
        ],
        "min_overlap_pixels": 1000,
        "max_mean_abs_diff": 1.0,
    }
    assert post(tiles, spec).status_code == 422


def test_five_images_chain():
    places = [(0, 0, 400, 400), (350, 0, 400, 400), (700, 0, 400, 400), (0, 350, 400, 400), (350, 350, 400, 400)]
    tiles = tile_set(places=places)
    offsets = {(0, 1): (350, 0), (1, 2): (350, 0), (0, 3): (0, 350), (3, 4): (350, 0)}
    spec = {
        "pairs": [
            {"a": a, "b": b, "window": window(off, radius=2)} for (a, b), off in offsets.items()
        ],
        "min_overlap_pixels": 10000,
        "max_mean_abs_diff": 2.0,
    }
    body = post(tiles, spec).json()
    assert body["status"] == "aligned", body
    assert [(p["x"], p["y"]) for p in body["positions"]] == [
        (0, 0),
        (350, 0),
        (700, 0),
        (0, 350),
        (350, 350),
    ]


def test_exposure_difference_within_threshold():
    tiles = tile_set(
        places=[(0, 0, 500, 500), (450, 0, 500, 500), (0, 450, 500, 500)],
        per_tile_gain=[1.0, 1.04, 0.97],
        offset=-2.0,
        noise_sigma=0.8,
    )
    # Single-position windows: the question here is whether a brightness
    # mismatch is tolerated, not whether the window is narrow enough.
    spec = {
        "pairs": [
            {"a": 0, "b": 1, "window": window((450, 0), radius=0)},
            {"a": 0, "b": 2, "window": window((0, 450), radius=0)},
        ],
        "min_overlap_pixels": 10000,
        "max_mean_abs_diff": 12.0,
    }
    body = post(tiles, spec).json()
    assert body["status"] == "aligned", body
    assert all(p["mean_abs_diff"] <= 12.0 for p in body["pairs"])


def test_ambiguous_always_names_an_uncertain_image_and_truncation_flag():
    # Five constant tiles, four chained pairs, each free to slide over a 9x9
    # window: enormous number of feasible layouts.  Enumeration is budgeted, so
    # the answer must be "ambiguous" with truncation flagged and at least one
    # named uncertain tile.
    blank = np.full((200, 200), 90, dtype=np.uint8)
    tiles = [blank.copy() for _ in range(5)]
    spec = {
        "pairs": [
            {
                "a": 0,
                "b": 1,
                "window": {"dx_min": 100, "dx_max": 108, "dy_min": 0, "dy_max": 8},
            },
            {
                "a": 1,
                "b": 2,
                "window": {"dx_min": 100, "dx_max": 108, "dy_min": 0, "dy_max": 8},
            },
            {
                "a": 2,
                "b": 3,
                "window": {"dx_min": 100, "dx_max": 108, "dy_min": 0, "dy_max": 8},
            },
            {
                "a": 3,
                "b": 4,
                "window": {"dx_min": 100, "dx_max": 108, "dy_min": 0, "dy_max": 8},
            },
        ],
        "min_overlap_pixels": 1000,
        "max_mean_abs_diff": 0.0,
    }
    body = post(tiles, spec).json()
    assert body["status"] == "ambiguous"
    assert body["search_truncated"] is True
    assert body["uncertain_image_ids"]
    assert set(body["uncertain_image_ids"]) >= {4}


def test_negative_coordinates_when_anchor_is_not_topleftmost():
    # Tile 0 sits at the bottom-right of a 2x2 mosaic, so tiles 1 and 2 must get
    # negative coordinates relative to it, and mosaic_bounds starts negative.
    canvas = np.arange(0, 1, dtype=np.uint8)  # placeholder, replaced below
    tiles = tile_set(places=[(450, 450, 600, 600), (0, 0, 600, 600), (0, 450, 600, 600)])
    spec = {
        "pairs": [
            {"a": 0, "b": 1, "candidates": [[-450, -450]]},
            {"a": 0, "b": 2, "candidates": [[-450, 0]]},
        ],
        "min_overlap_pixels": 20000,
        "max_mean_abs_diff": 0.1,
    }
    body = post(tiles, spec).json()
    assert body["status"] == "aligned", body
    assert [(p["x"], p["y"]) for p in body["positions"]] == [(0, 0), (-450, -450), (-450, 0)]
    bounds = body["mosaic_bounds"]
    assert bounds["x_min"] == -450 and bounds["y_min"] == -450
    assert bounds["width"] == 1050 and bounds["height"] == 1050


def test_rescan_prefers_isolated_tile():
    # Pairs (0,1) and (2,3) leave {2,3} stranded but mutually linked, while tile
    # 4 has no declared pair at all.  The isolated tile is the better rescan
    # target because adding one link cannot fix a stranded pair.
    tiles = tile_set(places=DEFAULT_PLACES + [(600, 600, 300, 300), (700, 700, 300, 300)])
    spec = {
        "pairs": [
            {"a": 0, "b": 1, "window": window((450, 0))},
            {"a": 2, "b": 3, "window": window((-450, 450))},
        ],
        "min_overlap_pixels": 5000,
        "max_mean_abs_diff": 4.0,
    }
    body = post(tiles, spec).json()
    assert body["status"] == "disconnected"
    assert body["unreachable_image_ids"] == [2, 3, 4]
    assert body["rescan_image_id"] == 4
