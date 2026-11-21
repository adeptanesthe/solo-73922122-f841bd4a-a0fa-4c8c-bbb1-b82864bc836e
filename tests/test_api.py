"""End-to-end tests through the HTTP API."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.demo_font import all_glyph_labels
from tests.fixtures import (
    HEIGHT,
    default_spec,
    glyph_samples,
    png_bytes,
    png_bytes_16bit,
    png_bytes_gray_alpha,
    png_bytes_nonbinary,
    png_bytes_palette,
    png_bytes_rgb,
    render_text,
)

client = TestClient(app)


def post(line, glyph_arrays, labels=None, spec=None):
    samples = glyph_samples()
    default_labels = list(samples.keys())
    spec = spec or default_spec()
    payload = dict(spec)
    # Explicit labels follow the uploads; otherwise keep whatever the spec
    # declares (default_spec already lists all 11 labels in upload order).
    if labels is not None:
        payload["glyph_labels"] = labels
    else:
        payload.setdefault("glyph_labels", default_labels)
    files = [("line", ("line.png", png_bytes(line), "image/png"))]
    for idx, arr in enumerate(glyph_arrays):
        files.append(("glyphs", (f"glyph_{idx}.png", png_bytes(arr), "image/png")))
    return client.post("/recognize", files=files, data={"spec": json.dumps(payload)})


def default_glyphs(labels=None):
    labels = labels or all_glyph_labels()
    samples = glyph_samples()
    return [samples[label] for label in labels]


def test_health_docs_and_demo_endpoints():
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/docs").status_code == 200
    assert "/recognize" in client.get("/openapi.json").json()["paths"]
    assert client.get("/demo/line").headers["content-type"] == "image/png"
    assert client.get("/demo/glyphs/5").headers["content-type"] == "image/png"
    assert client.get("/demo/glyphs/dot").headers["content-type"] == "image/png"
    assert client.get("/demo/glyphs/x").status_code == 404
    spec = client.get("/demo/spec").json()
    assert set(spec) == {"glyph_labels", "max_mismatch", "min_margin"}


def test_entry_ready_clean_line():
    line = render_text("12.50")
    r = post(line, default_glyphs())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "entry_ready"
    assert body["review_reasons"] == []
    assert body["search_truncated"] is False
    best = body["candidates"][0]
    assert best["raw_text"] == "12.50"
    assert best["value"] == "12.5"
    assert best["mismatches"] == 0
    assert [c["char"] for c in best["chars"]] == list("12.50")
    # Column ranges are half-open and ordered.
    for prev, nxt in zip(best["chars"], best["chars"][1:]):
        assert prev["end_col"] <= nxt["start_col"]
        assert nxt["end_col"] > nxt["start_col"]
    assert sum(c["mismatches"] for c in best["chars"]) == best["mismatches"]
    # A runner-up, if present, is a distinct, strictly worse value.
    if len(body["candidates"]) == 2:
        assert body["candidates"][1]["value"] != "12.5"
        assert body["candidates"][1]["mismatches"] > 0


def test_stuck_line_with_defects_still_recognized():
    line = render_text("12.50", stuck_after={0, 3}, flips=[(2, 4), (3, 9)])
    r = post(line, default_glyphs(), spec=default_spec(max_mismatch=12, min_margin=3))
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["status"] == "entry_ready", body
    assert body["candidates"][0]["raw_text"] == "12.50"
    assert 0 < body["candidates"][0]["mismatches"] <= 12


def test_mismatch_limit_exceeded_marks_review():
    line = render_text("7", flips=[(r, 3) for r in range(HEIGHT)])  # 7 flipped pixels
    r = post(line, default_glyphs(), spec=default_spec(max_mismatch=1, min_margin=3))
    body = r.json()
    assert body["status"] == "needs_review"
    assert "mismatch_limit_exceeded" in body["review_reasons"]
    assert body["candidates"]  # a reading exists, just too costly


def test_small_margin_marks_review():
    # An ambiguous glyph: erase the spine of "1" so several digits score close.
    from app.demo_font import glyph_bitmap

    bmp = glyph_bitmap("1").copy()
    bmp[:, 2] = False
    line = np.where(bmp, 0, 255).astype(np.uint8)
    r = post(line, default_glyphs(), spec=default_spec(max_mismatch=100, min_margin=100))
    body = r.json()
    assert body["status"] == "needs_review"
    assert "margin_too_small" in body["review_reasons"]
    assert len(body["candidates"]) == 2


def test_no_reading_marks_review():
    # Nine separated single ink columns with only a dot sample available:
    # dots alone are illegal and eight characters cannot cover nine runs.
    dot1 = np.full((HEIGHT, 1), 0, dtype=np.uint8)
    gap = np.full((HEIGHT, 1), 255, np.uint8)
    parts = [gap]
    for _ in range(9):
        parts.extend([dot1, gap])
    line = np.concatenate(parts, axis=1)
    r = post(line, [dot1], labels=["."], spec={"glyph_labels": ["."], "max_mismatch": 50, "min_margin": 1})
    body = r.json()
    assert body["status"] == "needs_review"
    assert body["review_reasons"] == ["no_reading"]
    assert body["candidates"] == []


def test_all_white_line_is_review_not_client_error():
    line = np.full((HEIGHT, 10), 255, dtype=np.uint8)
    r = post(line, default_glyphs())
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "needs_review"
    assert "no_reading" in body["review_reasons"]


def test_rejects_rgb_png():
    line = render_text("1")
    files = [("line", ("line.png", png_bytes_rgb(line), "image/png"))]
    for idx, arr in enumerate(default_glyphs()):
        files.append(("glyphs", (f"g{idx}.png", png_bytes(arr), "image/png")))
    r = client.post(
        "/recognize", files=files, data={"spec": json.dumps(default_spec())}
    )
    assert r.status_code == 400
    assert "颜色类型" in r.json()["detail"]


def test_rejects_16bit_png():
    line = render_text("1")
    files = [("line", ("line.png", png_bytes_16bit(line), "image/png"))]
    for idx, arr in enumerate(default_glyphs()):
        files.append(("glyphs", (f"g{idx}.png", png_bytes(arr), "image/png")))
    r = client.post("/recognize", files=files, data={"spec": json.dumps(default_spec())})
    assert r.status_code == 400
    assert "8 位" in r.json()["detail"]


def test_rejects_palette_and_alpha_png():
    line = render_text("1")
    for encoder, needle in (
        (png_bytes_palette, "颜色类型"),
        (png_bytes_gray_alpha, "颜色类型"),
    ):
        files = [("line", ("line.png", encoder(line), "image/png"))]
        for idx, arr in enumerate(default_glyphs()):
            files.append(("glyphs", (f"g{idx}.png", png_bytes(arr), "image/png")))
        r = client.post("/recognize", files=files, data={"spec": json.dumps(default_spec())})
        assert r.status_code == 400
        assert needle in r.json()["detail"]


def test_rejects_non_binary_pixel_values():
    line = render_text("1")
    files = [("line", ("line.png", png_bytes_nonbinary(line), "image/png"))]
    for idx, arr in enumerate(default_glyphs()):
        files.append(("glyphs", (f"g{idx}.png", png_bytes(arr), "image/png")))
    r = client.post("/recognize", files=files, data={"spec": json.dumps(default_spec())})
    assert r.status_code == 400
    assert "0/255" in r.json()["detail"]


def test_rejects_non_png_and_bad_json():
    files = [("line", ("line.png", b"not a png", "image/png"))]
    for idx, arr in enumerate(default_glyphs()):
        files.append(("glyphs", (f"g{idx}.png", png_bytes(arr), "image/png")))
    r = client.post("/recognize", files=files, data={"spec": json.dumps(default_spec())})
    assert r.status_code == 400

    line = render_text("1")
    r = client.post(
        "/recognize",
        files=[("line", ("line.png", png_bytes(line), "image/png"))]
        + [
            ("glyphs", (f"g{idx}.png", png_bytes(arr), "image/png"))
            for idx, arr in enumerate(default_glyphs())
        ],
        data={"spec": "{broken"},
    )
    assert r.status_code == 400


def test_rejects_label_count_mismatch():
    line = render_text("1")
    # Upload 11 glyph files but declare only 2 labels: order-wise correspondence
    # is impossible, so this is a 422 semantic error.
    spec = default_spec()
    spec["glyph_labels"] = ["1", "2"]
    r = post(line, default_glyphs(), spec=spec)
    assert r.status_code == 422
    assert "一一对应" in r.json()["detail"]


def test_rejects_illegal_label():
    line = render_text("1")
    # 11 uploads with 11 labels, but the first label is not a legal glyph label.
    labels = ["x"] + [str(d) for d in range(10)]
    glyphs = default_glyphs()
    r = post(line, glyphs, labels=labels)
    assert r.status_code == 422
    assert "glyph_labels" in r.json()["detail"]


def test_dot_only_label_set_yields_no_reading():
    line = render_text("1")
    r = post(
        line,
        [glyph_samples()["."]],
        labels=["."],
        spec={
            "glyph_labels": ["."],
            "max_mismatch": 5,
            "min_margin": 1,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "needs_review"
    assert body["review_reasons"] == ["no_reading"]


def test_rejects_mismatched_heights():
    line = render_text("1")
    glyphs = default_glyphs()
    glyphs[3] = np.concatenate([glyphs[3], np.full((1, glyphs[3].shape[1]), 255, np.uint8)], axis=0)
    r = post(line, glyphs)
    assert r.status_code == 422
    assert "高度" in r.json()["detail"]


def test_rejects_all_white_glyph():
    line = render_text("1")
    glyphs = default_glyphs()
    glyphs[0] = np.full_like(glyphs[0], 255)
    r = post(line, glyphs)
    assert r.status_code == 422
    assert "全白" in r.json()["detail"]


def test_rejects_oversize_line():
    line = np.full((HEIGHT, 200), 255, dtype=np.uint8)
    r = post(line, default_glyphs())
    assert r.status_code == 422
    assert "192" in r.json()["detail"]


def test_glyph_cap_boundary():
    line = render_text("1")
    # 33 uploads is the inclusive cap and must work; 34 is rejected.  The cap
    # is enforced on the uploads with a 400 (a malformed request shape).
    labels33 = (all_glyph_labels() * 3)[:33]
    glyphs33 = [glyph_samples()[label] for label in labels33]
    ok = post(line, glyphs33, labels=labels33)
    assert ok.status_code == 200, ok.text

    labels34 = (all_glyph_labels() * 4)[:34]
    glyphs34 = [glyph_samples()[label] for label in labels34]
    too_many = post(line, glyphs34, labels=labels34)
    assert too_many.status_code in (400, 422)
    assert "33" in too_many.json()["detail"]


def test_min_margin_must_be_positive():
    line = render_text("1")
    r = post(line, default_glyphs(), spec=default_spec(min_margin=0))
    assert r.status_code == 422


def test_demo_endpoint_request_is_entry_ready():
    line_bytes = client.get("/demo/line").content
    glyph_bytes = [client.get(f"/demo/glyphs/{'dot' if l == '.' else l}").content for l in all_glyph_labels()]
    spec = client.get("/demo/spec").json()
    files = [("line", ("line.png", line_bytes, "image/png"))]
    files += [("glyphs", (f"g{i}.png", data, "image/png")) for i, data in enumerate(glyph_bytes)]
    r = client.post("/recognize", files=files, data={"spec": json.dumps(spec)})
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["status"] == "entry_ready", body
    assert body["candidates"][0]["value"] == "12.5"


def test_subset_of_labels_works():
    # Only the glyphs actually appearing in the line are supplied.
    line = render_text("1.0")
    labels = ["0", "1", "."]
    r = post(line, default_glyphs(labels), labels=labels)
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["status"] == "entry_ready"
    assert body["candidates"][0]["raw_text"] == "1.0"
