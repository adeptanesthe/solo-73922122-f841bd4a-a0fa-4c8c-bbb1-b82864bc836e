"""Unit tests for the recognition solver, including a brute-force cross-check."""

from __future__ import annotations

import numpy as np
import pytest

from app.demo_font import glyph_bitmap
from app.solver import (
    GlyphSample,
    canonical_value,
    recognize,
)
from tests.fixtures import HEIGHT, glyph_samples, render_text


def samples_from(arrays: dict[str, np.ndarray] | None = None) -> list[GlyphSample]:
    arrays = arrays or glyph_samples()
    return [
        GlyphSample(label=label, width=arr.shape[1], ink=(arr == 0))
        for label, arr in arrays.items()
    ]


def ink_of(arr: np.ndarray) -> np.ndarray:
    return arr == 0


# --------------------------------------------------------------------------
# Canonicalization
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1", "1"),
        ("007", "7"),
        ("0", "0"),
        ("12.50", "12.5"),
        ("01.20", "1.2"),
        ("0.0", "0"),
        ("0.000", "0"),
        ("100.00", "100"),
        ("1024", "1024"),
    ],
)
def test_canonical_value(raw, expected):
    assert canonical_value(raw) == expected


# --------------------------------------------------------------------------
# Brute-force reference implementation for small line widths
# --------------------------------------------------------------------------


def brute_force(line: np.ndarray, sample_arrays: dict[str, np.ndarray]) -> dict[str, int]:
    """Enumerate every segmentation/spelling; return value -> minimum cost."""
    width = line.shape[1]
    line_ink = ink_of(line)
    by_width: dict[int, dict[str, np.ndarray]] = {}
    for label, arr in sample_arrays.items():
        by_width.setdefault(arr.shape[1], {})[label] = arr

    col_ink = line_ink.any(axis=0)

    def legal(text: str) -> bool:
        if not text or len(text.replace(".", "")) > 8:
            return False
        if "." in text:
            left, right = text.split(".")
            if not left or not right:
                return False
        return text.replace(".", "").isdigit()

    best: dict[str, int] = {}

    def search(col: int, text: str, cost: int) -> None:
        if col == width:
            if legal(text):
                value = canonical_value(text)
                if value not in best or cost < best[value]:
                    best[value] = cost
            return
        # skip an all-white column; inked columns must be covered by a segment
        if not col_ink[col]:
            search(col + 1, text, cost)
        # consume one segment of every available width/label
        for w, labels in by_width.items():
            end = col + w
            if end > width:
                continue
            if not col_ink[col:end].any():
                continue
            for label, arr in labels.items():
                candidate = text + label
                if len(candidate.replace(".", "")) > 8:
                    continue
                if "." in candidate and candidate.count(".") > 1:
                    continue
                if label == "." and (not text or text.endswith(".")):
                    continue
                seg_cost = int(np.count_nonzero(line_ink[:, col:end] != (arr == 0)))
                search(end, candidate, cost + seg_cost)

    search(0, "", 0)
    return best


# --------------------------------------------------------------------------
# Solver behaviour
# --------------------------------------------------------------------------


def test_clean_line_recognized_with_zero_mismatch():
    line = render_text("12.50")
    result = recognize(ink_of(line), samples_from())
    assert result.truncated is False
    best = result.readings[0]
    assert best.raw_text == "12.50"
    assert canonical_value(best.raw_text) == "12.5"
    assert best.mismatches == 0
    # Column ranges cover every character and are ordered, half-open.
    starts = [s.start_col for s in best.chars]
    ends = [s.end_col for s in best.chars]
    assert starts == sorted(starts)
    assert all(e > s for s, e in zip(starts, ends))
    assert [s.char for s in best.chars] == list("12.50")
    assert sum(s.mismatches for s in best.chars) == 0
    # Any runner-up must be a distinct value with a strictly positive cost.
    if len(result.readings) == 2:
        assert canonical_value(result.readings[1].raw_text) != "12.5"
        assert result.readings[1].mismatches > 0


def test_stuck_digits_are_segmented():
    # No blank gap between "1" and "2": the cut must fall between ink columns.
    line = render_text("12", stuck_after={0}, margin=1)
    result = recognize(ink_of(line), samples_from())
    assert result.readings[0].raw_text == "12"
    assert result.readings[0].mismatches == 0
    first, second = result.readings[0].chars
    assert first.end_col == second.start_col


def test_mismatch_cost_counts_pixel_differences():
    line = render_text("7", flips=[(1, 3), (4, 3), (6, 3)])
    result = recognize(ink_of(line), samples_from())
    assert result.readings[0].raw_text == "7"
    assert result.readings[0].mismatches == 3


def test_leading_and_trailing_zeros_group_to_one_value():
    # 01.20 and 1.20 and 1.2 all represent the same numeric value, so the best
    # reading is that single value; any runner-up must differ from it.
    line = render_text("01.20")
    result = recognize(ink_of(line), samples_from())
    assert result.readings[0].raw_text == "01.20"
    assert canonical_value(result.readings[0].raw_text) == "1.2"
    assert result.readings[0].mismatches == 0
    if len(result.readings) == 2:
        assert canonical_value(result.readings[1].raw_text) != "1.2"


def test_dot_needs_digit_on_both_sides():
    # Only a width-1 dot sample is available, and the line holds nine separated
    # ink columns.  Dots alone cannot form a legal reading, and eight characters
    # cannot cover nine runs (skips are only legal across all-white columns),
    # so no legal reading exists at all.
    dot1 = np.full((HEIGHT, 1), 0, dtype=np.uint8)
    gap = np.full((HEIGHT, 1), 255, dtype=np.uint8)
    parts = [gap]
    for _ in range(9):
        parts.extend([dot1, gap])
    line = np.concatenate(parts, axis=1)
    only_dot = [GlyphSample(label=".", width=1, ink=(dot1 == 0))]
    result = recognize(ink_of(line), only_dot)
    assert result.readings == []
    assert result.truncated is False


def test_all_white_line_has_no_reading():
    line = np.full((HEIGHT, 12), 255, dtype=np.uint8)
    result = recognize(ink_of(line), samples_from())
    assert result.readings == []
    assert result.truncated is False


def test_duplicate_sample_keeps_cheapest():
    # Two samples labelled "1": one perfect, one with an extra ink pixel.
    good = glyph_samples()["1"]
    bad = good.copy()
    bad[0, 0] = 0  # add ink
    samples = samples_from() + [GlyphSample(label="1", width=5, ink=(bad == 0))]
    line = render_text("1")
    result = recognize(ink_of(line), samples)
    assert result.readings[0].raw_text == "1"
    assert result.readings[0].mismatches == 0


def test_returns_two_distinct_values_sorted():
    # Deface "1" heavily so another digit wins or ties closely; build an
    # ambiguous single glyph whose two nearest templates are different digits.
    base = glyph_bitmap("1").copy()
    ambiguous = base.copy()
    ambiguous[:, 2] = False  # erase the spine -> closer to several templates
    line_arr = np.where(ambiguous, 0, 255).astype(np.uint8)
    result = recognize(ink_of(line_arr), samples_from())
    costs = [r.mismatches for r in result.readings]
    assert costs == sorted(costs)
    values = [canonical_value(r.raw_text) for r in result.readings]
    assert len(values) == len(set(values))


def test_more_than_eight_digits_rejected():
    line = render_text("123456789", margin=0)
    result = recognize(ink_of(line), samples_from())
    # No legal reading of length 9; the solver may still find shorter readings
    # by skipping columns, but none may contain 9 digits.
    for reading in result.readings:
        assert len(reading.raw_text.replace(".", "")) <= 8


def test_brute_force_matches_solver_on_small_lines():
    rng = np.random.default_rng(7)
    sample_arrays = glyph_samples()
    texts = ["5", "42", "7.3", "10.5", "00.10", "902", "3.14"]
    for text in texts:
        stuck = set(rng.choice(max(1, len(text) - 1), size=1))
        line = render_text(text, stuck_after=stuck, margin=0)
        # Add a couple of random pixel defects inside the rendered glyphs.
        for _ in range(2):
            r = int(rng.integers(0, HEIGHT))
            c = int(rng.integers(0, line.shape[1]))
            line[r, c] = 255 - line[r, c]
        expected = brute_force(line, sample_arrays)
        result = recognize(ink_of(line), samples_from())
        got = {canonical_value(r.raw_text): r.mismatches for r in result.readings}
        assert expected, "brute force reference found no readings"
        # The solver's best must equal the brute-force optimum.
        best_value = min(expected, key=lambda v: expected[v])
        assert got[best_value] == expected[best_value]
        # Every solver-reported value agrees with the reference cost.
        for value, cost in got.items():
            assert expected[value] == cost
        # At most two distinct values, and they are the two cheapest (with
        # deterministic tie handling).
        all_sorted = sorted(expected.items(), key=lambda kv: (kv[1], kv[0]))
        assert list(got) == [v for v, _ in all_sorted[:2]]


def test_brute_force_on_all_blank_widths():
    sample_arrays = glyph_samples()
    line = np.full((HEIGHT, 9), 255, dtype=np.uint8)
    assert brute_force(line, sample_arrays) == {}
    result = recognize(ink_of(line), samples_from())
    assert result.readings == []


def test_randomized_exact_costs_against_brute_force():
    # Property test over small random instances (kept narrow so the exhaustive
    # reference stays cheap): every reported value's cost must equal the
    # reference minimum for that value, best/runner-up costs must match, and
    # equal-cost ties are allowed to differ in which values are retained.
    rng = np.random.default_rng(11)
    h = 10
    for _ in range(25):
        n_lab = int(rng.integers(2, 5))
        labels = [str(d) for d in range(n_lab)]
        sample_maps, samples = {}, []
        for label in labels:
            w = int(rng.integers(1, 3))
            bmp = rng.random((h, w)) < 0.4
            sample_maps[label] = np.where(bmp, 0, 255).astype(np.uint8)
            samples.append(GlyphSample(label=label, width=w, ink=bmp))
        width = int(rng.integers(3, 9))
        line_ink = rng.random((h, width)) < 0.35
        line_arr = np.where(line_ink, 0, 255).astype(np.uint8)

        result = recognize(line_ink, samples)
        reference = brute_force(line_arr, sample_maps)
        ordered_ref = sorted(reference.items(), key=lambda kv: (kv[1], kv[0]))

        assert bool(result.readings) == bool(ordered_ref)
        assert result.truncated is False
        got = {canonical_value(r.raw_text): r.mismatches for r in result.readings}
        assert len(got) == len(result.readings)  # distinct values only
        for value, cost in got.items():
            assert reference[value] == cost  # exact per-value optimum
        if ordered_ref:
            assert result.readings[0].mismatches == ordered_ref[0][1]
        if len(ordered_ref) >= 2 and len(result.readings) >= 2:
            assert result.readings[1].mismatches == ordered_ref[1][1]
            assert result.readings[1].mismatches >= result.readings[0].mismatches


def test_segment_width_must_match_sample_width():
    # Only digit samples (width 5) available; a dot-only line cannot parse.
    digit_arrays = {str(d): glyph_samples()[str(d)] for d in range(10)}
    line = render_text("1.1")
    result = recognize(ink_of(line), samples_from(digit_arrays))
    # The dot columns contain ink but no width-2 sample exists, so a legal
    # reading covering that ink is impossible; returned readings must avoid it.
    for reading in result.readings:
        assert "." not in reading.raw_text
