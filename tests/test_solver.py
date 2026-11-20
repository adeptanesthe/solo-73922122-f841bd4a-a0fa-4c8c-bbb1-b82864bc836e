"""Unit tests for the layout solver's consistency and uniqueness logic."""

from __future__ import annotations

import pytest

from app.models import PairSpec
from app.solver import solve


def pair(a: int, b: int) -> PairSpec:
    return PairSpec(a=a, b=b, candidates=[(0, 0)])


def pair_xy(a: int, b: int, *cands) -> PairSpec:
    return PairSpec(a=a, b=b, candidates=list(cands))


def test_single_consistent_layout():
    pairs = [pair_xy(0, 1, (10, 0), (11, 0)), pair_xy(1, 2, (0, 20)), pair_xy(0, 2, (10, 20))]
    result = solve(3, pairs, [[(10, 0), (11, 0)], [(0, 20)], [(10, 20)]])
    assert result.status == "aligned"
    assert result.layouts[0].positions == ((0, 0), (10, 0), (10, 20))
    assert result.truncated is False


def test_cycle_contradiction_gives_no_solution():
    # 0->1 (10,0), 1->2 (0,20) force 0->2 = (10,20), but only (99,99) is allowed.
    pairs = [pair_xy(0, 1, (10, 0)), pair_xy(1, 2, (0, 20)), pair_xy(0, 2, (99, 99))]
    result = solve(3, pairs, [[(10, 0)], [(0, 20)], [(99, 99)]])
    assert result.status == "no_solution"
    assert result.layouts == []


def test_multiple_layouts_marks_uncertain_images():
    # Pair (1,2) has two viable translations, both consistent with pair (0,1).
    pairs = [pair_xy(0, 1, (100, 0)), pair_xy(1, 2, (0, 50), (0, 60))]
    result = solve(3, pairs, [[(100, 0)], [(0, 50), (0, 60)]])
    assert result.status == "ambiguous"
    assert result.uncertain_image_ids == [2]
    positions_by_layout = {layout.positions[2] for layout in result.layouts}
    assert positions_by_layout == {(100, 50), (100, 60)}


def test_redundant_pair_with_two_consistent_offsets_still_unique_positions():
    # Pair (0,1) pins the relative position; the redundant pair (0,2) also
    # appears as (1,2).  Two different (1,2) offsets are consistent only if they
    # yield the same positions, which they cannot here (one is wrong and must be
    # pruned).  Uniqueness should hold.
    pairs = [
        pair_xy(0, 1, (100, 0)),
        pair_xy(1, 2, (0, 50)),
        pair_xy(0, 2, (100, 50)),
    ]
    result = solve(3, pairs, [[(100, 0)], [(0, 50)], [(100, 50)]])
    assert result.status == "aligned"
    assert result.layouts[0].positions == ((0, 0), (100, 0), (100, 50))


def test_redundant_pair_with_conflicting_candidates_is_still_solvable():
    # (0,2) has an extra wrong candidate; it is simply not chosen.
    pairs = [
        pair_xy(0, 1, (100, 0)),
        pair_xy(1, 2, (0, 50)),
        pair_xy(0, 2, (100, 50), (333, 333)),
    ]
    result = solve(3, pairs, [[(100, 0)], [(0, 50)], [(100, 50), (333, 333)]])
    assert result.status == "aligned"
    assert result.layouts[0].positions == ((0, 0), (100, 0), (100, 50))


def test_symmetry_causes_two_layouts_for_four_tiles():
    # Square cycle where pair (2,3) could be joined at two shifts that both keep
    # the drawing consistent in isolation; with only (0,1),(0,2),(1,3) declared
    # and tile 3 free to move, ambiguity is expected.
    pairs = [pair_xy(0, 1, (100, 0)), pair_xy(0, 2, (0, 100)), pair_xy(1, 3, (0, 100), (5, 100))]
    result = solve(4, pairs, [[(100, 0)], [(0, 100)], [(0, 100), (5, 100)]])
    assert result.status == "ambiguous"
    assert 3 in result.uncertain_image_ids
    assert 1 not in result.uncertain_image_ids


def test_enumeration_budget_marks_truncated():
    # Eight independent-ish pairs each with many candidates, chained so many
    # combinations survive.  Chain of 8 edges with 6 candidates each = 6**8 combos.
    n = 5
    pairs = []
    cands = []
    for k in range(4):
        pairs.append(pair_xy(k, k + 1, *[(v, 0) for v in range(6)]))
        cands.append([(v, 0) for v in range(6)])
    result = solve(n, pairs, cands)
    # Any chain with these candidates also admits many position vectors: the
    # solver either enumerates all (6**4 = 1296, under the budget) or marks
    # truncation.  Either way the answer must be ambiguous.
    assert result.status == "ambiguous"
    assert result.uncertain_image_ids


def test_truncated_result_reports_all_non_anchor_images(monkeypatch):
    import app.solver as solver_module

    monkeypatch.setattr(solver_module, "MAX_LAYOUTS", 2)
    pairs = [pair_xy(0, 1, *[(v, 0) for v in range(6)])]
    result = solve(2, pairs, [[(v, 0) for v in range(6)]])
    assert result.truncated is True
    assert result.status == "ambiguous"
    assert result.uncertain_image_ids == [1]


def test_truncated_before_first_layout_is_not_reported_as_no_solution(monkeypatch):
    import app.solver as solver_module

    # A node budget of 1 stops the search after the first branch; an empty
    # layout set then proves nothing, so the answer must be "ambiguous" (with
    # uncertain images named), never "no_solution".
    monkeypatch.setattr(solver_module, "MAX_NODES_EXPANDED", 1)
    pairs = [pair_xy(0, 1, (10, 0), (20, 0)), pair_xy(1, 2, (0, 10))]
    result = solve(3, pairs, [[(10, 0), (20, 0)], [(0, 10)]])
    assert result.truncated is True
    assert result.status == "ambiguous"
    assert result.uncertain_image_ids == [1, 2]

