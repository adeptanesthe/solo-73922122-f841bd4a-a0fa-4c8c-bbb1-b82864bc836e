"""Mosaic layout solver.

Given the candidate offsets accepted for each declared pair, decide whether the
global layout is unique.

Coordinate convention: image ``i``'s top-left corner is ``(x_i, y_i)`` in a
common frame; image 0 is the origin ``(0, 0)``.  A declared pair ``(a, b)`` with
offset ``(dx, dy)`` means "b's origin sits at ``(x_a + dx, y_a + dy)``", i.e.::

    x_b - x_a = dx
    y_b - y_a = dy

Each pair may admit several offsets that satisfy the overlap and MAD gates on
its own.  A *layout* is a choice of one accepted offset for every declared pair
such that all difference constraints are mutually consistent.  Plain
"one candidate per pair" enumeration would blow up on redundant pairs (up to 10
pairs, 81 candidates each), so the solver instead:

1. builds a spanning tree of the pair graph rooted at image 0, preferring
   low-candidate-count edges, which fixes all image positions;
2. enumerates tree-candidate assignments, pruning any assignment that already
   contradicts an accepted offset of an edge whose two endpoints are placed;
3. at a leaf, verifies that every remaining pair has *some* accepted offset
   consistent with the resulting positions.

Distinct induced position vectors are the candidate layouts.  A fixed budget
bounds worst-case work; if exhausted, the result is conservatively reported as
ambiguous with ``truncated`` set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import PairSpec

# Distinct layouts retained.  Once the cap is hit, uniqueness cannot be proven.
MAX_LAYOUTS = 512
# Hard budget on search-tree nodes expanded.  A node is one placement of one
# image; this bounds work even when every branch dies before reaching a leaf
# (e.g. a redundant pair that rejects all combinations at the last level).
MAX_NODES_EXPANDED = 600_000


@dataclass
class Layout:
    positions: tuple[tuple[int, int], ...]  # indexed by image id, image 0 at (0, 0)
    pair_offsets: tuple[tuple[int, int], ...]  # indexed by pair id


@dataclass
class SolveResult:
    status: str  # "aligned" | "no_solution" | "ambiguous"
    layouts: list[Layout] = field(default_factory=list)
    uncertain_image_ids: list[int] = field(default_factory=list)
    truncated: bool = False
    candidate_counts: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class _TreeEdge:
    pid: int
    parent: int
    child: int
    sign: int  # child position = parent position + sign * (dx, dy) of pair pid


def _build_tree(
    n_images: int, pairs: list[PairSpec], counts: list[int]
) -> list[_TreeEdge] | None:
    """Prim's algorithm from image 0, cheapest edges first.

    Choosing low-candidate edges first minimises the number of tree assignments
    enumerated.  Returns None when some image is unreachable from image 0.
    """
    adjacency: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n_images)}
    for pid, pair in enumerate(pairs):
        adjacency[pair.a].append((pair.b, pid))
        adjacency[pair.b].append((pair.a, pid))

    in_tree = {0}
    used: set[int] = set()
    order: list[_TreeEdge] = []
    frontier: set[tuple[int, int]] = set(adjacency[0])  # (other, pid)

    while len(in_tree) < n_images and frontier:
        allowed = [(counts[pid], other, pid) for other, pid in frontier if other not in in_tree]
        if not allowed:
            break
        _count, node, pid = min(allowed)
        parent = pairs[pid].a if node == pairs[pid].b else pairs[pid].b
        sign = -1 if pairs[pid].a == node else +1
        order.append(_TreeEdge(pid=pid, parent=parent, child=node, sign=sign))
        used.add(pid)
        in_tree.add(node)
        frontier.discard((node, pid))
        for other, other_pid in adjacency[node]:
            if other not in in_tree and other_pid not in used:
                frontier.add((other, other_pid))
    if len(in_tree) < n_images:
        return None
    return order


def solve(
    n_images: int,
    pairs: list[PairSpec],
    candidates: list[list[tuple[int, int]]],
) -> SolveResult:
    """Enumerate globally consistent layouts.

    ``candidates[pid]`` holds the accepted offsets for pair ``pid``; the caller
    guarantees every list is non-empty and the pair graph connects image 0.
    """
    candidate_counts = [len(c) for c in candidates]
    result = SolveResult(status="no_solution", candidate_counts=candidate_counts)
    n_pairs = len(pairs)

    tree = _build_tree(n_images, pairs, candidate_counts)
    if tree is None:  # pragma: no cover - caller checks connectivity first
        result.status = "disconnected"
        return result

    endpoints = [(p.a, p.b) for p in pairs]
    tree_pids = {edge.pid for edge in tree}
    non_tree = [pid for pid in range(n_pairs) if pid not in tree_pids]
    cand_sets = [set(c) for c in candidates]

    # Enumerate tree edges parent-first so each step extends known positions.
    ordered_edges: list[_TreeEdge] = []
    placed = {0}
    pending = list(tree)
    while pending:
        edge = next((e for e in pending if e.parent in placed), pending[0])
        pending.remove(edge)
        ordered_edges.append(edge)
        placed.add(edge.child)

    layouts: dict[tuple[tuple[int, int], ...], Layout] = {}
    nodes_expanded = 0
    truncated = False

    def consistent(positions: dict[int, tuple[int, int]]) -> bool:
        """Check every fully-placed pair against the position differences."""
        for pid, (a, b) in enumerate(endpoints):
            if a in positions and b in positions:
                (ax, ay), (bx, by) = positions[a], positions[b]
                if (bx - ax, by - ay) not in cand_sets[pid]:
                    return False
        return True

    def record(positions: dict[int, tuple[int, int]]) -> None:
        norm = tuple(positions[i] for i in range(n_images))
        if norm in layouts:
            return
        pair_offsets: list[tuple[int, int]] = []
        for pid, (a, b) in enumerate(endpoints):
            (ax, ay), (bx, by) = positions[a], positions[b]
            pair_offsets.append((bx - ax, by - ay))
        layouts[norm] = Layout(positions=norm, pair_offsets=tuple(pair_offsets))

    def backtrack(k: int, positions: dict[int, tuple[int, int]]) -> None:
        nonlocal nodes_expanded, truncated
        if truncated:
            return
        if k == len(ordered_edges):
            record(positions)
            if len(layouts) >= MAX_LAYOUTS:
                truncated = True
            return
        edge = ordered_edges[k]
        px, py = positions[edge.parent]
        for dx, dy in candidates[edge.pid]:
            nodes_expanded += 1
            if nodes_expanded > MAX_NODES_EXPANDED:
                truncated = True
                return
            positions[edge.child] = (px + edge.sign * dx, py + edge.sign * dy)
            if consistent(positions):
                backtrack(k + 1, positions)
                if truncated:
                    del positions[edge.child]
                    return
            del positions[edge.child]

    backtrack(0, {0: (0, 0)})
    result.truncated = truncated

    if not layouts:
        if truncated:
            # The budget ran out before any layout was completed, so an empty
            # set does NOT prove infeasibility.  Reporting no_solution here
            # would claim a certainty we do not have.
            result.status = "ambiguous"
            result.uncertain_image_ids = list(range(1, n_images))
            return result
        return result  # no_solution: every combination conflicts

    ordered = list(layouts.values())
    result.layouts = ordered[: max(2, min(len(ordered), 5))]
    if len(ordered) == 1 and not truncated:
        result.status = "aligned"
        return result

    result.status = "ambiguous"
    if truncated:
        # Enumeration stopped early: several images may have further feasible
        # positions we never saw.  Report every non-anchor image as uncertain
        # rather than guessing from the layouts collected so far.
        result.uncertain_image_ids = list(range(1, n_images))
        return result
    result.uncertain_image_ids = [
        i
        for i in range(n_images)
        if len({layout.positions[i] for layout in ordered}) > 1
    ]
    return result
