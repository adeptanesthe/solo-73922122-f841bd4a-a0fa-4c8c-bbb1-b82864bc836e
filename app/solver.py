"""Dimension line recognition solver.

The input is one binary grayscale line image (ink = pixel 0) plus a set of
labelled glyph samples of the same height.  The line may be segmented at any
column boundary; all-white columns at the ends or between segments may be
skipped, but every kept segment must contain ink and must match a sample of
the *same width* pixel by pixel.  A segment's cost is the number of differing
pixels (XOR count) against the best sample carrying the candidate label; the
line's cost is the sum over its characters.

Legal readings contain 1..8 digits and optionally one decimal point with a
digit on each side; leading zeros and trailing fractional zeros are allowed.
Many (segmentation, spelling) pairs can represent the same numeric value
(e.g. ``1.5``/``1.50`` or alternative cut positions), so each distinct value
keeps only its cheapest representation.

A plain exhaustive search is exponential in the line width.  The solver
instead:

1. builds a table of every legal segment ``(start, width)`` with the cheapest
   matching label and its XOR cost;
2. computes an admissible backward dynamic program ``h[col][digits][mode]``
   giving the cheapest legal completion from any partial state, which also
   proves when no legal reading exists;
3. enumerates complete readings through a best-first heap ordered by
   ``g + h`` (A*), so the cheapest readings pop first, and stops once the
   second distinct value is provably optimal.

A fixed push budget bounds worst-case work; if it is exhausted the caller is
told to route the line to manual review rather than risk an unproven answer.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .models import MAX_DIGITS

# Modes of the partial-reading automaton.
MODE_DIGITS = 0  # digits only so far (last character is a digit, or empty at d=0)
MODE_AFTER_DOT = 1  # last character is the dot; a following digit is required
MODE_FRACTION = 2  # dot present and at least one digit follows it
ACCEPT_MODES = (MODE_DIGITS, MODE_FRACTION)

# Hard budget on heap pushes for one recognition request.  Hitting it means the
# proof of optimality cannot be completed, so the response must be reviewed.
MAX_HEAP_PUSHES = 300_000
# Safety cap on distinct completed values retained during enumeration.
MAX_COMPLETED_VALUES = 10_000
# Per-state dominance width.  At a fixed state (column, digit count, mode), at
# most ``d - 1`` distinct texts can spell the *same* numeric value when a dot is
# present (one per legal dot position, e.g. "0.10"/"00.1"), and just one
# without a dot.  Keeping MAX_DIGITS + 1 cheapest paths therefore always leaves
# room for the path leading to the second distinct value, so the two cheapest
# distinct values are recovered exactly.  Any further path to the same state is
# dominated (identical continuations, equal or higher cost) and dropped.
K_BEST_PER_STATE = MAX_DIGITS + 1

_INF = np.iinfo(np.int64).max


@dataclass(frozen=True)
class GlyphSample:
    label: str
    width: int
    ink: np.ndarray  # boolean array, shape (height, width)


@dataclass(frozen=True)
class CharSpan:
    char: str
    start_col: int
    end_col: int
    mismatches: int


@dataclass(frozen=True)
class Reading:
    raw_text: str
    mismatches: int
    chars: tuple[CharSpan, ...]


@dataclass
class RecognitionResult:
    readings: list[Reading]  # distinct values, cheapest representation each, sorted
    truncated: bool


def canonical_value(raw_text: str) -> str:
    """Normalize a legal reading to its canonical numeric spelling.

    Leading zeros of the integer part collapse to one zero; trailing zeros of
    the fractional part (and a dangling dot) are removed.
    """
    if "." in raw_text:
        integer, fraction = raw_text.split(".", 1)
        integer = integer.lstrip("0") or "0"
        fraction = fraction.rstrip("0")
        return integer if not fraction else f"{integer}.{fraction}"
    return str(int(raw_text))


def _build_edge_table(
    line_ink: np.ndarray, samples: list[GlyphSample]
) -> dict[int, list[tuple[int, int, str]]]:
    """For every start column, list ``(end_col, cost, label)`` segment options.

    Only segments containing ink are emitted.  When several samples share a
    label, the cheapest (minimum XOR) one wins per segment.
    """
    height, width = line_ink.shape
    # Prefix sum of inked columns gives an O(1) "segment has ink" test.
    col_ink = line_ink.any(axis=0).astype(np.int32)
    prefix = np.zeros(width + 1, dtype=np.int32)
    np.cumsum(col_ink, out=prefix[1:])

    # Group samples by width; labels may repeat across samples of equal width.
    by_width: dict[int, dict[str, list[np.ndarray]]] = {}
    for sample in samples:
        by_width.setdefault(sample.width, {}).setdefault(sample.label, []).append(sample.ink)

    # All sliding H x w windows of the line, indexed by start column.  A sample
    # wider than the line cannot match anywhere; the service layer rejects this
    # earlier, but guard here too so direct callers get an empty edge set.
    windows = {
        w: np.lib.stride_tricks.sliding_window_view(line_ink, (height, w)).reshape(
            width - w + 1, height, w
        )
        for w in by_width
        if w <= width
    }
    by_width = {w: labels for w, labels in by_width.items() if w <= width}

    edges: dict[int, list[tuple[int, int, str]]] = {}
    for w, label_templates in by_width.items():
        windowed = windows[w]  # (S, H, w) boolean
        for start in range(width - w + 1):
            if prefix[start + w] - prefix[start] == 0:
                continue  # all-white segment: not a character, skip it
            segment = windowed[start]
            options: list[tuple[int, str]] = []
            for label, templates in label_templates.items():
                best = min(int(np.count_nonzero(segment != tpl)) for tpl in templates)
                options.append((best, label))
            # Cheapest label first; ties order deterministically by label so the
            # solver explores stable branches first.
            options.sort()
            edges.setdefault(start, []).extend(
                (start + w, cost, label) for cost, label in options
            )
    return edges


def _backward_heuristic(
    width: int, edges: dict[int, list[tuple[int, int, str]]], col_has_ink: np.ndarray
) -> np.ndarray:
    """Cheapest legal completion cost from each state, or INF if none exists.

    State is ``(column, digits used, mode)``.  Advancing one column without
    consuming a character (skipping an all-white column) costs nothing and is
    legal only when that column carries no ink: every inked column has to be
    covered by a character segment.
    """
    h = np.full((width + 1, MAX_DIGITS + 1, 3), _INF, dtype=np.int64)
    # Accepting states at the right edge.
    for d in range(1, MAX_DIGITS + 1):
        h[width, d, MODE_DIGITS] = 0
        h[width, d, MODE_FRACTION] = 0

    for col in range(width - 1, -1, -1):
        # Free skip: the same state can simply move one column right, but only
        # across a fully white column (inked columns must belong to a segment).
        if not col_has_ink[col]:
            h[col] = h[col + 1].copy()
        for end, cost, label in edges.get(col, ()):
            if label == ".":
                # A dot can only follow at least one digit and starts MODE_AFTER_DOT.
                for d in range(1, MAX_DIGITS + 1):
                    nxt = h[end, d, MODE_AFTER_DOT]
                    if nxt < _INF:
                        cand = cost + nxt
                        if cand < h[col, d, MODE_DIGITS]:
                            h[col, d, MODE_DIGITS] = cand
            else:
                for d in range(MAX_DIGITS):
                    nd = d + 1
                    # Digit extending a digits-only prefix (also the first digit).
                    nxt = h[end, nd, MODE_DIGITS]
                    if nxt < _INF:
                        cand = cost + nxt
                        if cand < h[col, d, MODE_DIGITS]:
                            h[col, d, MODE_DIGITS] = cand
                    # Digit right after the dot: after-dot -> fraction.
                    nxt = h[end, nd, MODE_FRACTION]
                    if nxt < _INF:
                        cand = cost + nxt
                        if cand < h[col, d, MODE_AFTER_DOT]:
                            h[col, d, MODE_AFTER_DOT] = cand
                    # Digit extending the fractional part.
                    if nxt < _INF:
                        cand = cost + nxt
                        if cand < h[col, d, MODE_FRACTION]:
                            h[col, d, MODE_FRACTION] = cand
    return h


def _successor_mode(mode: int, label: str, d: int) -> int | None:
    """Automaton transition; None means the move is illegal in this state."""
    if label == ".":
        return MODE_AFTER_DOT if mode == MODE_DIGITS and d >= 1 else None
    if mode == MODE_DIGITS:
        return MODE_DIGITS
    if mode == MODE_AFTER_DOT:
        return MODE_FRACTION
    return MODE_FRACTION


def recognize(line_ink: np.ndarray, samples: list[GlyphSample]) -> RecognitionResult:
    """Return the cheapest readings (up to two distinct values) of one line."""
    width = line_ink.shape[1]
    col_has_ink = line_ink.any(axis=0)
    edges = _build_edge_table(line_ink, samples)
    h = _backward_heuristic(width, edges, col_has_ink)

    # Heap entry: (f, tie_counter, col, digits, mode, g, raw_text, spans).
    heap: list[tuple] = []
    counter = 0
    initial_h = int(h[0, 0, MODE_DIGITS])
    truncated = False
    if initial_h >= _INF:
        # The backward DP proves no legal reading exists; this is not truncation.
        return RecognitionResult(readings=[], truncated=False)
    heapq.heappush(heap, (initial_h, counter, 0, 0, MODE_DIGITS, 0, "", ()))
    pushes = 1

    # Dominance pruning: for every (col, digits, mode) keep only the costs of
    # the K_BEST_PER_STATE cheapest paths that reached it.  Reaching the same
    # state with a higher accumulated cost can never lead to a better reading,
    # because all continuations from that state are identical.
    state_best: dict[tuple[int, int, int], list[int]] = {}

    def admit(state: tuple[int, int, int], cost: int) -> bool:
        costs = state_best.get(state)
        if costs is None:
            state_best[state] = [cost]
            return True
        if len(costs) >= K_BEST_PER_STATE and cost >= costs[-1]:
            return False
        costs.append(cost)
        costs.sort()
        del costs[K_BEST_PER_STATE:]
        return True

    # Distinct numeric value -> best Reading representation seen so far.
    completed: dict[str, Reading] = {}
    best_key: dict[str, tuple] = {}

    def runner_up_cost() -> int | None:
        if len(completed) < 2:
            return None
        return sorted(reading.mismatches for reading in completed.values())[1]

    stop_at: int | None = None
    while heap:
        # h is admissible, so an entry whose lower bound f already exceeds the
        # runner-up cost cannot beat it.  Entries with f == stop_at still have to
        # be expanded because their real g may be exactly stop_at.
        if stop_at is not None and heap[0][0] > stop_at:
            break
        f, _tie, col, d, mode, g, text, spans = heapq.heappop(heap)
        if stop_at is not None and g > stop_at:
            continue
        if col == width:
            if mode in ACCEPT_MODES and d >= 1:
                value = canonical_value(text)
                key = (g, text, tuple((s.start_col, s.end_col, s.mismatches) for s in spans))
                if value not in completed or key < best_key[value]:
                    completed[value] = Reading(raw_text=text, mismatches=g, chars=spans)
                    best_key[value] = key
                    if len(completed) > MAX_COMPLETED_VALUES:
                        truncated = True
                        break
                    stop_at = runner_up_cost()
            continue

        def push_state(next_col: int, next_d: int, next_mode: int, next_g: int,
                       next_text: str, next_spans: tuple) -> None:
            nonlocal counter, pushes, truncated
            if truncated or not admit((next_col, next_d, next_mode), next_g):
                return
            bound = int(h[next_col, next_d, next_mode])
            if bound >= _INF:
                return
            heapq.heappush(
                heap,
                (next_g + bound, counter, next_col, next_d, next_mode, next_g, next_text, next_spans),
            )
            counter += 1
            pushes += 1
            if pushes > MAX_HEAP_PUSHES:
                truncated = True

        # Free skip of one all-white column (inked columns must be covered).
        if not col_has_ink[col]:
            push_state(col + 1, d, mode, g, text, spans)
            if truncated:
                break

        for end, cost, label in edges.get(col, ()):
            nm = _successor_mode(mode, label, d)
            if nm is None:
                continue
            nd = d if label == "." else d + 1
            if nd > MAX_DIGITS:
                continue
            ng = g + cost
            span = CharSpan(char=label, start_col=col, end_col=end, mismatches=cost)
            push_state(end, nd, nm, ng, text + label, spans + (span,))
            if truncated:
                break
        if truncated:
            break

    # Deterministic ordering: cheapest first; ties break by canonical value,
    # then by raw spelling and finally by column ranges.  Which tied values are
    # retained can depend on the dominance pruning, but every retained value's
    # cost is exact and equal-cost ties are always routed to review anyway
    # (their margin is 0, below any positive min_margin).
    ordered = sorted(
        completed.values(),
        key=lambda r: (
            r.mismatches,
            canonical_value(r.raw_text),
            r.raw_text,
            tuple((s.start_col, s.end_col) for s in r.chars),
        ),
    )
    return RecognitionResult(readings=ordered[:2], truncated=truncated)
