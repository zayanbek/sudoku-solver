"""Backtracking Sudoku solver.

Uses bitmasks for the row/column/box constraints and always branches on the
empty cell with the fewest candidates (MRV heuristic), which keeps even the
"hardest" published puzzles down to a few milliseconds.
"""

from __future__ import annotations

from typing import List, Optional

Grid = List[List[int]]

ALL = 0x3FE  # bits 1..9 set


def _box(r: int, c: int) -> int:
    return (r // 3) * 3 + c // 3


def is_valid(grid: Grid) -> bool:
    """Check that the given clues don't already break any Sudoku rule."""
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for r in range(9):
        for c in range(9):
            v = grid[r][c]
            if not v:
                continue
            if not 1 <= v <= 9:
                return False
            bit = 1 << v
            b = _box(r, c)
            if rows[r] & bit or cols[c] & bit or boxes[b] & bit:
                return False
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
    return True


def conflicts(grid: Grid) -> set[tuple[int, int]]:
    """Return every cell that clashes with another clue."""
    bad: set[tuple[int, int]] = set()
    for r in range(9):
        for c in range(9):
            v = grid[r][c]
            if not v:
                continue
            for rr in range(9):
                for cc in range(9):
                    if (rr, cc) == (r, c) or grid[rr][cc] != v:
                        continue
                    if rr == r or cc == c or _box(rr, cc) == _box(r, c):
                        bad.add((r, c))
    return bad


def solve(grid: Grid) -> Optional[Grid]:
    """Return a solved copy of ``grid`` or ``None`` if it has no solution."""
    if not is_valid(grid):
        return None

    board = [row[:] for row in grid]
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    empty: list[tuple[int, int]] = []

    for r in range(9):
        for c in range(9):
            v = board[r][c]
            if v:
                bit = 1 << v
                rows[r] |= bit
                cols[c] |= bit
                boxes[_box(r, c)] |= bit
            else:
                empty.append((r, c))

    def search() -> bool:
        if not empty:
            return True

        # Pick the most constrained cell.
        best_i, best_mask, best_count = -1, 0, 10
        for i, (r, c) in enumerate(empty):
            mask = ALL & ~(rows[r] | cols[c] | boxes[_box(r, c)])
            count = bin(mask).count("1")
            if count < best_count:
                best_i, best_mask, best_count = i, mask, count
                if count <= 1:
                    break
        if best_count == 0:
            return False

        r, c = empty[best_i]
        empty[best_i] = empty[-1]
        empty.pop()
        b = _box(r, c)

        mask = best_mask
        while mask:
            bit = mask & -mask
            mask ^= bit
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            board[r][c] = bit.bit_length() - 1
            if search():
                return True
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit

        board[r][c] = 0
        empty.append((r, c))
        empty[best_i], empty[-1] = empty[-1], empty[best_i]
        return False

    return board if search() else None


def format_grid(grid: Grid) -> str:
    lines = []
    for r, row in enumerate(grid):
        if r and r % 3 == 0:
            lines.append("------+-------+------")
        parts = []
        for c, v in enumerate(row):
            if c and c % 3 == 0:
                parts.append("|")
            parts.append(str(v) if v else ".")
        lines.append(" ".join(parts))
    return "\n".join(lines)
