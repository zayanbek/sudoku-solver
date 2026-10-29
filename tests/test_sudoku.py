import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from make_sample import PUZZLE, WORKSHEET, photograph, render, render_page  # noqa: E402
from sudoku import vision  # noqa: E402
from sudoku.solver import conflicts, is_valid, solve  # noqa: E402

HARD = [
    [8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 6, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 9, 0, 2, 0, 0],
    [0, 5, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 4, 5, 7, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 3, 0],
    [0, 0, 1, 0, 0, 0, 0, 6, 8],
    [0, 0, 8, 5, 0, 0, 0, 1, 0],
    [0, 9, 0, 0, 0, 0, 4, 0, 0],
]


def assert_solved(tc, puzzle, solution):
    tc.assertIsNotNone(solution)
    target = set(range(1, 10))
    for i in range(9):
        tc.assertEqual(set(solution[i]), target)
        tc.assertEqual({solution[r][i] for r in range(9)}, target)
        br, bc = (i // 3) * 3, (i % 3) * 3
        tc.assertEqual({solution[br + r][bc + c] for r in range(3) for c in range(3)}, target)
    for r in range(9):
        for c in range(9):
            if puzzle[r][c]:
                tc.assertEqual(puzzle[r][c], solution[r][c])


class SolverTests(unittest.TestCase):
    def test_classic(self):
        assert_solved(self, PUZZLE, solve(PUZZLE))

    def test_hard(self):
        assert_solved(self, HARD, solve(HARD))

    def test_does_not_mutate_input(self):
        before = [row[:] for row in PUZZLE]
        solve(PUZZLE)
        self.assertEqual(before, PUZZLE)

    def test_invalid_clues(self):
        bad = [row[:] for row in PUZZLE]
        bad[0][2] = 5  # duplicate 5 in the first row
        self.assertFalse(is_valid(bad))
        self.assertIsNone(solve(bad))
        self.assertIn((0, 0), conflicts(bad))
        self.assertIn((0, 2), conflicts(bad))

    def test_unsolvable(self):
        grid = [[0] * 9 for _ in range(9)]
        grid[0][:8] = [1, 2, 3, 4, 5, 6, 7, 8]
        grid[1][8] = 9
        self.assertIsNone(solve(grid))


class VisionTests(unittest.TestCase):
    def test_reads_flat_scan(self):
        det = vision.detect(render(PUZZLE))
        self.assertEqual(det.grid, PUZZLE)

    def test_reads_skewed_photo(self):
        det = vision.detect(photograph(render(PUZZLE)))
        self.assertEqual(det.grid, PUZZLE)

    def test_reads_worksheet(self):
        det = vision.detect(render_page(WORKSHEET))
        self.assertEqual(det.grid, WORKSHEET)

    def test_reads_worksheet_photo(self):
        det = vision.detect(photograph(render_page(WORKSHEET)))
        self.assertEqual(det.grid, WORKSHEET)

    def test_repair_uses_runner_up_guess(self):
        misread = [row[:] for row in PUZZLE]
        misread[0][1] = 5  # the 3 was read as a 5, clashing with the 5 beside it
        ranked = {(r, c): [misread[r][c]] + [d for d in range(1, 10) if d != misread[r][c]]
                  for r in range(9) for c in range(9) if misread[r][c]}
        ranked[(0, 1)] = [5, 3, 8, 6, 9, 1, 2, 4, 7]
        ranked[(0, 0)] = [5, 6, 8, 1, 2, 3, 4, 7, 9]  # 6 and 8 both clash in column 0
        self.assertEqual(vision.repair(misread, ranked), PUZZLE)

    def test_overlay_has_same_shape(self):
        img = photograph(render(PUZZLE))
        det = vision.detect(img)
        out = vision.overlay_solution(det, solve(det.grid))
        self.assertEqual(out.shape, img.shape)


if __name__ == "__main__":
    unittest.main()
