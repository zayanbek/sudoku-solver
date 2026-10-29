"""Finding the Sudoku grid in a photo and reading its digits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .ocr import DigitClassifier, normalize
from .solver import conflicts

WARP = 450  # side of the top-down grid image, 50px per cell
CELL = WARP // 9
SOLUTION_BGR = (224, 122, 31)  # same blue as the solution board


class GridNotFound(Exception):
    pass


@dataclass
class Detection:
    grid: List[List[int]]
    corners: np.ndarray  # 4x2 float32, TL TR BR BL in source coordinates
    warped: np.ndarray  # top-down BGR view of the puzzle
    source: np.ndarray  # the original BGR image
    found_cells: List[tuple[int, int]] = field(default_factory=list)


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],  # top-left
        pts[np.argmin(d)],  # top-right
        pts[np.argmax(s)],  # bottom-right
        pts[np.argmax(d)],  # bottom-left
    ], np.float32)


def _preprocess(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return cv2.dilate(thresh, np.ones((2, 2), np.uint8))


def find_grid(image: np.ndarray, min_frac: float = 0.05, max_frac: float = 1.0) -> np.ndarray:
    """Return the four corners of the largest square-ish contour."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = _preprocess(gray)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = image.shape[0] * image.shape[1]

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(cnt)
        if area > max_frac * total:
            continue
        if area < min_frac * total:
            break
        peri = cv2.arcLength(cnt, True)
        for eps in (0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return order_corners(approx)

        # Fall back to the minimum-area rectangle when the outline is ragged.
        box = cv2.boxPoints(cv2.minAreaRect(cnt))
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        if 0.7 < w / max(h, 1) < 1.4 and area > 0.6 * w * h:
            return order_corners(box)

    raise GridNotFound("Couldn't find a Sudoku grid in this image.")


def locate(image: np.ndarray) -> np.ndarray:
    """Find the grid, then look inside it in case we caught the paper's edge.

    On a photo of a sheet of paper the biggest quad is often the page itself,
    so after straightening it we search again for a smaller grid inside and
    map those corners back to the original image.
    """
    corners = find_grid(image)
    for _ in range(2):
        dst = np.array([[0, 0], [WARP - 1, 0], [WARP - 1, WARP - 1], [0, WARP - 1]], np.float32)
        m = cv2.getPerspectiveTransform(corners, dst)
        flat = cv2.warpPerspective(image, m, (WARP, WARP))
        try:
            inner = find_grid(flat, min_frac=0.3, max_frac=0.9)
        except GridNotFound:
            break
        back = cv2.perspectiveTransform(inner.reshape(-1, 1, 2), np.linalg.inv(m))
        corners = order_corners(back)
    return corners


def warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    dst = np.array([[0, 0], [WARP - 1, 0], [WARP - 1, WARP - 1], [0, WARP - 1]], np.float32)
    m = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, m, (WARP, WARP))


def _clear_border(binary: np.ndarray) -> np.ndarray:
    """Remove blobs touching the cell edge (grid lines bleeding in)."""
    h, w = binary.shape
    mask = binary.copy()
    flood = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(w):
        for y in (0, h - 1):
            if mask[y, x]:
                cv2.floodFill(mask, flood, (x, y), 0)
    for y in range(h):
        for x in (0, w - 1):
            if mask[y, x]:
                cv2.floodFill(mask, flood, (x, y), 0)
    return mask


def extract_digit(cell: np.ndarray) -> Optional[np.ndarray]:
    """Return a normalised digit image, or None if the cell looks empty."""
    h, w = cell.shape
    m = int(0.08 * h)
    cell = cell[m:h - m, m:w - m]
    h, w = cell.shape

    _, binary = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Otsu on a blank cell picks noise; reject if the contrast is tiny.
    if cell.max() - cell.min() < 40:
        return None
    binary = _clear_border(binary)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None

    # Keep the biggest blob that sits reasonably near the centre.
    best, best_area = 0, 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        cx, cy = x + bw / 2, y + bh / 2
        if abs(cx - w / 2) > w * 0.3 or abs(cy - h / 2) > h * 0.3:
            continue
        if bh < h * 0.3 or area < 0.02 * h * w:
            continue
        if area > best_area:
            best, best_area = i, area
    if not best:
        return None

    digit = np.where(labels == best, 255, 0).astype(np.uint8)
    return normalize(digit)


_classifier: Optional[DigitClassifier] = None


def classifier() -> DigitClassifier:
    global _classifier
    if _classifier is None:
        _classifier = DigitClassifier()
    return _classifier


def read_grid(warped: np.ndarray) -> tuple[List[List[int]], List[tuple[int, int]]]:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    digits, cells = [], []
    for r in range(9):
        for c in range(9):
            cell = gray[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
            d = extract_digit(cell)
            if d is not None:
                digits.append(d)
                cells.append((r, c))

    grid = [[0] * 9 for _ in range(9)]
    ranked = {}
    for (r, c), labels in zip(cells, classifier().rank(digits)):
        grid[r][c] = labels[0]
        ranked[(r, c)] = labels
    return repair(grid, ranked), cells


def repair(grid: List[List[int]], ranked: dict, max_fixes: int = 4) -> List[List[int]]:
    """Use the Sudoku rules to fix likely misreads.

    A misread digit usually clashes with a correctly read one in the same row,
    column or box (a 6 read as an 8 next to a real 8). While there are clashes,
    try the classifier's runner-up guesses (2nd and 3rd) for each clashing cell
    and keep the single change that removes the most clashes.
    """
    grid = [row[:] for row in grid]
    for _ in range(max_fixes):
        bad = conflicts(grid)
        if not bad:
            break
        best = None  # (clashes left, guess rank, cell, value)
        for cell in bad:
            r, c = cell
            original = grid[r][c]
            for rank_i, alt in enumerate(ranked[cell][1:3], start=1):
                grid[r][c] = alt
                score = (len(conflicts(grid)), rank_i, cell, alt)
                if score[0] < len(bad) and (best is None or score[:2] < best[:2]):
                    best = score
            grid[r][c] = original
        if best is None:
            break
        (r, c), alt = best[2], best[3]
        grid[r][c] = alt
    return grid


def detect(image: np.ndarray) -> Detection:
    corners = locate(image)
    top_down = warp(image, corners)
    grid, cells = read_grid(top_down)
    return Detection(grid, corners, top_down, image, cells)


def overlay_solution(det: Detection, solution: List[List[int]]) -> np.ndarray:
    """Draw the missing digits back onto the original photo."""
    # Draw onto a single-channel alpha mask so anti-aliased edges blend cleanly.
    alpha = np.zeros((WARP, WARP), np.uint8)
    for r in range(9):
        for c in range(9):
            if det.grid[r][c]:
                continue
            text = str(solution[r][c])
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
            x = c * CELL + (CELL - tw) // 2
            y = r * CELL + (CELL + th) // 2
            cv2.putText(alpha, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 2, cv2.LINE_AA)

    src = np.array([[0, 0], [WARP - 1, 0], [WARP - 1, WARP - 1], [0, WARP - 1]], np.float32)
    m = cv2.getPerspectiveTransform(src, det.corners)
    h, w = det.source.shape[:2]
    a = cv2.warpPerspective(alpha, m, (w, h)).astype(np.float32)[..., None] / 255.0
    color = np.array(SOLUTION_BGR, np.float32)
    out = det.source.astype(np.float32) * (1 - a) + color * a
    return out.astype(np.uint8)
