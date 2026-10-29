"""Render sample Sudoku images (a worksheet, a flat grid and a skewed "photo").

    python tools/make_sample.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

WORKSHEET = [
    [0, 0, 3, 0, 2, 0, 6, 0, 0],
    [9, 0, 0, 3, 0, 5, 0, 0, 1],
    [0, 0, 1, 8, 0, 6, 4, 0, 0],
    [0, 0, 8, 1, 0, 2, 9, 0, 0],
    [7, 0, 0, 0, 0, 0, 0, 0, 8],
    [0, 0, 6, 7, 0, 8, 2, 0, 0],
    [0, 0, 2, 6, 0, 9, 5, 0, 0],
    [8, 0, 0, 2, 0, 3, 0, 0, 9],
    [0, 0, 5, 0, 1, 0, 3, 0, 0],
]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ()
    for name in names + ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(puzzle, cell: int = 60, margin: int = 40) -> np.ndarray:
    side = cell * 9 + margin * 2
    img = Image.new("RGB", (side, side), "white")
    draw = ImageDraw.Draw(img)
    font = load_font(int(cell * 0.6))

    for i in range(10):
        w = 4 if i % 3 == 0 else 1
        p = margin + i * cell
        draw.line([(margin, p), (side - margin, p)], fill="black", width=w)
        draw.line([(p, margin), (p, side - margin)], fill="black", width=w)

    for r, row in enumerate(puzzle):
        for c, v in enumerate(row):
            if v:
                x = margin + c * cell + cell / 2
                y = margin + r * cell + cell / 2
                draw.text((x, y), str(v), fill="black", font=font, anchor="mm")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def photograph(flat: np.ndarray) -> np.ndarray:
    """Fake a phone photo: perspective tilt, uneven lighting, noise."""
    h, w = flat.shape[:2]
    out_w, out_h = int(w * 1.4), int(h * 1.35)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[90, 70], [out_w - 60, 110], [out_w - 120, out_h - 50], [50, out_h - 90]])
    m = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(flat, m, (out_w, out_h), borderValue=(150, 160, 170))

    yy, xx = np.mgrid[0:out_h, 0:out_w]
    light = 0.75 + 0.25 * (xx / out_w)
    photo = np.clip(photo * light[..., None], 0, 255).astype(np.uint8)
    noise = np.random.default_rng(0).normal(0, 6, photo.shape)
    photo = np.clip(photo + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(photo, (3, 3), 0)


def render_page(puzzle, cell: int = 52) -> np.ndarray:
    """A printable worksheet: title on top, shaded clue cells, footer."""
    margin_x, top = 24, 110
    grid = cell * 9
    width, height = grid + margin_x * 2, top + grid + 90
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    title_font = load_font(44, bold=True)
    draw.text((width / 2, top / 2), "Daily Sudoku", fill="black", font=title_font, anchor="mm")

    for r, row in enumerate(puzzle):
        for c, v in enumerate(row):
            if v:
                x, y = margin_x + c * cell, top + r * cell
                draw.rectangle([x, y, x + cell, y + cell], fill=(236, 236, 236))

    for i in range(10):
        w = 3 if i % 3 == 0 else 1
        draw.line([(margin_x, top + i * cell), (margin_x + grid, top + i * cell)], fill="black", width=w)
        draw.line([(margin_x + i * cell, top), (margin_x + i * cell, top + grid)], fill="black", width=w)

    font = load_font(int(cell * 0.42))
    for r, row in enumerate(puzzle):
        for c, v in enumerate(row):
            if v:
                x = margin_x + c * cell + cell / 2
                y = top + r * cell + cell / 2
                draw.text((x, y), str(v), fill=(30, 30, 30), font=font, anchor="mm")

    footer = load_font(16)
    draw.text((width / 2, height - 40), "Puzzle #1  ·  Medium", fill=(110, 110, 110),
              font=footer, anchor="mm")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "samples")
    os.makedirs(out_dir, exist_ok=True)
    flat = render(PUZZLE)
    cv2.imwrite(os.path.join(out_dir, "worksheet.png"), render_page(WORKSHEET))
    cv2.imwrite(os.path.join(out_dir, "puzzle.png"), flat)
    cv2.imwrite(os.path.join(out_dir, "puzzle_photo.jpg"), photograph(flat))
    print("Wrote samples to", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
