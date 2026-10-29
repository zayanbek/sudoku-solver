"""Printed digit recognition: HOG features + k-nearest-neighbours.

There is no pre-trained model shipped with the project. Instead the classifier
is trained on synthetic digits rendered from whatever TrueType fonts are on the
machine (plus OpenCV's built-in Hershey fonts as a fallback), with a bit of
rotation, scale, blur and thickness jitter so it generalises to photos.
The trained samples are cached next to this file so start-up stays fast.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Iterable, List

import cv2
import numpy as np

SIZE = 28  # side of the normalised digit image
CACHE = os.path.join(os.path.dirname(__file__), ".digit_samples.npz")
CACHE_VERSION = 2

FONT_NAMES = [
    "arial", "arialbd", "ARIALN", "calibri", "calibrib", "verdana", "verdanab",
    "tahoma", "tahomabd", "times", "timesbd", "georgia", "georgiab", "cour",
    "courbd", "segoeui", "segoeuib", "trebuc", "trebucbd", "consola",
    "DejaVuSans", "DejaVuSans-Bold", "DejaVuSerif", "LiberationSans-Regular",
    "LiberationSans-Bold", "LiberationSerif-Regular", "Helvetica", "Arial",
]
FONT_DIRS = [
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
    os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts"),
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    "/Library/Fonts",
    "/System/Library/Fonts",
]
HERSHEY = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_PLAIN,
]


def hog(img: np.ndarray, cell: int = 7, bins: int = 9) -> np.ndarray:
    """Histogram of oriented gradients over a grid of cells.

    Written out by hand rather than using cv2.HOGDescriptor, which isn't part of
    every OpenCV build (it moved out of the core package in OpenCV 5).
    """
    f = img.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = np.mod(ang, 180.0)  # unsigned gradients
    idx = np.minimum((ang / (180.0 / bins)).astype(np.int32), bins - 1)

    n = img.shape[0] // cell
    hist = np.zeros((n, n, bins), np.float32)
    for cy in range(n):
        for cx in range(n):
            sl = (slice(cy * cell, (cy + 1) * cell), slice(cx * cell, (cx + 1) * cell))
            hist[cy, cx] = np.bincount(idx[sl].ravel(), mag[sl].ravel(), bins)

    # Normalise over overlapping 2x2 blocks of cells.
    blocks = []
    for by in range(n - 1):
        for bx in range(n - 1):
            v = hist[by:by + 2, bx:bx + 2].ravel()
            blocks.append(v / (np.linalg.norm(v) + 1e-6))
    return np.concatenate(blocks)


def normalize(binary: np.ndarray) -> np.ndarray | None:
    """Crop a white-on-black digit and centre it in a SIZE x SIZE square."""
    pts = cv2.findNonZero(binary)
    if pts is None:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    digit = binary[y:y + h, x:x + w]

    # Fit the longest side into 20px like MNIST, keep the aspect ratio.
    scale = 20.0 / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    digit = cv2.resize(digit, (nw, nh), interpolation=cv2.INTER_AREA)

    out = np.zeros((SIZE, SIZE), np.uint8)
    ox, oy = (SIZE - nw) // 2, (SIZE - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = digit
    return out


def features(digits: Iterable[np.ndarray]) -> np.ndarray:
    rows = []
    for d in digits:
        small = cv2.resize(d, (14, 14), interpolation=cv2.INTER_AREA).ravel() / 255.0
        rows.append(np.concatenate([hog(d), small * 0.5]))
    return np.asarray(rows, np.float32)


def _find_fonts() -> List[str]:
    found = []
    for name in FONT_NAMES:
        for folder in FONT_DIRS:
            hits = glob.glob(os.path.join(folder, "**", name + ".tt[fc]"), recursive=True)
            if hits:
                found.append(hits[0])
                break
    return found


def _augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape
    angle = rng.uniform(-8, 8)
    scale = rng.uniform(0.85, 1.1)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += rng.uniform(-3, 3)
    m[1, 2] += rng.uniform(-3, 3)
    img = cv2.warpAffine(img, m, (w, h))

    k = rng.choice([0, 0, 1, 2])
    if k == 1:
        img = cv2.dilate(img, np.ones((2, 2), np.uint8))
    elif k == 2:
        img = cv2.erode(img, np.ones((2, 2), np.uint8))
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    # Half the time, simulate a small digit in a low-resolution photo: shrink
    # it so it's only 14-24px tall, blur, and threshold like the real cells are.
    if rng.random() < 0.5:
        pts = cv2.findNonZero(img)
        if pts is not None:
            glyph_h = cv2.boundingRect(pts)[3]
            f = rng.uniform(14, 24) / max(glyph_h, 1)
            small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (3, 3), 0)
            _, small = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            return small

    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    return img


def _render_ttf(path: str, digit: int) -> np.ndarray | None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    try:
        font = ImageFont.truetype(path, 48)
    except OSError:
        return None
    canvas = Image.new("L", (72, 72), 0)
    ImageDraw.Draw(canvas).text((14, 4), str(digit), fill=255, font=font)
    return np.array(canvas)


def _render_hershey(font: int, digit: int, thickness: int) -> np.ndarray:
    canvas = np.zeros((72, 72), np.uint8)
    cv2.putText(canvas, str(digit), (16, 56), font, 1.8, 255, thickness, cv2.LINE_AA)
    return canvas


def _build_samples(per_font: int = 12) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(1234)
    fonts = _find_fonts()
    samples, labels = [], []

    def add(img: np.ndarray | None, label: int) -> None:
        if img is None or not img.any():
            return
        for _ in range(per_font):
            d = normalize(_augment(img, rng))
            if d is not None:
                samples.append(d)
                labels.append(label)

    for digit in range(1, 10):
        for path in fonts:
            add(_render_ttf(path, digit), digit)
        for font in HERSHEY:
            for thickness in (2, 3, 4):
                add(_render_hershey(font, digit, thickness), digit)

    return features(samples), np.asarray(labels, np.int32)


class DigitClassifier:
    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.x, self.y = self._load_or_build()
        self._x_sq = (self.x ** 2).sum(axis=1)

    @staticmethod
    def _load_or_build() -> tuple[np.ndarray, np.ndarray]:
        if os.path.exists(CACHE):
            try:
                data = np.load(CACHE)
                if int(data["version"]) == CACHE_VERSION:
                    return data["x"], data["y"]
            except (OSError, KeyError, ValueError):
                pass
        x, y = _build_samples()
        try:
            np.savez_compressed(CACHE, x=x, y=y, version=CACHE_VERSION)
        except OSError:
            pass
        return x, y

    def predict(self, digits: List[np.ndarray]) -> List[int]:
        return [ranked[0] for ranked in self.rank(digits)]

    def rank(self, digits: List[np.ndarray]) -> List[List[int]]:
        """For each digit, every label 1-9 from most to least likely.

        The first choice is a distance-weighted vote of the k nearest samples;
        the rest are ordered by how close the nearest sample of that digit is.
        The alternatives let the caller fix misreads that break Sudoku rules.
        """
        if not digits:
            return []
        q = features(digits)
        # Squared euclidean distance to every training sample.
        dist = (q ** 2).sum(axis=1)[:, None] + self._x_sq[None, :] - 2.0 * q @ self.x.T
        nearest = np.argpartition(dist, self.k, axis=1)[:, :self.k]

        out = []
        for row, idx in zip(dist, nearest):
            votes = np.zeros(10, np.float32)
            for i in idx:
                votes[self.y[i]] += 1.0 / (1e-3 + max(row[i], 0.0))
            best = int(votes.argmax())
            closest = [(row[self.y == d].min(), d) for d in range(1, 10) if d != best]
            out.append([best] + [d for _, d in sorted(closest)])
        return out
