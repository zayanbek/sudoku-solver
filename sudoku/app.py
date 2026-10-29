"""Tkinter front-end: image on the left, solved grid on the right."""

from __future__ import annotations

import ctypes
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import vision
from .solver import conflicts, solve

BG = "#1e1f24"
PANEL = "#2a2c33"
TEXT = "#e8e8ea"
MUTED = "#9a9ca5"
ACCENT = "#4c9bff"
GIVEN = "#1b1b1f"
SOLVED = "#1f7ae0"
ERROR = "#e5484d"

PAD = 9


class SudokuApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Sudoku Solver")
        self.configure(bg=BG)
        self.minsize(900, 560)
        self.geometry("1180x720")

        self.detection: Optional[vision.Detection] = None
        self.grid: List[List[int]] = [[0] * 9 for _ in range(9)]
        self.solution: Optional[List[List[int]]] = None
        self.selected: Optional[tuple[int, int]] = None
        self.show_overlay = tk.BooleanVar(value=False)
        self._photo: Optional[ImageTk.PhotoImage] = None
        self.cell = 50.0
        self.origin = (0.0, 0.0)

        self._build_style()
        self._build_layout()
        self._draw_board()
        self.bind("<Key>", self._on_key)

        # Train/load the digit model in the background so the window opens instantly.
        self.status("Loading digit model…")
        threading.Thread(target=self._warm_up, daemon=True).start()

    # ------------------------------------------------------------------ UI

    def _build_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=PANEL, foreground=TEXT,
                        font=("Segoe UI Semibold", 13))
        style.configure("Status.TLabel", background=BG, foreground=MUTED)
        style.configure("TButton", font=("Segoe UI", 10), padding=(14, 7),
                        background=PANEL, foreground=TEXT, borderwidth=0)
        style.map("TButton", background=[("active", "#383b45"), ("disabled", PANEL)],
                  foreground=[("disabled", "#5c5e66")])
        style.configure("Accent.TButton", background=ACCENT, foreground="white")
        style.map("Accent.TButton", background=[("active", "#3a86e8"), ("disabled", "#2d4a70")])
        style.configure("TCheckbutton", background=PANEL, foreground=TEXT)
        style.map("TCheckbutton", background=[("active", PANEL)])

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1, uniform="half")
        root.columnconfigure(1, weight=1, uniform="half")
        root.rowconfigure(0, weight=1)

        # Left: uploaded image.
        left = ttk.Frame(root, style="Panel.TFrame", padding=14)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Puzzle image", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.image_canvas = tk.Canvas(left, bg="#202127", highlightthickness=0)
        self.image_canvas.grid(row=1, column=0, sticky="nsew", pady=10)
        self.image_canvas.bind("<Configure>", lambda e: self._render_image())
        self.image_canvas.bind("<Button-1>", lambda e: self.open_image())

        left_bar = ttk.Frame(left, style="Panel.TFrame")
        left_bar.grid(row=2, column=0, sticky="ew")
        self.open_btn = ttk.Button(left_bar, text="Upload image…", command=self.open_image)
        self.open_btn.pack(side="left")
        ttk.Checkbutton(left_bar, text="Show solution on photo", variable=self.show_overlay,
                        command=self._render_image).pack(side="left", padx=12)

        # Right: solution board.
        right = ttk.Frame(root, style="Panel.TFrame", padding=14)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Solution", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.board = tk.Canvas(right, width=300, height=300, bg=PANEL, highlightthickness=0)
        self.board.grid(row=1, column=0, sticky="nsew", pady=10)
        self.board.bind("<Button-1>", self._on_board_click)
        self.board.bind("<Configure>", lambda e: self._draw_board())

        right_bar = ttk.Frame(right, style="Panel.TFrame")
        right_bar.grid(row=2, column=0, sticky="ew")
        self.solve_btn = ttk.Button(right_bar, text="Solve", style="Accent.TButton",
                                    command=self.solve_puzzle)
        self.solve_btn.pack(side="left")
        ttk.Button(right_bar, text="Clear solution", command=self.clear_solution).pack(side="left", padx=8)
        ttk.Button(right_bar, text="Reset", command=self.reset).pack(side="left")

        hint = "Misread digit? Click the cell and type 1-9 (0 clears it)."
        ttk.Label(right, text=hint, background=PANEL, foreground=MUTED).grid(
            row=3, column=0, sticky="w", pady=(8, 0))

        self.status_var = tk.StringVar()
        ttk.Label(self, textvariable=self.status_var, style="Status.TLabel",
                  padding=(16, 0, 16, 10)).pack(fill="x")

    def status(self, text: str) -> None:
        self.status_var.set(text)

    # ------------------------------------------------------------- drawing

    def _render_image(self) -> None:
        c = self.image_canvas
        c.delete("all")
        cw, ch = max(c.winfo_width(), 50), max(c.winfo_height(), 50)

        if self.detection is None:
            c.create_text(cw / 2, ch / 2, fill=MUTED, font=("Segoe UI", 12),
                          text="Click here or press “Upload image…”\nto load a Sudoku photo",
                          justify="center")
            return

        det = self.detection
        if self.show_overlay.get() and self.solution:
            img = vision.overlay_solution(det, self.solution)
        else:
            img = det.source.copy()
            thickness = max(2, img.shape[1] // 300)
            cv2.polylines(img, [det.corners.astype(np.int32)], True, (255, 155, 76), thickness)

        h, w = img.shape[:2]
        scale = min(cw / w, ch / h)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        rgb = cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        c.create_image(cw / 2, ch / 2, image=self._photo)

    def _draw_board(self) -> None:
        b = self.board
        b.delete("all")
        size = max(min(b.winfo_width(), b.winfo_height()), 120)
        cell = (size - 2 * PAD) / 9
        ox = (b.winfo_width() - size) / 2 + PAD if b.winfo_width() > size else PAD
        oy = PAD
        self.cell, self.origin = cell, (ox, oy)
        end_x, end_y = ox + 9 * cell, oy + 9 * cell
        b.create_rectangle(ox, oy, end_x, end_y, fill="#fafafa", outline="")
        font_px = -int(cell * 0.55)  # negative = pixels, so it tracks the cell size

        bad = conflicts(self.grid)
        if self.selected:
            r, c = self.selected
            x, y = ox + c * cell, oy + r * cell
            b.create_rectangle(x, y, x + cell, y + cell, fill="#dbe9ff", outline="")
        for r, c in bad:
            x, y = ox + c * cell, oy + r * cell
            b.create_rectangle(x, y, x + cell, y + cell, fill="#fde2e2", outline="")

        for i in range(10):
            width = 3 if i % 3 == 0 else 1
            color = "#202020" if i % 3 == 0 else "#b8b8b8"
            px, py = ox + i * cell, oy + i * cell
            b.create_line(ox, py, end_x, py, width=width, fill=color)
            b.create_line(px, oy, px, end_y, width=width, fill=color)

        for r in range(9):
            for c in range(9):
                x, y = ox + (c + 0.5) * cell, oy + (r + 0.5) * cell
                given = self.grid[r][c]
                if given:
                    color = ERROR if (r, c) in bad else GIVEN
                    b.create_text(x, y, text=str(given), fill=color,
                                  font=("Segoe UI Semibold", font_px))
                elif self.solution:
                    b.create_text(x, y, text=str(self.solution[r][c]), fill=SOLVED,
                                  font=("Segoe UI", font_px))

    # ------------------------------------------------------------- actions

    def _warm_up(self) -> None:
        try:
            vision.classifier()
            self.after(0, self.status, "Ready. Upload a photo or screenshot of a Sudoku.")
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            self.after(0, self.status, f"Failed to load digit model: {exc}")

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a Sudoku image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"), ("All files", "*.*")],
        )
        if not path:
            return

        # cv2.imread can't handle non-ASCII paths on Windows, so decode from bytes.
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("Sudoku Solver", "That file doesn't look like an image.")
            return

        self.status("Reading puzzle…")
        self.open_btn.state(["disabled"])
        threading.Thread(target=self._detect, args=(image,), daemon=True).start()

    def _detect(self, image: np.ndarray) -> None:
        try:
            det = vision.detect(image)
        except vision.GridNotFound as exc:
            self.after(0, self._detect_failed, str(exc))
            return
        except Exception as exc:  # pragma: no cover
            self.after(0, self._detect_failed, f"Something went wrong: {exc}")
            return
        self.after(0, self._detect_done, det)

    def _detect_failed(self, msg: str) -> None:
        self.open_btn.state(["!disabled"])
        self.status(msg)
        messagebox.showwarning("Sudoku Solver", msg + "\n\nTry a straighter, well-lit photo "
                               "where the whole grid is visible.")

    def _detect_done(self, det: vision.Detection) -> None:
        self.open_btn.state(["!disabled"])
        self.detection = det
        self.grid = [row[:] for row in det.grid]
        self.solution = None
        self.selected = None
        self.show_overlay.set(False)
        self._render_image()
        self._draw_board()
        self.status(f"Found {len(det.found_cells)} clues. Check them, then press Solve.")
        self.solve_puzzle(quiet=True)

    def solve_puzzle(self, quiet: bool = False) -> None:
        if conflicts(self.grid):
            self.status("Some clues clash (shown in red). Fix them before solving.")
            return
        if sum(v != 0 for row in self.grid for v in row) < 17:
            if not quiet:
                self.status("Not enough clues — a valid Sudoku needs at least 17.")
            return

        result = solve(self.grid)
        if result is None:
            self.solution = None
            self.status("No solution exists. A digit was probably misread — click it to fix.")
        else:
            self.solution = result
            self.status("Solved!")
        self._draw_board()
        self._render_image()

    def clear_solution(self) -> None:
        self.solution = None
        self.show_overlay.set(False)
        self._draw_board()
        self._render_image()
        self.status("Solution cleared.")

    def reset(self) -> None:
        self.detection = None
        self.grid = [[0] * 9 for _ in range(9)]
        self.solution = None
        self.selected = None
        self._draw_board()
        self._render_image()
        self.status("Ready.")

    # --------------------------------------------------------------- input

    def _on_board_click(self, event: tk.Event) -> None:
        ox, oy = self.origin
        c = int((event.x - ox) // self.cell)
        r = int((event.y - oy) // self.cell)
        self.selected = (r, c) if 0 <= r < 9 and 0 <= c < 9 else None
        self._draw_board()

    def _on_key(self, event: tk.Event) -> None:
        if not self.selected:
            return
        r, c = self.selected
        moves = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}
        if event.keysym in moves:
            dr, dc = moves[event.keysym]
            self.selected = ((r + dr) % 9, (c + dc) % 9)
        elif event.char and event.char in "123456789":
            self.grid[r][c] = int(event.char)
            self.solution = None
        elif event.char == "0" or event.keysym in ("BackSpace", "Delete"):
            self.grid[r][c] = 0
            self.solution = None
        elif event.keysym == "Return":
            self.solve_puzzle()
            return
        elif event.keysym == "Escape":
            self.selected = None
        else:
            return
        self._draw_board()


def main() -> None:
    if sys.platform == "win32":
        # Render crisply on high-DPI displays instead of being bitmap-scaled.
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    app = SudokuApp()
    app.mainloop()
