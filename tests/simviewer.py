"""A fake continuous-scroll viewer for tests: renders slices of a synthetic page (no real book)."""
from __future__ import annotations

import numpy as np
from PIL import Image


def make_document(height: int = 7000, width: int = 640, seed: int = 1, figures: tuple = ((2600, 900),),
                  table_at: int | None = None) -> np.ndarray:
    """White page with text-like lines, blank gaps between paragraphs and tall figures."""
    rng = np.random.default_rng(seed)
    doc = np.full((height, width, 3), 255, dtype=np.uint8)
    y = 30
    fig_ranges = [(a, a + h) for a, h in figures]
    while y < height - 30:
        hit = next((r for r in fig_ranges if r[0] <= y < r[1]), None)
        if hit:
            a, b = hit[0], min(hit[1], height - 30)
            if b - a < 50:
                break
            yy, xx = np.mgrid[0:b - a, 0:width - 80]
            fig = (rng.integers(0, 90, size=(b - a, width - 80, 1)) + (yy // 3 % 60)[..., None]
                   + (xx // 5 % 40)[..., None]).astype(np.uint8)
            doc[a:b, 40:width - 40] = np.repeat(fig, 3, axis=2) // 2 + 20
            y = b + 40
            continue
        if table_at is not None and table_at <= y < table_at + 400:
            doc[y:y + 14, 40:width - 40] = 60      # identical rows: a periodic table
            y += 28
            continue
        if rng.random() < 0.12:
            y += int(rng.integers(30, 70))        # paragraph gap
            continue
        x = 40
        while x < width - 60:
            w = int(rng.integers(8, 46))
            if rng.random() < 0.9:
                doc[y:y + 14, x:x + w] = int(rng.integers(20, 110))
            x += w + int(rng.integers(6, 14))
        y += 26
    return doc


class SimViewer:
    """scroll(units) moves the page by units*ppu pixels; grab() shows the current view."""

    def __init__(self, doc: np.ndarray, view_h: int = 600, ppu: float = 100, header: int = 0,
                 lazy: int = 0, jitter: float = 0.0, seed: int = 3, on_scroll=None, dead: bool = False):
        self.doc, self.vh, self.ppu = doc, view_h, ppu
        self.header, self.lazy, self.jitter = header, lazy, jitter
        self.rng = np.random.default_rng(seed)
        self.pos = 0
        self.pending = 0
        self.rendered = (0, view_h)
        self.on_scroll = on_scroll
        self.dead = dead
        self.calls = 0
        hdr = np.random.default_rng(99).integers(0, 255, size=(max(header, 1), doc.shape[1], 3), dtype=np.uint8)
        self.header_img = hdr[:header]

    def scroll(self, units: int) -> None:
        self.calls += 1
        if self.dead:
            return
        step = units * self.ppu
        if self.jitter:
            step *= 1 + self.rng.uniform(-self.jitter, self.jitter)
        extra = self.on_scroll(self) if self.on_scroll else 0
        self.pos = int(np.clip(self.pos + round(step) + extra, 0, len(self.doc) - self.vh))
        self.pending = self.lazy

    def grab(self) -> Image.Image:
        view = self.doc[self.pos:self.pos + self.vh].copy()
        lo, hi = self.rendered
        if self.pending > 0:
            self.pending -= 1
            rows = np.arange(self.pos, self.pos + self.vh)
            view[(rows < lo) | (rows >= hi)] = 255
        else:
            self.rendered = (min(lo, self.pos), max(hi, self.pos + self.vh))
        if self.header:
            view[:self.header] = self.header_img
        return Image.fromarray(view)
