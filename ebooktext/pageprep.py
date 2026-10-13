"""Page preparation applied when importing images/PDFs: drop tablet status bars, split spreads."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class ImportOptions:
    strip_bars: bool = True        # remove dark OS status/home bars at the top/bottom edge
    split_spreads: bool = True     # split landscape two-page spreads into two pages
    right_to_left: bool = False    # reading order of a spread (True for vertical-text books)
    spread_ratio: float = 1.25     # width/height at or above this is treated as a spread


def strip_system_bars(image: Image.Image, dark_below: float = 140, max_frac: float = 0.08,
                      min_px: int = 12, margin: int = 4) -> Image.Image:
    """Crop uniform dark bands (status bar / home indicator) from the top and bottom edges."""
    rgb = image.convert("RGB")
    rows = np.asarray(rgb.convert("L"), dtype=np.float32).mean(axis=1)
    limit = int(len(rows) * max_frac)

    def band(seq) -> int:
        n = 0
        for v in seq[:limit]:
            if v < dark_below:
                n += 1
            else:
                break
        return n if n >= min_px else 0

    top, bottom = band(rows), band(rows[::-1])
    if not top and not bottom:
        return rgb
    top = top + margin if top else 0
    bottom = bottom + margin if bottom else 0
    return rgb.crop((0, top, rgb.width, rgb.height - bottom))


def find_gutter(image: Image.Image, search: float = 0.08, ink_below: int = 200) -> int:
    """x position to split a spread: centre of the ink-free column run nearest the middle."""
    gray = np.asarray(image.convert("L"))
    w = gray.shape[1]
    lo, hi = int(w * (0.5 - search)), int(w * (0.5 + search))
    blank = (gray[:, lo:hi] < ink_below).mean(axis=0) < 0.002
    runs, start = [], None
    for i, b in enumerate(list(blank) + [False]):
        if b and start is None:
            start = i
        elif not b and start is not None:
            runs.append((start, i))
            start = None
    if not runs:
        return w // 2
    longest = max(e - s for s, e in runs)
    good = [(s, e) for s, e in runs if (e - s) >= 0.6 * longest]
    s, e = min(good, key=lambda r: abs(lo + (r[0] + r[1]) // 2 - w // 2))
    return lo + (s + e) // 2


def is_spread(image: Image.Image, opts: ImportOptions) -> bool:
    return image.width / max(1, image.height) >= opts.spread_ratio


def prepare_page(image: Image.Image, opts: ImportOptions | None = None) -> list[Image.Image]:
    """Return the pages contained in one captured image, in reading order."""
    opts = opts or ImportOptions()
    img = strip_system_bars(image) if opts.strip_bars else image.convert("RGB")
    if opts.split_spreads and is_spread(img, opts):
        x = find_gutter(img)
        left, right = img.crop((0, 0, x, img.height)), img.crop((x, 0, img.width, img.height))
        return [right, left] if opts.right_to_left else [left, right]
    return [img]


def needs_processing(image: Image.Image, opts: ImportOptions) -> bool:
    if opts.split_spreads and is_spread(image, opts):
        return True
    if opts.strip_bars:
        return strip_system_bars(image).size != image.size
    return False
