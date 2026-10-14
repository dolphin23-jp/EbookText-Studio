"""Image helpers for continuous-scroll capture (pure numpy/Pillow, no display needed).

Three building blocks:

* ``estimate_shift``   how many pixels did the content move between two screenshots?
* ``detect_static_bands``  rows at the top/bottom that do not move (toolbars, footers)
* ``ScrollStitcher``   append new rows of every screenshot to one tall strip and cut the strip
                       into page-like slices at blank rows, so that text lines and figures are
                       not cut in half.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

INK_THRESHOLD = 28          # |gray - background| above this counts as ink
STATIC_DIFF = 1.5           # mean |a-b| below this: the screen did not move
BLANK_THRESHOLD = 12        # a pixel closer than this to the page background is not ink (cutting)


def to_gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32)


def _reduce_cols(a: np.ndarray, target: int = 192) -> np.ndarray:
    h, w = a.shape
    f = max(1, w // target)
    w2 = (w // f) * f
    return a[:, :w2].reshape(h, w2 // f, f).mean(axis=2)


def _reduce_rows(a: np.ndarray, f: int) -> np.ndarray:
    if f <= 1:
        return a
    h2 = (a.shape[0] // f) * f
    return a[:h2].reshape(h2 // f, f, a.shape[1]).mean(axis=1)


def column_background(gray: np.ndarray, step: int = 2) -> np.ndarray:
    """Most frequent gray level of every column (page colour, dark viewer margins, ...).

    The mode, not the median: on a text-dense page more than half of a column can be ink.
    """
    q = np.clip((gray[::step] // 8).astype(np.int32), 0, 31)
    w = q.shape[1]
    idx = (np.arange(w, dtype=np.int32)[None, :] * 32 + q).ravel()
    counts = np.bincount(idx, minlength=w * 32).reshape(w, 32)
    return counts.argmax(axis=1).astype(np.float32) * 8 + 4


def _ink_rows(a: np.ndarray, bg: np.ndarray, thr: float = INK_THRESHOLD) -> np.ndarray:
    """Rows where some column differs from the per-column background (page colour, dark margins...)."""
    return (np.abs(a - bg[None, :]) > thr).any(axis=1)


@dataclass
class ShiftResult:
    shift: int                 # content moved up by this many pixels (0 = did not move)
    error: float               # weighted mean abs gray difference on rows that contain ink
    second_error: float | None  # best error of a clearly different shift (ambiguity check)
    ink_rows: int              # rows with ink inside the overlap (reliability of the match)
    confident: bool
    static: bool = False       # the two screenshots are the same screen
    blank: bool = False        # overlap has (almost) no ink; shift taken from `expected`
    reason: str = ""


# Rows that have ink on one side only are expected while a page is still being drawn (late
# loading); they count, but less than rows that have ink on both sides.
ONE_SIDED_WEIGHT = 0.15
LOST_WEIGHT = 1.0


def _scan(A: np.ndarray, B: np.ndarray, inkA: np.ndarray, inkB: np.ndarray,
          shifts: list[int], min_ink: int) -> dict[int, float]:
    """Error for each shift: prev row y+s must equal cur row y. Only rows with ink count."""
    n = A.shape[0]
    out: dict[int, float] = {}
    for s in shifts:
        m = n - s
        if m <= 0:
            continue
        ia, ib = inkA[s:], inkB[:m]
        total = int((ia | ib).sum())
        if total < min_ink:
            continue
        both = ia & ib
        nb = int(both.sum())
        if nb < max(min_ink // 2, int(0.5 * int(ia.sum()))):
            continue                      # what the first screen showed is not (mostly) there
        d = np.abs(A[s:] - B[:m]).mean(axis=1)
        lost, late = ia & ~ib, ib & ~ia
        num = d[both].sum() + LOST_WEIGHT * d[lost].sum() + ONE_SIDED_WEIGHT * d[late].sum()
        den = nb + LOST_WEIGHT * int(lost.sum()) + ONE_SIDED_WEIGHT * int(late.sum())
        out[s] = float(num / den)
    return out


def estimate_shift(prev: np.ndarray, cur: np.ndarray, *, expected: float | None = None,
                   min_overlap: float = 0.10, max_error: float = 14.0,
                   ambiguity_margin: float = 1.5, min_ink_rows: int = 12) -> ShiftResult:
    """Vertical shift between two gray windows of the same size (scrolling down = positive).

    ``expected`` (pixels) is the shift the scroll command should produce. It limits the search
    range and breaks ties between equally good periodic matches. When the overlap holds very
    little (one text line, say) a near-exact match close to ``expected`` is still accepted.
    """
    res = _estimate_main(prev, cur, expected, min_overlap, max_error, ambiguity_margin, min_ink_rows)
    if res.confident or not expected or prev.shape != cur.shape:
        return res
    weak = _estimate_weak(prev, cur, expected, min_overlap)
    return weak or res


def _estimate_weak(prev: np.ndarray, cur: np.ndarray, expected: float, min_overlap: float,
                   max_error: float = 4.0, min_ink_rows: int = 4) -> ShiftResult | None:
    hw = prev.shape[0]
    smax = hw - max(8, int(hw * min_overlap))
    w = max(6, int(0.04 * expected))
    lo, hi = max(1, int(expected) - w), min(smax, int(expected) + w)
    if lo > hi:
        return None
    A, B = _reduce_cols(prev), _reduce_cols(cur)
    bg = column_background(np.concatenate([A, B], axis=0), step=1)
    inkA, inkB = _ink_rows(A, bg), _ink_rows(B, bg)
    scan = _scan(A, B, inkA, inkB, list(range(lo, hi + 1)), min_ink_rows)
    if not scan:
        return None
    best_err = min(scan.values())
    if best_err > max_error:
        return None
    near = sorted(s for s, e in scan.items() if e <= best_err + 1.0)
    best = min(near, key=lambda s: abs(s - expected))
    if len(near) > 1 and near[-1] - near[0] > 3 and abs(best - expected) > 3:
        return None                           # repeated pattern and not where the scroll was meant to land
    rows = int((inkA[best:] | inkB[:hw - best]).sum())
    return ShiftResult(best, scan[best], None, rows, True, reason="重なり部分の内容が少ないが一致")


def _estimate_main(prev: np.ndarray, cur: np.ndarray, expected: float | None,
                   min_overlap: float, max_error: float, ambiguity_margin: float,
                   min_ink_rows: int) -> ShiftResult:
    if prev.shape != cur.shape:
        return ShiftResult(0, float("inf"), None, 0, False, reason="画像サイズが一致しません")
    hw = prev.shape[0]
    if float(np.abs(prev - cur).mean()) < STATIC_DIFF:
        return ShiftResult(0, 0.0, None, 0, True, static=True, reason="画面が動いていません")
    min_ov = max(8, int(hw * min_overlap))
    smax = hw - min_ov
    if smax < 2:
        return ShiftResult(0, float("inf"), None, 0, False, reason="領域が小さすぎます")
    hi = smax
    if expected:
        hi = min(smax, int(expected * 1.6) + 8)
    A, B = _reduce_cols(prev), _reduce_cols(cur)
    bg = column_background(np.concatenate([A, B], axis=0), step=1)
    inkA, inkB = _ink_rows(A, bg), _ink_rows(B, bg)

    f = 4 if hw >= 240 else 1
    Ac, Bc = _reduce_rows(A, f), _reduce_rows(B, f)
    inkAc, inkBc = _ink_rows(Ac, bg, INK_THRESHOLD / 2), _ink_rows(Bc, bg, INK_THRESHOLD / 2)
    min_ink_c = max(3, min_ink_rows // f)
    coarse = _scan(Ac, Bc, inkAc, inkBc, list(range(1, hi // f + 1)), min_ink_c)

    def blank_result() -> ShiftResult:
        # No usable ink in the overlap: any alignment only touches blank rows, so the commanded
        # distance is a safe stand-in. Without it we cannot tell.
        if expected and 1 <= int(round(expected)) <= smax:
            return ShiftResult(int(round(expected)), 0.0, None, 0, True, blank=True,
                               reason="重なり部分に文字・図がないため指定量で結合")
        return ShiftResult(0, float("inf"), None, 0, False,
                           reason="重なり部分に十分な内容がなく移動量を判定できません")

    picked: list[int] = []
    for s in sorted(coarse, key=coarse.get):
        if all(abs(s - p) > 2 for p in picked):
            picked.append(s)
        if len(picked) >= 4:
            break
    cand: set[int] = set(range(1, min(hi, 8) + 1))      # tiny shifts (end of the document)
    for sc in picked:
        cand.update(range(max(1, sc * f - f - 1), min(smax, sc * f + f + 1) + 1))
    fine = _scan(A, B, inkA, inkB, sorted(cand), min_ink_rows)
    if not fine:
        return blank_result()
    best_err = min(fine.values())
    near = sorted(s for s, e in fine.items() if e <= best_err + ambiguity_margin)
    clusters: list[list[int]] = []
    for s in near:
        if clusters and s - clusters[-1][-1] <= 3:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    if len(clusters) == 1:
        chosen = clusters[0]
    elif expected:
        chosen = min(clusters, key=lambda c: min(abs(s - expected) for s in c))
    else:
        return ShiftResult(0, best_err, None, 0, False,
                           reason="同じ見た目の候補が複数あり移動量を一つに決められません")
    best = min(chosen, key=lambda s: (fine[s], abs(s - expected) if expected else 0))
    err = fine[best]
    others = [e for s, e in coarse.items() if abs(s * f - best) > 2 * f + 2]
    second = min(others) if others else None
    rows = int((inkA[best:] | inkB[:hw - best]).sum())
    if err > max_error:
        return ShiftResult(best, err, second, rows, False,
                           reason=f"重なり部分の一致度が低い（誤差 {err:.1f}）。読み込み途中か、画面が変化した可能性があります")
    return ShiftResult(best, err, second, rows, True)


def detect_static_bands(prev: Image.Image, cur: Image.Image, scrolled: int = 9,
                        max_frac: float = 0.25, min_px: int = 6) -> tuple[int, int]:
    """Toolbar / footer rows at the top and bottom that stay put while the content scrolled.

    Needs two screenshots taken before/after scrolling by ``scrolled`` (> 8) pixels. A band is a
    run of pixel-identical rows that reaches the edge, trimmed to the last row that differs from
    the page background: blank page margin that merely looks the same in both pictures is not a
    toolbar and must stay in the capture. Capped at ``max_frac`` of the height each.
    """
    a, b = to_gray(prev), to_gray(cur)
    if a.shape != b.shape or scrolled <= 8:
        return 0, 0
    same = np.abs(a - b).max(axis=1) < 2.0
    dev = (np.abs(a - column_background(np.concatenate([a, b], axis=0), step=1)[None, :]) > INK_THRESHOLD).any(axis=1)
    h = len(same)
    limit = int(h * max_frac)

    def band(seq: np.ndarray, ink: np.ndarray) -> int:
        n = 0
        for v in seq[:limit]:
            if not v:
                break
            n += 1
        hits = np.flatnonzero(ink[:n])
        size = int(hits[-1]) + 1 if len(hits) else 0
        return size if size >= min_px else 0

    return band(same, dev), band(same[::-1], dev[::-1])


# ---- stitching and slicing -------------------------------------------------------------------

@dataclass
class Slice:
    image: Image.Image
    forced: bool = False       # no blank row was available; cut at the least inked row
    rows: int = 0


class ScrollStitcher:
    """Tall strip built from scrolling screenshots, emitted as page-like slices.

    Invariant: the last ``window_h`` rows of the strip always equal the latest screenshot, so the
    next screenshot can replace the overlap (later screenshots are the more completely loaded ones)
    and only its new rows are appended. Rows ``[0, emitted)`` were already written out.
    """

    def __init__(self, window_h: int, width: int, slice_height: int = 0,
                 blank_run: int = 3, search_frac: float = 0.3, max_frac: float = 2.0):
        self.window_h = window_h
        self.width = width
        self.slice_height = slice_height or int(min(2400, max(600, width * 1.4)))
        self.blank_run = blank_run
        self.search_frac = search_frac
        self.max_frac = max_frac
        self.strip = np.zeros((0, width, 3), dtype=np.uint8)
        self.emitted = 0
        self.forced_cuts = 0
        self._bg = np.zeros(width, dtype=np.float32)

    # -- building --
    def start(self, window: Image.Image, emitted: bool = False) -> None:
        arr = np.asarray(window.convert("RGB"), dtype=np.uint8)
        self.strip = arr.copy()
        self.emitted = len(arr) if emitted else 0

    def load_pending(self, strip: Image.Image, emitted: int) -> None:
        self.strip = np.asarray(strip.convert("RGB"), dtype=np.uint8).copy()
        self.emitted = max(0, min(int(emitted), len(self.strip)))

    def append(self, window: Image.Image, shift: int) -> None:
        arr = np.asarray(window.convert("RGB"), dtype=np.uint8)
        if shift <= 0:
            overlap = min(len(arr), len(self.strip))
            if overlap > 0 and len(self.strip) - overlap >= self.emitted:
                self.strip[-overlap:] = arr[-overlap:]
            return
        ov = max(0, len(arr) - shift)
        if ov:
            lo = len(self.strip) - ov
            if lo >= self.emitted:
                self.strip[lo:] = arr[:ov]
            else:
                # Part of the overlap was already written out; keep those rows as they are.
                keep = self.emitted - lo
                self.strip[self.emitted:] = arr[keep:ov]
        self.strip = np.concatenate([self.strip, arr[ov:]], axis=0)

    @property
    def pending_rows(self) -> int:
        return len(self.strip) - self.emitted

    def last_window(self) -> Image.Image:
        return Image.fromarray(self.strip[-self.window_h:])

    # -- cutting --
    def _background(self) -> np.ndarray:
        """Per-column background colour of the whole strip."""
        return column_background(self.strip[::2].astype(np.float32).mean(axis=2), step=1)

    def _ink_count(self, rows: np.ndarray, bg: np.ndarray | None = None) -> np.ndarray:
        """Per row: number of pixels that differ from the column's background.

        The right-most columns are ignored because a viewer's scrollbar thumb lives there.
        """
        gray = rows.astype(np.float32).mean(axis=2)
        if bg is None:
            bg = column_background(gray)
        w = gray.shape[1]
        edge = max(8, int(w * 0.012)) if w > 200 else 0
        return (np.abs(gray - bg[None, :]) > BLANK_THRESHOLD)[:, :w - edge].sum(axis=1)

    def _blank_rows(self, rows: np.ndarray) -> np.ndarray:
        return self._ink_count(rows, self._bg) <= max(1, int(self.width * 0.001))

    def _find_cut(self, lo: int, hi: int, target: int, rows: np.ndarray) -> int | None:
        """Middle of the blank run (>= blank_run rows) nearest to ``target`` within [lo, hi)."""
        blank = self._blank_rows(rows[lo:hi])
        best, best_d = None, None
        start = None
        for i, b in enumerate(list(blank) + [False]):
            if b and start is None:
                start = i
            elif not b and start is not None:
                if i - start >= self.blank_run:
                    mid = lo + (start + i) // 2
                    d = abs(mid - target)
                    if best_d is None or d < best_d:
                        best, best_d = mid, d
                start = None
        return best

    def _least_ink_row(self, lo: int, hi: int) -> int:
        return lo + int(np.argmin(self._ink_count(self.strip[lo:hi], self._bg)))

    def pop_slices(self, final: bool = False) -> list[Slice]:
        """Write out every slice that can be cut safely.

        While scrolling continues the last ``window_h`` rows stay mutable (the next screenshot
        may replace them), so slices are only cut above them. ``final`` writes everything.
        """
        out: list[Slice] = []
        self._bg = self._background()
        t = self.slice_height
        tol = int(t * self.search_frac)
        max_h = int(t * self.max_frac)
        while True:
            base = self.emitted
            limit = len(self.strip) - (0 if final else self.window_h)
            avail = limit - base
            if avail <= 0:
                break
            if final and avail <= t + tol:
                out.append(self._take(limit, False))
                break
            if avail < t + tol:
                break
            cut = self._find_cut(base + t - tol, base + t + tol, base + t, self.strip)
            if cut is None:
                # A figure or table may be taller than the page: look further for a gap.
                far = min(limit, base + max_h)
                if far > base + t + tol:
                    cut = self._find_cut(base + t + tol, far, base + t + tol, self.strip)
            forced = False
            if cut is None:
                if avail < max_h:
                    if final:
                        out.append(self._take(limit, False))
                    break
                cut = self._least_ink_row(base + t - tol, base + t + tol)
                forced = True
                self.forced_cuts += 1
            out.append(self._take(cut, forced))
        return out

    def _take(self, cut: int, forced: bool) -> Slice:
        img = Image.fromarray(self.strip[self.emitted:cut])
        n = cut - self.emitted
        self.emitted = cut
        # Drop rows that are no longer needed (everything before the last window).
        drop = min(self.emitted, len(self.strip) - self.window_h)
        if drop > 0:
            self.strip = self.strip[drop:]
            self.emitted -= drop
        return Slice(img, forced, n)

    def hold_back_cut(self) -> int | None:
        """Last safe cut (middle of a blank run) in the pending rows, for stopping without a tail."""
        if self.pending_rows <= 0:
            return None
        self._bg = self._background()
        cut = None
        lo = self.emitted
        rows = self.strip
        blank = self._blank_rows(rows[lo:])
        start = None
        for i, b in enumerate(list(blank) + [False]):
            if b and start is None:
                start = i
            elif not b and start is not None:
                if i - start >= self.blank_run:
                    cut = lo + (start + i) // 2
                start = None
        return cut if cut and cut > lo else None

    def pop_until_hold_back(self) -> list[Slice]:
        """Emit pending rows only up to the last blank gap; the unfinished tail stays pending."""
        out = self.pop_slices(final=False)
        # Rows inside the mutable window are safe to write once no further screenshots follow.
        cut = self.hold_back_cut()
        if cut:
            out.append(self._take(cut, False))
        return out
