"""Page layout model for the structured Markdown: lines, NDLOCR-Lite layout blocks, ruled lines,
page separators and graphics.

Everything here works on the page image and the OCR line boxes that the pipeline already has.
NDLOCR-Lite also writes an XML file per page with its layout analysis (表組, 図版, 柱, ノンブル,
タイトル本文, キャプション and a reading order). When that XML is available it is used as a strong
hint; without it (Tesseract, or an old cache) the same steps run on image analysis alone.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------------------------
# Lines


@dataclass
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    kind: str = "本文"          # NDLOCR-Lite LINE TYPE (本文 / タイトル本文 / キャプション / 頭注 ...)
    repaired: bool = False      # added by the OCR-gap repair (re-read of a small crop)
    order: float | None = None  # NDLOCR-Lite reading order
    block: int | None = None    # NDLOCR-Lite TEXTBLOCK index (lines of one paragraph block)

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def vertical(self) -> bool:
        return self.h > 1.8 * self.w and len(self.text) > 1

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


Rect = tuple[float, float, float, float]


def rect_of_box(item: dict) -> Rect | None:
    coord = item.get("boundingBox")
    if not isinstance(coord, list) or len(coord) < 2:
        return None
    try:
        xs = [float(p[0]) for p in coord]
        ys = [float(p[1]) for p in coord]
    except (TypeError, ValueError, IndexError):
        return None
    if max(xs) <= min(xs) or max(ys) <= min(ys):
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def lines_from_boxes(boxes: list[dict]) -> list[Line]:
    out = []
    for item in boxes:
        text = str(item.get("text", "")).strip()
        r = rect_of_box(item)
        if text and r:
            out.append(Line(text, *r, repaired=bool(item.get("repaired"))))
    return out


def area(r: Rect) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def intersect(a: Rect, b: Rect) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: Rect, b: Rect) -> float:
    i = intersect(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def inside(r: Rect, outer: Rect, frac: float = 0.6) -> bool:
    """At least `frac` of r lies in outer."""
    a = area(r)
    return a > 0 and intersect(r, outer) / a >= frac


def center_in(r: Rect, outer: Rect, pad: float = 0) -> bool:
    cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
    return outer[0] - pad <= cx <= outer[2] + pad and outer[1] - pad <= cy <= outer[3] + pad


def union(rects: list[Rect]) -> Rect:
    return (min(r[0] for r in rects), min(r[1] for r in rects),
            max(r[2] for r in rects), max(r[3] for r in rects))


# ---------------------------------------------------------------------------------------------
# NDLOCR-Lite XML


@dataclass
class NdlBlock:
    kind: str     # 表組 / 図版 / 組織図 / 柱 / ノンブル / 数式 / ...
    rect: Rect


@dataclass
class NdlPage:
    lines: list[Line] = field(default_factory=list)
    blocks: list[NdlBlock] = field(default_factory=list)


def _float(elem, name: str, default: float = 0.0) -> float:
    try:
        return float(elem.get(name))
    except (TypeError, ValueError):
        return default


def _elem_rect(elem) -> Rect:
    x, y = _float(elem, "X"), _float(elem, "Y")
    return (x, y, x + _float(elem, "WIDTH"), y + _float(elem, "HEIGHT"))


def parse_ndl_xml(text: str) -> NdlPage:
    """Lines (with TYPE, ORDER and TEXTBLOCK) and BLOCKs from one NDLOCR-Lite page XML."""
    page = NdlPage()
    if not text:
        return page
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        # The tool writes bare '&' in rare cases; be tolerant.
        try:
            root = ET.fromstring(re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)", "&amp;", text))
        except ET.ParseError:
            return page
    tb_index = 0
    parent: dict[int, int] = {}
    for tb in root.iter("TEXTBLOCK"):
        for line in tb.iter("LINE"):
            parent[id(line)] = tb_index
        tb_index += 1
    for line in root.iter("LINE"):
        s = (line.get("STRING") or "").strip()
        r = _elem_rect(line)
        if not s or area(r) <= 0:
            continue
        order = line.get("ORDER")
        try:
            order_f = float(order) if order is not None else None
        except ValueError:
            order_f = None
        page.lines.append(Line(s, *r, kind=(line.get("TYPE") or "本文").strip(), order=order_f,
                               block=parent.get(id(line))))
    for block in root.iter("BLOCK"):
        r = _elem_rect(block)
        if area(r) > 0:
            page.blocks.append(NdlBlock((block.get("TYPE") or "").strip(), r))
    return page


def attach_ndl(lines: list[Line], ndl: NdlPage) -> None:
    """Copy kind / order / block from the matching NDL line (by position) onto each line.

    The pipeline's boxes are the source of truth for the text (they include repairs and replacement
    edits); the XML only contributes the layout labels."""
    if not ndl.lines:
        return
    for line in lines:
        best, score = None, 0.3
        for cand in ndl.lines:
            s = iou(line.rect, cand.rect)
            if s > score:
                best, score = cand, s
        if best is not None:
            line.kind, line.order, line.block = best.kind, best.order, best.block


FIGURE_BLOCKS = ("図版", "組織図", "グラフ", "写真")
TABLE_BLOCKS = ("表組",)
NOISE_BLOCKS = ("柱", "ノンブル")

# ---------------------------------------------------------------------------------------------
# Image helpers


def gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.int16)


def _runs_1d(flags: np.ndarray, max_gap: int, min_len: int) -> list[tuple[int, int]]:
    """[start, end) runs of True in a 1-D array, bridging gaps of <= max_gap False values."""
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > max_gap + 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]])) + 1
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_len]


@dataclass
class Seg:
    """A ruled line. Horizontal: pos = y, a..b = x range. Vertical: pos = x, a..b = y range."""
    pos: float
    a: float
    b: float
    thick: int = 1

    @property
    def length(self) -> float:
        return self.b - self.a


CONTRAST = 35        # a rule pixel is this much darker than the pixels a few px to either side
RULE_MAX_THICK = 4   # thicker dark bands are fills / separators, not rules


def _line_pixels(g: np.ndarray, axis: int, d: int = 3) -> np.ndarray:
    """Pixels darker than both neighbours at distance d across the line direction.

    axis=0 finds horizontal lines (compares with rows above/below)."""
    if axis == 0:
        before = np.empty_like(g)
        after = np.empty_like(g)
        before[d:] = g[:-d]
        before[:d] = 255
        after[:-d] = g[d:]
        after[-d:] = 255
    else:
        before = np.empty_like(g)
        after = np.empty_like(g)
        before[:, d:] = g[:, :-d]
        before[:, :d] = 255
        after[:, :-d] = g[:, d:]
        after[:, -d:] = 255
    return (np.minimum(before, after) - g > CONTRAST) & (g < 225)


def _merge_segments(raw: list[tuple[int, int, int]], tol: int = 1) -> list[Seg]:
    """raw = (pos, a, b) single-pixel-thick pieces; merge neighbouring positions with overlap."""
    raw.sort()
    segs: list[Seg] = []
    for pos, a, b in raw:
        for s in reversed(segs[-12:]):
            last = s.pos + s.thick - 1
            if pos - last <= tol and min(b, s.b) - max(a, s.a) > 0.5 * min(b - a, s.length):
                s.a, s.b = min(s.a, a), max(s.b, b)
                s.thick = pos - int(s.pos) + 1
                break
        else:
            segs.append(Seg(pos, a, b, 1))
    out = []
    for s in segs:
        if s.thick <= RULE_MAX_THICK:
            s.pos = s.pos + (s.thick - 1) / 2
            out.append(s)
    return out


def detect_rules(g: np.ndarray, min_len: int | None = None) -> tuple[list[Seg], list[Seg]]:
    """Horizontal and vertical ruled lines (table borders, box frames).

    Lines that cross nearly the whole image (screen seams, viewer guides) are dropped: a printed
    rule never runs into both page margins."""
    H, W = g.shape
    if min_len is None:
        min_len = max(30, int(0.04 * W))
    hp = _line_pixels(g, 0)
    vp = _line_pixels(g, 1)
    # Screen seams: thin lines across the whole capture, including the empty left margin. Text
    # lying on a seam breaks it into pieces that would otherwise look like short rules.
    zone = max(8, int(0.04 * W))
    seam = (hp[:, :zone].mean(axis=1) > 0.7) & (hp.mean(axis=1) > 0.45)
    for y in np.flatnonzero(seam):
        # Remove only the seam's own grey: a printed rule that happens to lie on the same row
        # is darker and survives.
        level = int(np.median(g[y, :zone]))
        for yy in range(max(0, y - 1), min(H, y + 2)):
            hp[yy, np.abs(g[yy] - level) <= 30] = False
    raw_h = []
    for y in range(H):
        for a, b in _runs_1d(hp[y], 2, min_len):
            raw_h.append((y, a, b))
    raw_v = []
    vmin = max(18, min_len // 2)
    for x in range(W):
        for a, b in _runs_1d(vp[:, x], 2, vmin):
            raw_v.append((x, a, b))
    edge = 4
    # Printed rules stay inside the page margins. Lines touching the left/right image edge are
    # screen seams, the page edge, or index tabs in the margin.
    hs = [s for s in _merge_segments(raw_h)
          if s.length < 0.94 * W and s.a > edge and s.b < W - edge]
    vs = [s for s in _merge_segments(raw_v)
          if edge < s.pos < W - edge]
    return hs, vs


def page_separators(g: np.ndarray, min_rows: int = 5, dark: int = 90, cover: float = 0.85) -> list[tuple[int, int]]:
    """Dark full-width bands that a scroll viewer draws between book pages: [y0, y1) ranges."""
    rows = (g < dark).mean(axis=1) >= cover
    return _runs_1d(rows, 1, min_rows)


def background_level(g: np.ndarray) -> int:
    hist = np.bincount(np.clip(g, 0, 255).ravel(), minlength=256)
    return int(np.argmax(hist))


def graphics_mask(image: Image.Image, text_rects: list[Rect], scale: int = 4,
                  tol: int = 12) -> tuple[np.ndarray, int]:
    """Downsampled mask of 'drawn' pixels (fills, frames, arrows, photos) with text lines erased.

    Returns (mask, scale). A cell is set when enough of its pixels differ from the page white."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    H, W = rgb.shape[:2]
    bg = np.array([np.bincount(rgb[..., c].ravel(), minlength=256).argmax() for c in range(3)])
    diff = np.abs(rgb - bg).max(axis=2) > tol
    for r in text_rects:
        x0, y0, x1, y1 = (int(max(0, r[0] - 1)), int(max(0, r[1] - 1)),
                          int(min(W, r[2] + 1)), int(min(H, r[3] + 1)))
        diff[y0:y1, x0:x1] = False
    h2, w2 = H // scale, W // scale
    small = diff[:h2 * scale, :w2 * scale].reshape(h2, scale, w2, scale).mean(axis=(1, 3)) > 0.12
    return small, scale


def components(mask: np.ndarray, dilate: int = 2) -> list[tuple[int, int, int, int, int]]:
    """Connected components of a small boolean mask (after a box dilation): (x0, y0, x1, y1, n)."""
    m = mask.copy()
    if dilate:
        p = np.pad(m, dilate)
        acc = np.zeros_like(m)
        for dy in range(2 * dilate + 1):
            for dx in range(2 * dilate + 1):
                acc |= p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
        m = acc
    H, W = m.shape
    seen = np.zeros_like(m)
    out = []
    for y0 in range(H):
        row = np.flatnonzero(m[y0] & ~seen[y0])
        for x0 in row:
            if seen[y0, x0]:
                continue
            stack = [(y0, x0)]
            seen[y0, x0] = True
            xs0 = xs1 = x0
            ys0 = ys1 = y0
            n = 0
            while stack:
                y, x = stack.pop()
                n += 1
                xs0, xs1, ys0, ys1 = min(xs0, x), max(xs1, x), min(ys0, y), max(ys1, y)
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < H and 0 <= xx < W and m[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            out.append((int(xs0), int(ys0), int(xs1) + 1, int(ys1) + 1, n))
    return out


def banded_components(mask: np.ndarray, dilate: int = 2, gap: int = 2) -> list[tuple[int, int, int, int, int]]:
    """components() computed separately inside horizontal bands of the mask that are separated by
    blank rows, so that a dilation never glues a drawing to the tinted box below it."""
    rows = mask.any(axis=1)
    out = []
    for a, b in _runs_1d(rows, gap, 1):
        for x0, y0, x1, y1, n in components(mask[a:b], dilate):
            out.append((x0, y0 + a, x1, min(y1 + a, b), n))
    return out
