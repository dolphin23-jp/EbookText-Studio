"""Structured Markdown: headings, paragraphs, lists, real tables and figures, in reading order.

Per page the OCR lines are sorted into
  * tables  — rebuilt from the ruled lines (see tables.py), the text put into their cells;
  * figures — drawn regions (diagrams, charts, photos) found from a caption badge, NDLOCR-Lite's
              图版 block or the image itself; the region is saved as an image and its labels are
              kept apart from the body text, grouped by the box they sit in;
  * noise   — running heads, page numbers (kept as page labels), index tabs in the margin;
  * body    — joined into paragraphs, headings and list items.
Pages are then put together into one book; a paragraph or table cut by a page edge is joined.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from . import tables as T
from .layout import (FIGURE_BLOCKS, NOISE_BLOCKS, Line, NdlPage, Rect, banded_components, center_in, components,
                     detect_rules, graphics_mask, gray, inside, intersect, page_separators, union)
from .textclean import _shape

# ---------------------------------------------------------------------------------------------
# Model


@dataclass
class Figure:
    rect: Rect
    caption: str = ""
    groups: list[str] = field(default_factory=list)
    image_name: str = ""
    page: str = ""
    cap_y: float | None = None


@dataclass
class Block:
    kind: str                     # heading / para / item / table / figure / caption / pagebreak
    y0: float
    y1: float
    text: str = ""
    level: int = 0
    table: T.Table | None = None
    figure: Figure | None = None
    left: float = 0.0             # x of the first line
    indent: float = 0.0           # first line's distance from the left edge of the text column
    right_full: bool = False      # the last line ran to the right edge of the text column
    page: str = ""


@dataclass
class PageModel:
    name: str
    size: tuple[int, int]
    blocks: list[Block] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)   # printed page numbers seen on this image


# ---------------------------------------------------------------------------------------------
# Helpers

SENTENCE_END = ("。", "．", ".", "!", "！", "?", "？", "」", "』", ")", "）")
_BULLET_MAIN = re.compile(r"^[◆◇■□●]\s*")
_BULLET_SUB = re.compile(r"^[・•▪◦○◎⊙▶►✓]\s*")
_NUMBERED = re.compile(r"^(?:[①-⑳❶-❿]|\(?\d{1,2}[).）]|（\d{1,2}）)\s*")
_CAPTION_TEXT = re.compile(r"^(?:表|図|Table|Fig\.?|Figure)\s*(?:[0-9０-９]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+|[IVX]+)")
_PAGE_NO = re.compile(r"^[-—ー]?\s*\d{1,4}\s*[-—ー]?$")


def _join(a: str, b: str) -> str:
    return T._join_text(a, b)


def _median(values: list[float], default: float) -> float:
    return float(np.median(values)) if values else default


def has_badge(rgb: np.ndarray, line: Line) -> bool:
    """A filled, saturated box at the start of the line: the 表/図 number badge of a caption."""
    h = max(int(line.h), 4)
    x0, y0 = int(max(0, line.x0 - 0.3 * h)), int(max(0, line.y0))
    x1, y1 = int(min(rgb.shape[1], line.x0 + 2.2 * h)), int(min(rgb.shape[0], line.y1))
    if x1 - x0 < 4 or y1 - y0 < 3:
        return False
    patch = rgb[y0:y1, x0:x1].astype(np.int16)
    sat = patch.max(axis=2) - patch.min(axis=2)
    if float((sat > 90).mean()) <= 0.35:
        return False
    # ...followed by ordinary text on the page background (a heading band stays coloured).
    a0, a1 = int(min(rgb.shape[1], line.x0 + 3 * h)), int(min(rgb.shape[1], line.x0 + 7 * h))
    if a1 - a0 < 4:
        return False
    after = rgb[y0:y1, a0:a1].astype(np.int16)
    sat2 = after.max(axis=2) - after.min(axis=2)
    return float((sat2 > 90).mean()) < 0.12


def read_badge(rgb: np.ndarray, line: Line) -> str | None:
    """Read the white-on-colour number badge at the start of a caption ("表17", "図6") with
    Tesseract. NDLOCR-Lite often misreads it (表7, ■■, (1)); the number matters because the body
    text refers to it."""
    try:
        import pytesseract
    except ImportError:
        return None
    h = max(int(line.h), 4)
    x0, x1 = int(max(0, line.x0 - 1.0 * h)), int(min(rgb.shape[1], line.x0 + 4 * h))
    y0, y1 = int(max(0, line.y0 - 3)), int(min(rgb.shape[0], line.y1 + 3))
    p = rgb[y0:y1, x0:x1].astype(np.int16)
    if p.size == 0:
        return None
    sat = (p.max(axis=2) - p.min(axis=2)) > 90
    cols = np.flatnonzero(sat.mean(axis=0) > 0.12)
    if not len(cols):
        return None
    runs = [[int(cols[0])]]
    for c in cols[1:]:
        if c - runs[-1][-1] <= 6:
            runs[-1].append(int(c))
        else:
            runs.append([int(c)])
    run = max(runs, key=len)
    if len(run) < 0.8 * h:
        return None
    rows = np.flatnonzero(sat[:, run[0]:run[-1] + 1].mean(axis=1) > 0.3)
    if not len(rows):
        return None
    sub = p[rows[0]:rows[-1] + 1, run[0]:run[-1] + 1]
    light = (sub.min(axis=2) > 170) & ((sub.max(axis=2) - sub.min(axis=2)) < 60)
    img = Image.fromarray(np.where(light, 0, 255).astype(np.uint8))
    canvas = Image.new("L", (img.width + 30, img.height + 30), 255)
    canvas.paste(img, (15, 15))
    scale = max(1.0, 60 / canvas.height)
    canvas = canvas.resize((int(canvas.width * scale), int(canvas.height * scale)), Image.Resampling.LANCZOS)
    try:
        text = pytesseract.image_to_string(canvas, lang="jpn", config="--psm 7").strip().replace(" ", "")
    except Exception:
        return None
    m = re.match(r"^(表|図)([0-9]{1,3})$", text)
    return f"{m.group(1)}{m.group(2)}" if m else None


_LABEL_PREFIX = re.compile(r"^(?:[(（]\s*[0-9０-９]{1,3}\s*[)）]|(?:表|図|Table|Fig\.?)\s*[0-9０-９ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVX]{0,3}|[■□]+)\s*")


def relabel_caption(text: str, label: str) -> str:
    rest = text
    for _ in range(2):
        m = _LABEL_PREFIX.match(rest)
        if not m or not m.group(0).strip() or len(m.group(0)) > 7:
            break
        rest = rest[m.end():]
    return f"{label} {rest}".strip()


# ---------------------------------------------------------------------------------------------
# Page analysis


def analyze_page(image: Image.Image, lines: list[Line], name: str, ndl: NdlPage | None = None,
                 running: set[str] | None = None, tesseract: bool = False) -> tuple[PageModel, list[T.Table]]:
    """Split one page image into ordered blocks. Returns the page and its tables (in order)."""
    W, H = image.size
    g = gray(image)
    rgb = np.asarray(image.convert("RGB"))
    model = PageModel(name, (W, H))
    running = running or set()
    blocks_ndl = ndl.blocks if ndl else []
    lines = [l for l in lines if l.text.strip()]

    # --- noise: running heads, page numbers, index tabs -------------------------------------
    seps = page_separators(g)
    wide = [l for l in lines if l.w > 0.45 * W and not l.vertical]
    col_x0 = float(np.percentile([l.x0 for l in wide], 10)) if len(wide) >= 3 else 0.1 * W
    col_x1 = float(np.percentile([l.x1 for l in wide], 90)) if len(wide) >= 3 else 0.9 * W
    body_h = _median([l.h for l in wide if l.kind in ("本文", "")], _median([l.h for l in lines], 18))
    noise_rects = [b.rect for b in blocks_ndl if b.kind in NOISE_BLOCKS]
    hs, vs = detect_rules(g)
    grid_rects = [t.rect for t in T.find_tables(hs, vs)]
    keep: list[Line] = []
    labels: list[tuple[float, str]] = []
    for l in lines:
        text = l.text.strip()
        # NDLOCR-Lite sometimes labels a large column title as 柱; real running heads are small.
        in_noise = any(inside(l.rect, r, 0.5) for r in noise_rects) and (
            l.h <= 1.1 * body_h or _PAGE_NO.match(text) is not None)
        margin = (l.x0 >= max(col_x1 + 0.02 * W, 0.92 * W) or l.x1 <= min(col_x0 - 0.02 * W, 0.08 * W)
                  or (l.vertical and (l.cx > 0.88 * W or l.cx < 0.12 * W)))
        outside = (l.x1 < col_x0 + 0.5 * body_h or l.x0 > col_x1 - 2.5 * body_h) and not any(
            center_in(l.rect, r) for r in grid_rects)
        if _PAGE_NO.match(text) and (in_noise or margin or (outside and _near_separator(l, seps, H))):
            labels.append((l.cy, re.sub(r"\D", "", text)))
            continue
        if in_noise or margin:
            continue
        if len(_shape(text)) >= 4 and _shape(text) in running:
            continue
        keep.append(l)
    lines = keep
    if tesseract:
        # Page numbers that NDLOCR-Lite found (ノンブル) but did not read.
        from .numcheck import read_number
        for b in blocks_ndl:
            if b.kind == "ノンブル" and not any(abs(y - (b.rect[1] + b.rect[3]) / 2) < 10 for y, _ in labels):
                r = b.rect
                crop = image.crop((int(max(0, r[0] - 6)), int(max(0, r[1] - 4)),
                                   int(min(W, r[2] + 6)), int(min(H, r[3] + 4))))
                big = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.LANCZOS)
                text, conf = read_number(big)
                if re.fullmatch(r"\d{1,4}", text) and conf >= 70:
                    labels.append(((r[1] + r[3]) / 2, text))
    model.labels = [lab for _, lab in sorted(labels)]

    # --- tables -------------------------------------------------------------------------------
    tables = [t for t in T.find_tables(hs, vs) if _plausible_table(t, lines)]
    frames = T.frames(hs, vs)

    # --- captions (badge or 表/図 text) --------------------------------------------------------
    captions: list[Line] = []
    for l in lines:
        if l.w > 0.9 * (col_x1 - col_x0) and not _CAPTION_TEXT.match(l.text):
            continue
        if has_badge(rgb, l) or (_CAPTION_TEXT.match(l.text) and (l.kind == "キャプション" or l.x0 < col_x0 + 2 * body_h)):
            captions.append(l)
    # A caption split into several boxes on one row (badge, then title) is one caption.
    for cap in sorted(captions, key=lambda l: l.x0):
        if not cap.text:
            continue
        mates = [l for l in lines if l is not cap and l.text and abs(l.cy - cap.cy) < 0.5 * cap.h
                 and l.x0 > cap.x0 and l.x0 - cap.x1 < 6 * body_h]
        for m in sorted(mates, key=lambda l: l.x0):
            cap.text = cap.text + " " + m.text
            cap.x1 = max(cap.x1, m.x1)
            m.text = ""
    captions = [c for c in captions if c.text]
    lines = [l for l in lines if l.text]
    if tesseract:
        for cap in captions:
            if True:
                label = read_badge(rgb, cap)
                if label:
                    cap.text = relabel_caption(cap.text, label)

    # --- figures ------------------------------------------------------------------------------
    text_rects = [l.rect for l in lines]
    mask, scale = graphics_mask(image, text_rects)
    # Index tabs, the page edge and screen borders live in the margins; they must not glue
    # unrelated drawings together.
    m0 = max(0, int((col_x0 - 0.04 * W) / scale))
    m1 = min(mask.shape[1], int((col_x1 + 0.03 * W) / scale) + 1)
    mask[:, :m0] = False
    mask[:, m1:] = False
    for s0, s1 in seps:
        mask[max(0, int(s0 / scale) - 1): int(s1 / scale) + 2, :] = False
    # Screen seams (thin grey lines across the capture) are not drawings.
    seam_rows = (np.abs(g[:, :max(8, int(0.04 * W))] - 255) > 20).mean(axis=1) > 0.9
    for y in np.flatnonzero(seam_rows):
        if 0 < y < H - 1 and not seam_rows[y - 1] or not seam_rows[min(H - 1, y + 1)]:
            mask[int(y / scale), :] = False
    comps = [(x0 * scale, y0 * scale, x1 * scale, y1 * scale, n * scale * scale)
             for x0, y0, x1, y1, n in banded_components(mask, dilate=2)]
    figures: list[Figure] = []

    def grow(seed: Rect) -> Rect:
        r = seed
        for _ in range(6):
            more = [c[:4] for c in comps if intersect(c[:4], (r[0] - 4, r[1] - 4, r[2] + 4, r[3] + 4)) > 0]
            nr = union([r] + more) if more else r
            if nr == r:
                break
            r = nr
        return r

    def is_table_caption(c: Line) -> bool:
        return c.text.startswith(("表", "Table")) or any(
            0 <= t.rect[1] - c.y1 < 4 * body_h and t.rect[0] - 10 <= c.x0 for t in tables)

    for cap in captions:
        if is_table_caption(cap):
            continue
        band = (col_x0 - 20, cap.y1, col_x1 + 20, cap.y1 + 3 * body_h)
        seeds = [c[:4] for c in comps if intersect(c[:4], band) > 0 and c[4] > 400]
        if not seeds:
            continue
        rect = grow(union(seeds))
        # Panels of one figure are often separated by white gaps: keep taking the next drawn
        # region below as long as no running text line sits in between.
        for _ in range(12):
            nxt = [c[:4] for c in comps if c[4] > 200 and rect[3] < c[1] <= rect[3] + 2.5 * body_h
                   and min(c[2], rect[2]) - max(c[0], rect[0]) > 0]
            if not nxt:
                break
            top = min(c[1] for c in nxt)
            gap = (col_x0, rect[3], col_x1, top)
            if any(l.w > 0.55 * (col_x1 - col_x0) and center_in(l.rect, gap) for l in lines):
                break
            rect = grow(union([rect] + nxt))
        rect = _stop_at(rect, cap, seps, captions)
        if rect[3] - rect[1] < 2 * body_h:
            continue
        figures.append(Figure(rect, caption=_caption_text(cap.text, "図"), page=name, cap_y=cap.y0))
    for b in blocks_ndl:
        if b.kind not in FIGURE_BLOCKS or any(inside(b.rect, f.rect, 0.5) for f in figures):
            continue
        r = b.rect
        x0, y0, x1, y1 = (int(r[0] / scale), int(r[1] / scale), int(r[2] / scale) + 1, int(r[3] / scale) + 1)
        density = float(mask[y0:y1, x0:x1].mean()) if y1 > y0 and x1 > x0 else 0.0
        if density < 0.08:
            continue           # an empty area labelled 図版 (happens next to page breaks)
        grown = grow(r)
        if any(intersect(t.rect, grown) > 0.3 * (t.rect[2] - t.rect[0]) * (t.rect[3] - t.rect[1])
               and intersect(t.rect, r) == 0 for t in tables):
            grown = r          # growing would swallow a separate table
        figures.append(Figure(grown, page=name))
    for c in comps:
        rect, n = c[:4], c[4]
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        if h < 6 * body_h or w < 0.3 * W or n < 0.02 * W * H:
            continue
        if any(_overlap(t.rect, rect) > 0.3 for t in tables):
            continue
        if any(inside(rect, f.rect, 0.5) for f in figures):
            continue
        inner = [l for l in lines if center_in(l.rect, rect)]
        if _text_box(inner, rect, body_h):
            continue
        if inner and sum(l.w > 0.75 * (col_x1 - col_x0) for l in inner) >= 0.5 * len(inner):
            continue           # mostly running text: a tinted text area, not a drawing
        smaller = [f for f in figures if inside(f.rect, rect, 0.5)]
        if smaller:
            for f in smaller:     # a partial detection (NDLOCR-Lite 図版) inside a larger drawing
                f.rect = union([f.rect, rect])
            continue
        figures.append(Figure(rect, page=name))

    # Tables inside a figure are part of the drawing.
    tables = [t for t in tables if not any(inside(t.rect, f.rect, 0.5) for f in figures)]

    # --- table text ---------------------------------------------------------------------------
    used: set[int] = set()
    kept_tables = []
    for t in tables:
        t.page = name
        t.pages = [name]
        cand = [l for l in lines if id(l) not in used]
        got = T.assign_lines(t, cand)
        filled = sum(1 for c in t.cells if c.text.strip())
        heading_box = (t.ncols == 1 and t.nrows <= 2 and 0 < len(got) <= 4
                       and max(l.h for l in got) >= 1.15 * body_h)
        if filled < max(2, 0.4 * len(t.cells)) or heading_box:
            for c in t.cells:
                c.lines, c.text = [], ""
            continue
        T.detect_header(t, g)
        used.update(id(l) for l in got)
        kept_tables.append(t)
    tables = kept_tables

    # --- figure text ----------------------------------------------------------------------------
    for f in figures:
        inner = [l for l in lines if id(l) not in used and center_in(l.rect, f.rect, pad=2)]
        used.update(id(l) for l in inner)
        if tesseract:
            # NDLOCR-Lite does not read text inside the regions it labels as 図版/組織図.
            inner += figure_text(image, f.rect, inner)
        x0, y0 = int(f.rect[0] / scale), int(f.rect[1] / scale)
        x1, y1 = int(f.rect[2] / scale) + 1, int(f.rect[3] / scale) + 1
        boxes = [((a + x0) * scale, (b + y0) * scale, (c + x0) * scale, (d + y0) * scale)
                 for a, b, c, d, n in components(mask[y0:y1, x0:x1], dilate=0) if n >= 30]
        boxes = [bx for bx in boxes if (bx[2] - bx[0]) < 0.95 * (f.rect[2] - f.rect[0])
                 or (bx[3] - bx[1]) < 0.95 * (f.rect[3] - f.rect[1])]
        f.groups = group_labels(inner, [fr for fr in frames if inside(fr, f.rect, 0.9)] + boxes)

    # --- body -----------------------------------------------------------------------------------
    body = [l for l in lines if id(l) not in used]
    blocks: list[Block] = []
    for t in tables:
        cap = _table_caption(t, captions, body, body_h)
        if cap is not None:
            t.caption = _caption_text(cap.text, "表")
            t.caption_shown = True
            body = [l for l in body if l is not cap]
            # The caption keeps its own place: subtitles between it and the grid stay in order.
            blocks.append(Block("caption", cap.y0, cap.y1, text=t.caption, page=name))
        blocks.append(Block("table", t.rect[1], t.rect[3], table=t, page=name))
    for f in figures:
        for cap in captions:
            if cap in body and f.caption and abs(cap.y1 - f.rect[1]) < 4 * body_h and _caption_text(cap.text, "図") == f.caption:
                body = [l for l in body if l is not cap]
        y0 = min(f.rect[1], f.cap_y) if f.cap_y is not None else f.rect[1]
        blocks.append(Block("figure", y0, f.rect[3], figure=f, page=name))
    caption_ids = {id(c) for c in captions}
    blocks += paragraphs(body, col_x0, col_x1, body_h, caption_ids, name)
    for y0, y1 in seps:
        label = ""
        above = [lab for y, lab in labels if y < y0 and y0 - y < 0.25 * H]
        if above:
            label = above[-1]
        blocks.append(Block("pagebreak", y0, y1, text=label, page=name))
    blocks.sort(key=lambda b: (b.y0, b.kind != "pagebreak"))
    model.blocks = blocks
    return model, tables


_FIG_BULLET = re.compile(r"^(?:[@©®◎⊙○●]|e(?= [A-Z]))\s*")


def figure_text(image: Image.Image, rect: Rect, have: list[Line], min_conf: float = 70) -> list[Line]:
    """Labels inside a figure read by Tesseract (sparse text mode), skipping what is already there."""
    try:
        import pytesseract
        from .linereread import _join as join_words, _words
    except ImportError:
        return []
    x0, y0 = int(max(0, rect[0])), int(max(0, rect[1]))
    x1, y1 = int(min(image.width, rect[2])), int(min(image.height, rect[3]))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return []
    crop = image.crop((x0, y0, x1, y1)).convert("RGB")
    crop = crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.LANCZOS)
    try:
        data = pytesseract.image_to_data(crop, lang="jpn+eng", config="--psm 11",
                                         output_type=pytesseract.Output.DICT)
    except Exception:
        return []
    rows: dict[tuple, list[dict]] = {}
    for w in _words(data):
        rows.setdefault(w["key"], []).append(w)
    out = []
    for ws in rows.values():
        conf = sum(w["conf"] * len(w["text"]) for w in ws) / max(1, sum(len(w["text"]) for w in ws))
        text = _FIG_BULLET.sub("・", join_words(ws)).strip()
        if conf < min_conf or len(text) < 2 or not re.search(r"[\w\u3040-\u30ff\u4e00-\u9fff]", text):
            continue
        bx = (x0 + min(w["box"][0] for w in ws) / 2, y0 + min(w["box"][1] for w in ws) / 2,
              x0 + max(w["box"][2] for w in ws) / 2, y0 + max(w["box"][3] for w in ws) / 2)
        if any(center_in(bx, (l.x0 - 4, l.y0 - 4, l.x1 + 4, l.y1 + 4)) or center_in(l.rect, bx) for l in have):
            continue
        out.append(Line(text, *bx))
    return out


def _plausible_table(t: T.Table, lines: list[Line]) -> bool:
    """Most cells of a real table hold text; flowcharts and framed drawings leave many empty."""
    filled = sum(1 for c in t.cells if any(center_in(l.rect, c.rect) for l in lines))
    return filled >= max(2, 0.5 * len(t.cells))


def _overlap(a: Rect, b: Rect) -> float:
    """Intersection over the smaller of the two areas."""
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    m = min(sa, sb)
    return intersect(a, b) / m if m > 0 else 0.0


def _near_separator(l: Line, seps: list[tuple[int, int]], H: int) -> bool:
    for y0, y1 in seps:
        if 0 < y0 - l.y1 < 0.12 * H or 0 < l.y0 - y1 < 0.06 * H:
            return True
    return l.y0 > 0.93 * H or l.y1 < 0.05 * H


def _stop_at(rect: Rect, cap: Line, seps: list[tuple[int, int]], captions: list[Line]) -> Rect:
    """A figure never extends over a page separator or into the next caption."""
    y1 = rect[3]
    for s0, _ in seps:
        if cap.y1 < s0 < y1:
            y1 = s0
    for c in captions:
        if c is not cap and cap.y1 < c.y0 < y1:
            y1 = c.y0 - 2
    return (rect[0], max(rect[1], cap.y1), rect[2], y1)


def _text_box(inner: list[Line], rect: Rect, body_h: float) -> bool:
    """A framed or tinted box of running text (a callout or a heading band), not a figure."""
    if not inner:
        return False
    if len(inner) <= 2 and all(l.h >= 1.15 * body_h for l in inner):
        return True
    widths = [l.w for l in inner]
    return len(inner) >= 3 and float(np.median(widths)) > 0.6 * (rect[2] - rect[0])


def _caption_text(text: str, kind: str) -> str:
    t = text.strip()
    t = re.sub(r"^[■□▪◼︎●◆]+", "", t).strip()
    if not _CAPTION_TEXT.match(t):
        t = f"{kind}　{t}" if t else kind
    return t


def _table_caption(t: T.Table, captions: list[Line], body: list[Line], body_h: float) -> Line | None:
    best = None
    for c in captions:
        if c in body and 0 <= t.rect[1] - c.y1 < 4.5 * body_h and c.x0 >= t.rect[0] - 12:
            if best is None or c.y1 > best.y1:
                best = c
    return best


def group_labels(lines: list[Line], frames: list[Rect]) -> list[str]:
    """Text inside a figure, grouped by the box it sits in (the smallest framed or filled region
    containing the line); lines outside any box are grouped when stacked closely. Groups are
    listed top to bottom, left to right."""
    if not lines:
        return []

    def frame_of(l: Line) -> int:
        best, area_best = -1, float("inf")
        for i, fr in enumerate(frames):
            a = (fr[2] - fr[0]) * (fr[3] - fr[1])
            if center_in(l.rect, fr) and a < area_best:
                best, area_best = i, a
        return best

    groups: dict[object, list[Line]] = {}
    loose: list[Line] = []
    for l in lines:
        f = frame_of(l)
        if f >= 0:
            groups.setdefault(("f", f), []).append(l)
        else:
            loose.append(l)
    # Loose lines: stack lines that overlap horizontally and follow closely.
    clusters: list[list[Line]] = []
    for l in sorted(loose, key=lambda l: (l.y0, l.x0)):
        for c in clusters:
            last = c[-1]
            if 0 <= l.y0 - last.y1 < 0.7 * max(l.h, last.h) and min(l.x1, last.x1) - max(l.x0, last.x0) > 0:
                c.append(l)
                break
        else:
            clusters.append([l])
    for i, c in enumerate(clusters):
        groups[("l", i)] = c
    ordered = sorted(groups.values(), key=lambda g: (round(min(l.y0 for l in g) / 24), min(l.x0 for l in g)))
    return [" / ".join(_group_text(part) for part in _split_columns(g)) for g in ordered]


def _split_columns(group: list[Line], depth: int = 0) -> list[list[Line]]:
    """Split a group at a vertical gutter that no line crosses (a two-column box)."""
    if depth > 1 or len(group) < 4:
        return [group]
    h = float(np.median([l.h for l in group]))
    spans = sorted((l.x0, l.x1) for l in group)
    best, end = None, spans[0][1]
    for a, b in spans[1:]:
        if a - end > 1.0 * h and (best is None or a - end > best[1] - best[0]):
            best = (end, a)
        end = max(end, b)
    if best is None:
        return [group]
    cut = (best[0] + best[1]) / 2
    left = [l for l in group if l.cx < cut]
    right = [l for l in group if l.cx >= cut]
    if len(left) < 2 or len(right) < 2:
        return [group]
    return _split_columns(left, depth + 1) + _split_columns(right, depth + 1)


def _group_text(group: list[Line]) -> str:
    text = ""
    for row in T._rows_of(group):
        piece = " ".join(l.text for l in row)
        if T._BULLET.match(piece):
            piece = "・" + T._BULLET.sub("", piece)
            text = (text + " " + piece).strip()
        else:
            text = _join(text, piece) if text else piece
    return text


# ---------------------------------------------------------------------------------------------
# Paragraphs


def paragraphs(lines: list[Line], col_x0: float, col_x1: float, body_h: float,
               caption_ids: set[int], page: str) -> list[Block]:
    out: list[Block] = []
    rows = T._rows_of(lines)
    cur: Block | None = None
    cur_block: int | None = None
    hang = []
    for i, row in enumerate(rows):
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        hang.append(bool(nxt) and nxt[0].y0 - max(l.y1 for l in row) < 0.9 * body_h
                    and nxt[0].x0 - row[0].x0 > 0.6 * body_h and row[-1].x1 >= col_x1 - 1.8 * body_h)
    for i, row in enumerate(rows):
        first = row[0]
        text = " ".join(l.text for l in row).strip()
        h = max(_median([l.h for l in row], body_h), 1.0)
        right_full = row[-1].x1 >= col_x1 - 1.8 * body_h
        kind, level = "para", 0
        if id(first) in caption_ids:
            kind = "caption"
        elif _is_heading(row, text, body_h, (col_x0, col_x1)):
            kind = "heading"
            level = 2 if h >= 1.45 * body_h else 3
        elif _BULLET_MAIN.match(text) or _NUMBERED.match(text):
            kind, level = "item", 0
        elif _BULLET_SUB.match(text):
            kind, level = "item", 1
        elif hang[i] and cur is not None and cur.kind == "item" and first.x0 > cur.left + 0.5 * body_h:
            # Starts left of its own continuation lines but right of the item above: a sub-item
            # whose bullet the OCR dropped.
            kind, level = "item", 1
        forced = kind == "item" and (hang[i] or _BULLET_MAIN.match(text) or _BULLET_SUB.match(text)
                                     or _NUMBERED.match(text))
        new = True
        if kind in ("para", "item") and cur is not None and cur.kind in ("para", "item") and not forced:
            gap = first.y0 - cur.y1
            same_block = first.block is not None and cur_block == first.block
            indent = first.x0 - cur.left
            if gap < 0.9 * body_h and (cur.right_full or same_block) and not (
                    indent > 0.7 * body_h and cur.text.endswith(SENTENCE_END) and cur.kind == "para"):
                new = False
            elif gap < 0.9 * body_h and cur.kind == "item" and first.x0 > cur.left + 0.5 * body_h:
                new = False   # hanging indent under a bullet
        if not new and cur is not None:
            cur.text = _join(cur.text, text)
            cur.y1 = max(cur.y1, max(l.y1 for l in row))
            cur.right_full = right_full
            cur_block = first.block if first.block is not None else cur_block
            continue
        # A line that starts further left than the next ones and is followed by indented lines is
        # a list item without a visible bullet (NDLOCR-Lite often drops the "・").
        cur = Block(kind, min(l.y0 for l in row), max(l.y1 for l in row), text, level,
                    left=first.x0, indent=first.x0 - col_x0, right_full=right_full, page=page)
        cur_block = first.block
        out.append(cur)
    _hanging_items(out, rows, body_h)
    return out


def _is_heading(row: list[Line], text: str, body_h: float, col: tuple[float, float] | None = None) -> bool:
    if len(text) > 60 or text.endswith(("。", "．", "、", ",", "，", ".")) and not re.fullmatch(r"\d{1,2}\.", text[-3:].strip()[-3:]):
        return False
    h = _median([l.h for l in row], body_h)
    if col is not None and h >= 1.12 * body_h and len(text) <= 40:
        mid = (col[0] + col[1]) / 2
        cx = (row[0].x0 + row[-1].x1) / 2
        if abs(cx - mid) < 0.06 * (col[1] - col[0]) and row[0].x0 > col[0] + 2 * body_h:
            return True      # a centred title (chapter heading, boxed title)
    if any(l.kind == "タイトル本文" for l in row) and h >= 1.1 * body_h and not _BULLET_MAIN.match(text):
        return True
    return h >= 1.3 * body_h and len(text) <= 40


def _hanging_items(blocks: list[Block], rows: list[list[Line]], body_h: float) -> None:
    """Paragraphs whose first line sticks out to the left of the following lines (a bullet the OCR
    dropped) become sub-items."""
    for b in blocks:
        if b.kind != "para":
            continue
        own = [r for r in rows if b.y0 <= r[0].y0 <= b.y1]
        if len(own) >= 2 and own[1][0].x0 - own[0][0].x0 > 0.6 * body_h and not b.text.startswith(("「", "(", "（")):
            b.kind, b.level = "item", 1


# ---------------------------------------------------------------------------------------------
# Book assembly


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith(SENTENCE_END) or text.rstrip().endswith(("：", ":"))


def assemble(pages: list[PageModel], all_tables: list[list[T.Table]]) -> list[Block]:
    """One block list for the book: paragraphs and tables cut by a page edge (or by a page
    separator inside a slice) are joined."""
    out: list[Block] = []
    last_content: Block | None = None
    for page in pages:
        first = True
        for b in page.blocks:
            if b.kind == "pagebreak":
                out.append(b)
                continue
            prev = last_content
            if (b.kind == "table" and not b.table.caption and prev is not None and prev.kind == "caption"
                    and out and out[-1] in (prev,) + tuple(x for x in out[-2:] if x.kind == "pagebreak")
                    and prev.text.startswith(("表", "Table"))):
                # The caption ended one page, the table starts the next.
                b.table.caption = prev.text
                out.remove(prev)
                prev = last_content = None
                for x in reversed(out):
                    if x.kind != "pagebreak":
                        prev = last_content = x
                        break
            if prev is not None and _continues(prev, b, out):
                if b.kind == "table":
                    T.merge_tables(prev.table, b.table)
                    prev.table.images += b.table.images
                    prev.y1 = b.y1
                else:
                    prev.text = _join(prev.text, b.text)
                    prev.right_full = b.right_full
                continue
            out.append(b)
            if first:
                b.page = page.name
            first = False
            last_content = b
    return out


def _continues(prev: Block, cur: Block, out: list[Block]) -> bool:
    between = out[out.index(prev) + 1:] if prev in out else []
    if any(x.kind not in ("pagebreak",) for x in between):
        return False
    crossed = bool(between) or prev.page != cur.page
    if not crossed:
        return False
    if prev.kind == "table" and cur.kind == "table":
        return (not cur.table.caption and T.same_columns(prev.table, cur.table)
                and (prev.table.open_bottom or cur.table.open_top or not cur.table.header_rows
                     or cur.table.header_rows == prev.table.header_rows))
    if prev.kind in ("para", "item") and cur.kind == "para":
        return prev.right_full and not _ends_sentence(prev.text) and cur.indent < 9
    return False


# ---------------------------------------------------------------------------------------------
# Rendering


def render(title: str, blocks: list[Block], figure_dir: str = "figures", table_dir: str = "tables") -> str:
    out = [f"# {title}", "",
           "<!-- 自動OCRによる構造化Markdownです。表・数値・薬剤名は原本（tables/・figures/の画像）と照合してください。 -->", ""]
    last_page = None
    fig_no = tab_no = 0
    for b in blocks:
        if b.page and b.page != last_page and b.kind != "pagebreak":
            out += [f"<!-- source: {b.page} -->", ""]
            last_page = b.page
        if b.kind == "pagebreak":
            out += [f"<!-- p.{b.text} ここまで -->" if b.text else "<!-- 改ページ -->", ""]
        elif b.kind == "heading":
            out += ["#" * b.level + " " + b.text, ""]
        elif b.kind == "caption":
            out += [f"**{b.text}**", ""]
        elif b.kind == "item":
            text = _BULLET_MAIN.sub("", _BULLET_SUB.sub("", b.text)) if not _NUMBERED.match(b.text) else b.text
            out += [("  " * b.level) + "- " + text, ""]
        elif b.kind == "para":
            out += [b.text, ""]
        elif b.kind == "table":
            tab_no += 1
            t = b.table
            if t.caption and not t.caption_shown:
                out += [f"**{t.caption}**", ""]
            out += [T.to_markdown(t), ""]
            if t.image_name:
                links = " ".join(f"[表の原画像{i + 1 if len(t.images) > 1 else ''}]({table_dir}/{n})"
                                 for i, n in enumerate(t.images or [t.image_name]))
                csv_name = Path(t.image_name).with_suffix(".csv").name
                out += [f"<sub>{links} / [CSV]({table_dir}/{csv_name})</sub>", ""]
        elif b.kind == "figure":
            fig_no += 1
            f = b.figure
            first = f.groups[0] if f.groups and len(f.groups[0]) <= 30 else ""
            cap = f.caption or (f"図：{first}" if first else "図")
            out += [f"**{cap}**", ""]
            if f.image_name:
                out += [f"![{cap}]({figure_dir}/{f.image_name})", ""]
            if f.groups:
                out += ["<details><summary>図中の文字</summary>", ""]
                out += [f"- {g}" for g in f.groups]
                out += ["", "</details>", ""]
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse blank lines between consecutive list items for a compact list.
    text = re.sub(r"(^\s*- .+)\n\n(?=\s*- )", r"\1\n", text, flags=re.M)
    return text.rstrip() + "\n"


# ---------------------------------------------------------------------------------------------
# Book driver


Reader = Callable[[list[Image.Image]], list[str]]


@dataclass
class StructureResult:
    markdown: str
    tables: int
    figures: int
    numchecks: list
    fixes: list


def build_book(title: str, images: list[Path], boxes_by_name: dict[str, list[dict]],
               ndl_xml: dict[str, str], output: Path, replacements: dict[str, str] | None = None,
               glossary: list[str] | None = None, running: set[str] | None = None,
               numbers: bool = True, stop: threading.Event | None = None,
               status: Callable[[str], None] = lambda s: None,
               texts: dict[str, str] | None = None) -> StructureResult:
    from .layout import lines_from_boxes, parse_ndl_xml, attach_ndl
    from .textfix import apply_lexicon, build_lexicon, fix_line
    from . import numcheck

    output = Path(output)
    fig_dir = output / "figures"
    tab_dir = output / "tables"
    for d in (fig_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*"):
            if old.is_file():
                old.unlink()
    # 1) lines + text fixes for every page (cheap), then the book lexicon
    page_lines: dict[str, list[Line]] = {}
    ndl_pages: dict[str, NdlPage] = {}
    fixes = []
    for p in images:
        lines = lines_from_boxes(boxes_by_name.get(p.name, []))
        # A gap repair that only produced a few digits or marks is noise from a decorative edge
        # (heading bands, icons); it would be glued to the heading on the same row.
        lines = [l for l in lines if not (l.repaired and _REPAIR_NOISE.match(l.text))]
        ndl = parse_ndl_xml(ndl_xml.get(p.name, ""))
        attach_ndl(lines, ndl)
        for l in lines:
            text = l.text
            for wrong, right in (replacements or {}).items():
                text = text.replace(wrong, right)
            text, fx = fix_line(text, p.name)
            fixes += fx
            l.text = text
        page_lines[p.name] = lines
        ndl_pages[p.name] = ndl
    lexicon = build_lexicon([l.text for ls in page_lines.values() for l in ls], glossary)
    for name, lines in page_lines.items():
        for l in lines:
            l.text, fx = apply_lexicon(l.text, lexicon, name)
            fixes += fx
    use_numbers = numbers and numcheck.tesseract_available()
    if numbers and not use_numbers:
        status("Tesseractが無いため、表の数値の読み直しを省略します")
    # 2) page analysis
    pages: list[PageModel] = []
    tables_by_page: list[list[T.Table]] = []
    checks = []
    for i, p in enumerate(images, 1):
        if stop is not None and stop.is_set():
            raise InterruptedError("構造化を停止しました。")
        with Image.open(p) as im:
            image = im.convert("RGB")
        lines = page_lines[p.name]
        if not lines and (texts or {}).get(p.name, "").strip():
            # No positions (text taken from the PDF itself): plain paragraphs.
            pages.append(_text_only_page(p.name, (texts or {})[p.name], replacements))
            tables_by_page.append([])
            continue
        if use_numbers:
            checks += numcheck.check_lines(image, lines, p.name)
        model, tabs = analyze_page(image, lines, p.name, ndl_pages[p.name], running, tesseract=use_numbers)
        pages.append(model)
        tables_by_page.append(tabs)
        _save_crops(image, model, tabs, fig_dir, tab_dir)
        if i % 10 == 0 or i == len(images):
            status(f"構造化 {i}/{len(images)}ページ")
    blocks = assemble(pages, tables_by_page)
    n_tab = 0
    for b in blocks:
        if b.kind == "table":
            n_tab += 1
            if b.table.image_name:
                (tab_dir / Path(b.table.image_name).with_suffix(".csv").name).write_text(
                    T.to_csv(b.table), encoding="utf-8-sig")
    n_fig = sum(1 for b in blocks if b.kind == "figure")
    return StructureResult(render(title, blocks), n_tab, n_fig, checks, fixes)


_REPAIR_NOISE = re.compile(r"^[\s\d.,:;'\-|lIL{}()\[\]!/\\●○◎・]{1,5}$")


def _text_only_page(name: str, text: str, replacements: dict[str, str] | None) -> PageModel:
    from .textfix import fix_line
    for wrong, right in (replacements or {}).items():
        text = text.replace(wrong, right)
    model = PageModel(name, (0, 0))
    y = 0.0
    for chunk in re.split(r"\n\s*\n", text):
        rows = [r.strip() for r in chunk.split("\n") if r.strip()]
        if not rows:
            continue
        joined = ""
        for r in rows:
            joined = _join(joined, fix_line(r, name)[0]) if joined else fix_line(r, name)[0]
        model.blocks.append(Block("para", y, y + 1, joined, page=name))
        y += 2
    return model


def _save_crops(image: Image.Image, model: PageModel, tabs: list[T.Table], fig_dir: Path, tab_dir: Path) -> None:
    stem = Path(model.name).stem
    W, H = image.size
    k = 0
    for b in model.blocks:
        if b.kind == "figure":
            k += 1
            r = b.figure.rect
            box = (max(0, int(r[0]) - 6), max(0, int(r[1]) - 6), min(W, int(r[2]) + 6), min(H, int(r[3]) + 6))
            name = f"{stem}_図{k}.png"
            image.crop(box).save(fig_dir / name)
            b.figure.image_name = name
    for j, t in enumerate(tabs, 1):
        r = t.rect
        box = (max(0, int(r[0]) - 4), max(0, int(r[1]) - 4), min(W, int(r[2]) + 4), min(H, int(r[3]) + 4))
        name = f"{stem}_表{j}.png"
        image.crop(box).save(tab_dir / name)
        t.image_name = name
        t.images = [name]
