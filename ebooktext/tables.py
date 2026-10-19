"""Tables: rebuild the cell grid from the ruled lines and put the OCR text into the cells.

A ruled table is found as a group of horizontal and vertical rules that cross each other. The
rule positions give the rows and columns; a missing piece of rule between two neighbouring cells
means they are one merged cell (rowspan/colspan). OCR lines are then placed into the cell that
contains their centre. A line that straddles a column border (the engine read across two cells)
is reported so the caller can re-read those cells one by one.

The result renders as a Markdown table (merged cells are repeated in every row/column they cover,
so each row reads on its own), as CSV, and as HTML with rowspan/colspan.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

import numpy as np

from .layout import Line, Rect, Seg, center_in, intersect, rect_of_box

SNAP = 5                 # rule positions closer than this are one grid line
MIN_CELL_W = 18
MIN_CELL_H = 12
COVER = 0.6              # a border exists when a rule covers this share of the cell edge


@dataclass
class Cell:
    r0: int
    c0: int
    r1: int              # inclusive
    c1: int
    rect: Rect
    lines: list[Line] = field(default_factory=list)
    text: str = ""
    reread: bool = False  # text came from a cell-by-cell re-read


@dataclass
class Table:
    rect: Rect
    xs: list[float]
    ys: list[float]
    cells: list[Cell]
    open_top: bool = False      # no rule along the top edge (continues from the previous page)
    open_bottom: bool = False
    header_rows: int = 0
    caption: str = ""
    caption_shown: bool = False  # the caption is rendered as its own block before the table
    page: str = ""              # page image name (first page for merged tables)
    pages: list[str] = field(default_factory=list)
    straddlers: list[Line] = field(default_factory=list)
    image_name: str = ""
    images: list[str] = field(default_factory=list)

    @property
    def nrows(self) -> int:
        return len(self.ys) - 1

    @property
    def ncols(self) -> int:
        return len(self.xs) - 1

    def grid(self) -> list[list[Cell | None]]:
        g: list[list[Cell | None]] = [[None] * self.ncols for _ in range(self.nrows)]
        for c in self.cells:
            for r in range(c.r0, c.r1 + 1):
                for k in range(c.c0, c.c1 + 1):
                    g[r][k] = c
        return g


# ---------------------------------------------------------------------------------------------
# Grid detection


def _snap(values: list[float], tol: float = SNAP) -> list[float]:
    out: list[list[float]] = []
    for v in sorted(values):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [float(np.mean(g)) for g in out]


def _crosses(h: Seg, v: Seg, tol: float = 6) -> bool:
    return h.a - tol <= v.pos <= h.b + tol and v.a - tol <= h.pos <= v.b + tol


def _groups(hs: list[Seg], vs: list[Seg]) -> list[tuple[list[Seg], list[Seg]]]:
    """Connected groups of crossing rules."""
    n = len(hs)
    parent = list(range(n + len(vs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, h in enumerate(hs):
        for j, v in enumerate(vs):
            if _crosses(h, v):
                a, b = find(i), find(n + j)
                if a != b:
                    parent[a] = b
    groups: dict[int, tuple[list[Seg], list[Seg]]] = {}
    for i, h in enumerate(hs):
        groups.setdefault(find(i), ([], []))[0].append(h)
    for j, v in enumerate(vs):
        groups.setdefault(find(n + j), ([], []))[1].append(v)
    return [g for g in groups.values() if g[0] and g[1]]


def _covered(segs: list[Seg], pos: float, a: float, b: float) -> float:
    """Share of [a, b] covered by segments lying at `pos`."""
    span = b - a
    if span <= 0:
        return 1.0
    pieces = sorted((max(a, s.a), min(b, s.b)) for s in segs if abs(s.pos - pos) <= SNAP and s.b > a and s.a < b)
    total, end = 0.0, a
    for lo, hi in pieces:
        lo = max(lo, end)
        if hi > lo:
            total += hi - lo
            end = hi
    return total / span


def _build(hs: list[Seg], vs: list[Seg]) -> Table | None:
    hs = [h for h in hs if h.length >= 40]
    vs = [v for v in vs if v.length >= 12]
    if len(vs) < 2 or not hs:
        return None
    xs = _snap([v.pos for v in vs])
    ys = _snap([h.pos for h in hs])
    # Rows cut off by the image edge: the vertical rules run on past the last horizontal rule.
    vtop, vbot = min(v.a for v in vs), max(v.b for v in vs)
    open_top = vtop < ys[0] - MIN_CELL_H
    open_bottom = vbot > ys[-1] + MIN_CELL_H
    if open_top:
        ys.insert(0, float(vtop))
    if open_bottom:
        ys.append(float(vbot))
    # Drop grid lines that make sliver cells (double rules, antialiasing).
    xs = [x for i, x in enumerate(xs) if i == 0 or x - xs[i - 1] >= MIN_CELL_W] if xs else xs
    ys = [y for i, y in enumerate(ys) if i == 0 or y - ys[i - 1] >= MIN_CELL_H] if ys else ys
    if len(xs) < 2 or len(ys) < 2:
        return None
    nr, nc = len(ys) - 1, len(xs) - 1
    # The outer frame must exist: left and right rules span most of the table height.
    height = ys[-1] - ys[0]
    if _covered(vs, xs[0], ys[0], ys[-1]) < 0.5 or _covered(vs, xs[-1], ys[0], ys[-1]) < 0.5:
        # Tables without side rules (only horizontal rules) are not handled here.
        return None
    parent = list(range(nr * nc))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for r in range(nr):
        for c in range(nc - 1):
            if _covered(vs, xs[c + 1], ys[r], ys[r + 1]) < COVER:
                join(r * nc + c, r * nc + c + 1)
    for r in range(nr - 1):
        for c in range(nc):
            if _covered(hs, ys[r + 1], xs[c], xs[c + 1]) < COVER:
                join(r * nc + c, (r + 1) * nc + c)
    members: dict[int, list[tuple[int, int]]] = {}
    for r in range(nr):
        for c in range(nc):
            members.setdefault(find(r * nc + c), []).append((r, c))
    cells = []
    for pos in members.values():
        r0, r1 = min(p[0] for p in pos), max(p[0] for p in pos)
        c0, c1 = min(p[1] for p in pos), max(p[1] for p in pos)
        cells.append(Cell(r0, c0, r1, c1, (xs[c0], ys[r0], xs[c1 + 1], ys[r1 + 1])))
    # A table needs at least two real cells; a single frame is a box (heading, callout, figure part).
    if len(cells) < 2:
        return None
    if height < 2 * MIN_CELL_H or xs[-1] - xs[0] < 80:
        return None
    cells.sort(key=lambda c: (c.r0, c.c0))
    return Table((xs[0], ys[0], xs[-1], ys[-1]), xs, ys, cells, open_top, open_bottom)


def find_tables(hs: list[Seg], vs: list[Seg]) -> list[Table]:
    out = []
    for gh, gv in _groups(hs, vs):
        t = _build(gh, gv)
        if t is not None:
            out.append(t)
    out.sort(key=lambda t: (t.rect[1], t.rect[0]))
    return out


def frames(hs: list[Seg], vs: list[Seg]) -> list[Rect]:
    """Closed rectangles (single boxes) among the rules: flowchart boxes, callouts, heading frames."""
    out = []
    for gh, gv in _groups(hs, vs):
        xs = _snap([v.pos for v in gv if v.length >= 12])
        ys = _snap([h.pos for h in gh if h.length >= 20])
        if len(xs) >= 2 and len(ys) >= 2 and xs[-1] - xs[0] >= 30 and ys[-1] - ys[0] >= 14:
            out.append((xs[0], ys[0], xs[-1], ys[-1]))
    return out


# ---------------------------------------------------------------------------------------------
# Text in cells

_BULLET = re.compile(r"^[◎⊙○●◉•・▪■□◆◇▶►※＊*]\s*|^[❶-❿①-⑳]\s*")
_CELL_BULLET = re.compile(r"^[◎⊙○◉●❶-❿]\s*")   # NDL reads the round cell-bullet icon as any of these


def _join_text(a: str, b: str) -> str:
    if not a:
        return b
    if a[-1].isascii() and a[-1].isalnum() and b[:1].isascii() and b[:1].isalnum():
        return a + " " + b
    if a.endswith("-") and len(a) > 1 and a[-2].isascii() and a[-2].isalpha() and b[:1].islower():
        return a[:-1] + b
    return a + b


def _rows_of(lines: list[Line]) -> list[list[Line]]:
    rows: list[list[Line]] = []
    for ln in sorted(lines, key=lambda l: (l.cy, l.x0)):
        if rows and abs(ln.cy - float(np.mean([l.cy for l in rows[-1]]))) < 0.5 * max(ln.h, 1):
            rows[-1].append(ln)
        else:
            rows.append([ln])
    return [sorted(r, key=lambda l: l.x0) for r in rows]


def cell_text(lines: list[Line], right: float | None = None) -> str:
    """Lines of one cell, top to bottom, as items separated by <br>.

    A line continues the previous item when the previous line ran up to the right edge of the
    cell (it wrapped) or when it is indented under a bullet (hanging indent). Otherwise it is a
    new item: in tables a short line is usually its own entry (「<30%減少」, 「30～50%減少」)."""
    if not lines:
        return ""
    rows = _rows_of(lines)
    if right is None:
        right = max(l.x1 for l in lines)
    left = min(l.x0 for l in lines)
    items: list[str] = []
    prev: list[Line] | None = None
    for row in rows:
        text = " ".join(l.text for l in row).strip()
        if not text:
            continue
        h = max(float(np.median([l.h for l in row])), 1.0)
        if _BULLET.match(text):
            items.append("・" + _CELL_BULLET.sub("", text) if _CELL_BULLET.match(text) else text)
        elif items and prev is not None and (prev[-1].x1 >= right - 1.6 * h or row[0].x0 > left + 0.6 * h):
            items[-1] = _join_text(items[-1], text)
        else:
            items.append(text)
        prev = row
    return "<br>".join(items)


def _ends_item(text: str) -> bool:
    return text.endswith(("。", "．", ".", ")", "）", "」"))


_SHORT = re.compile(r"^[\s\d.,+\-−–~～〜<>≦≧≤≥%％/()（）点]{1,10}$")


def split_implicit_rows(table: Table) -> None:
    """Split a ruled row into sub-rows when a column of short values (scores, numbers) lines up
    with lines in the other columns but the book drew no rule between them.

    The first column (the row label) and cells spanning several ruled rows stay merged."""
    changed = True
    while changed:
        changed = False
        for r in range(table.nrows):
            row_cells = [c for c in table.cells if c.r0 <= r <= c.r1]
            singles = [c for c in row_cells if c.r0 == c.r1 == r and c.c0 > 0]
            refs = [c for c in singles if len(_rows_of(c.lines)) >= 2
                    and all(_SHORT.match(l.text) for l in c.lines)]
            if not refs:
                continue
            ref = max(refs, key=lambda c: len(_rows_of(c.lines)))
            centers = [float(np.mean([l.cy for l in row])) for row in _rows_of(ref.lines)]
            # The value lines must line up with lines of another column.
            others = [l for c in singles if c is not ref for l in c.lines]
            hits = sum(1 for y in centers if any(abs(l.cy - y) < 0.6 * max(l.h, 1) for l in others))
            if hits < max(2, len(centers) - 1):
                continue
            cuts = [(a + b) / 2 for a, b in zip(centers, centers[1:])]
            _split_row(table, r, cuts)
            changed = True
            break


def _split_row(table: Table, r: int, cuts: list[float]) -> None:
    k = len(cuts)
    ys = table.ys[: r + 1] + cuts + table.ys[r + 1:]
    new_cells: list[Cell] = []
    for c in table.cells:
        if c.r1 < r:
            new_cells.append(c)
        elif c.r0 > r:
            new_cells.append(Cell(c.r0 + k, c.c0, c.r1 + k, c.c1, c.rect, c.lines, c.text, c.reread))
        elif c.r0 == c.r1 == r and c.c0 > 0:
            bounds = [table.ys[r]] + cuts + [table.ys[r + 1]]
            for i in range(k + 1):
                part = [l for l in c.lines if (i == 0 or l.cy >= bounds[i]) and (i == k or l.cy < bounds[i + 1])]
                rect = (c.rect[0], bounds[i], c.rect[2], bounds[i + 1])
                new_cells.append(Cell(r + i, c.c0, r + i, c.c1, rect, part))
        else:
            new_cells.append(Cell(c.r0, c.c0, c.r1 + k, c.c1, c.rect, c.lines, c.text, c.reread))
    table.ys = ys
    table.cells = sorted(new_cells, key=lambda c: (c.r0, c.c0))


def assign_lines(table: Table, lines: list[Line]) -> list[Line]:
    """Put each line inside the table into its cell. Returns the lines that were used."""
    used = []
    for ln in lines:
        if not center_in(ln.rect, table.rect, pad=2):
            continue
        hits = [c for c in table.cells if intersect(ln.rect, c.rect) > 0.2 * max(ln.w * ln.h, 1)]
        best = None
        for c in table.cells:
            if center_in(ln.rect, c.rect):
                best = c
                break
        if best is None and hits:
            best = max(hits, key=lambda c: intersect(ln.rect, c.rect))
        if best is None:
            continue
        if len({(c.c0, c.c1) for c in hits}) > 1 and ln.w > 1.3 * (best.rect[2] - best.rect[0]):
            table.straddlers.append(ln)
        best.lines.append(ln)
        used.append(ln)
    split_implicit_rows(table)
    for c in table.cells:
        c.text = cell_text(c.lines, c.rect[2] - 3)
    return used


def straddled_cells(table: Table) -> list[Cell]:
    """Cells touched by a line that crosses a column border: their text should be re-read."""
    out = []
    for ln in table.straddlers:
        for c in table.cells:
            if intersect(ln.rect, c.rect) > 0.15 * max(ln.w * ln.h, 1) and c not in out:
                out.append(c)
    return out


def detect_header(table: Table, g: np.ndarray) -> None:
    """The first row is a header when its background differs clearly from the body rows."""
    if table.open_top or table.nrows < 2:
        table.header_rows = 0
        return

    def bg(r: int) -> float:
        y0, y1 = int(table.ys[r]) + 2, int(table.ys[r + 1]) - 1
        x0, x1 = int(table.xs[0]) + 2, int(table.xs[-1]) - 1
        if y1 <= y0 or x1 <= x0:
            return 255.0
        return float(np.median(g[y0:y1, x0:x1]))

    first, rest = bg(0), [bg(r) for r in range(1, min(table.nrows, 4))]
    table.header_rows = 1 if rest and abs(first - float(np.median(rest))) > 25 else 0


# ---------------------------------------------------------------------------------------------
# Cross-page continuation


def same_columns(a: Table, b: Table, tol: float = 8) -> bool:
    return a.ncols == b.ncols and all(abs(x - y) <= tol for x, y in zip(a.xs, b.xs))


def merge_tables(a: Table, b: Table) -> None:
    """Append b's rows to a. When a ends in an open row and b starts with one (a row cut by the
    page edge), the two halves are joined cell by cell. A repeated header row is dropped."""
    join_cut = a.open_bottom and b.open_top
    shift = a.nrows - (1 if join_cut else 0)
    last_row = a.grid()[-1] if a.nrows else []
    head_a = [c.text for c in a.cells if c.r0 == 0]
    head_b = [c.text for c in b.cells if c.r0 == 0]
    drop_header = bool(a.header_rows and b.header_rows and not join_cut and head_a == head_b)
    off = shift - (1 if drop_header else 0)
    new_cells = []
    for c in b.cells:
        if drop_header and c.r0 == 0:
            continue
        if join_cut and c.r0 == 0 and c.c0 < len(last_row):
            t = last_row[c.c0]
            if t is not None and t.c0 == c.c0 and t.c1 == c.c1:
                t.text = _merge_text(t.text, c.text)
                t.r1 = max(t.r1, c.r1 + off)
                continue
        new_cells.append(Cell(c.r0 + off, c.c0, c.r1 + off, c.c1, c.rect, c.lines, c.text, c.reread))
    a.cells += new_cells
    a.cells.sort(key=lambda c: (c.r0, c.c0))
    nrows = max(c.r1 for c in a.cells) + 1
    a.ys = [a.ys[0] + i for i in range(nrows + 1)]   # geometry is per page; only the count matters now
    a.open_bottom = b.open_bottom
    a.pages.extend(p for p in b.pages if p not in a.pages)
    a.straddlers += b.straddlers


def _merge_text(a: str, b: str) -> str:
    if not a:
        return b
    if not b:
        return a
    if b.startswith("・"):
        return a + "<br>" + b
    return _join_text(a, b)


# ---------------------------------------------------------------------------------------------
# Rendering


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def to_rows(table: Table, fill: bool = True) -> list[list[str]]:
    """Cell texts as a plain rectangle; merged cells are repeated when fill=True."""
    rows = []
    for r, row in enumerate(table.grid()):
        out = []
        for k, c in enumerate(row):
            if c is None:
                out.append("")
            elif fill or (c.r0 == r and c.c0 == k):
                out.append(c.text)
            else:
                out.append("")
        rows.append(out)
    return rows


def to_markdown(table: Table) -> str:
    rows = to_rows(table, fill=True)
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    if len(rows[0]) == 1:
        # A one-column table is a list of boxed items.
        return "\n".join(f"> {_md_escape(r[0]).replace('<br>', '  ')}" if r[0] else ">" for r in rows)
    if table.header_rows:
        head, body = rows[0], rows[1:]
    else:
        head, body = [""] * ncols, rows
    out = ["| " + " | ".join(_md_escape(c) for c in head) + " |",
           "| " + " | ".join("---" for _ in head) + " |"]
    out += ["| " + " | ".join(_md_escape(c) for c in r) + " |" for r in body]
    return "\n".join(out)


def to_csv(table: Table) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for r in to_rows(table, fill=True):
        w.writerow([c.replace("<br>", "\n") for c in r])
    return buf.getvalue()


def to_html(table: Table) -> str:
    out = ["<table>"]
    grid = table.grid()
    for r, row in enumerate(grid):
        out.append("<tr>")
        for k, c in enumerate(row):
            if c is None:
                out.append("<td></td>")
                continue
            if c.r0 != r or c.c0 != k:
                continue
            tag = "th" if r < table.header_rows else "td"
            span = ""
            if c.r1 > c.r0:
                span += f' rowspan="{c.r1 - c.r0 + 1}"'
            if c.c1 > c.c0:
                span += f' colspan="{c.c1 - c.c0 + 1}"'
            text = c.text.replace("&", "&amp;").replace("<br>", "\x00").replace("<", "&lt;").replace("\x00", "<br>")
            out.append(f"<{tag}{span}>{text}</{tag}>")
        out.append("</tr>")
    out.append("</table>")
    return "\n".join(out)


def boxes_in(boxes: list[dict], rect: Rect) -> list[dict]:
    out = []
    for b in boxes:
        r = rect_of_box(b)
        if r and center_in(r, rect):
            out.append(b)
    return out
