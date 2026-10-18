"""Repair OCR gaps found by the audit: re-read only the suspicious spots and splice the result in.

For every audit finding a small crop is cut out of the page image, enlarged, and read again
with the same engine. The result is merged into the page text next to the OCR line it
belongs to, and a matching box is added so the searchable PDF stays consistent.

The merge is a best effort: it needs the surrounding OCR line to be found again in the page
text. When it cannot be placed with confidence, nothing is changed and the spot is listed as
"未反映" for a manual look. Every change is recorded in a Repair so the report can show it.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .audit import Finding, PageAudit, _rect_of

TARGET_LINE_H = 40         # crops are enlarged until a text line is about this tall
MAX_SCALE = 4.0
MAX_REGIONS = 200          # re-reading is slow; an absurd number of findings is a layout problem

Reader = Callable[[list[Image.Image]], list[str]]


@dataclass
class Repair:
    name: str
    kind: str       # finding kind
    action: str     # 行末に追記 / 行頭に追記 / 行として挿入 / 置換 / 未反映
    y0: int
    text: str       # text that was read from the crop
    old: str = ""   # text that was replaced (sparse)
    reason: str = ""

    @property
    def done(self) -> bool:
        return self.action != "未反映"


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _ascii_edge(a: str, b: str) -> bool:
    return bool(a) and bool(b) and a[-1].isascii() and a[-1].isalnum() and b[0].isascii() and b[0].isalnum()


def _dup(new: str, *neighbours: str) -> bool:
    """The re-read only repeats what the neighbouring line already says (the crop overlapped it)."""
    n = _norm(new)
    return len(n) >= 3 and any(n in _norm(line) for line in neighbours)


def _glue(a: str, b: str) -> str:
    return f"{a} {b}" if _ascii_edge(a, b) else a + b


def _clean_read(text: str) -> str:
    parts = [p.strip() for p in text.splitlines() if p.strip()]
    out = ""
    for p in parts:
        out = _glue(out, p) if out else p
    return out


def _acceptable(text: str) -> bool:
    if not any(c.isalnum() for c in text):
        return False
    return len(text) >= 2 or ord(text[0]) > 0x2E7F           # a lone Latin letter is noise, a lone kanji is fine


def make_crop(image: Image.Image, f: Finding) -> Image.Image:
    """The region of a finding, padded with its own background colour and enlarged for reading."""
    W, H = image.size
    h = max(f.y1 - f.y0, 1)
    px, py = int(0.3 * h) + 2, 4
    box = (max(0, f.x0 - px), max(0, f.y0 - py), min(W, f.x1 + px), min(H, f.y1 + py))
    crop = image.convert("RGB").crop(box)
    bg = tuple(int(v) for v in np.median(np.asarray(crop).reshape(-1, 3), axis=0))
    m = max(24, h)
    canvas = Image.new("RGB", (crop.width + 2 * m, crop.height + 2 * m), bg)
    canvas.paste(crop, (m, m))
    scale = min(MAX_SCALE, max(1.0, TARGET_LINE_H / h))
    if scale > 1.05:
        canvas = canvas.resize((round(canvas.width * scale), round(canvas.height * scale)),
                               Image.Resampling.LANCZOS)
    return canvas


def _find_line(lines: list[str], text: str) -> int | None:
    want = _norm(text)
    if not want:
        return None
    for i, line in enumerate(lines):
        if _norm(line) == want:
            return i
    for i, line in enumerate(lines):
        if want in _norm(line):
            return i
    return None


def _quad(f: Finding) -> list[list[float]]:
    return [[f.x0, f.y0], [f.x1, f.y0], [f.x1, f.y1], [f.x0, f.y1]]


def _row_key(rect: tuple[float, float, float, float], h: float) -> tuple[int, float]:
    return (int(round((rect[1] + rect[3]) / 2 / max(0.5 * h, 1))), rect[0])


def _apply_gap(f: Finding, new: str, lines: list[str], boxes: list[dict], name: str) -> Repair:
    h = max(f.y1 - f.y0, 1)
    cx = (f.x0 + f.x1) / 2
    rep = Repair(name, f.kind, "未反映", f.y0, new)
    anchor_text = None
    anchor_rect = None
    placed = False
    if f.kind == "partial":
        best = None
        for item in boxes:
            r = _rect_of(item)
            t = str(item.get("text", "")).strip()
            if r is None or not t:
                continue
            ov = min(f.y1, r[3]) - max(f.y0, r[1])
            if ov < 0.4 * min(h, max(r[3] - r[1], 1)):
                continue
            gap = max(r[0] - f.x1, f.x0 - r[2], 0)
            if best is None or gap < best[0]:
                best = (gap, t, r)
        if best:
            idx = _find_line(lines, best[1])
            if idx is not None and _dup(new, lines[idx]):
                rep.reason = "同じ文字列が隣の行に既にあります"
                return rep
            if idx is not None:
                if (best[2][0] + best[2][2]) / 2 <= cx:
                    lines[idx] = _glue(lines[idx].rstrip(), new)
                    rep.action = "行末に追記"
                else:
                    lines[idx] = _glue(new, lines[idx].lstrip())
                    rep.action = "行頭に追記"
                placed = True
    if not placed:
        key = (int(round((f.y0 + f.y1) / 2 / max(0.5 * h, 1))), f.x0)
        for item in boxes:
            r = _rect_of(item)
            t = str(item.get("text", "")).strip()
            if r is None or not t or _row_key(r, h) >= key:
                continue
            if anchor_rect is None or _row_key(r, h) > _row_key(anchor_rect, h):
                anchor_text, anchor_rect = t, r
        if anchor_text is None:
            lines.insert(0, new)
            rep.action = "行として挿入"
            placed = True
        else:
            idx = _find_line(lines, anchor_text)
            if idx is None:
                rep.reason = "前後のOCR行が本文中に見つかりません"
            elif _dup(new, *lines[idx:idx + 2]):
                rep.reason = "同じ文字列が隣の行に既にあります"
                return rep
            else:
                lines.insert(idx + 1, new)
                rep.action = "行として挿入"
                placed = True
    if placed:
        boxes.append({"text": new, "boundingBox": _quad(f), "repaired": True})
    return rep


def _apply_sparse(f: Finding, new: str, lines: list[str], boxes: list[dict], name: str) -> Repair:
    rep = Repair(name, f.kind, "未反映", f.y0, new)
    target = None
    for item in boxes:
        r = _rect_of(item)
        if r and (int(r[0]), int(r[1]), int(r[2]), int(r[3])) == (f.x0, f.y0, f.x1, f.y1):
            target = item
            break
    if target is None:
        rep.reason = "対象の枠が見つかりません"
        return rep
    old = str(target.get("text", "")).strip()
    rep.old = old
    if len(_norm(new)) < len(_norm(old)) + 2:
        rep.reason = "再読みしても文字数が増えませんでした"
        return rep
    idx = _find_line(lines, old)
    if idx is None:
        rep.reason = "元のOCR行が本文中に見つかりません"
        return rep
    if _norm(lines[idx]) == _norm(old):
        lines[idx] = new
    else:
        lines[idx] = lines[idx].replace(old, new, 1)
    target["text"] = new
    rep.action = "置換"
    return rep


def repair_book(images: list[Path], audits: list[PageAudit], texts: dict[str, str],
                boxes_by_name: dict[str, list[dict]], reader: Reader,
                stop: threading.Event | None = None) -> list[Repair]:
    """Re-read every finding and merge the result into `texts` and `boxes_by_name` in place."""
    by_name = {p.name: p for p in images}
    jobs: list[tuple[str, Finding]] = []
    for a in audits:
        for f in sorted(a.findings, key=lambda f: (f.y0, f.x0)):
            jobs.append((a.name, f))
    jobs = jobs[:MAX_REGIONS]
    if not jobs:
        return []
    crops = []
    opened: dict[str, Image.Image] = {}
    for name, f in jobs:
        if name not in opened:
            with Image.open(by_name[name]) as im:
                opened[name] = im.convert("RGB")
        crops.append(make_crop(opened[name], f))
    reads = [_clean_read(t) for t in reader(crops)]
    repairs: list[Repair] = []
    state: dict[str, list[str]] = {}
    for (name, f), new in zip(jobs, reads):
        if stop is not None and stop.is_set():
            raise InterruptedError("OCRの補修を停止しました。")
        lines = state.setdefault(name, texts[name].split("\n"))
        boxes = boxes_by_name.setdefault(name, [])
        if not _acceptable(new):
            repairs.append(Repair(name, f.kind, "未反映", f.y0, new, reason="再読みで文字を読めませんでした"))
            continue
        if f.kind == "sparse":
            repairs.append(_apply_sparse(f, new, lines, boxes, name))
        else:
            repairs.append(_apply_gap(f, new, lines, boxes, name))
    for name, lines in state.items():
        texts[name] = "\n".join(lines)
    return repairs
