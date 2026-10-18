"""Second opinion on short numbers (table cells, scores) with Tesseract.

NDLOCR-Lite reads running Japanese text very well but is unreliable on a lone short number in a
table cell: on real medical tables it returned "00" for 8, "11" for 1 or 12, "21" for -2 and
"14" for 24. Tesseract reading the same spot as one line (psm 7) gets these right with high
confidence. So every short line that is mostly digits is cropped, enlarged and read again; the
Tesseract reading replaces the original only when it is a clean number with high confidence.

Every replacement is recorded for the review list. Cases that stay doubtful (e.g. a trailing
"≦" that Tesseract sees as "=") are flagged instead of guessed.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from .layout import Line, Rect

# A line worth checking: short, contains a digit, and has no kana/kanji besides a unit-ish tail.
CANDIDATE = re.compile(r"^[\s\d.,+\-−–~～〜<>≦≧≤≥=%％/()（）〈〉oOlI|]{1,8}$")
CLEAN = re.compile(r"^[<>≦≧≤≥]?[+\-−]?\d+(?:[.,]\d+)?(?:\s?[~～〜\-−]\s?\d+(?:[.,]\d+)?)?%?$")
MIN_CONF = 80.0
TARGET_H = 48


@dataclass
class NumCheck:
    page: str
    old: str
    new: str
    conf: float
    action: str        # 置換 / 要確認
    note: str = ""


def candidates(lines: list[Line]) -> list[Line]:
    return [l for l in lines if CANDIDATE.match(l.text.strip()) and any(c.isdigit() for c in l.text)]


def crop_line(image: Image.Image, line: Line, clip: Rect | None = None) -> Image.Image:
    h = max(line.h, 4.0)
    x0, x1 = line.x0 - 0.6 * h, line.x1 + 0.6 * h
    y0, y1 = line.y0 - 0.25 * h, line.y1 + 0.25 * h
    if clip is not None:
        x0, y0 = max(x0, clip[0] + 3), max(y0, clip[1] + 2)
        x1, y1 = min(x1, clip[2] - 2), min(y1, clip[3] - 1)
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(image.width, int(x1)), min(image.height, int(y1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        x0, y0, x1, y1 = (int(line.x0), int(line.y0), int(line.x1), int(line.y1))
    crop = image.crop((x0, y0, x1, y1)).convert("L")
    crop = _erase_rules(crop)
    crop = ImageOps.autocontrast(crop, cutoff=1)
    pad = 24
    canvas = Image.new("L", (crop.width + 2 * pad, crop.height + 2 * pad), 255)
    canvas.paste(crop, (pad, pad))
    scale = max(1.0, TARGET_H / max(crop.height, 1))
    if scale > 1.05:
        canvas = canvas.resize((round(canvas.width * scale), round(canvas.height * scale)),
                               Image.Resampling.LANCZOS)
    return canvas


def _erase_rules(crop: Image.Image) -> Image.Image:
    """Paint table borders that fall inside the crop with the background: Tesseract reads a
    vertical rule as "|" or "1"."""
    a = np.asarray(crop, dtype=np.int16).copy()
    if a.size == 0:
        return crop
    bg = int(np.median(a))
    dark = a < bg - 35
    cols = dark.mean(axis=0) > 0.7
    rows = dark.mean(axis=1) > 0.8
    a[:, cols] = bg
    for y in np.flatnonzero(rows):
        # A rule or screen seam running through the digits: remove only its own grey, keep the
        # (darker) strokes of the characters it crosses.
        level = int(np.median(a[y]))
        row = a[y]
        row[np.abs(row - level) <= 25] = bg
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def tesseract_available() -> bool:
    try:
        import pytesseract
        from .ocr import _find_tesseract, _use_user_tessdata
        _use_user_tessdata()
        pytesseract.pytesseract.tesseract_cmd = _find_tesseract("")
        return "eng" in set(pytesseract.get_languages(config=""))
    except Exception:
        return False


def read_number(image: Image.Image) -> tuple[str, float]:
    import pytesseract
    data = pytesseract.image_to_data(image, lang="eng", config="--psm 7",
                                     output_type=pytesseract.Output.DICT)
    words, confs = [], []
    for t, c in zip(data["text"], data["conf"]):
        try:
            c = float(c)
        except (TypeError, ValueError):
            continue
        if str(t).strip() and c >= 0:
            words.append(str(t).strip())
            confs.append(c)
    if not words:
        return "", 0.0
    return "".join(words), min(confs)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def decide(old: str, new: str, conf: float) -> tuple[str, str]:
    """(action, value): 置換 with the new value, 要確認 keeping the old one, or '' (agree/ignore).

    The replacement is conservative: the Tesseract reading must be a clean number with high
    confidence and must not add or drop a decimal point (4.5 is never turned into 45)."""
    o = old.strip()
    new = new.strip().strip("|[]!")
    n = new.replace("−", "-").replace(",", ".") if re.fullmatch(r"\d+,\d+", new.strip()) else new.strip().replace("−", "-")
    if not n or n == o:
        return "", o
    if re.fullmatch(r"[A-Za-z]{2,4}", n) and conf >= 90:
        return "置換", n            # a word such as "or" that NDLOCR-Lite read as digits
    if not any(c.isdigit() for c in n):
        return "", o
    if n.endswith("=") and CLEAN.match(n[:-1]):
        # A trailing ≦/≧ that Tesseract sees as "=" and NDLOCR-Lite as an extra digit ("2≦" -> "25").
        base = n[:-1]
        if conf >= MIN_CONF and o.startswith(base) and len(_digits(o)) in (len(_digits(base)), len(_digits(base)) + 1):
            return "要確認", base + "≦"
        return "要確認", o
    same_points = o.count(".") == n.count(".")
    gained_point = n.count(".") > o.count(".") and _digits(n) == _digits(o) and conf >= 90
    if conf >= MIN_CONF and CLEAN.match(n) and (same_points or gained_point):
        if o[:1] in "<>≦≧〈〉" and n[:1] not in "<>≦≧":
            n = {"〈": "<", "〉": ">"}.get(o[0], o[0]) + n
        return "置換", n
    if _digits(n) != _digits(o) and conf >= 60:
        return "要確認", o
    return "", o


def check_lines(image: Image.Image, lines: list[Line], page: str, clips: dict[int, Rect] | None = None,
                workers: int = 0) -> list[NumCheck]:
    """Re-read candidate lines in place (line.text is replaced); returns the record."""
    todo = candidates(lines)
    if not todo:
        return []
    crops = [crop_line(image, l, (clips or {}).get(id(l))) for l in todo]
    n = workers or max(1, min(os.cpu_count() or 1, 8))
    saved = os.environ.get("OMP_THREAD_LIMIT")
    os.environ["OMP_THREAD_LIMIT"] = "1"
    try:
        with ThreadPoolExecutor(max_workers=n) as ex:
            reads = list(ex.map(read_number, crops))
    finally:
        if saved is None:
            os.environ.pop("OMP_THREAD_LIMIT", None)
        else:
            os.environ["OMP_THREAD_LIMIT"] = saved
    out = []
    for line, (new, conf) in zip(todo, reads):
        action, value = decide(line.text, new, conf)
        if not action:
            continue
        note = ""
        if new.strip().endswith("="):
            note = "末尾の≦/≧が読めていない可能性" + ("（≦にしました。≧の可能性あり）" if value.endswith("≦") else "")
        elif action == "置換" and value.count(".") > line.text.count("."):
            note = "小数点を補いました"
        out.append(NumCheck(page, line.text, value if value != line.text else new, conf, action, note))
        if value != line.text:
            line.text = value
    return out
