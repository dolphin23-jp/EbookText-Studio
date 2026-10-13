"""Post-OCR text cleanup (running headers/footers, page numbers) and review flags."""
from __future__ import annotations

import re
from collections import Counter

# Lines that are clearly not body text. Kept conservative: only patterns seen in real scans.
_STATUS_BAR = re.compile(r"^\s*\d{1,2}:\d{2}\s+\d{1,2}月\d{1,2}日\s*[（(][日月火水木金土][）)]")
_COPYRIGHT = re.compile(r"JCOPY|^\s*[©Ⓒ]\s*\S+", re.I)
_PAGE_NO = re.compile(r"^\s*[-—ー]?\s*(\d{1,4})\s*[-—ー]?\s*$")


def _shape(line: str) -> str:
    """Line with digits and spaces removed, to find headers that differ only by page/section number."""
    return re.sub(r"[\d\s０-９]+", "", line)


def find_running_lines(pages: list[str], edge: int = 2, min_pages: int = 3, min_ratio: float = 0.3) -> set[str]:
    """Lines (as _shape) that repeat near the top/bottom of many pages: 柱, 版権表記 etc."""
    counts: Counter[str] = Counter()
    for text in pages:
        lines = [l for l in text.split("\n") if l.strip()]
        seen = set()
        for l in lines[:edge] + lines[-edge:]:
            shape = _shape(l)
            # Headers/footers rarely end like a sentence; skipping those protects repeated body lines.
            if len(shape) >= 4 and not shape.rstrip().endswith(("。", ".", "．", "、", ",", "，")):
                seen.add(shape)
        counts.update(seen)
    need = max(min_pages, int(len(pages) * min_ratio + 0.999))
    return {s for s, c in counts.items() if c >= need}


def clean_page(text: str, running: set[str]) -> tuple[str, list[str]]:
    """Return (body text, notes). Removes status bar / copyright / running header lines and
    lifts a lone page number out of the body."""
    notes: list[str] = []
    lines = text.split("\n")
    kept: list[str] = []
    for l in lines:
        if _STATUS_BAR.match(l) or _COPYRIGHT.search(l):
            continue
        if l.strip() and len(_shape(l)) >= 4 and _shape(l) in running:
            continue
        kept.append(l)
    # Lone page numbers at the top or bottom (ignore blank lines).
    idx = [i for i, l in enumerate(kept) if l.strip()]
    for i in (idx[-1:] + idx[:1]) if idx else []:
        m = _PAGE_NO.match(kept[i])
        if m:
            notes.append(f"page: {m.group(1)}")
            kept[i] = ""
            break
    return "\n".join(kept).strip("\n"), notes


def clean_pages(texts: list[str]) -> tuple[list[str], list[list[str]], set[str]]:
    running = find_running_lines(texts) if len(texts) >= 3 else set()
    out, notes = [], []
    for t in texts:
        body, n = clean_page(t, running)
        out.append(body)
        notes.append(n)
    return out, notes, running


# ---- review flags -------------------------------------------------------------------------
_UNIT = r"(?:%|％|mg|μg|µg|g|kg|mL|ml|mℓ|dL|L|mmHg|mEq|mmol|mol|U|IU|ng|pg|回|日|週|ヵ月|か月|カ月|ヶ月|時間|分|秒|歳|点|mm|cm|℃|°C|/μL|/µL|/分|/日)"
_NUM_UNIT = re.compile(rf"\d+(?:\.\d+)?\s*{_UNIT}")
_RANGE = re.compile(r"\d\s*[~～〜一ーこ]\s*\d")
# Typical misreads: digit followed by a kana/kanji that stands in for 〜 or a letter in a unit.
_SUSPECT = re.compile(r"\d[こ一ー]\d|\dm6|\d[Oo]\d|[A-Za-z]\d[ぁ-ん]")


def review_reasons(line: str) -> list[str]:
    reasons = []
    if _SUSPECT.search(line):
        reasons.append("誤読の疑い")
    if _RANGE.search(line):
        reasons.append("範囲")
    if _NUM_UNIT.search(line):
        reasons.append("数値・単位")
    return reasons


def review_lines(pages: list[tuple[str, str]]) -> list[tuple[str, int, str, list[str]]]:
    """(page filename, line number, text, reasons) for lines that carry medical numbers."""
    found = []
    for name, text in pages:
        for no, line in enumerate(text.split("\n"), 1):
            reasons = review_reasons(line)
            if reasons:
                found.append((name, no, line.strip(), reasons))
    return found


def render_review(title: str, found: list[tuple[str, int, str, list[str]]]) -> str:
    out = [f"# {title} 要確認リスト", "",
           "数値・単位・範囲を含む行です。OCRの数字や「〜」の誤読が起きやすいため、原本と照合してください。",
           "「誤読の疑い」は `60こ70%` のように、数字の間の記号が文字に化けた可能性がある行です。", ""]
    last = None
    for name, no, text, reasons in found:
        if name != last:
            out += ["", f"## {name}", ""]
            last = name
        out.append(f"- L{no} [{'・'.join(reasons)}] {text}")
    return "\n".join(out).rstrip() + "\n"
