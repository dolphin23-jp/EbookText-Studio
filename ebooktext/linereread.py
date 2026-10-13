"""Tesseract page reading with a second pass on low-confidence lines.

Whole-page layout analysis garbles some lines (headings on tinted bars, table cells, short
labels). Re-reading just that line as a single-line image often fixes it, and Tesseract's own
confidence tells us which lines to retry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from PIL import Image, ImageOps


@dataclass
class ReadStats:
    lines: int = 0
    reread: int = 0
    improved: int = 0


def _words(data: dict) -> list[dict]:
    out = []
    for i, text in enumerate(data["text"]):
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if not str(text).strip() or conf < 0:
            continue
        out.append({"text": str(text).strip(), "conf": conf, "key": (data["block_num"][i], data["par_num"][i], data["line_num"][i]),
                    "box": (data["left"][i], data["top"][i], data["left"][i] + data["width"][i], data["top"][i] + data["height"][i])})
    return out


_ASCII_WORD = re.compile(r"[A-Za-z0-9]")


def _needs_space(prev: dict, cur: dict) -> bool:
    return bool(_ASCII_WORD.match(prev["text"][-1]) or _ASCII_WORD.match(cur["text"][0]))


def _line_box(text: str, words: list[dict], dx: float = 0, dy: float = 0) -> dict:
    """One text-layer box per line: searching the PDF for a phrase works only when the phrase sits
    in a single text run, and a run per token splits Japanese words."""
    box = (min(w["box"][0] for w in words), min(w["box"][1] for w in words),
           max(w["box"][2] for w in words), max(w["box"][3] for w in words))
    return {"text": text, "boundingBox": _quad(box, dx, dy)}


def _join(words: list[dict]) -> str:
    """Join word tokens close to Tesseract's text output.

    image_to_data drops Tesseract's word-space flag, and joining with " " would put a space
    between every Japanese token. Measured against image_to_string on real pages, "a space
    wherever one side is an ASCII letter/digit" reproduces it best (97% line-text similarity)."""
    text = words[0]["text"]
    for prev, cur in zip(words, words[1:]):
        if _needs_space(prev, cur):
            text += " "
        text += cur["text"]
    return text


def _line_conf(words: list[dict]) -> float:
    n = sum(len(w["text"]) for w in words)
    return sum(w["conf"] * len(w["text"]) for w in words) / max(n, 1)


def _quad(box: tuple[float, float, float, float], dx: float = 0, dy: float = 0) -> list[list[float]]:
    x0, y0, x1, y1 = box
    return [[x0 + dx, y0 + dy], [x0 + dx, y1 + dy], [x1 + dx, y1 + dy], [x1 + dx, y0 + dy]]


def read_page(image: Image.Image, lang: str, psm: int, reread: bool = True,
              threshold: float = 80, margin: float = 3,
              pad: int = 6) -> tuple[str, ReadStats, list[dict]]:
    """Returns (text, stats, boxes); boxes are {"text", "boundingBox"} per line in image pixels."""
    import pytesseract

    data = pytesseract.image_to_data(image, lang=lang, config=f"--psm {psm}",
                                     output_type=pytesseract.Output.DICT)
    lines: dict[tuple, list[dict]] = {}
    for w in _words(data):
        lines.setdefault(w["key"], []).append(w)
    stats = ReadStats(lines=len(lines))
    out: list[str] = []
    boxes: list[dict] = []
    last_par = None
    for key, ws in lines.items():
        text = _join(ws)
        line_boxes = [_line_box(text, ws)]
        conf = _line_conf(ws)
        if reread and conf < threshold:
            stats.reread += 1
            x0 = max(0, min(w["box"][0] for w in ws) - pad)
            y0 = max(0, min(w["box"][1] for w in ws) - pad)
            x1 = min(image.width, max(w["box"][2] for w in ws) + pad)
            y1 = min(image.height, max(w["box"][3] for w in ws) + pad)
            crop = ImageOps.autocontrast(image.crop((x0, y0, x1, y1)).convert("L"), cutoff=0.5).convert("RGB")
            new = _words(pytesseract.image_to_data(crop, lang=lang, config="--psm 7",
                                                   output_type=pytesseract.Output.DICT))
            if new and _line_conf(new) > conf + margin:
                text = _join(new)
                line_boxes = [_line_box(text, new, x0, y0)]
                stats.improved += 1
        if last_par is not None and key[:2] != last_par:
            out.append("")
        last_par = key[:2]
        out.append(text)
        boxes += line_boxes
    return "\n".join(out), stats, boxes
