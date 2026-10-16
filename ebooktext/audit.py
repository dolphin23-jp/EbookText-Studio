"""OCR coverage audit: find text that is visible in a page image but missing from the OCR result.

OCR engines sometimes drop a whole line, or a chunk in the middle of a line, and
the loss differs from run to run. The audit does not read the text. It only looks
at where ink is in the image and where the OCR boxes are, and reports ink that no
OCR box covers. Everything runs locally on the captured images.

Limits: horizontal text only; a figure that contains small labels can give false
alarms, so a report is a "please look here" hint, never a verdict.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

EDGE_THRESHOLD = 48        # summed |dx|+|dy| (0..510) above which a pixel counts as ink
RULE_WINDOW = 100          # a column that is inked in >= RULE_DENSITY of any such window is a
RULE_DENSITY = 0.7         # vertical rule (solid or dashed), not text
MIN_LINE_H = 6             # shorter ink bands are rules / noise
MAX_LINE_H_FACTOR = 3.0    # taller bands (relative to the median line) are figures
MIN_LINE_H_FACTOR = 0.4
EDGE_ZONE = 0.12           # top/bottom share of the image where page-boundary bars and footers sit
BAR_WIDTH = 0.9            # ...and a finding this wide there is a bar or footer, not body text


SPARSE_RATIO = 1.5         # box width per character, in line heights; full-width text is ~1.0
SPARSE_MIN_CHARS = 4

@dataclass
class Finding:
    kind: str     # "line" = nothing read on this line, "partial" = part of the line missing,
                  # "sparse" = the line box is much wider than the text read inside it
    y0: int
    y1: int
    x0: int
    x1: int


@dataclass
class PageAudit:
    name: str
    lines: int = 0                   # text-like lines found in the image
    findings: list[Finding] = field(default_factory=list)
    skipped: str = ""                # reason the page was not audited

    @property
    def ok(self) -> bool:
        return not self.findings


def _runs(flags: np.ndarray, merge_gap: int = 0) -> list[tuple[int, int]]:
    """[start, end) runs of True; runs separated by <= merge_gap False values are joined."""
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    out: list[list[int]] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev - 1 > merge_gap:
            out.append([start, prev + 1])
            start = i
        prev = i
    out.append([start, prev + 1])
    return [(a, b) for a, b in out]


def ink_mask(image: Image.Image) -> np.ndarray:
    """Boolean mask of pixels on a visible edge (dark-on-light and light-on-dark alike)."""
    g = np.asarray(image.convert("L"), dtype=np.int16)
    edge = np.zeros(g.shape, dtype=np.int16)
    edge[:, 1:] += np.abs(np.diff(g, axis=1))
    edge[1:, :] += np.abs(np.diff(g, axis=0))
    return _drop_vertical_rules(edge > EDGE_THRESHOLD)


def _drop_vertical_rules(mask: np.ndarray) -> np.ndarray:
    """Clear vertical rules (frame borders, side bars, dashed or solid): they would glue
    neighbouring text lines into one tall block and hide the line structure."""
    h = mask.shape[0]
    w = min(RULE_WINDOW, h)
    cs = np.vstack([np.zeros((1, mask.shape[1]), dtype=np.int32), np.cumsum(mask, axis=0, dtype=np.int32)])
    dense = (cs[w:] - cs[:-w]) >= RULE_DENSITY * w          # (h-w+1, W): window starting at row i
    if not dense.any():
        return mask
    flags = np.vstack([np.zeros((1, mask.shape[1]), dtype=np.int32), np.cumsum(dense, axis=0, dtype=np.int32)])
    out = mask.copy()
    for y in range(h):
        lo, hi = max(0, y - w + 1), min(y, h - w)           # window starts that contain row y
        if hi >= lo:
            out[y] &= (flags[hi + 1] - flags[lo]) == 0
    return out


def text_bands(mask: np.ndarray) -> list[tuple[int, int]]:
    """Row ranges that look like single text lines."""
    width = mask.shape[1]
    row_ink = mask.sum(axis=1)
    need = max(2, int(width * 0.002))
    bands = [(a, b) for a, b in _runs(row_ink >= need, merge_gap=2) if b - a >= MIN_LINE_H]
    heights = sorted(b - a for a, b in bands if b - a <= 80)
    if not heights:
        return []
    med = heights[len(heights) // 2]
    return [(a, b) for a, b in bands
            if MIN_LINE_H_FACTOR * med <= b - a <= MAX_LINE_H_FACTOR * med]


def _rect_of(item: dict) -> tuple[float, float, float, float] | None:
    quad = item.get("boundingBox")
    if not isinstance(quad, list) or len(quad) < 4:
        return None
    try:
        xs = [float(p[0]) for p in quad]
        ys = [float(p[1]) for p in quad]
    except (TypeError, ValueError, IndexError):
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _box_rects(boxes: list[dict]) -> list[tuple[float, float, float, float]]:
    rects = []
    for item in boxes:
        rect = _rect_of(item) if str(item.get("text", "")).strip() else None
        if rect:
            rects.append(rect)
    return rects


def _sparse_boxes(boxes: list[dict]) -> list[Finding]:
    """A box that spans far more width than its text needs: the engine kept the line box but
    lost characters inside it, which no ink-versus-box comparison can see."""
    out = []
    for item in boxes:
        text = str(item.get("text", "")).strip()
        rect = _rect_of(item)
        if rect is None or len(text) < SPARSE_MIN_CHARS:
            continue
        left, top, right, bottom = rect
        h, w = bottom - top, right - left
        if h <= 0 or w < h:
            continue                      # vertical or degenerate box
        if w / len(text) / h > SPARSE_RATIO:
            out.append(Finding("sparse", int(top), int(bottom), int(left), int(right)))
    return out


def audit_image(image: Image.Image, boxes: list[dict], name: str = "") -> PageAudit:
    """Compare the ink in `image` with the OCR line `boxes` (original image pixels)."""
    result = PageAudit(name=name)
    rects = _box_rects(boxes)
    if not rects:
        result.skipped = "OCRの位置情報がありません"
        return result
    result.findings += _sparse_boxes(boxes)
    mask = ink_mask(image)
    height, width = mask.shape
    bands = text_bands(mask)
    result.lines = len(bands)
    for y0, y1 in bands:
        h = y1 - y0
        # Ink columns of this line, joined into runs (word gaps are smaller than 1.2 line heights).
        segments = [(a, b) for a, b in _runs(mask[y0:y1].any(axis=0), merge_gap=int(h * 1.2))
                    if b - a >= 1.5 * h]
        if not segments:
            continue
        covering = []
        for left, top, right, bottom in rects:
            overlap = min(y1, bottom) - max(y0, top)
            if overlap >= 0.4 * min(h, max(bottom - top, 1)):
                covering.append((left - 0.5 * h, right + 0.5 * h))
        for a, b in segments:
            # Subtract covered intervals from the segment; what is left is unread ink.
            pieces = [(float(a), float(b))]
            for cl, cr in covering:
                nxt = []
                for pa, pb in pieces:
                    if cr <= pa or cl >= pb:
                        nxt.append((pa, pb))
                        continue
                    if cl > pa:
                        nxt.append((pa, cl))
                    if cr < pb:
                        nxt.append((cr, pb))
                pieces = nxt
            for pa, pb in pieces:
                if pb - pa < max(2.5 * h, 40):
                    continue
                # Text fills most of the line height; a rule, dashed underline or bar edge
                # only touches a couple of rows, so it is not unread text.
                if mask[y0:y1, int(pa):int(pb)].any(axis=1).sum() < 0.5 * h:
                    continue
                in_edge_zone = y1 <= height * EDGE_ZONE or y0 >= height * (1 - EDGE_ZONE)
                if in_edge_zone and pb - pa >= BAR_WIDTH * width:
                    continue
                result.findings.append(Finding("partial" if covering else "line",
                                               int(y0), int(y1), int(pa), int(pb)))
    return result


def audit_book(images: list[Path], boxes_by_name: dict[str, list[dict]]) -> list[PageAudit]:
    out = []
    for path in images:
        try:
            with Image.open(path) as im:
                out.append(audit_image(im.convert("RGB"), boxes_by_name.get(path.name, []), path.name))
        except OSError as exc:
            out.append(PageAudit(name=path.name, skipped=f"画像を開けません: {exc}"))
    return out


def annotate(image: Image.Image, audit: PageAudit) -> Image.Image:
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    for f in audit.findings:
        draw.rectangle([f.x0, f.y0, f.x1, f.y1], outline=(255, 0, 0), width=3)
    return img


def render_report(title: str, audits: list[PageAudit], repairs: list | None = None) -> str:
    flagged = [a for a in audits if a.findings]
    skipped = [a for a in audits if a.skipped]
    lines = [f"# OCR抜け確認: {title}", "",
             "画像に文字（インク）があるのに、OCRの読み取り枠が無い箇所を探した結果です。"
             "図表内の文字などで誤検出することがあるため、赤枠付き画像（output/audit/）で目視確認してください。", "",
             f"- 確認した画像: {len(audits) - len(skipped)}枚 / 疑い箇所のある画像: {len(flagged)}枚"]
    if skipped:
        lines.append(f"- 位置情報が無く確認できなかった画像: {len(skipped)}枚")
    lines.append("")
    if repairs:
        done = [r for r in repairs if r.done]
        lines += [f"## 自動で補った箇所（{len(done)}件）／補えなかった箇所（{len(repairs) - len(done)}件）", "",
                  "疑い箇所だけを切り出して読み直し、本文に入れたものです。内容は必ず画像と見比べてください。", ""]
        for r in repairs:
            snippet = r.text if len(r.text) <= 40 else r.text[:40] + "…"
            if r.done:
                extra = f"（元: {r.old}）" if r.old else ""
                lines.append(f"- {r.name} 上から{r.y0}px: {r.action} 「{snippet}」{extra}")
            else:
                lines.append(f"- {r.name} 上から{r.y0}px: 未反映（{r.reason}）")
        lines.append("")
        if flagged:
            lines += ["## 補修後もまだ疑いが残る箇所", ""]
    for a in flagged:
        lines.append(f"## {a.name}（画像内の行 約{a.lines}行）")
        for f in a.findings:
            what = {"line": "行ごと未認識", "partial": "行の途中が未認識",
                    "sparse": "行の途中の文字が欠けた疑い（枠の幅に対し読めた文字数が少ない）"}[f.kind]
            lines.append(f"- {what}: 上から{f.y0}〜{f.y1}px、左から{f.x0}〜{f.x1}px")
        lines.append("")
    if not flagged:
        lines.append("疑い箇所は見つかりませんでした。")
    return "\n".join(lines) + "\n"


def write_audit(book: Path, title: str, images: list[Path], audits: list[PageAudit],
                repairs: list | None = None) -> Path:
    """Write output/OCR抜け確認.md, output/ocr_audit.json and red-boxed copies in output/audit/."""
    import json
    output = Path(book) / "output"
    audit_dir = output / "audit"
    if audit_dir.is_dir():
        for old in audit_dir.glob("*.png"):
            old.unlink()
    by_name = {p.name: p for p in images}
    for a in audits:
        if not a.findings:
            continue
        audit_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(by_name[a.name]) as im:
            annotate(im, a).save(audit_dir / (Path(a.name).stem + "_audit.png"))
    (output / "ocr_audit.json").write_text(
        json.dumps([asdict(a) for a in audits], ensure_ascii=False, indent=1), encoding="utf-8")
    report = output / "OCR抜け確認.md"
    report.write_text(render_report(title, audits, repairs), encoding="utf-8")
    if repairs:
        (output / "ocr_repair.json").write_text(
            json.dumps([asdict(r) for r in repairs], ensure_ascii=False, indent=1), encoding="utf-8")
    return report
