"""Pure helpers and project management; safe to import on machines without a display."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageEnhance, ImageOps

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def safe_name(name: str) -> str:
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    clean = re.sub(r"\s+", " ", clean)
    return clean[:100] or "名称未設定"


def project_dir(base: Path, title: str) -> Path:
    return Path(base).expanduser() / safe_name(title)


def page_images(folder: Path) -> list[Path]:
    folder = Path(folder)
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda p: p.name.lower()) if folder.is_dir() else []


def save_manifest(book: Path, updates: dict[str, Any]) -> None:
    path = Path(book) / "manifest.json"
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
    old.update(updates)
    old["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(path, old)


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def looks_black(image: Image.Image) -> bool:
    g = image.convert("L").resize((64, 64))
    # A near-black picture or screenshot is not automatically a protected page:
    # we only flag it so the user can inspect what is being captured.
    return sum(g.histogram()[:12]) / (64 * 64) > 0.985


def nearly_identical(a: Image.Image, b: Image.Image, threshold: float = 1.2) -> bool:
    def normalized(im: Image.Image) -> Image.Image:
        return im.convert("L").resize((128, 128))
    left, right = normalized(a), normalized(b)
    diff = ImageChops.difference(left, right)
    return sum(i * count for i, count in enumerate(diff.histogram())) / (128 * 128) < threshold


def auto_trim_box(image: Image.Image, tolerance: int = 18, margin: int = 8) -> tuple[int, int, int, int] | None:
    """Bounding box of the content inside uniform white margins, or None when trimming is unsafe."""
    rgb = image.convert("RGB")
    # White-page trim must not consume significant image contents.
    small = rgb.copy()
    small.thumbnail((1500, 1500))
    inverted = ImageChops.difference(small, Image.new("RGB", small.size, "white")).convert("L")
    mask = inverted.point(lambda v: 255 if v > tolerance else 0)
    bbox = mask.getbbox()
    if not bbox:
        return None
    sx, sy = rgb.width / small.width, rgb.height / small.height
    left = max(0, int(bbox[0] * sx) - margin)
    top = max(0, int(bbox[1] * sy) - margin)
    right = min(rgb.width, int(bbox[2] * sx) + margin)
    bottom = min(rgb.height, int(bbox[3] * sy) + margin)
    # Suspicious crop of a page: do not trim automatically.
    if (right - left) < rgb.width * 0.55 or (bottom - top) < rgb.height * 0.55:
        return None
    return left, top, right, bottom


def auto_trim_white(image: Image.Image, tolerance: int = 18, margin: int = 8) -> Image.Image:
    """Trim uniform white margins only; keep pages unchanged if trim is ambiguous."""
    box = auto_trim_box(image, tolerance, margin)
    rgb = image.convert("RGB")
    return rgb.crop(box) if box else rgb


def prepare_ocr_ex(image: Image.Image, upscale: bool = True, autocontrast: bool = True,
                   trim: bool = False) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """prepare_ocr plus the transform back to the input image: x_in = x_out / sx + dx, y_in = y_out / sy + dy."""
    out = image.convert("RGB")
    dx = dy = 0.0
    sx = sy = 1.0
    if trim:
        box = auto_trim_box(out)
        if box:
            out = out.crop(box)
            dx, dy = float(box[0]), float(box[1])
    if upscale and min(out.size) < 1300:
        factor = min(2.0, 1300 / min(out.size))
        if factor > 1.05:
            w0, h0 = out.size
            out = out.resize((round(w0 * factor), round(h0 * factor)), Image.Resampling.LANCZOS)
            sx, sy = out.width / w0, out.height / h0
    if autocontrast:
        gray = ImageOps.autocontrast(out.convert("L"), cutoff=0.3)
        out = Image.merge("RGB", (gray, gray, gray))
    return out, (sx, sy, dx, dy)


def prepare_ocr(image: Image.Image, upscale: bool = True, autocontrast: bool = True,
                trim: bool = False) -> Image.Image:
    return prepare_ocr_ex(image, upscale, autocontrast, trim)[0]


def replacements_load(path: Path) -> dict[str, str]:
    if not path.exists():
        path.write_text("{}\n", encoding="utf-8")
        # No substitutions are performed until the user explicitly edits this file.
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in data.items()):
        raise ValueError("replacements.json は文字列:文字列のJSON辞書にしてください。")
    return {k: v for k, v in data.items() if k and not k.startswith("_")}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def render_markdown(title: str, page_texts: list[tuple[str, str]],
                    notes: dict[str, list[str]] | None = None) -> str:
    parts = [f"# {title}", "", "<!-- 自動OCRです。医療数値・薬剤名・図表は原本と照合してください。 -->"]
    for filename, body in page_texts:
        extra = "".join(f", {n}" for n in (notes or {}).get(filename, []))
        parts += ["", f"<!-- source: {filename}{extra} -->", "", normalize_text(body), ""]
    return "\n".join(parts).rstrip() + "\n"
