"""Import local images or PDFs supplied by the user."""
from __future__ import annotations

import json
from pathlib import Path
from shutil import copy2
from typing import Callable

from .core import page_images, save_manifest


def import_images(book: Path, source: Path, progress: Callable[[str], None]) -> int:
    source = Path(source).resolve()
    dst_dir = Path(book).resolve() / "pages"
    if source == dst_dir:
        raise ValueError("取り込み元はプロジェクトのpages以外のフォルダにしてください。")
    items = page_images(source)
    if not items:
        raise ValueError("画像がありません。PNG/JPG/WEBP/TIFF等に対応します。")
    dst_dir.mkdir(parents=True, exist_ok=True)
    if page_images(dst_dir):
        raise ValueError("プロジェクトに既に画像があります。新しい書名で取り込んでください。")
    for i, path in enumerate(items, 1):
        dst = dst_dir / f"page_{i:05d}{path.suffix.lower()}"
        copy2(path, dst)
        progress(f"画像取り込み: {i}/{len(items)}")
    save_manifest(book, {"source": "images", "images": len(items)})
    return len(items)


def import_pdf(book: Path, file: Path, progress: Callable[[str], None], dpi: int = 180) -> int:
    import fitz

    file = Path(file)
    dst_dir = Path(book) / "pages"
    dst_dir.mkdir(parents=True, exist_ok=True)
    if page_images(dst_dir):
        raise ValueError("プロジェクトに既に画像があります。新しい書名で取り込んでください。")
    native_text: dict[str, str] = {}
    with fitz.open(file) as doc:
        if doc.needs_pass:
            raise ValueError("パスワードで保護されたPDFは取り込めません。")
        count = len(doc)
        for i, page in enumerate(doc, 1):
            pm = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
            dest = dst_dir / f"page_{i:05d}.png"
            pm.save(dest)
            native = page.get_text("text", sort=True).strip()
            if len(native) > 40:
                native_text[dest.name] = native
            progress(f"PDF取り込み: {i}/{count}")
    (Path(book) / "native_text.json").write_text(json.dumps(native_text, ensure_ascii=False, indent=2), encoding="utf-8")
    save_manifest(book, {"source": "pdf", "source_pdf_filename": file.name, "images": count,
                         "native_text_pages": len(native_text)})
    return count
