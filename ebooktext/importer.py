"""Import local images or PDFs supplied by the user."""
from __future__ import annotations

import json
from pathlib import Path
from shutil import copy2
from typing import Callable

from PIL import Image

from .core import page_images, save_manifest
from .pageprep import ImportOptions, needs_processing, prepare_page


def _opts_dict(opts: ImportOptions) -> dict:
    return {"strip_bars": opts.strip_bars, "split_spreads": opts.split_spreads,
            "right_to_left": opts.right_to_left}


def _save_pages(images: list[Image.Image], dst_dir: Path, start: int) -> int:
    for k, im in enumerate(images):
        im.save(dst_dir / f"page_{start + k:05d}.png")
    return start + len(images)


def import_images(book: Path, source: Path, progress: Callable[[str], None],
                  opts: ImportOptions | None = None) -> int:
    opts = opts or ImportOptions()
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
    n = 1
    for i, path in enumerate(items, 1):
        with Image.open(path) as im:
            im.load()
            if needs_processing(im, opts):
                n = _save_pages(prepare_page(im, opts), dst_dir, n)
                progress(f"画像取り込み: {i}/{len(items)}")
                continue
        copy2(path, dst_dir / f"page_{n:05d}{path.suffix.lower()}")
        n += 1
        progress(f"画像取り込み: {i}/{len(items)}")
    save_manifest(book, {"source": "images", "source_images": len(items), "images": n - 1,
                         "import_options": _opts_dict(opts)})
    return n - 1


def _page_image(doc, page, dpi: int) -> Image.Image:
    """Embedded image at native resolution when the page is one full-page picture; else a render."""
    import fitz

    infos = page.get_image_info(xrefs=True)
    if len(infos) == 1 and infos[0].get("xref"):
        x0, y0, x1, y1 = infos[0]["bbox"]
        covers = (x1 - x0) >= page.rect.width * 0.97 and (y1 - y0) >= page.rect.height * 0.97
        if covers:
            try:
                pix = fitz.Pixmap(doc, infos[0]["xref"])
                if pix.n - pix.alpha > 3 or pix.colorspace is None:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img = Image.frombytes("RGB" if not pix.alpha else "RGBA", (pix.width, pix.height), pix.samples).convert("RGB")
                if abs(img.width / img.height - page.rect.width / page.rect.height) < 0.01 and not page.rotation:
                    return img
            except (RuntimeError, ValueError):
                pass
    pm = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    return Image.frombytes("RGB", (pm.width, pm.height), pm.samples)


def import_pdf(book: Path, file: Path, progress: Callable[[str], None], dpi: int = 180,
               opts: ImportOptions | None = None) -> int:
    import fitz

    opts = opts or ImportOptions()
    file = Path(file)
    dst_dir = Path(book) / "pages"
    dst_dir.mkdir(parents=True, exist_ok=True)
    if page_images(dst_dir):
        raise ValueError("プロジェクトに既に画像があります。新しい書名で取り込んでください。")
    native_text: dict[str, str] = {}
    n = 1
    with fitz.open(file) as doc:
        if doc.needs_pass:
            raise ValueError("パスワードで保護されたPDFは取り込めません。")
        count = len(doc)
        for i, page in enumerate(doc, 1):
            native = page.get_text("text", sort=True).strip()
            pages = prepare_page(_page_image(doc, page, dpi), opts)
            first = n
            n = _save_pages(pages, dst_dir, n)
            # Native text belongs to a whole source page; keep it only when the page was not split.
            if len(native) > 40 and len(pages) == 1:
                native_text[f"page_{first:05d}.png"] = native
            progress(f"PDF取り込み: {i}/{count}")
    (Path(book) / "native_text.json").write_text(json.dumps(native_text, ensure_ascii=False, indent=2), encoding="utf-8")
    save_manifest(book, {"source": "pdf", "source_pdf_filename": file.name, "source_pdf_pages": count,
                         "images": n - 1, "native_text_pages": len(native_text),
                         "import_options": _opts_dict(opts)})
    return n - 1
