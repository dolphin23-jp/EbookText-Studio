"""Local OCR pipeline: Tesseract or optionally NDLOCR-Lite.

All OCR runs on the user's computer. No cloud upload or copyrighted-book
transmission is performed by this program.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import fitz
from PIL import Image

from .audit import audit_book, write_audit
from .linereread import read_page
from .textclean import clean_pages, render_review, review_lines
from .core import (_write_json_atomic, file_sha256, normalize_text, page_images,
                   prepare_ocr_ex, render_markdown, replacements_load, save_manifest)


@dataclass
class OCROptions:
    engine: str = "Tesseract"
    lang: str = "jpn+eng"
    psm: int = 3
    trim: bool = False
    preprocess: bool = True
    upscale: bool = True
    prefer_native_pdf: bool = True
    ndl_command: str = "auto"
    tesseract_exe: str = ""
    reread_low_conf: bool = True   # re-read low-confidence lines one by one (Tesseract only)
    workers: int = 0          # parallel Tesseract pages; 0 = auto (CPU count, max 8)
    clean_text: bool = True   # drop status bar / copyright / running headers, lift page numbers
    audit: bool = True        # report ink the OCR left unread (output/OCR抜け確認.md)


def _find_tesseract(explicit: str = "") -> str:
    if explicit and Path(explicit).is_file():
        return explicit
    name = shutil.which("tesseract")
    if name:
        return name
    for candidate in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                      r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if Path(candidate).exists():
            return candidate
    raise RuntimeError("Tesseract本体が見つかりません。READMEのTesseract導入手順を確認してください。")


def user_tessdata_dir() -> Path:
    """Per-user language data folder that setup.ps1 downloads into (no admin rights needed)."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "EbookTextStudio" / "tessdata"


def _use_user_tessdata() -> None:
    """Point Tesseract at the per-user data folder when it holds the base languages."""
    folder = user_tessdata_dir()
    if (folder / "eng.traineddata").is_file() and (folder / "jpn.traineddata").is_file():
        os.environ["TESSDATA_PREFIX"] = str(folder)


def _check_tesseract(opts: OCROptions) -> None:
    import pytesseract

    _use_user_tessdata()
    pytesseract.pytesseract.tesseract_cmd = _find_tesseract(opts.tesseract_exe)
    available = set(pytesseract.get_languages(config=""))
    missing = [l for l in opts.lang.split("+") if l not in available]
    if missing:
        raise RuntimeError(f"Tesseractの言語データがありません: {', '.join(missing)}。jpn と eng をインストールしてください。")


def _read_native(book: Path) -> dict[str, str]:
    path = book / "native_text.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return {}


def ndl_support_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "EbookTextStudio"


def managed_ndl() -> list[str] | None:
    """Command for the NDLOCR-Lite that setup.bat installs (its own venv + the repo's src/ocr.py).

    EBOOKTEXT_NDL_HOME / EBOOKTEXT_NDL_PYTHON override the locations (used by tests and by people
    who keep NDLOCR-Lite elsewhere)."""
    home = Path(os.environ.get("EBOOKTEXT_NDL_HOME") or ndl_support_dir() / "ndlocr-lite")
    script = home / "src" / "ocr.py"
    if os.environ.get("EBOOKTEXT_NDL_PYTHON"):
        python = Path(os.environ["EBOOKTEXT_NDL_PYTHON"])
    else:
        venv = ndl_support_dir() / "ndl-venv"
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if script.is_file() and python.is_file():
        return [str(python), str(script)]
    return None


def ndl_available() -> bool:
    try:
        _check_ndl_command("auto")
        return True
    except RuntimeError:
        return False


def _check_ndl_command(command: str) -> list[str]:
    # "auto" (or empty): the NDLOCR-Lite installed by setup.bat, else an `ndlocr-lite` command on PATH.
    # Otherwise: an explicit command or the path of ocr.py.
    command = command.strip().strip('"')
    if command in ("", "auto"):
        managed = managed_ndl()
        if managed:
            return managed
        if shutil.which("ndlocr-lite"):
            return ["ndlocr-lite"]
        raise RuntimeError("NDLOCR-Liteが導入されていません。setup.batを実行するか、公式READMEに従ってCLIをセットアップしてください。")
    if command.lower().endswith(".py") and Path(command).is_file():
        if getattr(sys, "frozen", False):
            # sys.executable is the packaged exe, so a separate Python is needed to run ocr.py.
            python = shutil.which("py") or shutil.which("python")
            if not python:
                raise RuntimeError("exe版でNDLOCR-Liteのocr.pyを使うにはPythonが必要です。ndlocr-liteコマンドを指定してください。")
            return [python, command]
        return [sys.executable, command]
    if Path(command).is_file() or shutil.which(command):
        return [command]
    raise RuntimeError("NDLOCR-Liteが見つかりません。公式READMEに従ってCLIをセットアップするか、src/ocr.pyのパスを指定してください。")


def _ndlocrlite_pending(pending: list[Path], opts: OCROptions, stop: threading.Event,
                        status: Callable[[str], None], work_dir: Path) -> dict[str, tuple[str, list[dict]]]:
    cmd = _check_ndl_command(opts.ndl_command)
    # Invoke the official batch CLI once: loading the models per-page is prohibitively slow.
    batch = work_dir / "input"
    out = work_dir / "output"
    batch.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for path in pending:
        shutil.copy2(path, batch / path.name)
    args = [*cmd, "--sourcedir", str(batch), "--output", str(out)]
    status(f"NDLOCR-Liteで{len(pending)}ページ処理中。初回はモデルの準備に時間がかかります。")
    # Hide Windows console; avoid flooding a pipe that would block the child.
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with (work_dir / "ndlocr.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, **kwargs)
        while process.poll() is None:
            if stop.wait(0.5):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise InterruptedError("NDLOCR-Liteを停止しました。")
        if process.returncode != 0:
            content = (work_dir / "ndlocr.log").read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"NDLOCR-Liteが終了コード{process.returncode}で失敗しました。ログ末尾:\n{content[-2500:]}")
    results = {}
    for path in pending:
        stem = path.stem
        txt_path = out / (stem + ".txt")
        json_path = out / (stem + ".json")
        if not txt_path.is_file():
            raise RuntimeError(f"NDLOCR-LiteのTXTが見つかりません: {txt_path}。バージョン/設定を確認してください。")
        text = txt_path.read_text(encoding="utf-8-sig")
        boxes = []
        if json_path.is_file():
            raw = json.loads(json_path.read_text(encoding="utf-8-sig"))
            for group in raw.get("contents", []):
                if isinstance(group, list):
                    boxes += [item for item in group if isinstance(item, dict)]
        results[path.name] = (normalize_text(text), boxes)
    return results


def _add_image_page(doc: fitz.Document, image: Path) -> fitz.Page:
    with Image.open(image) as im:
        w, h = im.size
    # 1 pixel == 1 PDF point; coordinates from OCR can then be mapped exactly.
    page = doc.new_page(width=w, height=h)
    page.insert_image(page.rect, filename=str(image), keep_proportion=False)
    return page


def _add_invisible_boxes(page: fitz.Page, boxes: list[dict], img_width: int,
                         img_height: int) -> None:
    if not boxes:
        return
    sx = page.rect.width / max(img_width, 1)
    sy = page.rect.height / max(img_height, 1)
    for item in boxes:
        text = str(item.get("text", "")).strip()
        coord = item.get("boundingBox")
        if not text or not isinstance(coord, list) or len(coord) < 4:
            continue
        try:
            xs = [float(p[0]) for p in coord]
            ys = [float(p[1]) for p in coord]
            left, right = min(xs) * sx, max(xs) * sx
            top, bottom = min(ys) * sy, max(ys) * sy
        except (TypeError, ValueError, IndexError):
            continue
        if right <= left or bottom <= top:
            continue
        # Invisible Unicode/CJK text layer; font is bundled into PyMuPDF.
        size = max(3.0, min(70.0, (bottom - top) * .85))
        try:
            # Keep long lines inside their box so search highlights do not overrun it.
            unit = fitz.get_text_length(text, fontname="japan", fontsize=1)
            if unit > 0:
                size = max(3.0, min(size, (right - left) / unit))
        except (ValueError, RuntimeError):
            pass
        try:
            page.insert_text((left, min(page.rect.height - 1, top + size)), text,
                             fontsize=size, fontname="japan", render_mode=3,
                             overlay=True)
        except (ValueError, RuntimeError):
            # A bad box must not invalidate a whole book.
            continue


def _write_image_pdf(images: list[Path], dest: Path, stop: threading.Event,
                     status: Callable[[str], None]) -> None:
    with fitz.open() as doc:
        for i, path in enumerate(images, 1):
            if stop.is_set():
                raise InterruptedError("PDF生成を停止しました。")
            _add_image_page(doc, path)
            if i % 10 == 0 or i == len(images):
                status(f"画像PDF作成中 {i}/{len(images)}")
        tmp = dest.with_suffix(".partial.pdf")
        doc.save(tmp, garbage=3, deflate=True)
    tmp.replace(dest)


def _write_searchable_boxes(images: list[Path], box_data: dict[str, list[dict]],
                         dest: Path, stop: threading.Event, status: Callable[[str], None]) -> None:
    with fitz.open() as doc:
        for i, path in enumerate(images, 1):
            if stop.is_set():
                raise InterruptedError("検索可能PDF生成を停止しました。")
            page = _add_image_page(doc, path)
            with Image.open(path) as img:
                w, h = img.size
            _add_invisible_boxes(page, box_data.get(path.name, []), w, h)
            if i % 10 == 0 or i == len(images):
                status(f"検索可能PDFを統合中 {i}/{len(images)}")
        tmp = dest.with_suffix(".partial.pdf")
        doc.save(tmp, garbage=3, deflate=True)
    tmp.replace(dest)


def _tesseract_pages(paths: list[Path], opts: OCROptions, stop: threading.Event,
                     status: Callable[[str], None], total: int) -> dict[str, tuple[str, list[dict]]]:
    """OCR pages with Tesseract, several at a time. Returns name -> (text, boxes in original image pixels)."""
    workers = opts.workers if opts.workers > 0 else max(1, min(os.cpu_count() or 1, 8))
    workers = max(1, min(workers, len(paths)))
    saved_omp = os.environ.get("OMP_THREAD_LIMIT")
    if workers > 1:
        # One thread per Tesseract process: parallel pages scale better than parallel threads inside one page.
        os.environ["OMP_THREAD_LIMIT"] = "1"

    def one(path: Path) -> tuple[str, str, list[dict], object]:
        if stop.is_set():
            raise InterruptedError("OCRを停止しました。")
        with Image.open(path) as original:
            pre, (sx, sy, dx, dy) = prepare_ocr_ex(original.convert("RGB"), upscale=opts.upscale,
                                                  autocontrast=opts.preprocess, trim=opts.trim)
        text, stats, boxes = read_page(pre, opts.lang, opts.psm, reread=opts.reread_low_conf)
        for item in boxes:
            item["boundingBox"] = [[x / sx + dx, y / sy + dy] for x, y in item["boundingBox"]]
        return path.name, normalize_text(text), boxes, stats

    results: dict[str, tuple[str, list[dict]]] = {}
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [executor.submit(one, p) for p in paths]
        for done, future in enumerate(as_completed(futures), 1):
            name, text, boxes, stats = future.result()
            results[name] = (text, boxes)
            msg = f"Tesseract: {done}/{len(paths)}（全{total}ページ）"
            if stats.reread:
                msg += f" 低信頼{stats.reread}行を再読み→{stats.improved}行改善"
            status(msg)
            if stop.is_set():
                raise InterruptedError("OCRを停止しました。")
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if workers > 1:
            if saved_omp is None:
                os.environ.pop("OMP_THREAD_LIMIT", None)
            else:
                os.environ["OMP_THREAD_LIMIT"] = saved_omp
    executor.shutdown(wait=True)
    return results


def convert_book(book: Path, title: str, opts: OCROptions, stop: threading.Event,
                 status: Callable[[str], None]) -> int:
    book = Path(book)
    images = page_images(book / "pages")
    if not images:
        raise ValueError("pages内に画像がありません。まずキャプチャ/取り込みを行ってください。")
    if opts.engine not in ("Tesseract", "NDLOCR-Lite"):
        raise ValueError("未対応のOCRエンジンです。")
    if opts.psm not in (3, 4, 6, 11, 12):
        raise ValueError("不正なPSM指定です。")
    output = book / "output"
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = book / "ocr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_dir = output / "pages"
    text_dir.mkdir(parents=True, exist_ok=True)
    cache_json_path = cache_dir / "cache.json"
    if cache_json_path.exists():
        try:
            cache = json.loads(cache_json_path.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}
    else:
        cache = {}
    old_entries = cache.get("entries", {})
    new_entries = {}
    native_texts = _read_native(book) if opts.prefer_native_pdf else {}
    replacement_path = book / "replacements.json"
    replacements = replacements_load(replacement_path)
    new_text: dict[str, str] = {}
    boxes_by_name: dict[str, list[dict]] = {}
    # Replacements only affect user-facing plain text, not the underlying OCR cache/PDF.
    try:
        import pytesseract
    except ImportError:
        if opts.engine == "Tesseract":
            raise RuntimeError("pytesseractがありません。setup.batを実行してください。")
    if opts.engine == "Tesseract":
        _check_tesseract(opts)
    else:
        _check_ndl_command(opts.ndl_command)

    def cache_key(path: Path) -> str:
        return json.dumps({"sha256": file_sha256(path), "engine": opts.engine,
                           "lang": opts.lang if opts.engine == "Tesseract" else "",
                           "psm": opts.psm if opts.engine == "Tesseract" else 0,
                           "preprocess": opts.preprocess if opts.engine == "Tesseract" else False,
                           "upscale": opts.upscale if opts.engine == "Tesseract" else False,
                           "trim": opts.trim if opts.engine == "Tesseract" else False,
                           "reread": opts.reread_low_conf if opts.engine == "Tesseract" else False,
                           "v": 2,  # 2: Tesseract boxes are cached and drive the searchable PDF
                           "native": bool(opts.prefer_native_pdf and path.name in native_texts)}, sort_keys=True)

    keys = {p.name: cache_key(p) for p in images}
    pending = []
    for path in images:
        cached = old_entries.get(path.name, {})
        txt_path = cache_dir / f"{path.stem}.txt"
        valid = cached.get("key") == keys[path.name] and txt_path.is_file()
        if valid:
            new_text[path.name] = txt_path.read_text(encoding="utf-8")
            boxes_by_name[path.name] = cached.get("boxes", [])
            new_entries[path.name] = cached
        else:
            pending.append(path)
    status(f"{len(images)}ページ中{len(pending)}ページをOCR（キャッシュ{len(images)-len(pending)}ページ）")
    # Group source PDF text extraction before OCR. This is superior to OCR if selectable text exists.
    # For source-native pages, searchable PDFs are still rendered with a text layer below.
    native_pending = []
    for path in pending:
        if path.name in native_texts:
            raw = normalize_text(native_texts[path.name])
            raw_path = cache_dir / f"{path.stem}.txt"
            raw_path.write_text(raw, encoding="utf-8")
            # Directly extracted native text has no original OCR coordinates.
            new_text[path.name] = raw
            new_entries[path.name] = {"key": keys[path.name], "boxes": [], "native": True}
            native_pending.append(path)
    pending = [p for p in pending if p not in native_pending]

    if opts.engine == "NDLOCR-Lite" and pending:
        with tempfile.TemporaryDirectory(prefix="ebooktext_ndl_") as td:
            results = _ndlocrlite_pending(pending, opts, stop, status, Path(td))
        for i, path in enumerate(pending, 1):
            if stop.is_set():
                raise InterruptedError("OCRを停止しました。")
            raw, boxes = results[path.name]
            (cache_dir / f"{path.stem}.txt").write_text(raw, encoding="utf-8")
            new_text[path.name] = raw
            boxes_by_name[path.name] = boxes
            new_entries[path.name] = {"key": keys[path.name], "boxes": boxes}
            _write_json_atomic(cache_json_path, {"entries": {**old_entries, **new_entries}})
            status(f"NDLOCR-Lite結果保存: {i}/{len(pending)}")
    elif opts.engine == "Tesseract":
        todo = pending + native_pending
        results = _tesseract_pages(todo, opts, stop, status, len(images))
        for path in todo:
            raw, boxes = results[path.name]
            if path in native_pending:
                # Markdown uses the PDF's own text; the OCR boxes only feed the searchable PDF.
                raw = new_text[path.name]
            else:
                new_text[path.name] = raw
            (cache_dir / f"{path.stem}.txt").write_text(raw, encoding="utf-8")
            boxes_by_name[path.name] = boxes
            new_entries[path.name] = {"key": keys[path.name], "boxes": boxes,
                                      **({"native": True} if path in native_pending else {})}
        _write_json_atomic(cache_json_path, {"entries": {**old_entries, **new_entries}})
    else:
        pass
    if opts.engine != "Tesseract":
        for path in native_pending:
            boxes_by_name[path.name] = []

    _write_json_atomic(cache_json_path, {"entries": {**old_entries, **new_entries}})
    assembled = []
    for path in images:
        raw = new_text[path.name]
        corrected = raw
        for wrong, right in replacements.items():
            corrected = corrected.replace(wrong, right)
        text_path = text_dir / f"{path.stem}.txt"
        text_path.write_text(corrected + "\n", encoding="utf-8")
        assembled.append((path.name, corrected))
    notes: dict[str, list[str]] = {}
    if opts.clean_text:
        cleaned, note_lists, running = clean_pages([t for _, t in assembled])
        assembled = [(n, c) for (n, _), c in zip(assembled, cleaned)]
        notes = {n: nl for (n, _), nl in zip(assembled, note_lists) if nl}
        for (path, (_, body)) in zip(images, assembled):
            (text_dir / f"{path.stem}.txt").write_text(body + "\n", encoding="utf-8")
        if running:
            status(f"繰り返し行（柱・版権表記など）{len(running)}種類を本文から除外")
    (output / "全文.md").write_text(render_markdown(title, assembled, notes), encoding="utf-8")
    (output / "全文.txt").write_text("\n\n".join(t for _, t in assembled) + "\n", encoding="utf-8")
    flagged = review_lines(assembled)
    (output / "要確認.md").write_text(render_review(title, flagged), encoding="utf-8")
    status(f"数値・単位を含む要確認行: {len(flagged)}行（output/要確認.md）")
    if opts.audit:
        try:
            audits = audit_book(images, boxes_by_name)
            write_audit(book, title, images, audits)
            bad = [a for a in audits if a.findings]
            if bad:
                n = sum(len(a.findings) for a in bad)
                status(f"OCR抜けの疑い: {len(bad)}枚・{n}箇所（output/OCR抜け確認.md、赤枠画像はoutput/audit/）")
            else:
                status("OCR抜けの疑いは見つかりませんでした。")
        except Exception as exc:  # the audit is advisory and must never fail a conversion
            status(f"OCR抜け確認をスキップしました: {exc}")
    status("原画像PDFを作成中")
    _write_image_pdf(images, output / "画像PDF.pdf", stop, status)
    status("検索可能PDFを作成中")
    _write_searchable_boxes(images, boxes_by_name, output / "検索可能PDF.pdf", stop, status)
    save_manifest(book, {"ocr_engine": opts.engine, "ocr_count": len(images),
                         "output_files": ["output/全文.md", "output/全文.txt", "output/画像PDF.pdf", "output/検索可能PDF.pdf"]})
    return len(images)
