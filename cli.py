"""Optional command line processing of image folders and ordinary PDF files.

Examples:
  python cli.py one --input "C:/Books/screenshots" --base "C:/EbookText" --title "腎臓内科学"
  python cli.py batch --input "C:/Books/to_process" --base "C:/EbookText"
"""
from __future__ import annotations

import argparse
import threading
from pathlib import Path

from ebooktext.core import EXTENSIONS, page_images, project_dir
from ebooktext.importer import import_images, import_pdf
from ebooktext.ocr import OCROptions, convert_book, ndl_available
from ebooktext.pageprep import ImportOptions


def process_one(src: Path, base: Path, title: str, opts: OCROptions,
                imp: ImportOptions | None = None) -> Path:
    book = project_dir(base, title)
    if src.is_file() and src.suffix.lower() == '.pdf':
        if not (book / 'pages').is_dir():
            import_pdf(book, src, print, opts=imp)
    elif src.is_dir():
        if src.resolve() != (book / 'pages').resolve() and not page_images(book / 'pages'):
            import_images(book, src, print, imp)
    else:
        raise ValueError(f'画像フォルダまたはPDFが必要です: {src}')
    count = convert_book(book, title, opts, threading.Event(), print)
    print(f'完了: {title} ({count}ページ) → {book / "output"}')
    return book


def main() -> int:
    parser = argparse.ArgumentParser(description='EbookText Studio CLI / image folder or PDF to Markdown + PDF')
    parser.add_argument('action', choices=['one', 'batch'])
    parser.add_argument('--input', required=True, type=Path, help='one: image folder/PDF; batch: folder containing subfolders/PDFs')
    parser.add_argument('--base', required=True, type=Path, help='book project destination base directory')
    parser.add_argument('--title', help='book title (required for one)')
    parser.add_argument('--engine', choices=['Tesseract', 'NDLOCR-Lite'], default=None,
                        help='default: NDLOCR-Lite when installed (setup.bat), otherwise Tesseract')
    parser.add_argument('--lang', default='jpn+eng')
    parser.add_argument('--ndl-command', default='auto', help='auto = the install from setup.bat, else ndlocr-lite on PATH')
    parser.add_argument('--psm', type=int, default=3, choices=[3, 4, 6, 11, 12])
    parser.add_argument('--no-native', action='store_true')
    parser.add_argument('--no-strip-bars', action='store_true', help='keep dark status/home bars at the image edges')
    parser.add_argument('--no-split', action='store_true', help='do not split two-page spreads')
    parser.add_argument('--rtl', action='store_true', help='spreads read right-to-left (vertical-text books)')
    parser.add_argument('--workers', type=int, default=0, help='pages OCRed in parallel (Tesseract); 0 = auto')
    parser.add_argument('--no-reread', action='store_true', help='skip the second pass on low-confidence lines')
    parser.add_argument('--no-clean', action='store_true', help='keep running headers, page numbers, copyright lines')
    parser.add_argument('--no-structure', action='store_true', help='page-by-page 全文.md without tables/figures/headings')
    parser.add_argument('--no-numcheck', action='store_true', help='do not re-read short numbers with Tesseract')
    args = parser.parse_args()
    engine = args.engine or ('NDLOCR-Lite' if ndl_available() else 'Tesseract')
    print(f'OCRエンジン: {engine}')
    opts = OCROptions(engine=engine, lang=args.lang, psm=args.psm,
                      ndl_command=args.ndl_command, prefer_native_pdf=not args.no_native,
                      clean_text=not args.no_clean, reread_low_conf=not args.no_reread, workers=args.workers,
                      structure=not args.no_structure, number_check=not args.no_numcheck)
    imp = ImportOptions(strip_bars=not args.no_strip_bars, split_spreads=not args.no_split,
                        right_to_left=args.rtl)
    if args.action == 'one':
        if not args.title:
            parser.error('--title is required for one')
        process_one(args.input, args.base, args.title, opts, imp)
        return 0
    if not args.input.is_dir():
        parser.error('batch --input must be a directory')
    sources = sorted((p for p in args.input.iterdir()
                      if (p.is_dir() and page_images(p)) or (p.is_file() and p.suffix.lower() == '.pdf')),
                     key=lambda p: p.name.lower())
    if not sources:
        parser.error('No image subfolders or PDF files found')
    failures = []
    for num, src in enumerate(sources, 1):
        print(f'[{num}/{len(sources)}] {src.name}')
        try:
            process_one(src, args.base, src.stem, opts, imp)
        except (OSError, ValueError, RuntimeError, InterruptedError) as exc:
            print(f'失敗: {src.name}: {exc}')
            failures.append(src.name)
    print(f'全{len(sources)}件、失敗{len(failures)}件')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
