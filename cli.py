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
from ebooktext.ocr import OCROptions, convert_book


def process_one(src: Path, base: Path, title: str, opts: OCROptions) -> Path:
    book = project_dir(base, title)
    if src.is_file() and src.suffix.lower() == '.pdf':
        if not (book / 'pages').is_dir():
            import_pdf(book, src, print)
    elif src.is_dir():
        if src.resolve() != (book / 'pages').resolve() and not page_images(book / 'pages'):
            import_images(book, src, print)
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
    parser.add_argument('--engine', choices=['Tesseract', 'NDLOCR-Lite'], default='Tesseract')
    parser.add_argument('--lang', default='jpn+eng')
    parser.add_argument('--ndl-command', default='ndlocr-lite')
    parser.add_argument('--psm', type=int, default=3, choices=[3, 4, 6, 11, 12])
    parser.add_argument('--no-native', action='store_true')
    args = parser.parse_args()
    opts = OCROptions(engine=args.engine, lang=args.lang, psm=args.psm,
                      ndl_command=args.ndl_command, prefer_native_pdf=not args.no_native)
    if args.action == 'one':
        if not args.title:
            parser.error('--title is required for one')
        process_one(args.input, args.base, args.title, opts)
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
            process_one(src, args.base, src.stem, opts)
        except (OSError, ValueError, RuntimeError, InterruptedError) as exc:
            print(f'失敗: {src.name}: {exc}')
            failures.append(src.name)
    print(f'全{len(sources)}件、失敗{len(failures)}件')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
