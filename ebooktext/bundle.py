"""One self-contained Markdown file per book, made from an existing output folder.

`全文.md` points to images in `figures/` and to table images/CSV in `tables/`. Moving or uploading
`全文.md` alone breaks those links. This module writes a single file that needs nothing else:

* mode "text" (for RAG / reading by AI, the default): figures stay as their caption and the text
  read inside them ("図中の文字"), image links and table-image/CSV links are dropped, HTML that
  only helps viewers is turned into plain text. Smallest token count: a picture embedded as text
  costs tens of thousands of tokens and is not seen as a picture by most AI tools.
* mode "embed" (for viewers such as Obsidian / Typora): the same, but each figure image is put
  inside the file as a data URI (downscaled and JPEG-compressed when that is smaller).

It only reads the output folder, so it can be run any time after a conversion, without OCR.
"""
from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .core import safe_name

_IMAGE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)\s*$")
_TABLE_LINKS = re.compile(r"^<sub>\[表の原画像.*$")
_SOURCE = re.compile(r"^<!-- source: [^>]*-->\s*$")
_PAGE_END = re.compile(r"^<!-- p\.(\d+) ここまで -->\s*$")
_PAGE_BREAK = re.compile(r"^<!-- 改ページ -->\s*$")
_DETAILS_OPEN = re.compile(r"^<details><summary>(.*?)</summary>\s*$")
_DETAILS_CLOSE = re.compile(r"^</details>\s*$")
MAX_WIDTH = 1000


@dataclass
class BundleResult:
    path: Path
    chars: int
    images: int
    missing: list[str]


def _data_uri(path: Path) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.width > MAX_WIDTH:
            im = im.resize((MAX_WIDTH, round(im.height * MAX_WIDTH / im.width)), Image.Resampling.LANCZOS)
        png, jpg = io.BytesIO(), io.BytesIO()
        im.save(png, "PNG", optimize=True)
        im.save(jpg, "JPEG", quality=80, optimize=True)
    data, mime = (jpg.getvalue(), "image/jpeg") if jpg.tell() < png.tell() else (png.getvalue(), "image/png")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def bundle_text(markdown: str, base: Path, mode: str = "text") -> tuple[str, int, list[str]]:
    """Rewrite a structured Markdown so that it needs no other file. Returns (text, images, missing)."""
    if mode not in ("text", "embed"):
        raise ValueError("mode は text か embed です。")
    out: list[str] = []
    images = 0
    missing: list[str] = []
    for line in markdown.splitlines():
        if _TABLE_LINKS.match(line) or _SOURCE.match(line) or _PAGE_BREAK.match(line):
            continue
        m = _PAGE_END.match(line)
        if m:
            out.append(f"〔p.{m.group(1)} ここまで〕")
            continue
        m = _IMAGE.match(line)
        if m:
            if mode == "embed":
                path = base / m.group("path")
                if path.is_file():
                    out.append(f"![{m.group('alt')}]({_data_uri(path)})")
                    images += 1
                else:
                    missing.append(m.group("path"))
            continue
        m = _DETAILS_OPEN.match(line)
        if m:
            out.append(f"{m.group(1)}:")
            continue
        if _DETAILS_CLOSE.match(line):
            continue
        out.append(line)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # "図中の文字:" directly followed by its list reads better without the blank line.
    text = re.sub(r"(図中の文字:)\n\n(- )", r"\1\n\2", text)
    return text.strip() + "\n", images, missing


def book_title(markdown: str, fallback: str) -> str:
    first = markdown.lstrip().splitlines()[0] if markdown.strip() else ""
    return first[2:].strip() if first.startswith("# ") else fallback


def bundle_book(output: Path, mode: str = "text", source: str = "全文.md") -> BundleResult:
    """Write `<書籍名>.md` (text) or `<書籍名>_画像埋め込み.md` (embed) into the output folder."""
    output = Path(output)
    src = output / source
    if not src.is_file():
        raise FileNotFoundError(f"{src} がありません。先にOCRを実行してください。")
    markdown = src.read_text(encoding="utf-8")
    text, images, missing = bundle_text(markdown, output, mode)
    title = safe_name(book_title(markdown, output.parent.name))
    name = f"{title}.md" if mode == "text" else f"{title}_画像埋め込み.md"
    dest = output / name
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(dest)
    return BundleResult(dest, len(text), images, missing)


def describe(result: BundleResult) -> str:
    size = result.path.stat().st_size
    msg = f"1ファイルにまとめました: {result.path.name}（{result.chars:,}文字、{size / 1024 / 1024:.1f}MB"
    if result.images:
        msg += f"、画像{result.images}枚を埋め込み"
    msg += "）"
    if result.missing:
        msg += f"。見つからない画像{len(result.missing)}枚は省きました"
    return msg
