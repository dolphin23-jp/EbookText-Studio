import json
import threading
from pathlib import Path

import fitz
import pytest
from PIL import Image, ImageDraw, ImageFont

from ebooktext.ocr import OCROptions, convert_book, _add_invisible_boxes
from ebooktext.importer import import_pdf, import_images


def sample_book(tmp_path):
    path = tmp_path / 'sample'
    pages = path / 'pages'
    pages.mkdir(parents=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 35)
    for i in range(1, 3):
        im = Image.new('RGB', (800, 1100), 'white')
        d = ImageDraw.Draw(im)
        d.text((60, 100), f'Hello World {i}', fill='black', font=font)
        d.text((60, 170), 'Kidney function', fill='black', font=font)
        im.save(pages / f'page_{i:05d}.png')
    return path


def test_tesseract_end_to_end(tmp_path):
    book = sample_book(tmp_path)
    msgs = []
    settings = OCROptions(lang='eng', preprocess=False)
    count = convert_book(book, 'OCR検証', settings, threading.Event(), msgs.append)
    assert count == 2
    out = book / 'output'
    text = (out / '全文.md').read_text(encoding='utf-8')
    assert 'Hello World 1' in text and 'Hello World 2' in text
    assert (out / '全文.txt').exists()
    with fitz.open(out / '画像PDF.pdf') as pdf:
        assert len(pdf) == 2
        assert not pdf[0].get_text().strip()
    with fitz.open(out / '検索可能PDF.pdf') as pdf:
        assert len(pdf) == 2
        assert 'Hello World 1' in pdf[0].get_text()
        assert 'Hello World 2' in pdf[1].get_text()
    assert any('Tesseract' in s for s in msgs)
    # OCR rerun should use cache.
    new_msgs = []
    convert_book(book, 'OCR検証', settings, threading.Event(), new_msgs.append)
    assert any('0ページをOCR' in s for s in new_msgs)


def test_ndl_overlay_unicode():
    doc = fitz.open()
    page = doc.new_page(width=800, height=1000)
    _add_invisible_boxes(page, [{'text': '腎不全', 'boundingBox': [[20,30],[20,60],[140,30],[140,60]]}], 800,1000)
    assert '腎不全' in page.get_text()
    doc.close()


def test_pdf_import_with_native_layer(tmp_path):
    doc = fitz.open()
    p = doc.new_page(width=600,height=800)
    p.insert_text((40,100), 'This is text that can be extracted from the PDF document.', fontsize=15)
    src = tmp_path / 'in.pdf'; doc.save(src); doc.close()
    book = tmp_path / 'project'
    n = import_pdf(book, src, lambda _: None)
    assert n == 1
    native = json.loads((book/'native_text.json').read_text(encoding='utf-8'))
    assert native['page_00001.png'].startswith('This is text')


def test_import_images(tmp_path):
    original = sample_book(tmp_path)
    book = tmp_path / 'copy'
    n = import_images(book, original / 'pages', lambda _: None)
    assert n == 2
    assert (book / 'pages' / 'page_00001.png').is_file()
    with pytest.raises(ValueError):
        import_images(book, original / 'pages', lambda _: None)
