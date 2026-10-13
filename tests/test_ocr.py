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
        assert 'Hello World 1' in ' '.join(pdf[0].get_text().split())
        assert 'Hello World 2' in ' '.join(pdf[1].get_text().split())
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


def test_user_tessdata_is_used_when_complete(tmp_path, monkeypatch):
    import os
    from ebooktext.ocr import user_tessdata_dir, _use_user_tessdata
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    monkeypatch.delenv('TESSDATA_PREFIX', raising=False)   # records 'unset' so teardown restores that
    folder = user_tessdata_dir()
    folder.mkdir(parents=True)
    (folder / 'eng.traineddata').write_bytes(b'x')
    _use_user_tessdata()
    assert 'TESSDATA_PREFIX' not in os.environ  # jpn missing -> leave Tesseract defaults
    (folder / 'jpn.traineddata').write_bytes(b'x')
    _use_user_tessdata()
    assert os.environ['TESSDATA_PREFIX'] == str(folder)
    os.environ.pop('TESSDATA_PREFIX')   # set by the code under test, not by monkeypatch


def test_searchable_pdf_boxes_map_back_to_original_pixels(tmp_path):
    book = sample_book(tmp_path)
    convert_book(book, 't', OCROptions(lang='eng', preprocess=False, workers=2), threading.Event(), lambda _: None)
    with fitz.open(book / 'output' / '検索可能PDF.pdf') as pdf:
        # The page is 800x1100 px; "Kidney function" was drawn at (60, 170) with a 35 px font,
        # while OCR ran on a 1.6x upscaled copy: the box must be mapped back.
        hits = pdf[0].search_for('Kidney function')
        assert hits, 'phrase should be searchable as one run'
        r = hits[0]
        assert 40 < r.x0 < 90 and 150 < r.y0 < 200, r


def test_stop_event_interrupts_ocr(tmp_path):
    book = sample_book(tmp_path)
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        convert_book(book, 't', OCROptions(lang='eng', workers=2), stop, lambda _: None)
