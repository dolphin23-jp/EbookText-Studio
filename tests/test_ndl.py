import json
import sys
import threading
from pathlib import Path

import pytest

from ebooktext import ocr
from ebooktext.ocr import OCROptions, convert_book, managed_ndl, _check_ndl_command
from PIL import Image, ImageDraw

FAKE_NDL = '''
import argparse, json, pathlib
p = argparse.ArgumentParser()
p.add_argument("--sourcedir"); p.add_argument("--output")
a = p.parse_args()
for img in sorted(pathlib.Path(a.sourcedir).iterdir()):
    out = pathlib.Path(a.output)
    (out / (img.stem + ".txt")).write_text("第1章\\n50～60歳 " + img.stem, encoding="utf-8")
    box = [[10, 10], [10, 40], [300, 10], [300, 40]]
    (out / (img.stem + ".json")).write_text(json.dumps({"contents": [[{"boundingBox": box, "text": "50～60歳 " + img.stem}]]}), encoding="utf-8")
'''


def book_with_pages(tmp_path, n=2):
    book = tmp_path / 'book'
    (book / 'pages').mkdir(parents=True)
    for i in range(1, n + 1):
        im = Image.new('RGB', (400, 300), 'white')
        ImageDraw.Draw(im).text((10, 10), f'p{i}', fill='black')
        im.save(book / 'pages' / f'page_{i:05d}.png')
    return book


def test_managed_ndl_resolution_via_env(tmp_path, monkeypatch):
    home = tmp_path / 'ndl'
    (home / 'src').mkdir(parents=True)
    (home / 'src' / 'ocr.py').write_text('', encoding='utf-8')
    py = tmp_path / 'python'
    py.write_text('', encoding='utf-8')
    monkeypatch.setenv('EBOOKTEXT_NDL_HOME', str(home))
    monkeypatch.setenv('EBOOKTEXT_NDL_PYTHON', str(py))
    assert managed_ndl() == [str(py), str(home / 'src' / 'ocr.py')]
    assert _check_ndl_command('auto') == managed_ndl()
    assert ocr.ndl_available()


def test_auto_without_install_explains(tmp_path, monkeypatch):
    monkeypatch.setenv('EBOOKTEXT_NDL_HOME', str(tmp_path / 'none'))
    monkeypatch.delenv('EBOOKTEXT_NDL_PYTHON', raising=False)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'empty'))
    monkeypatch.setattr(ocr.shutil, 'which', lambda name: None)
    assert not ocr.ndl_available()
    with pytest.raises(RuntimeError, match='setup.bat'):
        _check_ndl_command('auto')


def test_ndl_engine_end_to_end_with_fake_cli(tmp_path):
    script = tmp_path / 'fake_ocr.py'
    script.write_text(FAKE_NDL, encoding='utf-8')
    book = book_with_pages(tmp_path)
    opts = OCROptions(engine='NDLOCR-Lite', ndl_command=str(script))
    assert convert_book(book, 'ndl', opts, threading.Event(), lambda _: None) == 2
    md = (book / 'output' / '全文.md').read_text(encoding='utf-8')
    assert '50～60歳 page_00001' in md and '50～60歳 page_00002' in md
    review = (book / 'output' / '要確認.md').read_text(encoding='utf-8')
    assert '範囲' in review
    import fitz
    with fitz.open(book / 'output' / '検索可能PDF.pdf') as pdf:
        assert pdf[0].search_for('50～60歳')
