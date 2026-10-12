from pathlib import Path
from PIL import Image, ImageDraw

from ebooktext.core import (safe_name, nearly_identical, auto_trim_white, render_markdown,
                           normalize_text, replacements_load, page_images)


def test_safe_name():
    assert safe_name('肺/腎: "本"?') == '肺_腎_ _本__'
    assert safe_name(' ') == '名称未設定'


def test_identical():
    a = Image.new('RGB', (300, 400), 'white')
    b = Image.new('RGB', (300, 400), 'white')
    c = Image.new('RGB', (300, 400), 'black')
    assert nearly_identical(a, b)
    assert not nearly_identical(a, c)


def test_auto_trim():
    a = Image.new('RGB', (600, 800), 'white')
    ImageDraw.Draw(a).rectangle((60, 70, 540, 740), fill='black')
    b = auto_trim_white(a)
    assert b.width < a.width and b.height < a.height
    assert b.width >= 480 and b.height >= 670


def test_markdown():
    text = render_markdown('医学書', [('page_00001.png','腎臓\n治療'),('page_00002.png','高K')])
    assert text.startswith('# 医学書')
    assert '<!-- source: page_00002.png -->' in text
    assert '腎臓\n治療' in text


def test_replacements_not_automatic(tmp_path):
    p = tmp_path / 'replacements.json'
    replacements = replacements_load(p)
    assert replacements == {}
    assert p.exists()
