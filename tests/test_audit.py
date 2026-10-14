import threading
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from ebooktext.audit import audit_image, render_report, write_audit, audit_book
from ebooktext.ocr import OCROptions, convert_book

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def page(lines, size=(900, 700), invert_rows=()):
    """White page with one text line every 60 px; returns image and a perfect set of OCR boxes."""
    font = ImageFont.truetype(FONT, 28)
    im = Image.new('RGB', size, 'white')
    d = ImageDraw.Draw(im)
    boxes = []
    for i, text in enumerate(lines):
        y = 60 + i * 60
        if i in invert_rows:
            d.rectangle([40, y - 6, 860, y + 40], fill=(30, 90, 160))
            d.text((60, y), text, fill='white', font=font)
        else:
            d.text((60, y), text, fill='black', font=font)
        w = d.textlength(text, font=font)
        boxes.append({'text': text, 'boundingBox': [[60, y], [60 + w, y], [60 + w, y + 34], [60, y + 34]]})
    return im, boxes


LINES = ['The first line of the page', 'A second line with more words in it',
         'Third line follows here', 'Fourth line closes the block']


def test_complete_ocr_has_no_findings():
    im, boxes = page(LINES)
    r = audit_image(im, boxes, 'p.png')
    assert r.lines == 4 and r.ok and not r.skipped


def test_missing_whole_line_is_reported_at_its_position():
    im, boxes = page(LINES)
    del boxes[2]
    r = audit_image(im, boxes)
    assert [f.kind for f in r.findings] == ['line']
    f = r.findings[0]
    assert f.y0 <= 60 + 2 * 60 + 5 and f.y1 >= 60 + 2 * 60 + 20


def test_missing_chunk_inside_a_line_is_reported_as_partial():
    im, boxes = page(LINES)
    font = ImageFont.truetype(FONT, 28)
    text = LINES[1]
    cut = ImageDraw.Draw(im).textlength(text[:14], font=font)
    # OCR kept only the beginning of the line: the tail is lost.
    boxes[1]['boundingBox'] = [[60, 120], [60 + cut, 120], [60 + cut, 154], [60, 154]]
    r = audit_image(im, boxes)
    assert len(r.findings) == 1 and r.findings[0].kind == 'partial'
    assert r.findings[0].x0 > 60 + cut - 5


def test_white_on_colour_heading_is_seen():
    im, boxes = page(LINES, invert_rows=(1,))
    del boxes[1]
    r = audit_image(im, boxes)
    assert any(f.kind == 'line' for f in r.findings)


def test_blank_page_and_missing_boxes():
    blank = Image.new('RGB', (600, 400), 'white')
    assert audit_image(blank, [{'text': 'x', 'boundingBox': [[0, 0], [5, 0], [5, 5], [0, 5]]}]).ok
    im, _ = page(LINES)
    assert audit_image(im, []).skipped


def test_big_figure_block_is_not_mistaken_for_text():
    im, boxes = page(LINES)
    ImageDraw.Draw(im).rectangle([100, 330, 800, 640], fill=(200, 60, 60))
    ImageDraw.Draw(im).ellipse([200, 380, 700, 600], fill=(20, 20, 120))
    r = audit_image(im, boxes)
    assert r.ok


def test_box_slightly_larger_than_text_is_fine():
    im, boxes = page(LINES)
    for b in boxes:
        (x0, y0), (x1, _), (_, y2), _ = b['boundingBox']
        b['boundingBox'] = [[x0 - 6, y0 - 4], [x1 + 6, y0 - 4], [x1 + 6, y2 + 4], [x0 - 6, y2 + 4]]
    assert audit_image(im, boxes).ok


def test_report_and_files(tmp_path):
    book = tmp_path / 'b'
    (book / 'pages').mkdir(parents=True)
    (book / 'output').mkdir()
    im, boxes = page(LINES)
    del boxes[3]
    path = book / 'pages' / 'page_00001.png'
    im.save(path)
    audits = audit_book([path], {path.name: boxes})
    report = write_audit(book, '検証', [path], audits)
    text = report.read_text(encoding='utf-8')
    assert 'page_00001.png' in text and '行ごと未認識' in text
    assert (book / 'output' / 'audit' / 'page_00001_audit.png').is_file()
    assert (book / 'output' / 'ocr_audit.json').is_file()
    ok_im, ok_boxes = page(LINES)
    assert '疑い箇所は見つかりませんでした' in render_report('t', [audit_image(ok_im, ok_boxes, 'ok.png')])


def test_convert_book_runs_audit_and_can_disable_it(tmp_path):
    def make(name):
        b = tmp_path / name
        (b / 'pages').mkdir(parents=True)
        im, _ = page(['Hello World one', 'Kidney function test'])
        im.save(b / 'pages' / 'page_00001.png')
        return b
    msgs = []
    b1 = make('on')
    convert_book(b1, 't', OCROptions(lang='eng', preprocess=False), threading.Event(), msgs.append)
    assert (b1 / 'output' / 'OCR抜け確認.md').is_file()
    assert any('OCR抜け' in m for m in msgs)
    b2 = make('off')
    convert_book(b2, 't', OCROptions(lang='eng', preprocess=False, audit=False), threading.Event(), lambda m: None)
    assert not (b2 / 'output' / 'OCR抜け確認.md').exists()


def test_sparse_box_flags_text_lost_inside_a_line_box():
    im, boxes = page(LINES)
    boxes[1]['text'] = 'A second'          # box still spans the whole line, text is short
    r = audit_image(im, boxes)
    assert [f.kind for f in r.findings] == ['sparse']
    assert r.findings[0].y0 <= 124


def test_sparse_box_ignores_short_labels_and_latin_text():
    im, boxes = page(LINES)
    boxes[0]['text'] = 'Hi'               # under the minimum length
    assert audit_image(im, boxes).ok
    # Latin text is narrower per character than full-width text: must never trip the ratio.
    assert audit_image(*page(LINES)).ok


def test_vertical_frame_does_not_hide_the_line_structure():
    im, boxes = page(LINES)
    d = ImageDraw.Draw(im)
    d.line([20, 0, 20, 700], fill='black', width=3)       # frame / side bar
    d.line([880, 0, 880, 700], fill='black', width=3)
    for y in range(0, 700, 12):                           # dashed bar
        d.line([34, y, 34, y + 8], fill=(60, 60, 200), width=2)
    del boxes[2]
    r = audit_image(im, boxes)
    assert r.lines == 4
    assert [f.kind for f in r.findings] == ['line']


def test_thin_rule_and_page_boundary_bar_are_not_text():
    im, boxes = page(LINES)
    d = ImageDraw.Draw(im)
    d.line([300, 100, 880, 100], fill=(80, 80, 200), width=2)   # underline beside a heading
    d.rectangle([0, 660, 900, 700], fill=(20, 20, 20))           # viewer bar at the bottom
    assert audit_image(im, boxes).ok
