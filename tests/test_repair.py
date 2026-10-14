import threading

import pytest
from PIL import Image, ImageDraw, ImageFont

import ebooktext.ocr as ocr
from ebooktext.audit import audit_book
from ebooktext.ocr import OCROptions, convert_book
from ebooktext.repair import make_crop, repair_book
from test_audit import FONT, LINES, page


def setup_page(tmp_path, drop=None):
    pages = tmp_path / 'pages'
    pages.mkdir()
    im, boxes = page(LINES)
    path = pages / 'page_00001.png'
    im.save(path)
    text = "\n".join(b['text'] for b in boxes)
    if drop is not None:
        removed = boxes.pop(drop)
        text = "\n".join(b['text'] for b in boxes)
    return path, text, {path.name: boxes}


def fake_reader(answers):
    seen = []

    def read(crops):
        seen.extend(crops)
        return [answers.pop(0) for _ in crops]
    read.seen = seen
    return read


def test_missing_line_is_reinserted_in_reading_order(tmp_path):
    path, text, boxes = setup_page(tmp_path, drop=2)
    audits = audit_book([path], boxes)
    texts = {path.name: text}
    reps = repair_book([path], audits, texts, boxes, fake_reader([LINES[2]]))
    assert [r.action for r in reps] == ['行として挿入']
    assert texts[path.name].split("\n") == LINES
    assert len(boxes[path.name]) == 4
    assert audit_book([path], boxes)[0].ok          # nothing left to flag


def test_missing_first_line_goes_to_the_top(tmp_path):
    path, text, boxes = setup_page(tmp_path, drop=0)
    texts = {path.name: text}
    repair_book([path], audit_book([path], boxes), texts, boxes, fake_reader([LINES[0]]))
    assert texts[path.name].split("\n") == LINES


def test_lost_tail_is_appended_to_its_line(tmp_path):
    path, text, boxes = setup_page(tmp_path)
    font = ImageFont.truetype(FONT, 28)
    cut = ImageDraw.Draw(Image.new('RGB', (10, 10))).textlength(LINES[1][:14], font=font)
    boxes[path.name][1]['text'] = LINES[1][:14]
    boxes[path.name][1]['boundingBox'] = [[60, 120], [60 + cut, 120], [60 + cut, 154], [60, 154]]
    texts = {path.name: text.replace(LINES[1], LINES[1][:14])}
    audits = audit_book([path], boxes)
    assert audits[0].findings[0].kind == 'partial'
    reps = repair_book([path], audits, texts, boxes, fake_reader([LINES[1][14:].strip()]))
    assert reps[0].action == '行末に追記'
    assert texts[path.name].split("\n")[1].replace(" ", "") == LINES[1].replace(" ", "")


def test_sparse_line_is_replaced_only_when_longer(tmp_path):
    path, text, boxes = setup_page(tmp_path)
    boxes[path.name][1]['text'] = 'A second'
    texts = {path.name: text.replace(LINES[1], 'A second')}
    audits = audit_book([path], boxes)
    assert audits[0].findings[0].kind == 'sparse'
    # A re-read that is not longer changes nothing.
    r0 = repair_book([path], audits, dict(texts), {k: [dict(b) for b in v] for k, v in boxes.items()},
                     fake_reader(['A sec']))
    assert r0[0].action == '未反映'
    reps = repair_book([path], audits, texts, boxes, fake_reader([LINES[1]]))
    assert reps[0].action == '置換' and reps[0].old == 'A second'
    assert texts[path.name].split("\n")[1] == LINES[1]
    assert audit_book([path], boxes)[0].ok


def test_garbage_duplicates_and_unplaceable_spots_change_nothing(tmp_path):
    path, text, boxes = setup_page(tmp_path, drop=2)
    for answer, why in (('|', '読めません'), (LINES[1], '隣の行に既に')):
        texts = {path.name: text}
        b = {k: [dict(x) for x in v] for k, v in boxes.items()}
        reps = repair_book([path], audit_book([path], b), texts, b, fake_reader([answer]))
        assert reps[0].action == '未反映' and why in reps[0].reason
        assert texts[path.name] == text and len(b[path.name]) == 3
    # The surrounding OCR line is missing from the text: do not guess a position.
    texts = {path.name: 'completely different text'}
    b = {k: [dict(x) for x in v] for k, v in boxes.items()}
    reps = repair_book([path], audit_book([path], b), texts, b, fake_reader([LINES[2]]))
    assert reps[0].action == '未反映' and texts[path.name] == 'completely different text'


def test_crop_is_padded_and_enlarged_to_a_readable_size(tmp_path):
    path, text, boxes = setup_page(tmp_path, drop=2)
    f = audit_book([path], boxes)[0].findings[0]
    with Image.open(path) as im:
        crop = make_crop(im, f)
    assert crop.height >= 40 * 2          # line height scaled to >= ~40px plus the margins
    assert crop.getpixel((2, 2)) == (255, 255, 255)


def test_convert_book_repairs_a_dropped_line_end_to_end(tmp_path, monkeypatch):
    book = tmp_path / 'b'
    (book / 'pages').mkdir(parents=True)
    im, _ = page(['Hello World one', 'Kidney function test', 'Another line here'])
    im.save(book / 'pages' / 'page_00001.png')
    real = ocr._tesseract_pages

    def lossy(paths, opts, stop, status, total):
        res = real(paths, opts, stop, status, total)
        out = {}
        for name, (text, boxes) in res.items():
            keep = [b for b in boxes if 'Kidney' not in b['text']]       # the engine "loses" one line
            out[name] = ("\n".join(l for l in text.split("\n") if 'Kidney' not in l), keep)
        return out
    monkeypatch.setattr(ocr, '_tesseract_pages', lossy)
    msgs = []
    convert_book(book, 't', OCROptions(lang='eng', preprocess=False), threading.Event(), msgs.append)
    md = (book / 'output' / '全文.md').read_text(encoding='utf-8')
    assert 'Kidney function' in md
    assert md.index('Hello World') < md.index('Kidney') < md.index('Another line')
    report = (book / 'output' / 'OCR抜け確認.md').read_text(encoding='utf-8')
    assert '自動で補った箇所' in report
    assert any('本文に補いました' in m for m in msgs)
    # With repair switched off the line stays lost and is reported instead.
    book2 = tmp_path / 'b2'
    (book2 / 'pages').mkdir(parents=True)
    im.save(book2 / 'pages' / 'page_00001.png')
    convert_book(book2, 't', OCROptions(lang='eng', preprocess=False, repair=False), threading.Event(), lambda m: None)
    assert 'Kidney' not in (book2 / 'output' / '全文.md').read_text(encoding='utf-8')
    assert '行ごと未認識' in (book2 / 'output' / 'OCR抜け確認.md').read_text(encoding='utf-8')
