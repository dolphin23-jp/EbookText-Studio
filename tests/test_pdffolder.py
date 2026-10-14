import json
from pathlib import Path

import fitz
import pytest

from ebooktext.importer import find_pdfs, import_pdfs


def make_pdf(path: Path, pages: int, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for i in range(pages):
        p = doc.new_page(width=600, height=800)
        p.insert_text((40, 100), f'{label} page {i + 1}: text that is long enough to count as a native text layer.', fontsize=12)
    doc.save(path)
    doc.close()


def test_find_pdfs_orders_chapters_naturally_and_skips_junk(tmp_path):
    for name in ('第10章 x', '第2章 y', '第1章 z'):
        make_pdf(tmp_path / name / 'scan.pdf', 1, name)
    make_pdf(tmp_path / '第３章 w' / 'scan.pdf', 1, 'w')          # full-width digit
    (tmp_path / 'md').mkdir()
    (tmp_path / 'md' / 'notes.txt').write_text('x')
    (tmp_path / '第1章 z' / '~$lock.pdf').write_bytes(b'junk')
    order = [p.parent.name for p in find_pdfs(tmp_path)]
    assert order == ['第1章 z', '第2章 y', '第３章 w', '第10章 x']


def test_import_pdfs_concatenates_pages_in_order(tmp_path):
    a, b = tmp_path / 'c1' / 'a.pdf', tmp_path / 'c2' / 'b.pdf'
    make_pdf(a, 2, 'AAA')
    make_pdf(b, 3, 'BBB')
    book = tmp_path / 'book'
    msgs = []
    n = import_pdfs(book, [a, b], msgs.append)
    assert n == 5
    assert [p.name for p in sorted((book / 'pages').glob('*.png'))] == [f'page_{i:05d}.png' for i in range(1, 6)]
    native = json.loads((book / 'native_text.json').read_text(encoding='utf-8'))
    assert native['page_00001.png'].startswith('AAA page 1')
    assert native['page_00003.png'].startswith('BBB page 1')
    manifest = json.loads((book / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['source_pdfs'] == [{'file': 'a.pdf', 'pdf_pages': 2, 'first_image': 1, 'last_image': 2},
                                       {'file': 'b.pdf', 'pdf_pages': 3, 'first_image': 3, 'last_image': 5}]
    assert manifest['images'] == 5 and manifest['source_pdf_pages'] == 5
    assert any('2/2冊目' in m for m in msgs)


def test_import_pdfs_refuses_existing_book_and_empty_list(tmp_path):
    a = tmp_path / 'a.pdf'
    make_pdf(a, 1, 'A')
    book = tmp_path / 'book'
    import_pdfs(book, [a], lambda _: None)
    with pytest.raises(ValueError):
        import_pdfs(book, [a], lambda _: None)
    with pytest.raises(ValueError):
        import_pdfs(tmp_path / 'other', [], lambda _: None)
