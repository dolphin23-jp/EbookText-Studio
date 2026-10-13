from PIL import Image, ImageDraw

from ebooktext.core import render_markdown
from ebooktext.importer import import_images
from ebooktext.pageprep import ImportOptions, find_gutter, prepare_page, strip_system_bars
from ebooktext.textclean import clean_pages, review_reasons


def spread(with_bars=True, w=1400, h=1000):
    im = Image.new('RGB', (w, h), 'white')
    d = ImageDraw.Draw(im)
    d.rectangle((100, 150, 600, 800), fill='black')      # left page text block
    d.rectangle((800, 150, 1300, 800), fill='black')     # right page text block
    if with_bars:
        d.rectangle((0, 0, w, 40), fill=(85, 85, 85))
        d.rectangle((0, h - 40, w, h), fill=(85, 85, 85))
    return im


def test_strip_bars_removes_dark_edges_only():
    out = strip_system_bars(spread())
    assert 1000 - 2 * 50 < out.height < 1000 - 2 * 40   # both bars gone, a little margin kept
    plain = Image.new('RGB', (500, 800), 'white')
    assert strip_system_bars(plain).size == (500, 800)


def test_gutter_in_blank_band():
    x = find_gutter(spread(False))
    assert 600 < x < 800


def test_split_spread_order_and_rtl():
    left, right = prepare_page(spread())
    assert left.width + right.width == 1400 and left.height == right.height
    # the left page contains the block at x=100..600 (black pixels on its left half)
    assert left.getpixel((300, left.height // 2)) == (0, 0, 0)
    assert right.getpixel((right.width - 300, right.height // 2)) == (0, 0, 0)
    r_first, _ = prepare_page(spread(), ImportOptions(right_to_left=True))
    assert r_first.size == right.size


def test_single_page_not_split():
    pages = prepare_page(Image.new('RGB', (900, 1300), 'white'))
    assert len(pages) == 1


def test_import_images_splits_and_numbers(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    spread().save(src / 'a.png')
    Image.new('RGB', (900, 1300), 'white').save(src / 'b.png')
    n = import_images(tmp_path / 'book', src, lambda _: None)
    assert n == 3
    assert (tmp_path / 'book' / 'pages' / 'page_00003.png').is_file()


def test_clean_pages_removes_running_and_lifts_page_number():
    bodies = ['腎臓の働き', '肝臓の働き', '心臓の働き', '肺の働き']
    pages = [f'第1章-{i} 循環を読み解く\n{b}です。\nJCOPY 498-12008\n{138 + i}' for i, b in enumerate(bodies)]
    cleaned, notes, running = clean_pages(pages)
    assert 'JCOPY' not in cleaned[0] and '第1章' not in cleaned[0]
    assert '腎臓の働きです。' in cleaned[0]
    assert notes[0] == ['page: 138']
    assert running


def test_status_bar_line_removed():
    cleaned, _, _ = clean_pages(['21:58 5月28日 (木)\n本文です。'] * 3)
    assert cleaned[0] == '本文です。'


def test_review_reasons():
    assert 'ー' not in ''.join(review_reasons('血圧 118/68mmHg'))
    assert '誤読の疑い' in review_reasons('死亡率 60こ70%で')
    assert review_reasons('これは普通の文章です。') == []


def test_markdown_notes():
    md = render_markdown('t', [('page_00001.png', '本文')], {'page_00001.png': ['page: 5']})
    assert '<!-- source: page_00001.png, page: 5 -->' in md
