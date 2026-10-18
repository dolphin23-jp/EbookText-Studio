"""Structured Markdown: ruled tables, merged cells, figures, paragraphs and text fixes.

The page images are drawn with PIL and the OCR lines are given directly, so these tests do not
need an OCR engine (except the number check, which is skipped without Tesseract)."""
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ebooktext import numcheck, structure as S, tables as T, textfix
from ebooktext.layout import Line, detect_rules, gray, parse_ndl_xml, page_separators

W, H = 780, 1100


def box(text, x0, y0, x1, y1, kind="本文"):
    return Line(text, x0, y0, x1, y1, kind=kind)


def ruled_table(draw, x0, y0, xs, ys, merged_rows=()):
    """Grid rules; (r, c) in merged_rows removes the rule below row r in column c."""
    for y in ys:
        draw.line([(xs[0], y), (xs[-1], y)], fill=(90, 90, 90), width=2)
    for x in xs:
        draw.line([(x, ys[0]), (x, ys[-1])], fill=(90, 90, 90), width=2)
    for r, c in merged_rows:
        y = ys[r + 1]
        draw.line([(xs[c] + 2, y), (xs[c + 1] - 2, y)], fill="white", width=4)


def blank():
    im = Image.new("RGB", (W, H), "white")
    return im, ImageDraw.Draw(im)


# ---------------------------------------------------------------------------------------------
# Tables


def test_grid_with_merged_label_cell():
    im, d = blank()
    xs, ys = [80, 280, 600, 700], [100, 140, 180, 220, 260]
    d.rectangle([80, 100, 700, 140], fill=(140, 140, 140))      # shaded header row
    ruled_table(d, 0, 0, xs, ys, merged_rows=[(1, 0), (2, 0)])   # first column spans rows 1-3
    hs, vs = detect_rules(gray(im))
    tables = T.find_tables(hs, vs)
    assert len(tables) == 1
    t = tables[0]
    assert (t.nrows, t.ncols) == (4, 3)
    label = [c for c in t.cells if c.c0 == 0 and c.r0 == 1][0]
    assert label.r1 == 3
    lines = [box("項目", 90, 110, 140, 130), box("基準", 290, 110, 340, 130), box("点数", 610, 110, 660, 130),
             box("年齢(歳)", 90, 190, 180, 210),
             box("<50", 290, 150, 330, 170), box("1", 610, 150, 620, 170),
             box("50~70", 290, 190, 350, 210), box("2", 610, 190, 620, 210),
             box("75≦", 290, 230, 330, 250), box("3", 610, 230, 620, 250)]
    T.assign_lines(t, lines)
    T.detect_header(t, gray(im))
    assert t.header_rows == 1
    md = T.to_markdown(t)
    assert md.splitlines()[0] == "| 項目 | 基準 | 点数 |"
    # The merged label is repeated on every row it covers, so each row reads on its own.
    assert "| 年齢(歳) | <50 | 1 |" in md and "| 年齢(歳) | 75≦ | 3 |" in md
    assert 'rowspan="3"' in T.to_html(t)


def test_screen_seam_and_page_edge_are_not_rules():
    im, d = blank()
    d.line([(0, 300), (W, 300)], fill=(194, 194, 194), width=1)       # seam across the capture
    d.line([(W - 2, 0), (W - 2, H)], fill=(60, 60, 60), width=2)       # page edge
    ruled_table(d, 0, 0, [80, 300, 700], [250, 290, 330])
    hs, vs = detect_rules(gray(im))
    t = T.find_tables(hs, vs)[0]
    assert t.ys == pytest.approx([250, 290, 330], abs=2)
    assert t.xs[-1] < 710


def test_implicit_rows_split_on_score_column():
    """A ruled row whose score column lists one value per line becomes one sub-row per value."""
    im, d = blank()
    ruled_table(d, 0, 0, [80, 280, 600, 700], [100, 140, 300])
    t = T.find_tables(*detect_rules(gray(im)))[0]
    lines = [box("評価", 90, 110, 140, 130), box("内容", 290, 110, 340, 130), box("点", 610, 110, 630, 130),
             box("血小板減少の", 90, 150, 220, 170), box("タイミング", 90, 175, 200, 195),
             box("<4日", 290, 150, 340, 170), box("-2", 610, 150, 630, 170),
             box("4日", 290, 180, 330, 200), box("2", 610, 180, 620, 200),
             box("5～10日", 290, 210, 360, 230), box("3", 610, 210, 620, 230)]
    T.assign_lines(t, lines)
    rows = T.to_rows(t)
    assert ["血小板減少の<br>タイミング", "<4日", "-2"] in rows
    assert ["血小板減少の<br>タイミング", "5～10日", "3"] in rows


def test_cell_text_keeps_items_and_joins_wraps():
    lines = [box("◎ジアゼパム5~10 mg", 335, 30, 445, 50), box("ボーラス", 352, 55, 420, 75),
             box("◎ロラゼパム", 335, 80, 430, 100)]
    assert T.cell_text(lines, right=448) == "・ジアゼパム5~10 mgボーラス<br>・ロラゼパム"
    short = [box("<30%減少", 280, 70, 380, 90), box("30~50%減少", 280, 95, 390, 115)]
    assert T.cell_text(short, right=630) == "<30%減少<br>30~50%減少"


def test_table_continues_on_next_page():
    def table(n_rows, open_top=False):
        ys = [0.0] + [40.0 * (i + 1) for i in range(n_rows)]
        t = T.Table((80, 0, 700, ys[-1]), [80, 300, 700], ys, [], open_top=open_top)
        for r in range(n_rows):
            for c in range(2):
                t.cells.append(T.Cell(r, c, r, c, (0, 0, 1, 1), text=f"{r}-{c}"))
        return t
    a, b = table(2), table(3)
    a.pages, b.pages = ["p1"], ["p2"]
    pa = S.PageModel("p1", (W, H), [S.Block("table", 900, 1000, table=a, page="p1")])
    pb = S.PageModel("p2", (W, H), [S.Block("table", 10, 120, table=b, page="p2")])
    blocks = S.assemble([pa, pb], [[a], [b]])
    assert [x.kind for x in blocks] == ["table"]
    assert blocks[0].table.nrows == 5


# ---------------------------------------------------------------------------------------------
# Page analysis


def test_page_analysis_caption_table_paragraphs_and_noise():
    im, d = blank()
    ruled_table(d, 0, 0, [80, 300, 700], [300, 340, 380, 420])
    d.rectangle([0, 700, W, 712], fill=(40, 40, 40))                  # viewer page separator
    lines = [
        box("ここは本文の一行目で、右端まで文字が続いている長い行です", 100, 100, 698, 120),
        box("二行目はここで終わる。", 80, 130, 300, 150),
        box("表3 検査の基準値", 80, 260, 260, 280, kind="キャプション"),
        box("項目", 90, 310, 150, 330), box("値", 310, 310, 350, 330),
        box("Na", 90, 350, 120, 370), box("140", 310, 350, 350, 370),
        box("K", 90, 390, 110, 410), box("4.0", 310, 390, 350, 410),
        box("22", 40, 660, 60, 676),                                    # page number in the margin
        box("循環", 735, 200, 750, 260),                                 # index tab
    ]
    model, tables = S.analyze_page(im, lines, "page_00001.png")
    kinds = [b.kind for b in model.blocks]
    assert kinds == ["para", "caption", "table", "pagebreak"]
    assert model.blocks[0].text.startswith("ここは本文の一行目") and model.blocks[0].text.endswith("終わる。")
    assert tables[0].caption.startswith("表3")
    assert model.blocks[3].text == "22"
    md = S.render("本", S.assemble([model], [tables]))
    assert "**表3 検査の基準値**" in md and "| Na | 140 |" in md and "<!-- p.22 ここまで -->" in md
    assert "循環" not in md


def test_figure_with_caption_is_kept_apart_from_body():
    im, d = blank()
    d.rectangle([150, 240, 640, 300], fill=(250, 210, 200), outline=(240, 90, 40), width=3)
    d.rectangle([150, 330, 640, 520], fill=(200, 200, 200))
    d.line([(395, 300), (395, 330)], fill=(0, 0, 0), width=3)
    lines = [box("図1 病態の概念モデル", 80, 200, 330, 220, kind="キャプション"),
             box("トリガー", 340, 260, 420, 280), box("基質", 360, 400, 410, 420),
             box("図のあとに続く本文の段落で、右端まで続く長い一行になっている", 80, 600, 698, 620)]
    model, _ = S.analyze_page(im, lines, "page_00002.png")
    figs = [b for b in model.blocks if b.kind == "figure"]
    assert len(figs) == 1
    f = figs[0].figure
    assert f.caption.startswith("図1")
    assert f.groups == ["トリガー", "基質"]
    paras = [b.text for b in model.blocks if b.kind == "para"]
    assert paras == ["図のあとに続く本文の段落で、右端まで続く長い一行になっている"]


def test_paragraph_continues_over_page_separator():
    im, d = blank()
    d.rectangle([0, 500, W, 512], fill=(40, 40, 40))
    lines = [box("前のページの最後の行は文の途中で右端まで続いていて", 80, 460, 698, 480),
             box("次のページの先頭で文が終わる。", 80, 560, 400, 580)]
    model, tables = S.analyze_page(im, lines, "page_00003.png")
    blocks = S.assemble([model], [tables])
    paras = [b.text for b in blocks if b.kind == "para"]
    assert paras == ["前のページの最後の行は文の途中で右端まで続いていて次のページの先頭で文が終わる。"]


def test_ndl_layout_types_are_used():
    xml = """<OCRDATASET><PAGE IMAGENAME="p.png" WIDTH="780" HEIGHT="1100">
    <TEXTBLOCK><LINE TYPE="タイトル本文" X="200" Y="100" WIDTH="300" HEIGHT="24" ORDER="0" STRING="換気設定" /></TEXTBLOCK>
    <BLOCK TYPE="柱" X="380" Y="20" WIDTH="300" HEIGHT="15" CONF="0.9" />
    </PAGE></OCRDATASET>"""
    page = parse_ndl_xml(xml)
    assert page.lines[0].kind == "タイトル本文" and page.blocks[0].kind == "柱"
    im, _ = blank()
    lines = [box("換気設定", 200, 100, 500, 124), box("基礎編:重症管理の基本", 380, 20, 680, 35),
             box("本文が続く行で、右端まで書かれている。本文が続く行で、右端まで", 80, 160, 698, 178),
             box("書かれている。", 80, 190, 200, 208)]
    from ebooktext.layout import attach_ndl
    attach_ndl(lines, page)
    model, _ = S.analyze_page(im, lines, "p.png", page)
    assert [(b.kind, b.text[:4]) for b in model.blocks] == [("heading", "換気設定"), ("para", "本文が続")]


# ---------------------------------------------------------------------------------------------
# Text fixes and numbers


@pytest.mark.parametrize("raw,fixed", [
    ("PEEP<16cmH,Oが目安", "PEEP<16cmH2Oが目安"),
    ("実際上14～16cmH3Oくらい", "実際上14～16cmH2Oくらい"),
    ("4～8mL/kqPBW", "4～8mL/kgPBW"),
    ("PaCO,が上昇", "PaCO2が上昇"),
    ("Pplat〈30cmH2O", "Pplat<30cmH2O"),
    ("カテ一テル", "カテーテル"),
    ("力テーテル留置", "カテーテル留置"),
    ("Stagel", "Stage1"),
    ("Grade I", "Grade I"),
    ("Grade Il", "Grade II"),
    ("ＡＢＣ１２３", "ABC123"),
    ("マスク一つ", "マスク一つ"),            # kanji 一 after katakana stays when a kana follows
    ("CO,CI", "CO,CI"),                      # cardiac output, not a blood gas
])
def test_line_fixes(raw, fixed):
    assert textfix.fix_line(raw)[0] == fixed


def test_book_lexicon_fixes_rare_confusable_spelling():
    texts = ["ヘルペス脳炎"] * 6 + ["ヘルベス"] + ["ボーラス投与"] * 5 + ["ポーラス"]
    lex = textfix.build_lexicon(texts)
    assert lex == {"ヘルベス": "ヘルペス", "ポーラス": "ボーラス"}
    assert textfix.apply_lexicon("ヘルベスが疑われる", lex)[0] == "ヘルペスが疑われる"


def test_number_decision_is_conservative():
    assert numcheck.decide("00", "8", 95) == ("置換", "8")
    assert numcheck.decide("11", "-1", 90) == ("置換", "-1")
    assert numcheck.decide("4.5", "45", 95)[0] != "置換"      # never drops a decimal point
    assert numcheck.decide("28", "28=", 89) == ("要確認", "28≦")  # a ≦ seen as "=", dropped by NDL
    assert numcheck.decide("25", "2=", 85) == ("要確認", "2≦")    # ... or read as an extra digit
    assert numcheck.decide("25", "2.5", 95) == ("置換", "2.5")
    assert numcheck.decide("15", "or", 95) == ("置換", "or")
    assert numcheck.decide("12", "12", 99) == ("", "12")
    assert numcheck.decide("〈6", "6", 95) == ("置換", "<6")


@pytest.mark.skipif(not numcheck.tesseract_available(), reason="Tesseract not installed")
def test_number_reread_fixes_doubled_digit():
    from PIL import ImageFont
    font = None
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(p).is_file():
            font = ImageFont.truetype(p, 18)
    if font is None:
        pytest.skip("no test font")
    im, d = blank()
    d.text((300, 200), "8", fill="black", font=font)
    line = box("00", 299, 199, 312, 221)
    checks = numcheck.check_lines(im, [line], "p")
    assert line.text == "8" and checks[0].action == "置換"


def test_caption_badge_label_replaces_misread_prefix():
    assert S.relabel_caption("表0PEEP/FIO2 table", "表10") == "表10 PEEP/FIO2 table"
    assert S.relabel_caption("■■急性AFの概念モデル", "図1") == "図1 急性AFの概念モデル"
    assert S.relabel_caption("表25 4T's score", "表25") == "表25 4T's score"
    assert S.relabel_caption("表39 IMPROVE Risk Score", "表39") == "表39 IMPROVE Risk Score"


def test_page_separator_detection():
    im, d = blank()
    d.rectangle([0, 400, W, 412], fill=(45, 45, 45))
    assert page_separators(gray(im)) == [(400, 413)]
