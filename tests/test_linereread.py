import pytesseract
from PIL import Image

from ebooktext.linereread import read_page


def _data(words):
    """words: (text, conf, block, par, line, left, top, w, h)"""
    keys = ["text", "conf", "block_num", "par_num", "line_num", "left", "top", "width", "height"]
    return {k: [w[i] for w in words] for i, k in enumerate(keys)}


def test_low_confidence_line_is_reread_and_replaced(monkeypatch):
    page = _data([("良い行", 95, 1, 1, 1, 10, 10, 100, 20),
                  ("PRE", 12, 1, 1, 2, 10, 50, 60, 20),
                  ("別段落", 90, 1, 2, 1, 10, 90, 80, 20)])
    calls = []

    def fake(image, lang=None, config=None, output_type=None):
        calls.append(config)
        if "--psm 7" in config:
            return _data([("感染性脳炎", 84, 1, 1, 1, 0, 0, 90, 20)])
        return page

    monkeypatch.setattr(pytesseract, "image_to_data", fake)
    text, stats, _ = read_page(Image.new("RGB", (200, 150), "white"), "jpn+eng", 3)
    assert text.split("\n") == ["良い行", "感染性脳炎", "", "別段落"]
    assert (stats.lines, stats.reread, stats.improved) == (3, 1, 1)
    assert calls.count("--psm 7") == 1


def test_reread_kept_only_when_clearly_better(monkeypatch):
    page = _data([("原文", 60, 1, 1, 1, 10, 10, 60, 20)])

    def fake(image, lang=None, config=None, output_type=None):
        return _data([("再読", 61, 1, 1, 1, 0, 0, 60, 20)]) if "--psm 7" in config else page

    monkeypatch.setattr(pytesseract, "image_to_data", fake)
    text, stats, _ = read_page(Image.new("RGB", (100, 60), "white"), "jpn+eng", 3)
    assert text == "原文" and stats.improved == 0


def test_reread_can_be_disabled(monkeypatch):
    page = _data([("低信頼", 10, 1, 1, 1, 10, 10, 60, 20)])
    monkeypatch.setattr(pytesseract, "image_to_data", lambda *a, **k: page)
    text, stats, _ = read_page(Image.new("RGB", (100, 60), "white"), "eng", 3, reread=False)
    assert text == "低信頼" and stats.reread == 0
