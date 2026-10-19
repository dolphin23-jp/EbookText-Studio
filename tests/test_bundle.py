"""One-file Markdown from an existing output folder (bundle.py)."""
from pathlib import Path

from PIL import Image

from ebooktext.bundle import bundle_book, bundle_text

SAMPLE = """# 腎臓の本

<!-- 自動OCRによる構造化Markdownです。 -->

<!-- source: page_00001.png -->

## 腎臓

本文の段落です。

**表17 KDIGO基準**

| 病期 | 尿量 |
| --- | --- |
| Stage1 | 0.5mL/kg/h未満 |

<sub>[表の原画像](tables/page_00001_表1.png) / [CSV](tables/page_00001_表1.csv)</sub>

**図1 概念モデル**

![図1 概念モデル](figures/page_00001_図1.png)

<details><summary>図中の文字</summary>

- トリガー
- 基質

</details>

<!-- p.56 ここまで -->

<!-- 改ページ -->

次の段落。
"""


def make_output(tmp_path: Path) -> Path:
    out = tmp_path / "腎臓の本" / "output"
    (out / "figures").mkdir(parents=True)
    Image.new("RGB", (1600, 400), (250, 210, 200)).save(out / "figures" / "page_00001_図1.png")
    (out / "全文.md").write_text(SAMPLE, encoding="utf-8")
    return out


def test_text_mode_keeps_only_text(tmp_path):
    text, images, missing = bundle_text(SAMPLE, tmp_path, "text")
    assert images == 0 and not missing
    assert "figures/" not in text and "tables/" not in text and "<details>" not in text
    assert "<!-- source" not in text and "改ページ" not in text
    assert "**図1 概念モデル**" in text and "図中の文字:\n- トリガー\n- 基質" in text
    assert "| Stage1 | 0.5mL/kg/h未満 |" in text
    assert "〔p.56 ここまで〕" in text


def test_bundle_book_writes_title_file_and_embeds_images(tmp_path):
    out = make_output(tmp_path)
    res = bundle_book(out, "text")
    assert res.path == out / "腎臓の本.md"
    plain = res.path.read_text(encoding="utf-8")
    res2 = bundle_book(out, "embed")
    assert res2.path == out / "腎臓の本_画像埋め込み.md" and res2.images == 1
    embedded = res2.path.read_text(encoding="utf-8")
    assert "![図1 概念モデル](data:image/" in embedded
    assert len(embedded) > len(plain)
    # The original files stay untouched.
    assert (out / "全文.md").read_text(encoding="utf-8") == SAMPLE


def test_missing_image_is_reported(tmp_path):
    out = make_output(tmp_path)
    (out / "figures" / "page_00001_図1.png").unlink()
    res = bundle_book(out, "embed")
    assert res.images == 0 and res.missing == ["figures/page_00001_図1.png"]
