"""Deterministic OCR corrections for Japanese medical books, every change logged.

Two kinds of fixes:

* Line rules: patterns that are almost always OCR confusions in this kind of text — a kanji
  "一" or a hyphen between katakana (カテ一テル → カテーテル), kanji look-alikes inside a katakana
  word (力/カ, 口/ロ, 工/エ, 夕/タ, 二/ニ, 八/ハ, 卜/ト), a lost subscript in blood-gas notation
  (cmH,O → cmH2O, FIO, → FIO2, PaCO, → PaCO2), unit typos after a number (mL/kq → mL/kg) and
  a CJK angle bracket used as a comparison sign (〈30 → <30).
* Book lexicon: a katakana word that occurs once or twice while a word differing in a single
  confusable character (ベ/ペ, バ/パ, ソ/ン, シ/ツ, dakuten lost ...) occurs many times in the same
  book is replaced by the frequent form (ヘルベス → ヘルペス when ヘルペス is common). The book is
  its own dictionary; an optional glossary file adds terms the user knows are right.

Nothing here touches digits themselves. Every change is returned so it can be reviewed.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

KATA = "ァ-ヴヵヶー"


@dataclass
class Fix:
    page: str
    old: str
    new: str
    rule: str
    context: str = ""


_LOOKALIKE = {"力": "カ", "口": "ロ", "工": "エ", "夕": "タ", "二": "ニ", "八": "ハ", "卜": "ト"}

_RULES: list[tuple[str, re.Pattern, object]] = [
    # Long vowel mark read as kanji one / hyphen / dash between katakana.
    ("長音符", re.compile(rf"(?<=[{KATA[:-1]}])[一‐－―−\-](?=[{KATA}])"), "ー"),
    # Kanji look-alikes inside a katakana word.
    ("カタカナの誤読", re.compile(rf"(?<=[{KATA}])[力口工夕二八卜](?=[{KATA}])"), None),
    ("カタカナの誤読", re.compile(rf"(?<![{KATA}一-鿿])[力口工夕八卜](?=[{KATA}]{{2,}})"), None),
    # Blood gas / respiratory notation with the subscript read as a comma, period or 3.
    ("下付き数字", re.compile(r"(?<![A-Za-z])((?:cm|mm)?\s?)H[,，.．、3₂]O(?![A-Za-z0-9])"), r"\1H2O"),
    ("下付き数字", re.compile(r"\b(Pa|Pv|PA|Et|ET|Pet|PET|P)CO[,，、₂](?![0-9])"), r"\1CO2"),
    ("下付き数字", re.compile(r"\b(Pa|Sp|Sa|Scv|Sv|Fi|FI|PA|Pv|Sc|SCV|D|V)O[,，、₂](?![0-9])"), r"\1O2"),
    ("下付き数字", re.compile(r"\bHCO[,，、₃](?![A-Za-z0-9])"), "HCO3"),
    # Latin l / I read for the digit 1 after a staging word.
    ("英数字の誤読", re.compile(r"\b(Stage|stage|Grade|grade|Class|class|Type|type|Phase|phase)[l|](?![A-Za-z0-9])"), r"\g<1>1"),
    # Roman numerals after a grading word: l / | read for I, o for 0.
    ("英数字の誤読", re.compile(r"\b(Grade|Stage|Class|Type|Phase|grade|stage|class|type)\s?([IlV|]*[l|][IlV|]*)(?![A-Za-z])"),
     lambda m: m.group(1) + " " + m.group(2).replace("l", "I").replace("|", "I")),
    ("英数字の誤読", re.compile(r"\b(Grade|Stage|grade|stage)o(?![A-Za-z])"), r"\g<1>0"),
    # Units right after a number or slash.
    ("単位", re.compile(r"(?<=[\d/])\s?kq(?![a-z])"), "kg"),
    ("単位", re.compile(r"(?<=[\d/])\s?mq(?![a-z])"), "mg"),
    ("単位", re.compile(r"(?<=[\d/])\s?[μµu]q(?![a-z])"), "μg"),
    ("単位", re.compile(r"mEg(?![A-Za-z])"), "mEq"),
    ("単位", re.compile(r"mmHq(?![A-Za-z])"), "mmHg"),
    # A comparison sign read as a CJK angle bracket (only when no matching bracket follows).
    ("不等号", re.compile(r"〈(?=\s?\d)(?![^〈〉]*〉)"), "<"),
    ("不等号", re.compile(r"^([^〈]*?)〉(?=\s?\d)"), r"\1>"),
]


def _halfwidth_alnum(text: str) -> str:
    return re.sub(r"[Ａ-Ｚａ-ｚ０-９]", lambda m: unicodedata.normalize("NFKC", m.group()), text)


def fix_line(text: str, page: str = "") -> tuple[str, list[Fix]]:
    fixes: list[Fix] = []
    new = _halfwidth_alnum(text)
    for rule, pattern, repl in _RULES:
        def sub(m: re.Match) -> str:
            old = m.group(0)
            if repl is None:
                rep = _LOOKALIKE.get(old, old)
            elif callable(repl):
                rep = repl(m)
            else:
                rep = m.expand(repl)
            if rep != old:
                fixes.append(Fix(page, old, rep, rule, text))
            return rep
        new = pattern.sub(sub, new)
    return new, fixes


# ---------------------------------------------------------------------------------------------
# Book lexicon

_PAIRS = [
    "ベペ", "バパ", "ビピ", "ブプ", "ボポ", "ハバパ", "ヒビピ", "フブプ", "ヘベペ", "ホボポ",
    "カガ", "キギ", "クグ", "ケゲ", "コゴ", "サザ", "シジ", "スズ", "セゼ", "ソゾ", "タダ", "チヂ",
    "ツヅ", "テデ", "トド", "シツ", "ソン", "ッツ", "ャヤ", "ュユ", "ョヨ", "ァア", "ィイ", "ゥウ",
    "ェエ", "ォオ", "クタ", "ウワ", "ロコ", "ヲヨ", "ルレ", "ミシ",
]
_CONFUSE: dict[str, set[str]] = {}
for group in _PAIRS:
    for a in group:
        _CONFUSE.setdefault(a, set()).update(set(group) - {a})

_KATA_WORD = re.compile(rf"[{KATA}]{{3,}}")


def load_glossary(path: Path) -> list[str]:
    """One correct term per line; lines starting with # are comments."""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def build_lexicon(texts: list[str], glossary: list[str] | None = None,
                  min_common: int = 4, ratio: float = 5.0) -> dict[str, str]:
    """rare word -> frequent word, for katakana words that differ in one confusable character."""
    counts: Counter[str] = Counter()
    for t in texts:
        counts.update(_KATA_WORD.findall(t))
    for g in glossary or []:
        for w in _KATA_WORD.findall(g):
            counts[w] += 10_000
    by_len: dict[int, list[str]] = {}
    for w, n in counts.items():
        if n >= min_common:
            by_len.setdefault(len(w), []).append(w)
    out: dict[str, str] = {}
    for w, n in counts.items():
        if n > 2 or len(w) < 4:
            continue
        best = None
        for v in by_len.get(len(w), []):
            if v == w or counts[v] < ratio * n:
                continue
            diff = [(a, b) for a, b in zip(w, v) if a != b]
            if len(diff) == 1 and diff[0][1] in _CONFUSE.get(diff[0][0], ()):
                if best is None or counts[v] > counts[best]:
                    best = v
        if best:
            out[w] = best
    return out


def apply_lexicon(text: str, lexicon: dict[str, str], page: str = "") -> tuple[str, list[Fix]]:
    if not lexicon:
        return text, []
    fixes: list[Fix] = []

    def sub(m: re.Match) -> str:
        w = m.group(0)
        rep = lexicon.get(w)
        if rep:
            fixes.append(Fix(page, w, rep, "書籍内の表記統一", text))
            return rep
        return w
    return _KATA_WORD.sub(sub, text), fixes


def render_fix_log(title: str, fixes: list[Fix]) -> str:
    out = [f"# {title} 自動補正ログ", "",
           "構造化Markdownを作る際に自動で直した箇所です。誤った補正があれば `replacements.json` で上書きできます。",
           "数字そのものは変更していません（表の数値の読み直しは別の節にあります）。", ""]
    by_rule: dict[str, list[Fix]] = {}
    for f in fixes:
        by_rule.setdefault(f.rule, []).append(f)
    for rule, items in by_rule.items():
        pairs = Counter((f.old, f.new) for f in items)
        out += [f"## {rule}（{len(items)}件）", ""]
        for (old, new), n in pairs.most_common():
            pages = sorted({f.page for f in items if f.old == old and f.new == new})
            where = ", ".join(pages[:5]) + (" ほか" if len(pages) > 5 else "")
            out.append(f"- `{old}` → `{new}` ×{n}（{where}）")
        out.append("")
    if not fixes:
        out.append("補正はありませんでした。")
    return "\n".join(out).rstrip() + "\n"
