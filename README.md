# EbookText Studio v0.1 — Windowsで電子書籍の文章化

Windows上に**実際に表示されているページ**を、通常のページ送り操作で撮影し、ローカルのOCRで日本語テキスト、Markdown、画像PDF、検索可能PDFを生成するツールです。Kindle専用の内部APIを利用しないので、画面を通常表示できてキーボードでページ送りできるサービスに共通して使えます。

**Kindle（Windows版またはブラウザ版）、楽天Kobo、Google Play ブックス、DMMブックス、その他のビューア**を想定しています。ただし、すべてのビューアでの動作を保証するものではありません。画面撮影を保護するビューアでは黒い画面が保存される場合があります。このソフトはDRMを解除したり、ブロックを回避したりしません。

## 目次

- [セットアップ（Windows 10/11）](#ダウンロード後のセットアップwindows-1011)
- [画面キャプチャ → OCR](#画面キャプチャ--ocr)
- [既存の画像・PDFからOCR](#既存の画像pdfからocr)
- [OCRエンジン](#ocrエンジン)
- [保存されるファイル](#保存されるファイル)
- [途中から再実行する](#途中から再実行する)
- [iPadから使う](#ipadから使う)
- [CLIでの一括OCR](#cliでの一括ocr上級者向け)
- [開発者向け](#開発者向け)
- [注意点・権利について](#注意点権利について)

## ダウンロード後のセットアップ（Windows 10/11）

1. ZIPを「すべて展開」し、`setup.bat` を実行してください。Python 3.11〜3.13が必要です。ない場合は **Python 3.12（python.org）** をインストールしてから再実行します。セットアップは専用の `.venv` を自動作成し、画面キャプチャ・OCR補助用のPythonライブラリをインストールします。
2. Tesseract OCR（**別途インストール必須**）をWindows Terminalで `winget install -e --id UB-Mannheim.TesseractOCR` などにより導入してください。インストール時に `Japanese (jpn)` と `English (eng)` の言語データを入れます。既定のインストールパスは `C:\Program Files\Tesseract-OCR\tesseract.exe`。`jpn_vert` は縦書き日本語をTesseractで読む際に便利です（言語欄に`jpn_vert+eng`）。
3. `run.bat` をダブルクリックしてGUIを起動します。

Tesseractの詳しい導入元: <https://github.com/UB-Mannheim/tesseract/wiki> 。wingetから日本語データが入らなかった場合は、対応する `jpn.traineddata` / `eng.traineddata` をインストール先の `tessdata` に入れてください。言語が欠けている場合はGUIが不足言語を表示します。

## 画面キャプチャ → OCR

1. Kindleアプリ（またはブラウザ）で対象書籍を開き、**単ページ表示・適切な拡大率**にして、取り込みたい最初のページへ移動します。
2. GUIの「書籍名」「保存先」を指定します。書籍単位でフォルダを分けます。
3. 「画面で撮影範囲を選択」をクリックし、画面上の**本文領域だけ**をドラッグして囲みます。選択が終わったら画像の上下が切れていないか確認します。
4. 「最大ページ数」を短め（まず**3〜5ページ**）にして動作テスト。ページ送りのキー（右矢印、左矢印、PageDown等）をビューアに合わせます。
5. 「自動キャプチャ開始」をクリックし、カウントダウン中に**電子書籍のウィンドウを前面に出してクリックでアクティブ化**してください。アプリは最小化されます。撮影中はキーボードを操作しないでください。
6. 最初のページを撮影し、指定間隔でページ送り・撮影を繰り返します。**F8で停止**、同一画面が3回続いても終了します。ページ途中で終える場合も途中画像は保存されます。次回は続きを表示してから「キャプチャ開始」を押すと連番で追記します。
7. アプリに戻り「OCR / Markdown / PDFを生成」をクリック。完成物は書籍フォルダの `output` 内に保存されます。OCR結果を修正して再変換する場合は `replacements.json` を編集してOCRを再実行します（置換が反映されます）。

**キャプチャの制限:** 1ページ表示推奨。本文がスクロール式・無限スクロール式のビューアでは1枚=1ページにならず、欠落や重複が発生することがあります。ページ送りキーは各ビューアで異なります。**マルチモニタではプライマリディスプレイ上**で範囲選択します。ウィンドウサイズや表示倍率は撮影途中で変えないでください。CPU速度やページの描画時間に応じ「待機秒数」を1.5〜3秒程度に調整してください。

## 既存の画像・PDFからOCR

画像が既にある場合は「画像フォルダを取り込む」。ファイル名のアルファベット順でページになります（例 `001.png`, `002.png`）。

PDFがある場合は「PDFを取り込む」。PDFは180dpiでページ画像に展開します。**PDF内に元々選択可能なテキストが40文字以上あるページは、その元テキストをMarkdown作成に優先利用**します。元PDFを完全に保存するわけではないため、必要であれば元ファイルは別途保管してください。

## OCRエンジン

### 1. Tesseract（標準、初回はこちら）

`Tesseract` を選択し、言語は `jpn+eng`、PSM=3（レイアウト自動）を推奨します。単段組ならPSM=4/6、文章の配置が変則的なら11も試せます。画像の色や文字の濃さにより「画質補正」を切り替えます。OCR前処理はテキスト抽出時のみで、検索可能PDFの画像には原スクリーンショットを用います。

**注意:** 数式、図表、縦組、医学用語は誤認識されます。検索可能PDFはOCRエンジン生成のテキストレイヤー付きですが原画像の文字と完全一致しないことがあります。

### 2. NDLOCR-Lite（日本語書籍向け・任意）

国立国会図書館の[NDLOCR-Lite公式リポジトリ](https://github.com/ndl-lab/ndlocr-lite)の**CLI版**を別途導入できます。Python環境やモデルのセットアップには公式READMEを参照し、`ndlocr-lite --version` の動作を確認してください。GUIの「OCRエンジン」でNDLOCR-Liteに切り替え、「NDLOCR-Liteコマンド」に `ndlocr-lite`、または公式ソース版の `src/ocr.py` のフルパスを指定します。`src/ocr.py` を指定した場合はEbookText Studioの `.venv` に公式の依存ライブラリとモデルがインストールされている必要があります。

NDLOCRの公式`--sourcedir`を1回実行して全ページOCRした後、生成TXTとJSON座標からMarkdownと検索可能PDFを作成します。テキストレイヤーはおおむね文字付近に配置しますが、特に縦書きの位置合わせは十分検証されていません。**NDLOCRエンジン本体はこのZIPに同梱していません。**

## 保存されるファイル

```text
Documents/EbookTextStudio/書籍名/
 ├─ pages/page_00001.png ...       # 原キャプチャ（再処理に使う）
 ├─ manifest.json                  # 撮影条件と処理履歴
 ├─ native_text.json               # PDF由来なら元のテキスト
 ├─ replacements.json              # 任意の修正辞書（手編集）
 ├─ ocr_cache/                     # 認識結果とPDF作業用（再開可能）
 └─ output/
     ├─ 全文.md                     # AIで読むのに便利
     ├─ 全文.txt                    # プレーンテキスト
     ├─ 画像PDF.pdf                 # 原画像から生成、画像のみ
     ├─ 検索可能PDF.pdf             # 原画像+OCR検索用テキスト層
     └─ pages/page_00001.txt ...   # ページ別テキスト
```

`replacements.json` は最初の実行時に `{}` として作成されます。必要なら `{"CKDl": "CKD1"}` のような辞書を**自分で追加**してください。誤った修正も生じうるため、医療用語の一括置換には慎重になってください。

## 途中から再実行する

- 撮影: 同じ書名のフォルダに追記されます。再開位置を**利用者がビューアで指定**してください。前の最後のページを再び写した場合は同一画像判定で重複防止を試みます。
- OCR: ページのハッシュと認識設定によるキャッシュが使われます。**エンジンやOCR設定を変えたときは再OCR**します。医学用語の置換辞書の編集だけならテキスト表示に反映され、OCR本体の再実行は不要です。
- OCR停止はページ単位（NDLOCRバッチの場合はプロセスを終了）。途中まで作成したキャッシュは残ります。作成途中のPDFは`.partial.pdf`として残る場合があり、完成PDFと区別します。

## iPadから使う

Windowsの保存先をiCloud Drive/OneDrive等に設定し、iPadから `output/全文.md` と `検索可能PDF.pdf` を読む構成が手軽です。クラウドAIに本を丸ごとアップロードする場合には著作権・電子書籍の契約条件・プライバシーに注意してください。

## CLIでの一括OCR（上級者向け）

画面撮影後の**画像フォルダやPDFが複数ある場合**は、コマンドで順番に処理できます。Kindleビューアから複数冊を自動選択する機能ではありません。

```powershell
# 画像フォルダ1冊を取り込み、OCRする
.venv\Scripts\python.exe cli.py one --input "C:\Books\my_pages" --base "C:\EbookText" --title "腎臓内科学"

# C:\Books\incoming の「サブフォルダ」または「PDF」を1冊ずつ処理
.venv\Scripts\python.exe cli.py batch --input "C:\Books\incoming" --base "C:\EbookText"
```

すでに変換済みの書籍は、キャッシュを使って再処理を最小限に抑えます。`--engine NDLOCR-Lite --ndl-command ndlocr-lite` で別のOCRエンジンを指定することもできます。

## 開発者向け

### ディレクトリ構成

```text
app.py              # GUI（tkinter）
cli.py              # CLI（one / batch）
ebooktext/
 ├─ capture.py      # 画面撮影・ページ送り・重複判定
 ├─ ocr.py          # Tesseract / NDLOCR-Lite、Markdown・PDF生成
 ├─ importer.py     # 画像フォルダ・PDFの取り込み
 └─ core.py         # 共通処理（ファイル名、画像比較、置換辞書など）
tests/              # pytest
```

### テストの実行

画面撮影とGUIはWindows専用ですが、OCR・取り込み・PDF生成のテストはLinux/macOSでも実行できます。OCRのテストにはTesseract本体と英語データ（`eng`）が必要です。

```bash
# Ubuntu/Debianの例
sudo apt-get install -y tesseract-ocr tesseract-ocr-jpn

python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q tests
```

テストフォント（DejaVu Sans）が `/usr/share/fonts/truetype/dejavu/` にない環境では、`tests/test_ocr.py` のフォントパスを調整してください。

## 注意点・権利について

**閲覧権限のあるコンテンツの、法令・利用規約で許可される範囲でのみ使用してください。** 著作権や電子書籍サービスの利用規約に従い、第三者への再配布や公開は行わないでください。利用者が見ることのできる画面の通常撮影・通常キー入力しか行わず、DRM解除や保護機能の回避には対応していません。

### 技術上の制約

- Kindleを含む各ビューア固有の動作はWindows実機で未検証です。キーボード操作や画面保護により動作しない場合があります。
- OCRの表構造再現、医学書の図表/数式の正確なMarkdown変換、章見出し検出は**未実装**です。MarkdownはページごとにOCR本文を連結し、元画像名をコメントとして残す形式です。
- PDFのテキストレイヤーは検索補助用で、特にNDLOCRの日本語/縦組では文字位置の厳密な保証がありません。
- 全処理はローカルで完結します。インストール時のpipや任意のNDLOCRセットアップだけがインターネットを使います。

### ソースコード

主な処理は`ebooktext/capture.py`、`ebooktext/ocr.py`、`ebooktext/importer.py`に分割しています。すべてPythonの通常のスクリプトで、利用者が自由に変更可能です。
