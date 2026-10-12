"""Japanese Windows GUI for ebook page capture and local OCR."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ebooktext import __version__
from ebooktext.capture import CaptureOptions, KEYS, run_capture
from ebooktext.core import project_dir
from ebooktext.importer import import_images, import_pdf
from ebooktext.ocr import OCROptions, convert_book
from ebooktext.pageprep import ImportOptions

NAVY = "#183153"
BLUE = "#2c5f9a"
WHITE = "#ffffff"
SUBTLE = "#e8eff7"
DEFAULT_DIR = Path.home() / "Documents" / "EbookTextStudio"


def set_windows_dpi_awareness():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError, ValueError):
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except (AttributeError, OSError):
                pass


class RegionSelector:
    """Select a rectangle from a frozen full-size screenshot of the primary monitor."""

    def __init__(self, root: tk.Tk):
        import mss
        from PIL import Image, ImageTk

        self.result: tuple[int, int, int, int] | None = None
        root.withdraw()
        root.update_idletasks()
        root.after(220)
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                im = sct.grab(monitor)
                shot = Image.frombytes("RGB", im.size, im.rgb)
        except Exception:
            root.deiconify()
            raise
        self.monitor = monitor
        self.window = tk.Toplevel(root)
        self.window.title("キャプチャ範囲選択")
        self.window.attributes("-topmost", True)
        self.window.geometry(f"{shot.width}x{shot.height}+0+0")
        self.window.overrideredirect(True)
        self.photo = ImageTk.PhotoImage(shot)
        self.canvas = tk.Canvas(self.window, width=shot.width, height=shot.height,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.create_rectangle(8, 8, 735, 54, fill=NAVY, outline=NAVY)
        self.canvas.create_text(18, 31, anchor="w", fill="white",
                                text="ドラッグで撮影領域を囲む → マウスを離すと確定 / Escでキャンセル",
                                font=("Yu Gothic UI", 12, "bold"))
        self.start = None
        self.rect_id = None
        self.canvas.bind("<Button-1>", self.down)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.up)
        self.window.bind("<Escape>", lambda e: self.cancel())
        self.window.focus_force()
        root.wait_window(self.window)
        root.deiconify()

    def down(self, ev):
        self.start = (ev.x, ev.y)
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(ev.x, ev.y, ev.x, ev.y,
                                                    outline="#ef4a42", width=3)

    def drag(self, ev):
        if self.start and self.rect_id is not None:
            self.canvas.coords(self.rect_id, *self.start, ev.x, ev.y)

    def up(self, ev):
        if not self.start:
            return
        x0, x1 = sorted((self.start[0], ev.x))
        y0, y1 = sorted((self.start[1], ev.y))
        if x1 - x0 >= 100 and y1 - y0 >= 100:
            self.result = (self.monitor["left"] + x0, self.monitor["top"] + y0,
                           x1 - x0, y1 - y0)
            self.window.destroy()
        else:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
            self.start = None

    def cancel(self):
        self.window.destroy()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"EbookText Studio {__version__} — 電子書籍の文章化")
        self.geometry("890x790")
        self.minsize(780, 700)
        self.configure(bg="#f2f6fb")
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self.busy = False
        self.stop_event = threading.Event()
        self.events: queue.Queue = queue.Queue()
        self.region = None

        sty = ttk.Style(self)
        if "vista" in sty.theme_names():
            sty.theme_use("vista")
        sty.configure("TFrame", background="#f2f6fb")
        sty.configure("TLabel", background="#f2f6fb", font=("Yu Gothic UI", 10))
        sty.configure("TLabelframe", background="#f2f6fb")
        sty.configure("TLabelframe.Label", foreground=NAVY, font=("Yu Gothic UI", 11, "bold"))
        sty.configure("Accent.TButton", font=("Yu Gothic UI", 10, "bold"), padding=(12, 7))
        sty.configure("TButton", padding=(8, 5))

        self.title_var = tk.StringVar(value="医学書サンプル")
        self.dir_var = tk.StringVar(value=str(DEFAULT_DIR))
        self.region_var = tk.StringVar(value="未選択")
        self.pages_var = tk.StringVar(value="30")
        self.delay_var = tk.StringVar(value="1.5")
        self.countdown_var = tk.StringVar(value="5")
        self.key_var = tk.StringVar(value=next(iter(KEYS)))
        self.format_var = tk.StringVar(value="PNG")
        self.engine_var = tk.StringVar(value="Tesseract")
        self.lang_var = tk.StringVar(value="jpn+eng")
        self.psm_var = tk.StringVar(value="3 (自動)")
        self.ndl_var = tk.StringVar(value="ndlocr-lite")
        self.trim_var = tk.BooleanVar(value=False)
        self.strip_bars_var = tk.BooleanVar(value=True)
        self.split_var = tk.BooleanVar(value=True)
        self.rtl_var = tk.BooleanVar(value=False)
        self.clean_var = tk.BooleanVar(value=True)
        self.native_var = tk.BooleanVar(value=True)
        self.preprocess_var = tk.BooleanVar(value=True)
        self.upscale_var = tk.BooleanVar(value=True)
        self.state_var = tk.StringVar(value="準備完了。まず画像のキャプチャ領域を選択してください。")
        self._build()
        self.after(130, self.process_events)

    def _build(self):
        header = tk.Frame(self, bg=NAVY, height=74)
        header.pack(fill="x")
        tk.Label(header, text="EbookText Studio", fg=WHITE, bg=NAVY,
                 font=("Yu Gothic UI", 20, "bold")).pack(side="left", padx=22, pady=15)
        tk.Label(header, text="画面キャプチャ  →  OCR  →  Markdown / PDF", fg="#cbd9ee", bg=NAVY,
                 font=("Yu Gothic UI", 10)).pack(side="left", padx=15)

        container = ttk.Frame(self, padding=(16, 12))
        container.pack(fill="both", expand=True)
        project = ttk.LabelFrame(container, text="① 保存先・書籍名", padding=10)
        project.pack(fill="x", pady=(0, 11))
        project.columnconfigure(1, weight=1)
        ttk.Label(project, text="書籍名").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(project, textvariable=self.title_var, width=48).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(project, text="保存先").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(project, textvariable=self.dir_var).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(project, text="参照…", command=self.pick_dir).grid(row=1, column=2, padx=6)
        ttk.Button(project, text="フォルダを開く", command=self.open_project).grid(row=0, column=2, padx=6)

        source = ttk.LabelFrame(container, text="② 画像を用意する（いずれか1つ）", padding=10)
        source.pack(fill="x", pady=(0, 11))
        line = ttk.Frame(source)
        line.pack(fill="x")
        self.region_btn = ttk.Button(line, text="画面で撮影範囲を選択", command=self.pick_region)
        self.region_btn.pack(side="left", padx=(0, 8))
        ttk.Label(line, textvariable=self.region_var).pack(side="left")
        opts = ttk.Frame(source)
        opts.pack(fill="x", pady=(9, 6))
        for label, var, width, options in [
            ("最大ページ数", self.pages_var, 6, None),
            ("待機秒数", self.delay_var, 6, None),
            ("開始猶予", self.countdown_var, 5, None),
        ]:
            ttk.Label(opts, text=label).pack(side="left", padx=(2, 5))
            ttk.Entry(opts, textvariable=var, width=width).pack(side="left", padx=(0, 12))
        ttk.Label(opts, text="ページ送り").pack(side="left", padx=(0, 5))
        ttk.Combobox(opts, textvariable=self.key_var, values=list(KEYS), state="readonly", width=17).pack(side="left")
        ttk.Label(opts, text="形式").pack(side="left", padx=(12, 5))
        ttk.Combobox(opts, textvariable=self.format_var, values=["PNG", "JPEG"], state="readonly", width=6).pack(side="left")
        actions = ttk.Frame(source)
        actions.pack(fill="x", pady=(3, 0))
        self.start_btn = ttk.Button(actions, text="▶ 自動キャプチャ開始", command=self.start_capture, style="Accent.TButton")
        self.start_btn.pack(side="left", padx=(0, 7))
        self.images_btn = ttk.Button(actions, text="画像フォルダを取り込む", command=self.pick_images)
        self.images_btn.pack(side="left", padx=7)
        prep = ttk.Frame(source)
        prep.pack(fill="x", pady=(6, 0))
        ttk.Label(prep, text="取り込み時の整形").pack(side="left", padx=(0, 8))
        ttk.Checkbutton(prep, text="上下のステータスバーを除去", variable=self.strip_bars_var).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(prep, text="見開きを左右2ページに分割", variable=self.split_var).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(prep, text="右開き（縦書き本）", variable=self.rtl_var).pack(side="left")
        self.pdf_btn = ttk.Button(actions, text="PDFを取り込む", command=self.pick_pdf)
        self.pdf_btn.pack(side="left", padx=7)
        ttk.Label(source, text="※ 撮影中は本を最前面にしてください。F8で停止。同じ画面が3回続くと終了。既存の画像には追記します。",
                  foreground="#455a77").pack(anchor="w", pady=(7, 0))

        ocr = ttk.LabelFrame(container, text="③ OCRして文章を作る", padding=10)
        ocr.pack(fill="x", pady=(0, 11))
        engine = ttk.Frame(ocr)
        engine.pack(fill="x")
        ttk.Label(engine, text="OCRエンジン").pack(side="left", padx=(0, 6))
        ttk.Combobox(engine, textvariable=self.engine_var, state="readonly", width=17,
                     values=["Tesseract", "NDLOCR-Lite"]).pack(side="left", padx=(0, 18))
        ttk.Label(engine, text="言語").pack(side="left", padx=(0, 6))
        ttk.Entry(engine, textvariable=self.lang_var, width=12).pack(side="left", padx=(0, 18))
        ttk.Label(engine, text="文字レイアウト").pack(side="left", padx=(0, 6))
        ttk.Combobox(engine, textvariable=self.psm_var, state="readonly", width=12,
                     values=["3 (自動)", "4 (単段)", "6 (ブロック)", "11 (散文)", "12 (散文/方向)"]).pack(side="left")
        ndl_line = ttk.Frame(ocr)
        ndl_line.pack(fill="x", pady=7)
        ttk.Label(ndl_line, text="NDLOCR-Liteコマンド / src/ocr.py パス").pack(side="left", padx=(0, 8))
        ttk.Entry(ndl_line, textvariable=self.ndl_var, width=55).pack(side="left", fill="x", expand=True)
        flags = ttk.Frame(ocr)
        flags.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(flags, text="画質補正", variable=self.preprocess_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(flags, text="小さい画像を拡大", variable=self.upscale_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(flags, text="白余白を自動トリミング", variable=self.trim_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(flags, text="PDFの既存テキスト層を優先", variable=self.native_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(flags, text="柱・ページ番号・版権表記を本文から外す", variable=self.clean_var).pack(side="left")
        run = ttk.Frame(ocr)
        run.pack(fill="x")
        self.ocr_btn = ttk.Button(run, text="▶ OCR / Markdown / PDF を生成", command=self.start_ocr,
                                  style="Accent.TButton")
        self.ocr_btn.pack(side="left", padx=(0, 10))
        self.stop_btn = ttk.Button(run, text="■ 停止", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 10))
        ttk.Label(run, text="出力: 全文.md、全文.txt、画像PDF.pdf、検索可能PDF.pdf",
                  foreground="#455a77").pack(side="left")

        logs = ttk.LabelFrame(container, text="実行状況", padding=8)
        logs.pack(fill="both", expand=True)
        ttk.Label(logs, textvariable=self.state_var, foreground=NAVY).pack(anchor="w", pady=(0, 5))
        self.log_text = tk.Text(logs, height=7, wrap="word", state="disabled",
                                font=("Consolas", 9), bg="white", fg="#1c334c")
        self.log_text.pack(fill="both", expand=True)
        ttk.Label(container, text="ローカル実行のみ・DRM解除機能なし。日本語・医学用語のOCR結果は必ず原本照合してください。",
                  foreground="#64748b").pack(anchor="w", pady=(8, 0))

    def pick_dir(self):
        v = filedialog.askdirectory(title="保存先の親フォルダを選択", initialdir=self.dir_var.get())
        if v:
            self.dir_var.set(v)

    def book(self) -> Path:
        if not self.title_var.get().strip():
            raise ValueError("書籍名を入力してください。")
        return project_dir(Path(self.dir_var.get()), self.title_var.get())

    def open_project(self):
        try:
            book = self.book()
            book.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(str(book))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(book)])
            else:
                subprocess.Popen(["xdg-open", str(book)])
        except Exception as exc:
            messagebox.showerror("フォルダ", str(exc))

    def pick_region(self):
        if self.busy:
            return
        try:
            selection = RegionSelector(self).result
            if selection:
                self.region = selection
                self.region_var.set(f"x={selection[0]}, y={selection[1]}, {selection[2]} × {selection[3]} px")
        except Exception as exc:
            messagebox.showerror("範囲選択失敗", str(exc))

    def emit(self, text):
        self.events.put(("status", text))

    def set_busy(self, busy: bool):
        self.busy = busy
        enable = "disabled" if busy else "normal"
        for btn in (self.region_btn, self.start_btn, self.images_btn, self.pdf_btn, self.ocr_btn):
            btn.configure(state=enable)
        self.stop_btn.configure(state="normal" if busy else "disabled")

    def launch(self, label: str, worker, minimize: bool = False):
        if self.busy:
            return
        self.stop_event = threading.Event()
        self.set_busy(True)
        self.emit(label)
        if minimize:
            self.iconify()

        def do_work():
            try:
                count = worker()
                self.events.put(("done", f"完了: {count} ページ処理しました。"))
            except InterruptedError as exc:
                self.events.put(("done", str(exc)))
            except Exception as exc:
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))
        threading.Thread(target=do_work, name="EbookText-worker", daemon=True).start()

    def start_capture(self):
        if not self.region:
            messagebox.showwarning("範囲が未指定", "まず「画面で撮影範囲を選択」を押してください。")
            return
        try:
            pages = int(self.pages_var.get())
            delay = float(self.delay_var.get())
            countdown = int(self.countdown_var.get())
            if not (1 <= pages <= 10000 and 0.2 <= delay <= 60 and 2 <= countdown <= 60):
                raise ValueError()
            opts = CaptureOptions(region=self.region, max_pages=pages, delay=delay, countdown=countdown,
                                  key_name=self.key_var.get(), image_format=self.format_var.get())
            book = self.book()
        except (ValueError, OSError):
            messagebox.showerror("設定", "最大ページ数(1～10000)、待機秒数(0.2～60)、開始猶予(2～60)を確認してください。")
            return
        self.launch("キャプチャを開始します。5秒以内に電子書籍を前面に出してください。",
                    lambda: run_capture(book, opts, self.stop_event, self.emit), minimize=True)

    def import_options(self) -> ImportOptions:
        return ImportOptions(strip_bars=self.strip_bars_var.get(), split_spreads=self.split_var.get(),
                             right_to_left=self.rtl_var.get())

    def pick_images(self):
        directory = filedialog.askdirectory(title="取り込む画像が入ったフォルダを選択")
        if directory:
            try:
                book = self.book()
                self.launch("画像取り込み中", lambda: import_images(book, Path(directory), self.emit, self.import_options()))
            except ValueError as exc:
                messagebox.showerror("書籍名", str(exc))

    def pick_pdf(self):
        file = filedialog.askopenfilename(title="取り込むPDFを選択", filetypes=[("PDF", "*.pdf")])
        if file:
            try:
                book = self.book()
                self.launch("PDF取り込み中", lambda: import_pdf(book, Path(file), self.emit, opts=self.import_options()))
            except ValueError as exc:
                messagebox.showerror("書籍名", str(exc))

    def start_ocr(self):
        try:
            book = self.book()
            opts = OCROptions(engine=self.engine_var.get(), lang=self.lang_var.get(),
                              psm=int(self.psm_var.get().split()[0]),
                              trim=self.trim_var.get(), preprocess=self.preprocess_var.get(),
                              upscale=self.upscale_var.get(), prefer_native_pdf=self.native_var.get(),
                              ndl_command=self.ndl_var.get(), clean_text=self.clean_var.get())
        except (ValueError, OSError) as exc:
            messagebox.showerror("設定", str(exc))
            return
        self.launch("OCR処理を開始します。", lambda: convert_book(book, self.title_var.get(), opts,
                                                          self.stop_event, self.emit))

    def stop(self):
        self.stop_event.set()
        self.emit("停止を要求しました。処理中のページが終わると停止します。")

    def process_events(self):
        try:
            while True:
                what, msg = self.events.get_nowait()
                self.state_var.set(msg)
                self.log_text.configure(state="normal")
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
                if what in ("done", "error"):
                    self.deiconify()
                    self.lift()
                    self.set_busy(False)
                    if what == "error":
                        messagebox.showerror("処理中のエラー", msg)
        except queue.Empty:
            pass
        self.after(130, self.process_events)

    def close_app(self):
        if self.busy:
            messagebox.showwarning("処理中", "先に停止ボタン（キャプチャ中はF8）を押して完了を確認してください。")
            return
        self.destroy()


if __name__ == "__main__":
    set_windows_dpi_awareness()
    App().mainloop()
