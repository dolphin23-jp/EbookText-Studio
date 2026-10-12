"""Capture the visible reading area, using ordinary page-turn keystrokes only.

No DRM decryption, protected-file parsing, or evasion of blocked capture.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from .core import looks_black, nearly_identical, page_images, save_manifest


KEYS = {
    "→ (次ページ)": "right",
    "← (前ページ)": "left",
    "PageDown": "page_down",
    "PageUp": "page_up",
    "↓ (下矢印)": "down",
    "Space": "space",
    "Enter": "enter",
}


@dataclass
class CaptureOptions:
    region: tuple[int, int, int, int]  # x, y, width, height
    max_pages: int = 50
    delay: float = 1.4
    countdown: int = 5
    key_name: str = "→ (次ページ)"
    stop_duplicates: int = 3
    image_format: str = "PNG"


def run_capture(book: Path, opts: CaptureOptions, stop: threading.Event,
                status: Callable[[str], None]) -> int:
    try:
        import mss
        from pynput.keyboard import Controller, Key, Listener
    except ImportError as exc:
        raise RuntimeError("必要なパッケージがありません。setup.bat を実行してください。") from exc

    x, y, w, h = opts.region
    if w < 100 or h < 100:
        raise ValueError("キャプチャ領域が小さすぎます。")
    if opts.key_name not in KEYS:
        raise ValueError("ページ送りキーが不明です。")
    book = Path(book)
    image_dir = book / "pages"
    image_dir.mkdir(parents=True, exist_ok=True)
    current = page_images(image_dir)
    # Never overwrite a previously captured page when resuming a project.
    max_num = max((int(p.stem.split("_")[-1]) for p in current
                   if p.stem.startswith("page_") and p.stem.split("_")[-1].isdigit()), default=0)
    key = getattr(Key, KEYS[opts.key_name])
    controller = Controller()
    duplicate = 0
    captured = 0
    prev = None
    if current:
        with Image.open(current[-1]) as old:
            prev = old.convert("RGB")

    def on_press(pressed):
        if pressed == Key.f8:
            stop.set()
            status("F8で停止しました。")
            return False
        return None

    listener = Listener(on_press=on_press)
    listener.start()
    try:
        for i in range(opts.countdown, 0, -1):
            if stop.is_set():
                return 0
            status(f"{i} 秒後に撮影。電子書籍を前面に出してください（F8で停止）")
            if stop.wait(1):
                return 0
        with mss.mss() as sct:
            for i in range(opts.max_pages):
                if stop.is_set():
                    break
                if captured or i:  # after first attempted capture, turn the page
                    controller.press(key)
                    controller.release(key)
                    if stop.wait(opts.delay):
                        break
                s = sct.grab({"left": x, "top": y, "width": w, "height": h})
                im = Image.frombytes("RGB", s.size, s.rgb)
                if looks_black(im):
                    status("ほぼ真っ黒な画像を検出。画面保護・領域設定を確認して終了します。")
                    break
                if prev is not None and nearly_identical(prev, im):
                    duplicate += 1
                    status(f"同一ページ候補 {duplicate}/{opts.stop_duplicates} 回（撮影 {captured} 枚）")
                    if duplicate >= opts.stop_duplicates:
                        status("同一画面が続いたため終了しました。")
                        break
                    continue
                duplicate = 0
                max_num += 1
                dst = image_dir / (f"page_{max_num:05d}.jpg" if opts.image_format == "JPEG" else f"page_{max_num:05d}.png")
                if opts.image_format == "JPEG":
                    im.save(dst, "JPEG", quality=94, subsampling=0)
                else:
                    im.save(dst, "PNG", optimize=False)
                prev = im
                captured += 1
                status(f"撮影: {captured} / {opts.max_pages} 枚（累計 {max_num} 枚）。F8で停止")
                # Guard against repetitive blank frames from the renderer.
        save_manifest(book, {"source": "screen_capture", "capture_region": list(opts.region),
                             "capture_key": opts.key_name, "images": len(page_images(image_dir))})
        return captured
    finally:
        listener.stop()
