"""Continuous-scroll capture: for viewers that scroll smoothly instead of turning pages.

The reading area is photographed, the viewer is scrolled by an ordinary key press or mouse-wheel
notch, and consecutive screenshots are aligned by comparing their overlap (``stitch.py``).
New rows are appended to one tall strip which is cut into page-like images at blank rows
(``pages/page_00001.png`` ...), so the OCR stage works exactly as for page-turn captures.

Only ordinary on-screen input is used; there is no DRM handling of any kind.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from PIL import Image

from .core import _write_json_atomic, looks_black, page_images, save_manifest
from .stitch import (ScrollStitcher, ShiftResult, Slice, column_background, detect_static_bands,
                     estimate_shift, to_gray)

SCROLL_METHODS = {
    "マウスホイール": "wheel",
    "↓ (下矢印)": "down",
    "PageDown": "page_down",
    "Space": "space",
}
STATE_VERSION = 1


@dataclass
class ScrollOptions:
    region: tuple[int, int, int, int]      # x, y, width, height of the reading area
    max_steps: int = 60                    # scroll steps (screenshots) in this run
    delay: float = 0.8                     # minimum wait after each scroll (seconds)
    poll: float = 0.25                     # interval of the "has the screen stopped changing" check
    settle_timeout: float = 8.0            # give up waiting for a stable screen after this long
    extra_wait: float = 0.3                # one more pause once the screen looks stable (lazy loading)
    countdown: int = 5
    method: str = "マウスホイール"
    units: int = 0                         # wheel notches / key presses per step; 0 = automatic
    target_overlap: float = 0.25           # share of the screen shown twice (automatic mode)
    min_overlap: float = 0.10              # below this overlap the match is not trusted
    stop_duplicates: int = 3               # unchanged screens in a row = end of the book
    slice_height: int = 0                  # height of the output images; 0 = from the width
    top_ignore: int = -1                   # fixed toolbar rows: -1 auto, 0 none, n pixels
    bottom_ignore: int = -1
    crop_right: int = 0                    # pixels of the right edge to drop (scrollbar)
    max_recoveries: int = 3
    blank_rechecks: int = 3                # extra looks at new rows that are completely blank
    flush_on_stop: bool = False            # also write the unfinished tail after F8 / a failure
    image_format: str = "PNG"


class ScrollBackend(Protocol):
    def grab(self) -> Image.Image: ...
    def scroll(self, units: int) -> None: ...   # positive = towards the end of the book


class ScreenBackend:
    """Real screen: mss for pictures, pynput for the wheel / keys."""

    def __init__(self, region: tuple[int, int, int, int], method: str):
        import mss
        from pynput.keyboard import Controller as Keyboard, Key
        from pynput.mouse import Controller as Mouse

        self.region = region
        self.kind = SCROLL_METHODS[method]
        self.sct = mss.mss()
        self.keyboard = Keyboard()
        self.mouse = Mouse()
        self.key = {"down": Key.down, "page_down": Key.page_down, "space": Key.space}.get(self.kind)

    def grab(self) -> Image.Image:
        x, y, w, h = self.region
        shot = self.sct.grab({"left": x, "top": y, "width": w, "height": h})
        return Image.frombytes("RGB", shot.size, shot.rgb)

    def scroll(self, units: int) -> None:
        import time

        n = abs(int(units))
        if self.kind == "wheel":
            x, y, w, h = self.region
            self.mouse.position = (x + w // 2, y + h // 2)
            for _ in range(n):
                self.mouse.scroll(0, -1 if units > 0 else 1)
                time.sleep(0.02)
        else:
            key = self.key
            if units < 0:
                from pynput.keyboard import Key
                key = Key.page_up if self.kind in ("page_down", "space") else Key.up
            for _ in range(n):
                self.keyboard.press(key)
                self.keyboard.release(key)
                time.sleep(0.04)

    def close(self) -> None:
        self.sct.close()


class ScrollStop(Exception):
    """The run cannot continue safely. ``reason`` is shown to the user."""


class ScrollCapture:
    def __init__(self, book: Path, opts: ScrollOptions, stop: threading.Event,
                 status: Callable[[str], None], backend: ScrollBackend,
                 wait: Callable[[float], bool] | None = None):
        if opts.method not in SCROLL_METHODS:
            raise ValueError("スクロール方法が不明です。")
        x, y, w, h = opts.region
        if w < 100 or h < 100:
            raise ValueError("キャプチャ領域が小さすぎます。")
        self.book = Path(book)
        self.opts = opts
        self.stop = stop
        self.say = status
        self.backend = backend
        self.wait = wait or stop.wait
        self.image_dir = self.book / "pages"
        self.state_dir = self.book / "scroll_state"
        self.top = max(0, opts.top_ignore)
        self.bottom = max(0, opts.bottom_ignore)
        self.stitcher: ScrollStitcher | None = None
        self.px_per_unit = 0.0
        self.recent: list[float] = []
        self.next_page = 1
        self.written = 0
        self.log: list[dict] = []
        self.warnings = 0
        self.recoveries = 0
        self.late_fixes = 0
        self.fixed_units = max(0, int(opts.units))
        self.unit_warned = False

    # ---- pictures -------------------------------------------------------------------------
    def window(self, frame: Image.Image) -> Image.Image:
        w, h = frame.size
        return frame.crop((0, self.top, w - self.opts.crop_right, h - self.bottom))

    @staticmethod
    def _same(a: np.ndarray, b: np.ndarray) -> bool:
        return a.shape == b.shape and float(np.abs(a - b).mean()) < 0.6

    @staticmethod
    def _small(im: Image.Image) -> np.ndarray:
        g = im.convert("L")
        return np.asarray(g.resize((max(1, g.width // 4), max(1, g.height // 4))), dtype=np.float32)

    def settle(self, extra: float = 0.0) -> Image.Image | None:
        """Wait until two pictures in a row are the same; None when stopped."""
        o = self.opts
        if self.wait(o.delay + extra):
            return None
        prev = self.backend.grab()
        prev_s = self._small(prev)
        stable = 0
        for _ in range(max(1, int(o.settle_timeout / o.poll))):
            if self.wait(o.poll):
                return None
            cur = self.backend.grab()
            cur_s = self._small(cur)
            if self._same(prev_s, cur_s):
                stable += 1
                if stable >= 2:
                    if o.extra_wait and self.wait(o.extra_wait):
                        return None
                    return self.backend.grab()
            else:
                stable = 0
            prev, prev_s = cur, cur_s
        self.say("画面が安定しないまま待機時間が過ぎました（読み込みが遅い可能性）。")
        return prev

    # ---- output ---------------------------------------------------------------------------
    def _save_slices(self, slices: list[Slice]) -> None:
        for sl in slices:
            ext = "jpg" if self.opts.image_format == "JPEG" else "png"
            dst = self.image_dir / f"page_{self.next_page:05d}.{ext}"
            if ext == "jpg":
                sl.image.save(dst, "JPEG", quality=94, subsampling=0)
            else:
                sl.image.save(dst, "PNG", optimize=False)
            self.next_page += 1
            self.written += 1
            if sl.forced:
                self.warnings += 1
                self.say(f"注意: {dst.name} は空白行がなく途中で切断しました（図表が大きい可能性）。")

    def save_state(self) -> None:
        st = self.stitcher
        if st is None:
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(st.strip).save(self.state_dir / "strip.png")
        _write_json_atomic(self.state_dir / "state.json", {
            "version": STATE_VERSION, "region": list(self.opts.region), "method": self.opts.method,
            "crop_right": self.opts.crop_right, "top": self.top, "bottom": self.bottom,
            "window": [st.width, st.window_h], "emitted": st.emitted,
            "px_per_unit": self.px_per_unit, "slice_height": st.slice_height})

    def load_state(self) -> bool:
        """Restore the unfinished strip of a previous run; flushes it when it cannot be continued."""
        sj, sp = self.state_dir / "state.json", self.state_dir / "strip.png"
        if not (sj.exists() and sp.exists()):
            return False
        try:
            data = json.loads(sj.read_text(encoding="utf-8"))
            with Image.open(sp) as im:
                strip = im.convert("RGB")
        except (OSError, ValueError):
            return False
        if data.get("version") != STATE_VERSION:
            return False
        o = self.opts
        w, h = o.region[2] - o.crop_right, o.region[3]
        top, bottom = int(data["top"]), int(data["bottom"])
        compatible = (data.get("window") == [w, h - top - bottom]
                      and data.get("crop_right") == o.crop_right
                      and (o.top_ignore < 0 or o.top_ignore == top)
                      and (o.bottom_ignore < 0 or o.bottom_ignore == bottom))
        st = ScrollStitcher(int(data["window"][1]), int(data["window"][0]),
                            o.slice_height or int(data.get("slice_height", 0)))
        st.load_pending(strip, int(data["emitted"]))
        if not compatible:
            self.say("前回と撮影範囲が違うため、前回の未保存分を保存して新しい区間から始めます。")
            self._save_slices(st.pop_slices(final=True))
            return False
        self.top, self.bottom = top, bottom
        self.px_per_unit = float(data.get("px_per_unit") or 0.0)
        self.stitcher = st
        return True

    # ---- matching -------------------------------------------------------------------------
    def _match(self, frame: Image.Image, expected: float | None) -> ShiftResult:
        assert self.stitcher is not None
        prev = to_gray(self.stitcher.last_window())
        return estimate_shift(prev, to_gray(self.window(frame)), expected=expected,
                              min_overlap=self.opts.min_overlap)

    def _plan_units(self) -> int:
        if self.fixed_units:
            return self.fixed_units
        win_h = self.stitcher.window_h
        ppu = max(1.0, self.px_per_unit)
        limit = win_h * (1 - self.opts.min_overlap)
        if ppu > limit:
            if not self.unit_warned:
                self.unit_warned = True
                self.say("警告: 1回の操作の移動量が大きすぎて画面の重なりを確保できません。"
                         "「マウスホイール」または「↓」への変更を推奨します。")
            return 1
        want = win_h * (1 - self.opts.target_overlap)
        return int(max(1, min(round(want / ppu), limit // ppu)))

    def _learn(self, shift: int, units: int, expected: float) -> None:
        if shift >= 0.8 * expected:         # smaller steps happen at the end of the book
            self.recent = (self.recent + [shift / units])[-5:]
            self.px_per_unit = float(np.median(self.recent))

    def _debug_dump(self, frame: Image.Image) -> str:
        d = self.book / "scroll_debug"
        d.mkdir(parents=True, exist_ok=True)
        assert self.stitcher is not None
        self.stitcher.last_window().save(d / "last_good_window.png")
        frame.save(d / "failed_frame.png")
        return str(d)

    # ---- the run --------------------------------------------------------------------------
    def run(self) -> int:
        o = self.opts
        self.image_dir.mkdir(parents=True, exist_ok=True)
        existing = page_images(self.image_dir)
        self.next_page = 1 + max((int(p.stem.split("_")[-1]) for p in existing
                                  if p.stem.startswith("page_") and p.stem.split("_")[-1].isdigit()), default=0)
        reason, message = "max", ""
        try:
            frame = self.settle()
            if frame is None:
                return 0
            if looks_black(frame):
                self.say("ほぼ真っ黒な画像を検出。画面保護・領域設定を確認して終了します。")
                return 0
            resumed = self.load_state()
            if resumed:
                self._resume(frame)
            else:
                self._begin(frame)
            self.say(f"連続スクロール開始（1回の移動量 約{self.px_per_unit:.0f}px、"
                     f"画面の高さ {self.stitcher.window_h}px）。F8で停止。")
            idle = 0
            for step in range(1, o.max_steps + 1):
                if self.stop.is_set():
                    reason = "stop"
                    break
                units = self._plan_units()
                expected = units * max(1.0, self.px_per_unit)
                self.backend.scroll(units)
                frame = self.settle()
                if frame is None:
                    reason = "stop"
                    break
                if looks_black(frame):
                    raise ScrollStop("ほぼ真っ黒な画像を検出しました。画面保護・領域設定を確認してください。")
                res = self._match(frame, expected)
                if not res.confident:
                    frame, res, units, expected = self._recover(frame, res, units, expected)
                    if frame is None:
                        reason = "stop"
                        break
                if res.static:
                    idle += 1
                    self.say(f"画面が動きません {idle}/{o.stop_duplicates}（本の末尾、またはビューアが操作を受けていません）")
                    if idle >= o.stop_duplicates:
                        reason = "end"
                        break
                    continue
                idle = 0
                frame, res = self._guard_late_load(frame, res)
                if frame is None:
                    reason = "stop"
                    break
                self.stitcher.append(self.window(frame), res.shift)
                self._learn(res.shift, units, expected)
                self.log.append({"step": step, "units": units, "shift": res.shift,
                                 "expected": round(expected, 1), "error": round(res.error, 2),
                                 "blank": res.blank})
                self._save_slices(self.stitcher.pop_slices())
                self.save_state()
                self.say(f"スクロール {step}/{o.max_steps}（移動 {res.shift}px、誤差 {res.error:.1f}）。"
                         f"出力 {self.written} 枚。F8で停止")
            else:
                reason = "max"
        except ScrollStop as exc:
            reason, message = "failure", str(exc)
        finally:
            self._finish(reason)
        if reason == "failure":
            raise InterruptedError(f"連続スクロールを安全停止しました: {message}（保存済み {self.written} 枚）")
        return self.written

    def _begin(self, frame: Image.Image) -> None:
        """First screen + one calibration scroll: learns the pixels per notch and fixed toolbars."""
        o = self.opts
        units = self.fixed_units or 1
        first = frame
        total = 0
        for attempt in range(3):
            self.backend.scroll(units)
            total += units
            frame2 = self.settle()
            if frame2 is None:
                raise ScrollStop("停止が要求されました。")
            if attempt == 0 and (o.top_ignore < 0 or o.bottom_ignore < 0):
                a, b = detect_static_bands(first, frame2)
                if o.top_ignore < 0:
                    self.top = a
                if o.bottom_ignore < 0:
                    self.bottom = b
                if self.top or self.bottom:
                    self.say(f"固定表示と思われる上 {self.top}px・下 {self.bottom}px を除外します。")
            w0, w1 = to_gray(self.window(first)), to_gray(self.window(frame2))
            res = estimate_shift(w0, w1, expected=None, min_overlap=o.min_overlap)
            if res.static:
                self.say(f"スクロールが反映されません（{attempt + 1}/3）。ビューアをクリックして操作できる状態にしてください。")
                continue
            if res.confident:
                st = ScrollStitcher(w0.shape[0], w0.shape[1], o.slice_height)
                st.start(self.window(first))
                st.append(self.window(frame2), res.shift)
                self.stitcher = st
                self.px_per_unit = res.shift / total
                self.recent = [self.px_per_unit]
                self.log.append({"step": 0, "units": total, "shift": res.shift, "error": round(res.error, 2)})
                self._save_slices(st.pop_slices())
                self.save_state()
                return
            self.say(f"移動量を測定できません: {res.reason}")
        raise ScrollStop("スクロール量を測定できませんでした。文字や図のあるページで開始し、"
                         "ビューアをクリックして操作できる状態にしてください。")

    def _resume(self, frame: Image.Image) -> None:
        st = self.stitcher
        assert st is not None
        win = self.window(frame)
        if win.size != (st.width, st.window_h):
            raise ScrollStop("前回と画面の大きさが変わっています。")
        res = estimate_shift(to_gray(st.last_window()), to_gray(win), expected=None,
                             min_overlap=self.opts.min_overlap)
        if res.confident:
            st.append(win, res.shift)
            self.say("前回の続きから再開します（位置が合いました）。" if res.shift == 0
                     else f"前回の続きから再開します（{res.shift}px 先に進んでいました）。")
            self._save_slices(st.pop_slices())
            return
        self.say("警告: 前回の最後の画面と一致しません。前回の分を保存し、新しい区間として始めます"
                 "（この境目で文章が重複・欠落していないか確認してください）。")
        self.warnings += 1
        self._save_slices(st.pop_slices(final=True))
        self.stitcher = None
        self.px_per_unit = 0.0
        self._begin(frame)

    def _recover(self, frame: Image.Image, res: ShiftResult, units: int, expected: float):
        """Mismatch: wait longer (late loading), then retreat half a step so the overlap grows."""
        o = self.opts
        self.say(f"画面の照合に失敗: {res.reason}。再試行します。")
        for attempt in range(1, o.max_recoveries + 1):
            self.recoveries += 1
            back = 0
            if attempt >= 2 and units >= 2:
                back = max(1, units // 2)
                self.backend.scroll(-back)
            frame = self.settle(extra=o.delay * attempt)
            if frame is None:
                return None, res, units, expected
            exp = (units - back) * max(1.0, self.px_per_unit)
            res = self._match(frame, exp)
            if res.confident:
                self.say("照合に成功しました（待機・スクロール量の調整で回復）。")
                self.warnings += 1
                return frame, res, units - back, exp
        d = self._debug_dump(frame)
        raise ScrollStop(f"{res.reason}。確認用画像: {d}")

    def _guard_late_load(self, frame: Image.Image, res: ShiftResult):
        """New rows that are completely blank may be a page that is still being drawn: look again.

        Real blank stretches (a blank page) cost a few extra waits; a page that is still loading
        gets up to ``blank_rechecks`` more chances before the blank rows are accepted.
        """
        if res.blank or res.shift < 40:
            return frame, res
        for attempt in range(1, self.opts.blank_rechecks + 1):
            if self._has_ink(frame, res.shift):
                break
            again = self.settle(extra=self.opts.delay * attempt)
            if again is None:
                return None, res
            res2 = self._match(again, res.shift)
            if res2.confident and not res2.static and abs(res2.shift - res.shift) <= 2:
                if self._has_ink(again, res2.shift):
                    self.late_fixes += 1
                frame, res = again, res2
        return frame, res

    def _has_ink(self, frame: Image.Image, shift: int) -> bool:
        win = to_gray(self.window(frame))
        new = win[-shift:]
        bg = column_background(win, step=2)
        return bool((np.abs(new - bg[None, :]) > 28).any())

    def _finish(self, reason: str) -> None:
        st = self.stitcher
        if st is None:
            return
        if reason in ("end", "max") or self.opts.flush_on_stop:
            self._save_slices(st.pop_slices(final=True))
        else:
            self._save_slices(st.pop_until_hold_back())
            if st.pending_rows > st.window_h // 4 and not (self.stop.is_set() and False):
                self.say(f"停止: 末尾の約{st.pending_rows}pxは文の途中のため保存を保留しました。"
                         "再開すると続きから取り込みます。")
        self.save_state()
        stats = {"steps": len([e for e in self.log if e["step"] > 0]), "warnings": self.warnings,
                 "recoveries": self.recoveries, "late_load_rechecks": self.late_fixes,
                 "forced_cuts": st.forced_cuts, "px_per_unit": round(self.px_per_unit, 2)}
        try:
            (self.book / "scroll_log.json").write_text(
                json.dumps({"stats": stats, "steps": self.log[-2000:]}, ensure_ascii=False, indent=1),
                encoding="utf-8")
            save_manifest(self.book, {
                "source": "screen_capture", "capture_mode": "continuous_scroll",
                "capture_region": list(self.opts.region), "scroll_method": self.opts.method,
                "scroll_stats": stats, "images": len(page_images(self.image_dir))})
        except OSError:
            pass
        self.say(f"連続スクロール終了: 今回 {self.written} 枚を保存（注意 {self.warnings} 件）。")


def run_scroll_capture(book: Path, opts: ScrollOptions, stop: threading.Event,
                       status: Callable[[str], None], backend: ScrollBackend | None = None,
                       wait: Callable[[float], bool] | None = None) -> int:
    """GUI entry point. With ``backend`` None the real screen is used and F8 stops the run."""
    listener = None
    own_backend = backend is None
    if own_backend:
        try:
            from pynput.keyboard import Key, Listener
            backend = ScreenBackend(opts.region, opts.method)
        except ImportError as exc:
            raise RuntimeError("必要なパッケージがありません。setup.bat を実行してください。") from exc

        def on_press(pressed):
            if pressed == Key.f8:
                stop.set()
                status("F8で停止しました。")
                return False
            return None

        listener = Listener(on_press=on_press)
        listener.start()
    try:
        if own_backend:
            for i in range(opts.countdown, 0, -1):
                if stop.is_set():
                    return 0
                status(f"{i} 秒後に撮影。電子書籍を前面に出し、クリックして操作できる状態にしてください（F8で停止）")
                if stop.wait(1):
                    return 0
        return ScrollCapture(book, opts, stop, status, backend, wait).run()
    finally:
        if listener is not None:
            listener.stop()
        if own_backend and hasattr(backend, "close"):
            backend.close()
