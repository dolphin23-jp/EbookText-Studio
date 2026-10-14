import threading

import numpy as np
import pytest
from PIL import Image

from ebooktext.core import page_images
from ebooktext.scrollcapture import ScrollCapture, ScrollOptions, run_scroll_capture
from ebooktext.stitch import (ScrollStitcher, detect_static_bands, estimate_shift,
                              to_gray)

from simviewer import SimViewer, make_document

REGION = (0, 0, 640, 600)


def no_wait(_t):
    return False


def run(tmp_path, viewer, stop=None, **kw):
    opts = ScrollOptions(region=(0, 0, 640, viewer.vh), **kw)
    stop = stop or threading.Event()
    msgs: list[str] = []
    n = run_scroll_capture(tmp_path, opts, stop, msgs.append, backend=viewer, wait=no_wait)
    return n, msgs


def stitched(tmp_path) -> np.ndarray:
    parts = [np.asarray(Image.open(p).convert("RGB")) for p in page_images(tmp_path / "pages")]
    return np.concatenate(parts, axis=0)


def covers(out: np.ndarray, doc: np.ndarray, start: int, end: int) -> bool:
    """The output is exactly doc[start:end] (nothing missing, nothing twice)."""
    return out.shape == doc[start:end].shape and np.array_equal(out, doc[start:end])


# ---- shift estimation ------------------------------------------------------------------------

def test_estimate_shift_exact_and_noisy():
    doc = make_document(3000)
    a, b = doc[400:1000], doc[400 + 437:1000 + 437]
    r = estimate_shift(to_gray(Image.fromarray(a)), to_gray(Image.fromarray(b)))
    assert r.confident and r.shift == 437
    noisy = np.clip(b.astype(int) + np.random.default_rng(0).integers(-6, 7, b.shape), 0, 255).astype(np.uint8)
    r = estimate_shift(to_gray(Image.fromarray(a)), to_gray(Image.fromarray(noisy)), expected=430)
    assert r.confident and r.shift == 437


def test_estimate_shift_unrelated_screens_not_confident():
    a = make_document(2000, seed=1)[0:600]
    b = make_document(2000, seed=2)[0:600]
    r = estimate_shift(to_gray(Image.fromarray(a)), to_gray(Image.fromarray(b)))
    assert not r.confident


def test_estimate_shift_static_and_tiny_shift():
    doc = make_document(2000)
    a = to_gray(Image.fromarray(doc[0:600]))
    assert estimate_shift(a, a.copy()).static
    r = estimate_shift(a, to_gray(Image.fromarray(doc[3:603])))
    assert r.confident and r.shift == 3


def test_estimate_shift_periodic_uses_expected():
    doc = make_document(2000, table_at=300)
    a, b = doc[300:900], doc[300 + 280:900 + 280]
    r = estimate_shift(to_gray(Image.fromarray(a)), to_gray(Image.fromarray(b)), expected=280)
    assert r.confident and r.shift == 280


def test_estimate_shift_blank_overlap_uses_expected():
    doc = np.full((2000, 640, 3), 255, dtype=np.uint8)
    doc[700:714, 100:500] = 20                                 # only in the new rows
    a, b = doc[0:600], doc[300:900]
    r = estimate_shift(to_gray(Image.fromarray(a)), to_gray(Image.fromarray(b)), expected=300)
    assert r.confident and r.blank and r.shift == 300
    assert not estimate_shift(to_gray(Image.fromarray(a)), to_gray(Image.fromarray(b))).confident


def test_detect_static_bands():
    doc = make_document(2000)
    v = SimViewer(doc, 600, 100, header=50)
    a = v.grab()
    v.scroll(1)
    b = v.grab()
    top, bottom = detect_static_bands(a, b)
    assert (top, bottom) == (50, 0)


# ---- stitching / slicing ---------------------------------------------------------------------

def test_stitcher_cuts_only_in_blank_rows_and_keeps_all_rows():
    doc = make_document(6000)
    st = ScrollStitcher(600, 640, slice_height=900)
    st.start(Image.fromarray(doc[0:600]))
    out, pos = [], 0
    while pos + 600 < 6000:
        pos = min(pos + 450, 6000 - 600)
        st.append(Image.fromarray(doc[pos:pos + 600]), 450 if pos % 450 == 0 else 0)
        out += st.pop_slices()
    out += st.pop_slices(final=True)
    cat = np.concatenate([np.asarray(s.image) for s in out], axis=0)
    assert np.array_equal(cat, doc[:len(cat)]) and len(cat) == pos + 600
    heights = np.cumsum([s.rows for s in out])[:-1]
    blank = (np.abs(doc.astype(int) - 255).sum(axis=2) < 40).all(axis=1)
    for cut in heights:
        assert blank[cut - 1] and blank[cut], f"cut at {cut} touches content"


def test_stitcher_does_not_cut_figure():
    doc = make_document(6000, figures=((1200, 1500),))
    st = ScrollStitcher(600, 640, slice_height=900)
    st.start(Image.fromarray(doc[0:600]))
    pos = 0
    out = []
    while pos < 5400:
        pos += 450
        st.append(Image.fromarray(doc[pos:pos + 600]), 450)
        out += st.pop_slices()
    out += st.pop_slices(final=True)
    cuts = np.cumsum([s.rows for s in out])[:-1]
    assert not any(1210 < c < 2690 for c in cuts), cuts
    assert not any(s.forced for s in out)


# ---- full capture runs against the fake viewer -----------------------------------------------

def test_fixed_scroll_complete_no_duplicates(tmp_path):
    doc = make_document(7000)
    v = SimViewer(doc, 600, 100)
    n, _ = run(tmp_path, v, max_steps=40, units=4)            # 400px per step, 200px overlap
    assert v.pos == len(doc) - 600                             # reached the end
    assert covers(stitched(tmp_path), doc, 0, len(doc))
    assert n >= 5


def test_auto_units_with_toolbar_and_lazy_loading(tmp_path):
    doc = make_document(7000)
    v = SimViewer(doc, 640, 80, header=48, lazy=5)
    run(tmp_path, v, max_steps=60)
    assert covers(stitched(tmp_path), doc, 48, len(doc))


def test_page_down_sized_step(tmp_path):
    doc = make_document(7000)
    v = SimViewer(doc, 600, 525)                               # one unit = 87.5% of the screen
    n, msgs = run(tmp_path, v, max_steps=40, method="PageDown")
    assert covers(stitched(tmp_path), doc, 0, len(doc))


def test_irregular_scroll_amounts(tmp_path):
    doc = make_document(7000)
    v = SimViewer(doc, 600, 60, jitter=0.15)
    run(tmp_path, v, max_steps=80)
    assert covers(stitched(tmp_path), doc, 0, len(doc))


def test_very_slow_loading_is_rechecked(tmp_path):
    doc = make_document(5000)
    v = SimViewer(doc, 600, 100, lazy=14)                      # placeholders outlast the settle loop
    run(tmp_path, v, max_steps=40, settle_timeout=2.0)
    assert covers(stitched(tmp_path), doc, 0, len(doc))


def test_scroll_jump_stops_safely_and_keeps_pages(tmp_path):
    doc = make_document(9000)
    jumped = {"n": 0}

    def jump(viewer):
        jumped["n"] += 1
        return 1400 if jumped["n"] == 6 else 0                 # a sudden jump larger than the screen

    v = SimViewer(doc, 600, 100, on_scroll=jump)
    with pytest.raises(InterruptedError) as err:
        run(tmp_path, v, max_steps=40, units=4)
    assert "安全停止" in str(err.value)
    assert (tmp_path / "scroll_debug" / "failed_frame.png").exists()
    out = stitched(tmp_path)
    assert np.array_equal(out, doc[:len(out)])                 # what was saved is correct and gap-free


def test_recovers_from_one_wrong_step_by_scrolling_back(tmp_path):
    doc = make_document(9000)
    state = {"n": 0}

    def overshoot(viewer):
        state["n"] += 1
        return 330 if state["n"] == 5 else 0                   # one oversized step, still < screen

    v = SimViewer(doc, 600, 100, on_scroll=overshoot)
    run(tmp_path, v, max_steps=60, units=4)
    assert covers(stitched(tmp_path), doc, 0, len(doc))


def test_viewer_that_does_not_scroll_stops_with_message(tmp_path):
    v = SimViewer(make_document(3000), 600, 100, dead=True)
    with pytest.raises(InterruptedError):
        run(tmp_path, v, max_steps=5)
    assert not page_images(tmp_path / "pages")


def test_end_of_book_stops_by_itself(tmp_path):
    doc = make_document(3000)
    v = SimViewer(doc, 600, 100)
    n, msgs = run(tmp_path, v, max_steps=500, units=4)
    assert any("末尾" in m for m in msgs)
    assert covers(stitched(tmp_path), doc, 0, len(doc))


def test_stop_and_resume_without_gap_or_duplicate(tmp_path):
    doc = make_document(8000)
    v = SimViewer(doc, 600, 100)
    stop = threading.Event()
    count = {"n": 0}
    orig = v.scroll

    def counting_scroll(units):
        count["n"] += 1
        if count["n"] == 7:
            stop.set()                                         # the F8 key
        orig(units)

    v.scroll = counting_scroll
    run(tmp_path, v, stop=stop, max_steps=40, units=4)
    first = stitched(tmp_path)
    assert len(first) < v.pos + 600                            # the unfinished tail is held back
    run(tmp_path, v, max_steps=60, units=4)                    # resume at the same position
    assert covers(stitched(tmp_path), doc, 0, len(doc))


def test_resume_after_user_scrolled_a_bit(tmp_path):
    doc = make_document(8000)
    v = SimViewer(doc, 600, 100)
    run(tmp_path, v, max_steps=5, units=4)                     # ends at "max": everything written
    v.pos += 150                                               # the user moved the page meanwhile
    run(tmp_path, v, max_steps=60, units=4)
    assert covers(stitched(tmp_path), doc, 0, len(doc))


def test_resume_at_unrelated_position_starts_new_segment_with_warning(tmp_path):
    doc = make_document(9000)
    v = SimViewer(doc, 600, 100)
    run(tmp_path, v, max_steps=4, units=4)
    first = len(stitched(tmp_path))
    v.pos = 6000                                               # far away, nothing in common
    n, msgs = run(tmp_path, v, max_steps=3, units=4)
    assert any("一致しません" in m for m in msgs)
    assert len(stitched(tmp_path)) > first


def test_region_change_does_not_corrupt_resume(tmp_path):
    doc = make_document(8000)
    v = SimViewer(doc, 600, 100)
    run(tmp_path, v, max_steps=4, units=4)
    v2 = SimViewer(doc, 560, 100)
    v2.pos = v.pos
    n, msgs = run(tmp_path, v2, max_steps=3, units=4)
    assert any("撮影範囲が違う" in m for m in msgs)


def test_page_images_are_ocr_compatible(tmp_path):
    doc = make_document(6000)
    run(tmp_path, SimViewer(doc, 600, 100), max_steps=40, units=4)
    pages = page_images(tmp_path / "pages")
    assert [p.name for p in pages] == [f"page_{i:05d}.png" for i in range(1, len(pages) + 1)]
    for p in pages:
        with Image.open(p) as im:
            assert im.width == 640 and 500 <= im.height <= 2400
    import json
    m = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert m["capture_mode"] == "continuous_scroll" and m["images"] == len(pages)


def test_existing_page_turn_images_are_not_overwritten(tmp_path):
    (tmp_path / "pages").mkdir()
    Image.new("RGB", (100, 100), "white").save(tmp_path / "pages" / "page_00007.png")
    run(tmp_path, SimViewer(make_document(4000), 600, 100), max_steps=40, units=4)
    names = [p.name for p in page_images(tmp_path / "pages")]
    assert "page_00007.png" in names and names[-1] > "page_00007.png"


def test_slice_pad_adds_blank_margin_and_keeps_rows(tmp_path):
    doc = make_document(6000)
    run(tmp_path, SimViewer(doc, 600, 100), max_steps=40, units=4, slice_pad=24)
    parts = [np.asarray(Image.open(p).convert("RGB")) for p in page_images(tmp_path / "pages")]
    assert all(np.all(p[:24] == p[24]) and np.all(p[-24:] == p[-25]) for p in parts)   # plain margins
    out = np.concatenate([p[24:-24] for p in parts], axis=0)
    assert np.array_equal(out, doc[:len(out)])                 # content itself is unchanged
