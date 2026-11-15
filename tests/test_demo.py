"""Step 7 tests: run with `python tests/test_demo.py`. Needs numpy + opencv (+ ffmpeg for one test)."""
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidsearch import curves as cv                       # noqa: E402
from vidsearch import demo                               # noqa: E402
from vidsearch import fusion as fu                       # noqa: E402


def _video(path, seconds=6, fps=10):
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    if not w.isOpened():
        return False
    for i in range(seconds * fps):
        w.write(np.full((48, 64, 3), (i * 4) % 255, np.uint8))
    w.release()
    return True


def test_config():
    cfg = demo.load_final_config(None)
    assert cfg["alpha_top1"] == 0.5 and "siglip" in cfg["frame_model"]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "step4_summary.json"
        json.dump({"final_config": {"alpha_top1": 0.65, "threshold_params": {"thr": 0.25, "smooth_k": 3,
                   "rank": "area"}, "windows_params": {"ctx": None, "smooth_k": 1, "min_len": 2, "max_frac": 1.0},
                   "frame_model": "SigLIP (notebook 01)"}}, open(p, "w"))
        cfg = demo.load_final_config(p)
        assert cfg["alpha_top1"] == 0.65 and cfg["windows_params"]["ctx"] is None
        assert cfg["frame_model"] == "google/siglip-base-patch16-224"          # descriptive name replaced
        json.dump({"model": "google/siglip-base-patch16-256", "n": 3}, open(Path(d) / "meta.json", "w"))
        assert demo.find_model_name(d) == "google/siglip-base-patch16-256"
        np.savez(Path(d) / "siglip_base_text.npz", qids=np.array(["test_00001", "test_00002"]),
                 emb=np.arange(8, dtype=np.float32).reshape(2, 4))
        tab = demo.load_qid_table(d, ["test_00001", "test_00002"])
        assert np.allclose(tab["test_00002"], [4, 5, 6, 7])
    print("PASS final config + model name lookup")


def test_search_matches_benchmark_path():
    rng = np.random.default_rng(0)
    cfg = demo.load_final_config(None)
    recs, S, A = [], {}, {}
    for i in range(30):
        n = int(rng.integers(20, 40))
        recs.append({"qid": f"q{i}", "vid": f"v{i}", "query": "x", "start": 5.0, "end": 12.0, "duration": n - 0.4})
        S[f"q{i}"], A[f"q{i}"] = rng.normal(size=n), rng.normal(size=n)
    for r in recs:
        segs, scores, _ = demo.predict_segments(S[r["qid"]], A[r["qid"]], cfg, r["duration"])
        # the Step 4 evaluation path for the same single query
        Pt = fu.predict_array(fu.fused_curves(S, A, [r], cfg["alpha_top1"]), [r], "threshold", cfg["threshold_params"])
        Pw = fu.predict_array(fu.fused_curves(S, A, [r], cfg["alpha_ranks_2_5"]), [r], "windows", cfg["windows_params"])
        H = fu.hybrid(Pt, Pw)[0]
        ref = [tuple(x) for x in H if not np.isnan(x[0])]
        assert np.allclose(segs, ref), (segs, ref)
        assert len(scores) == len(segs) and all(x1 <= r["duration"] + 1e-9 for _, x1 in segs)
    fig = demo.plot_search(S["q0"], A["q0"], cv.fuse(S["q0"], A["q0"], 0.5), segs, scores, "a query", (5, 12))
    assert len(fig.axes) == 1
    print("PASS demo search == Step 4 benchmark path (30 queries) + plot")


def test_clips_and_discovery():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "charades").mkdir(); (d / "ucf" / "Fighting").mkdir(parents=True)
        ok = _video(d / "charades" / "AAAAA.mp4") and _video(d / "ucf" / "Fighting" / "Fighting003_x264.mp4")
        if not ok:
            print("SKIP clips (no mp4 encoder)")
            return
        ex = demo.find_extra_videos(d, exclude_dir=d / "charades")
        assert len(ex) == 1 and ex[0]["queries"] == demo.SUGGESTIONS["fighting"]
        if shutil.which("ffmpeg"):
            paths = demo.cut_clips(d / "ucf" / "Fighting" / "Fighting003_x264.mp4", [(1.0, 3.0), (4.0, 5.5)], d / "out")
            assert all(p and os.path.getsize(p) > 0 for p in paths), paths
            print("PASS clip cutting (ffmpeg) + extra-video discovery")
        else:
            print("PASS extra-video discovery (ffmpeg not installed, clip cutting skipped)")


def _fake_gradio():
    """Just enough of gradio's API to build the app and call its callbacks."""
    g = types.ModuleType("gradio")
    handlers = {}

    class Dep:
        def then(self, fn, inputs=None, outputs=None):
            handlers[fn.__name__ + "_then"] = fn
            return Dep()

    class C:
        def __init__(self, *a, **k): self.kw = k
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def _on(self, fn, inputs=None, outputs=None):
            handlers[fn.__name__] = fn
            return Dep()
        click = submit = change = select = _on
    for name in ("Blocks", "Row", "Column", "Markdown", "State", "Video", "Dropdown", "Button", "Textbox",
                 "Plot", "Dataframe", "Radio", "HTML", "Accordion"):
        setattr(g, name, C)
    g.update = lambda **k: k
    g.Progress = lambda *a, **k: (lambda f, desc=None: None)
    g.SelectData = object
    g.themes = types.SimpleNamespace(Soft=lambda **k: "soft-theme")
    return g, handlers


def test_app_wiring():
    g, h = _fake_gradio()
    sys.modules["gradio"] = g

    class FakeSearch:
        cfg = demo.load_final_config(None)

        def index(self, path, progress=None):
            progress and progress(0.5, "half")
            return {"path": path, "duration": 20.0, "timing": {"total_s": 1.0}}

        def search(self, idx, q):
            s = np.sin(np.arange(20)); a = np.cos(np.arange(20))
            return {"segments": [(2.0, 6.0), (10.0, 14.0)], "scores": [1.2, 0.4], "clips": ["c1.mp4", "c2.mp4"],
                    "s": s, "a": a, "f": cv.fuse(s, a, 0.5), "query_ms": 9.5}

    with tempfile.TemporaryDirectory() as d:
        vid = Path(d) / "AAAAA.mp4"
        if not _video(vid):
            print("SKIP app wiring (no mp4 encoder)")
            return
        ex = [{"label": "example", "path": str(vid), "queries": ["a person opens the door", "person sits"]}]
        app = demo.build_app(FakeSearch(), ex, {demo.gt_key(vid, "a person opens the door"): (2.0, 5.0)})
        assert h["pick_example"]("example") == str(vid)
        # Gradio copies files into its cache: same file name, different folder
        cached = Path(d) / "cache"; cached.mkdir(); shutil.copy(vid, cached / "AAAAA.mp4")
        idx, msg, sugg, q = h["on_video"](str(cached / "AAAAA.mp4"))
        assert "Ready" in msg and sugg["choices"][0] == "a person opens the door" and q["value"] == ""
        assert h["use_suggestion"](types.SimpleNamespace(value="person sits")) == "person sits"
        fig, info, cards, moment, player, clips = h["do_search"](idx, str(cached / "AAAAA.mp4"), "a person opens the door")
        assert "Ground truth" in info and "IoU with ground truth" in cards and "vi-card top" in cards
        assert moment["value"].startswith("#1") and player == "c1.mp4" and clips == ["c1.mp4", "c2.mp4"]
        assert h["play_moment"](moment["choices"][1], clips) == "c2.mp4"
        assert "first" in h["do_search"](None, None, "x")[1]
        assert "Type what" in h["do_search"](idx, str(vid), " ")[1]
        assert "Pick an example" in h["on_video"](None)[1]
        assert "do_search_then" in h
        assert hasattr(app, "_vi_launch_kw")
    del sys.modules["gradio"]
    print("PASS app wiring (stand-in gradio): auto-index, suggestions, search, moment player")


if __name__ == "__main__":
    test_config()
    test_search_matches_benchmark_path()
    test_clips_and_discovery()
    test_app_wiring()
