"""Step 3 tests: run with `python tests/test_activity.py`. Needs numpy + opencv only."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidsearch import curves as cv                       # noqa: E402
from vidsearch import xclip_features as xf               # noqa: E402


def test_clip_plan():
    times, starts, idx = xf.clip_plan(30, 8, 4.0, 1.0)
    assert len(times) == 60 and np.isclose(times[0], 0.25) and np.isclose(times[-1], 29.75)
    assert len(starts) == 27 and starts[0] == 0 and starts[-1] == 26
    assert idx.shape == (27, 8) and list(idx[0]) == list(range(8)) and list(idx[-1]) == list(range(52, 60))
    _, s2, i2 = xf.clip_plan(3, 8, 4.0, 1.0)                 # shorter than one clip
    assert list(s2) == [0.0] and i2.max() < 8
    _, s3, i3 = xf.clip_plan(30, 32, 4.0, 1.0)               # 32-frame model -> 8 fps grid
    assert i3.shape == (27, 32) and i3[1, 0] == 8
    print("PASS clip_plan")


def test_clip_to_curve():
    c = xf.clip_to_curve([1.0, 3.0], [0, 1], 2, 3)
    assert np.allclose(c, [1.0, 2.0, 3.0]), c
    c = xf.clip_to_curve([5.0], [0], 4, 3)                   # clip longer than video
    assert np.allclose(c, [5, 5, 5])
    print("PASS clip_to_curve")


def test_metrics_and_proposals():
    recs = [{"qid": "a", "vid": "v", "query": "q", "start": 10.0, "end": 18.0, "duration": 40.0}]
    oracle, _ = cv.evaluate({"a": [(10.0, 18.0)]}, recs)
    assert oracle["R1@0.7"] == 100 and oracle["mIoU"] == 100
    miss, _ = cv.evaluate({"a": [(20.0, 30.0)]}, recs)
    assert miss["R5@0.3"] == 0 and miss["mIoU"] == 0
    rng = np.random.default_rng(0)
    curve = rng.normal(0, 0.2, 40)
    curve[10:18] += 2.0
    for method, params in (("windows", {"ctx": 0.5}), ("windows", {"ctx": None}),
                           ("threshold", {"thr": 0.5, "smooth_k": 1})):
        segs, _ = cv.METHODS[method](curve, **params)
        assert cv.tiou(segs[:1], (10, 18))[0] >= 0.7, (method, params, segs[:2])
    m, _ = cv.evaluate(cv.predict({"a": curve}, recs, "windows", {"ctx": 0.5}), recs)
    assert m["R1@0.7"] == 100
    best, rows = cv.tune({"a": curve}, recs, "threshold")
    assert rows[0]["R1@0.5"] == 100 and set(best) == {"thr", "smooth_k", "rank"}
    segs, _ = cv.nms([[0, 10], [1, 10], [20, 30]], [3, 2, 1], 0.5, 5)
    assert len(segs) == 2
    print("PASS metrics + proposals")


def test_records_and_pairs():
    r = cv.normalize_record({"video_id": "V1", "sentence": "x", "start": 1, "end": 2}, 7, "val")
    assert r["qid"] == "val_00007" and r["vid"] == "V1" and "duration" not in r
    recs = [
        {"qid": "1", "vid": "V", "query": "a person opens the window", "start": 0, "end": 5},
        {"qid": "2", "vid": "V", "query": "the person closes the window", "start": 10, "end": 15},
        {"qid": "3", "vid": "V", "query": "a person closes a laptop", "start": 20, "end": 25},
    ]
    pairs = cv.find_opposite_pairs(recs)
    assert len(pairs) == 1 and pairs[0][1]["qid"] == "1" and pairs[0][2]["qid"] == "2"
    t = np.arange(30)
    good = {"1": (t < 5).astype(float), "2": ((t >= 10) & (t < 15)).astype(float)}
    assert cv.direction_accuracy(good, pairs)[0] == 100
    same = {"1": (t < 5).astype(float), "2": (t < 5).astype(float)}
    assert cv.direction_accuracy(same, pairs)[0] == 50
    print("PASS records + opposite pairs")


def test_frame_loader_layouts():
    recs = [{"qid": f"test_{i:05d}", "vid": f"V{i % 3}", "query": f"query {i}",
             "start": 0.0, "end": 3.0} for i in range(6)]
    rng = np.random.default_rng(1)
    F = {f"V{j}": rng.normal(size=(10 + j, 8)).astype(np.float16) for j in range(3)}
    T = {r["qid"]: rng.normal(size=8).astype(np.float32) for r in recs}
    with tempfile.TemporaryDirectory() as d:          # layout 1: one array per id
        np.savez(Path(d) / "frame_feats.npz", **F)
        np.savez(Path(d) / "query_feats.npz", **T)
        S = cv.load_frame_curves(d, recs, verbose=False)
        assert S is not None and len(S) == 6 and len(S["test_00001"]) == 11
        f, t = F["V1"].astype(np.float32), T["test_00001"]
        ref = (f / np.linalg.norm(f, axis=1, keepdims=True)) @ (t / np.linalg.norm(t))
        assert np.allclose(S["test_00001"], ref, atol=1e-5)
    with tempfile.TemporaryDirectory() as d:          # layout 2/3: ids + stacked / offsets
        vids = sorted(F)
        off = np.cumsum([0] + [len(F[v]) for v in vids])
        np.savez(Path(d) / "frames.npz", vids=np.array(vids), offsets=off,
                 feats=np.concatenate([F[v] for v in vids]))
        np.savez(Path(d) / "text.npz", qids=np.array(list(T)), emb=np.stack(list(T.values())))
        S2 = cv.load_frame_curves(d, recs, verbose=False)
        assert S2 is not None and np.allclose(S2["test_00001"], S["test_00001"], atol=1e-5)
    print("PASS SigLIP feature loader (3 layouts)")


def test_decode_and_save():
    import cv2
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "toy.mp4")
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        if not w.isOpened():
            print("SKIP decode (no mp4 encoder on this machine)")
            return
        for i in range(65):                               # 6.5 s at 10 fps
            img = np.full((48, 64, 3), (i * 3) % 255, np.uint8)
            w.write(img)
        w.release()
        v = xf.load_video_clips(path, 8, 4.0, 1.0, size=32)
        assert v["frames"].shape == (14, 32, 32, 3), v["frames"].shape
        assert v["n_sec"] == 7 and list(v["starts"]) == [0, 1, 2, 3] and v["idx"].shape == (4, 8)
        assert v["frames"][0].mean() < v["frames"][-1].mean()   # time order preserved
        res = {"clip_emb": {"toy": np.ones((4, 5), np.float16)},
               "clip_scores": {"q": np.array([1, 2, 3, 4], np.float32)},
               "videos": {"toy": {"n_sec": 7, "duration": 6.5, "starts": [0, 1, 2, 3]}},
               "failed": [], "stats": {}}
        xf.save_activity(Path(d) / "out", res, {"q": np.ones(5, np.float16)}, {"clip_sec": 4.0})
        act = xf.load_activity(d)
        recs = [{"qid": "q", "vid": "toy", "query": "x", "start": 0, "end": 3}]
        a = xf.activity_curves(act, recs, "cond")["q"]
        assert len(a) == 7 and np.isclose(a[0], 1) and np.isclose(a[-1], 4)
        assert len(xf.activity_curves(act, recs, "plain")["q"]) == 7
        assert json.load(open(Path(d) / "out" / "xclip_meta.json"))["config"]["clip_sec"] == 4.0
    print("PASS decode + save/load + curves")


if __name__ == "__main__":
    test_clip_plan()
    test_clip_to_curve()
    test_metrics_and_proposals()
    test_records_and_pairs()
    test_frame_loader_layouts()
    test_decode_and_save()
