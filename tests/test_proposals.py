"""Run with:  python tests/test_proposals.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from vidsearch.proposals import window_proposals, threshold_segments, smooth
from vidsearch.metrics import iou_1d
from vidsearch.search import score_curves, predict, grid_search


def bump_curve(T=30, s=10, e=18, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.3, T)
    x[s:e] += 2.0
    return x


def test_windows_find_bump():
    top = window_proposals(bump_curve(), fps=1.0, duration=30.0)
    assert iou_1d([top[0][:2]], (10, 18))[0] >= 0.7, top[0]
    assert len(top) <= 5 and all(0 <= p[0] < p[1] <= 30 for p in top)


def test_threshold_find_bump():
    top = threshold_segments(bump_curve(), fps=1.0, duration=30.0, thr=0.5)
    assert iou_1d([top[0][:2]], (10, 18))[0] >= 0.7, top[0]


def test_fps2_and_short_video():
    top = window_proposals(bump_curve(T=60, s=20, e=36), fps=2.0, duration=30.0)
    assert iou_1d([top[0][:2]], (10, 18))[0] >= 0.7
    assert window_proposals(np.array([1.0, 2.0]), fps=1.0, duration=2.0)  # shorter than every window


def test_smooth_shape():
    assert smooth(np.arange(10.0), 5).shape == (10,)


def test_end_to_end_search():
    rng = np.random.default_rng(0)
    D, T = 16, 30
    q = rng.normal(size=D); q /= np.linalg.norm(q)
    V = rng.normal(size=(T, D)) * 0.1
    V[10:18] += q                                     # frames 10-17 match the query
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    recs = [{"qid": "q0", "vid": "v0", "start": 10.0, "end": 18.0, "duration": 30.0}]
    curves = score_curves(recs, {"v0": V}, {"q0": q})
    preds = predict(recs, curves, "windows")
    assert iou_1d([preds["q0"][0][:2]], (10, 18))[0] >= 0.7
    rows = grid_search(recs, curves, {"smooth_k": [1, 3], "ctx": [0, 0.5]})
    assert rows[0]["R1@0.5"] == 100


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
