"""Step 4 tests: run with `python tests/test_fusion.py`. Needs numpy only."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidsearch import curves as cv                       # noqa: E402
from vidsearch import fusion as fu                       # noqa: E402


def _toy(n_videos=40, seed=0):
    """Frame curves are smooth but peak ~3 s too early; activity curves sit exactly on the
    moment but are noisy. Neither is right alone; the mix should be."""
    rng = np.random.default_rng(seed)
    recs, S, A = [], {}, {}
    for v in range(n_videos):
        n = 30
        s0 = int(rng.integers(3, 18))
        e0 = s0 + int(rng.integers(5, 10))
        q = f"q{v}"
        recs.append({"qid": q, "vid": f"v{v}", "query": "x", "start": float(s0),
                     "end": float(e0), "duration": float(n)})
        t = np.arange(n)
        S[q] = np.exp(-0.5 * ((t - ((s0 + e0) / 2 - 3)) / 4.0) ** 2) + rng.normal(0, 0.1, n)
        A[q] = ((t >= s0) & (t < e0)).astype(float) + rng.normal(0, 1.2, n)
    return recs, S, A


def test_eval_matches_curves():
    recs, S, _ = _toy()
    P = fu.predict_array(S, recs, "windows", {"ctx": 0.5})
    m1, per = fu.evaluate_array(P, recs)
    m2, top1 = cv.evaluate(cv.predict(S, recs, "windows", {"ctx": 0.5}), recs)
    assert m1 == m2, (m1, m2)
    assert np.allclose(per["top1"], [top1[r["qid"]] for r in recs])
    P[3] = np.nan                                         # a query with no prediction scores 0
    assert fu.evaluate_array(P, recs)[1]["top1"][3] == 0
    print("PASS vectorised eval == curves.evaluate")


def test_hybrid():
    top = np.full((2, 5, 2), np.nan)
    top[0, 0], top[0, 1] = (10, 20), (0, 4)
    top[1, 0] = (5, 6)
    rest = np.full((2, 5, 2), np.nan)
    rest[0, :3] = [(11, 20), (30, 40), (50, 60)]          # (11, 20) duplicates the top-1
    rest[1, 0] = (5, 6)
    H = fu.hybrid(top, rest)
    assert np.allclose(H[0, :4], [(10, 20), (30, 40), (50, 60), (0, 4)]) and np.isnan(H[0, 4, 0])
    assert np.allclose(H[1, 0], (5, 6)) and np.isnan(H[1, 1, 0])
    empty = fu.hybrid(np.full((1, 5, 2), np.nan), rest[:1])
    assert np.allclose(empty[0, 0], (11, 20))            # no threshold top-1 -> first window
    print("PASS hybrid merge")


def test_sweep_prefers_mix():
    recs, S, A = _toy(60)
    grids = {"threshold": cv.grid(thr=[0.0, 0.5], smooth_k=[1, 3], rank=["mean"]),
             "windows": cv.grid(ctx=[0.5, None], smooth_k=[1], min_len=[2], max_frac=[1.0])}
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = fu.sweep(S, A, recs, alphas, grids=grids, n_jobs=1)
    assert len(rows) == len(alphas) * (len(grids["threshold"]) + len(grids["windows"]))
    frame, act = fu.best_row(rows, "threshold", [1.0]), fu.best_row(rows, "threshold", [0.0])
    mix = fu.best_row(rows, "threshold", [0.25, 0.5, 0.75])
    assert mix["R1@0.5"] > max(frame["R1@0.5"], act["R1@0.5"]), (frame["R1@0.5"], act["R1@0.5"], mix["R1@0.5"])
    w, m = fu.best_hybrid(rows, mix, recs, [0.25, 0.5, 0.75])
    assert m["R1@0.5"] == mix["R1@0.5"] and m["R5@0.5"] >= mix["R5@0.5"]
    tab = fu.alpha_table(rows)
    assert len(tab) == 2 * len(alphas)
    print("PASS sweep (mix beats both single curves on toy data) + hybrid search")


def test_bootstrap_and_categories():
    g = np.repeat(np.arange(100), 3)
    x = np.ones(300)
    same = fu.cluster_bootstrap(x, x, g, B=500)
    assert same["diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.5).astype(float)
    better = fu.cluster_bootstrap(np.maximum(y, rng.random(300) < 0.3), y, g, B=500)
    assert better["diff"] > 0 and better["ci_low"] > 0 and better["p_le_0"] < 0.05
    r = {"start": 0, "end": 2, "duration": 30}
    assert fu.duration_category(r).startswith("brief")
    assert fu.duration_category({**r, "end": 12}).startswith("long")
    assert fu.duration_category({**r, "end": 6}) == "medium"
    print("PASS cluster bootstrap + categories")


if __name__ == "__main__":
    test_eval_matches_curves()
    test_hybrid()
    test_sweep_prefers_mix()
    test_bootstrap_and_categories()
