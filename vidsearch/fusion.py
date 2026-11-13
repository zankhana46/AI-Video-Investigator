"""Step 4: fuse frame evidence s(t) (SigLIP) with activity evidence a(t) (X-CLIP).

    f(t) = alpha * z(s) + (1 - alpha) * z(a)        per query, on the common length (curves.fuse)

alpha = 1 is frame-only, alpha = 0 is activity-only, so the ablation rows and the fused rows are
all produced by the same code with the same tuning budget. Everything is numpy on cached curves:
  sweep              every (alpha, proposal method, settings) on val, keeping every prediction
  hybrid             top-1 from threshold proposals, ranks 2-5 from sliding windows
  evaluate_array     vectorised R1/R5 @ IoU + mIoU (same numbers as curves.evaluate)
  cluster_bootstrap  95% CI for "A minus B", resampling whole videos (queries share videos)
  duration_category  brief / medium / long moments, for the challenging-case breakdown
"""
import multiprocessing as mp
import os

import numpy as np

from . import curves as cv

K = 5


# --------------------------------------------------------------------------- curves
def fused_curves(S, A, recs, alpha):
    """{qid: f(t)} for every query that has both a frame and an activity curve."""
    out = {}
    for r in recs:
        s, a = S.get(r["qid"]), A.get(r["qid"])
        if s is not None and a is not None:
            out[r["qid"]] = cv.fuse(s, a, alpha)
    return out


# --------------------------------------------------------------------------- predictions
def predict_array(curves, recs, method, params, k=K):
    """Predictions as a (Q, k, 2) array in seconds, in recs order; NaN where there is none."""
    fn = cv.METHODS[method]
    P = np.full((len(recs), k, 2), np.nan)
    for i, r in enumerate(recs):
        c = curves.get(r["qid"])
        if c is None or len(c) == 0:
            continue
        segs, _ = fn(c, **params)
        sec = cv.to_seconds(segs, r.get("duration"), len(c))[:k]
        P[i, :len(sec)] = sec
    return P


def gt_array(recs):
    return np.array([[r["start"], r["end"]] for r in recs], np.float64).reshape(-1, 2)


def iou_array(P, G):
    """P (Q, k, 2) predictions, G (Q, 2) ground truth -> (Q, k) IoU, 0 where there is no prediction."""
    with np.errstate(invalid="ignore"):
        ps, pe = P[..., 0], P[..., 1]
        gs, ge = G[:, None, 0], G[:, None, 1]
        inter = np.clip(np.minimum(pe, ge) - np.maximum(ps, gs), 0, None)
        union = (pe - ps) + (ge - gs) - inter
        iou = inter / np.maximum(union, 1e-9)
    return np.nan_to_num(iou, nan=0.0)


def evaluate_array(P, recs, ks=(1, 5), ious=cv.IOUS):
    """Same metrics as curves.evaluate, vectorised. Returns (metrics, per-query arrays)."""
    I = iou_array(P, gt_array(recs))
    out = {}
    for k in ks:
        best = I[:, :k].max(1)
        for m in ious:
            out[f"R{k}@{m}"] = round(100 * float((best >= m).mean()), 2) if len(I) else 0.0
    top1 = I[:, 0]
    out["mIoU"] = round(100 * float(top1.mean()), 2) if len(I) else 0.0
    out["n_queries"] = len(recs)
    return out, {"top1": top1, "best5": I[:, :K].max(1)}


def hybrid(P_top, P_rest, k=K, nms_thr=0.5):
    """Top-1 from P_top (threshold proposals: best boundaries), then fill ranks 2..k with
    P_rest (sliding windows: best candidate sets), skipping near-duplicates (IoU >= nms_thr);
    leftover P_top segments fill any remaining slots."""
    Q = len(P_top)
    out = np.full((Q, k, 2), np.nan)
    out[:, 0] = P_top[:, 0]
    n = (~np.isnan(P_top[:, 0, 0])).astype(int)
    cands = np.concatenate([P_rest, P_top[:, 1:]], axis=1)
    for j in range(cands.shape[1]):
        c = cands[:, j]
        with np.errstate(invalid="ignore"):
            inter = np.clip(np.minimum(out[..., 1], c[:, None, 1])
                            - np.maximum(out[..., 0], c[:, None, 0]), 0, None)
            union = (out[..., 1] - out[..., 0]) + (c[:, None, 1] - c[:, None, 0]) - inter
            iou = np.nan_to_num(inter / np.maximum(union, 1e-9), nan=0.0)
        ok = ~np.isnan(c[:, 0]) & (n < k) & (iou < nms_thr).all(1)
        idx = np.flatnonzero(ok)
        out[idx, n[idx]] = c[idx]
        n[idx] += 1
    return out


# --------------------------------------------------------------------------- tuning
_CTX = {}


def _sweep_one(task):
    alpha, method, grid = task
    S, A, recs = _CTX["S"], _CTX["A"], _CTX["recs"]
    F = fused_curves(S, A, recs, alpha)
    res = []
    for p in grid:
        P = predict_array(F, recs, method, p)
        m, _ = evaluate_array(P, recs)
        res.append((m, P))
    return alpha, method, res


def sweep(S, A, recs, alphas, methods=("threshold", "windows"), grids=None, n_jobs=None):
    """Evaluate every (alpha, method, settings) on recs (use val!). Rows keep their predictions
    ("P") so hybrids can be searched afterwards without recomputing anything."""
    grids = grids or cv.GRIDS
    tasks = [(float(a), m, grids[m]) for m in methods for a in alphas]
    _CTX.update(S=S, A=A, recs=recs)
    n_jobs = n_jobs if n_jobs is not None else (os.cpu_count() or 1)
    results = None
    if n_jobs > 1:
        try:
            with mp.get_context("fork").Pool(n_jobs) as pool:
                results = pool.map(_sweep_one, tasks, chunksize=1)
        except Exception as e:  # noqa: BLE001
            print("parallel sweep unavailable, running serially:", e)
    if results is None:
        results = [_sweep_one(t) for t in tasks]
    rows = []
    for alpha, method, res in results:
        for p, (m, P) in zip(grids[method], res):
            rows.append({"alpha": alpha, "method": method, "params": p, **m, "P": P})
    return rows


def _in(alpha, alphas):
    return alphas is None or any(np.isclose(alpha, a) for a in alphas)


def best_row(rows, method, alphas=None, key="R1@0.5"):
    """Best row for a method among the given alphas (ties broken by mIoU)."""
    cand = [r for r in rows if r["method"] == method and _in(r["alpha"], alphas)]
    return max(cand, key=lambda r: (r[key], r["mIoU"]))


def best_hybrid(rows, top_row, recs, alphas=None, key="R5@0.5"):
    """Keep top_row's top-1 and pick the windows row (among alphas) whose candidates make the
    best ranks 2-5 on recs. Returns (windows_row, metrics_of_the_hybrid)."""
    best_r, best_m = None, None
    for r in rows:
        if r["method"] != "windows" or not _in(r["alpha"], alphas):
            continue
        m, _ = evaluate_array(hybrid(top_row["P"], r["P"]), recs)
        if best_m is None or (m[key], m["R5@0.7"]) > (best_m[key], best_m["R5@0.7"]):
            best_r, best_m = r, m
    return best_r, best_m


def alpha_table(rows, key="R1@0.5"):
    """Best score per (alpha, method) over settings, for the alpha-sweep plot."""
    out = {}
    for r in rows:
        k = (r["alpha"], r["method"])
        if k not in out or (r[key], r["mIoU"]) > (out[k][key], out[k]["mIoU"]):
            out[k] = r
    return [{"alpha": a, "method": m, key: r[key], "mIoU": r["mIoU"]}
            for (a, m), r in sorted(out.items())]


# --------------------------------------------------------------------------- statistics
def cluster_bootstrap(x, y, groups, B=5000, seed=0, chunk=500):
    """Difference of means x - y (per-query values, e.g. hit = 1/0 or IoU) in percentage points,
    with a 95% CI from resampling whole videos with replacement (queries of the same video are
    correlated, so resampling queries would overstate confidence).
    p_le_0 = share of resamples where the difference is <= 0 (one-sided)."""
    d = np.asarray(x, np.float64) - np.asarray(y, np.float64)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    G = inv.max() + 1
    sd = np.bincount(inv, weights=d, minlength=G)
    cnt = np.bincount(inv, minlength=G).astype(np.float64)
    rng = np.random.default_rng(seed)
    boots = []
    for s in range(0, B, chunk):
        W = rng.multinomial(G, np.full(G, 1.0 / G), size=min(chunk, B - s)).astype(np.float64)
        boots.append((W @ sd) / np.maximum(W @ cnt, 1e-9))
    boot = np.concatenate(boots)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": round(100 * float(d.mean()), 2), "ci_low": round(100 * float(lo), 2),
            "ci_high": round(100 * float(hi), 2), "p_le_0": round(float((boot <= 0).mean()), 4)}


def duration_category(r, brief=3.0, long_s=10.0, long_frac=0.3):
    """Moment-length category used in the challenging-case analysis."""
    d = r["end"] - r["start"]
    dur = r.get("duration") or r["end"]
    if d < brief:
        return "brief (<3 s)"
    if d > long_s or d > long_frac * dur:
        return "long (>10 s or >30%)"
    return "medium"
