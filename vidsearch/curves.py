"""Curve utilities for Steps 3-4 (activity evidence and fusion).

Everything here is plain numpy, so it runs on CPU (Kaggle or your laptop) in seconds:
  * load_jsonl          read annotations/{val,test}.jsonl (field names auto-detected)
  * threshold_proposals the proposal's baseline: keep high frames, merge neighbours
  * window_proposals    multi-scale sliding windows scored by contrast (inside - outside)
  * evaluate            R1/R5 @ IoU 0.3/0.5/0.7 + mIoU (standard Charades-STA metrics)
  * tune                grid search on val, pick by R1@0.5 (ties broken by mIoU)
  * load_frame_curves   rebuild SigLIP s(t) curves from notebook 01's cached features
  * find_opposite_pairs / direction_accuracy   "open vs close" style diagnostic

Convention (same as the rest of the repo): curve index i covers the time bin [i, i+1) s.
This module is self-contained on purpose, so frame-only and activity-only results in the
Step 3 notebook are scored by exactly the same code.
"""
import csv
import itertools
import json
import re
import time
from pathlib import Path

import numpy as np

IOUS = (0.3, 0.5, 0.7)

# --------------------------------------------------------------------------- annotations
_KEYS = {
    "vid": ("vid", "video_id", "video", "vid_id"),
    "qid": ("qid", "query_id", "id", "uid"),
    "query": ("query", "sentence", "text", "description"),
    "start": ("start", "st", "t_start", "gt_start"),
    "end": ("end", "en", "et", "t_end", "gt_end"),
    "duration": ("duration", "video_duration", "length", "dur", "vid_duration"),
}


def normalize_record(r, i, split):
    """Map one annotation dict onto {qid, vid, query, start, end, duration?}."""
    out = {}
    for k, cands in _KEYS.items():
        for c in cands:
            if c in r:
                out[k] = r[c]
                break
    if "start" not in out or "end" not in out:
        for c in ("timestamp", "gt", "segment", "times"):
            if c in r and len(r[c]) == 2:
                out["start"], out["end"] = r[c]
                break
    missing = [k for k in ("vid", "query", "start", "end") if k not in out]
    if missing:
        raise KeyError(f"annotation record has no field for {missing}; its keys are {list(r)}")
    if "qid" not in out:
        out["qid"] = f"{split}_{i:05d}"
    out["qid"], out["vid"], out["query"] = str(out["qid"]), str(out["vid"]), str(out["query"])
    out["start"], out["end"] = float(out["start"]), float(out["end"])
    if out.get("duration") is not None:
        out["duration"] = float(out["duration"])
    else:
        out.pop("duration", None)
    return out


def load_jsonl(path, split=None):
    path = Path(path)
    split = split or path.stem
    recs = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                recs.append(normalize_record(json.loads(line), i, split))
    return recs


def videos_of(recs):
    """Video ids in first-seen order."""
    return list(dict.fromkeys(r["vid"] for r in recs))


# --------------------------------------------------------------------------- small helpers
def zscore(x):
    x = np.asarray(x, np.float64)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-8 else 1.0)


def smooth(x, k):
    """Centred moving average (edges renormalised). k <= 1 is a no-op; even k is rounded down."""
    x = np.asarray(x, np.float64)
    k = min(int(k), len(x))
    if k % 2 == 0:
        k -= 1
    if k <= 1:
        return x.copy()
    w = np.ones(k)
    return np.convolve(x, w, mode="same") / np.convolve(np.ones_like(x), w, mode="same")


def seg_mean(curve, start, end):
    """Mean of the curve over the bins whose centre lies in [start, end) seconds."""
    c = np.asarray(curve, np.float64)
    t = np.arange(len(c)) + 0.5
    m = (t >= start) & (t < end)
    if not m.any():
        return float(c[min(len(c) - 1, max(0, int((start + end) / 2)))])
    return float(c[m].mean())


# --------------------------------------------------------------------------- metrics
def tiou(pred, gt):
    """Temporal IoU of each row of pred (N, 2) with one segment gt (2,)."""
    pred = np.asarray(pred, np.float64).reshape(-1, 2)
    g0, g1 = float(gt[0]), float(gt[1])
    inter = np.clip(np.minimum(pred[:, 1], g1) - np.maximum(pred[:, 0], g0), 0, None)
    union = (pred[:, 1] - pred[:, 0]) + (g1 - g0) - inter
    return inter / np.maximum(union, 1e-9)


def evaluate(preds, recs, ks=(1, 5), ious=IOUS):
    """preds: {qid: [(start_s, end_s), ...] best first}.
    Returns (metrics dict in %, {qid: IoU of the top-1 prediction})."""
    hits = {f"R{k}@{m}": [] for k in ks for m in ious}
    top1 = {}
    for r in recs:
        p = preds.get(r["qid"])
        if not p:
            for key in hits:
                hits[key].append(0.0)
            top1[r["qid"]] = 0.0
            continue
        iou = tiou(np.asarray(p, np.float64)[: max(ks)], (r["start"], r["end"]))
        for k in ks:
            best = iou[:k].max()
            for m in ious:
                hits[f"R{k}@{m}"].append(float(best >= m))
        top1[r["qid"]] = float(iou[0])
    out = {k: round(100 * float(np.mean(v)), 2) if v else 0.0 for k, v in hits.items()}
    out["mIoU"] = round(100 * float(np.mean(list(top1.values()))), 2) if top1 else 0.0
    out["n_queries"] = len(recs)
    return out, top1


# --------------------------------------------------------------------------- proposals
def nms(segs, scores, thr=0.5, topk=5):
    """Greedy 1-D non-maximum suppression. segs (N, 2), scores (N,)."""
    segs = np.asarray(segs, np.float64).reshape(-1, 2)
    scores = np.asarray(scores, np.float64)
    order = np.argsort(-scores, kind="stable")
    segs, scores = segs[order], scores[order]
    alive = np.ones(len(segs), bool)
    keep = []
    while len(keep) < topk:
        rest = np.flatnonzero(alive)
        if len(rest) == 0:
            break
        i = rest[0]
        keep.append(i)
        alive &= tiou(segs, segs[i]) < thr
        alive[i] = False
    return segs[keep], scores[keep]


def window_proposals(curve, ctx=0.5, min_len=2, max_frac=1.0, smooth_k=1,
                     lengths=None, nms_thr=0.5, topk=5):
    """Multi-scale sliding windows (stride 1 bin) scored by contrast.

    score = mean(z inside the window) - mean(z in its surroundings)
    ctx      surrounding context on each side as a fraction of the window length;
             None = compare against *all* bins outside the window.
    lengths  window lengths in bins; default = every length from min_len to max_frac * n.
    Returns (segs (k, 2) in bin units [start, end), scores (k,)).
    """
    x = smooth(zscore(curve), smooth_k)
    n = len(x)
    if n <= 1:
        return np.array([[0.0, float(max(n, 1))]]), np.array([0.0])
    cs = np.concatenate([[0.0], np.cumsum(x)])
    total = cs[-1]
    if lengths is None:
        top = max(min_len, int(round(max_frac * n)))
        lengths = range(max(1, int(min_len)), min(n, top) + 1)
    all_segs, all_scores = [], []
    for L in lengths:
        L = int(L)
        if L < 1 or L > n:
            continue
        st = np.arange(0, n - L + 1)
        en = st + L
        inside_sum = cs[en] - cs[st]
        inside = inside_sum / L
        if ctx is None:
            n_out = n - L
            outside = (total - inside_sum) / n_out if n_out > 0 else np.zeros_like(inside)
        else:
            c = max(1, int(round(L * float(ctx))))
            a0 = np.maximum(0, st - c)
            b1 = np.minimum(n, en + c)
            cnt = (st - a0) + (b1 - en)
            ctx_sum = (cs[st] - cs[a0]) + (cs[b1] - cs[en])
            outside = np.where(cnt > 0, ctx_sum / np.maximum(cnt, 1), x.mean())
        all_segs.append(np.stack([st, en], 1))
        all_scores.append(inside - outside)
    if not all_segs:
        return np.array([[0.0, float(n)]]), np.array([0.0])
    return nms(np.concatenate(all_segs), np.concatenate(all_scores), nms_thr, topk)


def threshold_proposals(curve, thr=0.0, smooth_k=3, rank="mean", min_len=1,
                        nms_thr=0.5, topk=5):
    """The proposal's baseline: keep bins whose z-scored (smoothed) score is above thr,
    merge neighbouring kept bins into segments, rank segments by
    rank="mean" (mean z inside) or rank="area" (sum of z - thr inside, favours longer runs)."""
    x = smooth(zscore(curve), smooth_k)
    n = len(x)
    on = x > thr
    segs, scores = [], []
    i = 0
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            if j - i >= min_len:
                segs.append((i, j))
                scores.append(x[i:j].mean() if rank == "mean" else (x[i:j] - thr).sum())
            i = j
        else:
            i += 1
    if not segs:
        k = int(np.argmax(x))
        segs, scores = [(max(0, k - 1), min(n, k + 2))], [float(x[k])]
    return nms(np.array(segs, np.float64), np.array(scores), nms_thr, topk)


METHODS = {"threshold": threshold_proposals, "windows": window_proposals}


def to_seconds(segs, duration, n):
    """Bin-unit segments -> seconds, clipped to the video length, order preserved."""
    end_cap = float(duration) if duration else float(n)
    out = []
    for a, b in np.asarray(segs, np.float64).reshape(-1, 2):
        a, b = max(0.0, min(a, end_cap)), max(0.0, min(b, end_cap))
        if b > a:
            out.append((a, b))
    return out or [(0.0, end_cap)]


def predict(curves, recs, method, params):
    fn = METHODS[method]
    preds = {}
    for r in recs:
        c = curves.get(r["qid"])
        if c is None or len(c) == 0:
            continue
        segs, _ = fn(c, **params)
        preds[r["qid"]] = to_seconds(segs, r.get("duration"), len(c))
    return preds


def grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*axes.values())]


GRIDS = {
    "threshold": grid(thr=[-0.25, 0.0, 0.25, 0.5, 0.75, 1.0], smooth_k=[1, 3, 5],
                      rank=["mean", "area"]),
    "windows": grid(ctx=[0.25, 0.5, 1.0, None], smooth_k=[1, 3], min_len=[2, 4],
                    max_frac=[0.6, 1.0]),
}


def tune(curves, recs, method, param_grid=None, key="R1@0.5"):
    """Grid search on recs (use val!). Returns (best_params, rows sorted best first)."""
    param_grid = param_grid if param_grid is not None else GRIDS[method]
    rows = []
    for p in param_grid:
        m, _ = evaluate(predict(curves, recs, method, p), recs)
        rows.append({**p, **m})
    rows.sort(key=lambda r: (r[key], r["mIoU"]), reverse=True)
    best = {k: rows[0][k] for k in param_grid[0]}
    return best, rows


# --------------------------------------------------------------------------- diagnostics
def inside_vs_outside(curves, recs):
    """% of queries whose curve is higher inside the GT than outside it (50% = no signal)."""
    wins = []
    for r in recs:
        c = curves.get(r["qid"])
        if c is None or len(c) < 2:
            continue
        t = np.arange(len(c)) + 0.5
        m = (t >= r["start"]) & (t < r["end"])
        if m.all() or not m.any():
            continue
        wins.append(np.mean(c[m]) > np.mean(c[~m]))
    return (round(100 * float(np.mean(wins)), 1) if wins else float("nan")), len(wins)


OPPOSITES = [
    ("open/close", r"\bopen", r"\bclos", True),
    ("turn on/off", r"\b(turn|switch)\w*\b.*\bon\b", r"\b(turn|switch)\w*\b.*\boff\b", True),
    ("sit/stand", r"\bsit", r"\bstand|\bget\w* up\b", False),
]
_STOP = set("""person persons their them they then with into onto from while there this that
after before some something another other again over back down front inside outside around
open opens opening opened close closes closing closed turn turns turning turned switch
switches switched sits sitting stand stands standing stood gets getting puts putting takes
taking starts start begins begin walks walking light lights""".split())


def _content_words(q):
    return {w for w in re.findall(r"[a-z]+", q.lower()) if len(w) >= 4 and w not in _STOP}


def find_opposite_pairs(recs, max_gt_iou=0.3):
    """Pairs of queries on the same video describing opposite actions (e.g. opening vs closing
    the same window) whose GT moments barely overlap. A model that only recognises objects
    scores both queries alike; one that sees motion should not."""
    by_vid = {}
    for r in recs:
        by_vid.setdefault(r["vid"], []).append(r)
    pairs = []
    for rs in by_vid.values():
        for name, pa, pb, need_obj in OPPOSITES:
            A = [r for r in rs if re.search(pa, r["query"].lower()) and not re.search(pb, r["query"].lower())]
            B = [r for r in rs if re.search(pb, r["query"].lower()) and not re.search(pa, r["query"].lower())]
            for a in A:
                for b in B:
                    if tiou([(a["start"], a["end"])], (b["start"], b["end"]))[0] >= max_gt_iou:
                        continue
                    if need_obj and not (_content_words(a["query"]) & _content_words(b["query"])):
                        continue
                    pairs.append((name, a, b))
    return pairs


def direction_accuracy(curves, pairs):
    """% of paired queries whose curve is higher on their *own* moment than on the opposite one."""
    ok = []
    for _, a, b in pairs:
        for q, own, other in ((a, a, b), (b, b, a)):
            c = curves.get(q["qid"])
            if c is None:
                continue
            ok.append(seg_mean(c, own["start"], own["end"]) > seg_mean(c, other["start"], other["end"]))
    return (round(100 * float(np.mean(ok)), 1) if ok else float("nan")), len(ok)


# --------------------------------------------------------------------------- SigLIP curves
def _l2n(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _mappings(z):
    """All id -> array tables we can read out of one npz, whatever its layout."""
    arrays = {k: z[k] for k in z.files}
    maps = [dict(arrays)]  # layout 1: one array per id
    strs = {k: v for k, v in arrays.items() if v.dtype.kind in "US" and v.ndim == 1}
    nums = {k: v for k, v in arrays.items() if v.dtype.kind in "fiu" and v.ndim >= 1}
    for _, ids in strs.items():
        ids = [str(i) for i in ids]
        for _, v in nums.items():  # layout 2: ids + stacked rows
            if v.dtype.kind == "f" and v.ndim == 2 and len(v) == len(ids):
                maps.append(dict(zip(ids, v)))
        for _, off in nums.items():  # layout 3: ids + offsets + concatenated frames
            if off.dtype.kind in "iu" and off.ndim == 1 and len(off) == len(ids) + 1:
                for _, v in nums.items():
                    if v.dtype.kind == "f" and v.ndim == 2 and len(v) == int(off[-1]):
                        maps.append({i: v[off[j]:off[j + 1]] for j, i in enumerate(ids)})
    return maps


def load_frame_curves(feat_dir, recs, verbose=True):
    """Rebuild SigLIP frame curves s(t) = cos(frame_t, query) from notebook 01's cached features.

    The npz layout is detected from its contents (which arrays are keyed by video id, query id
    or query text), so this works whatever names notebook 01 used. Returns {qid: curve} or
    None, printing what it found so the layout can be fixed if it is not recognised."""
    vids = {r["vid"] for r in recs}
    qids = {r["qid"] for r in recs}
    texts = {r["query"] for r in recs}
    frames = by_qid = by_text = curves_by_qid = None
    for p in sorted(Path(feat_dir).rglob("*.npz")):
        if p.name.startswith("xclip_"):
            continue
        try:
            z = np.load(p, allow_pickle=False)
            maps = _mappings(z)
            if verbose:
                ex = {k: tuple(z[k].shape) for k in z.files[:3]}
                print(f"  {p.name}: {len(z.files)} arrays, e.g. {ex}")
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  {p.name}: could not read ({e})")
            continue
        for m in maps:
            keys = set(m)
            if frames is None and len(keys & vids) >= 0.8 * len(vids) and \
                    all(np.ndim(m[v]) == 2 for v in list(keys & vids)[:20]):
                frames = m
            elif len(keys & qids) >= 0.8 * len(qids):
                sample = [np.asarray(m[q]).squeeze() for q in list(keys & qids)[:50]]
                if all(s.ndim == 1 for s in sample) and len({len(s) for s in sample}) == 1:
                    by_qid = by_qid or m          # fixed-size vectors -> text embeddings
                else:
                    curves_by_qid = curves_by_qid or m   # varying lengths -> already curves
            elif len(keys & texts) >= 0.8 * len(texts):
                by_text = by_text or m
    if frames is None and curves_by_qid is not None:
        return {r["qid"]: np.asarray(curves_by_qid[r["qid"]], np.float64).ravel()
                for r in recs if r["qid"] in curves_by_qid}
    text = by_qid if by_qid is not None else by_text
    if frames is None or text is None:
        if verbose:
            print("  -> layout not recognised (frames found: %s, query features found: %s)"
                  % (frames is not None, text is not None))
        return None
    out = {}
    cache = {}
    for r in recs:
        F = frames.get(r["vid"])
        t = text.get(r["qid"]) if text is by_qid else text.get(r["query"])
        if F is None or t is None:
            continue
        t = np.asarray(t).reshape(-1)
        if np.shape(F)[1] != len(t):
            if verbose:
                print(f"  -> dimension mismatch: frames {np.shape(F)}, query {t.shape}")
            return None
        if r["vid"] not in cache:
            cache[r["vid"]] = _l2n(F)
        out[r["qid"]] = (cache[r["vid"]] @ _l2n(t)).astype(np.float64)
    return out


def align(a, b):
    """Trim two curves of the same video to a common length (they can differ by a bin)."""
    n = min(len(a), len(b))
    return np.asarray(a[:n], np.float64), np.asarray(b[:n], np.float64)


def fuse(s, a, alpha):
    """f(t) = alpha * z(s) + (1 - alpha) * z(a), on the common length."""
    s, a = align(s, a)
    return alpha * zscore(s) + (1 - alpha) * zscore(a)


# --------------------------------------------------------------------------- logging
def append_csv(path, row):
    """Append one row to a CSV, widening the header if new columns appear."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    rows.append({k: ("" if v is None else v) for k, v in row.items()})
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def log_row(path, method, split, metrics, config, **extra):
    append_csv(path, {"time": time.strftime("%Y-%m-%d %H:%M"), "method": method, "split": split,
                      **{k: v for k, v in metrics.items()}, **extra,
                      "config": json.dumps(config, default=str)})
