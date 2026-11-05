"""Turn a per-frame relevance curve into ranked time segments (training-free)."""
import numpy as np


def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


def smooth(x, k=1):
    """Moving average over k frames (k odd). k<=1 returns x unchanged."""
    x = np.asarray(x, dtype=np.float32)
    if k <= 1 or len(x) < 2:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")[:len(x)]


def nms(props, iou_thr=0.5, top_k=5):
    """props: array [N,3] of (start, end, score). Greedy 1-D NMS, best first."""
    if len(props) == 0:
        return []
    props = props[np.argsort(-props[:, 2])]
    keep = []
    for p in props:
        if keep:
            k = np.array(keep)
            inter = np.clip(np.minimum(k[:, 1], p[1]) - np.maximum(k[:, 0], p[0]), 0, None)
            union = np.maximum(k[:, 1], p[1]) - np.minimum(k[:, 0], p[0])
            if (inter / np.maximum(union, 1e-8) >= iou_thr).any():
                continue
        keep.append(p)
        if len(keep) == top_k:
            break
    return [(float(s), float(e), float(sc)) for s, e, sc in keep]


def threshold_segments(scores, fps=1.0, duration=None, thr=0.5, top_k=5):
    """Baseline (i) from the proposal: keep frames whose z-score >= thr, merge
    consecutive kept frames into segments, rank segments by mean z-score."""
    z = zscore(scores)
    T = len(z)
    duration = duration or T / fps
    mask = np.concatenate([[False], z >= thr, [False]])
    edges = np.flatnonzero(np.diff(mask.astype(int)))
    starts, ends = edges[::2], edges[1::2]
    if len(starts) == 0:                      # nothing above threshold: best single frame
        i = int(np.argmax(z))
        return [(i / fps, min((i + 1) / fps, duration), float(z[i]))]
    segs = [(s / fps, min(e / fps, duration), float(z[s:e].mean())) for s, e in zip(starts, ends)]
    return sorted(segs, key=lambda p: -p[2])[:top_k]


def window_proposals(scores, fps=1.0, duration=None, lengths_s=(2, 4, 8, 16, 32),
                     stride=1, ctx=0.5, top_k=5, nms_thr=0.5):
    """Baseline (ii): multi-scale sliding windows scored by contrast,
    score = mean(inside) - mean(neighbouring context of width ctx*L), then NMS.
    ctx=0 compares against the whole-video mean instead."""
    s = np.asarray(scores, dtype=np.float64)
    T = len(s)
    duration = duration or T / fps
    cs = np.concatenate([[0.0], np.cumsum(s)])
    gmean = s.mean()
    all_props = []
    for Ls in sorted(set(min(max(1, int(round(L * fps))), T) for L in lengths_s)):
        starts = np.arange(0, T - Ls + 1, stride)
        ends = starts + Ls
        inside = (cs[ends] - cs[starts]) / Ls
        c = int(round(Ls * ctx)) if ctx > 0 else 0
        lo, hi = np.maximum(starts - c, 0), np.minimum(ends + c, T)
        out_n = (starts - lo) + (hi - ends)
        out_sum = (cs[starts] - cs[lo]) + (cs[hi] - cs[ends])
        outside = np.where(out_n > 0, out_sum / np.maximum(out_n, 1), gmean)
        all_props.append(np.stack([starts / fps, np.minimum(ends / fps, duration),
                                   inside - outside], axis=1))
    return nms(np.concatenate(all_props), nms_thr, top_k)
