"""Score curves -> predictions -> metrics, plus a CPU grid search for tuning on val."""
from itertools import product

import numpy as np

from .metrics import evaluate
from .proposals import smooth, zscore, threshold_segments, window_proposals


def score_curves(records, video_feats, text_feats):
    """Cosine similarity of every frame with the query: {qid: array[T]}.
    Features are L2-normalized, so a dot product is the cosine."""
    curves = {}
    for r in records:
        V = video_feats[r["vid"]].astype(np.float32)
        q = text_feats[r["qid"]].astype(np.float32)
        curves[r["qid"]] = V @ q
    return curves


def predict(records, curves, method="windows", fps=1.0, smooth_k=1, **kw):
    preds = {}
    for r in records:
        s = zscore(smooth(curves[r["qid"]], smooth_k))
        if method == "windows":
            preds[r["qid"]] = window_proposals(s, fps=fps, duration=r["duration"], **kw)
        elif method == "threshold":
            preds[r["qid"]] = threshold_segments(s, fps=fps, duration=r["duration"], **kw)
        else:
            raise ValueError(method)
    return preds


def grid_search(records, curves, grid, method="windows", fps=1.0,
                select=("R1@0.5", "R1@0.7", "mIoU")):
    """Try every combination in `grid` ({param: [values]}). Sorted best first by
    the sum of the `select` metrics. Use on VAL only."""
    rows = []
    for values in product(*grid.values()):
        cfg = dict(zip(grid.keys(), values))
        summary, _ = evaluate(predict(records, curves, method, fps=fps, **cfg), records)
        rows.append({**cfg, **summary, "select": sum(summary[k] for k in select)})
    return sorted(rows, key=lambda r: -r["select"])
