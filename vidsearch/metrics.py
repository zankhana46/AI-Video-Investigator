"""Moment-retrieval metrics: R@k at IoU thresholds, mIoU, plus 1-D NMS."""
import numpy as np


def iou_1d(preds, gt):
    """IoU between each predicted segment [N,2] and one GT segment (s, e)."""
    preds = np.asarray(preds, dtype=float).reshape(-1, 2)
    inter = np.clip(np.minimum(preds[:, 1], gt[1]) - np.maximum(preds[:, 0], gt[0]), 0, None)
    union = np.maximum(preds[:, 1], gt[1]) - np.minimum(preds[:, 0], gt[0])
    return inter / np.maximum(union, 1e-8)


def nms_1d(proposals, iou_thr=0.5, top_k=5):
    """Greedy NMS. proposals: list of (start, end, score). Returns top_k, best first."""
    props = sorted(proposals, key=lambda p: -p[2])
    keep = []
    for p in props:
        if all(iou_1d([p[:2]], k[:2])[0] < iou_thr for k in keep):
            keep.append(p)
            if len(keep) == top_k:
                break
    return keep


def evaluate(predictions, records, ks=(1, 5), iou_thrs=(0.3, 0.5, 0.7)):
    """predictions: {qid: [(start, end, score), ...] sorted best first}.

    Returns (summary dict in %, per-query list). Keep the per-query results for
    the challenging-case breakdown later.
    """
    per_query = []
    for r in records:
        preds = predictions.get(r["qid"], [])
        ious = iou_1d([p[:2] for p in preds], (r["start"], r["end"])) if preds else np.zeros(0)
        row = {"qid": r["qid"], "iou_top1": float(ious[0]) if len(ious) else 0.0}
        for k in ks:
            for t in iou_thrs:
                row[f"R{k}@{t}"] = float((ious[:k] >= t).any()) if len(ious) else 0.0
        per_query.append(row)

    summary = {key: float(100 * np.mean([q[key] for q in per_query]))
               for key in per_query[0] if key.startswith("R")}
    summary["mIoU"] = float(100 * np.mean([q["iou_top1"] for q in per_query]))
    summary["n_queries"] = len(per_query)
    return summary, per_query
