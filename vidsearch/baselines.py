"""Sanity-check baselines for the evaluation pipeline."""
import numpy as np


def oracle_predictions(records):
    return {r["qid"]: [(r["start"], r["end"], 1.0)] for r in records}


def random_predictions(records, k=5, seed=0):
    rng = np.random.default_rng(seed)
    preds = {}
    for r in records:
        d = r["duration"]
        out = []
        for _ in range(k):
            length = rng.uniform(0.1, 1.0) * d
            start = rng.uniform(0, d - length)
            out.append((start, start + length, 0.0))
        preds[r["qid"]] = out
    return preds


def prior_predictions(train_records, test_records):
    """Always predict the average normalized GT position: measures dataset bias."""
    s = np.mean([r["start"] / r["duration"] for r in train_records])
    e = np.mean([r["end"] / r["duration"] for r in train_records])
    return {r["qid"]: [(s * r["duration"], e * r["duration"], 0.0)] for r in test_records}
