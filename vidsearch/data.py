"""Charades-STA annotation loading, cleaning, and train/val/test splits."""
import csv
import json
import random
from pathlib import Path


def parse_sta(path):
    """Parse a Charades-STA file. Each line: 'VID start end##query sentence'."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "##" not in line:
                continue
            meta, query = line.split("##", 1)
            vid, start, end = meta.split()
            records.append({"vid": vid, "start": float(start), "end": float(end),
                            "query": query.strip().rstrip(".").strip()})
    return records


def load_durations(*csv_paths):
    """Video lengths (seconds) from Charades_v1_train.csv / Charades_v1_test.csv."""
    durations = {}
    for p in csv_paths:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                durations[row["id"]] = float(row["length"])
    return durations


def attach_durations(records, durations, split):
    """Add duration, clip GT to [0, duration], drop broken rows, assign query ids."""
    out, dropped = [], 0
    for i, r in enumerate(records):
        dur = durations.get(r["vid"])
        if dur is None:
            dropped += 1
            continue
        s, e = max(0.0, r["start"]), min(r["end"], dur)
        if e - s < 0.1:
            dropped += 1
            continue
        out.append({**r, "start": s, "end": e, "duration": dur,
                    "split": split, "qid": f"{split}_{i:05d}"})
    return out, dropped


def make_val_split(train_records, n_val_videos=300, seed=0):
    """Hold out whole videos from train for tuning (Charades-STA has no val split)."""
    vids = sorted({r["vid"] for r in train_records})
    random.Random(seed).shuffle(vids)
    val_vids = set(vids[:n_val_videos])
    return [{**r, "split": "val", "qid": r["qid"].replace("train_", "val_")}
            for r in train_records if r["vid"] in val_vids]


def save_jsonl(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
