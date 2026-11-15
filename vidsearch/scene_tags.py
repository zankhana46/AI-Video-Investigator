"""Step 6: tag every test query with the challenging conditions named in the proposal.

  * crowded scenes      people counted with YOLOv8n on the 1-fps frames inside the GT moment
  * dark scenes         mean brightness of those frames (bottom share of the test set)
  * viewpoint changes   abrupt drops in similarity between consecutive SigLIP frame embeddings
                        (cached from notebook 01, so no video decoding needed for this one)
  * multi-action        queries describing several actions ("... and ...", "... then ...")
Brief / long moments and opposite-action queries come from fusion.py / curves.py.

Timing convention as everywhere else: index i covers [i, i+1) s and is sampled at i + 0.5 s.
ultralytics / torch are only needed by count_people, and are passed in, so the rest is testable.
"""
import math
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import curves as cv


# --------------------------------------------------------------------------- video
def decode_1fps(path):
    """One BGR frame per second (at i + 0.5 s) in a single sequential pass.
    Returns (frames list, brightness (n,) mean grey level 0-255, duration_s)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 1e-3 or nfr <= 0:
        cap.release()
        raise IOError(f"bad metadata in {path}")
    duration = nfr / fps
    n_sec = max(1, int(math.ceil(duration - 1e-6)))
    want = np.clip(np.floor((np.arange(n_sec) + 0.5) * fps).astype(int), 0, nfr - 1)
    frames, fi, last = [], -1, None
    for target in want:
        while fi < target and cap.grab():
            fi += 1
        if fi >= target:
            ok, frame = cap.retrieve()
            if ok:
                last = frame
        if last is None:
            break
        frames.append(last)
    cap.release()
    if not frames:
        raise IOError(f"no frames decoded from {path}")
    while len(frames) < n_sec:                       # container shorter than its metadata
        frames.append(frames[-1])
    bright = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in frames], np.float32)
    return frames, bright, duration


def frame_at(path, t):
    """The BGR frame at time t seconds (for example figures)."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000.0 * float(t))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def count_people(model, frames, batch=64, conf=0.25, device=None, imgsz=640):
    """Number of people YOLO finds in each frame (COCO class 0)."""
    counts = []
    half = device not in (None, "cpu")
    for i in range(0, len(frames), batch):
        res = model.predict(source=list(frames[i:i + batch]), classes=[0], conf=conf, imgsz=imgsz,
                            device=device, half=half, verbose=False)
        counts.extend(len(r.boxes) for r in res)
    return np.array(counts, np.float32)


def tag_videos(model, videos, device=None, conf=0.25, workers=4, prefetch=8, log_every=200):
    """videos: list of (vid, path). CPU threads decode upcoming videos while YOLO runs.
    Returns ({vid: {"persons": (n,), "bright": (n,), "duration": s}}, failed, stats)."""
    def load(item):
        vid, path = item
        try:
            return vid, decode_1fps(path), None
        except Exception as e:  # noqa: BLE001
            return vid, None, repr(e)

    out, failed, n_frames, t0 = {}, [], 0, time.time()
    it = iter(videos)
    with ThreadPoolExecutor(workers) as ex:
        pending = deque(ex.submit(load, v) for v in [next(it, None) for _ in range(prefetch)] if v)
        while pending:
            vid, v, err = pending.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(ex.submit(load, nxt))
            if v is None:
                failed.append([vid, err])
                continue
            frames, bright, duration = v
            out[vid] = {"persons": count_people(model, frames, conf=conf, device=device),
                        "bright": bright, "duration": float(duration)}
            n_frames += len(frames)
            if log_every and len(out) % log_every == 0:
                print(f"  {len(out)}/{len(videos)} videos | {(time.time() - t0) / 60:.1f} min")
    el = time.time() - t0
    return out, failed, {"videos": len(out), "frames": n_frames, "elapsed_s": round(el, 1),
                         "frames_per_s": round(n_frames / max(el, 1e-9), 1)}


def save_tags(path, tags):
    arrays = {}
    for vid, d in tags.items():
        arrays[f"persons/{vid}"] = d["persons"]
        arrays[f"bright/{vid}"] = d["bright"]
    np.savez(path, **arrays)


def load_tags(path):
    z = np.load(path)
    tags = {}
    for k in z.files:
        kind, vid = k.split("/", 1)
        tags.setdefault(vid, {})[kind] = z[k]
    return tags


# --------------------------------------------------------------------------- SigLIP frames
def load_frame_table(feat_dir, vids):
    """{vid: (T, D)} SigLIP frame features from notebook 01, whatever the npz layout."""
    vids = set(vids)
    for p in sorted(Path(feat_dir).rglob("*.npz")):
        if p.name.startswith("xclip_"):
            continue
        for m in cv._mappings(np.load(p, allow_pickle=False)):
            hit = set(m) & vids
            if len(hit) >= 0.8 * len(vids) and all(np.ndim(m[v]) == 2 for v in list(hit)[:20]):
                return {v: np.asarray(m[v], np.float32) for v in hit}
    return None


def visual_change(F):
    """1 - cosine similarity between consecutive frames: (T-1,), entry i = change from i to i+1."""
    F = np.asarray(F, np.float32)
    F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-8)
    return 1.0 - (F[1:] * F[:-1]).sum(1)


# --------------------------------------------------------------------------- query tags
MULTI_ACTION = re.compile(r"\b(and then|then|and|while|before|after)\b")


def is_multi_action(query):
    """Query describes more than one action, e.g. 'opens the door and walks in'."""
    q = query.lower()
    return bool(MULTI_ACTION.search(q)) and len(re.findall(r"[a-z]+", q)) >= 5


def gt_bins(r, n):
    """Indices of the 1-s bins whose centre lies inside the GT (at least one)."""
    t = np.arange(n) + 0.5
    m = np.flatnonzero((t >= r["start"]) & (t < r["end"]))
    if len(m) == 0:
        m = np.array([min(n - 1, max(0, int((r["start"] + r["end"]) / 2)))])
    return m


def query_tags(recs, tags, changes):
    """Per-query measurements: mean / max people in the GT, mean brightness, the largest
    consecutive-frame change inside the GT (NaN if the GT spans fewer than 2 bins)."""
    out = []
    for r in recs:
        row = {"qid": r["qid"], "vid": r["vid"], "gt_len": r["end"] - r["start"],
               "persons_mean": np.nan, "persons_max": np.nan, "bright": np.nan, "max_change": np.nan,
               "multi_action": is_multi_action(r["query"])}
        t = tags.get(r["vid"])
        if t is not None:
            b = gt_bins(r, len(t["persons"]))
            row["persons_mean"] = float(t["persons"][b].mean())
            row["persons_max"] = float(t["persons"][b].max())
            row["bright"] = float(t["bright"][b].mean())
        d = changes.get(r["vid"])
        if d is not None and len(d):
            b = gt_bins(r, len(d) + 1)
            bs = set(b.tolist())
            inside = [i for i in b if i + 1 in bs]              # transitions i -> i+1 within the GT
            if inside:
                row["max_change"] = float(d[inside].max())
        out.append(row)
    return out
