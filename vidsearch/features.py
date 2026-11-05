"""GPU feature extraction with image-text models (SigLIP / CLIP) via Hugging Face.

Video features: {vid: float16 array [T, D]} at `fps`, L2-normalized.
Text features:  {qid: float16 array [D]}, L2-normalized.
Everything is saved in single .npz files (Kaggle keeps max 500 output files).
"""
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

from .video_io import sample_frames


def load_model(name="google/siglip-base-patch16-224", device="cuda"):
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(name)
    return model, processor


def _as_tensor(out):
    """Some transformers versions return an output object instead of a tensor."""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        if getattr(out, attr, None) is not None:
            return getattr(out, attr)
    return out[0]


@torch.no_grad()
def encode_images(model, processor, frames, batch_size=64, device="cuda"):
    out = []
    for i in range(0, len(frames), batch_size):
        px = processor(images=list(frames[i:i + batch_size]), return_tensors="pt")["pixel_values"]
        e = _as_tensor(model.get_image_features(pixel_values=px.to(device, model.dtype)))
        out.append(F.normalize(e.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def encode_texts(model, processor, texts, batch_size=256, device="cuda", template="{}"):
    # SigLIP was trained with fixed-length padding; CLIP uses dynamic padding.
    padding = "max_length" if "siglip" in model.config.model_type else True
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [template.format(t) for t in texts[i:i + batch_size]]
        inputs = processor(text=batch, padding=padding, truncation=True, return_tensors="pt").to(device)
        e = _as_tensor(model.get_text_features(**inputs))
        out.append(F.normalize(e.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


def extract_video_features(vids, video_dir, model, processor, fps=1.0, batch_size=64,
                           max_side=320, workers=4, prefetch=8, device="cuda"):
    """CPU threads decode upcoming videos while the GPU encodes the current one."""
    def load(v):
        try:
            return v, sample_frames(Path(video_dir) / f"{v}.mp4", fps=fps, max_side=max_side)
        except Exception as e:                       # keep going if one video is broken
            return v, e

    feats, failed, n_frames = {}, [], 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    it = iter(vids)
    with ThreadPoolExecutor(workers) as ex:
        queue = deque(ex.submit(load, v) for _, v in zip(range(prefetch), it))
        for _ in tqdm(range(len(vids)), desc="videos"):
            v, res = queue.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                queue.append(ex.submit(load, nxt))
            if isinstance(res, Exception):
                failed.append(v)
                continue
            frames, _ = res
            feats[v] = encode_images(model, processor, frames, batch_size, device).astype(np.float16)
            n_frames += len(frames)
    secs = time.perf_counter() - t0
    stats = {"seconds": secs, "n_videos": len(feats), "n_frames": n_frames,
             "video_minutes": n_frames / fps / 60, "failed": failed,
             "sec_per_video_minute": secs / max(n_frames / fps / 60, 1e-9),
             "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0}
    return feats, stats


def save_video_features(feats, path, meta=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **feats)
    Path(str(path).replace(".npz", ".json")).write_text(json.dumps(meta or {}, indent=1))


def load_video_features(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def save_text_features(qids, emb, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, qids=np.array(qids), emb=emb.astype(np.float16))


def load_text_features(path):
    d = np.load(path)
    return dict(zip(d["qids"].tolist(), d["emb"]))
