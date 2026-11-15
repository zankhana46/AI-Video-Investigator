"""Step 7: the AI Video Investigator demo.

Upload (or pick) a video, it is indexed once with both models:
    SigLIP  one embedding per second   -> frame evidence s(t)
    X-CLIP  one embedding per 4-s clip -> activity evidence a(t)
then every typed query is answered in milliseconds with exactly the system from Step 4:
    fused f(t) = alpha * z(s) + (1 - alpha) * z(a); top-1 from threshold proposals,
    ranks 2-5 from sliding windows (the hybrid), using the settings saved in step4_summary.json.

The search logic (predict_segments) is plain numpy and is the same code path the benchmark used.
torch / transformers / gradio are imported lazily.
"""
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from . import curves as cv
from . import fusion as fu
from . import scene_tags as st
from . import xclip_features as xf

DEFAULT_CONFIG = {
    "alpha_top1": 0.5, "threshold_params": {"thr": 0.0, "smooth_k": 5, "rank": "mean"},
    "alpha_ranks_2_5": 0.5, "windows_params": {"ctx": 0.25, "smooth_k": 3, "min_len": 4, "max_frac": 0.6},
    "activity_model": "microsoft/xclip-base-patch16-zero-shot", "activity_variant": "plain",
    "frame_model": "google/siglip-base-patch16-224", "clip_sec": 4.0,
}


def load_final_config(summary_path=None, frame_model=None):
    """Final configuration chosen on val in Step 4 (falls back to defaults if not found)."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if summary_path and Path(summary_path).exists():
        cfg.update(json.load(open(summary_path)).get("final_config", {}))
    if frame_model:
        cfg["frame_model"] = frame_model
    if "siglip" not in str(cfg.get("frame_model", "")).lower() or "/" not in str(cfg["frame_model"]):
        cfg["frame_model"] = DEFAULT_CONFIG["frame_model"]
    return cfg


def find_model_name(feat_dir, keyword="siglip"):
    """Look for the model name notebook 01 stored next to its features (any JSON string)."""
    for p in sorted(Path(feat_dir).rglob("*.json")):
        try:
            found = []

            def walk(o):
                if isinstance(o, dict):
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
                elif isinstance(o, str) and keyword in o.lower() and "/" in o and len(o) < 100:
                    found.append(o)
            walk(json.load(open(p)))
            if found:
                return found[0]
        except Exception:  # noqa: BLE001
            continue
    return None


def load_qid_table(feat_dir, qids, skip_prefix="xclip_"):
    """{qid: vector} for cached query embeddings (e.g. notebook 01's SigLIP text features)."""
    qids = set(qids)
    for p in sorted(Path(feat_dir).rglob("*.npz")):
        if p.name.startswith(skip_prefix):
            continue
        for m in cv._mappings(np.load(p, allow_pickle=False)):
            hit = set(m) & qids
            if hit and len(hit) >= 0.8 * len(qids) and all(np.ndim(m[q]) == 1 for q in list(hit)[:20]):
                return {q: np.asarray(m[q], np.float32) for q in hit}
    return None


# --------------------------------------------------------------------------- search (numpy)
def _pad(segs, k=fu.K):
    P = np.full((1, k, 2), np.nan)
    segs = list(segs)[:k]
    if segs:
        P[0, :len(segs)] = segs
    return P


def predict_segments(s, a, cfg, duration, k=fu.K):
    """Top-k moments for one query on one video, exactly as in the Step 4 benchmark."""
    f_top = cv.fuse(s, a, cfg["alpha_top1"])
    f_rest = cv.fuse(s, a, cfg["alpha_ranks_2_5"])
    seg_t, _ = cv.threshold_proposals(f_top, **cfg["threshold_params"])
    seg_w, _ = cv.window_proposals(f_rest, **cfg["windows_params"])
    P = fu.hybrid(_pad(cv.to_seconds(seg_t, duration, len(f_top))),
                  _pad(cv.to_seconds(seg_w, duration, len(f_rest))), k=k)[0]
    segs = [(float(x0), float(x1)) for x0, x1 in P if not np.isnan(x0)]
    z = cv.zscore(f_top)
    scores = [round(cv.seg_mean(z, x0, x1), 2) for x0, x1 in segs]
    return segs, scores, f_top


def cut_clips(video_path, segments, out_dir, pad=0.0):
    """One browser-playable mp4 per segment (H.264, no audio). Returns the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    paths = []
    for i, (s0, s1) in enumerate(segments):
        out = out_dir / f"moment_{i + 1}_{s0:.0f}-{s1:.0f}s_{int(time.time() * 1000) % 100000}.mp4"
        start, dur = max(0.0, s0 - pad), max(0.5, s1 - s0 + 2 * pad)
        if ffmpeg:
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(video_path),
                            "-t", f"{dur:.2f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                            "-pix_fmt", "yuv420p", "-an", str(out)], check=False)
        paths.append(str(out) if out.exists() and out.stat().st_size > 0 else None)
    return paths


def plot_search(s, a, f, segs, scores, query, gt=None):
    """s(t), a(t), f(t) with the top moments shaded; GT in green when known."""
    from matplotlib.figure import Figure          # no pyplot: no global state, works in any backend
    s, a = cv.align(s, a)
    t = np.arange(len(f)) + 0.5
    fig = Figure(figsize=(10, 3.2))
    ax = fig.subplots()
    if gt is not None:
        ax.axvspan(gt[0], gt[1], color="green", alpha=0.2, label="ground truth")
    for i, (x0, x1) in enumerate(segs):                       # top moments as stacked bars, #1 on top
        y = 3.6 - 0.35 * i
        ax.plot([x0, x1], [y, y], color="red" if i == 0 else "salmon", lw=5 if i == 0 else 3,
                solid_capstyle="butt", label="top moment" if i == 0 else None)
        ax.text(x1 + 0.2, y, f"#{i + 1}", va="center", fontsize=7, color="darkred")
    ax.plot(t, cv.zscore(s), color="tab:blue", alpha=0.7, label="frame s(t), SigLIP")
    ax.plot(t, cv.zscore(a), color="tab:orange", alpha=0.7, label="activity a(t), X-CLIP")
    ax.plot(t, cv.zscore(f), color="black", lw=2, label="fused f(t)")
    ax.set_ylim(-3, 4); ax.set_xlabel("seconds"); ax.set_ylabel("z-score")
    ax.set_title(f"\u201c{query}\u201d", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", ncol=4)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- models
def _tensor(out, *names):
    """A feature tensor from either a plain tensor or a HF ModelOutput."""
    import torch
    if torch.is_tensor(out):
        return out
    for n in names + ("pooler_output",):
        v = getattr(out, n, None)
        if v is not None:
            return v
    return out[0]


class FrameEncoder:
    """SigLIP image/text embeddings, L2-normalised, as numpy arrays."""

    def __init__(self, model_name="google/siglip-base-patch16-224", device=None, fp16=True,
                 text_padding="max_length"):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)
        if fp16 and self.device.startswith("cuda"):
            self.model = self.model.half()
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.dtype = next(self.model.parameters()).dtype
        self.model_name = model_name
        self.text_padding = text_padding        # "max_length" (SigLIP's training setup) or "longest"

    def encode_frames(self, frames_bgr, batch=64):
        torch, out = self.torch, []
        with torch.no_grad():
            for i in range(0, len(frames_bgr), batch):
                rgb = [f[:, :, ::-1].copy() for f in frames_bgr[i:i + batch]]
                px = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
                out.append(_tensor(self.model.get_image_features(pixel_values=px), "image_embeds").float())
        e = torch.cat(out).cpu().numpy()
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)

    def encode_text(self, texts):
        torch = self.torch
        with torch.no_grad():
            pad = "max_length" if self.text_padding == "max_length" else True
            t = self.proc(text=list(texts), padding=pad, truncation=True, max_length=64,
                          return_tensors="pt").to(self.device)
            kw = {"input_ids": t["input_ids"]}
            if "attention_mask" in t:
                kw["attention_mask"] = t["attention_mask"]
            e = _tensor(self.model.get_text_features(**kw), "text_embeds").float().cpu().numpy()
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)


class VideoSearch:
    """Index videos once with both models, then answer queries with the Step 4 configuration."""

    def __init__(self, cfg, frame_encoder, xclip_scorer, clip_dir="/tmp/demo_clips"):
        self.cfg, self.fe, self.xs, self.clip_dir = cfg, frame_encoder, xclip_scorer, Path(clip_dir)

    def index(self, path):
        t0 = time.time()
        frames, _, duration = st.decode_1fps(path)
        siglip = self.fe.encode_frames(frames)
        t1 = time.time()
        v = xf.load_video_clips(path, self.xs.frames_per_clip, self.cfg["clip_sec"], 1.0, self.xs.size)
        emb, _ = self.xs.score_video(v["frames"], v["idx"], [])
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
        t2 = time.time()
        return {"path": str(path), "duration": float(duration), "n_sec": int(v["n_sec"]),
                "siglip": siglip, "xclip": emb, "starts": v["starts"],
                "timing": {"frame_s": round(t1 - t0, 2), "activity_s": round(t2 - t1, 2),
                           "total_s": round(t2 - t0, 2),
                           "s_per_video_minute": round((t2 - t0) / max(duration / 60, 1e-9), 2)}}

    def curves(self, idx, query):
        s = idx["siglip"] @ self.fe.encode_text([query])[0]
        ta = self.xs.encode_text([query]).cpu().numpy()[0]
        ta = ta / max(np.linalg.norm(ta), 1e-8)
        a = xf.clip_to_curve(idx["xclip"] @ ta, idx["starts"], self.cfg["clip_sec"], idx["n_sec"])
        return s.astype(np.float64), a

    def search(self, idx, query, k=fu.K, clips=True):
        t0 = time.time()
        s, a = self.curves(idx, query)
        segs, scores, f = predict_segments(s, a, self.cfg, idx["duration"], k)
        ms = 1000 * (time.time() - t0)
        paths = cut_clips(idx["path"], segs, self.clip_dir) if clips else [None] * len(segs)
        return {"segments": segs, "scores": scores, "clips": paths, "s": s, "a": a, "f": f,
                "query_ms": round(ms, 1)}


# --------------------------------------------------------------------------- demo videos
SUGGESTIONS = {
    "robbery": ["a person points a gun at someone", "a person runs out of the store"],
    "fighting": ["two people fighting", "a person falls to the ground"],
    "shoplifting": ["a person hides something in their clothes", "a person walks out of the store"],
    "burglary": ["a person breaks into a house", "a person climbs through a window"],
    "roadaccidents": ["a car crashes into another car", "a person is hit by a car"],
    "assault": ["a person attacks someone", "a person falls to the ground"],
    "vandalism": ["a person breaks a window", "a person damages a car"],
    "stealing": ["a person takes something and runs away", "a person opens a car door"],
    "arson": ["a person sets a fire", "smoke rises"],
    "explosion": ["an explosion", "people run away"],
}
GENERIC = ["a person walks into the room", "a person runs away", "two people fighting", "a car drives past"]


def find_extra_videos(root, exclude_dir=None, limit=8, max_minutes=10):
    """Short .mp4 videos attached besides Charades (e.g. UCF-Crime), with suggested queries."""
    out = []
    exclude = str(exclude_dir) if exclude_dir else None
    for p in sorted(Path(root).rglob("*.mp4")):
        if exclude and str(p).startswith(exclude):
            continue
        cap = cv2.VideoCapture(str(p))
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps <= 0 or n <= 0 or n / fps > 60 * max_minutes:
            continue
        name = p.stem.lower()
        sugg = next((v for k, v in SUGGESTIONS.items() if k in name or k in str(p.parent).lower()), GENERIC)
        out.append({"label": f"{p.stem}  ({n / fps / 60:.1f} min)", "path": str(p), "queries": sugg})
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- app
def build_app(searcher, examples, gt_lookup=None, max_minutes=15):
    """examples: list of {"label", "path", "queries"}; gt_lookup: {(path, query): (start, end)}."""
    import gradio as gr
    gt_lookup = gt_lookup or {}
    by_label = {e["label"]: e for e in examples}

    def _path(v):
        if isinstance(v, dict):
            v = v.get("video") or v.get("path") or v.get("name")
        return v

    def do_index(upload, choice):
        path = _path(upload) or (by_label[choice]["path"] if choice in by_label else None)
        if not path:
            return None, "Upload a video or pick an example first.", gr.update()
        cap = cv2.VideoCapture(str(path))
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps <= 0 or n <= 0:
            return None, "Could not read that video.", gr.update()
        if n / fps > 60 * max_minutes:
            return None, f"Please use a video shorter than {max_minutes} minutes.", gr.update()
        idx = searcher.index(path)
        tm = idx["timing"]
        msg = (f"Indexed **{Path(path).name}**: {idx['duration']:.0f} s of video, {len(idx['siglip'])} frames, "
               f"{len(idx['starts'])} clips in {tm['total_s']} s "
               f"({tm['frame_s']} s SigLIP + {tm['activity_s']} s X-CLIP).")
        sugg = by_label[choice]["queries"] if (not _path(upload) and choice in by_label) else GENERIC
        return idx, msg, gr.update(value=sugg[0] if sugg else "")

    def do_search(idx, query):
        empty = [None] * fu.K
        if idx is None:
            return [None, "Index a video first.", None] + empty
        if not query or not query.strip():
            return [None, "Type what you are looking for.", None] + empty
        res = searcher.search(idx, query.strip())
        gt = gt_lookup.get((idx["path"], query.strip().lower()))
        fig = plot_search(res["s"], res["a"], res["f"], res["segments"], res["scores"], query, gt)
        rows = [[i + 1, f"{x0:.1f}", f"{x1:.1f}", f"{x1 - x0:.1f}", sc]
                for i, ((x0, x1), sc) in enumerate(zip(res["segments"], res["scores"]))]
        msg = f"Found {len(rows)} moments in {res['query_ms']} ms."
        if gt is not None:
            iou = fu.iou_array(np.array([[res["segments"][0]]]), np.array([gt]))[0, 0] if rows else 0.0
            msg += f" Ground truth: {gt[0]:.1f}–{gt[1]:.1f} s, IoU of the top moment = {iou:.2f}."
        clips = (res["clips"] + empty)[:fu.K]
        return [fig, msg, rows] + clips

    with gr.Blocks(title="AI Video Investigator") as app:
        gr.Markdown("# AI Video Investigator\nFind the moment in a video that matches a description. "
                    "Frame evidence (SigLIP) and activity evidence (X-CLIP) are fused as in the report.")
        state = gr.State(None)
        with gr.Row():
            upload = gr.Video(label="Upload a video")
            with gr.Column():
                choice = gr.Dropdown(choices=list(by_label), label="…or pick an example",
                                     value=next(iter(by_label), None))
                index_btn = gr.Button("1. Index video", variant="primary")
                status = gr.Markdown()
        with gr.Row():
            query = gr.Textbox(label="2. What are you looking for?", placeholder="a person opens the door",
                               scale=4)
            search_btn = gr.Button("Search", variant="primary", scale=1)
        info = gr.Markdown()
        plot = gr.Plot(label="Evidence over time")
        table = gr.Dataframe(headers=["rank", "start (s)", "end (s)", "length (s)", "score"], label="Top moments")
        with gr.Row():
            vids = [gr.Video(label=f"#{i + 1}") for i in range(fu.K)]
        index_btn.click(do_index, [upload, choice], [state, status, query])
        search_btn.click(do_search, [state, query], [plot, info, table] + vids)
        query.submit(do_search, [state, query], [plot, info, table] + vids)
    return app
