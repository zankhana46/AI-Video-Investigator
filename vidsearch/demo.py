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


def plot_search(s, a, f, segs, scores, query, gt=None, transparent=False):
    """s(t), a(t), f(t) with the top moments as bars; GT in green when known.
    transparent=True blends into light and dark app themes."""
    from matplotlib.figure import Figure          # no pyplot: no global state, works in any backend
    s, a = cv.align(s, a)
    t = np.arange(len(f)) + 0.5
    fig = Figure(figsize=(10, 3.2))
    ax = fig.subplots()
    fused_col, label_col = ("#8b5cf6", "#ef4444") if transparent else ("black", "darkred")  # readable on dark & light
    if gt is not None:
        ax.axvspan(gt[0], gt[1], color="#22c55e" if transparent else "green", alpha=0.28 if transparent else 0.2,
                   label="ground truth")
    for i, (x0, x1) in enumerate(segs):                       # top moments as stacked bars, #1 on top
        y = 3.6 - 0.35 * i
        ax.plot([x0, x1], [y, y], color="red" if i == 0 else "salmon", lw=5 if i == 0 else 3,
                solid_capstyle="butt", label="top moment" if i == 0 else None)
        ax.text(x1 + 0.2, y, f"#{i + 1}", va="center", fontsize=7, color=label_col)
    ax.plot(t, cv.zscore(s), color="tab:blue", alpha=0.7, label="frame s(t), SigLIP")
    ax.plot(t, cv.zscore(a), color="tab:orange", alpha=0.7, label="activity a(t), X-CLIP")
    ax.plot(t, cv.zscore(f), color=fused_col, lw=2.2, label="fused f(t)")
    ax.set_ylim(-3, 4); ax.set_xlabel("seconds"); ax.set_ylabel("z-score")
    ax.set_title(f"\u201c{query}\u201d", fontsize=10)
    leg = ax.legend(fontsize=7, loc="lower right", ncol=5, frameon=not transparent)
    if transparent:
        grey = "#8b949e"
        fig.patch.set_alpha(0); ax.set_facecolor("none")
        for sp in ax.spines.values():
            sp.set_color(grey)
        ax.tick_params(colors=grey); ax.xaxis.label.set_color(grey); ax.yaxis.label.set_color(grey)
        ax.title.set_color(grey); ax.grid(alpha=0.15)
        for txt in leg.get_texts():
            txt.set_color(grey)
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

    def index(self, path, progress=None):
        """Embed a video with both models. Cached per file, so re-picking a video is instant.
        progress: optional callable(fraction, description)."""
        p = Path(path)
        key = (str(p.resolve()), p.stat().st_size, p.stat().st_mtime) if p.exists() else str(path)
        cache = self.__dict__.setdefault("_cache", {})
        if key in cache:
            return cache[key]
        say = progress or (lambda f, d: None)
        t0 = time.time()
        say(0.05, "Reading frames")
        frames, _, duration = st.decode_1fps(path)
        say(0.25, "Frame evidence (SigLIP)")
        siglip = self.fe.encode_frames(frames)
        t1 = time.time()
        say(0.45, "Activity evidence (X-CLIP)")
        v = xf.load_video_clips(path, self.xs.frames_per_clip, self.cfg["clip_sec"], 1.0, self.xs.size)
        emb, _ = self.xs.score_video(v["frames"], v["idx"], [])
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
        t2 = time.time()
        say(1.0, "Ready")
        idx = {"path": str(path), "duration": float(duration), "n_sec": int(v["n_sec"]),
                "siglip": siglip, "xclip": emb, "starts": v["starts"],
                "timing": {"frame_s": round(t1 - t0, 2), "activity_s": round(t2 - t1, 2),
                           "total_s": round(t2 - t0, 2),
                           "s_per_video_minute": round((t2 - t0) / max(duration / 60, 1e-9), 2)}}
        if len(cache) >= 12:
            cache.pop(next(iter(cache)))
        cache[key] = idx
        return idx

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
CSS = """
.gradio-container {max-width: 1180px !important; margin: auto !important;}
#vi-header h1 {margin-bottom: 0.1em;}
#vi-header p {opacity: 0.75; margin-top: 0;}
.vi-card {border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 9px 14px;
          margin-bottom: 7px; display: flex; justify-content: space-between; align-items: center;
          background: var(--block-background-fill);}
.vi-card.top {border-left: 5px solid var(--color-accent);}
.vi-card .meta {opacity: 0.7; font-size: 0.9em;}
.vi-empty {opacity: 0.6; padding: 12px 2px;}
"""


def gt_key(path, query):
    """Ground truth is looked up by file name, so it survives Gradio copying files to its cache."""
    return (Path(str(path)).name, str(query).strip().lower())


def result_cards(segs, scores, gt=None):
    """HTML cards for the top moments (theme-aware colours)."""
    if not segs:
        return "<div class='vi-empty'>No moments found.</div>"
    out = []
    for i, ((x0, x1), sc) in enumerate(zip(segs, scores)):
        iou = ""
        if gt is not None:
            v = fu.iou_array(np.array([[[x0, x1]]]), np.array([gt]))[0, 0]
            iou = f" &middot; IoU with ground truth {v:.2f}"
        out.append(f"<div class='vi-card{' top' if i == 0 else ''}'><span><b>#{i + 1}</b>&nbsp;&nbsp;"
                   f"{x0:.1f} – {x1:.1f} s <span class='meta'>({x1 - x0:.1f} s)</span></span>"
                   f"<span class='meta'>score {sc:+.2f}{iou}</span></div>")
    return "".join(out)


def _make(cls, required, optional):
    """Create a component with optional styling arguments, dropping them on older/newer Gradio."""
    try:
        return cls(**required, **optional)
    except TypeError:
        return cls(**required)


def build_app(searcher, examples, gt_lookup=None, max_minutes=15):
    """examples: [{"label", "path", "queries"}]; gt_lookup: {gt_key(path, query): (start, end)}."""
    import gradio as gr
    gt_lookup = gt_lookup or {}
    by_label = {e["label"]: e for e in examples}
    by_name = {Path(e["path"]).name: e for e in examples}
    cfg = searcher.cfg
    K = fu.K

    def _path(v):
        if isinstance(v, dict):
            v = v.get("video") or v.get("path") or v.get("name")
        return str(v) if v else None

    def pick_example(label):
        e = by_label.get(label)
        return e["path"] if e else gr.update()

    def on_video(video, progress=gr.Progress()):
        path = _path(video)
        if not path:
            return None, "Pick an example or upload a video to begin.", gr.update(choices=[], value=None), gr.update()
        cap = cv2.VideoCapture(path)
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps <= 0 or n <= 0:
            return None, "**Could not read that video.** Try an .mp4 file.", gr.update(choices=[], value=None), gr.update()
        if n / fps > 60 * max_minutes:
            return (None, f"**That video is {n / fps / 60:.0f} minutes long.** Please use one under {max_minutes} minutes.",
                    gr.update(choices=[], value=None), gr.update())
        idx = searcher.index(path, progress=lambda f, d: progress(f, desc=d))
        tm = idx["timing"]
        ex = by_name.get(Path(path).name)
        sugg = (ex["queries"] if ex else GENERIC)[:6]
        msg = (f"**Ready:** {Path(path).name} &middot; {idx['duration']:.0f} s of video &middot; "
               f"indexed in {tm['total_s']} s. Type a query or click a suggestion.")
        return idx, msg, gr.update(choices=sugg, value=None), gr.update(value="")

    def use_suggestion(evt: gr.SelectData):
        return evt.value

    def do_search(idx, video, query, progress=gr.Progress()):
        path = _path(video)
        blank = (None, "", "<div class='vi-empty'>Results will appear here.</div>",
                 gr.update(choices=[], value=None), None, [])
        if not path:
            return (None, "Pick an example or upload a video first.") + blank[2:]
        if not query or not str(query).strip():
            return (None, "Type what you are looking for, or click a suggestion.") + blank[2:]
        if idx is None or Path(idx["path"]).name != Path(path).name:
            idx = searcher.index(path, progress=lambda f, d: progress(f, desc=d))
        q = str(query).strip()
        res = searcher.search(idx, q)
        gt = gt_lookup.get(gt_key(path, q))
        fig = plot_search(res["s"], res["a"], res["f"], res["segments"], res["scores"], q, gt, transparent=True)
        msg = f"Found {len(res['segments'])} moments in {res['query_ms']:.0f} ms."
        if gt is not None:
            msg += f" Ground truth (green): {gt[0]:.1f} – {gt[1]:.1f} s."
        names = [f"#{i + 1}  {x0:.1f}–{x1:.1f} s" for i, (x0, x1) in enumerate(res["segments"])]
        clips = res["clips"]
        return (fig, msg, result_cards(res["segments"], res["scores"], gt),
                gr.update(choices=names, value=names[0] if names else None),
                clips[0] if clips else None, clips)

    def play_moment(choice, clips):
        if not choice or not clips:
            return gr.update()
        i = int(choice.split()[0].lstrip("#")) - 1
        return clips[i] if 0 <= i < len(clips) else gr.update()

    theme = None
    try:
        theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="sky", neutral_hue="slate")
    except Exception:  # noqa: BLE001
        pass
    try:
        app_ctx = gr.Blocks(title="AI Video Investigator", theme=theme, css=CSS)
        launch_kw = {}
    except TypeError:                               # newer Gradio: theme and css go to launch()
        app_ctx = gr.Blocks(title="AI Video Investigator")
        launch_kw = {"theme": theme, "css": CSS}

    with app_ctx as app:
        gr.Markdown("# AI Video Investigator\nDescribe a moment in plain words and find where it happens in a video.",
                    elem_id="vi-header")
        idx_state, clips_state = gr.State(None), gr.State([])
        with gr.Row():
            with gr.Column(scale=5):
                choice = gr.Dropdown(choices=list(by_label), value=None, label="Pick an example video")
                video = _make(gr.Video, {"label": "…or upload your own (up to %d min)" % max_minutes},
                              {"height": 300})
                status = gr.Markdown("Pick an example or upload a video to begin.")
            with gr.Column(scale=6):
                query = _make(gr.Textbox, {"label": "What are you looking for?"},
                              {"placeholder": "e.g. a person opens the door", "lines": 1})
                with gr.Row():
                    search_btn = _make(gr.Button, {"value": "Search"}, {"variant": "primary"})
                suggestions = gr.Radio(choices=[], value=None, label="Suggestions (click to search)")
                info = gr.Markdown()
                cards = gr.HTML("<div class='vi-empty'>Results will appear here.</div>")
        with gr.Row():
            with gr.Column(scale=6):
                plot = gr.Plot(label="Evidence over time")
            with gr.Column(scale=5):
                moment = gr.Radio(choices=[], value=None, label="Play moment")
                player = _make(gr.Video, {"label": "Selected moment"}, {"interactive": False, "autoplay": True,
                                                                         "height": 300})
        with gr.Accordion("How it works", open=False):
            gr.Markdown(
                f"Each video is indexed once: **SigLIP** embeds one frame per second (what is visible) and "
                f"**X-CLIP** embeds {cfg['clip_sec']:.0f}-second clips (what is happening). For a query, both give a "
                f"score over time; they are standardised and mixed as *f(t) = α·frame + (1 − α)·activity* "
                f"(α = {cfg['alpha_top1']} for the top moment, {cfg['alpha_ranks_2_5']} for ranks 2–5), the values "
                f"chosen on the validation set in the report. The top moment comes from thresholding f(t); "
                f"the alternatives come from multi-scale sliding windows.")

        search_outputs = [plot, info, cards, moment, player, clips_state]
        choice.change(pick_example, [choice], [video])
        video.change(on_video, [video], [idx_state, status, suggestions, query])
        suggestions.select(use_suggestion, None, [query]).then(
            do_search, [idx_state, video, query], search_outputs)
        search_btn.click(do_search, [idx_state, video, query], search_outputs)
        query.submit(do_search, [idx_state, video, query], search_outputs)
        moment.change(play_moment, [moment, clips_state], [player])
    app._vi_launch_kw = launch_kw
    return app


def launch_app(app, **kw):
    """Launch, passing theme/css here when this Gradio version wants them at launch time."""
    extra = getattr(app, "_vi_launch_kw", {})
    try:
        return app.launch(**extra, **kw)
    except TypeError:
        return app.launch(**kw)
