"""Step 3: activity-level evidence with X-CLIP.

X-CLIP turns a short clip (8 frames over ~4 s for microsoft/xclip-base-patch32) into one vector,
so unlike SigLIP it sees motion. Its text side is *video-conditioned*: a small "prompt generator"
adjusts each query embedding with the clip's visual tokens before the similarity is taken. So we
score every clip of a video against that video's own queries during extraction (cheap: ~3 queries
per video), and also cache the plain clip/text embeddings for a no-conditioning ablation and the demo.

Timing: clip j covers seconds [starts[j], starts[j] + clip_sec). Its score is spread over those
seconds and averaged where clips overlap, giving a curve a(t) whose index i covers [i, i+1) s,
the same convention as the SigLIP curve s(t).

torch / transformers are imported lazily so the numpy helpers can be unit-tested without them.
"""
import json
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- frame sampling
def clip_plan(n_sec, frames_per_clip, clip_sec=4.0, stride_sec=1.0):
    """Frame grid and clip layout for a video of n_sec seconds.

    Frames are sampled at fps = frames_per_clip / clip_sec, at bin centres (k + 0.5) / fps.
    Clip j starts at starts[j] seconds and uses grid frames idx[j, :].
    Returns times (G,), starts (C,), idx (C, F)."""
    fps = frames_per_clip / clip_sec
    n_grid = max(frames_per_clip, int(math.ceil(n_sec * fps - 1e-9)))
    times = (np.arange(n_grid) + 0.5) / fps
    last = max(0.0, n_sec - clip_sec)
    starts = np.arange(0.0, last + 1e-9, stride_sec)
    if len(starts) == 0:
        starts = np.array([0.0])
    if starts[-1] < last - 1e-9:          # make sure the final seconds are covered
        starts = np.append(starts, last)
    first = np.round(starts * fps).astype(int)
    idx = np.clip(first[:, None] + np.arange(frames_per_clip)[None, :], 0, n_grid - 1)
    return times, starts, idx


def _prep(bgr, size):
    """Resize the shortest side to `size`, centre-crop to size x size, BGR -> RGB uint8."""
    h, w = bgr.shape[:2]
    s = size / min(h, w)
    nh, nw = max(size, int(round(h * s))), max(size, int(round(w * s)))
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y0, x0 = (nh - size) // 2, (nw - size) // 2
    return np.ascontiguousarray(img[y0:y0 + size, x0:x0 + size, ::-1])


def load_video_clips(path, frames_per_clip, clip_sec=4.0, stride_sec=1.0, size=224):
    """Decode exactly the frames X-CLIP needs, in one sequential pass.
    Returns dict(frames (G, size, size, 3) uint8, starts (C,), idx (C, F), n_sec, duration)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 1e-3 or nfr <= 0:
        cap.release()
        raise IOError(f"bad metadata in {path} (fps={fps}, frames={nfr})")
    duration = nfr / fps
    n_sec = max(1, int(math.ceil(duration - 1e-6)))
    times, starts, idx = clip_plan(n_sec, frames_per_clip, clip_sec, stride_sec)
    want = np.clip(np.floor(times * fps).astype(int), 0, nfr - 1)
    frames = np.zeros((len(times), size, size, 3), np.uint8)
    order = np.argsort(want, kind="stable")
    ptr, fi, last = 0, -1, None
    while ptr < len(order):
        target = want[order[ptr]]
        while fi < target and cap.grab():
            fi += 1
        if fi < target:
            break                         # container shorter than its metadata
        ok, frame = cap.retrieve()
        if not ok:
            break
        last = _prep(frame, size)
        while ptr < len(order) and want[order[ptr]] == fi:
            frames[order[ptr]] = last
            ptr += 1
    cap.release()
    if last is None:
        raise IOError(f"no frames decoded from {path}")
    while ptr < len(order):               # pad anything past the real end with the last frame
        frames[order[ptr]] = last
        ptr += 1
    return {"frames": frames, "starts": starts, "idx": idx, "n_sec": n_sec, "duration": duration}


# --------------------------------------------------------------------------- model wrapper
def _get(out, name, index):
    """Field of a HF ModelOutput, or the same position of a tuple output."""
    v = getattr(out, name, None)
    return v if v is not None else out[index]


class XClipScorer:
    """Wraps an X-CLIP model for clip embeddings and video-conditioned query scores."""

    def __init__(self, model_name="microsoft/xclip-base-patch32", device=None, fp16=True,
                 max_frames_per_batch=256, model=None, processor=None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            from transformers import XCLIPModel, XCLIPProcessor
            model = XCLIPModel.from_pretrained(model_name)
            processor = XCLIPProcessor.from_pretrained(model_name)
        self.model_name = model_name
        self.model = model.eval().to(self.device)
        if fp16 and self.device.startswith("cuda"):
            self.model = self.model.half()
        self.proc = processor
        self.dtype = next(self.model.parameters()).dtype
        vc = self.model.config.vision_config
        self.frames_per_clip = int(getattr(vc, "num_frames", 8))
        self.size = int(getattr(vc, "image_size", 224))
        ip = None
        for name in ("image_processor", "video_processor", "feature_extractor"):
            ip = getattr(processor, name, None) if processor is not None else None
            if ip is not None:
                break
        self.mean_list = list(getattr(ip, "image_mean", None) or [0.485, 0.456, 0.406])
        self.std_list = list(getattr(ip, "image_std", None) or [0.229, 0.224, 0.225])
        self.mean = torch.tensor(self.mean_list, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self.std = torch.tensor(self.std_list, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self.max_frames = max_frames_per_batch
        self.use_forward = False   # set True by self_check if the fast path disagrees with X-CLIP

    # ---- text
    def tokenize(self, texts):
        return self.proc.tokenizer(list(texts), padding=True, truncation=True, max_length=77,
                                   return_tensors="pt").to(self.device)

    def encode_text(self, texts, batch_size=256):
        """Plain (not video-conditioned) query embeddings, (Q, D) float32 on the device."""
        torch, m = self.torch, self.model
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                t = self.tokenize(texts[i:i + batch_size])
                o = m.text_model(input_ids=t["input_ids"], attention_mask=t["attention_mask"])
                outs.append(m.text_projection(_get(o, "pooler_output", 1)).float())
        return torch.cat(outs)

    # ---- video
    def _pixels(self, frames_u8):
        x = self.torch.from_numpy(frames_u8).to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).to(self.dtype).div_(255)
        return (x - self.mean) / self.std

    def _encode_clips(self, pix):
        """pix (B, F, 3, H, W) -> video embeddings (B, D), prompt visual tokens (B, P, D).
        Mirrors XCLIPModel.forward step by step."""
        m = self.model
        B, F = pix.shape[:2]
        vo = m.vision_model(pixel_values=pix.reshape(B * F, *pix.shape[2:]))
        last, pooled = _get(vo, "last_hidden_state", 0), _get(vo, "pooler_output", 1)
        cls = m.visual_projection(pooled).view(B, F, -1)
        vid = _get(m.mit(cls), "pooler_output", 1)
        img = m.prompts_visual_layernorm(last[:, 1:, :]) @ m.prompts_visual_projection
        img = img.view(B, F, -1, vid.shape[-1]).mean(dim=1)
        return vid, img

    def _cond_scores(self, vid, img, txt):
        """Cosine between each clip and each query after video-specific prompting -> (B, K)."""
        m = self.model
        t = txt.to(self.dtype).unsqueeze(0).expand(vid.shape[0], -1, -1)
        t = t + m.prompts_generator(t, img)
        t = t / t.norm(dim=-1, keepdim=True)
        v = vid / vid.norm(dim=-1, keepdim=True)
        return self.torch.einsum("bd,bkd->bk", v.float(), t.float())

    def _forward_scores(self, pix, tok):
        """Reference path: X-CLIP's own forward pass (slower, used only if the fast path disagrees)."""
        m = self.model
        out = m(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], pixel_values=pix)
        scores = _get(out, "logits_per_video", 0).float() / m.logit_scale.exp().float()
        return _get(out, "video_embeds", 3).float(), scores

    def score_video(self, frames_u8, idx, texts):
        """All clips of one video against its queries.
        Returns clip embeddings (C, D) float32 and conditioned cosine scores (C, K) float32 (or None)."""
        torch = self.torch
        C, F = idx.shape
        bs = max(1, self.max_frames // F)
        embs, conds = [], []
        with torch.no_grad():
            x = self._pixels(frames_u8)
            idx_t = torch.from_numpy(np.ascontiguousarray(idx)).to(self.device)
            txt = self.encode_text(texts) if texts else None
            tok = self.tokenize(texts) if (texts and self.use_forward) else None
            for s in range(0, C, bs):
                pix = x[idx_t[s:s + bs]]
                if self.use_forward and tok is not None:
                    vid, sc = self._forward_scores(pix, tok)
                    embs.append(vid)
                    conds.append(sc)
                    continue
                vid, img = self._encode_clips(pix)
                embs.append(vid.float())
                if txt is not None:
                    conds.append(self._cond_scores(vid, img, txt))
        emb = torch.cat(embs).cpu().numpy()
        cond = torch.cat(conds).cpu().numpy() if conds else None
        return emb, cond

    def self_check(self, frames_u8, idx, texts, tol=2e-2):
        """Compare the fast decomposed scoring with X-CLIP's own forward on two clips.
        If they disagree, switch to the forward path (slower but guaranteed correct)."""
        torch = self.torch
        with torch.no_grad():
            pix = self._pixels(frames_u8)[torch.from_numpy(np.ascontiguousarray(idx[:2])).to(self.device)]
            _, ref = self._forward_scores(pix, self.tokenize(texts))
            vid, img = self._encode_clips(pix)
            mine = self._cond_scores(vid, img, self.encode_text(texts))
        err = float((ref - mine).abs().max())
        self.use_forward = not (err < tol)
        return err, ref.cpu().numpy(), mine.cpu().numpy()

    def query_latency_ms(self, frames_u8, idx, text, repeats=20):
        """Time to answer one new query on an already-indexed video (text encode + prompting +
        scoring). Visual tokens are kept in memory here, as an interactive demo would."""
        torch = self.torch
        sync = torch.cuda.synchronize if self.device.startswith("cuda") else (lambda: None)
        with torch.no_grad():
            pix = self._pixels(frames_u8)[torch.from_numpy(np.ascontiguousarray(idx)).to(self.device)]
            vid, img = self._encode_clips(pix)
            self._cond_scores(vid, img, self.encode_text([text]))      # warm-up
            sync()
            t0 = time.perf_counter()
            for _ in range(repeats):
                self._cond_scores(vid, img, self.encode_text([text])).cpu()
            sync()
        return 1000 * (time.perf_counter() - t0) / repeats


# --------------------------------------------------------------------------- extraction
def extract(scorer, videos, vid_queries, query_text, clip_sec=4.0, stride_sec=1.0,
            workers=4, prefetch=8, log_every=200):
    """videos: list of (vid, path). vid_queries: {vid: [qid, ...]}. query_text: {qid: text}.
    CPU threads decode upcoming videos while the GPU encodes the current one."""
    torch = scorer.torch
    F = scorer.frames_per_clip

    def load(item):
        vid, path = item
        try:
            return vid, load_video_clips(path, F, clip_sec, stride_sec, scorer.size), None
        except Exception as e:  # noqa: BLE001
            return vid, None, repr(e)

    if scorer.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    clip_emb, clip_scores, meta, failed = {}, {}, {}, []
    total_sec, t0 = 0.0, time.time()
    it = iter(videos)
    with ThreadPoolExecutor(workers) as ex:
        pending = deque(ex.submit(load, v) for v in [next(it, None) for _ in range(prefetch)] if v)
        done = 0
        while pending:
            vid, v, err = pending.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(ex.submit(load, nxt))
            if v is None:
                failed.append([vid, err])
                continue
            qids = vid_queries.get(vid, [])
            emb, cond = scorer.score_video(v["frames"], v["idx"], [query_text[q] for q in qids])
            clip_emb[vid] = emb.astype(np.float16)
            for j, q in enumerate(qids):
                clip_scores[q] = cond[:, j].astype(np.float32)
            meta[vid] = {"n_sec": int(v["n_sec"]), "duration": round(float(v["duration"]), 3),
                         "starts": [float(s) for s in v["starts"]]}
            total_sec += v["duration"]
            done += 1
            if log_every and done % log_every == 0:
                el = time.time() - t0
                print(f"  {done}/{len(videos)} videos | {el / 60:.1f} min | "
                      f"{el / max(total_sec / 60, 1e-9):.2f} s per video-minute")
    elapsed = time.time() - t0
    stats = {
        "videos": len(meta), "failed": len(failed), "video_minutes": round(total_sec / 60, 1),
        "elapsed_s": round(elapsed, 1),
        "index_s_per_video_minute": round(elapsed / max(total_sec / 60, 1e-9), 3),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if scorer.device.startswith("cuda") else 0.0,
    }
    return {"clip_emb": clip_emb, "clip_scores": clip_scores, "videos": meta,
            "failed": failed, "stats": stats}


def save_activity(out_dir, res, text_emb, config):
    """Four files, so the Kaggle output stays far below its 500-file limit."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "xclip_clip_emb.npz", **res["clip_emb"])
    np.savez(out / "xclip_clip_scores.npz", **res["clip_scores"])
    np.savez(out / "xclip_text_emb.npz", **text_emb)
    with open(out / "xclip_meta.json", "w") as f:
        json.dump({"config": config, "videos": res["videos"], "stats": res["stats"],
                   "failed": res["failed"]}, f)


def load_activity(feat_dir):
    """Load what save_activity wrote (searches subfolders, e.g. a Kaggle input dataset)."""
    feat_dir = Path(feat_dir)
    hit = sorted(feat_dir.rglob("xclip_meta.json"))
    if not hit:
        raise FileNotFoundError(f"no xclip_meta.json under {feat_dir}")
    d = hit[0].parent
    with open(d / "xclip_meta.json") as f:
        meta = json.load(f)

    def tab(name):
        z = np.load(d / name)
        return {k: z[k] for k in z.files}

    return {"clip_emb": tab("xclip_clip_emb.npz"), "clip_scores": tab("xclip_clip_scores.npz"),
            "text_emb": tab("xclip_text_emb.npz"), **meta}


def clip_to_curve(clip_scores, starts, clip_len, n_sec):
    """Spread each clip's score over the seconds it covers; average where clips overlap."""
    acc, cnt = np.zeros(n_sec), np.zeros(n_sec)
    for s, v in zip(starts, clip_scores):
        a = int(round(s))
        b = min(n_sec, int(round(s + clip_len)))
        if b <= a:
            b = min(n_sec, a + 1)
        acc[a:b] += v
        cnt[a:b] += 1
    ok = cnt > 0
    if ok.all():
        return acc / cnt
    if not ok.any():
        return np.zeros(n_sec)
    i = np.arange(n_sec)
    return np.interp(i, i[ok], acc[ok] / cnt[ok])


def activity_curves(act, recs, variant="cond"):
    """{qid: a(t)}. variant='cond' uses X-CLIP's video-conditioned scores (the full model);
    'plain' uses cosine of the cached clip and text embeddings (no prompt generator)."""
    clip_len = act["config"]["clip_sec"]
    out, normed = {}, {}
    for r in recs:
        v = act["videos"].get(r["vid"])
        if v is None:
            continue
        if variant == "cond":
            s = act["clip_scores"].get(r["qid"])
            if s is None:
                continue
        else:
            if r["vid"] not in normed:
                e = act["clip_emb"][r["vid"]].astype(np.float32)
                normed[r["vid"]] = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
            t = act["text_emb"].get(r["qid"])
            if t is None:
                continue
            t = t.astype(np.float32)
            s = normed[r["vid"]] @ (t / max(np.linalg.norm(t), 1e-8))
        out[r["qid"]] = clip_to_curve(s, v["starts"], clip_len, v["n_sec"])
    return out


# --------------------------------------------------------------------------- DEV comparison
def compare_models(model_names, recs, vid_dir, clip_sec=4.0, stride_sec=1.0, n_full_videos=1634):
    """Run each X-CLIP variant on a small val subset and report signal strength, accuracy with
    default (untuned) settings, and the estimated GPU time of a full run."""
    import torch
    from . import curves as cv
    vids = cv.videos_of(recs)
    videos = [(v, Path(vid_dir) / f"{v}.mp4") for v in vids]
    vq, qtext = {}, {}
    for r in recs:
        vq.setdefault(r["vid"], []).append(r["qid"])
        qtext[r["qid"]] = r["query"]
    rows = []
    for name in model_names:
        print(f"-- {name}")
        sc = XClipScorer(name)
        res = extract(sc, videos, vq, qtext, clip_sec, stride_sec, log_every=0)
        temb = sc.encode_text([r["query"] for r in recs]).cpu().numpy()
        act = {"config": {"clip_sec": clip_sec}, "videos": res["videos"],
               "clip_scores": res["clip_scores"], "clip_emb": res["clip_emb"],
               "text_emb": {r["qid"]: temb[i] for i, r in enumerate(recs)}}
        row = {"model": name, "frames/clip": sc.frames_per_clip}
        for variant in ("cond", "plain"):
            C = activity_curves(act, recs, variant)
            io, _ = cv.inside_vs_outside(C, recs)
            m, _ = cv.evaluate(cv.predict(C, recs, "windows", {"ctx": 0.5}), recs)
            row[f"in>out % ({variant})"] = io
            row[f"R1@0.5 ({variant})"] = m["R1@0.5"]
        per_video = res["stats"]["elapsed_s"] / max(res["stats"]["videos"], 1)
        row["est. full run (min)"] = round(per_video * n_full_videos / 60, 1)
        row["peak GPU GB"] = res["stats"]["peak_gpu_gb"]
        rows.append(row)
        del sc
        torch.cuda.empty_cache()
    return rows
