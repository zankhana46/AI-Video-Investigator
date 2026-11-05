"""Video decoding with OpenCV (preinstalled on Kaggle/Colab, no decord needed)."""
import cv2
import numpy as np


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {"fps": fps, "n_frames": n, "duration": n / fps if fps > 0 else 0.0}


def sample_frames(path, fps=1.0, max_side=None):
    """Decode a video at `fps` frames per second.

    Frame i represents the time bin [i/fps, (i+1)/fps) and is taken from the bin
    centre. Returns (frames uint8 [T,H,W,3] RGB, times [T] in seconds).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {path}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n_out = max(1, int(np.ceil(n / native * fps)))
    times = (np.arange(n_out) + 0.5) / fps
    targets = np.minimum((times * native).astype(int), max(n - 1, 0))

    frames, t_i, idx = [], 0, 0
    while t_i < len(targets):
        if not cap.grab():          # header frame count can be wrong; stop at real end
            break
        while t_i < len(targets) and targets[t_i] == idx:
            ok, frame = cap.retrieve()
            if ok:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if max_side:
                    h, w = frame.shape[:2]
                    s = max_side / max(h, w)
                    if s < 1:
                        frame = cv2.resize(frame, (int(w * s), int(h * s)),
                                           interpolation=cv2.INTER_AREA)
                frames.append(frame)
            t_i += 1
        idx += 1
    cap.release()
    if not frames:
        raise IOError(f"No frames decoded from {path}")
    return np.stack(frames), times[:len(frames)]
