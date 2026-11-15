"""Step 6 tests: run with `python tests/test_scene_tags.py`. Needs numpy + opencv only
(YOLO is replaced by a stand-in model)."""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidsearch import scene_tags as st                    # noqa: E402


class _FakeYOLO:
    """Pretends to detect one person per 100 grey levels of brightness."""
    class _R:
        def __init__(self, n):
            self.boxes = [0] * n

    def predict(self, source, **kw):
        assert kw["classes"] == [0]
        return [self._R(int(f.mean() // 100)) for f in source]


def _toy_video(d, seconds=5, fps=10):
    import cv2
    path = os.path.join(d, "toy.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    if not w.isOpened():
        return None
    for i in range(seconds * fps):
        w.write(np.full((48, 64, 3), min(250, 20 + i * 5), np.uint8))   # gets brighter over time
    w.release()
    return path


def test_decode_and_count():
    with tempfile.TemporaryDirectory() as d:
        path = _toy_video(d)
        if path is None:
            print("SKIP decode (no mp4 encoder on this machine)")
            return
        frames, bright, dur = st.decode_1fps(path)
        assert len(frames) == 5 and abs(dur - 5) < 0.2 and np.all(np.diff(bright) > 0)
        counts = st.count_people(_FakeYOLO(), frames, batch=2)
        assert len(counts) == 5 and counts[0] == 0 and counts[-1] >= 2
        tags, failed, stats = st.tag_videos(_FakeYOLO(), [("toy", path), ("bad", Path(d) / "nope.mp4")],
                                            log_every=0)
        assert set(tags) == {"toy"} and len(failed) == 1 and stats["frames"] == 5
        st.save_tags(Path(d) / "t.npz", tags)
        back = st.load_tags(Path(d) / "t.npz")
        assert np.allclose(back["toy"]["persons"], tags["toy"]["persons"])
        assert st.frame_at(path, 2.5) is not None
    print("PASS decode + people count + save/load")


def test_visual_change_and_tags():
    F = np.array([[1, 0], [1, 0.05], [0, 1], [0, 1.02]], np.float32)   # cut between 1 and 2
    d = st.visual_change(F)
    assert d.argmax() == 1 and d[0] < 0.01 and d[1] > 0.9
    tags = {"v": {"persons": np.array([0, 1, 3, 3]), "bright": np.array([10, 20, 30, 40.0])}}
    recs = [{"qid": "a", "vid": "v", "query": "person opens the door and walks in", "start": 1.0, "end": 3.0},
            {"qid": "b", "vid": "v", "query": "a person sneezes", "start": 0.0, "end": 1.0}]
    rows = st.query_tags(recs, tags, {"v": d})
    a, b = rows
    assert a["persons_mean"] == 2.0 and a["persons_max"] == 3 and a["bright"] == 25.0
    assert a["max_change"] > 0.9 and a["multi_action"]
    assert np.isnan(b["max_change"]) and not b["multi_action"] and b["persons_mean"] == 0
    assert not st.is_multi_action("person standing")
    print("PASS visual change + query tags")


def test_frame_table():
    with tempfile.TemporaryDirectory() as d:
        np.savez(Path(d) / "siglip_base_video_1fps.npz", A=np.ones((3, 4)), B=np.ones((5, 4)))
        np.savez(Path(d) / "siglip_base_text.npz", qids=np.array(["q1"]), emb=np.ones((1, 4)))
        T = st.load_frame_table(d, ["A", "B"])
        assert set(T) == {"A", "B"} and T["B"].shape == (5, 4)
    print("PASS SigLIP frame table loader")


if __name__ == "__main__":
    test_decode_and_count()
    test_visual_change_and_tags()
    test_frame_table()
