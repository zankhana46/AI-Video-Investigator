"""Run with:  python tests/test_basics.py   (or pytest)"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from vidsearch.metrics import iou_1d, nms_1d, evaluate
from vidsearch.data import parse_sta, attach_durations
from vidsearch.video_io import sample_frames
from vidsearch.baselines import oracle_predictions, random_predictions


def test_iou():
    assert np.isclose(iou_1d([[0, 10]], (0, 10))[0], 1.0)
    assert np.isclose(iou_1d([[0, 10]], (5, 15))[0], 5 / 15)
    assert iou_1d([[0, 5]], (6, 10))[0] == 0.0


def test_nms():
    keep = nms_1d([(0, 10, .9), (1, 10, .8), (20, 30, .7)], iou_thr=0.5)
    assert [k[2] for k in keep] == [.9, .7]


def test_parse_and_eval():
    d = Path(tempfile.mkdtemp())
    (d / "sta.txt").write_text("AO8RW 0.0 6.9##a person is putting a book on a shelf.\n"
                               "AO8RW 5.0 40.0##person opens the door.\n")
    recs, dropped = attach_durations(parse_sta(d / "sta.txt"), {"AO8RW": 33.0}, "test")
    assert dropped == 0 and recs[1]["end"] == 33.0 and recs[0]["query"].endswith("shelf")
    s, _ = evaluate(oracle_predictions(recs), recs)
    assert s["R1@0.7"] == 100 and np.isclose(s["mIoU"], 100)
    s, _ = evaluate(random_predictions(recs), recs)
    assert 0 <= s["R1@0.5"] <= 100


def test_sample_frames():
    p = Path(tempfile.mkdtemp()) / "v.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 48))
    for i in range(24 * 5):  # 5-second video
        w.write(np.full((48, 64, 3), i % 255, np.uint8))
    w.release()
    frames, times = sample_frames(p, fps=1.0)
    assert frames.shape == (5, 48, 64, 3) and np.allclose(times, [.5, 1.5, 2.5, 3.5, 4.5])
    frames, _ = sample_frames(p, fps=2.0)
    assert len(frames) == 10


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
