# AI Video Investigator

Semantic video search: natural-language query → time-stamped video segments.
CS 5330 project, Mithun Dineshkumar & Zankhana Mehta.

## Layout
```
vidsearch/        reusable code (import this from every notebook)
  data.py         Charades-STA parsing, cleaning, val split
  video_io.py     frame sampling with OpenCV
  metrics.py      R@k at IoU, mIoU, 1-D NMS
  baselines.py    oracle / random / prior sanity baselines
  download.py     fetch only the needed Charades videos
  utils.py        timer, results.csv logger
notebooks/        one notebook per project step (run on Kaggle)
tests/            python tests/test_basics.py
```

## Workflow rules
1. Code lives in `vidsearch/`; notebooks only call it. Commit and push, then re-clone in Kaggle.
2. GPU is used only for feature extraction. Features are cached as .npy and everything else runs on CPU.
3. Every experiment is logged with `log_result(...)` to `results/results.csv`.
4. Tune on `val`, report on `test`, never the other way round.
