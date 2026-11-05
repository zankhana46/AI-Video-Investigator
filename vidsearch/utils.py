import csv
import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


@contextmanager
def timer(name, store=None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if store is not None:
        store[name] = dt
    print(f"[{name}] {dt:.2f}s")


def log_result(csv_path, method, split, summary, config=None):
    """Append one experiment row to a CSV. Your report tables come from this file."""
    row = {"time": datetime.now().isoformat(timespec="seconds"), "method": method,
           "split": split,
           **{k: round(v, 2) if isinstance(v, float) else v for k, v in summary.items()},
           "config": json.dumps(config or {})}
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
