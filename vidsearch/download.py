"""Download only the Charades videos we need, straight out of the remote zip."""
from pathlib import Path

from tqdm.auto import tqdm

CHARADES_480_URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"
CHARADES_ANN_URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip"


def download_subset(vids, out_dir, url=CHARADES_480_URL):
    """Uses HTTP range requests (remotezip), so the full multi-GB zip is never
    downloaded. Safe to re-run: videos already on disk are skipped."""
    from remotezip import RemoteZip

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in out_dir.glob("*.mp4")}
    with RemoteZip(url) as z:
        members = {Path(n).stem: n for n in z.namelist() if n.endswith(".mp4")}
        todo = sorted(v for v in set(vids) if v in members and v not in have)
        missing = sorted(set(vids) - set(members))
        for v in tqdm(todo, desc="videos"):
            (out_dir / f"{v}.mp4").write_bytes(z.read(members[v]))
    return {"downloaded": len(todo), "already_had": len(have), "missing_in_zip": missing}
