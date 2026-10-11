"""Download GloVe 50-d word vectors (gensim-data release, public domain) for the glove embedder.

The file is 66 MB so it is not committed; it lands in data/vectors/ (gitignored) and its
SHA-256 is checked. Hugging Face and Kaggle were unreachable from the build sandbox, GitHub
release assets were, which is why this source was chosen.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://github.com/RaRe-Technologies/gensim-data/releases/download/"
    "glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"
)
SHA256 = "5c55f98957aa9fed8d2ac5fb1dcff57af3b23c5a3ee7af3f7945f8d49198eb24"
DEST = Path(__file__).resolve().parents[1] / "data" / "vectors" / "glove-wiki-gigaword-50.gz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if DEST.exists() and sha256(DEST) == SHA256:
        print(f"{DEST}: already present, hash ok")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL} ...")
    urllib.request.urlretrieve(URL, DEST)  # noqa: S310 - fixed https URL
    got = sha256(DEST)
    if got != SHA256:
        DEST.unlink()
        print(f"hash mismatch: {got}", file=sys.stderr)
        return 1
    print(f"ok: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
