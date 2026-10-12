"""Reproduce the pool of (query, posting) pairs behind data/eval/judged.csv.

The pool is the union of the top-10 lists of every available embedder at alpha 0, 0.5 and 1
for the ten judged queries (TF-IDF always; GloVe when `make vectors` has been run). Every
pair in the pool was judged by hand once (data/eval/JUDGING.md). `--check` verifies that the
committed judged.csv still covers the pool, so a change to the generator, the recommender or
the skill dictionary that moves new postings into a top-10 is caught in CI rather than
silently counted as "unjudged".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobrec.data import load_eval, load_judged, load_postings  # noqa: E402
from jobrec.embed import VECTORS_PATH, GloveEmbedder, TfidfEmbedder  # noqa: E402
from jobrec.recommend import Recommender  # noqa: E402

JUDGED_QUERIES = ("Q02", "Q07", "Q11", "Q16", "Q18", "Q22", "Q32", "Q39", "Q48", "Q50")
ALPHAS = (0.0, 0.5, 1.0)
K = 10


def build_pool() -> dict[str, dict[str, list[str]]]:
    """{query_id: {posting_id: ["tfidf@0.5:rank", ...]}} for the judged queries."""
    postings = load_postings()
    queries, _ = load_eval()
    backends = [("tfidf", TfidfEmbedder)]
    if VECTORS_PATH.exists():
        backends.append(("glove", GloveEmbedder))
    recs = {
        (name, alpha): Recommender(postings, cls(), alpha=alpha)
        for name, cls in backends
        for alpha in ALPHAS
    }
    pool: dict[str, dict[str, list[str]]] = {}
    for q in queries.itertuples(index=False):
        if q.id not in JUDGED_QUERIES:
            continue
        hits: dict[str, list[str]] = {}
        for (name, alpha), rec in recs.items():
            for rank, r in enumerate(rec.recommend(q.cv_text, K), start=1):
                hits.setdefault(r.posting.id, []).append(f"{name}@{alpha}:{rank}")
        pool[q.id] = hits
    return pool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify judged.csv covers the pool")
    args = parser.parse_args(argv)
    pool = build_pool()
    n = sum(len(v) for v in pool.values())
    if args.check:
        judged = load_judged()
        missing = [(q, p) for q, ids in pool.items() for p in ids if p not in judged.get(q, {})]
        if missing:
            print(f"{len(missing)} pooled pairs are not in judged.csv: {missing[:10]}")
            return 1
        print(f"judged.csv covers the pool ({n} pairs, {len(pool)} queries)")
        return 0
    for q, ids in pool.items():
        print(f"{q}: {len(ids)} postings")
        for pid, where in sorted(ids.items()):
            print(f"  {pid}  {' '.join(where)}")
    print(f"{n} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
