"""Precision@k and MRR of each embedder/alpha on the 50-query evaluation set.

Writes figures/eval_results.json and prints a Markdown table. `--quick` runs TF-IDF only
(used by `make check`); otherwise every embedder whose dependencies are present is run.
Numbers quoted in the README come from this script; re-run it after changing anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobrec.data import load_eval, load_postings  # noqa: E402
from jobrec.embed import VECTORS_PATH, Embedder, GloveEmbedder, TfidfEmbedder  # noqa: E402
from jobrec.recommend import Recommender  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "eval_results.json"
KS = (1, 5, 10)
ALPHAS = (0.0, 0.5, 1.0)


def available_embedders(quick: bool) -> list[tuple[str, type]]:
    out: list[tuple[str, type]] = [("tfidf", TfidfEmbedder)]
    if quick:
        return out
    if VECTORS_PATH.exists():
        out.append(("glove", GloveEmbedder))
    try:
        from jobrec.embed import MiniLMEmbedder

        MiniLMEmbedder()  # fails if sentence-transformers or the model is unavailable
        out.append(("minilm", MiniLMEmbedder))
    except Exception:  # noqa: BLE001 - optional backend, any failure means "skip"
        pass
    return out


def evaluate(rec: Recommender, queries, relevant) -> dict:
    hits = {k: [] for k in KS}
    rr = []
    for q in queries.itertuples(index=False):
        ranked = [r.posting.id for r in rec.recommend(q.cv_text, k=max(KS))]
        rel = relevant[q.id]
        for k in KS:
            hits[k].append(sum(pid in rel for pid in ranked[:k]) / k)
        first = next((i for i, pid in enumerate(ranked) if pid in rel), None)
        rr.append(1 / (first + 1) if first is not None else 0.0)
    return {
        **{f"p@{k}": round(float(np.mean(hits[k])), 4) for k in KS},
        "mrr": round(float(np.mean(rr)), 4),
        "n_queries": len(queries),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="TF-IDF only, no JSON written")
    parser.add_argument("--figure", action="store_true", help="also write figures/eval_p5.png")
    args = parser.parse_args(argv)
    postings = load_postings()
    queries, relevant = load_eval()
    n_rel = np.mean([len(v) for v in relevant.values()])
    chance = n_rel / len(postings)
    rows = []
    for name, cls in available_embedders(args.quick):
        for alpha in ALPHAS:
            t0 = time.perf_counter()
            emb: Embedder = cls()
            rec = Recommender(postings, emb, alpha=alpha)
            res = evaluate(rec, queries, relevant)
            res.update(embedder=name, alpha=alpha, seconds=round(time.perf_counter() - t0, 2))
            rows.append(res)
    print(f"| embedder | alpha | P@1 | P@5 | P@10 | MRR |  (chance P@k = {chance:.3f})")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['embedder']} | {r['alpha']} | {r['p@1']:.2f} | {r['p@5']:.2f} | "
            f"{r['p@10']:.2f} | {r['mrr']:.2f} |"
        )
    if not args.quick:
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(
            json.dumps({"chance_precision": round(float(chance), 4), "results": rows}, indent=1)
            + "\n"
        )
        print(f"wrote {OUT}")
        if args.figure:
            make_figure(rows, float(chance))
    return 0


def make_figure(rows: list[dict], chance: float) -> None:
    """Grouped bars of P@5 per embedder and alpha, with the chance level as a line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    embedders = list(dict.fromkeys(r["embedder"] for r in rows))
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    width = 0.8 / len(ALPHAS)
    for j, alpha in enumerate(ALPHAS):
        vals = [
            next(r["p@5"] for r in rows if r["embedder"] == e and r["alpha"] == alpha)
            for e in embedders
        ]
        xs = [i + (j - (len(ALPHAS) - 1) / 2) * width for i in range(len(embedders))]
        bars = ax.bar(xs, vals, width * 0.95, label=f"alpha = {alpha}")
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.axhline(chance, color="grey", ls="--", lw=1, label=f"chance ({chance:.2f})")
    ax.set_xticks(range(len(embedders)), embedders)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("precision@5 (50 queries)")
    ax.set_title("Recommender precision@5 on the synthetic evaluation set")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = OUT.parent / "eval_p5.png"
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
