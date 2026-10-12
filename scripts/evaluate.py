"""Precision@k and MRR of each embedder/alpha on the 50-query evaluation set, plus P@10 on the
hand-judged subset (10 queries, data/eval/judged.csv, criteria in data/eval/JUDGING.md).

Writes figures/eval_results.json and prints two Markdown tables. `--quick` runs TF-IDF only
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

from jobrec.data import load_eval, load_judged, load_postings  # noqa: E402
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


def evaluate_judged(rec: Recommender, queries, relevant, judged) -> dict:
    """P@10 on the judged queries under three readings of 'relevant': the family rule (as in
    `evaluate`), the strict hand judgement (relevant only) and the lenient one (relevant or
    arguable). A top-10 pair that was never judged counts as not relevant and is reported
    as `unjudged` so the reader can see how much of the ranking the judgements cover."""
    k = 10
    strict, lenient, family, unjudged = [], [], [], 0
    for q in queries.itertuples(index=False):
        if q.id not in judged:
            continue
        ranked = [r.posting.id for r in rec.recommend(q.cv_text, k=k)]
        js = [judged[q.id].get(pid) for pid in ranked]
        unjudged += js.count(None)
        strict.append(sum(j == "relevant" for j in js) / k)
        lenient.append(sum(j in ("relevant", "arguable") for j in js) / k)
        family.append(sum(pid in relevant[q.id] for pid in ranked) / k)
    return {
        "p@10_family_rule": round(float(np.mean(family)), 4),
        "p@10_strict": round(float(np.mean(strict)), 4),
        "p@10_lenient": round(float(np.mean(lenient)), 4),
        "unjudged_pairs": unjudged,
        "n_queries": len(strict),
    }


def judged_agreement(relevant, judged) -> dict:
    """How the family rule and the hand judgement agree on the judged pairs: a 2x3 count of
    (rule says relevant / not) x (relevant / arguable / wrong)."""
    table = {"rule_relevant": {}, "rule_not_relevant": {}}
    for q, pairs in judged.items():
        for pid, j in pairs.items():
            row = "rule_relevant" if pid in relevant[q] else "rule_not_relevant"
            table[row][j] = table[row].get(j, 0) + 1
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="TF-IDF only, no JSON written")
    parser.add_argument("--figure", action="store_true", help="also write figures/eval_p5.png")
    args = parser.parse_args(argv)
    postings = load_postings()
    queries, relevant = load_eval()
    judged = load_judged()
    n_rel = np.mean([len(v) for v in relevant.values()])
    chance = n_rel / len(postings)
    rows, judged_rows = [], []
    for name, cls in available_embedders(args.quick):
        for alpha in ALPHAS:
            t0 = time.perf_counter()
            emb: Embedder = cls()
            rec = Recommender(postings, emb, alpha=alpha)
            res = evaluate(rec, queries, relevant)
            res.update(embedder=name, alpha=alpha, seconds=round(time.perf_counter() - t0, 2))
            rows.append(res)
            jres = evaluate_judged(rec, queries, relevant, judged)
            jres.update(embedder=name, alpha=alpha)
            judged_rows.append(jres)
    print(f"| embedder | alpha | P@1 | P@5 | P@10 | MRR |  (chance P@k = {chance:.3f})")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['embedder']} | {r['alpha']} | {r['p@1']:.2f} | {r['p@5']:.2f} | "
            f"{r['p@10']:.2f} | {r['mrr']:.2f} |"
        )
    n_j = judged_rows[0]["n_queries"]
    print()
    print(
        f"| embedder | alpha | P@10 family rule | P@10 strict | P@10 lenient | unjudged |"
        f"  ({n_j} judged queries)"
    )
    print("|---|---|---|---|---|---|")
    for r in judged_rows:
        print(
            f"| {r['embedder']} | {r['alpha']} | {r['p@10_family_rule']:.2f} | "
            f"{r['p@10_strict']:.2f} | {r['p@10_lenient']:.2f} | {r['unjudged_pairs']} |"
        )
    agreement = judged_agreement(relevant, judged)
    print(f"family rule vs judgement on the judged pairs: {agreement}")
    if not args.quick:
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(
            json.dumps(
                {
                    "chance_precision": round(float(chance), 4),
                    "results": rows,
                    "judged": judged_rows,
                    "judged_agreement": agreement,
                },
                indent=1,
            )
            + "\n"
        )
        print(f"wrote {OUT}")
        if args.figure:
            make_figure(rows, float(chance))
            make_judged_figure(judged_rows)
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


def make_judged_figure(rows: list[dict]) -> None:
    """P@10 on the judged queries: family rule vs strict vs lenient, one group per config."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['embedder']}\nα={r['alpha']}" for r in rows]
    series = [
        ("family rule", "p@10_family_rule"),
        ("hand-judged, strict", "p@10_strict"),
        ("hand-judged, lenient", "p@10_lenient"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    width = 0.8 / len(series)
    for j, (label, key) in enumerate(series):
        xs = [i + (j - (len(series) - 1) / 2) * width for i in range(len(rows))]
        bars = ax.bar(xs, [r[key] for r in rows], width * 0.95, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)
    ax.set_xticks(range(len(rows)), labels, fontsize=7)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(f"precision@10 ({rows[0]['n_queries']} judged queries)")
    ax.set_title("Family-rule labels vs hand judgement on the judged subset")
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = OUT.parent / "eval_judged_p10.png"
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
