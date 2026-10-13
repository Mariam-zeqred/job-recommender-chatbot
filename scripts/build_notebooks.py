"""Write notebooks/01_evaluation.ipynb with outputs cleared (no nbformat dependency).

The notebook reproduces scripts/evaluate.py interactively, shows the misses per query and
compares the family-rule labels with the hand-judged subset. Run
`python scripts/build_notebooks.py` after changing the evaluation code; `--execute` then runs
the notebook once with nbclient (optional dependency) to prove the cells still work, and
rewrites it with outputs cleared, so nothing executed is committed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_evaluation.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        "# Evaluation of the recommender\n\n"
        "Reproduces `scripts/evaluate.py` and looks at the queries that miss. "
        "Data: `data/postings.csv` (synthetic, see README) and `data/eval/`.",
    ),
    (
        "code",
        "import sys\nsys.path.insert(0, '../src'); sys.path.insert(0, '../scripts')\n"
        "import pandas as pd\n"
        "from jobrec.data import load_postings, load_eval, load_judged\n"
        "from jobrec.embed import TfidfEmbedder, GloveEmbedder, VECTORS_PATH\n"
        "from jobrec.recommend import Recommender\n"
        "from evaluate import evaluate, evaluate_judged, judged_agreement\n\n"
        "postings = load_postings('../data/postings.csv')\n"
        "queries, relevant = load_eval('../data/eval')\n"
        "judged = load_judged('../data/eval')\n"
        "len(postings), len(queries), sum(len(v) for v in judged.values())",
    ),
    ("markdown", "## Precision@k and MRR per embedder and α"),
    (
        "code",
        "rows = []\n"
        "backends = [('tfidf', TfidfEmbedder)]\n"
        "if VECTORS_PATH.exists():\n"
        "    backends.append(('glove', GloveEmbedder))\n"
        "for name, cls in backends:\n"
        "    for alpha in (0.0, 0.5, 1.0):\n"
        "        rec = Recommender(postings, cls(), alpha=alpha)\n"
        "        res = evaluate(rec, queries, relevant)\n"
        "        rows.append({'embedder': name, 'alpha': alpha, **res})\n"
        "pd.DataFrame(rows)",
    ),
    ("markdown", "## Which queries miss, and what they get instead"),
    (
        "code",
        "rec = Recommender(postings, TfidfEmbedder(), alpha=0.5)\n"
        "for q in queries.itertuples(index=False):\n"
        "    top = rec.recommend(q.cv_text, 5)\n"
        "    wrong = [(r.posting.id, r.posting.role_family, r.posting.title) for r in top\n"
        "             if r.posting.id not in relevant[q.id]]\n"
        "    if wrong:\n"
        "        print(q.id, q.role_family, '->', wrong)",
    ),
    (
        "markdown",
        "## Family rule vs hand judgement (10 judged queries, `data/eval/JUDGING.md`)\n\n"
        "The same top-10 lists scored three ways: the family rule, the strict judgement "
        "(`relevant` only) and the lenient one (`relevant` or `arguable`). The judged subset "
        "was chosen to contain the cross-family CVs, so it is harder than the full set.",
    ),
    (
        "code",
        "jrows = []\n"
        "for name, cls in backends:\n"
        "    for alpha in (0.0, 0.5, 1.0):\n"
        "        rec = Recommender(postings, cls(), alpha=alpha)\n"
        "        jrows.append({'embedder': name, 'alpha': alpha,\n"
        "                      **evaluate_judged(rec, queries, relevant, judged)})\n"
        "pd.DataFrame(jrows)",
    ),
    (
        "code",
        "# Where the two label sets disagree: every disagreement is an 'arguable' pair.\n"
        "pd.DataFrame(judged_agreement(relevant, judged)).fillna(0).astype(int)",
    ),
    (
        "code",
        "# The arguable pairs, by reason: seniority mismatches vs adjacent roles.\n"
        "jd = pd.read_csv('../data/eval/judged.csv')\n"
        "fam = dict(zip(queries.id, queries.role_family))\n"
        "jd['same_family'] = [fam[q] == f for q, f in zip(jd.query_id, jd.posting_family)]\n"
        "jd[jd.judgement == 'arguable'].groupby(['same_family', 'seniority']).size()",
    ),
    (
        "code",
        "# Q18 (data platform CV) at alpha = 0.5: the family rule calls 9 of 10 wrong,\n"
        "# a human calls 7 of them arguable and 2 wrong. Read the reasons.\n"
        "rec = Recommender(postings, TfidfEmbedder(), alpha=0.5)\n"
        "q18 = queries[queries.id == 'Q18'].iloc[0]\n"
        "for r in rec.recommend(q18.cv_text, 10):\n"
        "    row = jd[(jd.query_id == 'Q18') & (jd.posting_id == r.posting.id)].iloc[0]\n"
        "    print(f'{r.posting.id} {row.judgement:9} {r.posting.title[:32]:32} | {row.reason}')",
    ),
    ("markdown", "## One query in detail: matched and missing skills"),
    (
        "code",
        "q = queries.iloc[6]  # Q07, the incubator CV\n"
        "print(q.cv_text)\n"
        "for r in rec.recommend(q.cv_text, 5):\n"
        "    print(f'{r.score:.2f} sim={r.similarity:.2f} cov={r.skill_coverage:.2f} '\n"
        "          f'{r.posting.id} {r.posting.title} | {r.explanation}')",
    ),
]


def build() -> dict:
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": src,
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, src in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def execute(nb: dict) -> int:
    """Run every cell once (needs nbformat, nbclient, ipykernel). Fails loudly on any error,
    prints a short digest of each code cell's output, and does not write anything."""
    import nbformat
    from nbclient import NotebookClient

    node = nbformat.from_dict(nb)
    NotebookClient(
        node, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(OUT.parent)}}
    ).execute()
    for i, cell in enumerate(node.cells):
        if cell.cell_type != "code":
            continue
        texts = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                texts.append(out["text"])
            elif "data" in out:
                texts.append(out["data"].get("text/plain", ""))
            elif out.get("output_type") == "error":
                print(f"cell {i} raised {out['ename']}: {out['evalue']}", file=sys.stderr)
                return 1
        digest = " ".join("".join(texts).split())
        print(f"cell {i}: {digest[:160]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the cells once, then clear")
    args = parser.parse_args(argv)
    nb = build()
    if args.execute and execute(nb) != 0:
        return 1
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {OUT} (outputs cleared)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
