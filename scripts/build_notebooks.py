"""Write notebooks/01_evaluation.ipynb with outputs cleared (no nbformat dependency).

The notebook reproduces scripts/evaluate.py interactively and shows the misses per query.
Run `python scripts/build_notebooks.py` after changing the evaluation code.
"""

from __future__ import annotations

import json
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
        "from jobrec.data import load_postings, load_eval\n"
        "from jobrec.embed import TfidfEmbedder, GloveEmbedder, VECTORS_PATH\n"
        "from jobrec.recommend import Recommender\n"
        "from evaluate import evaluate\n\n"
        "postings = load_postings('../data/postings.csv')\n"
        "queries, relevant = load_eval('../data/eval')\n"
        "len(postings), len(queries)",
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


def main() -> int:
    nb = {
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
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
