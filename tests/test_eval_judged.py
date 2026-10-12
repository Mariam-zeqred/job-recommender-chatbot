"""The hand-judged subset: file shape, pool coverage, and the metrics computed from it."""

import csv
import sys
from pathlib import Path

import pytest

from jobrec.data import EVAL_DIR, JUDGEMENTS, load_eval, load_judged
from jobrec.embed import TfidfEmbedder
from jobrec.recommend import Recommender

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_judging_pool import JUDGED_QUERIES, build_pool  # noqa: E402
from evaluate import evaluate_judged, judged_agreement  # noqa: E402


@pytest.fixture(scope="module")
def judged():
    return load_judged()


def test_judged_file_shape(judged, postings):
    rows = list(csv.DictReader((EVAL_DIR / "judged.csv").open(encoding="utf-8")))
    assert set(rows[0]) == {
        "query_id",
        "posting_id",
        "posting_family",
        "posting_title",
        "seniority",
        "judgement",
        "reason",
    }
    assert all(r["reason"].strip() for r in rows), "every judgement needs a reason"
    assert set(judged) == set(JUDGED_QUERIES)
    ids = {p.id for p in postings}
    assert all(pid in ids for pairs in judged.values() for pid in pairs)
    assert {j for pairs in judged.values() for j in pairs.values()} == set(JUDGEMENTS)
    assert sum(len(v) for v in judged.values()) == len(rows) == 255


def test_pool_is_covered(judged):
    pool = build_pool()  # TF-IDF only in CI; GloVe too when vectors are present
    missing = [(q, p) for q, ids in pool.items() for p in ids if p not in judged[q]]
    assert missing == []


def test_judged_metrics(judged, postings):
    queries, relevant = load_eval()
    rec = Recommender(postings, TfidfEmbedder(), alpha=0.5)
    res = evaluate_judged(rec, queries, relevant, judged)
    assert res["n_queries"] == len(JUDGED_QUERIES)
    assert res["unjudged_pairs"] == 0
    assert 0 <= res["p@10_strict"] <= res["p@10_lenient"] <= 1
    assert res["p@10_strict"] >= 0.6  # the committed judgements; see README "Evaluation"
    agreement = judged_agreement(relevant, judged)
    assert sum(sum(v.values()) for v in agreement.values()) == 255
    # The family rule never calls a 'wrong' pair relevant, and never misses a 'relevant' one:
    # every disagreement is an 'arguable' pair (seniority or adjacent role).
    assert "wrong" not in agreement["rule_relevant"]
    assert "relevant" not in agreement["rule_not_relevant"]
