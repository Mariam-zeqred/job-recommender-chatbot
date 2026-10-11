import numpy as np
import pytest

from jobrec.data import Posting
from jobrec.embed import TfidfEmbedder, make_embedder
from jobrec.recommend import Recommender

CV_EMBEDDED = (
    "Embedded firmware engineer. C on ESP32 and STM32 with FreeRTOS, UART/I2C/SPI drivers, "
    "debugging with an oscilloscope, git."
)
CV_ANALYST = "Data analyst: SQL, Excel, Python with pandas, Power BI dashboards, statistics."


def test_returns_k_sorted_results(postings):
    rec = Recommender(postings)
    out = rec.recommend(CV_EMBEDDED, k=7)
    assert len(out) == 7
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= r.score <= 1.0 for r in out)


def test_top_hits_are_from_the_right_family(postings):
    rec = Recommender(postings)
    top = rec.recommend(CV_EMBEDDED, k=5)
    assert sum(r.posting.role_family == "embedded_firmware" for r in top) >= 4
    top = rec.recommend(CV_ANALYST, k=5)
    assert sum(r.posting.role_family == "data_analyst" for r in top) >= 4


def test_explanation_partitions_required_skills(postings):
    rec = Recommender(postings)
    for r in rec.recommend(CV_EMBEDDED, k=5):
        assert set(r.matched_skills) | set(r.missing_skills) == set(r.posting.required_skills)
        assert not set(r.matched_skills) & set(r.missing_skills)
        assert r.skill_coverage == pytest.approx(
            len(r.matched_skills) / len(r.posting.required_skills)
        )
        assert "matches" in r.explanation or "missing" in r.explanation


def test_alpha_extremes(postings):
    pure_text = Recommender(postings, alpha=0.0).recommend(CV_ANALYST, k=3)
    assert all(r.score == pytest.approx(r.similarity) for r in pure_text)
    pure_skill = Recommender(postings, alpha=1.0).recommend(CV_ANALYST, k=3)
    assert all(r.score == pytest.approx(r.skill_coverage) for r in pure_skill)
    with pytest.raises(ValueError):
        Recommender(postings, alpha=1.5)


def test_empty_cv_rejected(postings):
    with pytest.raises(ValueError):
        Recommender(postings[:10]).recommend("   ")


def test_required_skills_fall_back_to_extraction():
    ps = [
        Posting("a", "Analyst", "X", "Y", "Needs SQL and Excel"),
        Posting("b", "Firmware", "X", "Y", "Needs C and FreeRTOS on ESP32"),
    ]
    rec = Recommender(ps)
    top = rec.recommend("I write C firmware with FreeRTOS", k=1)[0]
    assert top.posting.id == "b"
    assert set(top.matched_skills) == {"c", "freertos"}
    assert top.missing_skills == ("esp32",)


def test_tfidf_rows_are_unit_norm():
    emb = TfidfEmbedder()
    emb.fit(["alpha beta", "beta gamma"])
    m = emb.transform(["alpha beta gamma"]).toarray()
    assert np.linalg.norm(m[0]) == pytest.approx(1.0)


def test_make_embedder_names():
    assert make_embedder("tfidf").name == "tfidf"
    with pytest.raises(ValueError):
        make_embedder("word2vec")
