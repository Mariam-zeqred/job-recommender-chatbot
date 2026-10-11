import subprocess
import sys
from pathlib import Path

import pytest

from jobrec.data import POSTINGS_CSV, load_eval, load_postings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_postings  # noqa: E402


def test_generator_is_deterministic():
    assert generate_postings.generate(seed=1, n=20) == generate_postings.generate(seed=1, n=20)
    assert generate_postings.generate(seed=1, n=20) != generate_postings.generate(seed=2, n=20)


def test_committed_csv_matches_generator():
    assert generate_postings.generate() == POSTINGS_CSV.read_text(encoding="utf-8")


def test_generator_check_flag():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_postings.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


def test_load_postings_schema(postings):
    assert len(postings) == 500
    ids = {p.id for p in postings}
    assert len(ids) == 500
    p = postings[0]
    assert p.required_skills and all(s == s.lower() for s in p.required_skills)
    assert "Required:" in p.description
    assert p.text.startswith(p.title)


def test_load_postings_requires_core_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("id,title\nx,y\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_postings(bad)


def test_load_postings_accepts_minimal_real_world_columns(tmp_path):
    minimal = tmp_path / "min.csv"
    minimal.write_text(
        "id,title,company,location,description\n"
        "r1,Data Analyst,Acme,Cairo,SQL and Python with pandas; dashboards in Power BI\n"
    )
    ps = load_postings(minimal)
    assert ps[0].required_skills == ()
    assert ps[0].years_min == 0


def test_eval_set_is_consistent(postings):
    queries, relevant = load_eval()
    ids = {p.id for p in postings}
    assert len(queries) == 50
    assert set(queries.columns) >= {"id", "role_family", "cv_text"}
    for qid, rel in relevant.items():
        assert rel, f"{qid} has no relevant postings"
        assert rel <= ids


def test_committed_labels_match_relevance_rule():
    import build_eval_labels

    assert build_eval_labels.build() == (ROOT / "data" / "eval" / "labels.csv").read_text(
        encoding="utf-8"
    )
