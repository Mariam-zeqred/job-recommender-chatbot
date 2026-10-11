import pytest
from fastapi.testclient import TestClient

from jobrec.api import State, create_app

CV = (
    "Data analyst with SQL, Excel, pandas and Power BI. Built weekly dashboards and ran "
    "A/B tests for a retail client."
)


@pytest.fixture(scope="module")
def client(postings):
    return TestClient(create_app(State(postings)))


def test_health_and_index(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["postings"] == 500 and h["llm_provider"] == "none"
    r = client.get("/")
    assert r.status_code == 200 and "<title>Job recommender</title>" in r.text


def test_recommend_endpoint(client):
    r = client.post("/recommend", json={"cv_text": CV, "k": 4})
    assert r.status_code == 200
    body = r.json()
    assert "sql" in body["cv_skills"]
    assert len(body["results"]) == 4
    first = body["results"][0]
    assert first["posting"]["id"].startswith("P")
    assert first["explanation"]
    assert set(first["matched_skills"]) | set(first["missing_skills"]) == set(
        first["posting"]["required_skills"]
    )


def test_recommend_validation(client):
    assert client.post("/recommend", json={"cv_text": "short", "k": 3}).status_code == 422
    assert client.post("/recommend", json={"cv_text": CV, "k": 0}).status_code == 422


def test_posting_lookup(client):
    assert client.get("/postings/P0001").json()["id"] == "P0001"
    assert client.get("/postings/nope").status_code == 404


def test_interview_round_trip(client):
    pid = client.post("/recommend", json={"cv_text": CV, "k": 1}).json()["results"][0]["posting"][
        "id"
    ]
    start = client.post("/interview/start", json={"posting_id": pid, "cv_text": CV}).json()
    sid = start["session_id"]
    total = start["total_questions"]
    assert start["question"]["id"] == 0
    done = False
    for i in range(total):
        r = client.post(
            "/interview/answer",
            json={"session_id": sid, "answer": f"Answer {i}: we measured it on 12 datasets."},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        done = body["done"]
        if i < total - 1:
            assert body["next_question"]["id"] == i + 1 and body["summary"] is None
    assert done
    assert body["next_question"] is None
    assert body["summary"]["answered"] == total
    assert (
        client.post("/interview/answer", json={"session_id": sid, "answer": "x"}).status_code == 409
    )
    assert (
        client.post("/interview/answer", json={"session_id": "bad", "answer": "x"}).status_code
        == 404
    )
    assert client.post("/interview/start", json={"posting_id": "nope"}).status_code == 404
