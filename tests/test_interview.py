import pytest

from jobrec.interview import MAX_QUESTIONS, InterviewSession, build_questions, score_answer
from jobrec.providers import NullProvider, get_provider

CV = "ESP32 firmware in C with FreeRTOS, MQTT telemetry, Python with pandas."


def _posting(postings, family):
    return next(p for p in postings if p.role_family == family)


def test_question_bank_shape(postings):
    p = _posting(postings, "iot_engineer")
    qs = build_questions(p, CV)
    assert qs[0].kind == "intro" and p.title in qs[0].text
    assert qs[-1].kind == "behavioural"
    assert len(qs) <= MAX_QUESTIONS
    assert [q.id for q in qs] == list(range(len(qs)))
    skills_asked = [q.skill for q in qs if q.kind == "skill"]
    assert set(skills_asked) <= set(p.required_skills)
    # Skills the CV has are asked before the ones it lacks.
    has = [s in CV.lower() for s in skills_asked]
    assert has == sorted(has, reverse=True)


def test_gap_question_only_with_cv_and_not_duplicated(postings):
    p = _posting(postings, "iot_engineer")
    assert all(q.kind != "gap" for q in build_questions(p, ""))
    qs = build_questions(p, CV)
    gaps = [q for q in qs if q.kind == "gap"]
    if gaps:
        assert gaps[0].skill not in CV.lower()
        asked = [q.skill for q in qs if q.kind == "skill"]
        if len(set(p.required_skills) - set(asked)) > 0:
            assert gaps[0].skill not in asked


def test_scoring_rewards_coverage_length_and_concreteness(postings):
    q = build_questions(_posting(postings, "iot_engineer"), CV)[0]
    empty = score_answer(q, "")
    thin = score_answer(q, "I like this job.")
    good = score_answer(
        q,
        "I have two years of experience building IoT devices; my main project was a sensor "
        "node and I want to join your team because I can learn cloud skills here. We built "
        "and shipped 40 units.",
    )
    assert empty.score == 0 and empty.comment == "No answer given."
    assert thin.score < good.score <= 100
    assert set(good.covered) >= {"experience", "project", "team", "learn"}
    assert good.missed == tuple(k for k in q.keywords if k not in good.covered)


def test_session_flow_and_summary(postings):
    p = _posting(postings, "embedded_firmware")
    s = InterviewSession(p, CV, NullProvider())
    n = len(s.questions)
    for i in range(n):
        assert not s.done
        fb = s.answer(f"Answer {i}: we built it and tested it on 3 boards with a watchdog.")
        assert 0 <= fb.score <= 100
        if fb.score < 75 and fb.missed:
            assert fb.follow_up and fb.follow_up.startswith("Could you say more about")
    assert s.done
    with pytest.raises(RuntimeError):
        s.answer("too late")
    summary = s.summary()
    assert summary.answered == n
    assert 0 <= summary.average_score <= 100
    assert set(summary.missing_skills) == set(p.required_skills) - set(
        __import__("jobrec.skills", fromlist=["x"]).extract_skills(CV)
    )
    assert len(s.transcript) == 2 * n


def test_default_provider_is_off(monkeypatch):
    monkeypatch.delenv("JOBREC_LLM_PROVIDER", raising=False)
    assert get_provider().name == "none"
    assert get_provider().complete([{"role": "user", "content": "hi"}]) is None
    with pytest.raises(ValueError):
        get_provider("gpt")
    monkeypatch.setenv("JOBREC_LLM_PROVIDER", "http")
    with pytest.raises(ValueError):  # no URL/model configured -> refuse, do not guess
        get_provider()
