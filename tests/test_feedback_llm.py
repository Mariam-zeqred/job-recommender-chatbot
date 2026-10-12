"""The LLM feedback path: the provider may reword, the rule keeps the score."""

from collections.abc import Sequence

from fastapi.testclient import TestClient

from jobrec.api import State, create_app
from jobrec.interview import InterviewSession, parse_review, rule_tip, score_answer
from jobrec.providers import Message, NullProvider

CV = "ESP32 firmware in C with FreeRTOS, MQTT telemetry, Python with pandas."
THIN = "I used MQTT once."


class CannedProvider:
    """Returns a fixed reply and records what it was asked."""

    name = "canned"

    def __init__(self, reply: str | None) -> None:
        self.reply = reply
        self.calls: list[Sequence[Message]] = []

    def complete(self, messages: Sequence[Message], max_tokens: int = 200) -> str | None:
        self.calls.append(messages)
        return self.reply


def _posting(postings):
    return next(p for p in postings if p.role_family == "iot_engineer")


def test_parse_review_is_tolerant():
    text = (
        "Sure! Here you go:\n"
        "- FEEDBACK: You named the broker but not QoS.\n"
        "next time: Say which QoS level you chose and why.\n"
        'FOLLOW-UP: "Which QoS did your telemetry use?"\n'
    )
    got = parse_review(text)
    assert got == {
        "feedback": "You named the broker but not QoS.",
        "next time": "Say which QoS level you chose and why.",
        "follow-up": "Which QoS did your telemetry use?",
    }
    assert parse_review(None) == {} and parse_review("no labels here") == {}
    assert parse_review("FEEDBACK:   \nNEXT TIME: x") == {"next time": "x"}


def test_rule_tip_covers_every_case():
    assert "mention a" in rule_tip(("a", "b"), False) and "number" in rule_tip(("a",), False)
    assert rule_tip(("a",), True) == "Next time, also cover a."
    assert "concrete example" in rule_tip((), False)
    assert rule_tip((), True).startswith("Keep this shape")


def test_null_provider_gives_rule_text(postings):
    s = InterviewSession(_posting(postings), CV, NullProvider())
    q = s.current_question()
    fb = s.answer(THIN)
    rule = score_answer(q, THIN)
    assert (fb.score, fb.covered, fb.missed, fb.comment, fb.tip) == (
        rule.score,
        rule.covered,
        rule.missed,
        rule.comment,
        rule.tip,
    )
    assert fb.source == "rule"
    assert fb.follow_up == f"Could you say more about {rule.missed[0]}?"


def test_llm_rewords_but_never_scores(postings):
    provider = CannedProvider(
        "FEEDBACK: Good start, but say what you did with it.\n"
        "NEXT TIME: Describe one project and one number.\n"
        "FOLLOW-UP: What did the device publish, and how often?"
    )
    s = InterviewSession(_posting(postings), CV, provider)
    q = s.current_question()
    fb = s.answer(THIN)
    rule = score_answer(q, THIN)
    assert fb.score == rule.score and fb.covered == rule.covered and fb.missed == rule.missed
    assert fb.comment == "Good start, but say what you did with it."
    assert fb.tip == "Describe one project and one number."
    assert fb.follow_up == "What did the device publish, and how often?"
    assert fb.source == "canned"
    # One call per answer, and the prompt carries the rule's verdict so the LLM presents it.
    assert len(provider.calls) == 1
    user = provider.calls[0][-1]["content"]
    assert f"Rule-based score: {rule.score}/100" in user and THIN in user


def test_llm_follow_up_only_when_the_rule_wants_one(postings):
    provider = CannedProvider("FEEDBACK: Fine.\nNEXT TIME: Add a number.\nFOLLOW-UP: Why?")
    s = InterviewSession(_posting(postings), CV, provider)
    strong = (
        "I have three years of experience with ESP32 devices; my main project was a sensor "
        "node and I want to join your team to learn cloud skills. We built and shipped 40 units."
    )
    fb = s.answer(strong)
    assert fb.score >= 75
    assert fb.follow_up is None  # rule did not want one, so the LLM's is dropped
    assert fb.comment == "Fine." and fb.source == "canned"
    provider2 = CannedProvider("FEEDBACK: Thin.\nNEXT TIME: More.\nFOLLOW-UP: none")
    s2 = InterviewSession(_posting(postings), CV, provider2)
    fb2 = s2.answer(THIN)
    assert fb2.follow_up == f"Could you say more about {fb2.missed[0]}?"


def test_garbage_or_failed_llm_falls_back(postings):
    for reply in (None, "", "I cannot help with that.", "FEEDBACK:\nNEXT TIME:\n"):
        s = InterviewSession(_posting(postings), CV, CannedProvider(reply))
        fb = s.answer(THIN)
        rule = score_answer(s.questions[0], THIN)
        assert fb.score == rule.score and fb.comment == rule.comment and fb.tip == rule.tip
        assert fb.source == "rule"
    # Partial reply: the labelled lines that exist are used, the rest stays rule-based.
    s = InterviewSession(_posting(postings), CV, CannedProvider("NEXT TIME: Name the broker."))
    fb = s.answer(THIN)
    assert fb.tip == "Name the broker." and fb.comment == score_answer(s.questions[0], THIN).comment
    assert fb.source == "canned"


def test_empty_answer_never_calls_the_llm(postings):
    provider = CannedProvider("FEEDBACK: x\nNEXT TIME: y\nFOLLOW-UP: z")
    s = InterviewSession(_posting(postings), CV, provider)
    fb = s.answer("   ")
    assert fb.score == 0 and fb.comment == "No answer given." and provider.calls == []


def test_api_exposes_tip_and_source(postings):
    st = State(postings)
    st.provider = CannedProvider("FEEDBACK: Clear.\nNEXT TIME: Quote a number.\nFOLLOW-UP: none")
    client = TestClient(create_app(st))
    start = client.post(
        "/interview/start", json={"posting_id": _posting(postings).id, "cv_text": CV}
    ).json()
    r = client.post(
        "/interview/answer", json={"session_id": start["session_id"], "answer": THIN}
    ).json()
    fb = r["feedback"]
    assert fb["tip"] == "Quote a number." and fb["source"] == "canned" and fb["comment"] == "Clear."
    assert isinstance(fb["score"], int)
