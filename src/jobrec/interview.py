"""Mock interview for one posting: a question bank built from its requirements, rule-based
answer scoring, and an optional LLM provider for follow-ups and feedback wording.

The rules are simple on purpose: a candidate (or a professor) can read exactly why an answer
scored what it did. The LLM, if enabled, only rewrites feedback and asks one follow-up; it
never changes the score.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from jobrec.data import Posting
from jobrec.providers import LLMProvider, Message, NullProvider
from jobrec.skills import category_of, extract_skills, vocabulary
from jobrec.text import clean

MAX_QUESTIONS = 8


@dataclass(frozen=True)
class Question:
    id: int
    kind: str  # intro | skill | gap | behavioural
    skill: str | None
    text: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class Feedback:
    question_id: int
    score: int  # 0-100
    covered: tuple[str, ...]
    missed: tuple[str, ...]
    comment: str
    follow_up: str | None = None


@dataclass
class Summary:
    posting_id: str
    answered: int
    average_score: float
    strongest: str | None
    weakest: str | None
    missing_skills: tuple[str, ...]
    advice: str


@lru_cache(maxsize=1)
def _bank() -> dict:
    return json.loads(
        resources.files("jobrec.resources").joinpath("questions.json").read_text("utf-8")
    )


def build_questions(posting: Posting, cv_text: str = "") -> list[Question]:
    """Intro, then one question per required skill (specific template if one exists, else the
    category template), then one 'gap' question for the first missing skill, then one
    behavioural question. Capped at MAX_QUESTIONS."""
    bank = _bank()
    cv_skills = set(extract_skills(cv_text)) if cv_text else set()
    required = list(posting.required_skills) or extract_skills(posting.description)
    qs: list[Question] = []
    intro = bank["intro"][0]
    qs.append(
        Question(
            0,
            "intro",
            None,
            intro["text"].format(title=posting.title, company=posting.company),
            tuple(intro["keywords"]),
        )
    )
    # Skills the CV already has come first: they are what an interviewer probes deepest.
    ordered = [s for s in required if s in cv_skills] + [s for s in required if s not in cv_skills]
    category_use: dict[str, int] = {}
    for skill in ordered:
        if len(qs) >= MAX_QUESTIONS - 2:
            break
        if skill in bank["by_skill"]:
            tpl = bank["by_skill"][skill]
        else:
            cat = category_of(skill) if skill in vocabulary() else "process"
            templates = bank["by_category"].get(cat, bank["by_category"]["process"])
            tpl = templates[category_use.get(cat, 0) % len(templates)]
            category_use[cat] = category_use.get(cat, 0) + 1
        qs.append(
            Question(
                len(qs), "skill", skill, tpl["text"].format(skill=skill), tuple(tpl["keywords"])
            )
        )
    missing = [s for s in required if s not in cv_skills]
    asked = {q.skill for q in qs}
    # Prefer a missing skill that did not get its own question, so nothing is asked twice.
    missing = [s for s in missing if s not in asked] or missing
    if cv_text and missing:
        gap = bank["gap"][0]
        qs.append(
            Question(
                len(qs),
                "gap",
                missing[0],
                gap["text"].format(skill=missing[0]),
                tuple(gap["keywords"]),
            )
        )
    beh = bank["behavioural"][len(posting.id) % len(bank["behavioural"])]
    qs.append(Question(len(qs), "behavioural", None, beh["text"], tuple(beh["keywords"])))
    return qs[:MAX_QUESTIONS]


_NUMBER = re.compile(r"\d")


def score_answer(question: Question, answer: str) -> Feedback:
    """Rule-based score: 60 % keyword coverage, 25 % length (saturates at 60 words), 15 % for
    giving a concrete example (a number or an 'I/we did' phrase)."""
    text = clean(answer)
    words = text.split()
    covered = tuple(k for k in question.keywords if k in text)
    missed = tuple(k for k in question.keywords if k not in text)
    coverage = len(covered) / len(question.keywords) if question.keywords else 0.0
    length = min(len(words), 60) / 60
    concrete = bool(_NUMBER.search(text)) or bool(
        re.search(r"\b(i|we) (built|wrote|used|fixed|measured|designed|tested|ran)\b", text)
    )
    score = round(100 * (0.60 * coverage + 0.25 * length + 0.15 * concrete))
    if not words:
        comment = "No answer given."
    elif score >= 75:
        comment = "Strong answer: specific and covers the main points."
    elif score >= 45:
        comment = "Reasonable answer. It would be stronger with a concrete example"
        comment += f" and by mentioning: {', '.join(missed[:3])}." if missed else "."
    else:
        comment = "Too thin. An interviewer would expect you to talk about"
        comment += f" {', '.join(missed[:4])}" if missed else " the question"
        comment += " and to give one concrete example with numbers."
    return Feedback(question.id, score, covered, missed, comment)


class InterviewSession:
    """State for one candidate interviewing for one posting."""

    def __init__(
        self, posting: Posting, cv_text: str = "", provider: LLMProvider | None = None
    ) -> None:
        self.posting = posting
        self.cv_text = cv_text
        self.provider = provider or NullProvider()
        self.questions = build_questions(posting, cv_text)
        self.feedback: list[Feedback] = []
        self.transcript: list[Message] = []

    @property
    def done(self) -> bool:
        return len(self.feedback) >= len(self.questions)

    def current_question(self) -> Question | None:
        return None if self.done else self.questions[len(self.feedback)]

    def answer(self, text: str) -> Feedback:
        q = self.current_question()
        if q is None:
            raise RuntimeError("interview is finished")
        fb = score_answer(q, text)
        follow_up = self._follow_up(q, text, fb)
        fb = Feedback(fb.question_id, fb.score, fb.covered, fb.missed, fb.comment, follow_up)
        self.feedback.append(fb)
        self.transcript += [
            {"role": "assistant", "content": q.text},
            {"role": "user", "content": text},
        ]
        return fb

    def _follow_up(self, q: Question, answer: str, fb: Feedback) -> str | None:
        if fb.score >= 75 or not fb.missed:
            return None
        rule = f"Could you say more about {fb.missed[0]}?"
        prompt: list[Message] = [
            {
                "role": "system",
                "content": (
                    "You are a friendly technical interviewer. Ask ONE short follow-up question "
                    "(max 25 words) that probes what the candidate's answer did not cover. "
                    "Do not grade, do not add preamble."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Role: {self.posting.title}\nQuestion: {q.text}\nAnswer: {answer}\n"
                    f"Points not covered: {', '.join(fb.missed)}"
                ),
            },
        ]
        return self.provider.complete(prompt, max_tokens=60) or rule

    def summary(self) -> Summary:
        scored = [(fb, self.questions[fb.question_id]) for fb in self.feedback]
        if not scored:
            return Summary(self.posting.id, 0, 0.0, None, None, (), "No answers yet.")
        avg = sum(fb.score for fb, _ in scored) / len(scored)
        by_skill = [(fb.score, q.skill) for fb, q in scored if q.skill]
        strongest = max(by_skill)[1] if by_skill else None
        weakest = min(by_skill)[1] if by_skill else None
        cv_skills = set(extract_skills(self.cv_text)) if self.cv_text else set()
        missing = tuple(s for s in self.posting.required_skills if s not in cv_skills)
        advice = (
            f"Average {avg:.0f}/100 over {len(scored)} answers. "
            + (f"Best on {strongest}, weakest on {weakest}. " if by_skill else "")
            + (f"Before applying, prepare examples for: {', '.join(missing)}." if missing else "")
        ).strip()
        return Summary(
            self.posting.id, len(scored), round(avg, 1), strongest, weakest, missing, advice
        )
