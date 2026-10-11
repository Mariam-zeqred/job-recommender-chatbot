"""FastAPI app: recommendations and a mock-interview session, plus the static front end.

Run:  uvicorn jobrec.api:app --app-dir src --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from jobrec import __version__
from jobrec.data import POSTINGS_CSV, Posting, load_postings
from jobrec.embed import make_embedder
from jobrec.interview import InterviewSession
from jobrec.providers import get_provider
from jobrec.recommend import Recommender

STATIC = Path(__file__).parent / "static"
MAX_SESSIONS = 200


class State:
    """Everything loaded at start-up. Kept on the app so tests can build their own."""

    def __init__(self, postings: list[Posting] | None = None) -> None:
        self.postings = postings or load_postings(os.environ.get("JOBREC_POSTINGS", POSTINGS_CSV))
        self.by_id = {p.id: p for p in self.postings}
        self.recommender = Recommender(self.postings, make_embedder())
        self.provider = get_provider()
        self.sessions: dict[str, InterviewSession] = {}


class RecommendRequest(BaseModel):
    cv_text: str = Field(min_length=20, max_length=20_000)
    k: int = Field(default=5, ge=1, le=20)


class PostingOut(BaseModel):
    id: str
    title: str
    company: str
    location: str
    seniority: str
    required_skills: list[str]
    nice_to_have: list[str]
    description: str


class RecommendationOut(BaseModel):
    posting: PostingOut
    score: float
    similarity: float
    skill_coverage: float
    matched_skills: list[str]
    missing_skills: list[str]
    bonus_skills: list[str]
    explanation: str


class RecommendResponse(BaseModel):
    cv_skills: list[str]
    embedder: str
    results: list[RecommendationOut]


class StartRequest(BaseModel):
    posting_id: str
    cv_text: str = Field(default="", max_length=20_000)


class QuestionOut(BaseModel):
    id: int
    kind: str
    skill: str | None
    text: str


class StartResponse(BaseModel):
    session_id: str
    posting: PostingOut
    total_questions: int
    question: QuestionOut


class AnswerRequest(BaseModel):
    session_id: str
    answer: str = Field(max_length=5_000)


class FeedbackOut(BaseModel):
    question_id: int
    score: int
    covered: list[str]
    missed: list[str]
    comment: str
    follow_up: str | None


class SummaryOut(BaseModel):
    posting_id: str
    answered: int
    average_score: float
    strongest: str | None
    weakest: str | None
    missing_skills: list[str]
    advice: str


class AnswerResponse(BaseModel):
    feedback: FeedbackOut
    next_question: QuestionOut | None
    done: bool
    summary: SummaryOut | None


def _posting_out(p: Posting) -> PostingOut:
    return PostingOut(
        id=p.id,
        title=p.title,
        company=p.company,
        location=p.location,
        seniority=p.seniority,
        required_skills=list(p.required_skills),
        nice_to_have=list(p.nice_to_have),
        description=p.description,
    )


def create_app(state: State | None = None) -> FastAPI:
    app = FastAPI(title="job-recommender-chatbot", version=__version__)
    st = state or State()
    app.state.jobrec = st

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return (STATIC / "index.html").read_text("utf-8")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "postings": len(st.postings),
            "embedder": st.recommender.embedder.name,
            "llm_provider": st.provider.name,
        }

    @app.get("/postings/{posting_id}", response_model=PostingOut)
    def posting(posting_id: str) -> PostingOut:
        p = st.by_id.get(posting_id)
        if p is None:
            raise HTTPException(404, f"no posting {posting_id}")
        return _posting_out(p)

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest) -> RecommendResponse:
        from jobrec.skills import extract_skills

        recs = st.recommender.recommend(req.cv_text, req.k)
        return RecommendResponse(
            cv_skills=extract_skills(req.cv_text),
            embedder=st.recommender.embedder.name,
            results=[
                RecommendationOut(
                    posting=_posting_out(r.posting),
                    score=round(r.score, 4),
                    similarity=round(r.similarity, 4),
                    skill_coverage=round(r.skill_coverage, 4),
                    matched_skills=list(r.matched_skills),
                    missing_skills=list(r.missing_skills),
                    bonus_skills=list(r.bonus_skills),
                    explanation=r.explanation,
                )
                for r in recs
            ],
        )

    @app.post("/interview/start", response_model=StartResponse)
    def interview_start(req: StartRequest) -> StartResponse:
        p = st.by_id.get(req.posting_id)
        if p is None:
            raise HTTPException(404, f"no posting {req.posting_id}")
        if len(st.sessions) >= MAX_SESSIONS:
            st.sessions.pop(next(iter(st.sessions)))  # drop the oldest
        session = InterviewSession(p, req.cv_text, st.provider)
        sid = uuid.uuid4().hex
        st.sessions[sid] = session
        q = session.current_question()
        assert q is not None
        return StartResponse(
            session_id=sid,
            posting=_posting_out(p),
            total_questions=len(session.questions),
            question=QuestionOut(id=q.id, kind=q.kind, skill=q.skill, text=q.text),
        )

    @app.post("/interview/answer", response_model=AnswerResponse)
    def interview_answer(req: AnswerRequest) -> AnswerResponse:
        session = st.sessions.get(req.session_id)
        if session is None:
            raise HTTPException(404, "unknown session")
        if session.done:
            raise HTTPException(409, "interview already finished")
        fb = session.answer(req.answer)
        q = session.current_question()
        summary = session.summary() if session.done else None
        return AnswerResponse(
            feedback=FeedbackOut(
                question_id=fb.question_id,
                score=fb.score,
                covered=list(fb.covered),
                missed=list(fb.missed),
                comment=fb.comment,
                follow_up=fb.follow_up,
            ),
            next_question=QuestionOut(id=q.id, kind=q.kind, skill=q.skill, text=q.text)
            if q
            else None,
            done=session.done,
            summary=SummaryOut(**summary.__dict__) if summary else None,
        )

    return app


app = create_app()
