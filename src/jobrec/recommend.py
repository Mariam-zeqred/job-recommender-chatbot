"""Top-k posting recommendation from CV text, with a skill-based explanation per hit."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from jobrec.data import Posting
from jobrec.embed import Embedder, TfidfEmbedder
from jobrec.skills import extract_skills


@dataclass(frozen=True)
class Recommendation:
    posting: Posting
    score: float  # final blended score in [0, 1]
    similarity: float  # cosine similarity of the texts
    skill_coverage: float  # fraction of the posting's required skills found in the CV
    matched_skills: tuple[str, ...]
    missing_skills: tuple[str, ...]
    bonus_skills: tuple[str, ...]  # nice-to-have skills the CV has

    @property
    def explanation(self) -> str:
        parts = []
        if self.matched_skills:
            parts.append("matches " + ", ".join(self.matched_skills))
        if self.missing_skills:
            parts.append("missing " + ", ".join(self.missing_skills))
        if self.bonus_skills:
            parts.append("bonus " + ", ".join(self.bonus_skills))
        return "; ".join(parts) if parts else "no skill overlap found; ranked by text similarity"


class Recommender:
    """Fit once on the postings, then `recommend` any number of CVs.

    score = (1 - alpha) * cosine_similarity + alpha * required_skill_coverage

    alpha=0 is pure text similarity; alpha=1 ranks only by the share of required skills the
    CV mentions. The default 0.5 is what the evaluation in scripts/evaluate.py compares.
    """

    def __init__(
        self, postings: Sequence[Posting], embedder: Embedder | None = None, alpha: float = 0.5
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.postings = list(postings)
        self.embedder = embedder or TfidfEmbedder()
        self.alpha = alpha
        texts = [p.text for p in self.postings]
        self.embedder.fit(texts)
        self._matrix = self.embedder.transform(texts)
        # Required skills: the CSV column if present, else extracted from the description.
        self._required = [
            p.required_skills or tuple(extract_skills(p.description)) for p in self.postings
        ]
        self._nice = [p.nice_to_have for p in self.postings]

    def similarities(self, cv_text: str) -> np.ndarray:
        q = self.embedder.transform([cv_text])
        sims = self._matrix @ q.T
        sims = sims.toarray() if hasattr(sims, "toarray") else np.asarray(sims)
        return np.clip(sims.ravel(), 0.0, 1.0)

    def recommend(self, cv_text: str, k: int = 5) -> list[Recommendation]:
        if not cv_text.strip():
            raise ValueError("cv_text is empty")
        cv_skills = set(extract_skills(cv_text))
        sims = self.similarities(cv_text)
        coverage = np.array(
            [len(cv_skills.intersection(req)) / len(req) if req else 0.0 for req in self._required]
        )
        scores = (1 - self.alpha) * sims + self.alpha * coverage
        order = np.argsort(-scores, kind="stable")[:k]
        out: list[Recommendation] = []
        for i in order:
            req = self._required[i]
            matched = tuple(s for s in req if s in cv_skills)
            missing = tuple(s for s in req if s not in cv_skills)
            bonus = tuple(s for s in self._nice[i] if s in cv_skills)
            out.append(
                Recommendation(
                    posting=self.postings[i],
                    score=float(scores[i]),
                    similarity=float(sims[i]),
                    skill_coverage=float(coverage[i]),
                    matched_skills=matched,
                    missing_skills=missing,
                    bonus_skills=bonus,
                )
            )
        return out
