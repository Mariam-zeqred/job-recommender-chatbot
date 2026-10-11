"""Loading the postings table and the evaluation set."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POSTINGS_CSV = ROOT / "data" / "postings.csv"
EVAL_DIR = ROOT / "data" / "eval"

REQUIRED_COLUMNS = {"id", "title", "company", "location", "description"}


@dataclass(frozen=True)
class Posting:
    id: str
    title: str
    company: str
    location: str
    description: str
    required_skills: tuple[str, ...] = field(default_factory=tuple)
    nice_to_have: tuple[str, ...] = field(default_factory=tuple)
    role_family: str = ""
    seniority: str = ""
    years_min: int = 0

    @property
    def text(self) -> str:
        """The text the recommender embeds: title first (it carries most of the signal)."""
        return f"{self.title}. {self.title}. {self.description}"


def _split(value: object) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return ()
    return tuple(s.strip() for s in str(value).split(";") if s.strip())


def load_postings(path: Path | str = POSTINGS_CSV) -> list[Posting]:
    """Read a postings CSV. Only the columns in REQUIRED_COLUMNS are mandatory, so a real
    dataset with `id,title,company,location,description` columns can be dropped in."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    out: list[Posting] = []
    for row in df.to_dict("records"):
        out.append(
            Posting(
                id=row["id"],
                title=row["title"],
                company=row["company"],
                location=row["location"],
                description=row["description"],
                required_skills=_split(row.get("required_skills")),
                nice_to_have=_split(row.get("nice_to_have")),
                role_family=row.get("role_family", ""),
                seniority=row.get("seniority", ""),
                years_min=int(row["years_min"]) if row.get("years_min", "") != "" else 0,
            )
        )
    return out


def load_eval(eval_dir: Path | str = EVAL_DIR) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """Return (queries, relevant) where queries has columns id,role_family,cv_text and
    relevant maps a query id to the set of posting ids labelled relevant for it."""
    eval_dir = Path(eval_dir)
    queries = pd.read_csv(eval_dir / "queries.csv", dtype=str, keep_default_na=False)
    labels = pd.read_csv(eval_dir / "labels.csv", dtype=str, keep_default_na=False)
    relevant: dict[str, set[str]] = {q: set() for q in queries["id"]}
    for row in labels.itertuples(index=False):
        relevant.setdefault(row.query_id, set()).add(row.posting_id)
    return queries, relevant
