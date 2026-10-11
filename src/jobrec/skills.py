"""Skill vocabulary and extraction.

The vocabulary (data/skills.json) maps a canonical skill name to a category and a list of
aliases. Extraction is dictionary-based: every canonical name and alias is matched as a
whole phrase in the cleaned text. This is deliberately simple and fully inspectable; its
limits (no fuzzy matching, no context) are discussed in the README.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from jobrec.text import clean


@dataclass(frozen=True)
class Skill:
    name: str
    category: str
    aliases: tuple[str, ...]


@lru_cache(maxsize=1)
def vocabulary() -> dict[str, Skill]:
    raw = json.loads(resources.files("jobrec.resources").joinpath("skills.json").read_text("utf-8"))
    return {
        name: Skill(name, entry["category"], tuple(entry["aliases"])) for name, entry in raw.items()
    }


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    # Whole-phrase match. Skill names may end in "+" or "#", so the boundary after the phrase
    # is "not followed by a letter/digit" rather than \b.
    return re.compile(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9+#])")


@lru_cache(maxsize=1)
def _patterns() -> list[tuple[re.Pattern[str], str]]:
    out: list[tuple[re.Pattern[str], str]] = []
    for skill in vocabulary().values():
        for phrase in (skill.name, *skill.aliases):
            out.append((_phrase_pattern(clean(phrase)), skill.name))
    # Longer phrases first so "aws iot core" wins over "aws" when both are present.
    out.sort(key=lambda p: -len(p[0].pattern))
    return out


def extract_skills(text: str) -> list[str]:
    """Canonical skill names found in `text`, in order of first appearance, no duplicates."""
    cleaned = clean(text)
    found: dict[str, int] = {}
    for pattern, name in _patterns():
        m = pattern.search(cleaned)
        if m and (name not in found or m.start() < found[name]):
            found[name] = m.start()
    return [name for name, _ in sorted(found.items(), key=lambda kv: kv[1])]


def category_of(skill: str) -> str:
    return vocabulary()[skill].category
