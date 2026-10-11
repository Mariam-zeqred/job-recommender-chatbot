"""Text normalisation shared by the skill extractor and the embedders."""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
# Keep characters that occur inside skill names: letters, digits, + # . / - and spaces.
_JUNK = re.compile(r"[^a-z0-9+#./\- ]")


def clean(text: str) -> str:
    """Lower-case, strip accents, drop punctuation that never appears in skill names.

    >>> clean("Senior C++/Python Dev (Remote)!")
    'senior c++/python dev remote'
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("\n", " ")
    text = _JUNK.sub(" ", text)
    return _WS.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    """Whitespace tokens of the cleaned text, with trailing punctuation removed."""
    return [t.strip(".,/-") for t in clean(text).split() if t.strip(".,/-")]
