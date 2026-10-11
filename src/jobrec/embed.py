"""Embedding backends behind one small interface.

    TfidfEmbedder   - word + bigram TF-IDF (scikit-learn). Default; no download.
    GloveEmbedder   - mean of GloVe 50-d word vectors (`make vectors`, 66 MB, public domain).
    MiniLMEmbedder  - sentence-transformers/all-MiniLM-L6-v2, optional, needs the model locally.

All return L2-normalised dense or sparse matrices, so cosine similarity is a dot product.
"""

from __future__ import annotations

import gzip
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from jobrec.text import clean, tokens

_ROOT = Path(__file__).resolve().parents[2]
VECTORS_PATH = _ROOT / "data" / "vectors" / "glove-wiki-gigaword-50.gz"


class Embedder(Protocol):
    name: str

    def fit(self, texts: Sequence[str]) -> None: ...

    def transform(self, texts: Sequence[str]) -> np.ndarray: ...


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self) -> None:
        self._vec = TfidfVectorizer(
            preprocessor=clean,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w+#.]+\b",
        )

    def fit(self, texts: Sequence[str]) -> None:
        self._vec.fit(texts)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        # Sparse matrix; rows are already L2-normalised by TfidfVectorizer.
        return self._vec.transform(texts)


class GloveEmbedder:
    """Average of word vectors, weighted by inverse document frequency learned in fit()."""

    name = "glove"

    def __init__(self, path: Path = VECTORS_PATH, max_words: int = 100_000) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `make vectors`")
        self._vectors: dict[str, np.ndarray] = {}
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            header = fh.readline().split()
            self.dim = int(header[1])
            for i, line in enumerate(fh):
                if i >= max_words:
                    break
                word, *vals = line.rstrip().split(" ")
                self._vectors[word] = np.asarray(vals, dtype=np.float32)
        self._idf: dict[str, float] = {}

    def fit(self, texts: Sequence[str]) -> None:
        n = len(texts)
        df: dict[str, int] = {}
        for text in texts:
            for tok in set(tokens(text)):
                df[tok] = df.get(tok, 0) + 1
        self._idf = {tok: float(np.log((1 + n) / (1 + c)) + 1.0) for tok, c in df.items()}

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            acc = np.zeros(self.dim, dtype=np.float32)
            weight = 0.0
            for tok in tokens(text):
                vec = self._vectors.get(tok)
                if vec is None:
                    continue
                w = self._idf.get(tok, 1.0)
                acc += w * vec
                weight += w
            if weight:
                out[i] = acc / weight
        return normalize(out)


class MiniLMEmbedder:
    """Optional sentence-transformer backend. Never required to run the demo or the tests."""

    name = "minilm"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pip install -r requirements-minilm.txt to use minilm") from exc
        self._model = SentenceTransformer(model_name)

    def fit(self, texts: Sequence[str]) -> None:  # pragma: no cover - nothing to fit
        return None

    def transform(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        return self._model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)


def make_embedder(name: str | None = None) -> Embedder:
    name = (name or os.environ.get("JOBREC_EMBEDDER", "tfidf")).lower()
    if name == "tfidf":
        return TfidfEmbedder()
    if name == "glove":
        return GloveEmbedder()
    if name == "minilm":
        return MiniLMEmbedder()
    raise ValueError(f"unknown embedder {name!r}; use tfidf, glove or minilm")
