"""Text embeddings for support-ticket retrieval.

Both backends share the small ``Embedder`` interface: ``fit`` on the ticket
texts, then ``encode``. ``encode`` always returns float32 rows of unit length,
so the inner product of two rows is their cosine similarity, which is what the
FAISS index ranks by.

  sentence_transformers  a pretrained MiniLM model. It knows paraphrases
                         ("hangs" ~ "times out") but needs the model files.
  lsa                    TF-IDF + truncated SVD fitted on the tickets
                         themselves. Fully offline; the fallback.

``make_embedder`` picks the backend from the config. ``"auto"`` tries
sentence-transformers from the local Hugging Face cache only (no download, so
an offline machine does not wait on network retries) and falls back to LSA,
with a warning, when the model cannot be loaded (package missing, model not
cached, ...). An explicit ``"sentence_transformers"`` may download the model.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from kpi_rca.config import RetrievalConfig

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str  # reported as RetrievalResult.backend

    def fit(self, texts: Sequence[str]) -> Embedder: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """float32 array of shape (len(texts), d), C-contiguous, every row of unit length."""
        ...


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise each row. An all-zero row (a text with no known words) stays zero, not NaN."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


# --- backends -----------------------------------------------------------------


class LSAEmbedder:
    """Latent semantic analysis: TF-IDF of words and word pairs, reduced by truncated SVD."""

    name = "lsa"

    def __init__(self, dim: int = 128, random_state: int = 42) -> None:
        self.dim = dim
        self.random_state = random_state
        self._tfidf: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None

    def fit(self, texts: Sequence[str]) -> LSAEmbedder:
        tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english")
        weights = tfidf.fit_transform(texts)
        n_docs, n_terms = weights.shape
        # TruncatedSVD needs fewer components than terms, and we keep fewer than documents.
        # A tiny corpus still gets one component.
        n_components = max(1, min(self.dim, n_terms - 1, n_docs - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=self.random_state).fit(weights)
        self._tfidf = tfidf
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._tfidf is None or self._svd is None:
            raise RuntimeError("LSAEmbedder.encode() called before fit()")
        return _unit_rows(self._svd.transform(self._tfidf.transform(texts)))


class SentenceTransformerEmbedder:
    """A pretrained sentence-transformers model; nothing is learned from the tickets."""

    name = "sentence_transformers"

    def __init__(self, model_name: str, local_files_only: bool = False) -> None:
        # Must be set before the import: huggingface_hub reads it once, when it is imported.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from sentence_transformers import SentenceTransformer  # lazy: it imports torch, which is slow

        self.model_name = model_name
        # local_files_only=True reads the model from the Hugging Face cache and never downloads
        self._model = SentenceTransformer(model_name, local_files_only=local_files_only)

    def fit(self, texts: Sequence[str]) -> SentenceTransformerEmbedder:
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.ascontiguousarray(vectors, dtype=np.float32)


# --- choosing a backend -------------------------------------------------------------


def _load_sentence_transformer(model_name: str, local_files_only: bool = False) -> SentenceTransformerEmbedder:
    try:
        return SentenceTransformerEmbedder(model_name, local_files_only=local_files_only)
    except Exception as exc:  # package missing, no network, unknown model name, ...
        raise RuntimeError(
            f"could not load sentence-transformers model {model_name!r}: {type(exc).__name__}: {exc}"
        ) from exc


def make_embedder(cfg: RetrievalConfig, backend: str | None = None) -> Embedder:
    """The embedder for `backend`, or for cfg.backend when `backend` is None."""
    backend = cfg.backend if backend is None else backend
    if backend == "lsa":
        return LSAEmbedder(cfg.lsa_dim, cfg.random_state)
    if backend == "sentence_transformers":
        return _load_sentence_transformer(cfg.model_name)
    if backend == "auto":
        try:
            # cache only: "auto" must stay fast offline; ask for "sentence_transformers" to download
            return _load_sentence_transformer(cfg.model_name, local_files_only=True)
        except RuntimeError as exc:
            log.warning("%s; falling back to LSA embeddings", exc)
            return LSAEmbedder(cfg.lsa_dim, cfg.random_state)
    raise ValueError(f"unknown retrieval backend {backend!r}; expected 'auto', 'sentence_transformers' or 'lsa'")
