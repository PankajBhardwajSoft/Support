"""Local SentenceTransformers embedding model, loaded from ./models with no network."""

from __future__ import annotations

import os
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = os.path.join(ROOT, "models", "all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_model():
    if not (os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR)):
        raise RuntimeError(
            f"Embedding model not found at {MODEL_DIR}. "
            "Run `python scripts/download_model.py` once (needs internet)."
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_DIR, device="cpu")


def embed(texts: list[str]) -> list[list[float]]:
    """Unit-length embeddings, so a dot product equals cosine similarity."""
    vectors = get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]
