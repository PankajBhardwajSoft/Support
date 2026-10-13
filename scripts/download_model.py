"""One-time download of the embedding model into ./models (needs internet once).

Run:  python scripts/download_model.py

After this, every other script loads the model from ./models with
HF_HUB_OFFLINE=1, so graded runs need no network.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ola_agent.rag.embeddings import MODEL_DIR, MODEL_NAME  # noqa: E402


def main() -> None:
    from sentence_transformers import SentenceTransformer

    if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
        print(f"Model already present at {MODEL_DIR}")
        return
    print(f"Downloading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    model.save(MODEL_DIR)
    print(f"Saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
