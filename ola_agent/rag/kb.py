"""Loads the knowledge-base documents from ./kb."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

from ola_agent.rag.embeddings import ROOT

KB_DIR = os.path.join(ROOT, "kb")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


def parse_document(raw: str) -> Document:
    """Parse a KB file: `doc_id:` line, `title:` line, blank line, body."""
    header, _, body = raw.strip().partition("\n\n")
    fields = {}
    for line in header.splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    if not fields.get("doc_id") or not body.strip():
        raise ValueError("KB document needs a doc_id line and a body")
    return Document(doc_id=fields["doc_id"], title=fields.get("title", ""), text=" ".join(body.split()))


def load_documents(kb_dir: str = KB_DIR) -> list[Document]:
    docs = []
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.md"))):
        with open(path, encoding="utf-8") as f:
            docs.append(parse_document(f.read()))
    return docs
