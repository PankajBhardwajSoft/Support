"""Policy search tool for the Retrieval Agent: step 2 retrieval over the recommended collection."""

from __future__ import annotations

import json
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from ola_agent.rag.generate import RECOMMENDED_COLLECTION, TOP_K, retrieve


class PolicySearchInput(BaseModel):
    query: str = Field(description="The customer's question")


class PolicySearchTool(BaseTool):
    name: str = "search_ola_policies"
    description: str = "Search the Ola support policy documents and return the closest passages with their similarity."
    args_schema: type[BaseModel] = PolicySearchInput
    client: Any = Field(default=None, exclude=True)  # ChromaDB client; None means ./chroma_db
    _calls: list[str] = PrivateAttr(default_factory=list)

    @property
    def calls(self) -> list[str]:
        return self._calls

    def _run(self, query: str) -> str:
        self._calls.append(query)
        chunks = retrieve(query, RECOMMENDED_COLLECTION, TOP_K, self.client)
        return json.dumps({
            "query": query,
            "collection": RECOMMENDED_COLLECTION,
            "chunks": [
                {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text, "similarity": round(c.similarity, 3)}
                for c in chunks
            ],
        })
