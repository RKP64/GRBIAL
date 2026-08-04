from __future__ import annotations

from abc import ABC, abstractmethod


class Retriever(ABC):
    """Entry-point selection seam.

    The query service only calls `search(...) -> list[node_id]`. Graph expansion
    happens afterwards and is identical for every retriever, so swapping FAISS
    for Azure AI Search is a new class here and nothing else.
    """

    name: str = "base"

    @abstractmethod
    async def search(self, domain: str, query: str, top_k: int) -> list[str]:
        ...

    async def build(self, domain: str) -> dict:
        """Optional index build. No-op for retrievers that need no index."""
        return {"retriever": self.name, "indexed": 0, "note": "No index is required."}
