from __future__ import annotations

import re

from ..stores.base import GraphStore
from .base import Retriever

_TOKEN = re.compile(r"[a-z0-9]+")


class KeywordRetriever(Retriever):
    """Lexical entry into the graph. No index, no embedding cost."""

    name = "text"

    def __init__(self, store: GraphStore) -> None:
        self.store = store

    async def search(self, domain: str, query: str, top_k: int) -> list[str]:
        terms = _TOKEN.findall(query.lower())
        return await self.store.keyword_search(domain, terms, top_k)
