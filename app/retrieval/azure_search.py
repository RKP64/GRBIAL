from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..config import get_settings
from ..services.llm import embed_texts
from .base import Retriever

log = logging.getLogger(__name__)

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.aio import SearchClient as AsyncSearchClient
    from azure.search.documents.models import VectorizedQuery
    AZURE_SEARCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    AZURE_SEARCH_AVAILABLE = False

# Reuse SSL/TCP connections across concurrent searches — creating a client per
# query is the single biggest avoidable latency cost.
_CLIENTS: dict[str, Any] = {}

CONTENT_FIELDS = ("chunk", "content", "text", "page_content", "body")
TITLE_FIELDS = ("page_title", "document_title", "title", "group_name", "autocomplete")
URL_FIELDS = ("citation_url", "source_url", "url", "source_reference", "filepath")


class AzureAISearchRetriever(Retriever):
    """Hybrid retrieval against an Azure AI Search index.

    Hybrid means one request carries both a lexical query (`search_text`, BM25)
    and a vector query (`vector_queries`); Azure fuses the rankings. Semantic
    ranking adds a reranker pass on top when a semantic configuration exists.

    Robustness follows the production pattern this was modelled on:
      * a cached client per index
      * a hard timeout so the SDK cannot retry silently for a minute
      * a four-step fallback ladder, because index schemas vary and a rejected
        field name should degrade the query, not fail it
    """

    name = "documents"

    def __init__(self) -> None:
        if not AZURE_SEARCH_AVAILABLE:
            raise RuntimeError("azure-search-documents is not installed")
        s = get_settings()
        if not (s.azure_search_endpoint and s.azure_search_api_key):
            raise RuntimeError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY are required")
        self.endpoint = s.azure_search_endpoint
        self.api_key = s.azure_search_api_key
        self.index = s.azure_search_index
        self.vector_field = s.azure_search_vector_field
        self.semantic_config = s.azure_search_semantic_config
        self.timeout = s.azure_search_timeout_seconds

    # ---------------------------------------------------------------- client
    def _client(self, index_name: str):
        if index_name not in _CLIENTS:
            _CLIENTS[index_name] = AsyncSearchClient(
                endpoint=self.endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(self.api_key),
            )
        return _CLIENTS[index_name]

    async def _attempt(self, client, kwargs: dict) -> list[dict]:
        """Run one search with a timeout shield and drain the async pager."""
        results = await asyncio.wait_for(client.search(**kwargs), timeout=self.timeout)
        docs: list[dict] = []
        async for doc in results:
            docs.append(dict(doc))
        return docs

    # ---------------------------------------------------------------- search
    async def search_documents(
        self,
        query: str,
        *,
        index_name: str | None = None,
        k: int = 5,
        use_semantic: bool = True,
        filter_expression: str | None = None,
    ) -> list[dict]:
        """Return normalised, ranked, deduplicated documents."""
        index_name = index_name or self.index
        if not index_name:
            raise RuntimeError("No Azure Search index configured")
        client = self._client(index_name)
        cleaned = (query or "").strip()
        fetch_k = max(k * 2, k)

        # Vector half of the hybrid query. If embeddings fail (rate limit,
        # timeout) we degrade to lexical-only rather than failing the request.
        vector: list[float] | None = None
        try:
            vecs = await asyncio.wait_for(embed_texts([cleaned]), timeout=self.timeout)
            vector = vecs[0] if vecs else None
        except Exception as exc:
            log.warning("Embedding failed, using lexical-only search: %s", exc)

        base: dict[str, Any] = {
            "search_text": cleaned or "*",
            "top": fetch_k,
        }
        if filter_expression:
            base["filter"] = filter_expression
        if vector and self.vector_field:
            base["vector_queries"] = [
                VectorizedQuery(vector=vector, k_nearest_neighbors=fetch_k,
                                fields=self.vector_field)
            ]

        docs: list[dict] | None = None
        semantic_used = False
        last_error: Exception | None = None

        # 1) semantic reranking over the hybrid result set
        if cleaned and use_semantic and self.semantic_config:
            try:
                kwargs = dict(base, query_type="semantic",
                              semantic_configuration_name=self.semantic_config)
                docs = await self._attempt(client, kwargs)
                semantic_used = True
            except Exception as exc:
                last_error = exc
                log.warning("Semantic search failed on '%s': %s", index_name, exc)

        # 2) plain hybrid
        if docs is None:
            try:
                docs = await self._attempt(client, dict(base, query_type="simple"))
            except Exception as exc:
                last_error = exc
                log.warning("Hybrid search failed on '%s': %s", index_name, exc)

        # 3) lexical only — drops the vector clause in case the field name is wrong
        if docs is None:
            try:
                lexical = {kk: vv for kk, vv in base.items() if kk != "vector_queries"}
                docs = await self._attempt(client, dict(lexical, query_type="simple"))
            except Exception as exc:
                last_error = exc

        # 4) minimal — no filter, no vector, nothing that can be rejected
        if docs is None:
            try:
                docs = await self._attempt(client, {"search_text": cleaned or "*",
                                                    "top": fetch_k, "query_type": "simple"})
            except Exception as exc:
                last_error = exc

        if docs is None:
            raise last_error or RuntimeError("Azure Search returned no result set")
        if not docs:
            return []

        normalised = [n for n in (self._normalise(d, i) for i, d in enumerate(docs, 1)) if n]
        normalised.sort(
            key=lambda d: (
                d.get("reranker_score") if d.get("reranker_score") is not None else -1,
                d.get("score") if d.get("score") is not None else -1,
            ),
            reverse=True,
        )

        seen: set[str] = set()
        deduped: list[dict] = []
        for d in normalised:
            key = f"{d['url'] or d['title']}|{' '.join(d['content'].split()).lower()[:400]}"
            if key in seen:
                continue
            seen.add(key)
            d["semantic_used"] = semantic_used
            deduped.append(d)
        return deduped[:k]

    @staticmethod
    def _normalise(doc: dict, idx: int) -> dict | None:
        content = ""
        for f in CONTENT_FIELDS:
            if doc.get(f):
                content = str(doc[f]).strip()
                break
        if not content:
            return None
        title = next((str(doc[f]).strip() for f in TITLE_FIELDS if doc.get(f)),
                     f"Result {idx}")
        url = next((str(doc[f]).strip() for f in URL_FIELDS
                    if doc.get(f) and str(doc[f]).startswith("http")), "")
        return {
            "title": title,
            "url": url,
            "content": content,
            "score": doc.get("@search.score"),
            "reranker_score": doc.get("@search.reranker_score"),
        }

    # ------------------------------------------------- Retriever interface
    async def search(self, domain: str, query: str, top_k: int) -> list[str]:
        """Not a graph entry point — this retriever returns documents, not nodes.

        Kept so the class satisfies the interface; graph entry stays with the
        keyword or FAISS retriever while this supplies passage context.
        """
        return []

    async def build(self, domain: str) -> dict:
        return {"retriever": self.name, "indexed": 0,
                "note": "Document search maintains its own index; nothing to build here."}


def format_documents(docs: list[dict]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        lines = [f"Source {i}: {d['title']}"]
        if d.get("url"):
            lines.append(f"URL: {d['url']}")
        lines.append("Content:")
        lines.append(d["content"])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks).strip()
