from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from ..stores import get_store
from .base import Retriever
from .keyword import KeywordRetriever


@lru_cache
def get_retriever() -> Retriever:
    s = get_settings()
    store = get_store()
    if s.retriever == "faiss":
        from .faiss_retriever import FaissRetriever

        return FaissRetriever(store, s.data_dir)
    return KeywordRetriever(store)


@lru_cache
def get_azure_search() -> "Retriever | None":
    """Document retriever used for hybrid answering. Independent of the graph
    entry-point retriever above — the two are combined in the query service."""
    if not get_settings().azure_search_configured:
        return None
    try:
        from .azure_search import AzureAISearchRetriever

        return AzureAISearchRetriever()
    except Exception:  # missing SDK or bad credentials
        return None


@lru_cache
def get_keyword_fallback() -> Retriever:
    return KeywordRetriever(get_store())


@lru_cache
def get_chunk_index():
    """Local passage store. Always available — no external service required."""
    from .chunk_index import ChunkIndex

    return ChunkIndex(get_settings().data_dir)
