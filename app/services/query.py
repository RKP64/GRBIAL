from __future__ import annotations

import asyncio
from typing import Any, Literal

from ..config import get_settings
from ..retrieval import (get_azure_search, get_chunk_index,
                         get_keyword_fallback, get_retriever)
from ..stores import get_store
from ..usage.recorder import attribute_to
from .llm import complete

Mode = Literal["graph", "hybrid"]
DocSource = Literal["auto", "service", "local", "none"]

GRAPH_SYSTEM = (
    "You answer strictly from the provided knowledge-graph context. "
    "If the context does not contain the answer, say so plainly and do not guess. "
    "Be concise and specific."
)

HYBRID_SYSTEM = (
    "You answer from two sources: a knowledge graph of verified entities and "
    "relationships, and retrieved document passages.\n"
    "Prefer the graph for facts about entities and how they connect; use the "
    "documents for detail, wording, and anything the graph does not cover.\n"
    "If the two disagree, say so rather than silently choosing one. "
    "If neither contains the answer, say so plainly. Never invent facts. "
    "When a document supplies the answer, cite it as [Source N]."
)


def _render_graph_context(sub: dict[str, Any]) -> str:
    by_id = {n["id"]: n for n in sub["nodes"]}
    lines: list[str] = []
    for node_id in sub["entry_points"]:
        node = by_id.get(node_id)
        if not node:
            continue
        lines.append(f"Node: {node_id} (type: {node.get('type','?')})")
        if node.get("evidence"):
            lines.append(f"  evidence: {node['evidence']}")
        for e in sub["edges"]:
            if e["source"] == node_id:
                lines.append(f"  -> {e['relation']} -> {e['target']} "
                             f"({by_id.get(e['target'],{}).get('type','')})")
            elif e["target"] == node_id:
                lines.append(f"  <- {e['relation']} <- {e['source']} "
                             f"({by_id.get(e['source'],{}).get('type','')})")
        lines.append("")
    return "\n".join(lines).strip()


async def _graph_part(domain: str, question: str, top_k: int, hops: int,
                      max_neighbours: int) -> dict[str, Any]:
    store = get_store()
    retriever = get_retriever()
    entry = await retriever.search(domain, question, top_k)
    used = retriever.name
    if not entry and retriever.name != "text":
        entry = await get_keyword_fallback().search(domain, question, top_k)
        used = "text match"
    if not entry:
        return {"retriever": used, "entry_points": [],
                "subgraph": {"nodes": [], "edges": []}, "context": ""}
    sub = await store.neighbourhood(domain, entry, hops=hops, max_neighbours=max_neighbours)
    return {
        "retriever": used,
        "entry_points": sub["entry_points"],
        "subgraph": {"nodes": sub["nodes"], "edges": sub["edges"]},
        "context": _render_graph_context(sub),
    }


def _format_documents(docs: list[dict[str, Any]]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        lines = [f"Source {i}: {d['title']}"]
        if d.get("url"):
            lines.append(f"URL: {d['url']}")
        lines.append("Content:")
        lines.append(d["content"])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks).strip()


async def _documents_part(domain: str, question: str, k: int, index_name: str | None,
                          filter_expression: str | None,
                          source: DocSource = "auto") -> dict[str, Any]:
    """Retrieve passages.

    Preference order under "auto": the connected search service if there is one,
    otherwise the passages captured locally during ingestion. Naming a source
    explicitly overrides that, which is how you compare the two.
    """
    if source == "none":
        return {"available": False, "documents": [], "context": "",
                "source": "none", "note": None}

    searcher = get_azure_search()
    use_service = source == "service" or (source == "auto" and searcher is not None)

    if use_service:
        if searcher is None:
            return {"available": False, "documents": [], "context": "", "source": "service",
                    "note": "The document search service is not connected."}
        try:
            docs = await searcher.search_documents(
                question, index_name=index_name, k=k, filter_expression=filter_expression
            )
            return {"available": True, "documents": docs,
                    "context": _format_documents(docs), "source": "service", "note": None}
        except Exception as exc:
            return {"available": True, "documents": [], "context": "", "source": "service",
                    "note": f"Document search failed: {exc}"}

    store = get_chunk_index()
    if not store.exists(domain):
        return {"available": False, "documents": [], "context": "", "source": "local",
                "note": "No source passages have been kept for this domain yet."}
    try:
        docs = await store.search(domain, question, k=k)
        return {"available": True, "documents": docs,
                "context": _format_documents(docs), "source": "local", "note": None}
    except Exception as exc:
        return {"available": True, "documents": [], "context": "", "source": "local",
                "note": f"Passage search failed: {exc}"}


async def answer(
    domain: str,
    question: str,
    *,
    mode: Mode = "graph",
    top_k: int = 5,
    hops: int = 1,
    max_neighbours: int = 25,
    doc_k: int = 5,
    index_name: str | None = None,
    filter_expression: str | None = None,
    doc_source: DocSource = "auto",
    synthesise: bool = True,
) -> dict[str, Any]:
    """Answer from the graph alone, or from the graph and documents together.

    In hybrid mode the two retrievals run concurrently — the document search does
    not wait on graph traversal.
    """
    if mode == "hybrid":
        graph, docs = await asyncio.gather(
            _graph_part(domain, question, top_k, hops, max_neighbours),
            _documents_part(domain, question, doc_k, index_name,
                            filter_expression, doc_source),
        )
    else:
        graph = await _graph_part(domain, question, top_k, hops, max_neighbours)
        docs = {"available": False, "documents": [], "context": "",
                "source": "none", "note": None}

    result: dict[str, Any] = {
        "question": question,
        "mode": mode,
        "retriever": graph["retriever"],
        "entry_points": graph["entry_points"],
        "subgraph": graph["subgraph"],
        "context": graph["context"],
        "documents": docs["documents"],
        "document_context": docs["context"],
        "document_source": docs.get("source", "none"),
        "search_note": docs["note"],
        "answer": None,
    }

    has_graph = bool(graph["context"])
    has_docs = bool(docs["context"])
    if not has_graph and not has_docs:
        result["answer"] = (
            "Nothing in the graph matches that question yet."
            if mode == "graph"
            else "Neither the graph nor the search index returned anything for that question."
        )
        return result

    if not synthesise:
        return result

    if mode == "hybrid":
        parts = []
        if has_graph:
            parts.append(f"KNOWLEDGE GRAPH CONTEXT:\n{graph['context']}")
        if has_docs:
            parts.append(f"RETRIEVED DOCUMENTS:\n{docs['context']}")
        user = "\n\n".join(parts) + f"\n\nQuestion: {question}"
        with attribute_to("answering", domain=domain):
            result["answer"] = await complete(HYBRID_SYSTEM, user, temperature=0.2)
    else:
        with attribute_to("answering", domain=domain):
            result["answer"] = await complete(
                GRAPH_SYSTEM, f"Context:\n{graph['context']}\n\nQuestion: {question}",
                temperature=0.2,
            )
    return result
