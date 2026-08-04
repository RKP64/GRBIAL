"""Tools an agent may call.

Each tool is a thin wrapper over a capability the platform already exposes, with
two additions that matter for agent use: a schema the model can read, and a
result that is bounded in size. An unbounded tool result is the usual cause of a
loop that runs out of context halfway through.

Tools are deliberately narrow. A model choosing between four well-described
tools behaves far better than one choosing between fifteen overlapping ones.
"""
from __future__ import annotations

import logging
from typing import Any

from ..retrieval import get_azure_search, get_chunk_index, get_keyword_fallback, get_retriever
from ..stores import get_store
from ..verification import verify_answer

log = logging.getLogger(__name__)

MAX_CHARS = 4000


def _truncate(text: str, limit: int = MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… truncated, {len(text) - limit} more characters"


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_graph",
        "description": (
            "Find entities in the knowledge graph and see how they connect. Use "
            "this for questions about specific things and the relationships "
            "between them. Returns matched entities with their neighbours."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "What to look for, in natural language."},
                "hops": {"type": "integer", "minimum": 1, "maximum": 2, "default": 1,
                         "description": "1 for direct connections, 2 to follow one "
                                        "step further. Use 2 only when the question "
                                        "needs two facts joined."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "expand_entity",
        "description": (
            "Show every recorded relationship for one named entity. Use this after "
            "search_graph when you need the full picture for a specific entity "
            "rather than a general search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string",
                           "description": "The exact entity name as it appears in the graph."},
            },
            "required": ["entity"],
        },
    },
    {
        "name": "search_documents",
        "description": (
            "Search the source documents for passages. Use this for wording, "
            "detail, procedure, or anything the graph does not model as a "
            "relationship."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "describe_schema",
        "description": (
            "List the entity types and relationships this graph holds. Use this "
            "first when unsure whether the graph can answer a question at all."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


async def available_tools(include_external: bool = True) -> list[dict[str, Any]]:
    """Built-in tools, plus anything reachable tool servers offer.

    External tools are added rather than substituted: the graph tools are what
    make this platform worth pointing an agent at, and a badly behaved external
    server should not be able to shadow them.
    """
    tools = list(TOOL_SCHEMAS)
    if not include_external:
        return tools
    try:
        from ..mcp.registry import get_server_registry

        builtin = {t["name"] for t in tools}
        external = await get_server_registry().tools()
        tools += [t.as_schema() for t in external if t.qualified not in builtin]
    except Exception as exc:
        log.warning("External tools could not be listed: %s", exc)
    return tools


async def run_tool(name: str, arguments: dict[str, Any], *, domain: str) -> str:
    """Execute a tool and return a compact text result for the model."""
    builtin = {t["name"] for t in TOOL_SCHEMAS}
    if name not in builtin and "__" in name:
        from ..mcp.registry import get_server_registry

        return _truncate(await get_server_registry().call(name, arguments))
    try:
        if name == "search_graph":
            return await _search_graph(domain, str(arguments.get("query", "")),
                                       int(arguments.get("hops", 1) or 1))
        if name == "expand_entity":
            return await _expand_entity(domain, str(arguments.get("entity", "")))
        if name == "search_documents":
            return await _search_documents(domain, str(arguments.get("query", "")))
        if name == "describe_schema":
            return await _describe_schema(domain)
        return f"There is no tool called '{name}'."
    except Exception as exc:
        log.warning("Tool %s failed: %s", name, exc)
        # Returned rather than raised: the model can recover by trying something
        # else, whereas an exception ends the conversation.
        return f"That tool could not run: {exc}"


async def _search_graph(domain: str, query: str, hops: int) -> str:
    if not query.strip():
        return "No query was given."
    store = get_store()
    retriever = get_retriever()
    entry = await retriever.search(domain, query, 6)
    if not entry:
        entry = await get_keyword_fallback().search(domain, query, 6)
    if not entry:
        return ("Nothing in the graph matches that. The entity may not have been "
                "extracted, or may be named differently.")
    sub = await store.neighbourhood(domain, entry, hops=min(hops, 2), max_neighbours=20)
    types = {n["id"]: n.get("type", "") for n in sub["nodes"]}
    lines = [f"Matched {len(sub['entry_points'])} entities."]
    for node_id in sub["entry_points"]:
        lines.append(f"\n{node_id} ({types.get(node_id, '')})")
        related = [e for e in sub["edges"] if node_id in (e["source"], e["target"])]
        if not related:
            lines.append("  no recorded relationships")
        for e in related[:15]:
            if e["source"] == node_id:
                lines.append(f"  {node_id} — {e['relation'].replace('_',' ')} → {e['target']}")
            else:
                lines.append(f"  {e['source']} — {e['relation'].replace('_',' ')} → {node_id}")
    return _truncate("\n".join(lines))


async def _expand_entity(domain: str, entity: str) -> str:
    if not entity.strip():
        return "No entity was given."
    store = get_store()
    graph = await store.export_json(domain)
    ids = {n["id"] for n in graph["nodes"]}
    if entity not in ids:
        close = [n for n in ids if entity.lower() in n.lower()
                 or n.lower() in entity.lower()][:5]
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        return f"'{entity}' is not in the graph.{hint}"
    types = {n["id"]: n.get("type", "") for n in graph["nodes"]}
    out = [f"{entity} ({types.get(entity, '')})"]
    for e in graph["edges"]:
        if e["source"] == entity:
            out.append(f"  {entity} — {e['relation'].replace('_',' ')} → "
                       f"{e['target']} ({types.get(e['target'],'')})")
        elif e["target"] == entity:
            out.append(f"  {e['source']} ({types.get(e['source'],'')}) — "
                       f"{e['relation'].replace('_',' ')} → {entity}")
    if len(out) == 1:
        out.append("  no recorded relationships")
    return _truncate("\n".join(out))


async def _search_documents(domain: str, query: str) -> str:
    if not query.strip():
        return "No query was given."
    searcher = get_azure_search()
    docs: list[dict] = []
    if searcher is not None:
        try:
            docs = await searcher.search_documents(query, k=4)
        except Exception as exc:
            log.warning("Document service search failed: %s", exc)
    if not docs:
        store = get_chunk_index()
        if store.exists(domain):
            docs = await store.search(domain, query, k=4)
    if not docs:
        return "No passages were found. There may be no documents kept for this domain."
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(f"[{i}] {d['title']}\n{d['content'][:900]}")
    return _truncate("\n\n".join(blocks))


async def _describe_schema(domain: str) -> str:
    from ..ontology import get_ontology

    ontology = get_ontology(domain)
    stats = await get_store().stats(domain)
    lines = [f"Domain: {ontology.name}. {ontology.description}",
             f"Graph holds {stats['nodes']} entities and {stats['edges']} relationships.",
             "", "Entity types:"]
    counts = stats.get("node_types", {})
    for name in ontology.entity_types:
        lines.append(f"  {name} ({counts.get(name, 0)} recorded)")
    lines.append("")
    lines.append("Relationships:")
    for s, r, t in sorted(ontology.allowed_triples):
        lines.append(f"  {s} — {r.replace('_',' ')} → {t}")
    return _truncate("\n".join(lines))
