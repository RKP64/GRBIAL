"""What else is touched if something changes.

Not simulation. Simulation needs numeric properties and propagation rules the
graph does not carry — you cannot compute revenue impact from a graph that never
recorded revenue. What the graph can answer is reachability: what connects to
this, how closely, and by which relationship. That is the question people
usually reach for simulation to answer, and it is answerable honestly.

The output is grouped by distance because distance is the signal. Something one
hop away is directly attached; something three hops away is related in a way
that may or may not matter, and the person reading needs to judge that rather
than be handed a single number that hides it.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from ..stores import get_store

log = logging.getLogger(__name__)

MAX_HOPS = 4
DEFAULT_HOPS = 2


async def impact(domain: str, entity: str, *, hops: int = DEFAULT_HOPS,
                 limit: int = 200) -> dict[str, Any]:
    """Walk outward from one entity and report what is reachable.

    Traversal is undirected on purpose. An incident affecting a gate is stored
    as Incident -> Gate, so asking what a gate change touches has to follow that
    edge backwards or the live operational state is invisible.
    """
    hops = max(1, min(MAX_HOPS, hops))
    store = get_store()

    graph = await store.export_json(domain)
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    if entity not in nodes:
        lowered = entity.lower()
        match = next((nid for nid in nodes if nid.lower() == lowered), None)
        if match is None:
            candidates = [nid for nid in nodes if lowered in nid.lower()][:5]
            return {
                "entity": entity, "found": False,
                "suggestions": candidates,
                "note": f"'{entity}' is not in {domain}."
                        + (f" Closest: {', '.join(candidates)}." if candidates else ""),
            }
        entity = match

    adjacency: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for e in graph.get("edges", []):
        s, t = e.get("source"), e.get("target")
        rel = e.get("relation") or e.get("type") or ""
        if not s or not t:
            continue
        adjacency[s].append((t, rel, "out"))
        adjacency[t].append((s, rel, "in"))

    # Breadth-first, so the first time a node is reached is by its shortest path.
    seen = {entity: 0}
    how: dict[str, tuple[str, str, str]] = {}   # node -> (via, relation, direction)
    frontier = [entity]
    for distance in range(1, hops + 1):
        nxt: list[str] = []
        for current in frontier:
            for neighbour, rel, direction in adjacency.get(current, []):
                if neighbour in seen or len(seen) >= limit:
                    continue
                seen[neighbour] = distance
                how[neighbour] = (current, rel, direction)
                nxt.append(neighbour)
        frontier = nxt
        if not frontier:
            break

    by_distance: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, int] = defaultdict(int)
    for nid, distance in seen.items():
        if nid == entity:
            continue
        via, rel, direction = how.get(nid, ("", "", ""))
        node_type = (nodes.get(nid) or {}).get("type") or "unknown"
        by_type[node_type] += 1
        by_distance[distance].append({
            "id": nid, "type": node_type,
            "via": via, "relation": rel, "direction": direction,
        })

    return {
        "entity": entity,
        "found": True,
        "type": (nodes.get(entity) or {}).get("type"),
        "hops": hops,
        "total": len(seen) - 1,
        "truncated": len(seen) >= limit,
        "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "by_distance": {str(d): sorted(v, key=lambda x: (x["type"], x["id"]))
                        for d, v in sorted(by_distance.items())},
    }


def format_impact(result: dict[str, Any], per_level: int = 12) -> str:
    """Render for a model: grouped, counted, and capped.

    Capped per level because a wide graph produces hundreds of entries at two
    hops, and a list that long stops being an answer.
    """
    if not result.get("found"):
        return result.get("note", "That entity was not found.")

    hops = result["hops"]
    lines = [f"{result['entity']} ({result.get('type') or 'unknown'}) — "
             f"{result['total']} connected entities within "
             f"{hops} hop{'s' if hops != 1 else ''}."]

    if result["by_type"]:
        lines.append("By type: " + ", ".join(f"{t} {n}" for t, n
                                             in result["by_type"].items()))

    for distance, entries in result["by_distance"].items():
        plural = "s" if str(distance) != "1" else ""
        lines.append(f"\n{distance} hop{plural} away ({len(entries)}):")
        for item in entries[:per_level]:
            arrow = "->" if item["direction"] == "out" else "<-"
            lines.append(f"  {item['id']} ({item['type']}) "
                         f"[{item['via']} {arrow} {item['relation']}]")
        if len(entries) > per_level:
            lines.append(f"  ... and {len(entries) - per_level} more")

    if result.get("truncated"):
        lines.append("\nTraversal was capped; there may be more.")

    lines.append("\nThis is what connects to the entity, not a prediction of "
                 "consequences.")
    return "\n".join(lines)
