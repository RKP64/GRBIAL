from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

import networkx as nx

from ..ontology.models import EdgeIn, NodeIn
from .base import GraphStore


class NetworkXStore(GraphStore):
    """In-process graph with GraphML persistence.

    GraphML is the portable system of record: the same artefact feeds Gephi,
    analytics, synthetic-data generation, and bulk load into Gremlin.
    """

    name = "local"

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._graphs: dict[str, nx.MultiDiGraph] = {}
        self._lock = asyncio.Lock()

    def _path(self, domain: str) -> Path:
        return self.data_dir / f"{domain}.graphml"

    def _graph(self, domain: str) -> nx.MultiDiGraph:
        if domain not in self._graphs:
            path = self._path(domain)
            if path.exists():
                self._graphs[domain] = self._read(path)
            else:
                self._graphs[domain] = nx.MultiDiGraph()
        return self._graphs[domain]

    @staticmethod
    def _read(path: Path) -> nx.MultiDiGraph:
        """Load a GraphML file, restoring relation names as edge keys.

        NetworkX does not round-trip MultiDiGraph edge keys — on read they come
        back as 0, 1, 2… so the relation name is recovered from the `label`
        attribute. Without this, every reload renames all relations and
        re-ingestion duplicates edges instead of updating them.
        """
        raw = nx.read_graphml(path)
        out = nx.MultiDiGraph()
        out.add_nodes_from(raw.nodes(data=True))
        edges = raw.edges(keys=True, data=True) if raw.is_multigraph() else (
            (u, v, None, d) for u, v, d in raw.edges(data=True)
        )
        for u, v, key, data in edges:
            relation = str(data.get("label") or key or "related_to")
            attrs = dict(data)
            attrs["label"] = relation
            if not out.has_edge(u, v, key=relation):
                out.add_edge(u, v, key=relation, **attrs)
        return out

    async def upsert(self, domain: str, nodes: list[NodeIn], edges: list[EdgeIn]) -> tuple[int, int]:
        async with self._lock:
            g = self._graph(domain)
            added_n = added_e = 0
            for n in nodes:
                if g.has_node(n.id):
                    ev = str(n.metadata.get("evidence", ""))[:280]
                    if ev and not g.nodes[n.id].get("evidence"):
                        g.nodes[n.id]["evidence"] = ev
                    g.nodes[n.id]["mentions"] = int(g.nodes[n.id].get("mentions", 1)) + 1
                else:
                    g.add_node(
                        n.id,
                        type=n.type,
                        label=n.id[:40],
                        evidence=str(n.metadata.get("evidence", ""))[:280],
                        mentions=1,
                    )
                    added_n += 1
            for e in edges:
                if g.has_node(e.source) and g.has_node(e.target):
                    if g.has_edge(e.source, e.target, key=e.type):
                        data = g[e.source][e.target][e.type]
                        data["weight"] = int(data.get("weight", 1)) + 1
                    else:
                        g.add_edge(e.source, e.target, key=e.type, label=e.type, weight=1)
                        added_e += 1
            return added_n, added_e

    async def persist(self, domain: str) -> Path:
        """Write the working copy. Edge keys stay as relation names so a reload
        round-trips correctly."""
        async with self._lock:
            g = self._graph(domain)
            path = self._path(domain)
            nx.write_graphml(g, path)
            return path

    async def export_bytes(self, domain: str) -> bytes:
        """Gephi-compatible export.

        Strict GraphML readers (graphology, Gephi Lite) require globally unique
        edge ids, but those ids must never become the persisted relation keys —
        so the renaming happens only on this throwaway copy, and the relation
        name is preserved in the `label` attribute.
        """
        async with self._lock:
            g = self._graph(domain)
            out = nx.MultiDiGraph()
            out.add_nodes_from(g.nodes(data=True))
            for i, (u, v, k, d) in enumerate(g.edges(keys=True, data=True)):
                attrs = dict(d)
                attrs["label"] = attrs.get("label") or str(k)
                out.add_edge(u, v, key=f"e{i}", **attrs)
            buf = io.BytesIO()
            nx.write_graphml(out, buf)
            return buf.getvalue()

    async def stats(self, domain: str) -> dict[str, Any]:
        g = self._graph(domain)
        types: dict[str, int] = {}
        for _, d in g.nodes(data=True):
            types[d.get("type", "?")] = types.get(d.get("type", "?"), 0) + 1
        rels: dict[str, int] = {}
        for _, _, k in g.edges(keys=True):
            rels[k] = rels.get(k, 0) + 1
        return {
            "domain": domain,
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "node_types": dict(sorted(types.items(), key=lambda kv: -kv[1])),
            "relationships": dict(sorted(rels.items(), key=lambda kv: -kv[1])),
        }

    async def keyword_search(self, domain: str, terms: list[str], limit: int) -> list[str]:
        g = self._graph(domain)
        terms = [t for t in terms if len(t) > 2]
        if not terms:
            return []
        scored: list[tuple[int, str]] = []
        for node_id, d in g.nodes(data=True):
            hay = f"{node_id} {d.get('type','')} {d.get('evidence','')}".lower()
            score = sum(1 for t in terms if t in hay)
            if score:
                scored.append((score * 100 + int(d.get("mentions", 1)), node_id))
        scored.sort(reverse=True)
        return [nid for _, nid in scored[:limit]]

    async def neighbourhood(self, domain: str, node_ids: list[str], hops: int, max_neighbours: int) -> dict[str, Any]:
        g = self._graph(domain)
        keep: set[str] = {n for n in node_ids if g.has_node(n)}
        frontier = set(keep)
        for _ in range(max(0, hops)):
            nxt: set[str] = set()
            for n in frontier:
                nbrs = list(g.successors(n))[:max_neighbours] + list(g.predecessors(n))[:max_neighbours]
                nxt.update(nbrs)
            nxt -= keep
            keep |= nxt
            frontier = nxt
            if not frontier:
                break
        sub = g.subgraph(keep)
        return {
            "entry_points": [n for n in node_ids if g.has_node(n)],
            "nodes": [
                {"id": n, "type": d.get("type", ""), "evidence": d.get("evidence", ""),
                 "mentions": int(d.get("mentions", 1))}
                for n, d in sub.nodes(data=True)
            ],
            "edges": [
                {"source": u, "relation": k, "target": v, "weight": int(d.get("weight", 1))}
                for u, v, k, d in sub.edges(keys=True, data=True)
            ],
        }

    async def node_texts(self, domain: str, max_relations: int = 12) -> list[tuple[str, str]]:
        """Describe each node by what it is and how it connects.

        "Section 194C. A Provision. applies to TDS; cited by ASMT-10." embeds far
        better than the bare id, because the surrounding entities carry the
        vocabulary a question is likely to use.
        """
        g = self._graph(domain)
        out: list[tuple[str, str]] = []
        for node_id, data in g.nodes(data=True):
            parts = [f"{node_id}."]
            node_type = data.get("type", "")
            if node_type:
                parts.append(f"A {node_type}.")
            relations: list[str] = []
            for _, target, rel in list(g.out_edges(node_id, keys=True))[:max_relations]:
                relations.append(f"{str(rel).replace('_', ' ')} {target}")
            remaining = max_relations - len(relations)
            if remaining > 0:
                for source, _, rel in list(g.in_edges(node_id, keys=True))[:remaining]:
                    relations.append(f"{source} {str(rel).replace('_', ' ')} this")
            if relations:
                parts.append("; ".join(relations) + ".")
            evidence = data.get("evidence", "")
            if evidence:
                parts.append(str(evidence))
            out.append((node_id, " ".join(parts).strip()))
        return out

    async def export_graphml(self, domain: str) -> bytes:
        await self.persist(domain)
        return await self.export_bytes(domain)

    async def export_json(self, domain: str) -> dict[str, Any]:
        """Node-link JSON — the shape D3, Cytoscape and most graph libraries read."""
        g = self._graph(domain)
        return {
            "domain": domain,
            "directed": True,
            "multigraph": True,
            "nodes": [
                {"id": n, "type": d.get("type", ""), "label": d.get("label", n),
                 "evidence": d.get("evidence", ""), "mentions": int(d.get("mentions", 1)),
                 "degree": g.degree(n)}
                for n, d in g.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, "relation": k, "weight": int(d.get("weight", 1))}
                for u, v, k, d in g.edges(keys=True, data=True)
            ],
        }

    async def sample(self, domain: str, limit: int, node_type: str | None = None) -> dict[str, Any]:
        """Highest-degree nodes plus the edges among them.

        Picking by degree keeps the sample connected; picking arbitrarily
        produces a screen of unlinked dots that tells the viewer nothing.
        """
        g = self._graph(domain)
        candidates = [n for n, d in g.nodes(data=True)
                      if not node_type or d.get("type") == node_type]
        top = sorted(candidates, key=lambda n: g.degree(n), reverse=True)[:limit]
        keep = set(top)
        sub = g.subgraph(keep)
        return {
            "truncated": len(candidates) > len(top),
            "total_nodes": g.number_of_nodes(),
            "nodes": [
                {"id": n, "type": d.get("type", ""), "evidence": d.get("evidence", ""),
                 "mentions": int(d.get("mentions", 1)), "degree": g.degree(n)}
                for n, d in sub.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, "relation": k, "weight": int(d.get("weight", 1))}
                for u, v, k, d in sub.edges(keys=True, data=True)
            ],
        }
