from __future__ import annotations

import asyncio
from typing import Any

from ..ontology.models import EdgeIn, NodeIn
from .base import GraphStore

try:
    from gremlin_python.driver import client as gremlin_client
    from gremlin_python.driver import serializer
    GREMLIN_AVAILABLE = True
except ImportError:  # pragma: no cover
    GREMLIN_AVAILABLE = False


class GremlinStore(GraphStore):
    """Azure Cosmos DB for Apache Gremlin.

    Cosmos specifics that are easy to get wrong and are handled here:
      * username is the resource path /dbs/<db>/colls/<graph>
      * GraphSON v2 only — v3 fails to negotiate
      * all queries parameterised via bindings (injection-safe, quote-safe)
      * fold/coalesce upserts so re-ingestion never duplicates
    """

    name = "cloud"

    def __init__(self, endpoint: str, key: str, database: str, collection: str) -> None:
        if not GREMLIN_AVAILABLE:
            raise RuntimeError("gremlinpython is not installed")
        if not endpoint or not key:
            raise RuntimeError("COSMOS_GREMLIN_ENDPOINT and COSMOS_KEY are required")
        self._client = gremlin_client.Client(
            url=endpoint,
            traversal_source="g",
            username=f"/dbs/{database}/colls/{collection}",
            password=key,
            message_serializer=serializer.GraphSONSerializersV2d0(),
        )

    async def _submit(self, query: str, bindings: dict[str, Any] | None = None) -> list[Any]:
        def run() -> list[Any]:
            return self._client.submit(query, bindings or {}).all().result()

        return await asyncio.to_thread(run)

    async def upsert(self, domain: str, nodes: list[NodeIn], edges: list[EdgeIn]) -> tuple[int, int]:
        added_n = added_e = 0
        for n in nodes:
            await self._submit(
                "g.V(vid).fold().coalesce(unfold(), "
                "addV(vlabel).property('id', vid).property('pk', vpk))"
                ".property('domain', vdom).property('evidence', vev)",
                {
                    "vid": n.id,
                    "vlabel": n.type,
                    "vpk": domain,
                    "vdom": domain,
                    "vev": str(n.metadata.get("evidence", ""))[:256],
                },
            )
            added_n += 1
        for e in edges:
            await self._submit(
                "g.V(src).as('a').V(dst)"
                ".coalesce(inE(rel).where(outV().as('a')), addE(rel).from('a'))",
                {"src": e.source, "dst": e.target, "rel": e.type},
            )
            added_e += 1
        return added_n, added_e

    async def stats(self, domain: str) -> dict[str, Any]:
        nodes = await self._submit("g.V().has('domain', d).count()", {"d": domain})
        edges = await self._submit("g.V().has('domain', d).outE().count()", {"d": domain})
        by_type = await self._submit(
            "g.V().has('domain', d).groupCount().by(label())", {"d": domain}
        )
        return {
            "domain": domain,
            "nodes": nodes[0] if nodes else 0,
            "edges": edges[0] if edges else 0,
            "node_types": by_type[0] if by_type else {},
            "relationships": {},
        }

    async def keyword_search(self, domain: str, terms: list[str], limit: int) -> list[str]:
        found: list[str] = []
        for term in [t for t in terms if len(t) > 2][:3]:
            rows = await self._submit(
                "g.V().has('domain', d).has('id', containing(t)).limit(k).id()",
                {"d": domain, "t": term, "k": limit},
            )
            found.extend(str(r) for r in rows)
        seen: set[str] = set()
        return [x for x in found if not (x in seen or seen.add(x))][:limit]

    async def neighbourhood(self, domain: str, node_ids: list[str], hops: int, max_neighbours: int) -> dict[str, Any]:
        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        for nid in node_ids:
            rows = await self._submit(
                "g.V(vid).project('id','label','out','in')"
                ".by(id()).by(label())"
                ".by(outE().limit(k).project('rel','to').by(label()).by(inV().id()).fold())"
                ".by(inE().limit(k).project('rel','from').by(label()).by(outV().id()).fold())",
                {"vid": nid, "k": max_neighbours},
            )
            for r in rows:
                nodes[r["id"]] = {"id": r["id"], "type": r["label"], "evidence": "", "mentions": 1}
                for o in r.get("out", []):
                    edges.append({"source": r["id"], "relation": o["rel"], "target": o["to"], "weight": 1})
                    nodes.setdefault(o["to"], {"id": o["to"], "type": "", "evidence": "", "mentions": 1})
                for i in r.get("in", []):
                    edges.append({"source": i["from"], "relation": i["rel"], "target": r["id"], "weight": 1})
                    nodes.setdefault(i["from"], {"id": i["from"], "type": "", "evidence": "", "mentions": 1})
        return {"entry_points": node_ids, "nodes": list(nodes.values()), "edges": edges}

    async def node_texts(self, domain: str) -> list[tuple[str, str]]:
        rows = await self._submit(
            "g.V().has('domain', d).project('id','label').by(id()).by(label())", {"d": domain}
        )
        return [(r["id"], f"{r['id']}. Type: {r['label']}.") for r in rows]

    async def export_graphml(self, domain: str) -> bytes:
        raise NotImplementedError("Export from Gremlin is not supported; use the NetworkX artefact")

    async def export_json(self, domain: str) -> dict[str, Any]:
        nodes = await self._submit(
            "g.V().has('domain', d).project('id','label').by(id()).by(label())", {"d": domain})
        edges = await self._submit(
            "g.V().has('domain', d).outE().project('s','r','t')"
            ".by(outV().id()).by(label()).by(inV().id())", {"d": domain})
        return {
            "domain": domain, "directed": True, "multigraph": True,
            "nodes": [{"id": n["id"], "type": n["label"], "label": n["id"],
                       "evidence": "", "mentions": 1, "degree": 0} for n in nodes],
            "edges": [{"source": e["s"], "target": e["t"], "relation": e["r"],
                       "weight": 1} for e in edges],
        }

    async def sample(self, domain: str, limit: int, node_type: str | None = None) -> dict[str, Any]:
        q = "g.V().has('domain', d)"
        bindings = {"d": domain, "k": limit}
        if node_type:
            q += ".hasLabel(lbl)"
            bindings["lbl"] = node_type
        nodes = await self._submit(
            q + ".limit(k).project('id','label').by(id()).by(label())", bindings)
        ids = [n["id"] for n in nodes]
        edges = []
        for nid in ids:
            rows = await self._submit(
                "g.V(vid).outE().limit(10).project('s','r','t')"
                ".by(outV().id()).by(label()).by(inV().id())", {"vid": nid})
            edges.extend({"source": r["s"], "target": r["t"], "relation": r["r"],
                          "weight": 1} for r in rows if r["t"] in set(ids))
        return {
            "truncated": len(ids) >= limit, "total_nodes": len(ids),
            "nodes": [{"id": n["id"], "type": n["label"], "evidence": "",
                       "mentions": 1, "degree": 0} for n in nodes],
            "edges": edges,
        }

    async def health(self) -> dict[str, Any]:
        try:
            await self._submit("g.V().limit(1)")
            return {"mode": "cloud", "status": "ok"}
        except Exception as exc:
            return {"mode": "cloud", "status": "error", "detail": str(exc)}
