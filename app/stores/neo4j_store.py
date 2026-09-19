from __future__ import annotations

import re
from typing import Any

from ..ontology.models import EdgeIn, NodeIn
from .base import GraphStore

try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:  # pragma: no cover
    NEO4J_AVAILABLE = False


_REL_SAFE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


def _rel_type(relation: str) -> str:
    """Return a Cypher-safe relationship type.

    Neo4j will not accept a relationship type as a query parameter, so the
    value has to be interpolated. Anything that is not a plain identifier is
    rejected and stored under RELATED_TO with the original text kept as a
    property, so a malformed relation name can never become Cypher.
    """
    candidate = (relation or "").strip().upper().replace(" ", "_").replace("-", "_")
    return candidate if _REL_SAFE.match(candidate) else "RELATED_TO"


class Neo4jStore(GraphStore):
    """Neo4j via the Bolt protocol.

    Deployment-agnostic: the same code talks to Neo4j running on an Azure
    virtual machine inside the VNet (bolt://) and to Neo4j AuraDB (neo4j+s://).
    Only the URI changes.

    Design notes:
      * every node carries a single :Entity label plus a `type` property, so
        the ontology owns the typing rather than the database schema
      * (domain, id) is the uniqueness key — re-ingestion merges rather than
        duplicates, matching the behaviour of the other stores
      * relationship types are interpolated through _rel_type(), never
        through string formatting of raw model output
    """

    name = "neo4j"

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        if not NEO4J_AVAILABLE:
            raise RuntimeError("neo4j driver is not installed (pip install neo4j)")
        if not uri:
            raise RuntimeError("NEO4J_URI is required")
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._database = database or "neo4j"
        self._prepared = False

    async def _run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, **params)
            return [record.data() async for record in result]

    async def _prepare(self) -> None:
        """Create the constraint and indexes once per process.

        Without the composite constraint, MERGE on (domain, id) does a full
        scan and ingestion slows to a crawl past a few thousand nodes.
        """
        if self._prepared:
            return
        await self._run(
            "CREATE CONSTRAINT entity_domain_id IF NOT EXISTS "
            "FOR (n:Entity) REQUIRE (n.domain, n.id) IS UNIQUE"
        )
        await self._run(
            "CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type)"
        )
        await self._run(
            "CREATE INDEX entity_id_lower IF NOT EXISTS FOR (n:Entity) ON (n.id_lower)"
        )
        self._prepared = True

    # ---- writes ---------------------------------------------------------

    async def upsert(self, domain: str, nodes: list[NodeIn], edges: list[EdgeIn]) -> tuple[int, int]:
        await self._prepare()
        added_n = added_e = 0

        if nodes:
            payload = [
                {
                    "id": n.id,
                    "id_lower": n.id.lower(),
                    "type": n.type,
                    "evidence": str(n.metadata.get("evidence", ""))[:256],
                }
                for n in nodes
                if n.id
            ]
            rows = await self._run(
                "UNWIND $rows AS row "
                "MERGE (n:Entity {domain: $domain, id: row.id}) "
                "ON CREATE SET n.created = timestamp(), n.mentions = 1 "
                "ON MATCH SET n.mentions = coalesce(n.mentions, 1) + 1 "
                "SET n.type = CASE WHEN row.type <> '' THEN row.type ELSE n.type END, "
                "    n.id_lower = row.id_lower, "
                "    n.evidence = CASE WHEN row.evidence <> '' THEN row.evidence "
                "                      ELSE n.evidence END "
                "RETURN count(n) AS written",
                rows=payload,
                domain=domain,
            )
            added_n = rows[0]["written"] if rows else 0

        # Relationship types cannot be parameterised, so edges are grouped by
        # sanitised type and each group written in one UNWIND.
        grouped: dict[str, list[dict[str, str]]] = {}
        for e in edges:
            if not (e.source and e.target):
                continue
            grouped.setdefault(_rel_type(e.type), []).append(
                {"source": e.source, "target": e.target, "relation": e.type}
            )

        for rel_type, rows in grouped.items():
            result = await self._run(
                "UNWIND $rows AS row "
                "MATCH (a:Entity {domain: $domain, id: row.source}) "
                "MATCH (b:Entity {domain: $domain, id: row.target}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                "ON CREATE SET r.weight = 1, r.relation = row.relation "
                "ON MATCH SET r.weight = coalesce(r.weight, 1) + 1 "
                "RETURN count(r) AS written",
                rows=rows,
                domain=domain,
            )
            added_e += result[0]["written"] if result else 0

        return added_n, added_e

    # ---- reads ----------------------------------------------------------

    async def stats(self, domain: str) -> dict[str, Any]:
        await self._prepare()
        counts = await self._run(
            "MATCH (n:Entity {domain: $domain}) "
            "OPTIONAL MATCH (n)-[r]->(:Entity {domain: $domain}) "
            "RETURN count(DISTINCT n) AS nodes, count(r) AS edges",
            domain=domain,
        )
        by_type = await self._run(
            "MATCH (n:Entity {domain: $domain}) "
            "RETURN coalesce(n.type, '') AS type, count(*) AS c",
            domain=domain,
        )
        by_rel = await self._run(
            "MATCH (:Entity {domain: $domain})-[r]->(:Entity {domain: $domain}) "
            "RETURN type(r) AS rel, count(*) AS c",
            domain=domain,
        )
        head = counts[0] if counts else {"nodes": 0, "edges": 0}
        return {
            "domain": domain,
            "nodes": head["nodes"],
            "edges": head["edges"],
            "node_types": {r["type"]: r["c"] for r in by_type},
            "relationships": {r["rel"]: r["c"] for r in by_rel},
        }

    async def keyword_search(self, domain: str, terms: list[str], limit: int) -> list[str]:
        await self._prepare()
        useful = [t.lower() for t in terms if len(t) > 2][:3]
        if not useful:
            return []
        rows = await self._run(
            "UNWIND $terms AS term "
            "MATCH (n:Entity {domain: $domain}) "
            "WHERE n.id_lower CONTAINS term "
            "RETURN DISTINCT n.id AS id LIMIT $limit",
            terms=useful,
            domain=domain,
            limit=limit,
        )
        return [r["id"] for r in rows]

    async def neighbourhood(
        self, domain: str, node_ids: list[str], hops: int, max_neighbours: int
    ) -> dict[str, Any]:
        await self._prepare()
        if not node_ids:
            return {"entry_points": [], "nodes": [], "edges": []}
        hops = max(1, min(int(hops or 1), 4))
        rows = await self._run(
            "MATCH (start:Entity {domain: $domain}) WHERE start.id IN $ids "
            f"CALL {{ WITH start "
            f"  MATCH path = (start)-[*1..{hops}]-(m:Entity {{domain: $domain}}) "
            f"  RETURN path LIMIT $cap }} "
            "WITH nodes(path) AS ns, relationships(path) AS rs "
            "UNWIND ns AS n WITH collect(DISTINCT n) AS allns, rs "
            "UNWIND rs AS r WITH allns, collect(DISTINCT r) AS allrs "
            "RETURN "
            "  [n IN allns | {id: n.id, type: coalesce(n.type,''), "
            "                 evidence: coalesce(n.evidence,''), "
            "                 mentions: coalesce(n.mentions,1)}] AS nodes, "
            "  [r IN allrs | {source: startNode(r).id, relation: type(r), "
            "                 target: endNode(r).id, "
            "                 weight: coalesce(r.weight,1)}] AS edges",
            domain=domain,
            ids=node_ids,
            cap=max_neighbours * max(1, len(node_ids)),
        )
        if not rows:
            return {"entry_points": node_ids, "nodes": [], "edges": []}
        return {
            "entry_points": node_ids,
            "nodes": rows[0].get("nodes") or [],
            "edges": rows[0].get("edges") or [],
        }

    async def node_texts(self, domain: str) -> list[tuple[str, str]]:
        """Embeddable text per node.

        The relationship summary is included deliberately: an entity mentioned
        once has a thin name but a meaningful set of connections, and that is
        what makes it findable by meaning rather than by string match.
        """
        await self._prepare()
        rows = await self._run(
            "MATCH (n:Entity {domain: $domain}) "
            "OPTIONAL MATCH (n)-[r]->(m:Entity {domain: $domain}) "
            "WITH n, collect(DISTINCT type(r) + ' ' + m.id)[..8] AS outs "
            "OPTIONAL MATCH (n)<-[r2]-(p:Entity {domain: $domain}) "
            "WITH n, outs, collect(DISTINCT p.id + ' ' + type(r2))[..8] AS ins "
            "RETURN n.id AS id, coalesce(n.type,'') AS type, "
            "       coalesce(n.evidence,'') AS evidence, outs, ins",
            domain=domain,
        )
        texts: list[tuple[str, str]] = []
        for r in rows:
            parts = [f"{r['id']}. Type: {r['type']}."]
            if r.get("evidence"):
                parts.append(f"Evidence: {r['evidence']}.")
            if r.get("outs"):
                parts.append("Connects to: " + "; ".join(r["outs"]) + ".")
            if r.get("ins"):
                parts.append("Referenced by: " + "; ".join(r["ins"]) + ".")
            texts.append((r["id"], " ".join(parts)))
        return texts

    # ---- exports --------------------------------------------------------

    async def export_json(self, domain: str) -> dict[str, Any]:
        await self._prepare()
        nodes = await self._run(
            "MATCH (n:Entity {domain: $domain}) "
            "OPTIONAL MATCH (n)-[r]-(:Entity {domain: $domain}) "
            "RETURN n.id AS id, coalesce(n.type,'') AS type, "
            "       coalesce(n.evidence,'') AS evidence, "
            "       coalesce(n.mentions,1) AS mentions, count(r) AS degree",
            domain=domain,
        )
        edges = await self._run(
            "MATCH (a:Entity {domain: $domain})-[r]->(b:Entity {domain: $domain}) "
            "RETURN a.id AS source, type(r) AS relation, b.id AS target, "
            "       coalesce(r.weight,1) AS weight",
            domain=domain,
        )
        return {
            "domain": domain,
            "directed": True,
            "multigraph": True,
            "nodes": [
                {
                    "id": n["id"],
                    "type": n["type"],
                    "label": n["id"],
                    "evidence": n["evidence"],
                    "mentions": n["mentions"],
                    "degree": n["degree"],
                }
                for n in nodes
            ],
            "edges": edges,
        }

    async def export_graphml(self, domain: str) -> bytes:
        """GraphML built from the exported node-link data.

        Neo4j has no native GraphML writer over Bolt, so the graph is rebuilt
        in NetworkX and written out — the same artefact the other stores
        produce, so downstream tools see no difference.
        """
        import io

        import networkx as nx

        data = await self.export_json(domain)
        g = nx.MultiDiGraph()
        for n in data["nodes"]:
            g.add_node(
                n["id"],
                type=n["type"],
                label=n["label"],
                evidence=n["evidence"],
                mentions=n["mentions"],
            )
        for e in data["edges"]:
            g.add_edge(e["source"], e["target"], relation=e["relation"], weight=e["weight"])
        buf = io.BytesIO()
        nx.write_graphml(g, buf)
        return buf.getvalue()

    async def sample(self, domain: str, limit: int, node_type: str | None = None) -> dict[str, Any]:
        await self._prepare()
        if node_type:
            nodes = await self._run(
                "MATCH (n:Entity {domain: $domain}) WHERE n.type = $type "
                "RETURN n.id AS id, coalesce(n.type,'') AS type, "
                "       coalesce(n.evidence,'') AS evidence, "
                "       coalesce(n.mentions,1) AS mentions LIMIT $limit",
                domain=domain,
                type=node_type,
                limit=limit,
            )
        else:
            # Prefer well-connected nodes so the sample looks like a network
            # rather than a scatter of isolated points.
            nodes = await self._run(
                "MATCH (n:Entity {domain: $domain}) "
                "OPTIONAL MATCH (n)-[r]-(:Entity {domain: $domain}) "
                "WITH n, count(r) AS deg ORDER BY deg DESC LIMIT $limit "
                "RETURN n.id AS id, coalesce(n.type,'') AS type, "
                "       coalesce(n.evidence,'') AS evidence, "
                "       coalesce(n.mentions,1) AS mentions, deg AS degree",
                domain=domain,
                limit=limit,
            )
        ids = [n["id"] for n in nodes]
        edges = []
        if ids:
            edges = await self._run(
                "MATCH (a:Entity {domain: $domain})-[r]->(b:Entity {domain: $domain}) "
                "WHERE a.id IN $ids AND b.id IN $ids "
                "RETURN a.id AS source, type(r) AS relation, b.id AS target, "
                "       coalesce(r.weight,1) AS weight",
                domain=domain,
                ids=ids,
            )
        return {
            "truncated": len(ids) >= limit,
            "total_nodes": len(ids),
            "nodes": [
                {
                    "id": n["id"],
                    "type": n["type"],
                    "evidence": n["evidence"],
                    "mentions": n["mentions"],
                    "degree": n.get("degree", 0),
                }
                for n in nodes
            ],
            "edges": edges,
        }

    async def health(self) -> dict[str, Any]:
        try:
            await self._run("RETURN 1 AS ok")
            return {"mode": "neo4j", "status": "ok", "database": self._database}
        except Exception as exc:
            return {"mode": "neo4j", "status": "error", "detail": str(exc)}

    async def close(self) -> None:
        await self._driver.close()
