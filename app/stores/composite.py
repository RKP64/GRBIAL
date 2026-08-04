from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..ontology.models import EdgeIn, NodeIn
from .base import GraphStore

log = logging.getLogger(__name__)


class CompositeStore(GraphStore):
    """Writes to a local store and a cloud store together.

    Reads are served by the local store: traversal and layout need many small
    lookups, and doing those over the network would make the console sluggish
    for no benefit. Writes go to both, so the cloud graph is the shared
    system of record and the local file stays a portable, exportable mirror.

    A cloud write failure is logged and surfaced on the job, never raised —
    losing the mirror should not abort an extraction that is otherwise fine.
    """

    name = "dual"

    def __init__(self, local: GraphStore, cloud: GraphStore) -> None:
        self.local = local
        self.cloud = cloud
        self.cloud_errors = 0

    async def upsert(self, domain: str, nodes: list[NodeIn], edges: list[EdgeIn]) -> tuple[int, int]:
        local_task = self.local.upsert(domain, nodes, edges)
        cloud_task = self.cloud.upsert(domain, nodes, edges)
        results = await asyncio.gather(local_task, cloud_task, return_exceptions=True)
        local_result, cloud_result = results
        if isinstance(cloud_result, Exception):
            self.cloud_errors += 1
            log.warning("Cloud mirror write failed: %s", cloud_result)
        if isinstance(local_result, Exception):
            raise local_result
        return local_result

    async def stats(self, domain: str) -> dict[str, Any]:
        return await self.local.stats(domain)

    async def keyword_search(self, domain: str, terms: list[str], limit: int) -> list[str]:
        return await self.local.keyword_search(domain, terms, limit)

    async def neighbourhood(self, domain: str, node_ids: list[str], hops: int,
                            max_neighbours: int) -> dict[str, Any]:
        return await self.local.neighbourhood(domain, node_ids, hops, max_neighbours)

    async def node_texts(self, domain: str) -> list[tuple[str, str]]:
        return await self.local.node_texts(domain)

    async def export_graphml(self, domain: str) -> bytes:
        return await self.local.export_graphml(domain)

    async def export_json(self, domain: str) -> dict[str, Any]:
        return await self.local.export_json(domain)

    async def sample(self, domain: str, limit: int, node_type: str | None = None) -> dict[str, Any]:
        return await self.local.sample(domain, limit, node_type)

    async def health(self) -> dict[str, Any]:
        local, cloud = await asyncio.gather(
            self.local.health(), self.cloud.health(), return_exceptions=True
        )
        cloud_ok = isinstance(cloud, dict) and cloud.get("status") == "ok"
        return {
            "mode": "dual",
            "status": "ok" if cloud_ok else "degraded",
            "detail": None if cloud_ok else "The shared graph is unreachable; "
                                            "work is being saved locally only.",
            "mirror_write_failures": self.cloud_errors,
        }
