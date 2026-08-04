from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..ontology.models import EdgeIn, NodeIn


class GraphStore(ABC):
    """Storage seam.

    The API layer only ever talks to this interface, so swapping NetworkX for
    Cosmos Gremlin (or anything else) changes no calling code.
    """

    name: str = "base"

    @abstractmethod
    async def upsert(self, domain: str, nodes: list[NodeIn], edges: list[EdgeIn]) -> tuple[int, int]:
        """Idempotent write. Returns (nodes_added, edges_added)."""

    @abstractmethod
    async def stats(self, domain: str) -> dict[str, Any]:
        ...

    @abstractmethod
    async def keyword_search(self, domain: str, terms: list[str], limit: int) -> list[str]:
        ...

    @abstractmethod
    async def neighbourhood(self, domain: str, node_ids: list[str], hops: int, max_neighbours: int) -> dict[str, Any]:
        """Return nodes+edges around the entry points, as a serialisable subgraph."""

    @abstractmethod
    async def node_texts(self, domain: str) -> list[tuple[str, str]]:
        """(node_id, embeddable_text) for every node.

        The text should describe the entity's role in the graph, not just its
        name — an entity mentioned once has a thin name but a meaningful set of
        relationships, and that is what makes it findable by meaning.
        """

    @abstractmethod
    async def export_graphml(self, domain: str) -> bytes:
        ...

    @abstractmethod
    async def export_json(self, domain: str) -> dict[str, Any]:
        """Portable node-link JSON for use in other applications."""

    @abstractmethod
    async def sample(self, domain: str, limit: int, node_type: str | None = None) -> dict[str, Any]:
        """A connected sample for visualisation."""

    async def health(self) -> dict[str, Any]:
        return {"mode": self.name, "status": "ok"}
