from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from .base import GraphStore
from .networkx_store import NetworkXStore


def _cloud_store() -> GraphStore:
    s = get_settings()
    from .gremlin_store import GremlinStore

    return GremlinStore(
        s.cosmos_gremlin_endpoint, s.cosmos_key, s.cosmos_database, s.cosmos_collection
    )


def _neo4j_store() -> GraphStore:
    s = get_settings()
    from .neo4j_store import Neo4jStore

    return Neo4jStore(s.neo4j_uri, s.neo4j_user, s.neo4j_password, s.neo4j_database)


def _remote_store() -> GraphStore:
    """The non-local backend for this deployment.

    Neo4j wins when it is configured, because a deployment that has set a
    Neo4j URI has chosen it deliberately; Cosmos Gremlin remains the fallback
    so existing deployments keep working with no configuration change.
    """
    s = get_settings()
    if s.neo4j_uri:
        return _neo4j_store()
    return _cloud_store()


@lru_cache
def get_store() -> GraphStore:
    """Select the storage mode.

    local — file-backed graph on this host; portable and dependency-free.
    cloud — the shared managed graph service (Cosmos DB Gremlin).
    neo4j — Neo4j over Bolt; a virtual machine inside the VNet in production,
            or AuraDB for development and demonstration.
    dual  — write to both; read locally for speed, mirror to the remote graph.

    If a remote graph is requested but unreachable, the platform falls back to
    local rather than refusing to start, and says so on the status endpoint.
    """
    s = get_settings()
    local = NetworkXStore(s.data_dir)
    mode = (s.graph_backend or "local").lower()

    if mode == "local":
        return local

    try:
        if mode == "neo4j":
            remote = _neo4j_store()
        elif mode == "cloud":
            remote = _cloud_store()
        else:
            remote = _remote_store()
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "Remote graph unavailable (%s); using local storage.", exc
        )
        return local

    if mode in {"cloud", "neo4j"}:
        return remote

    from .composite import CompositeStore

    return CompositeStore(local, remote)
