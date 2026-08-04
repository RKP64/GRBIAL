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


@lru_cache
def get_store() -> GraphStore:
    """Select the storage mode.

    local — file-backed graph on this host; portable and dependency-free.
    cloud — the shared managed graph service.
    dual  — write to both; read locally for speed, mirror to the shared graph.

    If cloud is requested but unreachable, the platform falls back to local
    rather than refusing to start, and says so on the status endpoint.
    """
    s = get_settings()
    local = NetworkXStore(s.data_dir)
    mode = (s.graph_backend or "local").lower()

    if mode == "local":
        return local
    try:
        cloud = _cloud_store()
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "Cloud graph unavailable (%s); using local storage.", exc
        )
        return local
    if mode == "cloud":
        return cloud
    from .composite import CompositeStore

    return CompositeStore(local, cloud)
