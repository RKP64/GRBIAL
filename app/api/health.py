from fastapi import APIRouter

from ..config import get_settings
from ..ontology import list_ontologies
from ..providers import provider_status
from ..retrieval import get_retriever
from ..stores import get_store

router = APIRouter(tags=["system"])


@router.get("/healthz", summary="Liveness")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness — capability report")
async def readyz() -> dict:
    """Reports what the platform can currently do.

    Deliberately describes capabilities rather than the components behind them:
    the console should not have to know, or show, which libraries or services
    are in use.
    """
    s = get_settings()
    store_health = await get_store().health()
    try:
        retrieval_mode = get_retriever().name
        retrieval_ok = True
    except Exception:
        retrieval_mode, retrieval_ok = "text", False
    return {
        "status": "ok" if store_health.get("status") == "ok" else "degraded",
        "storage": {
            "mode": store_health.get("mode", "local"),
            "status": store_health.get("status", "ok"),
            "detail": store_health.get("detail"),
        },
        "retrieval": {"mode": retrieval_mode, "ready": retrieval_ok},
        "extraction_ready": s.llm_configured and provider_status()["ready"],
        "semantic_search_ready": s.embeddings_configured,
        "document_search_ready": s.azure_search_configured,
        "domains": [o["key"] for o in list_ontologies()],
    }
