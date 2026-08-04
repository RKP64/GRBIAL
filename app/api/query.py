from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..config import get_settings
from ..retrieval import get_chunk_index
from ..access import Principal, current_principal, viewer
from ..services.query import answer

router = APIRouter(prefix="/query", tags=["query"], dependencies=[Depends(viewer)])


class QueryRequest(BaseModel):
    domain: str
    question: str = Field(min_length=2)
    mode: Literal["graph", "hybrid"] = "graph"
    top_k: int = Field(default=5, ge=1, le=25)
    hops: int = Field(default=1, ge=0, le=3)
    max_neighbours: int = Field(default=25, ge=1, le=200)
    doc_k: int = Field(default=5, ge=1, le=20)
    index_name: str | None = None
    filter_expression: str | None = None
    doc_source: Literal["auto", "service", "local", "none"] = "auto"
    synthesise: bool = True
    # Check the generated answer against the graph before returning it.
    verify: bool = False


@router.get("/modes", summary="Which answering modes are available right now")
async def modes(domain: str | None = None) -> dict:
    """Reports what can answer a question now.

    Passages kept during ingestion make hybrid answering possible with no
    external service, so availability is reported per domain.
    """
    s = get_settings()
    local_ready = bool(domain) and get_chunk_index().exists(domain)
    service_ready = s.azure_search_configured
    sources = []
    if service_ready:
        sources.append("service")
    if local_ready:
        sources.append("local")
    return {
        "graph": {"available": True, "label": "Graph only"},
        "hybrid": {
            "available": bool(sources),
            "label": "Graph + documents",
            "sources": sources,
            "note": None if sources
                    else "No document source yet. Extract a document, or ask an "
                         "administrator to connect a search service.",
        },
    }


@router.post("", summary="Retrieve and answer")
async def run_query(req: QueryRequest) -> dict:
    try:
        result = await answer(
            req.domain, req.question, mode=req.mode, top_k=req.top_k, hops=req.hops,
            max_neighbours=req.max_neighbours, doc_k=req.doc_k,
            index_name=req.index_name, filter_expression=req.filter_expression,
            doc_source=req.doc_source, synthesise=req.synthesise,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if req.verify and result.get("answer"):
        from ..verification import verify_answer

        try:
            result["verification"] = (
                await verify_answer(req.domain, result["answer"])
            ).as_dict()
        except Exception as exc:  # verification must never fail the answer
            result["verification"] = {"error": str(exc)}
    return result
