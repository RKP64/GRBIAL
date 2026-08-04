import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from ..retrieval import get_chunk_index, get_retriever
from ..access import Principal, current_principal, editor, require_domain, viewer
from ..stores import get_store

router = APIRouter(prefix="/graph", tags=["graph"], dependencies=[Depends(viewer)])


@router.get("/{domain}/stats", summary="Node and edge counts by type")
async def stats(domain: str, principal: Principal = Depends(current_principal)) -> dict:
    require_domain(principal, domain)
    return await get_store().stats(domain)


@router.get("/{domain}/visualize", summary="Connected sample for the graph view")
async def visualize(
    domain: str,
    limit: int = Query(150, ge=10, le=600),
    node_type: str | None = None,
) -> dict:
    return await get_store().sample(domain, limit=limit, node_type=node_type)


@router.get("/{domain}/subgraph", summary="Neighbourhood around matching nodes")
async def subgraph(
    domain: str,
    q: str = Query(..., description="Text to locate entry-point nodes"),
    top_k: int = 8,
    hops: int = 1,
    max_neighbours: int = 25,
) -> dict:
    store = get_store()
    entry = await store.keyword_search(domain, [t for t in q.lower().split() if len(t) > 2], top_k)
    return await store.neighbourhood(domain, entry, hops=hops, max_neighbours=max_neighbours)


@router.post("/{domain}/index", summary="Build the vector index for this domain",
             dependencies=[Depends(editor)])
async def build_index(domain: str) -> dict:
    try:
        return await get_retriever().build(domain)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/embeddings/warm", tags=["system"],
             summary="Load the embedding model before it is first needed")
async def warm_embeddings() -> dict:
    """Loading a local model can involve a download on first use. Calling this
    ahead of an ingestion run keeps that cost out of the run itself."""
    from ..providers import get_embedder

    embedder = get_embedder()
    warm = getattr(embedder, "warm", None)
    if warm is None:
        return {"warmed": False, "note": "This embedding source needs no warm-up."}
    try:
        detail = await warm()
        return {"warmed": True, **detail}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{domain}/passages", summary="How many source passages are kept locally")
async def passage_stats(domain: str) -> dict:
    return get_chunk_index().stats(domain)


@router.post("/{domain}/passages/index",
             summary="Enable meaning-based passage search for kept passages",
             dependencies=[Depends(editor)])
async def build_passage_vectors(domain: str) -> dict:
    try:
        return await get_chunk_index().build_vectors(domain)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{domain}/passages", status_code=204,
               summary="Discard the kept source passages for this domain",
               dependencies=[Depends(editor)])
async def clear_passages(domain: str) -> None:
    get_chunk_index().clear(domain)


@router.get("/{domain}/export.graphml", summary="Download GraphML (Gephi, Neo4j, yEd)")
async def export_graphml(domain: str) -> Response:
    try:
        payload = await get_store().export_graphml(domain)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return Response(content=payload, media_type="application/xml",
                    headers={"Content-Disposition": f'attachment; filename="{domain}.graphml"'})


@router.get("/{domain}/export.json", summary="Download node-link JSON (D3, Cytoscape, custom apps)")
async def export_json(domain: str) -> Response:
    payload = await get_store().export_json(domain)
    return Response(
        content=json.dumps(payload, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{domain}.json"'},
    )
