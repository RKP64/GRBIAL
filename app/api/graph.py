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


def _spine_lookup(domain: str):
    """type -> spine type for this domain, or an empty map if none is declared."""
    try:
        from ..ontology import get_ontology
        onto = get_ontology(domain)
        return {t: onto.spine_of(t) for t in onto.entity_types if onto.spine_of(t)}
    except Exception:
        return {}


@router.get("/{domain}/visualize", summary="Connected sample for the graph view")
async def visualize(
    domain: str,
    limit: int = Query(250, ge=10, le=3000),
    node_type: str | None = None,
    principal: Principal = Depends(current_principal),
) -> dict:
    require_domain(principal, domain)
    data = await get_store().sample(domain, limit=limit, node_type=node_type)
    spine = _spine_lookup(domain)
    for n in data.get("nodes", []):
        n["spine"] = spine.get(n.get("type") or "")
    data["edge_count"] = len(data.get("edges", []))
    data["spine_declared"] = bool(spine)
    return data


@router.get("/{domain}/search", summary="Find entities by name for the graph view")
async def search_entities(
    domain: str,
    q: str = Query(..., min_length=1),
    limit: int = Query(12, ge=1, le=50),
    principal: Principal = Depends(current_principal),
) -> list[dict]:
    """Type-ahead lookup across the whole domain, not only the drawn sample.

    Ranked exact match first, then prefix, then substring, and within each by
    how connected the entity is — the well-connected match is usually the one
    the person meant.
    """
    require_domain(principal, domain)
    needle = q.strip().lower()
    graph = await get_store().export_json(domain)
    spine = _spine_lookup(domain)
    hits = []
    for n in graph.get("nodes", []):
        nid = str(n.get("id", ""))
        low = nid.lower()
        if needle not in low:
            continue
        rank = 0 if low == needle else 1 if low.startswith(needle) else 2
        hits.append((rank, -int(n.get("degree") or 0), nid, n))
    hits.sort(key=lambda h: (h[0], h[1], h[2]))
    return [
        {"id": nid, "type": n.get("type") or "", "degree": int(n.get("degree") or 0),
         "spine": spine.get(n.get("type") or "")}
        for _, _, nid, n in hits[:limit]
    ]


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


@router.get("/{domain}/impact", summary="What connects to an entity")
async def impact_report(domain: str, entity: str,
                        hops: int = Query(2, ge=1, le=4),
                        limit: int = Query(600, ge=20, le=3000),
                        principal: Principal = Depends(current_principal)) -> dict:
    """Reachability from one entity, grouped by distance.

    Deliberately named impact rather than simulation: it reports what is
    connected, not what would happen. The graph carries no numeric properties to
    predict consequences from.
    """
    if not principal.may_read(domain):
        raise HTTPException(status_code=403,
                            detail=f"You do not have access to '{domain}'.")
    from ..services.impact import impact
    result = await impact(domain, entity, hops=hops, limit=limit)
    spine = _spine_lookup(domain)
    for n in result.get("nodes", []) or []:
        n["spine"] = spine.get(n.get("type") or "")
    return result


# ------------------------------------------------------------------ resolution

@router.get("/{domain}/resolution", summary="Entity merges awaiting a decision")
async def pending_merges(domain: str,
                         principal: Principal = Depends(current_principal)) -> list[dict]:
    if not principal.may_read(domain):
        raise HTTPException(status_code=403,
                            detail=f"You do not have access to '{domain}'.")
    from ..services.resolution import Resolver
    return Resolver(domain).pending()


@router.post("/{domain}/resolution/{merge_id}", summary="Accept or reject a merge",
             dependencies=[Depends(editor)])
async def decide_merge(domain: str, merge_id: str, accept: bool = True) -> dict:
    """Apply a proposed merge, or dismiss it.

    Accepting rewrites every edge on the incoming node onto the existing one and
    removes the duplicate. There is no undo, which is why the ambiguous band
    reaches a person rather than being decided automatically.
    """
    from ..services.resolution import Resolver

    resolver = Resolver(domain)
    match = next((r for r in resolver.pending() if r["id"] == merge_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="No such pending merge.")

    if not accept:
        resolver.dismiss(merge_id)
        return {"merge_id": merge_id, "applied": False}

    store = get_store()
    graph = await store.export_json(domain)
    incoming, existing = match["incoming"], match["existing"]
    types = {n["id"]: n.get("type") for n in graph.get("nodes", [])}
    if incoming not in types or existing not in types:
        resolver.dismiss(merge_id)
        raise HTTPException(status_code=409,
                            detail="One of those entities is no longer in the graph.")

    from ..ontology.models import EdgeIn, NodeIn
    edges = []
    for e in graph.get("edges", []):
        source, target = e.get("source"), e.get("target")
        relation = e.get("relation") or e.get("type") or ""
        if incoming not in (source, target):
            continue
        edges.append(EdgeIn(
            source=existing if source == incoming else source,
            relation=relation,
            target=existing if target == incoming else target))

    await store.upsert(domain,
                       [NodeIn(id=existing, type=types[existing] or "",
                               metadata={"also_seen_as": incoming})],
                       edges)
    resolver.dismiss(merge_id)
    return {"merge_id": merge_id, "applied": True,
            "merged_into": existing, "edges_moved": len(edges),
            "note": "The duplicate node remains until the domain is rebuilt; "
                    "its relationships now also exist on the surviving entity."}
