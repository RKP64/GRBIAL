from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..ontology import get_ontology, list_ontologies
from ..ontology.registry import Ontology, delete_ontology, save_ontology
from ..access import Principal, current_principal, editor, viewer

router = APIRouter(prefix="/ontologies", tags=["ontology"],
                   dependencies=[Depends(viewer)])


class EntityTypeIn(BaseModel):
    id_rule: str = ""
    examples: list[str] = Field(default_factory=list)


class TripleIn(BaseModel):
    source: str
    relation: str
    target: str


class IdTransformIn(BaseModel):
    """A domain-defined identifier rule: regex in, canonical form out."""

    pattern: str = ""
    replace: str = ""
    case: Literal["", "upper", "lower", "title"] = ""
    note: str = ""


class OntologyIn(BaseModel):
    key: str
    name: str = ""
    description: str = ""
    entity_types: dict[str, EntityTypeIn]
    allowed_triples: list[TripleIn] = Field(default_factory=list)
    normalization_rules: list[str] = Field(default_factory=list)
    id_transforms: list[IdTransformIn] = Field(default_factory=list)
    strip_type_prefixes: bool = True
    collapse_whitespace: bool = True
    open_relations: bool = False
    custom_prompt: str = ""


def _detail(o: Ontology) -> dict:
    return {
        "key": o.key, "name": o.name, "description": o.description,
        "builtin": o.builtin,
        "entity_types": o.entity_types,
        "allowed_triples": [{"source": s, "relation": r, "target": t}
                            for s, r, t in sorted(o.allowed_triples)],
        "normalization_rules": o.normalization_rules,
        "id_transforms": [t.to_spec() for t in o.id_transforms],
        "transform_errors": [
            {"pattern": t.pattern_src, "error": t.error} for t in o.id_transforms if t.error
        ],
        "strip_type_prefixes": o.strip_type_prefixes,
        "collapse_whitespace": o.collapse_whitespace,
        "open_relations": o.open_relations,
        "custom_prompt": o.custom_prompt,
        "generated_prompt": o.generated_prompt(),
        "effective_prompt": o.extraction_prompt(),
    }


@router.get("", summary="List domain ontologies")
async def list_all() -> list[dict]:
    return list_ontologies()


@router.get("/{key}", summary="Full schema contract for one domain")
async def detail(key: str) -> dict:
    try:
        return _detail(get_ontology(key))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("", status_code=201, summary="Create or replace a domain ontology",
             dependencies=[Depends(editor)])
async def upsert(body: OntologyIn) -> dict:
    spec = {
        "key": body.key,
        "name": body.name or body.key,
        "description": body.description,
        "entity_types": {k: v.model_dump() for k, v in body.entity_types.items()},
        "allowed_triples": [[t.source, t.relation, t.target] for t in body.allowed_triples],
        "open_relations": body.open_relations,
        "normalization": {
            "strip_type_prefixes": body.strip_type_prefixes,
            "collapse_whitespace": body.collapse_whitespace,
            "id_transforms": [t.model_dump() for t in body.id_transforms],
            "rules_text": body.normalization_rules,
        },
        "custom_prompt": body.custom_prompt,
    }
    try:
        return _detail(save_ontology(spec))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{key}/duplicate", status_code=201,
             summary="Copy an ontology under a new key (use to edit a built-in)",
             dependencies=[Depends(editor)])
async def duplicate(key: str, new_key: str) -> dict:
    try:
        source = get_ontology(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    spec = source.to_spec()
    spec["key"] = new_key
    spec["name"] = f"{source.name} (copy)"
    try:
        return _detail(save_ontology(spec))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{key}", status_code=204, summary="Delete a user-defined ontology",
               dependencies=[Depends(editor)])
async def remove(key: str) -> None:
    try:
        delete_ontology(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
