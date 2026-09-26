"""The shared upper ontology — the spine every domain ontology extends.

Each domain keeps its own depth: an airport domain has gates, stands and
belts; an asset domain has fixtures, circuits and transformers. Left alone,
those domains never join, because nothing says a gate and a fixture are both
physical assets at a location maintained by a vendor.

The spine is the small, centrally owned vocabulary that says exactly that. A
domain entity type declares which spine type it extends; nothing else changes
about how the domain is modelled. Questions that cross functions then join on
the spine rather than on hand-written mappings between every pair of domains.

The list is deliberately short. A spine that grows to fifty types stops being
a join layer and becomes a second ontology that every domain has to fight.
"""
from __future__ import annotations

from typing import Any

SPINE: dict[str, str] = {
    "Person": "An individual — employee, passenger, contractor, officer.",
    "Asset": "A physical or logical thing that is owned, operated or maintained.",
    "Vendor": "An external organisation that supplies goods or services.",
    "Location": "A place — site, terminal, zone, base, building, room.",
    "Process": "A defined activity or workflow — maintenance, check-in, audit.",
    "Document": "A record — procedure, manual, circular, contract, report.",
    "Organisation": "An internal unit — department, formation, team, airline.",
    "Event": "Something that happened at a time — incident, flight, inspection.",
}


def is_spine_type(name: str | None) -> bool:
    return bool(name) and name in SPINE


def spine_types() -> list[dict[str, str]]:
    return [{"type": k, "description": v} for k, v in SPINE.items()]


def validate_extends(entity_types: dict[str, dict]) -> list[str]:
    """Return a message per entity type whose `extends` names no spine type.

    Checked on save rather than at extraction time: an ontology that points at
    a spine type that does not exist would silently fail to join, which is the
    exact failure the spine exists to prevent.
    """
    problems = []
    for name, spec in (entity_types or {}).items():
        target = (spec or {}).get("extends")
        if target and not is_spine_type(target):
            problems.append(
                f"entity type '{name}' extends '{target}', which is not a spine type "
                f"(allowed: {', '.join(SPINE)})"
            )
    return problems


def join_points(ontologies: list[Any]) -> dict[str, list[dict[str, str]]]:
    """For each spine type, the domain entity types that extend it.

    This is the answer to "where do these functions meet?" — every spine type
    with more than one domain behind it is a place a cross-domain question can
    join.
    """
    out: dict[str, list[dict[str, str]]] = {k: [] for k in SPINE}
    for onto in ontologies:
        for type_name, spec in (onto.entity_types or {}).items():
            target = (spec or {}).get("extends")
            if is_spine_type(target):
                out[target].append({"domain": onto.key, "entity_type": type_name})
    return out
