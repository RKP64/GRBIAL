"""Actions an agent may perform, declared in the ontology alongside the types.

The ontology says what exists. Without this it cannot say what can be *done*,
so an agent can only ever answer. Declaring actions turns the graph from
something read into something acted through — and, because every action writes
its outcome back, each one leaves the graph knowing more than before.

Four guardrails, in the order they matter:

  Typed parameters. A parameter declared as an entity type resolves against the
  graph before anything runs. An agent cannot raise a ticket against a gate it
  invented, because the reference fails to resolve and the action never fires.
  This is the single thing that makes agent writes safe.

  The signed-in user's permission, not the agent's. An agent configured with an
  action is not authorisation to run it.

  Confirmation before anything irreversible, showing resolved parameters, so a
  person approves an action against a specific node rather than a sentence.

  An audit entry naming the user, the parameters and the outcome.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,48}$")


@dataclass
class ActionParameter:
    name: str
    description: str = ""
    # Exactly one of these decides how the value is handled. An entity_type
    # makes the parameter a reference that must resolve to a real node; without
    # one it is a plain value.
    entity_type: str = ""
    type: str = "string"          # string | number | boolean
    enum: list[str] = field(default_factory=list)
    required: bool = True

    def as_schema(self) -> dict[str, Any]:
        prop: dict[str, Any] = {"type": "string" if self.entity_type else self.type}
        described = self.description
        if self.entity_type:
            described = (described + f" Must be an existing {self.entity_type} "
                                     f"in the graph.").strip()
        prop["description"] = described
        if self.enum:
            prop["enum"] = self.enum
        return prop


@dataclass
class Action:
    name: str
    description: str = ""
    parameters: list[ActionParameter] = field(default_factory=list)
    requires_role: str = "editor"     # viewer | editor | admin
    confirm: bool = True
    # What the action does. Either calls an external tool, writes to the graph,
    # or both — the external call first, so its result can be recorded.
    external_tool: str = ""
    writes_node_type: str = ""
    writes_relation: str = ""
    writes_target_parameter: str = ""

    def as_schema(self) -> dict[str, Any]:
        return {
            "name": f"action__{self.name}",
            "description": (self.description +
                            (" Requires confirmation before it runs."
                             if self.confirm else "")).strip(),
            "parameters": {
                "type": "object",
                "properties": {p.name: p.as_schema() for p in self.parameters},
                "required": [p.name for p in self.parameters if p.required],
            },
        }


def parse_actions(raw: list[dict[str, Any]] | None) -> dict[str, Action]:
    """Read the actions block, discarding anything malformed.

    A broken action is dropped rather than raised: one bad entry should not stop
    a domain loading and take its extraction and querying down with it.
    """
    actions: dict[str, Action] = {}
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip().lower()
        if not NAME_RE.fullmatch(name):
            log.warning("Skipping action with unusable name: %r", item.get("name"))
            continue

        params: list[ActionParameter] = []
        for p in item.get("parameters") or []:
            if not isinstance(p, dict):
                continue
            pname = str(p.get("name", "")).strip()
            if not pname:
                continue
            params.append(ActionParameter(
                name=pname,
                description=str(p.get("description", "")).strip(),
                entity_type=str(p.get("entity_type", "")).strip(),
                type=str(p.get("type", "string")).strip() or "string",
                enum=[str(e) for e in (p.get("enum") or [])],
                required=bool(p.get("required", True)),
            ))

        effect = item.get("effect") or {}
        writes = effect.get("writes") or {}
        node = writes.get("node") or {}
        edge = writes.get("edge") or {}

        role = str(item.get("requires_role", "editor")).strip().lower()
        if role not in {"viewer", "editor", "admin"}:
            role = "editor"

        actions[name] = Action(
            name=name,
            description=str(item.get("description", "")).strip(),
            parameters=params,
            requires_role=role,
            confirm=bool(item.get("confirm", True)),
            external_tool=str(effect.get("external", "")).strip(),
            writes_node_type=str(node.get("type", "")).strip(),
            writes_relation=str(edge.get("relation", "")).strip(),
            writes_target_parameter=str(edge.get("target", "")).strip(),
        )
    return actions


def as_spec(actions: dict[str, Action]) -> list[dict[str, Any]]:
    """Serialise back to the shape the ontology file stores."""
    out = []
    for a in actions.values():
        entry: dict[str, Any] = {
            "name": a.name,
            "description": a.description,
            "requires_role": a.requires_role,
            "confirm": a.confirm,
            "parameters": [
                {k: v for k, v in {
                    "name": p.name, "description": p.description,
                    "entity_type": p.entity_type,
                    "type": p.type if not p.entity_type else None,
                    "enum": p.enum or None,
                    "required": p.required,
                }.items() if v not in (None, "", [])}
                for p in a.parameters
            ],
        }
        effect: dict[str, Any] = {}
        if a.external_tool:
            effect["external"] = a.external_tool
        writes: dict[str, Any] = {}
        if a.writes_node_type:
            writes["node"] = {"type": a.writes_node_type}
        if a.writes_relation and a.writes_target_parameter:
            writes["edge"] = {"relation": a.writes_relation,
                              "target": a.writes_target_parameter}
        if writes:
            effect["writes"] = writes
        if effect:
            entry["effect"] = effect
        out.append(entry)
    return out
