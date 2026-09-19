"""Run a declared action: resolve, authorise, execute, record.

Kept separate from the agent loop because the loop decides *what* to attempt and
this decides whether it is allowed to happen. Mixing the two makes it easy for a
later change to the loop to quietly bypass a check.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..access import Principal, Role
from ..ontology import get_ontology
from ..ontology.models import EdgeIn, NodeIn
from ..stores import get_store

log = logging.getLogger(__name__)

ROLE_ORDER = {"viewer": 0, "editor": 1, "admin": 2}


@dataclass
class ActionOutcome:
    ok: bool
    message: str
    needs_confirmation: bool = False
    resolved: dict[str, Any] | None = None
    wrote: dict[str, Any] | None = None


async def _resolve_entity(domain: str, entity_type: str, value: str) -> str | None:
    """Find a node of the right type matching this value.

    Exact id first, then a keyword search filtered to the declared type. Returns
    None when nothing matches, which is what stops an action running against an
    entity the model invented.
    """
    value = (value or "").strip()
    if not value:
        return None
    store = get_store()

    exported = await store.export_json(domain)
    by_id = {n["id"]: n for n in exported.get("nodes", [])}

    node = by_id.get(value)
    if node is not None and (not entity_type or node.get("type") == entity_type):
        return value

    # Case-insensitive exact match before anything fuzzier: "gate a12" should
    # find "Gate A12" without opening the door to a loose match.
    lowered = value.lower()
    for nid, n in by_id.items():
        if nid.lower() == lowered and (not entity_type or n.get("type") == entity_type):
            return nid

    candidates = [nid for nid, n in by_id.items()
                  if (not entity_type or n.get("type") == entity_type)
                  and lowered in nid.lower()]
    # One unambiguous candidate is accepted; several is not, because picking the
    # first would mean acting on the wrong record without anyone noticing.
    return candidates[0] if len(candidates) == 1 else None


def _record(principal: Principal, domain: str, action_name: str,
            outcome: str, detail: str = "", resolved: dict | None = None) -> None:
    """Write the attempt to the audit log.

    Refusals are recorded as well as successes: a run of blocked attempts
    against entities that do not exist is a signal worth being able to see, and
    a log that only holds what succeeded cannot answer "what was tried".

    Never allowed to raise — an audit failure must not turn into a failed
    action, or worse, an action that ran but reported failure.
    """
    try:
        from ..auth import AuditEvent, get_audit_log
        params = ", ".join(f"{k}={v}" for k, v in (resolved or {}).items())
        get_audit_log().record(AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            user=principal.name,
            action=f"action:{action_name}",
            path=f"/{domain}",
            resource=action_name,
            detail=f"{outcome}"
                   + (f" | {params}" if params else "")
                   + (f" | {detail}" if detail else ""),
        ))
    except Exception as exc:
        log.warning("Could not record action in the audit log: %s", exc)


async def run_action(
    *,
    domain: str,
    action_name: str,
    arguments: dict[str, Any],
    principal: Principal,
    confirmed: bool = False,
) -> ActionOutcome:
    ontology = get_ontology(domain)
    action = ontology.actions.get(action_name)
    if action is None:
        _record(principal, domain, action_name, "unknown-action")
        return ActionOutcome(False, f"'{action_name}' is not an action in this domain.")

    # Permission belongs to the person, not the agent. An agent being configured
    # with an action is not authorisation to run it.
    needed = ROLE_ORDER.get(action.requires_role, 1)
    held = ROLE_ORDER.get(principal.role.value, 0)
    if held < needed:
        _record(principal, domain, action_name, "refused",
                f"needs {action.requires_role}, has {principal.role.value}")
        return ActionOutcome(
            False,
            f"This action needs the {action.requires_role} role. You are signed in "
            f"as {principal.role.value}.")

    if not principal.may_read(domain):
        _record(principal, domain, action_name, "refused", "no access to domain")
        return ActionOutcome(False, f"You do not have access to '{domain}'.")

    # Resolve before doing anything. A reference that does not resolve stops the
    # action here, before an external system is touched.
    resolved: dict[str, Any] = {}
    for p in action.parameters:
        raw = arguments.get(p.name)
        if raw in (None, ""):
            if p.required:
                _record(principal, domain, action_name, "rejected",
                        f"missing {p.name}")
                return ActionOutcome(False, f"'{p.name}' is required.")
            continue
        if p.entity_type:
            match = await _resolve_entity(domain, p.entity_type, str(raw))
            if match is None:
                _record(principal, domain, action_name, "rejected",
                        f"{p.name}={raw!r} did not resolve to a {p.entity_type}")
                return ActionOutcome(
                    False,
                    f"No {p.entity_type} matching '{raw}' exists in {domain}. "
                    f"Check the name, or search the graph first.")
            resolved[p.name] = match
        elif p.enum and str(raw) not in p.enum:
            _record(principal, domain, action_name, "rejected",
                    f"{p.name}={raw!r} not in allowed values")
            return ActionOutcome(
                False, f"'{p.name}' must be one of: {', '.join(p.enum)}.")
        else:
            resolved[p.name] = raw

    if action.confirm and not confirmed:
        detail = ", ".join(f"{k}: {v}" for k, v in resolved.items())
        _record(principal, domain, action_name, "awaiting-confirmation",
                resolved=resolved)
        return ActionOutcome(
            False,
            f"Ready to run '{action.name}' with {detail}. Confirm to proceed.",
            needs_confirmation=True, resolved=resolved)

    # External effect first, so anything it returns can be recorded.
    external_result = ""
    if action.external_tool:
        from ..mcp.registry import get_server_registry
        try:
            external_result = await get_server_registry().call(
                action.external_tool, resolved)
        except Exception as exc:
            log.exception("Action %s external call failed", action.name)
            _record(principal, domain, action_name, "failed",
                    f"external step: {exc}", resolved)
            return ActionOutcome(False, f"The external step failed: {exc}")

    # Write the outcome back, so the next question about this entity sees it.
    wrote: dict[str, Any] = {}
    if action.writes_node_type:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        node_id = f"{action.writes_node_type}-{stamp}"
        node = NodeIn(id=node_id, type=action.writes_node_type,
                      metadata={
                          "created_by": principal.name,
                          "created_at": datetime.now(timezone.utc).isoformat(),
                          "action": action.name,
                          "source": "action",
                          "detail": (external_result or "")[:400],
                      })
        edges = []
        target_param = action.writes_target_parameter
        if action.writes_relation and target_param in resolved:
            edges.append(EdgeIn(source=node_id, relation=action.writes_relation,
                                target=str(resolved[target_param])))
        await get_store().upsert(domain, [node], edges)
        wrote = {"node": node_id, "type": action.writes_node_type,
                 "edges": [e.type for e in edges]}

    parts = [f"'{action.name}' completed."]
    if external_result:
        parts.append(external_result[:600])
    if wrote:
        parts.append(f"Recorded {wrote['node']} in the graph.")
    _record(principal, domain, action_name, "completed",
            f"wrote {wrote['node']}" if wrote else "", resolved)
    return ActionOutcome(True, " ".join(parts), resolved=resolved, wrote=wrote)
