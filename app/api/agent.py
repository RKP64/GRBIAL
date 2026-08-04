import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..agent.loop import run_agent
from ..agent.registry import ALL_TOOLS, get_registry
from ..agent.tools import TOOL_SCHEMAS
from ..providers import get_provider
from ..access import Principal, current_principal, editor, require_domain, viewer

router = APIRouter(prefix="/agents", tags=["agent"], dependencies=[Depends(viewer)])


# ------------------------------------------------------------------ catalogue
@router.get("/tools", summary="Tools an agent may be given")
async def tools() -> dict:
    try:
        available = get_provider().tools_available
        note = None if available else (
            "The configured language model does not support tool use."
        )
    except Exception as exc:
        available, note = False, str(exc)
    return {
        "available": available,
        "note": note,
        "tools": [{"name": t["name"], "description": t["description"]}
                  for t in TOOL_SCHEMAS],
    }


# ------------------------------------------------------------------ agents
class AgentIn(BaseModel):
    key: str
    name: str = ""
    description: str = ""
    domains: list[str] = Field(default_factory=list)
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=lambda: list(ALL_TOOLS))
    max_steps: int = Field(default=6, ge=1, le=12)
    temperature: float = Field(default=0.2, ge=0, le=1)
    verify: bool = True
    starters: list[str] = Field(default_factory=list)


@router.get("", summary="Saved agents")
async def list_agents() -> list[dict]:
    return [a.as_dict() for a in get_registry().list()]


@router.get("/{key}", summary="One agent")
async def get_agent(key: str) -> dict:
    try:
        return get_registry().get(key).as_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("", status_code=201, summary="Create or replace an agent",
             dependencies=[Depends(editor)])
async def save_agent(body: AgentIn) -> dict:
    try:
        return get_registry().save(body.model_dump()).as_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{key}", status_code=204, summary="Delete an agent",
               dependencies=[Depends(editor)])
async def delete_agent(key: str) -> None:
    try:
        get_registry().delete(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ------------------------------------------------------------------ asking
class AskRequest(BaseModel):
    question: str = Field(min_length=2)
    domain: str | None = None      # must be one the agent may read
    max_steps: int | None = Field(default=None, ge=1, le=12)
    verify: bool | None = None


@router.post("/{key}/ask", summary="Ask a saved agent")
async def ask_agent(key: str, body: AskRequest) -> dict:
    try:
        agent = get_registry().get(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    domain = body.domain or agent.default_domain()
    try:
        result = await run_agent(
            domain, body.question,
            system_prompt=agent.system_prompt,
            allowed_tools=agent.tools,
            max_steps=body.max_steps or agent.max_steps,
            temperature=agent.temperature,
            verify=agent.verify if body.verify is None else body.verify,
            permitted_domains=agent.domains,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    payload = result.as_dict()
    payload.update(question=body.question, domain=domain,
                   agent={"key": agent.key, "name": agent.name})
    return payload


# ------------------------------------------------------------------ export / import

EXPORT_VERSION = 1


def _export_envelope(kind: str, items: list[dict]) -> dict:
    """Wrap one or more agents/teams in a portable envelope."""
    return {
        "platform": "knowledge-graph-platform",
        "export_version": EXPORT_VERSION,
        "kind": kind,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
    }


def _strip_timestamps(d: dict) -> dict:
    """Remove created_at/updated_at — the target instance sets its own."""
    return {k: v for k, v in d.items() if k not in ("created_at", "updated_at")}


@router.get("/{key}/export", summary="Export one agent as a portable JSON file")
async def export_agent(key: str) -> Response:
    try:
        agent = get_registry().get(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    payload = _export_envelope("agent", [_strip_timestamps(agent.as_dict())])
    return Response(
        content=json.dumps(payload, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{key}.agent.json"'},
    )


@router.get("/export/all", summary="Export every agent as a single file")
async def export_all_agents() -> Response:
    agents = get_registry().list()
    payload = _export_envelope(
        "agent", [_strip_timestamps(a.as_dict()) for a in agents])
    return Response(
        content=json.dumps(payload, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="agents.json"'},
    )


class ImportResult(BaseModel):
    imported: list[str] = Field(default_factory=list)
    skipped: list[dict] = Field(default_factory=list)


@router.post("/import", status_code=201,
             summary="Import agents from an export file",
             dependencies=[Depends(editor)])
async def import_agents(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400,
                            detail=f"Invalid JSON: {exc}") from exc

    if data.get("kind") not in ("agent",):
        raise HTTPException(
            status_code=400,
            detail=f"Expected kind 'agent', got '{data.get('kind')}'.")

    items = data.get("items", [])
    if not items:
        raise HTTPException(status_code=400, detail="No agents in the file.")

    registry = get_registry()
    result: dict = {"imported": [], "skipped": []}
    for spec in items:
        try:
            agent = registry.save(spec)
            result["imported"].append(agent.key)
        except (ValueError, TypeError) as exc:
            result["skipped"].append({"key": spec.get("key", "?"),
                                      "reason": str(exc)})
    return result


# ------------------------------------------------------------------ ad-hoc
class AdHocRequest(BaseModel):
    domain: str
    question: str = Field(min_length=2)
    system_prompt: str = ""
    allowed_tools: list[str] | None = None
    max_steps: int = Field(default=6, ge=1, le=12)
    temperature: float = Field(default=0.2, ge=0, le=1)
    verify: bool = True


@router.post("/ask", summary="Ask without a saved agent")
async def ask_adhoc(body: AdHocRequest) -> dict:
    """Unscoped by design — this is the builder's own path, equivalent to
    querying a domain directly. Saved agents are the scoped route."""
    try:
        result = await run_agent(
            body.domain, body.question, system_prompt=body.system_prompt,
            allowed_tools=body.allowed_tools, max_steps=body.max_steps,
            temperature=body.temperature, verify=body.verify,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    payload = result.as_dict()
    payload.update(question=body.question, domain=body.domain)
    return payload


# ------------------------------------------------------------------ teams
class MemberIn(BaseModel):
    agent: str
    hands_off_to: list[str] = Field(default_factory=list)
    when: str = ""


class TeamIn(BaseModel):
    key: str
    name: str = ""
    description: str = ""
    entry: str = ""
    members: list[MemberIn] = Field(default_factory=list)
    max_handoffs: int = Field(default=6, ge=1, le=12)
    max_steps_per_agent: int = Field(default=4, ge=1, le=8)
    verify: bool = True
    starters: list[str] = Field(default_factory=list)


class TeamAskRequest(BaseModel):
    question: str = Field(min_length=2)
    verify: bool | None = None


teams_router = APIRouter(prefix="/teams", tags=["agent"],
                         dependencies=[Depends(viewer)])


@teams_router.get("", summary="Saved teams")
async def list_teams() -> list[dict]:
    from ..agent.teams import get_team_registry

    return [t.as_dict() for t in get_team_registry().list()]


@teams_router.get("/{key}", summary="One team")
async def get_team(key: str) -> dict:
    from ..agent.teams import get_team_registry

    try:
        return get_team_registry().get(key).as_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@teams_router.post("", status_code=201, summary="Create or replace a team",
                   dependencies=[Depends(editor)])
async def save_team(body: TeamIn) -> dict:
    from ..agent.teams import get_team_registry

    try:
        return get_team_registry().save(body.model_dump()).as_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@teams_router.delete("/{key}", status_code=204, summary="Delete a team",
                     dependencies=[Depends(editor)])
async def delete_team(key: str) -> None:
    from ..agent.teams import get_team_registry

    try:
        get_team_registry().delete(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@teams_router.get("/{key}/export",
                   summary="Export one team and its agents as a portable file")
async def export_team(key: str) -> Response:
    from ..agent.teams import get_team_registry

    try:
        team = get_team_registry().get(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Bundle the agents this team references so the file is self-contained.
    agents_reg = get_registry()
    agent_items = []
    for member in team.members:
        try:
            agent_items.append(
                _strip_timestamps(agents_reg.get(member.agent).as_dict()))
        except KeyError:
            pass  # agent was deleted; team still exports

    payload = _export_envelope("team", [{
        "team": _strip_timestamps(team.as_dict()),
        "agents": agent_items,
    }])
    return Response(
        content=json.dumps(payload, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{key}.team.json"'},
    )


@teams_router.get("/export/all",
                   summary="Export every team (with agents) as a single file")
async def export_all_teams() -> Response:
    from ..agent.teams import get_team_registry

    teams = get_team_registry().list()
    agents_reg = get_registry()
    all_agent_keys: set[str] = set()
    team_items = []
    for t in teams:
        for m in t.members:
            all_agent_keys.add(m.agent)
        team_items.append(_strip_timestamps(t.as_dict()))

    agent_items = []
    for k in sorted(all_agent_keys):
        try:
            agent_items.append(_strip_timestamps(agents_reg.get(k).as_dict()))
        except KeyError:
            pass

    payload = _export_envelope("team", [{
        "teams": team_items,
        "agents": agent_items,
    }])
    return Response(
        content=json.dumps(payload, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="teams.json"'},
    )


@teams_router.post("/import", status_code=201,
                    summary="Import teams (and their agents) from an export file",
                    dependencies=[Depends(editor)])
async def import_teams(file: UploadFile = File(...)) -> dict:
    from ..agent.teams import get_team_registry

    raw = await file.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400,
                            detail=f"Invalid JSON: {exc}") from exc

    if data.get("kind") != "team":
        raise HTTPException(
            status_code=400,
            detail=f"Expected kind 'team', got '{data.get('kind')}'.")

    items = data.get("items", [])
    if not items:
        raise HTTPException(status_code=400, detail="No teams in the file.")

    agents_reg = get_registry()
    teams_reg = get_team_registry()
    result: dict = {"agents_imported": [], "agents_skipped": [],
                    "teams_imported": [], "teams_skipped": []}

    # Import agents first — teams reference them.
    for item in items:
        for spec in (item.get("agents") or []):
            try:
                agent = agents_reg.save(spec)
                result["agents_imported"].append(agent.key)
            except (ValueError, TypeError) as exc:
                result["agents_skipped"].append(
                    {"key": spec.get("key", "?"), "reason": str(exc)})

        # Handle both single-team and multi-team exports.
        team_specs = item.get("teams") or []
        if not team_specs and "team" in item:
            team_specs = [item["team"]]
        for spec in team_specs:
            try:
                team = teams_reg.save(spec)
                result["teams_imported"].append(team.key)
            except (ValueError, TypeError) as exc:
                result["teams_skipped"].append(
                    {"key": spec.get("key", "?"), "reason": str(exc)})

    return result


@teams_router.post("/{key}/ask", summary="Ask a team")
async def ask_team(key: str, body: TeamAskRequest) -> dict:
    """The conversation starts with the team's entry agent and may pass between
    members. Every handoff is returned in `steps` with the stated reason."""
    from ..agent.teams import get_team_registry, run_team

    try:
        team = get_team_registry().get(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        result = await run_team(team, body.question, verify=body.verify)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    payload = result.as_dict()
    payload.update(question=body.question,
                   team={"key": team.key, "name": team.name})
    return payload
