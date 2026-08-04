from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..access import editor
from ..agent.registry import get_registry
from ..agent.teams import get_team_registry
from ..config import get_settings
from ..design.architect import propose
from ..ontology.registry import save_ontology
from ..services.parsing import SUPPORTED
from pydantic import BaseModel, Field

router = APIRouter(prefix="/design", tags=["design"], dependencies=[Depends(editor)])

MAX_BYTES = 50 * 1024 * 1024


@router.post("/propose", summary="Draft a design from sample documents")
async def propose_design(
    goal: str = Form(""),
    files: list[UploadFile] = File(...),
) -> dict:
    """Reads a spread of the sample and proposes an ontology, agents and — where
    the goal warrants it — a team.

    Nothing is created. The response is a draft to review and edit; saving it
    goes through the normal endpoints, so the same validation applies.
    """
    uploads: list[tuple[str, bytes]] = []
    total = 0
    for upload in files:
        ext = "." + (upload.filename or "").rsplit(".", 1)[-1].lower()
        if ext not in SUPPORTED:
            raise HTTPException(
                status_code=400,
                detail=f"{upload.filename}: unsupported type. Supported: {sorted(SUPPORTED)}")
        data = await upload.read()
        total += len(data)
        if total > MAX_BYTES:
            raise HTTPException(status_code=413,
                                detail="Sample exceeds 50 MB. A few representative "
                                       "files are enough — this is a sample, not an "
                                       "ingestion.")
        uploads.append((upload.filename or "upload", data))

    settings = get_settings()
    try:
        proposal = await propose(uploads, goal,
                                 rows_per_chunk=settings.rows_per_chunk,
                                 chunk_size=settings.chunk_size,
                                 overlap=settings.chunk_overlap)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return proposal.as_dict()


class ApplyRequest(BaseModel):
    ontology: dict = Field(default_factory=dict)
    agents: list[dict] = Field(default_factory=list)
    team: dict | None = None


@router.post("/apply", status_code=201, summary="Save a reviewed design")
async def apply_design(body: ApplyRequest) -> dict:
    """Creates the domain, then the agents, then the team.

    Ordered so a failure leaves something coherent: an ontology with no agents is
    usable, agents pointing at a domain that does not exist are not. Each part
    goes through the same validation as if it had been created by hand.
    """
    created: dict[str, list[str]] = {"ontology": [], "agents": [], "team": []}
    try:
        ontology = save_ontology({
            "key": body.ontology.get("key"),
            "name": body.ontology.get("name"),
            "description": body.ontology.get("description"),
            "entity_types": body.ontology.get("entity_types") or {},
            "allowed_triples": [[t["source"], t["relation"], t["target"]]
                                for t in (body.ontology.get("allowed_triples") or [])],
            "open_relations": body.ontology.get("open_relations", False),
            "normalization": {
                "strip_type_prefixes": body.ontology.get("strip_type_prefixes", True),
                "collapse_whitespace": body.ontology.get("collapse_whitespace", True),
                "id_transforms": body.ontology.get("id_transforms") or [],
                "rules_text": body.ontology.get("normalization_rules") or [],
            },
            "custom_prompt": body.ontology.get("custom_prompt", ""),
        })
        created["ontology"].append(ontology.key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Domain: {exc}") from exc

    agents = get_registry()
    for spec in body.agents:
        try:
            agents.save({**spec, "domains": spec.get("domains") or [ontology.key]})
            created["agents"].append(spec.get("key", ""))
        except ValueError as exc:
            created.setdefault("skipped", []).append(f"{spec.get('key')}: {exc}")

    if body.team and created["agents"]:
        try:
            team = get_team_registry().save(body.team)
            created["team"].append(team.key)
        except ValueError as exc:
            created.setdefault("skipped", []).append(f"team: {exc}")

    return {"created": created,
            "next": f"Upload documents on the Ingest tab with domain '{ontology.key}'."}
