import secrets

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..access import ALL_DOMAINS, Principal, Role, admin, current_principal, get_store

router = APIRouter(prefix="/access", tags=["access"])


class PrincipalIn(BaseModel):
    key: str = Field(min_length=8, max_length=128)
    name: str = Field(min_length=1)
    role: Role = Role.VIEWER
    domains: list[str] = Field(default_factory=lambda: [ALL_DOMAINS])


@router.get("/me", summary="Who am I, and what may I do")
async def me(principal: Principal = Depends(current_principal)) -> dict:
    """Called by the console on load so it can hide what the caller cannot use.
    The server enforces regardless — this only avoids offering dead buttons."""
    return {
        "name": principal.name,
        "role": principal.role.value,
        "domains": principal.domains,
        "all_domains": principal.all_domains,
        "can": {
            "read": True,
            "ingest": principal.at_least(Role.EDITOR),
            "manage_schemas": principal.at_least(Role.EDITOR),
            "manage_agents": principal.at_least(Role.EDITOR),
            "train": principal.at_least(Role.ADMIN),
            "manage_access": principal.at_least(Role.ADMIN),
        },
    }


@router.get("/keys", summary="Keys that exist", dependencies=[Depends(admin)])
async def list_keys() -> list[dict]:
    """Keys are masked. A key that has been issued cannot be read back — reissue
    instead, so a leaked list is not a leaked credential."""
    return [p.as_dict() for p in get_store().list()]


@router.post("/keys/generate", summary="Suggest a strong key",
             dependencies=[Depends(admin)])
async def generate_key() -> dict:
    return {"key": secrets.token_urlsafe(32)}


@router.post("/keys", status_code=201, summary="Create or update a key",
             dependencies=[Depends(admin)])
async def save_key(body: PrincipalIn) -> dict:
    try:
        principal = get_store().save(body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return principal.as_dict()


@router.delete("/keys/{key}", status_code=204, summary="Revoke a key",
               dependencies=[Depends(admin)])
async def delete_key(key: str) -> None:
    try:
        get_store().delete(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
