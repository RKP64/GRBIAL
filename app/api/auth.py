"""Sign in, current user, user administration, audit trail."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..access import ALL_DOMAINS, Principal, Role, current_principal
from ..auth import (AuditEvent, User, get_audit_log, get_user_store,
                    create_token, hash_password)

router = APIRouter(prefix="/auth", tags=["auth"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ip(request: Request) -> str:
    return request.client.host if request.client else ""


class LoginRequest(BaseModel):
    username: str
    password: str


class NewUser(BaseModel):
    username: str = Field(min_length=3)
    password: str = Field(min_length=6)
    email: str = ""
    display_name: str = ""
    role: str = "editor"
    domains: list[str] = Field(default_factory=lambda: [ALL_DOMAINS])


@router.post("/login", summary="Sign in")
async def login(body: LoginRequest, request: Request) -> dict:
    user = get_user_store().authenticate(body.username, body.password)
    if user is None:
        # One message for both wrong-user and wrong-password: distinguishing
        # them tells an attacker which accounts exist.
        get_audit_log().record(AuditEvent(
            timestamp=_now(), user=body.username, action="login_failed",
            ip=_ip(request)))
        raise HTTPException(status_code=401,
                            detail="That email and password do not match.")

    get_audit_log().record(AuditEvent(
        timestamp=_now(), user=user.username, action="login", ip=_ip(request)))

    return {
        "access_token": create_token({"sub": user.username, "role": user.role}),
        "token_type": "bearer",
        "user": user.safe_dict(),
    }


@router.get("/me", summary="Who is signed in, and what they may do")
async def me(principal: Principal = Depends(current_principal)) -> dict:
    profile: dict = {"name": principal.name, "role": principal.role.value,
                     "domains": principal.domains,
                     "all_domains": principal.all_domains}
    if principal.key.startswith("jwt:"):
        user = get_user_store().find(principal.key[4:])
        if user is not None:
            profile.update(user.safe_dict())
    profile["can"] = {
        "read": True,
        "ingest": principal.at_least(Role.EDITOR),
        "manage_schemas": principal.at_least(Role.EDITOR),
        "manage_agents": principal.at_least(Role.EDITOR),
        "train": principal.at_least(Role.ADMIN),
        "manage_access": principal.at_least(Role.ADMIN),
    }
    return profile


@router.get("/users", summary="List users")
async def list_users(principal: Principal = Depends(current_principal)) -> list[dict]:
    if not principal.at_least(Role.ADMIN):
        raise HTTPException(status_code=403, detail="Administrators only.")
    return [u.safe_dict() for u in get_user_store().list()]


@router.post("/users", status_code=201, summary="Add a user")
async def create_user(body: NewUser, request: Request,
                      principal: Principal = Depends(current_principal)) -> dict:
    if not principal.at_least(Role.ADMIN):
        raise HTTPException(status_code=403, detail="Administrators only.")
    if body.role not in {r.value for r in Role}:
        raise HTTPException(status_code=400,
                            detail=f"Role must be one of: "
                                   f"{', '.join(r.value for r in Role)}.")
    store = get_user_store()
    if store.find(body.username) is not None:
        raise HTTPException(status_code=400, detail="That username already exists.")

    user = store.save(User(
        username=body.username, email=body.email or body.username,
        password_hash=hash_password(body.password),
        display_name=body.display_name or body.username.split("@")[0],
        role=body.role, domains=body.domains or [ALL_DOMAINS],
        created_at=_now()))

    get_audit_log().record(AuditEvent(
        timestamp=_now(), user=principal.name, action="create_user",
        resource=body.username, ip=_ip(request)))
    return user.safe_dict()


@router.delete("/users/{username}", status_code=204, summary="Remove a user")
async def delete_user(username: str, request: Request,
                      principal: Principal = Depends(current_principal)):
    # No return annotation on purpose. This module uses
    # `from __future__ import annotations`, which turns `-> None` into the
    # string "None"; FastAPI resolves that to NoneType, treats it as a response
    # model, and refuses it because 204 forbids a body.
    if not principal.at_least(Role.ADMIN):
        raise HTTPException(status_code=403, detail="Administrators only.")
    store = get_user_store()
    if store.find(username) is None:
        raise HTTPException(status_code=404, detail="No such user.")
    # Removing the account you are signed in as locks you out of the instance.
    if principal.key == f"jwt:{username}":
        raise HTTPException(status_code=400,
                            detail="You cannot remove your own account.")
    admins = [u for u in store.list() if u.role == Role.ADMIN.value and u.is_active]
    if len(admins) <= 1 and any(u.username == username for u in admins):
        raise HTTPException(status_code=400,
                            detail="This is the last administrator. Add another "
                                   "before removing this one.")
    store.delete(username)
    get_audit_log().record(AuditEvent(
        timestamp=_now(), user=principal.name, action="delete_user",
        resource=username, ip=_ip(request)))


@router.get("/audit", summary="Recent activity")
async def audit(limit: int = 100, user: str | None = None,
                principal: Principal = Depends(current_principal)) -> list[dict]:
    if not principal.at_least(Role.ADMIN):
        raise HTTPException(status_code=403, detail="Administrators only.")
    return get_audit_log().recent(limit=min(limit, 500), user=user)
