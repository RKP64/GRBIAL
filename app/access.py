"""Who may do what.

Three roles, chosen because a fourth is rarely used and every extra role
multiplies the cases a reviewer has to reason about:

    viewer  — read the graph, ask questions, use agents
    editor  — the above, plus ingest documents and manage schemas, agents, teams
    admin   — the above, plus manage people and run training

Alongside the role, each principal carries a domain list. A viewer of the tax
domain cannot read the telecom graph, whatever the role would otherwise allow.
Domain scoping and role are checked separately because they answer different
questions: what kind of action, and over which data.

Principals are defined in one file so an operator can read the whole access
model at a glance. Production deployments should replace this with directory
groups; the enforcement points below do not change when they do.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import Depends, Header, HTTPException, status

from .config import get_settings

KEY_RE = re.compile(r"^[A-Za-z0-9_\-]{8,128}$")


class Role(str, Enum):
    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"


RANK = {Role.VIEWER: 1, Role.EDITOR: 2, Role.ADMIN: 3}

ALL_DOMAINS = "*"


@dataclass
class Principal:
    key: str
    name: str
    role: Role = Role.VIEWER
    domains: list[str] = field(default_factory=lambda: [ALL_DOMAINS])
    created_at: str = ""
    last_seen: str = ""

    def as_dict(self, *, reveal_key: bool = False) -> dict[str, Any]:
        return {
            "key": self.key if reveal_key else _mask(self.key),
            "name": self.name, "role": self.role.value, "domains": self.domains,
            "created_at": self.created_at, "last_seen": self.last_seen,
        }

    @property
    def all_domains(self) -> bool:
        return ALL_DOMAINS in self.domains

    def may_read(self, domain: str) -> bool:
        return self.all_domains or domain in self.domains

    def at_least(self, role: Role) -> bool:
        return RANK[self.role] >= RANK[role]


def _mask(key: str) -> str:
    return f"{key[:4]}…{key[-4:]}" if len(key) > 10 else "…"


class AccessStore:
    def __init__(self, data_dir: Path) -> None:
        self.path = Path(data_dir) / "principals.json"

    def _read(self) -> list[Principal]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        out = []
        for item in raw:
            try:
                item["role"] = Role(item.get("role", "viewer"))
                out.append(Principal(**item))
            except (TypeError, ValueError):
                continue
        return out

    def _write(self, principals: list[Principal]) -> None:
        self.path.write_text(
            json.dumps([
                {"key": p.key, "name": p.name, "role": p.role.value,
                 "domains": p.domains, "created_at": p.created_at,
                 "last_seen": p.last_seen}
                for p in principals
            ], indent=2),
            encoding="utf-8",
        )

    def list(self) -> list[Principal]:
        return self._read()

    def find(self, key: str) -> Principal | None:
        return next((p for p in self._read() if p.key == key), None)

    def save(self, spec: dict[str, Any]) -> Principal:
        key = str(spec.get("key", "")).strip()
        if not KEY_RE.fullmatch(key):
            raise ValueError(
                "A key must be 8-128 characters of letters, digits, hyphen or "
                "underscore. Generate a random one rather than choosing it."
            )
        name = str(spec.get("name", "")).strip()
        if not name:
            raise ValueError("Give this key a name, so it is clear who holds it.")
        raw_role = spec.get("role", "viewer")
        # Pydantic hands back a Role instance, an operator hands back a string,
        # and str(Role.VIEWER) is "Role.VIEWER" rather than "viewer" — so take
        # the value explicitly instead of stringifying.
        if isinstance(raw_role, Role):
            role = raw_role
        else:
            try:
                role = Role(str(raw_role).strip().lower())
            except ValueError as exc:
                raise ValueError("Role must be viewer, editor or admin.") from exc
        domains = [d for d in (spec.get("domains") or [ALL_DOMAINS]) if d] or [ALL_DOMAINS]

        principals = self._read()
        existing = next((p for p in principals if p.key == key), None)
        now = datetime.now(timezone.utc).isoformat()
        principal = Principal(
            key=key, name=name, role=role, domains=domains,
            created_at=existing.created_at if existing else now,
            last_seen=existing.last_seen if existing else "",
        )
        principals = [p for p in principals if p.key != key] + [principal]
        self._require_an_admin_remains(principals)
        self._write(principals)
        return principal

    @staticmethod
    def _require_an_admin_remains(principals: list[Principal]) -> None:
        """Refuse a change that would leave nobody able to administer.

        Configured keys count: they are always administrators and cannot be
        deleted through this API, so a deployment that has them can never be
        locked out and should not be blocked from ordinary edits.
        """
        if get_settings().api_key_set:
            return
        if not any(p.role is Role.ADMIN for p in principals):
            raise ValueError(
                "That would leave no administrator. Promote another key first."
            )

    def delete(self, key: str) -> None:
        principals = self._read()
        if not any(p.key == key for p in principals):
            raise KeyError("No such key.")
        remaining = [p for p in principals if p.key != key]
        self._require_an_admin_remains(remaining)
        self._write(remaining)

    def touch(self, key: str) -> None:
        principals = self._read()
        for p in principals:
            if p.key == key:
                p.last_seen = datetime.now(timezone.utc).isoformat()
                self._write(principals)
                return


def get_store() -> AccessStore:
    return AccessStore(get_settings().data_dir)


# ------------------------------------------------------------------ dependency
async def current_principal(x_api_key: str | None = Header(default=None)) -> Principal:
    """Resolve the caller.

    Keys listed in configuration are treated as administrators, so a deployment
    works before anyone has been added and cannot lock itself out by deleting
    the wrong record.
    """
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Provide an X-API-Key header.")
    settings = get_settings()
    if x_api_key in settings.api_key_set:
        return Principal(key=x_api_key, name="Configured key", role=Role.ADMIN,
                         domains=[ALL_DOMAINS])
    store = get_store()
    principal = store.find(x_api_key)
    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="That key is not recognised.")
    store.touch(x_api_key)
    return principal


def require(role: Role):
    """Dependency factory: this route needs at least `role`."""
    async def dependency(principal: Principal = Depends(current_principal)) -> Principal:
        if not principal.at_least(role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This needs {role.value} access. Yours is {principal.role.value}.",
            )
        return principal
    return dependency


def require_domain(principal: Principal, domain: str) -> None:
    """Called inside a route once the domain is known from the path or body."""
    if not principal.may_read(domain):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You do not have access to '{domain}'.",
        )


viewer = require(Role.VIEWER)
editor = require(Role.EDITOR)
admin = require(Role.ADMIN)
