"""Saved agents.

An agent is configuration, not code: which domains it may read, what it is for,
which tools it may use, and how hard it may work. Everything else — the loop,
the tools, verification — is shared.

The domain list is the important field. It is not a convenience for the user
picking a default; it is the boundary the loop enforces. An agent that chooses
its own next hop can otherwise wander across domains that a person would have
had to be granted access to, and a model deciding where to look is a harder
boundary to police after the fact than before.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..ontology import get_ontology
from .tools import TOOL_SCHEMAS

KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,48}$")
ALL_TOOLS = [t["name"] for t in TOOL_SCHEMAS]


@dataclass
class Agent:
    key: str
    name: str
    description: str = ""
    domains: list[str] = field(default_factory=list)
    system_prompt: str = ""
    tools: list[str] = field(default_factory=lambda: list(ALL_TOOLS))
    max_steps: int = 6
    temperature: float = 0.2
    verify: bool = True
    starters: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "name": self.name, "description": self.description,
            "domains": self.domains, "system_prompt": self.system_prompt,
            "tools": self.tools, "max_steps": self.max_steps,
            "temperature": self.temperature, "verify": self.verify,
            "starters": self.starters,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }

    def may_read(self, domain: str) -> bool:
        return domain in self.domains

    def default_domain(self) -> str:
        return self.domains[0] if self.domains else ""


class AgentRegistry:
    def __init__(self, data_dir: Path) -> None:
        self.dir = Path(data_dir) / "agents"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def list(self) -> list[Agent]:
        out: list[Agent] = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                out.append(Agent(**json.loads(path.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, TypeError):
                continue
        return out

    def get(self, key: str) -> Agent:
        path = self._path(key)
        if not path.exists():
            raise KeyError(f"There is no agent called '{key}'.")
        return Agent(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, spec: dict[str, Any]) -> Agent:
        key = str(spec.get("key", "")).strip().lower()
        if not KEY_RE.fullmatch(key):
            raise ValueError(
                "Key must be 2-49 characters: lowercase letters, digits, hyphen "
                "or underscore."
            )
        domains = [d for d in (spec.get("domains") or []) if d]
        if not domains:
            raise ValueError("An agent must be given at least one domain to read.")
        for domain in domains:
            try:
                get_ontology(domain)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc

        # External tool names are not validated against a fixed list: a server
        # may be registered after the agent, and removing a server should not
        # invalidate agents that referenced it.
        requested = spec.get("tools") or ALL_TOOLS
        tools = [t for t in requested if t in ALL_TOOLS or "__" in t]
        if not tools:
            raise ValueError("An agent must be allowed at least one tool.")

        now = datetime.now(timezone.utc).isoformat()
        existing = None
        if self._path(key).exists():
            try:
                existing = self.get(key)
            except Exception:
                existing = None

        agent = Agent(
            key=key,
            name=str(spec.get("name") or key).strip(),
            description=str(spec.get("description") or "").strip(),
            domains=domains,
            system_prompt=str(spec.get("system_prompt") or "").strip(),
            tools=tools,
            max_steps=max(1, min(12, int(spec.get("max_steps") or 6))),
            temperature=max(0.0, min(1.0, float(spec.get("temperature") or 0.2))),
            verify=bool(spec.get("verify", True)),
            starters=[s for s in (spec.get("starters") or []) if str(s).strip()][:6],
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        self._path(key).write_text(json.dumps(agent.as_dict(), indent=2), encoding="utf-8")
        return agent

    def delete(self, key: str) -> None:
        path = self._path(key)
        if not path.exists():
            raise KeyError(f"There is no agent called '{key}'.")
        path.unlink()


def get_registry() -> AgentRegistry:
    return AgentRegistry(get_settings().data_dir)
