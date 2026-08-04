"""Configured tool servers.

Kept as a small registry rather than environment variables because servers are
added and removed by operators during normal use, and because each one needs a
health state that outlives a single request.

A server that is unreachable is not removed and does not raise. Its tools simply
do not appear until it recovers — an agent that was working yesterday should
degrade rather than break because something else went down.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import get_settings
from .client import MCPClient, RemoteTool

log = logging.getLogger(__name__)

KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,32}$")


@dataclass
class ServerConfig:
    key: str
    name: str
    url: str
    token: str = ""
    enabled: bool = True
    timeout: float = 20.0
    created_at: str = ""

    def as_dict(self, *, reveal_token: bool = False) -> dict[str, Any]:
        return {"key": self.key, "name": self.name, "url": self.url,
                "token": (self.token if reveal_token
                          else ("set" if self.token else "")),
                "enabled": self.enabled, "timeout": self.timeout,
                "created_at": self.created_at}


class ServerRegistry:
    def __init__(self, data_dir: Path) -> None:
        self.path = Path(data_dir) / "mcp_servers.json"
        self._clients: dict[str, MCPClient] = {}

    def _read(self) -> list[ServerConfig]:
        if not self.path.exists():
            return []
        try:
            return [ServerConfig(**item)
                    for item in json.loads(self.path.read_text(encoding="utf-8"))]
        except (json.JSONDecodeError, TypeError):
            return []

    def _write(self, servers: list[ServerConfig]) -> None:
        self.path.write_text(
            json.dumps([s.as_dict(reveal_token=True) for s in servers], indent=2),
            encoding="utf-8")

    def list(self) -> list[ServerConfig]:
        return self._read()

    def get(self, key: str) -> ServerConfig:
        server = next((s for s in self._read() if s.key == key), None)
        if server is None:
            raise KeyError(f"There is no tool server called '{key}'.")
        return server

    def save(self, spec: dict[str, Any]) -> ServerConfig:
        key = str(spec.get("key", "")).strip().lower()
        if not KEY_RE.fullmatch(key):
            raise ValueError("Key must be 2-33 characters: lowercase letters, "
                             "digits, hyphen or underscore.")
        url = str(spec.get("url", "")).strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError("The address must be an http or https URL.")

        servers = self._read()
        existing = next((s for s in servers if s.key == key), None)
        token = str(spec.get("token") or "")
        if not token and existing:
            token = existing.token          # blank means "leave it alone"

        server = ServerConfig(
            key=key, name=str(spec.get("name") or key).strip(), url=url, token=token,
            enabled=bool(spec.get("enabled", True)),
            timeout=max(1.0, min(120.0, float(spec.get("timeout") or 20.0))),
            created_at=existing.created_at if existing
            else datetime.now(timezone.utc).isoformat(),
        )
        self._write([s for s in servers if s.key != key] + [server])
        self._clients.pop(key, None)
        return server

    def delete(self, key: str) -> None:
        servers = self._read()
        if not any(s.key == key for s in servers):
            raise KeyError(f"There is no tool server called '{key}'.")
        self._write([s for s in servers if s.key != key])
        self._clients.pop(key, None)

    def client(self, key: str) -> MCPClient:
        if key not in self._clients:
            config = self.get(key)
            self._clients[key] = MCPClient(config.key, config.url, token=config.token,
                                           timeout=config.timeout)
        return self._clients[key]

    # ---------------------------------------------------------------- tools
    async def tools(self) -> list[RemoteTool]:
        """Every tool from every reachable, enabled server.

        Servers are queried concurrently and independently: one that is down
        costs its own timeout, not everyone else's.
        """
        enabled = [s for s in self._read() if s.enabled]
        if not enabled:
            return []

        async def fetch(config: ServerConfig) -> list[RemoteTool]:
            try:
                return await asyncio.wait_for(self.client(config.key).list_tools(),
                                              timeout=config.timeout)
            except Exception as exc:
                log.warning("Tool server '%s' is unavailable: %s", config.key, exc)
                return []

        results = await asyncio.gather(*(fetch(s) for s in enabled))
        return [tool for group in results for tool in group]

    async def call(self, qualified: str, arguments: dict[str, Any]) -> str:
        server_key, _, tool_name = qualified.partition("__")
        if not tool_name:
            return f"'{qualified}' is not a recognised tool."
        try:
            config = self.get(server_key)
        except KeyError:
            return f"No tool server called '{server_key}' is configured."
        if not config.enabled:
            return f"The '{config.name}' tool server is currently turned off."
        try:
            return await asyncio.wait_for(
                self.client(server_key).call(tool_name, arguments),
                timeout=config.timeout)
        except asyncio.TimeoutError:
            return (f"The '{config.name}' server did not respond within "
                    f"{config.timeout:.0f} seconds.")
        except Exception as exc:
            return f"The '{config.name}' server could not run that: {exc}"

    async def health(self) -> list[dict[str, Any]]:
        async def check(config: ServerConfig) -> dict[str, Any]:
            if not config.enabled:
                return {**config.as_dict(), "reachable": False, "tools": 0,
                        "detail": "Turned off."}
            state = await self.client(config.key).health()
            return {**config.as_dict(), "reachable": state.reachable,
                    "tools": state.tools, "detail": state.detail}

        servers = self._read()
        if not servers:
            return []
        return list(await asyncio.gather(*(check(s) for s in servers)))


_registry: ServerRegistry | None = None


def get_server_registry() -> ServerRegistry:
    global _registry
    if _registry is None:
        _registry = ServerRegistry(get_settings().data_dir)
    return _registry
