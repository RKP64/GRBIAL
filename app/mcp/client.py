"""Client for external tool servers speaking the Model Context Protocol.

MCP is JSON-RPC over a transport. This implements the HTTP transport, which is
the one that makes sense for a server-side platform — stdio assumes the tool
runs as a child process on the same machine, which is not how a shared service
should reach a shared capability.

Three properties matter more than protocol completeness here:

  * A slow or dead server must not stall a conversation. Every call is
    timeout-shielded and failures come back as text the model can react to.
  * Tool names are namespaced on arrival. Two servers may both offer "search",
    and an agent must be able to tell them apart.
  * Discovery is cached but refreshable. Listing tools on every request would
    add a round trip to every turn; never refreshing would hide new tools.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"


@dataclass
class RemoteTool:
    server: str
    name: str                 # as the server calls it
    description: str
    input_schema: dict[str, Any]

    @property
    def qualified(self) -> str:
        """Namespaced so two servers offering the same name stay distinct."""
        return f"{self.server}__{self.name}"

    def as_schema(self) -> dict[str, Any]:
        return {
            "name": self.qualified,
            "description": f"[{self.server}] {self.description}",
            "input_schema": self.input_schema or {"type": "object", "properties": {}},
        }


@dataclass
class ServerHealth:
    reachable: bool
    tools: int = 0
    detail: str = ""
    checked_at: float = field(default_factory=time.time)


class MCPClient:
    """One client per configured server."""

    def __init__(self, key: str, url: str, *, token: str = "",
                 headers: dict[str, str] | None = None, timeout: float = 20.0) -> None:
        self.key = key
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._headers = {"Content-Type": "application/json",
                         "Accept": "application/json, text/event-stream"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._headers.update(headers or {})
        self._session_id: str | None = None
        self._initialised = False
        self._tools: list[RemoteTool] = []

    async def _rpc(self, method: str, params: dict[str, Any] | None = None,
                   *, notify: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method,
                                   "params": params or {}}
        if not notify:
            payload["id"] = int(time.time() * 1000) % 1_000_000
        headers = dict(self._headers)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            session = response.headers.get("Mcp-Session-Id")
            if session:
                self._session_id = session
            if notify or not response.content:
                return {}
            body = _parse(response.text)

        if "error" in body:
            error = body["error"]
            raise RuntimeError(f"{error.get('code')}: {error.get('message')}")
        return body.get("result", {})

    async def initialise(self) -> None:
        """Perform the handshake once.

        Guarded by a plain flag rather than a lock: a lock created here would
        bind to whichever event loop first built this client, and the client is
        cached across requests that each run their own loop. The handshake is
        idempotent, so a duplicate under concurrency costs one extra call and
        nothing else — far cheaper than a lock that fails in a later request.
        """
        if self._initialised:
            return
        await self._rpc("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "knowledge-graph-platform", "version": "1.1.0"},
        })
        try:
            await self._rpc("notifications/initialized", notify=True)
        except Exception:
            # Optional in practice; a server that rejects it still works.
            pass
        self._initialised = True

    async def list_tools(self, *, refresh: bool = False) -> list[RemoteTool]:
        if self._tools and not refresh:
            return self._tools
        await self.initialise()
        result = await self._rpc("tools/list")
        self._tools = [
            RemoteTool(server=self.key, name=t.get("name", ""),
                       description=t.get("description", ""),
                       input_schema=t.get("inputSchema") or t.get("input_schema") or {})
            for t in result.get("tools", []) if t.get("name")
        ]
        return self._tools

    async def call(self, tool: str, arguments: dict[str, Any]) -> str:
        await self.initialise()
        result = await self._rpc("tools/call", {"name": tool, "arguments": arguments})
        return _render(result)

    async def health(self) -> ServerHealth:
        try:
            tools = await asyncio.wait_for(self.list_tools(refresh=True),
                                           timeout=self.timeout)
            return ServerHealth(reachable=True, tools=len(tools))
        except Exception as exc:
            return ServerHealth(reachable=False, detail=str(exc)[:200])


def _parse(text: str) -> dict[str, Any]:
    """Accept a plain JSON body or a single server-sent event carrying one."""
    import json

    stripped = text.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    for line in stripped.splitlines():
        if line.startswith("data:"):
            candidate = line[5:].strip()
            if candidate.startswith("{"):
                return json.loads(candidate)
    raise RuntimeError("The server returned something that was not a JSON-RPC response.")


def _render(result: dict[str, Any], limit: int = 4000) -> str:
    """Flatten a tool result into text the model can read.

    Content blocks vary by server — text, images, embedded resources. Anything
    that is not text is described rather than dropped, so the model knows
    something was returned even when it cannot read it.
    """
    import json

    parts: list[str] = []
    for block in result.get("content", []) or []:
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind == "resource":
            resource = block.get("resource", {})
            parts.append(resource.get("text")
                         or f"[resource: {resource.get('uri', 'unnamed')}]")
        elif kind:
            parts.append(f"[{kind} returned, which cannot be shown as text]")
    if not parts and result:
        parts.append(json.dumps(result)[:limit])
    text = "\n".join(p for p in parts if p).strip() or "(the tool returned nothing)"
    if result.get("isError"):
        text = f"The tool reported an error: {text}"
    return text if len(text) <= limit else text[:limit] + "\n… truncated"
