from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..access import admin, editor, viewer
from ..agent.tools import available_tools
from ..mcp.registry import get_server_registry

router = APIRouter(prefix="/tool-servers", tags=["tools"])


class ServerIn(BaseModel):
    key: str
    name: str = ""
    url: str
    token: str = ""            # blank on update means "leave it unchanged"
    enabled: bool = True
    timeout: float = Field(default=20.0, ge=1, le=120)


@router.get("", summary="Configured tool servers and whether they answer",
            dependencies=[Depends(viewer)])
async def list_servers() -> list[dict]:
    return await get_server_registry().health()


@router.post("", status_code=201, summary="Add or update a tool server",
             dependencies=[Depends(admin)])
async def save_server(body: ServerIn) -> dict:
    registry = get_server_registry()
    try:
        server = registry.save(body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    state = await registry.client(server.key).health()
    return {**server.as_dict(), "reachable": state.reachable,
            "tools": state.tools, "detail": state.detail}


@router.delete("/{key}", status_code=204, summary="Remove a tool server",
               dependencies=[Depends(admin)])
async def delete_server(key: str) -> None:
    try:
        get_server_registry().delete(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/tools", summary="Every tool an agent could be given",
            dependencies=[Depends(viewer)])
async def all_tools() -> dict:
    """Built-in tools and those offered by reachable servers, in the form an
    agent's tool list expects."""
    tools = await available_tools()
    return {
        "tools": [{"name": t["name"], "description": t["description"],
                   "external": "__" in t["name"]} for t in tools],
    }
