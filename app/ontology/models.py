from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class NodeIn(BaseModel):
    """A node as proposed by the extractor, before schema validation."""

    id: str
    type: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("type"):
                data["type"] = data.get("entity_type") or data.get("label") or ""
            if not data.get("id"):
                data["id"] = data.get("name") or data.get("entity") or ""
        return data


class EdgeIn(BaseModel):
    """An edge as proposed by the extractor.

    Models routinely name the relation key `relation` or `rel` instead of
    `type`; all three are accepted so valid edges are never rejected over
    naming alone.
    """

    source: str
    target: str
    type: str = ""

    @model_validator(mode="before")
    @classmethod
    def _accept_relation_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("type"):
            data["type"] = data.get("relation") or data.get("rel") or data.get("label") or ""
        return data


class Extraction(BaseModel):
    nodes: list[NodeIn] = Field(default_factory=list)
    edges: list[EdgeIn] = Field(default_factory=list)


class RejectRecord(BaseModel):
    kind: Literal["node", "edge"]
    reason: str
    payload: dict[str, Any]


class ValidationResult(BaseModel):
    """What survived validation, and everything that did not."""

    nodes: list[NodeIn] = Field(default_factory=list)
    edges: list[EdgeIn] = Field(default_factory=list)
    rejects: list[RejectRecord] = Field(default_factory=list)

    @property
    def reject_count(self) -> int:
        return len(self.rejects)
