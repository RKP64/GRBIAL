"""Wrap a provider so every call is recorded.

A decorator rather than edits inside each provider: the accounting concern stays
in one file, and a provider added later is instrumented by the factory without
its author thinking about it.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from ..providers.base import LLMProvider
from .recorder import UsageEvent, current_attribution, get_recorder


def _estimate(text: str) -> int:
    """Roughly four characters per token. Only used where a provider reports
    nothing, and always flagged so it is never mistaken for a measurement."""
    return max(1, len(text or "") // 4)


class MeteredProvider(LLMProvider):
    """Passes everything through, recording what it cost on the way."""

    def __init__(self, inner: LLMProvider) -> None:
        self.inner = inner
        self.name = inner.name

    # -------------------------------------------------------------- passthrough
    @property
    def embeddings_available(self) -> bool:
        return self.inner.embeddings_available

    @property
    def tools_available(self) -> bool:
        return self.inner.tools_available

    def describe(self) -> dict:
        return self.inner.describe()

    def __getattr__(self, item: str) -> Any:
        # Provider-specific extras (warm, client, model ids) stay reachable.
        return getattr(self.inner, item)

    # -------------------------------------------------------------- metered
    def _record(self, *, model: str, started: float, prompt: str, completion: str,
                reported: dict[str, int] | None, operation_default: str) -> None:
        operation, domain, principal = current_attribution()
        recorder = get_recorder()
        if reported and reported.get("input") is not None:
            input_tokens = int(reported.get("input") or 0)
            output_tokens = int(reported.get("output") or 0)
            estimated = False
        else:
            input_tokens = _estimate(prompt)
            output_tokens = _estimate(completion)
            estimated = True
        recorder.record(UsageEvent(
            at=datetime.now(timezone.utc).isoformat(),
            operation=operation if operation != "other" else operation_default,
            provider=self.name, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            estimated_tokens=estimated,
            cost=recorder.estimate_cost(model, input_tokens, output_tokens),
            duration_ms=int((time.perf_counter() - started) * 1000),
            domain=domain, principal=principal,
        ))

    def _model_name(self) -> str:
        for attr in ("model", "deployment", "model_id"):
            value = getattr(self.inner, attr, None)
            if value:
                return str(value)
        return self.name

    async def complete(self, system: str, user: str, *, temperature: float = 0.1,
                       json_mode: bool = False) -> str:
        started = time.perf_counter()
        result = await self.inner.complete(system, user, temperature=temperature,
                                           json_mode=json_mode)
        self._record(model=self._model_name(), started=started,
                     prompt=f"{system}\n{user}", completion=result,
                     reported=None, operation_default="answering")
        return result

    async def complete_json(self, system: str, user: str, *, temperature: float = 0.05) -> dict:
        started = time.perf_counter()
        result = await self.inner.complete_json(system, user, temperature=temperature)
        self._record(model=self._model_name(), started=started,
                     prompt=f"{system}\n{user}", completion=str(result),
                     reported=None, operation_default="extraction")
        return result

    async def embed(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        result = await self.inner.embed(texts)
        model = getattr(self.inner, "embedding_model_id", None) \
            or getattr(self.inner, "embedding_deployment", None) \
            or getattr(self.inner, "embedding_model", None) \
            or getattr(self.inner, "model_name", None) or "embedding"
        joined = "\n".join(texts)
        recorder = get_recorder()
        operation, domain, principal = current_attribution()
        tokens = _estimate(joined)
        recorder.record(UsageEvent(
            at=datetime.now(timezone.utc).isoformat(),
            operation="embedding", provider=self.name, model=str(model),
            input_tokens=tokens, output_tokens=0, estimated_tokens=True,
            cost=recorder.estimate_cost(str(model), tokens, 0),
            duration_ms=int((time.perf_counter() - started) * 1000),
            domain=domain, principal=principal,
            detail={"texts": len(texts)},
        ))
        return result

    async def converse(self, system: str, messages: list[dict], tools: list[dict],
                       *, temperature: float = 0.2) -> dict:
        started = time.perf_counter()
        result = await self.inner.converse(system, messages, tools,
                                           temperature=temperature)
        self._record(
            model=self._model_name(), started=started,
            prompt=system + "".join(str(m.get("content", "")) for m in messages),
            completion=result.get("text", "") + str(result.get("tool_calls", "")),
            reported=None, operation_default="agent",
        )
        return result
