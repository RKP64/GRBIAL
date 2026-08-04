from __future__ import annotations

import json

from typing import Any

from openai import AsyncOpenAI

from ..config import Settings
from .base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    """Any endpoint speaking the OpenAI chat-completions API.

    Covers hosted OpenAI, self-hosted model servers, and fine-tuned models
    deployed behind a compatible gateway — all with one base URL and key.
    """

    name = "openai_compatible"

    def __init__(self, s: Settings) -> None:
        self.model = s.openai_model
        self.embedding_model = s.openai_embedding_model
        self.client = AsyncOpenAI(
            base_url=s.openai_base_url or None,
            api_key=s.openai_api_key or "not-required",
            timeout=s.extraction_timeout_seconds,
            max_retries=1,
        )

    async def complete(self, system: str, user: str, *, temperature: float = 0.1,
                       json_mode: bool = False) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    async def embed(self, texts: list[str], batch_size: int = 256) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            resp = await self.client.embeddings.create(
                model=self.embedding_model, input=texts[i : i + batch_size]
            )
            out.extend(d.embedding for d in resp.data)
        return out

    @property
    def embeddings_available(self) -> bool:
        return bool(self.embedding_model)

    @property
    def tools_available(self) -> bool:
        return True

    async def converse(self, system: str, messages: list[dict], tools: list[dict],
                       *, temperature: float = 0.2) -> dict:
        payload: list[dict] = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m["role"] == "tool":
                payload.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                                "content": m["content"]})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                payload.append({
                    "role": "assistant",
                    "content": m.get("content") or None,
                    "tool_calls": [{
                        "id": c["id"], "type": "function",
                        "function": {"name": c["name"],
                                     "arguments": json.dumps(c["arguments"])},
                    } for c in m["tool_calls"]],
                })
            else:
                payload.append({"role": m["role"], "content": m["content"]})

        response = await self.client.chat.completions.create(
            model=self.deployment if hasattr(self, "deployment") else self.model,
            messages=payload, temperature=temperature,
            tools=[{"type": "function", "function": {
                "name": t["name"], "description": t["description"],
                "parameters": t["input_schema"]}} for t in tools],
        )
        choice = response.choices[0].message
        calls = []
        for c in (choice.tool_calls or []):
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": c.id, "name": c.function.name, "arguments": args})
        return {"text": choice.content or "", "tool_calls": calls,
                "stop": "tool_use" if calls else "end"}
