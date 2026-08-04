from __future__ import annotations

import json

from typing import Any

from openai import AsyncAzureOpenAI

from ..config import Settings
from .base import LLMProvider


class AzureOpenAIProvider(LLMProvider):
    name = "azure_openai"

    def __init__(self, s: Settings) -> None:
        self.deployment = s.azure_openai_deployment
        self.embedding_deployment = s.azure_openai_embedding_deployment
        self.client = AsyncAzureOpenAI(
            azure_endpoint=s.azure_openai_endpoint,
            api_key=s.azure_openai_api_key,
            api_version=s.azure_openai_api_version,
            timeout=s.extraction_timeout_seconds,
            max_retries=1,
        )

    async def complete(self, system: str, user: str, *, temperature: float = 0.1,
                       json_mode: bool = False) -> str:
        kwargs: dict[str, Any] = {
            "model": self.deployment,
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
                model=self.embedding_deployment, input=texts[i : i + batch_size]
            )
            out.extend(d.embedding for d in resp.data)
        return out

    @property
    def embeddings_available(self) -> bool:
        return bool(self.embedding_deployment)

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
