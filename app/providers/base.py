"""Language-model provider seam.

Everything above this layer — extraction, answering, embeddings — calls three
functions and never learns which vendor is behind them. Adding a provider means
adding one class here and nothing else.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def complete(self, system: str, user: str, *, temperature: float = 0.1,
                       json_mode: bool = False) -> str:
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    async def complete_json(self, system: str, user: str, *,
                            temperature: float = 0.05) -> dict:
        """Structured output with tolerant parsing.

        Not every model honours a strict JSON mode, so the response is cleaned
        of code fences and surrounding prose before parsing. This is the one
        place that has to be forgiving, and it belongs here rather than in
        every provider.
        """
        raw = await self.complete(system, user, temperature=temperature, json_mode=True)
        return parse_json_response(raw)

    @property
    def embeddings_available(self) -> bool:
        return True

    @property
    def tools_available(self) -> bool:
        """Whether this provider can drive a tool-calling loop."""
        return False

    async def converse(self, system: str, messages: list[dict], tools: list[dict],
                       *, temperature: float = 0.2) -> dict:
        """One turn of a tool-calling conversation.

        Returns a provider-neutral shape so the agent loop never learns which
        vendor is behind it:

            {"text": str,
             "tool_calls": [{"id": str, "name": str, "arguments": dict}],
             "stop": "tool_use" | "end"}

        `messages` uses the same neutral shape: entries are
        {"role": "user"|"assistant", "content": str} or
        {"role": "tool", "tool_call_id": str, "content": str}.
        """
        raise RuntimeError(f"{self.name} does not support tool use.")

    def describe(self) -> dict:
        return {"provider": self.name}


def parse_json_response(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise
