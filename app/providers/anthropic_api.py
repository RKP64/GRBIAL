from __future__ import annotations

import logging

from ..config import Settings
from .base import LLMProvider

log = logging.getLogger(__name__)

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    ANTHROPIC_AVAILABLE = False


class AnthropicProvider(LLMProvider):
    """Anthropic's API directly.

    The Messages API takes the system prompt as a top-level argument rather than
    a message, and requires max_tokens — both handled here.

    Anthropic does not offer an embedding model. Meaning-based search therefore
    needs an embedding source configured separately; without one the platform
    still works, using keyword matching for entities and passages. That is a
    real limitation and the readiness endpoint reports it rather than failing
    at the point of use.
    """

    name = "anthropic"

    def __init__(self, s: Settings) -> None:
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError("The anthropic package is not installed.")
        if not s.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self.model = s.anthropic_model
        self.max_tokens = s.anthropic_max_tokens
        self.client = AsyncAnthropic(
            api_key=s.anthropic_api_key,
            base_url=s.anthropic_base_url or None,
            timeout=float(s.extraction_timeout_seconds),
            max_retries=1,
        )
        # Optional companion for embeddings, since Anthropic has none.
        self._embedder: LLMProvider | None = None
        if s.embedding_provider and s.embedding_provider.lower() != "none":
            try:
                self._embedder = _build_embedder(s)
            except Exception as exc:
                log.warning("Embedding provider unavailable: %s", exc)

    async def complete(self, system: str, user: str, *, temperature: float = 0.1,
                       json_mode: bool = False) -> str:
        prompt = user
        if json_mode:
            prompt = (f"{user}\n\nRespond with a single valid JSON object and nothing "
                      f"else. Do not wrap it in code fences or add commentary.")
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=system or None,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content
                       if getattr(block, "type", "") == "text")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedder is None:
            raise RuntimeError(
                "No embedding model is configured. Anthropic does not provide one — "
                "set EMBEDDING_PROVIDER and its credentials, or leave meaning-based "
                "search disabled and rely on keyword matching."
            )
        return await self._embedder.embed(texts)

    @property
    def embeddings_available(self) -> bool:
        return self._embedder is not None

    @property
    def tools_available(self) -> bool:
        return True

    async def converse(self, system: str, messages: list[dict], tools: list[dict],
                       *, temperature: float = 0.2) -> dict:
        payload = []
        for m in messages:
            if m["role"] == "tool":
                payload.append({"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": m["tool_call_id"],
                    "content": m["content"],
                }]})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for call in m["tool_calls"]:
                    blocks.append({"type": "tool_use", "id": call["id"],
                                   "name": call["name"], "input": call["arguments"]})
                payload.append({"role": "assistant", "content": blocks})
            else:
                payload.append({"role": m["role"], "content": m["content"]})

        response = await self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=temperature,
            system=system or None, messages=payload,
            tools=[{"name": t["name"], "description": t["description"],
                    "input_schema": t["input_schema"]} for t in tools],
        )
        text = "".join(b.text for b in response.content
                       if getattr(b, "type", "") == "text")
        calls = [{"id": b.id, "name": b.name, "arguments": dict(b.input or {})}
                 for b in response.content if getattr(b, "type", "") == "tool_use"]
        return {"text": text, "tool_calls": calls,
                "stop": "tool_use" if calls else "end"}


def _build_embedder(s: Settings) -> LLMProvider:
    """Embeddings can come from a different vendor than chat.

    Nothing requires one provider to do both, and Anthropic makes that explicit.
    """
    choice = s.embedding_provider.lower()
    if choice == "local":
        from .local_embeddings import LocalEmbeddingProvider

        return LocalEmbeddingProvider(s)
    if choice in ("bedrock", "aws"):
        from .bedrock import BedrockProvider

        return BedrockProvider(s)
    if choice in ("azure", "azure_openai"):
        from .azure_openai import AzureOpenAIProvider

        return AzureOpenAIProvider(s)
    if choice in ("openai", "openai_compatible"):
        from .openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(s)
    raise RuntimeError(f"Unknown embedding provider '{s.embedding_provider}'.")
