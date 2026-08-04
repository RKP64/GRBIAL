from __future__ import annotations

import logging
from functools import lru_cache

from ..config import get_settings
from .base import LLMProvider, parse_json_response  # noqa: F401

log = logging.getLogger(__name__)


@lru_cache
def get_provider() -> LLMProvider:
    """Build the configured provider.

    LLM_PROVIDER selects it. When left as `auto`, whichever set of credentials
    is present wins — so a deployment only has to fill in the block for the
    cloud it actually has.
    """
    s = get_settings()
    choice = (s.llm_provider or "auto").lower()

    if choice == "auto":
        if s.azure_openai_endpoint and s.azure_openai_api_key and s.azure_openai_deployment:
            choice = "azure"
        elif s.anthropic_api_key:
            choice = "anthropic"
        elif s.bedrock_model_id and (s.aws_region or s.aws_profile):
            choice = "bedrock"
        elif s.openai_model:
            choice = "openai"
        else:
            raise RuntimeError(
                "No language model is configured. Fill in the credentials for "
                "Anthropic, Azure OpenAI, Amazon Bedrock, or an OpenAI-compatible "
                "endpoint."
            )

    if choice in ("azure", "azure_openai"):
        from .azure_openai import AzureOpenAIProvider

        return _metered(AzureOpenAIProvider(s))
    if choice in ("bedrock", "aws"):
        from .bedrock import BedrockProvider

        return _metered(BedrockProvider(s))
    if choice == "anthropic":
        from .anthropic_api import AnthropicProvider

        return _metered(AnthropicProvider(s))
    if choice in ("openai", "openai_compatible"):
        from .openai_compatible import OpenAICompatibleProvider

        return _metered(OpenAICompatibleProvider(s))
    raise RuntimeError(
        f"Unknown provider '{choice}'. Use azure, bedrock, anthropic or openai."
    )


def _metered(provider: LLMProvider) -> LLMProvider:
    """Wrap for usage recording. Applied in the factory so a provider added
    later is measured without its author having to remember."""
    from ..usage.instrument import MeteredProvider

    return MeteredProvider(provider)


@lru_cache
def get_embedder() -> LLMProvider:
    """The embedding source.

    Falls back to the chat provider when no separate one is named, so a single
    well-configured provider needs no extra setting.
    """
    s = get_settings()
    choice = (s.embedding_provider or "none").lower()
    if choice == "none":
        return get_provider()
    if choice == "local":
        from .local_embeddings import LocalEmbeddingProvider

        return LocalEmbeddingProvider(s)
    if choice in ("bedrock", "aws"):
        from .bedrock import BedrockProvider

        return _metered(BedrockProvider(s))
    if choice in ("azure", "azure_openai"):
        from .azure_openai import AzureOpenAIProvider

        return _metered(AzureOpenAIProvider(s))
    if choice in ("openai", "openai_compatible"):
        from .openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(s)
    raise RuntimeError(f"Unknown embedding provider '{choice}'.")


def provider_status() -> dict:
    """Non-raising summary for the readiness endpoint."""
    try:
        provider = get_provider()
        return {"ready": True, "embeddings_ready": provider.embeddings_available}
    except Exception as exc:
        return {"ready": False, "embeddings_ready": False, "detail": str(exc)}
