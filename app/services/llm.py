"""Language-model access.

A thin pass-through to the configured provider, kept so that call sites read the
same regardless of which cloud is behind them.
"""
from __future__ import annotations

import logging

from ..providers import get_embedder, get_provider

log = logging.getLogger(__name__)


async def complete(system: str, user: str, *, temperature: float = 0.1,
                   json_mode: bool = False) -> str:
    return await get_provider().complete(system, user, temperature=temperature,
                                         json_mode=json_mode)


async def complete_json(system: str, user: str, *, temperature: float = 0.05) -> dict:
    return await get_provider().complete_json(system, user, temperature=temperature)


async def embed_texts(texts: list[str], batch_size: int = 256) -> list[list[float]]:
    """Embed text using the configured embedding source, which may be a
    different provider than the one answering questions."""
    if not texts:
        return []
    return await get_embedder().embed(texts)
