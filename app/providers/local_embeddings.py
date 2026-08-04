from __future__ import annotations

import asyncio
import logging
import threading

from ..config import Settings
from .base import LLMProvider

log = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class LocalEmbeddingProvider(LLMProvider):
    """Embeddings from a sentence-transformers model running in this process.

    Chosen when embeddings should cost nothing per call and no text should leave
    the host — which also makes it the natural companion to a chat provider that
    has no embedding model of its own.

    The model is loaded lazily and once: importing this module must stay cheap,
    and a first request should not race a second into loading it twice.
    """

    name = "local_embeddings"

    def __init__(self, s: Settings) -> None:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it, or choose a "
                "different embedding source."
            )
        self.model_name = s.local_embedding_model
        self.device = s.local_embedding_device or None
        self.batch_size = s.local_embedding_batch_size
        self.normalize = s.local_embedding_normalize
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    log.info("Loading embedding model '%s' (first use may download it).",
                             self.model_name)
                    self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    async def warm(self) -> dict:
        """Load the model ahead of first use, so an ingestion run does not stall
        on a download partway through."""
        def run() -> dict:
            model = self._load()
            return {"model": self.model_name,
                    "dimensions": model.get_sentence_embedding_dimension()}
        return await asyncio.to_thread(run)

    async def complete(self, system: str, user: str, *, temperature: float = 0.1,
                       json_mode: bool = False) -> str:
        raise RuntimeError(
            "This provider supplies embeddings only. Set LLM_PROVIDER to a chat "
            "provider and keep this one as EMBEDDING_PROVIDER."
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        def run() -> list[list[float]]:
            model = self._load()
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [v.tolist() for v in vectors]

        # Encoding is CPU-bound and releases the GIL inside torch, so a worker
        # thread keeps the event loop responsive during a long index build.
        return await asyncio.to_thread(run)

    @property
    def embeddings_available(self) -> bool:
        return True

    def describe(self) -> dict:
        return {"provider": self.name, "model": self.model_name}
