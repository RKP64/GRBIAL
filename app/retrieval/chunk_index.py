"""Local passage index.

The third retrieval layer. It answers "which passage discusses this?" without
depending on an external search service, so a standalone or air-gapped
deployment is not limited to what the ontology happened to model.

It is deliberately separate from the entity index: that one finds *which entity*
a question is about, this one finds *what the source text said*. A deployment
can run either, both, or neither.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..services.llm import embed_texts

log = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    FAISS_AVAILABLE = False

_TOKEN = re.compile(r"[a-z0-9]+")


class ChunkIndex:
    """Passage store with hybrid scoring.

    Vector similarity finds passages that mean the same thing; a lexical score
    finds passages that use the same rare words. Combining them recovers the
    behaviour a managed hybrid search gives you, without the service.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------- paths
    def _vec_path(self, domain: str) -> Path:
        return self.data_dir / f"{domain}.chunks.faiss"

    def _meta_path(self, domain: str) -> Path:
        return self.data_dir / f"{domain}.chunks.json"

    def exists(self, domain: str) -> bool:
        return self._meta_path(domain).exists()

    # ------------------------------------------------------------- write
    async def add(self, domain: str, passages: list[dict[str, str]]) -> int:
        """Append passages captured during ingestion.

        Called from the extraction pipeline, so the passage index is built from
        the same chunks the graph was extracted from — no second pass over the
        files, no second cost.
        """
        if not passages:
            return 0
        meta = self._load_meta(domain)
        start = len(meta["passages"])
        meta["passages"].extend(passages)
        self._meta_path(domain).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        self._cache.pop(domain, None)
        return len(meta["passages"]) - start

    async def build_vectors(self, domain: str) -> dict[str, Any]:
        """Embed everything captured so far. Optional — lexical search works without it."""
        meta = self._load_meta(domain)
        passages = meta["passages"]
        if not passages:
            return {"indexed": 0, "vectors": False,
                    "note": "No passages have been captured for this domain yet."}
        if not FAISS_AVAILABLE:
            return {"indexed": len(passages), "vectors": False,
                    "note": "Passages are searchable by keyword; meaning-based "
                            "passage search is unavailable on this deployment."}
        vectors = await embed_texts([p["text"] for p in passages])
        arr = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(arr)
        index = faiss.IndexFlatIP(arr.shape[1])
        index.add(arr)
        faiss.write_index(index, str(self._vec_path(domain)))
        self._cache.pop(domain, None)
        return {"indexed": len(passages), "vectors": True}

    def clear(self, domain: str) -> None:
        for path in (self._vec_path(domain), self._meta_path(domain)):
            if path.exists():
                path.unlink()
        self._cache.pop(domain, None)

    # ------------------------------------------------------------- read
    def _load_meta(self, domain: str) -> dict[str, Any]:
        path = self._meta_path(domain)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("Passage store for '%s' was unreadable; starting fresh.", domain)
        return {"passages": []}

    def _load(self, domain: str) -> dict[str, Any] | None:
        if domain in self._cache:
            return self._cache[domain]
        meta = self._load_meta(domain)
        if not meta["passages"]:
            return None
        index = None
        if FAISS_AVAILABLE and self._vec_path(domain).exists():
            try:
                index = faiss.read_index(str(self._vec_path(domain)))
                if index.ntotal != len(meta["passages"]):
                    # More passages were added than were embedded; fall back to
                    # lexical rather than returning misaligned results.
                    log.info("Passage vectors for '%s' are out of date.", domain)
                    index = None
            except Exception as exc:
                log.warning("Could not read passage vectors for '%s': %s", domain, exc)
        loaded = {"passages": meta["passages"], "index": index}
        self._cache[domain] = loaded
        return loaded

    def stats(self, domain: str) -> dict[str, Any]:
        loaded = self._load(domain)
        if not loaded:
            return {"passages": 0, "vectors": False}
        return {"passages": len(loaded["passages"]),
                "vectors": loaded["index"] is not None}

    @staticmethod
    def _lexical_scores(query: str, passages: list[dict[str, str]]) -> np.ndarray:
        terms = set(_TOKEN.findall(query.lower()))
        if not terms:
            return np.zeros(len(passages), dtype="float32")
        scores = np.zeros(len(passages), dtype="float32")
        for i, p in enumerate(passages):
            words = set(_TOKEN.findall(p["text"].lower()))
            if words:
                scores[i] = len(terms & words) / len(terms)
        return scores

    async def search(self, domain: str, query: str, k: int = 5,
                     vector_weight: float = 0.65) -> list[dict[str, Any]]:
        loaded = self._load(domain)
        if not loaded:
            return []
        passages = loaded["passages"]
        lexical = self._lexical_scores(query, passages)

        combined = lexical
        if loaded["index"] is not None:
            try:
                qv = np.asarray(await embed_texts([query]), dtype="float32")
                faiss.normalize_L2(qv)
                scores, idxs = loaded["index"].search(qv, min(len(passages), max(k * 4, 20)))
                semantic = np.zeros(len(passages), dtype="float32")
                for score, i in zip(scores[0], idxs[0]):
                    if i != -1:
                        semantic[i] = max(0.0, float(score))
                combined = vector_weight * semantic + (1 - vector_weight) * lexical
            except Exception as exc:
                log.warning("Passage vector search failed, using keywords: %s", exc)

        order = np.argsort(-combined)[: k * 3]
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for i in order:
            if combined[i] <= 0:
                continue
            passage = passages[int(i)]
            fingerprint = " ".join(passage["text"].split()).lower()[:200]
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            results.append({
                "title": passage.get("source", "Source document"),
                "url": "",
                "content": passage["text"],
                "score": float(combined[i]),
                "reranker_score": None,
            })
            if len(results) >= k:
                break
        return results
