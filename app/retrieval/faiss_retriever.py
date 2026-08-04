from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..services.llm import embed_texts
from ..stores.base import GraphStore
from .base import Retriever

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    FAISS_AVAILABLE = False


class FaissRetriever(Retriever):
    """Semantic entry via in-process vector search.

    Answers questions with no lexical overlap ("somewhere to freshen up" finds
    Restroom / Shower). Falls back to keyword when the index is missing.
    """

    name = "semantic"

    def __init__(self, store: GraphStore, data_dir: Path, min_score: float = 0.2) -> None:
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss-cpu is not installed")
        self.store = store
        self.data_dir = Path(data_dir)
        self.min_score = min_score
        self._cache: dict[str, tuple] = {}

    def _paths(self, domain: str) -> tuple[Path, Path]:
        return self.data_dir / f"{domain}.faiss", self.data_dir / f"{domain}.ids.json"

    async def build(self, domain: str) -> dict:
        pairs = await self.store.node_texts(domain)
        if not pairs:
            return {"retriever": self.name, "indexed": 0, "note": "graph is empty"}
        ids = [p[0] for p in pairs]
        vecs = await embed_texts([p[1] for p in pairs])
        arr = np.asarray(vecs, dtype="float32")
        faiss.normalize_L2(arr)
        index = faiss.IndexFlatIP(arr.shape[1])
        index.add(arr)
        idx_path, ids_path = self._paths(domain)
        faiss.write_index(index, str(idx_path))
        ids_path.write_text(json.dumps(ids), encoding="utf-8")
        self._cache[domain] = (index, ids)
        return {"retriever": self.name, "indexed": len(ids)}

    def _load(self, domain: str):
        if domain in self._cache:
            return self._cache[domain]
        idx_path, ids_path = self._paths(domain)
        if not (idx_path.exists() and ids_path.exists()):
            return None
        index = faiss.read_index(str(idx_path))
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        self._cache[domain] = (index, ids)
        return self._cache[domain]

    async def search(self, domain: str, query: str, top_k: int) -> list[str]:
        loaded = self._load(domain)
        if loaded is None:
            return []
        index, ids = loaded
        vecs = await embed_texts([query])
        q = np.asarray(vecs, dtype="float32")
        faiss.normalize_L2(q)
        scores, idxs = index.search(q, min(top_k, len(ids)))
        return [ids[i] for score, i in zip(scores[0], idxs[0]) if i != -1 and score >= self.min_score]
