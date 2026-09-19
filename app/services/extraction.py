from __future__ import annotations

import asyncio
import logging

from ..config import get_settings
from ..ontology import get_ontology
from ..retrieval import get_chunk_index
from ..stores import get_store
from ..usage.recorder import attribute_to
from ..stores.networkx_store import NetworkXStore
from .jobs import Job, JobState
from .llm import complete_json
from .parsing import Chunk, chunk_file

log = logging.getLogger(__name__)


async def run_ingestion(job: Job, uploads: list[tuple[str, bytes]]) -> None:
    """Extract a knowledge graph from uploaded files.

    Chunks are processed concurrently under a semaphore, and every chunk emits a
    trace event the moment it lands — so a long run is observable from the first
    second instead of silent until the end.
    """
    settings = get_settings()
    store = get_store()

    try:
        ontology = get_ontology(job.domain)
    except KeyError as exc:
        job.state, job.error = JobState.FAILED, str(exc)
        job.finished_at = _now()
        job.emit("error", str(exc))
        return

    job.state = JobState.RUNNING
    job.started_at = _now()

    # ---- parse ---------------------------------------------------------
    chunks: list[Chunk] = []
    for filename, data in uploads:
        try:
            parsed = chunk_file(
                filename,
                data,
                rows_per_chunk=settings.rows_per_chunk,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
            chunks.extend(parsed)
            job.emit("info", f"Parsed {filename} into {len(parsed)} chunks", file=filename,
                     chunks=len(parsed))
        except Exception as exc:
            job.counters["chunk_errors"] += 1
            job.emit("error", f"Could not read {filename}: {exc}", file=filename)

    job.total = len(chunks)
    if not job.total:
        job.state = JobState.FAILED
        job.error = "No readable content was found in the uploaded files."
        job.finished_at = _now()
        job.emit("error", job.error)
        return

    job.emit("info", f"Extracting from {job.total} chunks with concurrency "
                     f"{settings.extraction_concurrency}", total=job.total)

    system_prompt = ontology.extraction_prompt()
    semaphore = asyncio.Semaphore(settings.extraction_concurrency)

    # Capture the same chunks the graph is extracted from, so local passage
    # search costs no extra pass over the files.
    captured: list[dict[str, str]] = []

    async def process(chunk: Chunk) -> None:
        if job.cancelled:
            return
        async with semaphore:
            if job.cancelled:
                return
            try:
                with attribute_to("extraction", domain=job.domain):
                    raw = await complete_json(system_prompt, chunk.text)
            except Exception as exc:
                job.counters["chunk_errors"] += 1
                job.done += 1
                job.emit("error", f"Chunk {chunk.index + 1} failed: {type(exc).__name__}",
                         chunk=chunk.index + 1, detail=str(exc)[:200])
                return

            result = ontology.validate(raw)

            nodes = result.nodes
            if settings.resolve_entities and nodes:
                from .resolution import resolve_batch
                try:
                    nodes, decisions = await resolve_batch(
                        job.domain, nodes,
                        adjudicate=settings.resolve_adjudicate)
                    merged = sum(1 for d in decisions
                                 if d.decision == "merge" and d.existing != d.incoming)
                    review = sum(1 for d in decisions if d.decision == "review")
                    if merged or review:
                        job.emit("info",
                                 f"Resolved {merged} into existing entities"
                                 + (f", {review} need review" if review else ""),
                                 merged=merged, review=review)
                except Exception as exc:
                    # Resolution is an improvement, not a gate. If it fails the
                    # nodes still go in under their own names.
                    log.warning("Entity resolution skipped for this chunk: %s", exc)

            added_n, added_e = await store.upsert(job.domain, nodes, result.edges)

            if settings.capture_passages:
                captured.append({"source": chunk.source, "text": chunk.text[:4000]})

            job.counters["nodes_added"] += added_n
            job.counters["edges_added"] += added_e
            job.counters["rejects"] += result.reject_count
            for rec in result.rejects[:5]:
                job.rejects.append(rec.model_dump())
            job.done += 1
            job.emit(
                "chunk",
                f"chunk {job.done}/{job.total} · +{added_n} nodes · +{added_e} edges"
                + (f" · {result.reject_count} rejected" if result.reject_count else ""),
                chunk=chunk.index + 1,
                source=chunk.source,
                nodes=added_n,
                edges=added_e,
                rejects=result.reject_count,
            )

    await asyncio.gather(*(process(c) for c in chunks))

    if isinstance(store, NetworkXStore):
        path = await store.persist(job.domain)
        job.emit("info", f"Graph snapshot written to {path.name}", artefact=path.name)

    if captured and not job.cancelled:
        try:
            added = await get_chunk_index().add(job.domain, captured)
            job.emit("info", f"{added} source passages kept for passage search.",
                     passages=added)
        except Exception as exc:
            job.emit("warning", f"Source passages could not be kept: {exc}")

    if job.cancelled:
        job.state = JobState.CANCELLED
        job.emit("info", "Ingestion cancelled.")
    else:
        job.state = JobState.SUCCEEDED
        stats = await store.stats(job.domain)
        job.emit("info",
                 f"Ingestion complete — graph now holds {stats['nodes']} nodes and "
                 f"{stats['edges']} edges.", **stats)
    job.finished_at = _now()


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
