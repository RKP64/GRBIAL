"""Golden sets, evaluation runs, and comparison between them."""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import re

from fastapi import (APIRouter, Depends, File, Form, HTTPException, UploadFile)
from pydantic import BaseModel, Field

from ..access import Principal, current_principal, editor, viewer
from ..services import evals
from ..services.jobs import JobState, registry as jobs

log = logging.getLogger(__name__)

router = APIRouter(prefix="/evals", tags=["evals"], dependencies=[Depends(viewer)])

_RUNNING: set = set()   # strong references to in-flight evaluation runs

KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,48}$")
MAX_BYTES = 10 * 1024 * 1024

# Column names people actually use. Matched case-insensitively so a set exported
# from a spreadsheet does not need editing before it can be uploaded.
QUESTION_COLUMNS = ("question", "query", "prompt", "input")
EXPECTED_COLUMNS = ("expected", "answer", "expected_answer", "response",
                    "golden_answer", "ground_truth")


def _pick(row: dict, names: tuple[str, ...]) -> str:
    """Find a value by any of several column names.

    Headings are normalised before matching, because a spreadsheet exports
    "Expected Answer" where the code expects "expected_answer", and requiring
    the person to rename columns before uploading is a pointless obstacle.
    """
    def norm(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")

    lowered = {norm(k): v for k, v in row.items()}
    for name in names:
        if lowered.get(norm(name)) not in (None, ""):
            return str(lowered[norm(name)]).strip()
    return ""


@router.get("/sets", summary="Golden sets")
async def list_sets() -> list[dict]:
    return evals.list_sets()


@router.post("/sets", status_code=201, summary="Upload a golden set",
             dependencies=[Depends(editor)])
async def upload_set(key: str = Form(...), name: str = Form(""),
                     file: UploadFile = File(...)) -> dict:
    key = key.strip().lower()
    if not KEY_RE.fullmatch(key):
        raise HTTPException(status_code=400,
                            detail="Key must be 2-49 characters: lowercase "
                                   "letters, digits, hyphen or underscore.")

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="That file is over 10 MB.")

    filename = (file.filename or "").lower()
    rows: list[dict] = []
    try:
        if filename.endswith(".json"):
            parsed = json.loads(data.decode("utf-8"))
            rows = parsed if isinstance(parsed, list) else parsed.get("items", [])
        elif filename.endswith((".xlsx", ".xls")):
            import pandas as pd
            frame = pd.read_excel(io.BytesIO(data))
            rows = frame.fillna("").to_dict("records")
        else:
            text = data.decode("utf-8-sig", errors="replace")
            rows = list(csv.DictReader(io.StringIO(text)))
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"That file could not be read: {exc}") from exc

    items = [{"question": _pick(r, QUESTION_COLUMNS),
              "expected": _pick(r, EXPECTED_COLUMNS),
              "note": _pick(r, ("note", "comment"))}
             for r in rows if isinstance(r, dict)]

    try:
        return evals.save_set(key, name, items).summary()
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{exc} Columns found: {', '.join(str(k) for k in (rows[0] or {}))}"
                   if rows else str(exc)) from exc


@router.get("/sets/{key}", summary="One golden set")
async def get_set(key: str) -> dict:
    try:
        gset = evals.load_set(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {**gset.summary(),
            "questions": [{"question": i.question, "expected": i.expected}
                          for i in gset.items]}


@router.delete("/sets/{key}", status_code=204, summary="Remove a golden set",
               dependencies=[Depends(editor)])
async def remove_set(key: str):
    evals.delete_set(key)


class RunRequest(BaseModel):
    set_key: str
    agent_key: str = Field(min_length=1)


@router.post("/runs", status_code=202, summary="Run a golden set against an agent",
             dependencies=[Depends(editor)])
async def start_run(body: RunRequest,
                    principal: Principal = Depends(current_principal)) -> dict:
    """Start a run in the background.

    A hundred questions is minutes of work, so the request returns a job to
    follow rather than holding the connection open.
    """
    try:
        gset = evals.load_set(body.set_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # Fail now, with a clear message, rather than minutes later inside the job.
    try:
        from ..agent.registry import get_registry
        get_registry().get(body.agent_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    job = jobs.create("eval", body.agent_key)
    job.total = len(gset.items)
    job.state = JobState.RUNNING

    async def work() -> None:
        try:
            def progress(done: int, total: int, result) -> None:
                job.done = done
                job.emit("info", f"{done}/{total} · {result.verdict}",
                         question=result.question[:80], verdict=result.verdict)

            run = await evals.run_eval(body.set_key, body.agent_key,
                                       principal=principal, on_progress=progress)
            job.state = JobState.SUCCEEDED
            job.finished_at = run.finished_at
            s = run.scores()
            headline = (f"Accuracy {s['accuracy']:.0%} over {s['scored']} judged"
                        if s.get("accuracy") is not None
                        else "No question could be judged")
            if s.get("errors") or s.get("unjudged"):
                headline += (f" · {s.get('errors', 0)} failed, "
                             f"{s.get('unjudged', 0)} not judged")
            job.emit("info", headline, run_id=run.id, scores=s)
        except Exception as exc:
            job.state = JobState.FAILED
            job.error = str(exc)
            job.emit("error", f"Run failed: {exc}")
            log.exception("Evaluation run failed")

    # asyncio only keeps a weak reference to a task. Without holding one here,
    # a long run can be garbage-collected part-way and simply stop.
    task = asyncio.create_task(work())
    _RUNNING.add(task)
    task.add_done_callback(_RUNNING.discard)
    return {"job_id": job.id, "questions": len(gset.items)}


@router.get("/runs", summary="Past runs, newest first")
async def list_runs(limit: int = 50) -> list[dict]:
    return evals.list_runs(limit=limit)


@router.get("/runs/{run_id}", summary="One run, question by question")
async def get_run(run_id: str) -> dict:
    try:
        return evals.load_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/compare", summary="Difference between two runs")
async def compare(a: str, b: str) -> dict:
    """Which questions improved and which regressed.

    The per-question movement is the point: an unchanged headline accuracy can
    hide ten questions breaking and ten being fixed.
    """
    try:
        return evals.compare(evals.load_run(a), evals.load_run(b))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
