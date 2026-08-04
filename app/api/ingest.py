import asyncio

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..ontology import get_ontology
from ..access import Principal, current_principal, editor
from ..services.extraction import run_ingestion
from ..services.jobs import registry
from ..services.parsing import SUPPORTED

router = APIRouter(prefix="/ingest", tags=["ingest"],
                   dependencies=[Depends(editor)])

MAX_BYTES = 200 * 1024 * 1024


@router.post("", status_code=202, summary="Start an extraction job")
async def start_ingestion(
    domain: str = Form(...),
    files: list[UploadFile] = File(...),
) -> dict:
    try:
        get_ontology(domain)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    uploads: list[tuple[str, bytes]] = []
    total = 0
    for upload in files:
        ext = "." + (upload.filename or "").rsplit(".", 1)[-1].lower()
        if ext not in SUPPORTED:
            raise HTTPException(
                status_code=400,
                detail=f"{upload.filename}: unsupported type. Supported: {sorted(SUPPORTED)}",
            )
        data = await upload.read()
        total += len(data)
        if total > MAX_BYTES:
            raise HTTPException(status_code=413, detail="Upload exceeds the 200 MB limit.")
        uploads.append((upload.filename or "upload", data))

    job = registry.create("ingest", domain, [u[0] for u in uploads])
    asyncio.create_task(run_ingestion(job, uploads))
    return {"job_id": job.id, "state": job.state.value,
            "poll": f"/jobs/{job.id}", "stream": f"/jobs/{job.id}/stream"}
