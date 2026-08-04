import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..access import Principal, current_principal, viewer
from ..services.jobs import registry

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[Depends(viewer)])


@router.get("", summary="Recent jobs")
async def list_jobs(limit: int = 50) -> list[dict]:
    return registry.list(limit)


@router.get("/{job_id}", summary="Job status")
async def get_job(job_id: str) -> dict:
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="No job with that id.")
    payload = job.summary()
    payload["trace"] = [e.as_dict() for e in list(job.trace)[-100:]]
    payload["rejects"] = list(job.rejects)[-50:]
    return payload


@router.post("/{job_id}/cancel", summary="Cancel a running job")
async def cancel_job(job_id: str) -> dict:
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="No job with that id.")
    job.cancel()
    return {"job_id": job.id, "state": job.state.value, "cancelling": True}


@router.get("/{job_id}/stream", summary="Live progress (server-sent events)")
async def stream_job(job_id: str) -> StreamingResponse:
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="No job with that id.")

    async def gen():
        async for message in registry.stream(job):
            yield f"data: {json.dumps(message)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
