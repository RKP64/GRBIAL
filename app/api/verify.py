from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..access import Principal, current_principal, viewer
from ..verification import verify_answer

router = APIRouter(prefix="/verify", tags=["verification"],
                   dependencies=[Depends(viewer)])


class VerifyRequest(BaseModel):
    domain: str
    answer: str = Field(min_length=1)
    question: str | None = None
    max_facts: int = Field(default=40, ge=5, le=200)


@router.post("", summary="Check an answer against the graph, claim by claim")
async def verify(body: VerifyRequest) -> dict:
    """Verify any text — including output from a system that is not this one.

    Useful for auditing an existing assistant against a governed graph without
    routing its traffic through this platform.
    """
    try:
        result = await verify_answer(body.domain, body.answer, max_facts=body.max_facts)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    payload = result.as_dict()
    payload["question"] = body.question
    payload["domain"] = body.domain
    return payload
