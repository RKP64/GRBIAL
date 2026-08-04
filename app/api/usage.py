from fastapi import APIRouter, Depends, Query

from ..access import admin, viewer
from ..usage.recorder import get_recorder

router = APIRouter(prefix="/usage", tags=["usage"])


@router.get("", summary="What has been spent, and on what",
            dependencies=[Depends(viewer)])
async def summary(days: int = Query(30, ge=1, le=365)) -> dict:
    """Grouped by operation, model, domain and day.

    Costs are indicative: they come from a local price table rather than a live
    feed, so treat them as relative cost between features rather than a bill.
    """
    return get_recorder().summary(days)


@router.get("/prices", summary="The price table in use",
            dependencies=[Depends(admin)])
async def prices() -> dict:
    recorder = get_recorder()
    return {
        "prices": recorder.prices,
        "unit": "per million tokens",
        "override": "Place usage_prices.json in the data directory to change these.",
    }


@router.get("/events", summary="Individual calls, newest first",
            dependencies=[Depends(admin)])
async def events(days: int = Query(7, ge=1, le=90), limit: int = Query(200, ge=1, le=2000)) -> list[dict]:
    rows = get_recorder().events(days)
    rows.sort(key=lambda e: e.get("at", ""), reverse=True)
    return rows[:limit]
