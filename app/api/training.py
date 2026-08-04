from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..config import get_settings
from ..access import Principal, current_principal, admin
from ..training.datasets import DatasetBuilder, GenerationOptions
from ..training.registry import TrainingRegistry, build_trainers
from ..training.trainers import TrainingRequest

router = APIRouter(prefix="/training", tags=["training"],
                   dependencies=[Depends(admin)])


def _builder() -> DatasetBuilder:
    return DatasetBuilder(get_settings().data_dir)


def _registry() -> TrainingRegistry:
    return TrainingRegistry(get_settings().data_dir)


# ----------------------------------------------------------------- datasets
class GenerateRequest(BaseModel):
    domain: str
    include_facts: bool = True
    include_inverse: bool = True
    include_multihop: bool = True
    include_refusals: bool = True
    include_passages: bool = False
    max_per_relation: int = Field(default=400, ge=10, le=5000)
    refusal_fraction: float = Field(default=0.08, ge=0.0, le=0.3)
    validation_fraction: float = Field(default=0.05, ge=0.01, le=0.3)
    seed: int = 42
    system_prompt: str = ""
    exclude_types: list[str] = Field(default_factory=list)


@router.post("/datasets", status_code=201, summary="Build a training set from the graph")
async def generate_dataset(body: GenerateRequest) -> dict:
    options = GenerationOptions(
        include_facts=body.include_facts, include_inverse=body.include_inverse,
        include_multihop=body.include_multihop, include_refusals=body.include_refusals,
        include_passages=body.include_passages, max_per_relation=body.max_per_relation,
        refusal_fraction=body.refusal_fraction,
        validation_fraction=body.validation_fraction, seed=body.seed,
        system_prompt=body.system_prompt, exclude_types=body.exclude_types,
    )
    try:
        return await _builder().generate(body.domain, options)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/datasets", summary="Training sets built so far")
async def list_datasets() -> list[dict]:
    return _builder().list_datasets()


@router.get("/datasets/{dataset_id}", summary="Dataset detail with a sample")
async def dataset_detail(dataset_id: str, preview: int = 8) -> dict:
    b = _builder()
    try:
        meta = b.get(dataset_id)
        meta["preview"] = b.preview(dataset_id, "train", preview)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return meta


@router.get("/datasets/{dataset_id}/download", summary="Download a split as JSONL")
async def download_dataset(dataset_id: str, split: str = "train") -> Response:
    path = _builder().path(dataset_id, split)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No {split} split for this dataset.")
    return Response(
        content=path.read_bytes(), media_type="application/jsonl",
        headers={"Content-Disposition": f'attachment; filename="{dataset_id}.{split}.jsonl"'},
    )


@router.delete("/datasets/{dataset_id}", status_code=204, summary="Delete a training set")
async def delete_dataset(dataset_id: str) -> None:
    try:
        _builder().delete(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ----------------------------------------------------------------- providers
@router.get("/providers", summary="What each provider can actually do")
async def providers() -> list[dict]:
    out = []
    for key, trainer in build_trainers(get_settings()).items():
        caps = trainer.capabilities()
        out.append({
            "key": key,
            "can_train": caps.can_train,
            "can_deploy": caps.can_deploy,
            "hyperparameters": caps.hyperparameters,
            "base_models": caps.base_models,
            "requirements": caps.requirements,
            "note": caps.note,
        })
    return out


# ----------------------------------------------------------------- jobs
class StartRequest(BaseModel):
    provider: str
    dataset_id: str
    base_model: str
    suffix: str = ""
    epochs: int | None = Field(default=None, ge=1, le=10)
    learning_rate_multiplier: float | None = Field(default=None, gt=0, le=10)
    batch_size: int | None = Field(default=None, ge=1, le=64)
    learning_rate: float | None = Field(default=None, gt=0, le=1)
    seed: int | None = None
    # Adapter settings, honoured only where the trainer really configures one.
    lora_rank: int | None = Field(default=None, ge=1, le=256)
    lora_alpha: int | None = Field(default=None, ge=1, le=512)
    lora_dropout: float | None = Field(default=None, ge=0, le=0.5)
    quantization: str | None = None            # none | 4bit | 8bit
    max_seq_length: int | None = Field(default=None, ge=128, le=32768)


def _adapter(body: "StartRequest") -> dict | None:
    """Collect adapter settings, or nothing when none were supplied."""
    fields = {
        "rank": body.lora_rank, "alpha": body.lora_alpha, "dropout": body.lora_dropout,
        "quantization": body.quantization, "max_seq_length": body.max_seq_length,
    }
    present = {k: v for k, v in fields.items() if v is not None}
    return present or None


@router.post("/jobs", status_code=202, summary="Start a fine-tuning job")
async def start_job(body: StartRequest) -> dict:
    b = _builder()
    try:
        meta = b.get(body.dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    train_path = b.path(body.dataset_id, "train")
    validation_path = b.path(body.dataset_id, "validation")
    if not train_path.exists():
        raise HTTPException(status_code=400, detail="This dataset has no training split.")

    request = TrainingRequest(
        base_model=body.base_model, train_file=train_path,
        validation_file=validation_path if validation_path.exists() else None,
        suffix=body.suffix, epochs=body.epochs,
        learning_rate_multiplier=body.learning_rate_multiplier,
        learning_rate=body.learning_rate,
        batch_size=body.batch_size, seed=body.seed,
        adapter=_adapter(body),
    )
    try:
        return await _registry().start(body.provider, request,
                                       dataset_id=body.dataset_id, domain=meta["domain"])
    except (KeyError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs", summary="Training jobs, refreshed from the provider")
async def list_jobs() -> list[dict]:
    return await _registry().refresh_all()


@router.get("/jobs/{job_id}", summary="One training job")
async def job_detail(job_id: str) -> dict:
    try:
        return await _registry().refresh(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/cancel", summary="Cancel a training job")
async def cancel_job(job_id: str) -> dict:
    try:
        return await _registry().cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class DeployRequest(BaseModel):
    deployment_name: str = Field(min_length=1)


@router.post("/jobs/{job_id}/deploy", summary="Make the trained model available")
async def deploy_job(job_id: str, body: DeployRequest) -> dict:
    try:
        return await _registry().deploy(job_id, body.deployment_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/models", summary="Models produced here")
async def models() -> list[dict]:
    return [
        {
            "job_id": r["id"], "domain": r["domain"], "dataset_id": r["dataset_id"],
            "base_model": r["base_model"], "model_ref": r["model_ref"],
            "provider": r["provider"], "deployment": r.get("deployment"),
            "created_at": r["created_at"],
        }
        for r in _registry().list()
        if r.get("model_ref")
    ]
