"""Training jobs and the models they produce."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from .trainers import (BedrockTrainer, InferenceOnlyTrainer, OpenAICompatibleTrainer,
                       RemoteGPUTrainer, ShaktiTrainer, TogetherTrainer, Trainer,
                       TrainingRequest)

log = logging.getLogger(__name__)

TERMINAL = {"succeeded", "failed", "cancelled"}


def build_trainers(s: Settings) -> dict[str, Trainer]:
    """One entry per provider the deployment could plausibly use."""
    trainers: dict[str, Trainer] = {
        "hosted": OpenAICompatibleTrainer(s),
        "bedrock": BedrockTrainer(s),
        "your_gpu": RemoteGPUTrainer(s),
        "together": TogetherTrainer(s),
        "shakti": ShaktiTrainer(s),
    }
    if s.groq_api_key:
        trainers["groq"] = InferenceOnlyTrainer(s, "Groq")
    return trainers


class TrainingRegistry:
    """Job records on disk, so a restart does not lose a running job."""

    def __init__(self, data_dir: Path) -> None:
        self.dir = Path(data_dir) / "training"
        self.dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- io
    def _path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.json"

    def _save(self, record: dict) -> dict:
        self._path(record["id"]).write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    def get(self, job_id: str) -> dict:
        path = self._path(job_id)
        if not path.exists():
            raise KeyError(f"No training job '{job_id}'.")
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        out = []
        for path in sorted(self.dir.glob("*.json"), reverse=True):
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return sorted(out, key=lambda r: r.get("created_at", ""), reverse=True)

    # ---------------------------------------------------------------- jobs
    async def start(self, provider: str, request: TrainingRequest, *,
                    dataset_id: str, domain: str) -> dict:
        trainers = build_trainers(get_settings())
        trainer = trainers.get(provider)
        if trainer is None:
            raise KeyError(f"Unknown training provider '{provider}'.")
        caps = trainer.capabilities()
        if not caps.can_train:
            raise RuntimeError(caps.note or f"{provider} cannot run training jobs.")

        record = {
            "id": uuid.uuid4().hex[:12],
            "provider": provider,
            "dataset_id": dataset_id,
            "domain": domain,
            "base_model": request.base_model,
            "suffix": request.suffix,
            "state": "starting",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "job_ref": None,
            "model_ref": None,
            "error": None,
            "hyperparameters": {
                "epochs": request.epochs,
                "learning_rate_multiplier": request.learning_rate_multiplier,
                "learning_rate": request.learning_rate,
                "batch_size": request.batch_size,
                "seed": request.seed,
                "adapter": request.adapter,
            },
        }
        self._save(record)
        try:
            started = await trainer.start(request)
            record.update(job_ref=started.get("job_ref"),
                          state=started.get("state", "queued"))
        except Exception as exc:
            log.exception("Training job could not be started")
            record.update(state="failed", error=str(exc))
        return self._save(record)

    async def refresh(self, job_id: str) -> dict:
        record = self.get(job_id)
        if record["state"] in TERMINAL or not record.get("job_ref"):
            return record
        trainer = build_trainers(get_settings()).get(record["provider"])
        if trainer is None:
            return record
        try:
            status = await trainer.status(record["job_ref"])
            record.update(state=status.get("state", record["state"]),
                          model_ref=status.get("model_ref") or record.get("model_ref"),
                          error=status.get("error"))
            if status.get("trained_tokens"):
                record["trained_tokens"] = status["trained_tokens"]
        except Exception as exc:
            record["error"] = f"Could not read job status: {exc}"
        return self._save(record)

    async def refresh_all(self) -> list[dict]:
        records = self.list()
        pending = [r for r in records if r["state"] not in TERMINAL and r.get("job_ref")]
        if pending:
            await asyncio.gather(*(self.refresh(r["id"]) for r in pending),
                                 return_exceptions=True)
            records = self.list()
        return records

    async def cancel(self, job_id: str) -> dict:
        record = self.get(job_id)
        trainer = build_trainers(get_settings()).get(record["provider"])
        if trainer and record.get("job_ref"):
            try:
                await trainer.cancel(record["job_ref"])
            except Exception as exc:
                record["error"] = str(exc)
        record["state"] = "cancelled"
        return self._save(record)

    async def deploy(self, job_id: str, deployment_name: str) -> dict:
        record = self.get(job_id)
        if not record.get("model_ref"):
            raise RuntimeError("This job has not produced a model yet.")
        trainer = build_trainers(get_settings()).get(record["provider"])
        result = await trainer.deploy(record["model_ref"], deployment_name)
        record["deployment"] = result
        return self._save(record)
