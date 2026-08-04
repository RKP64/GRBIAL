"""Fine-tuning back ends.

Not every provider can train, and pretending otherwise produces a UI that
offers buttons which cannot work. Each trainer therefore declares what it
actually supports, and the console renders only what is genuinely available.

Current reality, as of writing:

  * OpenAI-compatible (includes Azure OpenAI) — hosted supervised fine-tuning.
    The provider manages the hardware, so this is "serverless" from our side:
    upload two files, start a job, poll it, deploy the result.
  * Amazon Bedrock — hosted customisation. Training data must sit in S3 and the
    job needs an IAM role it can assume, so this trainer prepares and validates
    rather than silently failing later.
  * Inference-only providers (for example Groq) — cannot train. They can serve
    an adapter trained elsewhere, so they are offered for deployment only.

Adapter choice is likewise not free: hosted services expose a fixed set of
knobs (epochs, learning-rate multiplier, batch size) and choose the adapter
method themselves. Presenting a "LoRA rank" field the service ignores would be
dishonest, so only the knobs a given trainer really passes through are shown.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings

log = logging.getLogger(__name__)


@dataclass
class TrainerCapabilities:
    can_train: bool = False
    can_deploy: bool = False
    hyperparameters: list[str] = field(default_factory=list)
    base_models: list[str] = field(default_factory=list)
    note: str = ""
    requirements: list[str] = field(default_factory=list)


@dataclass
class TrainingRequest:
    base_model: str
    train_file: Path
    validation_file: Path | None
    suffix: str = ""
    epochs: int | None = None
    # Hosted services take a multiplier against their own tuned rate; a worker
    # you control takes an absolute rate. Both are carried so neither has to be
    # faked into the other's units.
    learning_rate_multiplier: float | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    seed: int | None = None
    # Only meaningful where the adapter is genuinely configurable.
    adapter: dict | None = None


class Trainer(ABC):
    name: str = "base"

    @abstractmethod
    def capabilities(self) -> TrainerCapabilities:
        ...

    async def start(self, request: TrainingRequest) -> dict[str, Any]:
        raise RuntimeError(f"{self.name} cannot run training jobs.")

    async def status(self, job_ref: str) -> dict[str, Any]:
        raise RuntimeError(f"{self.name} cannot report training jobs.")

    async def cancel(self, job_ref: str) -> dict[str, Any]:
        raise RuntimeError(f"{self.name} cannot cancel training jobs.")

    async def deploy(self, model_ref: str, deployment_name: str) -> dict[str, Any]:
        raise RuntimeError(f"{self.name} cannot deploy models.")


class OpenAICompatibleTrainer(Trainer):
    """Hosted supervised fine-tuning over the OpenAI fine-tuning API.

    Azure OpenAI implements the same surface, so one trainer covers both. The
    provider owns the GPUs; we upload, start, and poll.
    """

    name = "openai_compatible"

    def __init__(self, s: Settings) -> None:
        self.settings = s
        self._client = None

    def _get_client(self):
        if self._client is None:
            if self.settings.training_provider_mode == "azure":
                from openai import AsyncAzureOpenAI

                self._client = AsyncAzureOpenAI(
                    azure_endpoint=self.settings.azure_openai_endpoint,
                    api_key=self.settings.azure_openai_api_key,
                    api_version=self.settings.azure_finetune_api_version,
                    timeout=120.0,
                )
            else:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(
                    base_url=self.settings.openai_base_url or None,
                    api_key=self.settings.openai_api_key,
                    timeout=120.0,
                )
        return self._client

    def capabilities(self) -> TrainerCapabilities:
        configured = bool(
            (self.settings.azure_openai_endpoint and self.settings.azure_openai_api_key)
            or self.settings.openai_api_key
        )
        models = [m.strip() for m in self.settings.training_base_models.split(",") if m.strip()]
        return TrainerCapabilities(
            can_train=configured,
            can_deploy=self.settings.training_provider_mode == "azure" and configured,
            hyperparameters=["epochs", "learning_rate_multiplier", "batch_size", "seed", "suffix"],
            base_models=models,
            note="" if configured else "Add the credentials for this provider to enable training.",
            requirements=["A base model that the provider permits fine-tuning on"],
        )

    async def start(self, request: TrainingRequest) -> dict[str, Any]:
        client = self._get_client()
        train = await client.files.create(
            file=open(request.train_file, "rb"), purpose="fine-tune")
        validation = None
        if request.validation_file and Path(request.validation_file).exists():
            validation = await client.files.create(
                file=open(request.validation_file, "rb"), purpose="fine-tune")

        hyper: dict[str, Any] = {}
        if request.epochs:
            hyper["n_epochs"] = request.epochs
        if request.learning_rate_multiplier:
            hyper["learning_rate_multiplier"] = request.learning_rate_multiplier
        if request.batch_size:
            hyper["batch_size"] = request.batch_size

        kwargs: dict[str, Any] = {
            "model": request.base_model,
            "training_file": train.id,
        }
        if validation:
            kwargs["validation_file"] = validation.id
        if hyper:
            kwargs["hyperparameters"] = hyper
        if request.suffix:
            kwargs["suffix"] = request.suffix
        if request.seed is not None:
            kwargs["seed"] = request.seed

        job = await client.fine_tuning.jobs.create(**kwargs)
        return {"job_ref": job.id, "state": job.status,
                "base_model": request.base_model, "raw": _safe(job)}

    async def status(self, job_ref: str) -> dict[str, Any]:
        job = await self._get_client().fine_tuning.jobs.retrieve(job_ref)
        return {
            "job_ref": job.id,
            "state": job.status,
            "model_ref": getattr(job, "fine_tuned_model", None),
            "error": getattr(getattr(job, "error", None), "message", None),
            "trained_tokens": getattr(job, "trained_tokens", None),
            "raw": _safe(job),
        }

    async def cancel(self, job_ref: str) -> dict[str, Any]:
        job = await self._get_client().fine_tuning.jobs.cancel(job_ref)
        return {"job_ref": job.id, "state": job.status}

    async def deploy(self, model_ref: str, deployment_name: str) -> dict[str, Any]:
        # Azure deployments are created through the management plane, which
        # needs a subscription id and resource group rather than the data-plane
        # key used everywhere else. Rather than half-implement it, the platform
        # registers the model and tells the operator what remains to be done.
        return {
            "deployment": deployment_name,
            "model_ref": model_ref,
            "state": "registered",
            "manual_step": (
                "Create the deployment for this model in your provider's console, "
                "then set it as the answering model to use it here."
            ),
        }


class BedrockTrainer(Trainer):
    """Amazon Bedrock model customisation.

    Bedrock reads training data from S3 and writes output there, and the job
    runs under an IAM role. Those are real prerequisites, not incidental
    configuration, so they are declared up front and validated before a job is
    submitted.
    """

    name = "bedrock"

    def __init__(self, s: Settings) -> None:
        self.settings = s

    def _client(self):
        import boto3

        session_kwargs: dict[str, Any] = {"region_name": self.settings.aws_region}
        if self.settings.aws_access_key_id and self.settings.aws_secret_access_key:
            session_kwargs.update(
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
            )
            if self.settings.aws_session_token:
                session_kwargs["aws_session_token"] = self.settings.aws_session_token
        elif self.settings.aws_profile:
            session_kwargs["profile_name"] = self.settings.aws_profile
        return boto3.Session(**session_kwargs).client("bedrock")

    def capabilities(self) -> TrainerCapabilities:
        ready = bool(self.settings.bedrock_training_role_arn
                     and self.settings.bedrock_training_s3_uri)
        models = [m.strip() for m in self.settings.bedrock_training_base_models.split(",")
                  if m.strip()]
        return TrainerCapabilities(
            can_train=ready,
            can_deploy=False,
            hyperparameters=["epochs", "learning_rate_multiplier", "batch_size"],
            base_models=models,
            note="" if ready else
                 "Training on this provider needs an S3 location for the data and "
                 "an IAM role the service can assume.",
            requirements=[
                "An S3 bucket for training data and job output",
                "An IAM role the training service can assume",
                "Provisioned throughput before a customised model can serve traffic",
            ],
        )

    async def start(self, request: TrainingRequest) -> dict[str, Any]:
        import asyncio
        import uuid

        caps = self.capabilities()
        if not caps.can_train:
            raise RuntimeError(caps.note)

        s3_uri = self.settings.bedrock_training_s3_uri.rstrip("/")
        bucket, _, prefix = s3_uri.removeprefix("s3://").partition("/")
        key = f"{prefix + '/' if prefix else ''}{request.train_file.name}"

        def run() -> dict[str, Any]:
            import boto3

            session_kwargs: dict[str, Any] = {"region_name": self.settings.aws_region}
            if self.settings.aws_access_key_id:
                session_kwargs.update(
                    aws_access_key_id=self.settings.aws_access_key_id,
                    aws_secret_access_key=self.settings.aws_secret_access_key,
                )
            session = boto3.Session(**session_kwargs)
            session.client("s3").upload_file(str(request.train_file), bucket, key)

            hyper = {}
            if request.epochs:
                hyper["epochCount"] = str(request.epochs)
            if request.batch_size:
                hyper["batchSize"] = str(request.batch_size)
            if request.learning_rate_multiplier:
                hyper["learningRateMultiplier"] = str(request.learning_rate_multiplier)

            job_name = f"kg-{request.suffix or 'slm'}-{uuid.uuid4().hex[:8]}"
            response = session.client("bedrock", region_name=self.settings.aws_region
                                      ).create_model_customization_job(
                jobName=job_name,
                customModelName=job_name,
                roleArn=self.settings.bedrock_training_role_arn,
                baseModelIdentifier=request.base_model,
                trainingDataConfig={"s3Uri": f"s3://{bucket}/{key}"},
                outputDataConfig={"s3Uri": f"{s3_uri}/output/"},
                hyperParameters=hyper or {"epochCount": "2"},
            )
            return {"job_ref": response["jobArn"], "state": "queued",
                    "base_model": request.base_model}

        return await asyncio.to_thread(run)

    async def status(self, job_ref: str) -> dict[str, Any]:
        import asyncio

        def run() -> dict[str, Any]:
            response = self._client().get_model_customization_job(jobIdentifier=job_ref)
            state = response.get("status", "").lower()
            return {
                "job_ref": job_ref,
                "state": {"inprogress": "running", "completed": "succeeded",
                          "failed": "failed", "stopped": "cancelled"}.get(state, state),
                "model_ref": response.get("outputModelArn"),
                "error": response.get("failureMessage"),
            }

        return await asyncio.to_thread(run)


class InferenceOnlyTrainer(Trainer):
    """A provider that serves models but cannot train them.

    Kept so the console can explain the limitation rather than hiding the
    provider entirely — an adapter trained elsewhere can still be served here.
    """

    name = "inference_only"

    def __init__(self, s: Settings, label: str = "This provider") -> None:
        self.label = label

    def capabilities(self) -> TrainerCapabilities:
        return TrainerCapabilities(
            can_train=False, can_deploy=False,
            note=f"{self.label} serves models but does not run training. Train "
                 f"elsewhere, then serve the result here.",
        )


def _safe(obj: Any) -> dict:
    try:
        return json.loads(obj.model_dump_json())
    except Exception:
        return {}


class RemoteGPUTrainer(Trainer):
    """Training on hardware the operator controls.

    The platform never loads a model or touches a GPU. It posts a job to a
    worker running next to the hardware and polls it — the same shape as a
    hosted provider, which is why nothing above this class changes.

    Why a separate worker rather than training in this process:

      * A run lasts hours and holds gigabytes. An out-of-memory kill would take
        the console down with it.
      * The API should stay small and CPU-only. Deep-learning dependencies are
        large and are needed on exactly one machine.
      * A worker can be a bare process on a workstation, a container, a
        Kubernetes pod, or a job on a managed cluster, without the platform
        knowing which.

    The worker contract is deliberately small — four endpoints, JSON in and out,
    documented in `worker/README.md`. Anything that implements it will work,
    including a wrapper around an existing internal training pipeline.
    """

    name = "remote_gpu"

    def __init__(self, s: Settings) -> None:
        self.base_url = (s.training_worker_url or "").rstrip("/")
        self.token = s.training_worker_token
        self.timeout = s.training_worker_timeout_seconds
        self.base_models = [m.strip() for m in s.training_worker_base_models.split(",")
                            if m.strip()]

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def capabilities(self) -> TrainerCapabilities:
        configured = bool(self.base_url)
        return TrainerCapabilities(
            can_train=configured,
            can_deploy=configured,
            # Unlike a hosted service, the adapter really is configurable here,
            # so these are offered honestly.
            hyperparameters=["epochs", "learning_rate", "batch_size", "seed",
                             "lora_rank", "lora_alpha", "lora_dropout",
                             "quantization", "max_seq_length"],
            base_models=self.base_models,
            note="" if configured else
                 "Set the address of your training worker to enable this.",
            requirements=[
                "A worker process running next to the GPU, implementing the worker contract",
                "Enough VRAM for the chosen base model and sequence length",
            ],
        )

    async def start(self, request: TrainingRequest) -> dict[str, Any]:
        import httpx

        caps = self.capabilities()
        if not caps.can_train:
            raise RuntimeError(caps.note)

        files = {
            "train_file": (request.train_file.name, request.train_file.read_bytes(),
                           "application/jsonl"),
        }
        if request.validation_file and Path(request.validation_file).exists():
            files["validation_file"] = (
                Path(request.validation_file).name,
                Path(request.validation_file).read_bytes(), "application/jsonl",
            )
        data = {"config": json.dumps({
            "base_model": request.base_model,
            "suffix": request.suffix,
            "epochs": request.epochs,
            "learning_rate": request.learning_rate,
            "batch_size": request.batch_size,
            "seed": request.seed,
            "lora": request.adapter,
        })}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/jobs", data=data, files=files,
                                         headers=self._headers())
            response.raise_for_status()
            body = response.json()
        return {"job_ref": body["job_id"], "state": body.get("state", "queued"),
                "base_model": request.base_model}

    async def status(self, job_ref: str) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/jobs/{job_ref}",
                                        headers=self._headers())
            response.raise_for_status()
            body = response.json()
        return {
            "job_ref": job_ref,
            "state": body.get("state", "running"),
            "model_ref": body.get("adapter_path"),
            "error": body.get("error"),
            "progress": body.get("progress"),
            "metrics": body.get("metrics"),
        }

    async def cancel(self, job_ref: str) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/jobs/{job_ref}/cancel",
                                         headers=self._headers())
            response.raise_for_status()
        return {"job_ref": job_ref, "state": "cancelled"}

    async def deploy(self, model_ref: str, deployment_name: str) -> dict[str, Any]:
        """Ask the worker to serve the adapter.

        What "serving" means is the worker's business — it may load the adapter
        into a running inference server, merge it into the base weights, or
        simply expose the files for download.
        """
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/deployments",
                json={"adapter_path": model_ref, "name": deployment_name},
                headers=self._headers(),
            )
            response.raise_for_status()
            body = response.json()
        return {"deployment": deployment_name, "model_ref": model_ref,
                "state": body.get("state", "serving"),
                "endpoint": body.get("endpoint"),
                "manual_step": body.get("manual_step")}
