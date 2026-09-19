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


def parse_models(spec: str) -> list[dict[str, str]]:
    """Turn a configured model list into id/label pairs.

    Written as "Label|provider/model-id" so the console can show a neutral
    name while the API still receives the identifier the service expects.
    A bare id falls back to showing itself, which keeps existing config valid.
    """
    out: list[dict[str, str]] = []
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "|" in item:
            label, _, ident = item.partition("|")
            out.append({"id": ident.strip(), "label": label.strip()})
        else:
            out.append({"id": item, "label": item})
    return out


@dataclass
class TrainerCapabilities:
    # What the console shows. Deliberately describes the arrangement rather
    # than the supplier, so the platform can be presented without disclosing
    # which services sit behind it.
    label: str = ""
    can_train: bool = False
    can_deploy: bool = False
    hyperparameters: list[str] = field(default_factory=list)
    base_models: list[dict] = field(default_factory=list)
    note: str = ""
    requirements: list[str] = field(default_factory=list)
    # Whether the produced artefact can be pulled down and kept. This is the
    # difference between exclusive use of a hosted model and owning a file, and
    # it decides whether a sovereignty claim survives scrutiny — so it is
    # surfaced rather than left for someone to discover after training.
    can_download_weights: bool = False
    # Where the training data and the resulting weights sit, stated as a
    # posture rather than a place: "In-region", "Offshore", "Your infrastructure".
    data_residency: str = ""


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
        models = parse_models(self.settings.training_base_models)
        return TrainerCapabilities(
            label="Managed service",
            can_train=configured,
            can_deploy=self.settings.training_provider_mode == "azure" and configured,
            hyperparameters=["epochs", "learning_rate_multiplier", "batch_size", "seed", "suffix"],
            base_models=models,
            can_download_weights=False,
            data_residency="Provider region",
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
            label="Managed service (object-store)",
            can_download_weights=False,
            data_residency="Provider region",
            note="" if ready else
                 "Training on this provider needs an object-store location for the "
                 "data and a role the service can assume.",
            requirements=[
                "An object-store bucket for training data and job output",
                "A role the training service can assume",
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


class TogetherTrainer(Trainer):
    """Together AI hosted fine-tuning of open-weight base models.

    The reason to reach for this rather than a hosted provider that keeps the
    result: Together fine-tunes open-weight bases and lets you pull the
    resulting adapter or merged checkpoint down and run it anywhere. What you
    end up with is a file, not a deployment name.

    The trade is jurisdictional. Together is US-incorporated, so the training
    data leaves India. That is fine for a pilot on non-sensitive or synthetic
    data and wrong for production BIAL material — which is why the residency is
    reported here instead of being left implicit.
    """

    name = "together"
    BASE = "https://api.together.xyz/v1"

    def __init__(self, s: Settings) -> None:
        self.settings = s
        self.timeout = 120.0

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.together_api_key}"}

    def capabilities(self) -> TrainerCapabilities:
        configured = bool(self.settings.together_api_key)
        models = parse_models(self.settings.together_base_models)
        return TrainerCapabilities(
            label="Portable open-weight",
            can_train=configured,
            can_deploy=configured,
            hyperparameters=["epochs", "learning_rate", "batch_size", "suffix"],
            base_models=models,
            can_download_weights=True,
            data_residency="Offshore",
            note=("" if configured
                  else "Add the credentials for this provider to enable training."),
            requirements=[
                "An open-weight base model this provider supports for fine-tuning",
                "Training data leaves the region — suited to pilots, not regulated data",
            ],
        )

    async def start(self, request: TrainingRequest) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with open(request.train_file, "rb") as fh:
                upload = await client.post(
                    f"{self.BASE}/files",
                    headers=self._headers(),
                    files={"file": (Path(request.train_file).name, fh, "application/jsonl")},
                    data={"purpose": "fine-tune"},
                )
            upload.raise_for_status()
            train_id = upload.json().get("id")

            validation_id = None
            if request.validation_file and Path(request.validation_file).exists():
                with open(request.validation_file, "rb") as fh:
                    val = await client.post(
                        f"{self.BASE}/files",
                        headers=self._headers(),
                        files={"file": (Path(request.validation_file).name, fh,
                                        "application/jsonl")},
                        data={"purpose": "fine-tune"},
                    )
                if val.is_success:
                    validation_id = val.json().get("id")

            payload: dict[str, Any] = {
                "model": request.base_model,
                "training_file": train_id,
            }
            if validation_id:
                payload["validation_file"] = validation_id
            if request.epochs:
                payload["n_epochs"] = request.epochs
            if request.learning_rate:
                payload["learning_rate"] = request.learning_rate
            if request.batch_size:
                payload["batch_size"] = request.batch_size
            if request.suffix:
                payload["suffix"] = request.suffix
            # LoRA keeps the artefact small and the cost low; a full fine-tune
            # is available by setting adapter.method to "full".
            adapter = request.adapter or {}
            if adapter.get("method", "lora") == "lora":
                payload["lora"] = True
                if adapter.get("rank"):
                    payload["lora_r"] = adapter["rank"]

            job = await client.post(f"{self.BASE}/fine-tunes",
                                    headers=self._headers(), json=payload)
            job.raise_for_status()
            body = job.json()

        return {"job_ref": body.get("id"), "state": body.get("status", "queued"),
                "base_model": request.base_model, "raw": body}

    async def status(self, job_ref: str) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.BASE}/fine-tunes/{job_ref}",
                                        headers=self._headers())
            response.raise_for_status()
            body = response.json()

        state = str(body.get("status", "running")).lower()
        # Together reports states such as "completed"; the registry treats
        # "succeeded" as terminal, so the vocabulary is aligned here rather
        # than special-cased in the registry.
        if state in {"completed", "complete"}:
            state = "succeeded"
        elif state in {"error", "user_error"}:
            state = "failed"
        return {
            "job_ref": job_ref,
            "state": state,
            "model_ref": body.get("output_name") or body.get("model_output_name"),
            "error": body.get("error"),
            "trained_tokens": body.get("total_price"),
            "raw": body,
        }

    async def cancel(self, job_ref: str) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.BASE}/fine-tunes/{job_ref}/cancel",
                                         headers=self._headers())
            response.raise_for_status()
        return {"job_ref": job_ref, "state": "cancelled"}

    async def deploy(self, model_ref: str, deployment_name: str) -> dict[str, Any]:
        """Together serves the tuned model directly by name.

        The weights are also retrievable, which is the point of choosing this
        provider, so the download route is reported alongside the endpoint.
        """
        return {
            "deployment": deployment_name,
            "model_ref": model_ref,
            "state": "serving",
            "endpoint": f"{self.BASE}/chat/completions",
            "download": f"{self.BASE}/finetune/download?ft_id={model_ref}",
            "manual_step": (
                "Call the model by name against the provider endpoint, or pull the "
                "weights with the download route and serve them yourself."
            ),
        }


class ShaktiTrainer(Trainer):
    """Yotta Shakti Studio fine-tuning.

    Kept deliberately generic: Shakti Studio exposes fine-tuning for open bases
    such as Llama and Qwen, and the endpoint is configured rather than hardcoded
    so this works whether the tenant is given an OpenAI-compatible surface or a
    dedicated one. Confirm the exact route with Yotta before relying on it in
    production — SHAKTI_BASE_URL is the single place that changes.

    The reason it is worth wiring at all: the compute and the data stay inside
    Indian jurisdiction, which is the one thing Together cannot offer.
    """

    name = "shakti"

    def __init__(self, s: Settings) -> None:
        self.settings = s
        self.timeout = 120.0

    @property
    def base_url(self) -> str:
        return (self.settings.shakti_base_url or "").rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.shakti_api_key}"}

    def capabilities(self) -> TrainerCapabilities:
        configured = bool(self.settings.shakti_api_key and self.base_url)
        models = parse_models(self.settings.shakti_base_models)
        return TrainerCapabilities(
            label="In-region open-weight",
            can_train=configured,
            can_deploy=configured,
            hyperparameters=["epochs", "learning_rate", "batch_size", "suffix"],
            base_models=models,
            # Bringing a fine-tuned model in is documented; exporting one out is
            # not. Claimed as false until Yotta confirms it in writing, because
            # the opposite error is the expensive one.
            can_download_weights=False,
            data_residency="In-region",
            note=("" if configured
                  else "Add the endpoint and credentials for this provider to enable training."),
            requirements=[
                "A tenant with fine-tuning enabled",
                "Confirm with the provider whether trained weights can be exported",
            ],
        )

    async def start(self, request: TrainingRequest) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with open(request.train_file, "rb") as fh:
                upload = await client.post(
                    f"{self.base_url}/files",
                    headers=self._headers(),
                    files={"file": (Path(request.train_file).name, fh, "application/jsonl")},
                    data={"purpose": "fine-tune"},
                )
            upload.raise_for_status()
            train_id = upload.json().get("id")

            payload: dict[str, Any] = {
                "model": request.base_model,
                "training_file": train_id,
            }
            if request.epochs:
                payload["n_epochs"] = request.epochs
            if request.learning_rate:
                payload["learning_rate"] = request.learning_rate
            if request.batch_size:
                payload["batch_size"] = request.batch_size
            if request.suffix:
                payload["suffix"] = request.suffix

            job = await client.post(f"{self.base_url}/fine-tunes",
                                    headers=self._headers(), json=payload)
            job.raise_for_status()
            body = job.json()

        return {"job_ref": body.get("id") or body.get("job_id"),
                "state": body.get("status", "queued"),
                "base_model": request.base_model, "raw": body}

    async def status(self, job_ref: str) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/fine-tunes/{job_ref}",
                                        headers=self._headers())
            response.raise_for_status()
            body = response.json()

        state = str(body.get("status", "running")).lower()
        if state in {"completed", "complete"}:
            state = "succeeded"
        elif state in {"error", "user_error"}:
            state = "failed"
        return {
            "job_ref": job_ref,
            "state": state,
            "model_ref": body.get("output_name") or body.get("fine_tuned_model"),
            "error": body.get("error"),
            "progress": body.get("progress"),
            "raw": body,
        }

    async def cancel(self, job_ref: str) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/fine-tunes/{job_ref}/cancel",
                                         headers=self._headers())
            response.raise_for_status()
        return {"job_ref": job_ref, "state": "cancelled"}

    async def deploy(self, model_ref: str, deployment_name: str) -> dict[str, Any]:
        return {
            "deployment": deployment_name,
            "model_ref": model_ref,
            "state": "registered",
            "endpoint": f"{self.base_url}/chat/completions",
            "manual_step": (
                "Create the serverless endpoint for this model in the provider "
                "console, then set it as the answering model here."
            ),
        }


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
            label="Serving only",
            data_residency="Provider region",
            note="This provider serves models but does not run training. Train "
                 "elsewhere, then serve the result here.",
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
            base_models=parse_models(",".join(self.base_models))
                        if isinstance(self.base_models, list) else self.base_models,
            label="Your own hardware",
            can_download_weights=True,
            data_residency="Your infrastructure",
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
