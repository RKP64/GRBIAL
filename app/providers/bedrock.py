from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ..config import Settings
from .base import LLMProvider

log = logging.getLogger(__name__)

try:
    import boto3
    from botocore.config import Config as BotoConfig
    BOTO_AVAILABLE = True
except ImportError:  # pragma: no cover
    BOTO_AVAILABLE = False


class BedrockProvider(LLMProvider):
    """Amazon Bedrock.

    Chat goes through the Converse API, which presents one request shape across
    model families, so switching the configured model does not change this code.

    Embeddings use InvokeModel, because the request body differs per family:
      * Titan takes one text per call and returns `embedding`
      * Cohere takes a batch and returns `embeddings`
    Both are handled below and selected from the configured model id.

    boto3 is synchronous, so every call runs in a worker thread. The extraction
    pipeline already limits concurrency with a semaphore, so the thread pool is
    bounded by that rather than by the number of chunks.
    """

    name = "bedrock"

    def __init__(self, s: Settings) -> None:
        if not BOTO_AVAILABLE:
            raise RuntimeError("boto3 is not installed")
        self.model_id = s.bedrock_model_id
        self.embedding_model_id = s.bedrock_embedding_model_id
        self.max_tokens = s.bedrock_max_tokens
        self.embedding_dimensions = s.bedrock_embedding_dimensions

        session_kwargs: dict[str, Any] = {"region_name": s.aws_region}
        # Explicit keys when supplied; otherwise boto3's own chain — IAM role,
        # profile, environment — so the platform works unchanged on EC2/ECS.
        if s.aws_access_key_id and s.aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = s.aws_access_key_id
            session_kwargs["aws_secret_access_key"] = s.aws_secret_access_key
            if s.aws_session_token:
                session_kwargs["aws_session_token"] = s.aws_session_token
        elif s.aws_profile:
            session_kwargs["profile_name"] = s.aws_profile

        session = boto3.Session(**session_kwargs)
        client_kwargs: dict[str, Any] = {
            "config": BotoConfig(
                read_timeout=int(s.extraction_timeout_seconds),
                connect_timeout=10,
                retries={"max_attempts": 2, "mode": "standard"},
            )
        }
        if s.bedrock_endpoint_url:
            client_kwargs["endpoint_url"] = s.bedrock_endpoint_url
        self.client = session.client("bedrock-runtime", **client_kwargs)

    # ------------------------------------------------------------- chat
    async def complete(self, system: str, user: str, *, temperature: float = 0.1,
                       json_mode: bool = False) -> str:
        prompt = user
        if json_mode:
            # Converse has no JSON mode flag; the instruction plus the tolerant
            # parser in the base class does the same job across model families.
            prompt = (f"{user}\n\nRespond with a single valid JSON object and nothing "
                      f"else. Do not wrap it in code fences or add commentary.")

        def run() -> str:
            response = self.client.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                system=[{"text": system}] if system else [],
                inferenceConfig={"temperature": temperature, "maxTokens": self.max_tokens},
            )
            blocks = response["output"]["message"]["content"]
            return "".join(b.get("text", "") for b in blocks)

        return await asyncio.to_thread(run)

    # ------------------------------------------------------------- embeddings
    async def embed(self, texts: list[str], batch_size: int = 96) -> list[list[float]]:
        if not self.embedding_model_id:
            raise RuntimeError("BEDROCK_EMBEDDING_MODEL_ID is not set.")
        model = self.embedding_model_id.lower()
        if "cohere" in model:
            return await self._embed_cohere(texts, batch_size)
        return await self._embed_titan(texts)

    async def _embed_titan(self, texts: list[str]) -> list[list[float]]:
        def one(text: str) -> list[float]:
            body: dict[str, Any] = {"inputText": text}
            if self.embedding_dimensions:
                body["dimensions"] = self.embedding_dimensions
                body["normalize"] = True
            response = self.client.invoke_model(
                modelId=self.embedding_model_id, body=json.dumps(body)
            )
            return json.loads(response["body"].read())["embedding"]

        # Titan embeds one input per call, so requests are issued in bounded
        # parallel batches rather than one long serial loop.
        out: list[list[float]] = []
        window = 8
        for i in range(0, len(texts), window):
            chunk = texts[i : i + window]
            out.extend(await asyncio.gather(*(asyncio.to_thread(one, t) for t in chunk)))
        return out

    async def _embed_cohere(self, texts: list[str], batch_size: int) -> list[list[float]]:
        def batch(items: list[str]) -> list[list[float]]:
            body = {"texts": items, "input_type": "search_document"}
            response = self.client.invoke_model(
                modelId=self.embedding_model_id, body=json.dumps(body)
            )
            return json.loads(response["body"].read())["embeddings"]

        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(await asyncio.to_thread(batch, texts[i : i + batch_size]))
        return out

    @property
    def embeddings_available(self) -> bool:
        return bool(self.embedding_model_id)

    def describe(self) -> dict:
        return {"provider": self.name, "model": self.model_id,
                "embedding_model": self.embedding_model_id}
