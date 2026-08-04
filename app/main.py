"""Knowledge Graph Platform — API service."""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import (access, agent, design, graph, health, ingest, jobs, mcp,
                  ontology, query, training, usage, verify)
from .config import get_settings
from .logging_setup import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
log = logging.getLogger("app")

app = FastAPI(
    title="Knowledge Graph Platform",
    version="1.1.0",
    description=(
        "Builds ontology-governed knowledge graphs from documents, answers "
        "questions over them, and verifies those answers against the graph.\n\n"
        "**Designed to be embedded.** Every capability is a JSON endpoint, so the "
        "console shipped with this service is one client among many — another "
        "front end, an agent, or an existing application can use the same API.\n\n"
        "Authenticate with an `X-API-Key` header. Start with `GET /readyz` to see "
        "what the deployment can currently do.\n\n"
        "Typical sequence for building a graph:\n"
        "1. `GET /ontologies` — pick or create a domain schema\n"
        "2. `POST /ingest` — upload documents, returns a job id\n"
        "3. `GET /jobs/{id}/stream` — follow progress live, or poll `GET /jobs/{id}`\n"
        "4. `GET /graph/{domain}/stats` — confirm what was built\n"
        "5. `POST /query` — ask questions, optionally with `verify: true`\n"
        "6. `GET /graph/{domain}/export.json` — take the graph elsewhere"
    ),
    openapi_tags=[
        {"name": "system", "description": "Health and capability reporting."},
        {"name": "access", "description": "Who may do what."},
        {"name": "usage", "description": "What has been spent, and on what."},
        {"name": "tools", "description": "External tool servers."},
        {"name": "tools", "description": "External tool servers an agent may use."},
        {"name": "ontology", "description": "Domain schemas: the contract extraction obeys."},
        {"name": "design", "description": "Propose a domain, agents and a team from samples."},
        {"name": "ingest", "description": "Turn documents into graph. Returns a job."},
        {"name": "jobs", "description": "Job progress, live or polled."},
        {"name": "graph", "description": "Read, visualise and export the graph."},
        {"name": "query", "description": "Answer questions from the graph and documents."},
        {"name": "verification", "description": "Check answers claim by claim against the graph."},
        {"name": "agent", "description": "Agents, teams, and answering with tool use."},
        {"name": "training", "description": "Build training sets from the graph and fine-tune."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    log.info(
        "%s %s -> %s", request.method, request.url.path, response.status_code,
        extra={"event": "http", "job_id": request_id},
    )
    return response


@app.get("/", include_in_schema=False)
async def root() -> dict:
    """Point an integrator at the things they need."""
    return {
        "service": "Knowledge Graph Platform",
        "version": app.version,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "readiness": "/readyz",
    }


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong on the server. Check service logs."},
    )


app.include_router(health.router)
app.include_router(access.router)
app.include_router(usage.router)
app.include_router(mcp.router)
app.include_router(design.router)
app.include_router(ontology.router)
app.include_router(ingest.router)
app.include_router(jobs.router)
app.include_router(graph.router)
app.include_router(query.router)
app.include_router(training.router)
app.include_router(verify.router)
app.include_router(agent.router)
app.include_router(agent.teams_router)
