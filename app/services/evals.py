"""Measuring whether an agent actually answers correctly.

Without this, a change to the ontology, the model, or the retrieval mode can
only be judged by reading a few answers and forming an impression. That is how
a fine-tune that quietly degraded half the corpus gets shipped.

The unit is a golden set: questions with answers a person considers correct.
Any set can be run against any agent, so the same questions measure a change of
model, a change of retrieval mode, or two agents against each other. Runs are
stored whole, which is what makes "did this get better or worse" answerable.

Three numbers per question, because one is not enough:

  correctness  — a model judges the answer against the expected one. String
                 matching fails on valid paraphrase, so it cannot be used.
  grounding    — the platform's own claim check. An answer can be correct and
                 ungrounded, which is luck rather than retrieval working.
  latency      — the cost argument for a small model rests on it.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..services.llm import complete_json

log = logging.getLogger(__name__)

VERDICTS = ("correct", "partial", "incorrect", "refused")

JUDGE_SYSTEM = """You compare an answer against a reference answer and judge it.

Return JSON only: {"verdict": "...", "reason": "one short sentence"}

verdict is one of:
  correct   — conveys the same facts as the reference. Wording may differ
              entirely; that is not a fault.
  partial   — some of the reference facts, missing others, nothing contradicted.
  incorrect — contradicts the reference, or asserts something it does not support.
  refused   — declines to answer or says the information is not available.

Judge only against the reference. Do not use your own knowledge of the subject —
an answer that is true in the world but absent from the reference is not
evidence the system worked.

A refusal where the reference contains a real answer is a miss, not a refusal —
mark it incorrect. Reserve "refused" for cases where declining was right.
"""


# ------------------------------------------------------------------ storage

@dataclass
class GoldenItem:
    question: str
    expected: str
    note: str = ""


@dataclass
class GoldenSet:
    key: str
    name: str
    items: list[GoldenItem] = field(default_factory=list)
    created_at: str = ""

    def summary(self) -> dict[str, Any]:
        return {"key": self.key, "name": self.name,
                "items": len(self.items), "created_at": self.created_at}


@dataclass
class Result:
    question: str
    expected: str
    answer: str
    verdict: str
    reason: str = ""
    grounding: float | None = None
    tool_calls: int = 0
    seconds: float = 0.0
    error: str = ""


@dataclass
class Run:
    id: str
    set_key: str
    agent_key: str
    model: str = ""
    started_at: str = ""
    finished_at: str = ""
    results: list[Result] = field(default_factory=list)

    def scores(self) -> dict[str, Any]:
        total = len(self.results)
        if not total:
            return {"total": 0}
        counts = {v: sum(1 for r in self.results if r.verdict == v) for v in VERDICTS}
        grounded = [r.grounding for r in self.results if r.grounding is not None]
        return {
            "total": total,
            **counts,
            "errors": sum(1 for r in self.results if r.error),
            # Partial counts half: an answer carrying some of the reference is
            # worth more than a wrong one and less than a right one.
            "accuracy": round((counts["correct"] + counts["partial"] * 0.5) / total, 3),
            "grounding": round(sum(grounded) / len(grounded), 3) if grounded else None,
            "median_seconds": round(
                sorted(r.seconds for r in self.results)[total // 2], 2),
        }

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "set_key": self.set_key, "agent_key": self.agent_key,
                "model": self.model or "platform default",
                "started_at": self.started_at, "finished_at": self.finished_at,
                "scores": self.scores()}


def _dir(name: str) -> Path:
    d = get_settings().data_dir / "evals" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_set(key: str, name: str, items: list[dict[str, Any]]) -> GoldenSet:
    parsed = [
        GoldenItem(question=str(i.get("question", "")).strip(),
                   expected=str(i.get("expected", "")).strip(),
                   note=str(i.get("note", "")).strip())
        for i in items
    ]
    parsed = [i for i in parsed if i.question and i.expected]
    if not parsed:
        raise ValueError("No usable rows. Each needs a question and an expected answer.")
    gset = GoldenSet(key=key, name=name or key, items=parsed,
                     created_at=datetime.now(timezone.utc).isoformat())
    (_dir("sets") / f"{key}.json").write_text(
        json.dumps(asdict(gset), indent=2), encoding="utf-8")
    return gset


def load_set(key: str) -> GoldenSet:
    path = _dir("sets") / f"{key}.json"
    if not path.exists():
        raise KeyError(f"No golden set called '{key}'.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return GoldenSet(key=raw["key"], name=raw["name"],
                     created_at=raw.get("created_at", ""),
                     items=[GoldenItem(**i) for i in raw["items"]])


def list_sets() -> list[dict[str, Any]]:
    out = []
    for path in sorted(_dir("sets").glob("*.json")):
        try:
            out.append(load_set(path.stem).summary())
        except Exception as exc:
            log.warning("Skipping unreadable golden set %s: %s", path.name, exc)
    return out


def delete_set(key: str) -> None:
    (_dir("sets") / f"{key}.json").unlink(missing_ok=True)


def save_run(run: Run) -> None:
    (_dir("runs") / f"{run.id}.json").write_text(
        json.dumps({**asdict(run), "scores": run.scores()}, indent=2),
        encoding="utf-8")


def load_run(run_id: str) -> dict[str, Any]:
    path = _dir("runs") / f"{run_id}.json"
    if not path.exists():
        raise KeyError(f"No run called '{run_id}'.")
    return json.loads(path.read_text(encoding="utf-8"))


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    paths = sorted(_dir("runs").glob("*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            out.append({k: raw.get(k) for k in
                        ("id", "set_key", "agent_key", "model",
                         "started_at", "finished_at", "scores")})
        except Exception:
            continue
    return out


# ------------------------------------------------------------------ judging

async def judge(question: str, expected: str, answer: str) -> tuple[str, str]:
    """Score one answer. A judge failure is not scored as a wrong answer."""
    if not answer.strip():
        return "incorrect", "No answer was produced."
    prompt = (f"Question:\n{question}\n\n"
              f"Reference answer:\n{expected}\n\n"
              f"Answer to judge:\n{answer}")
    try:
        raw = await complete_json(JUDGE_SYSTEM, prompt, temperature=0.0)
    except Exception as exc:
        log.warning("Judge failed: %s", exc)
        return "", f"Could not be judged: {exc}"
    verdict = str(raw.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        return "", f"Judge returned an unusable verdict: {verdict!r}"
    return verdict, str(raw.get("reason", "")).strip()


# ------------------------------------------------------------------ running

async def run_eval(set_key: str, agent_key: str, *, principal: Any = None,
                   on_progress=None) -> Run:
    """Run every question in a set against one agent.

    Questions run one at a time rather than concurrently. A run is not
    time-critical, and firing a hundred agent loops at once distorts the latency
    figure that is one of the things being measured.
    """
    from ..agent.loop import run_agent
    from ..agent.registry import get_registry

    gset = load_set(set_key)
    agent = get_registry().get(agent_key)
    domain = agent.default_domain()

    tools = list(agent.tools)
    mode = agent.retrieval_mode or "graph+rag"
    if mode == "rag":
        tools = [t for t in tools if t not in ("search_graph", "expand_entity")]
    elif mode == "graph":
        tools = [t for t in tools if t != "search_documents"]

    run = Run(id=uuid.uuid4().hex[:12], set_key=set_key, agent_key=agent_key,
              model=agent.model, started_at=datetime.now(timezone.utc).isoformat())

    for index, item in enumerate(gset.items, 1):
        started = time.perf_counter()
        answer, grounding, calls, error = "", None, 0, ""
        try:
            result = await run_agent(
                domain, item.question,
                system_prompt=agent.system_prompt,
                allowed_tools=tools or agent.tools,
                max_steps=agent.max_steps, temperature=agent.temperature,
                verify=agent.verify, permitted_domains=agent.domains,
                model=agent.model, principal=principal)
            payload = result.as_dict()
            answer = payload.get("answer") or ""
            calls = payload.get("tool_calls") or 0
            verification = payload.get("verification") or {}
            grounding = verification.get("score")
        except Exception as exc:
            error = str(exc)
            log.warning("Eval question failed: %s", exc)

        seconds = round(time.perf_counter() - started, 2)
        verdict, reason = ("", error) if error else await judge(
            item.question, item.expected, answer)

        run.results.append(Result(
            question=item.question, expected=item.expected, answer=answer,
            verdict=verdict or "incorrect", reason=reason, grounding=grounding,
            tool_calls=calls, seconds=seconds, error=error))

        if on_progress:
            on_progress(index, len(gset.items), run.results[-1])

    run.finished_at = datetime.now(timezone.utc).isoformat()
    save_run(run)
    return run


def compare(run_a: dict[str, Any], run_b: dict[str, Any]) -> dict[str, Any]:
    """Difference between two runs, question by question.

    The per-question regressions matter more than the headline number: an
    unchanged accuracy can hide ten questions that broke and ten that were
    fixed, which is not the same as nothing having happened.
    """
    by_question_a = {r["question"]: r for r in run_a.get("results", [])}
    rank = {"correct": 3, "partial": 2, "refused": 1, "incorrect": 0}

    improved, regressed = [], []
    for result_b in run_b.get("results", []):
        result_a = by_question_a.get(result_b["question"])
        if not result_a:
            continue
        before, after = rank.get(result_a["verdict"], 0), rank.get(result_b["verdict"], 0)
        entry = {"question": result_b["question"],
                 "from": result_a["verdict"], "to": result_b["verdict"],
                 "reason": result_b.get("reason", "")}
        if after > before:
            improved.append(entry)
        elif after < before:
            regressed.append(entry)

    scores_a = run_a.get("scores") or {}
    scores_b = run_b.get("scores") or {}
    return {
        "a": {"id": run_a.get("id"), "agent": run_a.get("agent_key"),
              "model": run_a.get("model"), "scores": scores_a},
        "b": {"id": run_b.get("id"), "agent": run_b.get("agent_key"),
              "model": run_b.get("model"), "scores": scores_b},
        "accuracy_delta": round((scores_b.get("accuracy") or 0)
                                - (scores_a.get("accuracy") or 0), 3),
        "improved": improved,
        "regressed": regressed,
    }
