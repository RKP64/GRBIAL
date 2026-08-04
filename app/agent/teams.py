"""Teams: several agents working one conversation.

The orchestration question is how control passes between agents. Two approaches
are common:

  * Prose master logic — a description of the flow, interpreted at runtime.
    Easy to author, but handoff conditions are soft, and a run can only be
    understood by replaying a trace afterwards.
  * Explicit handoff — each agent declares which agents it may pass to, and
    passing is an action it takes deliberately.

This uses the second. Handoff is exposed to the model as a tool, so a transfer
appears in the trace as a decision with a stated reason, the permitted routes are
declared configuration rather than an emergent property of a prompt, and an
agent can only reach agents it was given. The same mechanism is testable without
a model in the loop.

The conversation is shared: each agent sees what came before, so the applicant
is never asked twice for the same thing.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..providers import get_provider
from ..verification import verify_answer
from .loop import Step
from .registry import get_registry
from .tools import available_tools, run_tool

KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,48}$")

HANDOFF_TOOL = "hand_off"


@dataclass
class Member:
    """One agent's place in a team."""
    agent: str
    hands_off_to: list[str] = field(default_factory=list)
    when: str = ""          # stated in the prompt so the model knows when to pass

    def as_dict(self) -> dict[str, Any]:
        return {"agent": self.agent, "hands_off_to": self.hands_off_to, "when": self.when}


@dataclass
class Team:
    key: str
    name: str
    description: str = ""
    entry: str = ""
    members: list[Member] = field(default_factory=list)
    max_handoffs: int = 6
    max_steps_per_agent: int = 4
    verify: bool = True
    starters: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "name": self.name, "description": self.description,
                "entry": self.entry, "members": [m.as_dict() for m in self.members],
                "max_handoffs": self.max_handoffs,
                "max_steps_per_agent": self.max_steps_per_agent,
                "verify": self.verify, "starters": self.starters,
                "created_at": self.created_at, "updated_at": self.updated_at}

    def member(self, agent_key: str) -> Member | None:
        return next((m for m in self.members if m.agent == agent_key), None)


class TeamRegistry:
    def __init__(self, data_dir: Path) -> None:
        self.dir = Path(data_dir) / "teams"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def list(self) -> list[Team]:
        out: list[Team] = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                raw["members"] = [Member(**m) for m in raw.get("members", [])]
                out.append(Team(**raw))
            except (json.JSONDecodeError, TypeError):
                continue
        return out

    def get(self, key: str) -> Team:
        path = self._path(key)
        if not path.exists():
            raise KeyError(f"There is no team called '{key}'.")
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["members"] = [Member(**m) for m in raw.get("members", [])]
        return Team(**raw)

    def save(self, spec: dict[str, Any]) -> Team:
        key = str(spec.get("key", "")).strip().lower()
        if not KEY_RE.fullmatch(key):
            raise ValueError("Key must be 2-49 characters: lowercase letters, "
                             "digits, hyphen or underscore.")
        members = [Member(**m) if isinstance(m, dict) else m
                   for m in (spec.get("members") or [])]
        if not members:
            raise ValueError("A team needs at least one agent.")

        agents = get_registry()
        names = {m.agent for m in members}
        for m in members:
            try:
                agents.get(m.agent)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            unknown = [t for t in m.hands_off_to if t not in names]
            if unknown:
                raise ValueError(
                    f"'{m.agent}' is set to hand off to {', '.join(unknown)}, which "
                    f"{'is' if len(unknown) == 1 else 'are'} not in this team."
                )
            if m.agent in m.hands_off_to:
                raise ValueError(f"'{m.agent}' cannot hand off to itself.")

        entry = str(spec.get("entry") or members[0].agent)
        if entry not in names:
            raise ValueError(f"The starting agent '{entry}' is not in this team.")

        now = datetime.now(timezone.utc).isoformat()
        existing = None
        if self._path(key).exists():
            try:
                existing = self.get(key)
            except Exception:
                existing = None

        team = Team(
            key=key, name=str(spec.get("name") or key).strip(),
            description=str(spec.get("description") or "").strip(),
            entry=entry, members=members,
            max_handoffs=max(1, min(12, int(spec.get("max_handoffs") or 6))),
            max_steps_per_agent=max(1, min(8, int(spec.get("max_steps_per_agent") or 4))),
            verify=bool(spec.get("verify", True)),
            starters=[s for s in (spec.get("starters") or []) if str(s).strip()][:6],
            created_at=existing.created_at if existing else now, updated_at=now,
        )
        self._path(key).write_text(json.dumps(team.as_dict(), indent=2), encoding="utf-8")
        return team

    def delete(self, key: str) -> None:
        path = self._path(key)
        if not path.exists():
            raise KeyError(f"There is no team called '{key}'.")
        path.unlink()


def get_team_registry() -> TeamRegistry:
    return TeamRegistry(get_settings().data_dir)


# --------------------------------------------------------------------- running
def _handoff_schema(targets: list[dict[str, str]]) -> dict[str, Any]:
    listed = "\n".join(f"- {t['agent']}: {t['description']}" for t in targets)
    return {
        "name": HANDOFF_TOOL,
        "description": (
            "Pass this conversation to a colleague who is better placed to "
            "continue. Use it when the next thing needed is outside what you "
            "handle. Do not use it to avoid answering something you can answer.\n\n"
            f"Available:\n{listed}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "enum": [t["agent"] for t in targets],
                       "description": "Who should continue."},
                "reason": {"type": "string",
                           "description": "Why, in one sentence — the next person sees this."},
                "summary": {"type": "string",
                            "description": "What has been established so far that they need."},
            },
            "required": ["to", "reason"],
        },
    }


@dataclass
class TeamResult:
    answer: str
    steps: list[Step]
    path: list[str]
    handoffs: int
    tool_calls: int
    stopped_at_limit: bool
    verification: dict | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"answer": self.answer, "steps": [s.as_dict() for s in self.steps],
                "path": self.path, "handoffs": self.handoffs,
                "tool_calls": self.tool_calls,
                "stopped_at_limit": self.stopped_at_limit,
                "verification": self.verification}


async def run_team(team: Team, question: str, *, verify: bool | None = None) -> TeamResult:
    provider = get_provider()
    if not provider.tools_available:
        raise RuntimeError(
            "The configured language model does not support tool use, which teams "
            "require. Use a single query instead."
        )

    agents = get_registry()
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    steps: list[Step] = []
    path: list[str] = []
    handoffs = 0
    tool_calls = 0
    step_no = 0
    answer = ""
    stopped_at_limit = False
    current = team.entry
    carried = ""          # what the previous agent passed along
    visits: dict[str, int] = {}

    while True:
        member = team.member(current)
        agent = agents.get(current)
        path.append(current)
        visits[current] = visits.get(current, 0) + 1
        step_no += 1
        steps.append(Step(step_no, "agent_start", f"{agent.name} took over.",
                          data={"agent": current, "reason": carried}))

        targets = []
        for t in (member.hands_off_to if member else []):
            try:
                other = agents.get(t)
                targets.append({"agent": t,
                                "description": other.description or other.name})
            except KeyError:
                continue

        tools = [t for t in await available_tools() if t["name"] in agent.tools]
        if targets:
            tools = tools + [_handoff_schema(targets)]

        system = _system_for(agent, member, carried)
        handed_to: str | None = None

        for _ in range(team.max_steps_per_agent):
            step_no += 1
            started = time.perf_counter()
            try:
                turn = await provider.converse(system, messages, tools, temperature=agent.temperature)
            except Exception as exc:
                steps.append(Step(step_no, "limit", f"The model call failed: {exc}"))
                break
            elapsed = int((time.perf_counter() - started) * 1000)

            if turn["text"]:
                steps.append(Step(step_no, "thinking", turn["text"][:500],
                                  data={"agent": current}, duration_ms=elapsed))

            if not turn["tool_calls"]:
                answer = turn["text"].strip()
                steps.append(Step(step_no, "answer", "Answered.", data={"agent": current}))
                break

            messages.append({"role": "assistant", "content": turn["text"],
                             "tool_calls": turn["tool_calls"]})

            for call in turn["tool_calls"]:
                if call["name"] == HANDOFF_TOOL:
                    target = str(call["arguments"].get("to", ""))
                    reason = str(call["arguments"].get("reason", ""))
                    summary = str(call["arguments"].get("summary", ""))
                    permitted = member.hands_off_to if member else []
                    if target not in permitted:
                        result = (f"You cannot hand off to '{target}'. You may pass to: "
                                  f"{', '.join(permitted) or 'nobody'}.")
                    elif handoffs >= team.max_handoffs:
                        result = ("No further handoffs are available. Answer with what "
                                  "the conversation already contains.")
                    elif visits.get(target, 0) >= 2:
                        result = (f"{target} has already handled this twice. Continue "
                                  f"yourself rather than passing back.")
                    else:
                        handed_to = target
                        carried = f"{reason}\n{summary}".strip()
                        steps.append(Step(step_no, "handoff",
                                          f"{agent.name} → {target}: {reason}",
                                          data={"from": current, "to": target,
                                                "reason": reason, "summary": summary}))
                        result = f"Handed to {target}."
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": result})
                    continue

                steps.append(Step(step_no, "tool_call",
                                  f"{call['name']}({_args(call['arguments'])})",
                                  data={"agent": current, "tool": call["name"]}))
                domain = agent.default_domain()
                result = await run_tool(call["name"], call["arguments"], domain=domain)
                tool_calls += 1
                steps.append(Step(step_no, "tool_result", _summary(result),
                                  data={"agent": current, "result": result[:1500]}))
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": result})

            if handed_to:
                break

        if handed_to:
            handoffs += 1
            current = handed_to
            continue

        if answer:
            break

        # The agent neither answered nor handed off within its budget.
        if handoffs >= team.max_handoffs:
            stopped_at_limit = True
        steps.append(Step(step_no + 1, "limit",
                          f"{agent.name} reached its step budget."))
        try:
            final = await provider.converse(system, messages, [], temperature=agent.temperature)
            answer = final["text"].strip()
        except Exception as exc:
            answer = f"I could not complete this request: {exc}"
        break

    if not answer:
        answer = ("I was unable to reach an answer. The conversation moved between "
                  f"{len(set(path))} agents without settling.")

    verification = None
    should_verify = team.verify if verify is None else verify
    if should_verify and answer:
        domain = agents.get(path[-1]).default_domain()
        try:
            verification = (await verify_answer(domain, answer)).as_dict()
        except Exception as exc:
            verification = {"error": str(exc)}

    return TeamResult(answer=answer, steps=steps, path=path, handoffs=handoffs,
                      tool_calls=tool_calls, stopped_at_limit=stopped_at_limit,
                      verification=verification)


def _system_for(agent, member, carried: str) -> str:
    parts = [agent.system_prompt.strip() or
             f"You are {agent.name}. {agent.description}".strip()]
    if member and member.when:
        parts.append(f"Hand off when: {member.when}")
    if member and member.hands_off_to:
        parts.append(
            "You are part of a team. If the next thing needed is outside what you "
            f"handle, use {HANDOFF_TOOL} to pass the conversation on, with a reason "
            "and a summary of what you established. Otherwise answer yourself."
        )
    if carried:
        parts.append(f"You were handed this conversation because:\n{carried}")
    return "\n\n".join(parts)


def _args(arguments: dict[str, Any]) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    return text if len(text) <= 100 else text[:97] + "…"


def _summary(result: str) -> str:
    first = result.strip().splitlines()[0] if result.strip() else "(empty)"
    return first[:130]
