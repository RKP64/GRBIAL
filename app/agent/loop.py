"""The agent loop.

One agent, deciding for itself which tools to use and when it has enough to
answer. This replaces the fixed retrieve-then-answer pipeline, where the user
had to choose the retrieval mode.

Three constraints make it safe to run against real traffic:

  * A step ceiling. A model that keeps calling tools without converging must
    stop, and stop with a usable partial answer rather than an exception.
  * Bounded tool results. Every tool truncates, so one large return cannot
    consume the context the loop still needs.
  * A recorded trace. Every decision, tool call and result is captured, because
    an agent that cannot be inspected cannot be debugged — and this trace is
    what makes the difference between a demo and something operable.

The answer is then verified against the graph, so the trace shows not only what
the agent did but whether what it said was grounded.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..providers import get_provider
from ..usage.recorder import attribute_to
from ..verification import verify_answer
from .tools import available_tools, run_tool

log = logging.getLogger(__name__)

DEFAULT_SYSTEM = """You answer questions about {domain} using the tools provided.

How to work:
1. Decide which tool will move you closest to an answer. Call describe_schema
   first only if you are unsure what this graph contains.
2. Read what came back before deciding the next step. If a search returns
   nothing useful, try different wording or a different tool rather than
   repeating the same call.
3. Stop calling tools as soon as you can answer. Do not gather more than you need.

How to answer:
- Use only what the tools returned. Do not add facts from your own knowledge.
- If the tools did not provide the answer, say so plainly and say what is
  missing. A clear "that is not recorded" is a correct answer.
- Be specific and concise. Name the entities and relationships you relied on.
"""


@dataclass
class Step:
    index: int
    kind: str                 # thinking | tool_call | tool_result | answer | limit
    detail: str
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "kind": self.kind, "detail": self.detail,
                "data": self.data, "duration_ms": self.duration_ms}


@dataclass
class AgentResult:
    answer: str
    steps: list[Step]
    tool_calls: int
    stopped_at_limit: bool
    verification: dict | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": [s.as_dict() for s in self.steps],
            "tool_calls": self.tool_calls,
            "stopped_at_limit": self.stopped_at_limit,
            "verification": self.verification,
        }


async def run_agent(
    domain: str,
    question: str,
    *,
    system_prompt: str = "",
    allowed_tools: list[str] | None = None,
    max_steps: int = 6,
    temperature: float = 0.2,
    verify: bool = True,
    permitted_domains: list[str] | None = None,
    model: str = "",
    principal: Any = None,
) -> AgentResult:
    """Answer with tool use.

    `permitted_domains` is the read boundary. When a saved agent runs, it is that
    agent's declared domain list, and a request for anything outside it is
    refused before a tool executes — the model is never in a position to reach
    data the agent was not granted.
    """
    if permitted_domains is not None and domain not in permitted_domains:
        raise PermissionError(
            f"This agent may not read '{domain}'. It is limited to: "
            f"{', '.join(permitted_domains)}."
        )

    # A blank model keeps the platform default, so agents that do not name one
    # are unaffected.
    try:
        provider = get_provider(model)
    except RuntimeError as exc:
        raise RuntimeError(
            f"This agent asks for the model '{model}', which could not be "
            f"prepared: {exc}"
        ) from exc
    if not provider.tools_available:
        raise RuntimeError(
            "The configured language model does not support tool use. Use the "
            "standard query instead."
        )

    available = await available_tools(domain=domain)
    tools = [t for t in available
             if not allowed_tools or t["name"] in allowed_tools]
    if not tools:
        raise RuntimeError("No tools are available to this agent.")

    system = (system_prompt.strip()
              or DEFAULT_SYSTEM.format(domain=domain.replace("_", " ")))
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    steps: list[Step] = []
    tool_calls = 0
    answer = ""
    stopped_at_limit = False
    # Guards against the common failure of asking the same thing repeatedly.
    seen_calls: set[tuple[str, str]] = set()

    for step_index in range(1, max_steps + 1):
        started = time.perf_counter()
        try:
            with attribute_to("agent", domain=domain):
                turn = await provider.converse(system, messages, tools,
                                               temperature=temperature)
        except Exception as exc:
            steps.append(Step(step_index, "limit", f"The model call failed: {exc}"))
            answer = answer or ("I could not complete this request because the "
                                "language model was unavailable.")
            break
        elapsed = int((time.perf_counter() - started) * 1000)

        if turn["text"]:
            steps.append(Step(step_index, "thinking", turn["text"][:600],
                              duration_ms=elapsed))

        if turn["stop"] == "end" or not turn["tool_calls"]:
            answer = turn["text"].strip()
            steps.append(Step(step_index, "answer", "Answered.", duration_ms=elapsed))
            break

        messages.append({"role": "assistant", "content": turn["text"],
                         "tool_calls": turn["tool_calls"]})

        for call in turn["tool_calls"]:
            signature = (call["name"], str(sorted(call["arguments"].items())))
            steps.append(Step(step_index, "tool_call",
                              f"{call['name']}({_render_args(call['arguments'])})",
                              data={"tool": call["name"], "arguments": call["arguments"]}))
            if signature in seen_calls:
                result = ("That exact call was already made and returned the same "
                          "result. Try different wording, a different tool, or "
                          "answer with what you have.")
            else:
                seen_calls.add(signature)
                tool_started = time.perf_counter()
                result = await run_tool(call["name"], call["arguments"],
                                        domain=domain, principal=principal)
                steps.append(Step(step_index, "tool_result",
                                  _summarise(call["name"], result),
                                  data={"tool": call["name"], "result": result[:2000]},
                                  duration_ms=int((time.perf_counter() - tool_started) * 1000)))
            tool_calls += 1
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": result})
    else:
        # Ran out of steps with tools still pending. Ask once for a final answer
        # using what was gathered, rather than returning nothing.
        stopped_at_limit = True
        steps.append(Step(max_steps + 1, "limit",
                          f"Reached the {max_steps}-step limit; answering with what "
                          f"was gathered."))
        messages.append({"role": "user",
                         "content": "Answer now using only what the tools returned. "
                                    "If it is not enough, say what is missing."})
        try:
            # No tools on this call. Offering them would let a model that wants
            # another lookup return a tool call and no text, leaving the user
            # with nothing after a full run.
            final = await provider.converse(system, messages, [], temperature=temperature)
            answer = final["text"].strip()
        except Exception as exc:
            answer = f"I could not complete this request: {exc}"
        if not answer:
            gathered = [s for s in steps if s.kind == "tool_result"]
            answer = (
                "I could not reach a confident answer within the step limit. "
                f"I consulted {len(gathered)} source"
                f"{'s' if len(gathered) != 1 else ''} without finding enough to "
                "answer — the information may not be recorded in this domain."
            )

    verification = None
    if verify and answer:
        try:
            verification = (await verify_answer(domain, answer)).as_dict()
        except Exception as exc:      # verification must never fail an answer
            verification = {"error": str(exc)}

    return AgentResult(answer=answer or "I was unable to produce an answer.",
                       steps=steps, tool_calls=tool_calls,
                       stopped_at_limit=stopped_at_limit, verification=verification)


def _render_args(arguments: dict[str, Any]) -> str:
    parts = [f"{k}={v!r}" for k, v in arguments.items()]
    text = ", ".join(parts)
    return text if len(text) <= 120 else text[:117] + "…"


def _summarise(tool: str, result: str) -> str:
    first = result.strip().splitlines()[0] if result.strip() else "(empty)"
    lines = len(result.strip().splitlines())
    return f"{first[:140]}" + (f"  (+{lines - 1} more lines)" if lines > 1 else "")
