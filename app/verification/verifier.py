"""Answer verification against the knowledge graph.

Most platforms measure hallucination by asking a judging model for an opinion.
Because a governed graph is ground truth we own, this measures something
stronger: whether each claim in an answer corresponds to a relationship that
survived schema validation, and cites the specific edge.

The method is deliberately hybrid rather than purely model-based:

  1. A model splits the answer into atomic claims. This is a linguistic task and
     is what models are reliably good at.
  2. Entity linking is deterministic — claims are matched to graph nodes by
     normalised string matching, not by asking a model which entities exist.
     Asking a model to recall the graph would reintroduce the very failure the
     layer is meant to detect.
  3. Candidate edges are gathered from the graph around the linked entities.
  4. A model decides only whether a specific candidate edge supports a specific
     claim — a narrow judgement over evidence placed in front of it, not a
     recall task.

Contradiction is reported conservatively. It is claimed only when the graph
holds the same subject and relation with a different object *and* that relation
behaves as single-valued everywhere else in the graph. Anything weaker is
reported as unsupported, because an absent edge is far more often an extraction
gap than a falsehood, and calling a true statement a contradiction is the worse
error.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..services.llm import complete_json
from ..stores import get_store
from ..usage.recorder import attribute_to

log = logging.getLogger(__name__)

_WORD = re.compile(r"[A-Za-z0-9]+")


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    NOT_CHECKABLE = "not_checkable"     # hedges, offers of help, meta-statements


@dataclass
class Claim:
    text: str
    status: ClaimStatus = ClaimStatus.UNSUPPORTED
    entities: list[str] = field(default_factory=list)
    evidence: list[dict[str, str]] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"claim": self.text, "status": self.status.value,
                "entities": self.entities, "evidence": self.evidence,
                "reason": self.reason}


@dataclass
class VerificationResult:
    claims: list[Claim]
    grounding_score: float
    checkable_claims: int
    supported: int
    unsupported: int
    contradicted: int
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "grounding_score": self.grounding_score,
            "checkable_claims": self.checkable_claims,
            "supported": self.supported,
            "unsupported": self.unsupported,
            "contradicted": self.contradicted,
            "claims": [c.as_dict() for c in self.claims],
            "note": self.note,
        }


DECOMPOSE_SYSTEM = """You split an answer into atomic factual claims.

An atomic claim states exactly one fact and can be judged true or false on its own.

Rules:
1. Split compound sentences. "A operates B and departs from C" is two claims.
2. Resolve pronouns and references so each claim stands alone.
3. Keep the original entity names exactly as written. Do not normalise, expand
   abbreviations, or correct spelling.
4. Mark a claim as checkable=false when it states no verifiable fact: greetings,
   offers of further help, hedges such as "I don't have that information",
   instructions to the reader, or statements about the assistant itself.
5. Do not add claims the answer does not make. Do not infer.

Return JSON: {"claims": [{"text": "...", "checkable": true}]}"""

JUDGE_SYSTEM = """You decide whether the supplied graph facts support a claim.

You are given one claim and a list of facts, each written as
subject — relation — object. These facts come from a validated knowledge graph.

Answer only from the facts given. Do not use outside knowledge. Do not infer
beyond what a fact states.

Return JSON:
{"verdict": "supported" | "unsupported" | "contradicted",
 "fact_indexes": [0, 2],
 "reason": "one short sentence"}

Use "supported" when the listed facts state what the claim states.
Use "contradicted" when a fact states something incompatible with the claim.
Use "unsupported" when the facts neither state nor contradict it — including
when they are merely about the same entities.

fact_indexes must list only the facts you actually relied on."""


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD.findall(text) if len(t) > 1}


def _link_entities(claim: str, node_ids: list[str], limit: int = 8) -> list[str]:
    """Match graph nodes named in a claim.

    Deterministic on purpose: a substring match after normalisation, then a
    token-overlap fallback for multi-word names. Longer identifiers are
    preferred so "Section 194C" wins over a node called "C".
    """
    lowered = claim.lower()
    exact = [n for n in node_ids if len(n) > 2 and n.lower() in lowered]
    if exact:
        return sorted(exact, key=len, reverse=True)[:limit]

    claim_tokens = _tokens(claim)
    scored: list[tuple[float, str]] = []
    for node in node_ids:
        node_tokens = _tokens(node)
        if not node_tokens:
            continue
        overlap = len(node_tokens & claim_tokens) / len(node_tokens)
        if overlap >= 0.75:
            scored.append((overlap * len(node_tokens), node))
    scored.sort(reverse=True)
    return [n for _, n in scored[:limit]]


def _single_valued(edges: list[dict], relation: str,
                   min_subjects: int = 5) -> bool:
    """Does this relation hold at most one object per subject, everywhere?

    `min_subjects` guards against a sparse graph. With two or three examples
    every relation looks single-valued, and concluding so would let an
    incomplete graph brand true statements as contradictions. Below the
    threshold the relation is treated as multi-valued, which downgrades the
    verdict to unsupported — the safer error.
    """
    seen: dict[str, set[str]] = {}
    for e in edges:
        if e["relation"] != relation:
            continue
        seen.setdefault(e["source"], set()).add(e["target"])
    if len(seen) < min_subjects:
        return False
    return all(len(v) == 1 for v in seen.values())


async def verify_answer(domain: str, answer: str, *,
                        subgraph: dict[str, Any] | None = None,
                        max_facts: int = 40) -> VerificationResult:
    """Check an answer against the graph, claim by claim."""
    if not (answer or "").strip():
        return VerificationResult([], 1.0, 0, 0, 0, 0, "Nothing to verify.")

    graph = await get_store().export_json(domain)
    node_ids = [n["id"] for n in graph["nodes"]]
    node_type = {n["id"]: n.get("type", "") for n in graph["nodes"]}
    edges = graph["edges"]
    if not node_ids:
        return VerificationResult([], 0.0, 0, 0, 0, 0,
                                  "This domain has no graph to verify against.")

    by_source: dict[str, list[dict]] = {}
    by_target: dict[str, list[dict]] = {}
    for e in edges:
        by_source.setdefault(e["source"], []).append(e)
        by_target.setdefault(e["target"], []).append(e)

    # ---- 1. decompose -------------------------------------------------
    try:
        with attribute_to("verification", domain=domain):
            decomposed = await complete_json(DECOMPOSE_SYSTEM, answer)
        raw_claims = decomposed.get("claims", []) or []
    except Exception as exc:
        log.warning("Claim decomposition failed: %s", exc)
        return VerificationResult([], 0.0, 0, 0, 0, 0,
                                  f"The answer could not be broken into claims: {exc}")

    claims: list[Claim] = []
    for item in raw_claims:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        if not item.get("checkable", True):
            claims.append(Claim(text=text, status=ClaimStatus.NOT_CHECKABLE,
                                reason="States no verifiable fact."))
            continue
        claims.append(Claim(text=text))

    # ---- 2-4. link, gather candidates, judge ---------------------------
    for claim in claims:
        if claim.status is ClaimStatus.NOT_CHECKABLE:
            continue

        linked = _link_entities(claim.text, node_ids)
        claim.entities = linked
        if not linked:
            claim.reason = "No entity in this claim appears in the graph."
            continue

        candidates: list[dict] = []
        seen: set[tuple] = set()
        for entity in linked:
            for e in by_source.get(entity, []) + by_target.get(entity, []):
                key = (e["source"], e["relation"], e["target"])
                if key not in seen:
                    seen.add(key)
                    candidates.append(e)
        if not candidates:
            claim.reason = ("The entities are in the graph but have no recorded "
                            "relationships.")
            continue

        # Prefer edges between two entities the claim names — those are the ones
        # most likely to decide it.
        linked_set = set(linked)
        candidates.sort(
            key=lambda e: (e["source"] in linked_set and e["target"] in linked_set),
            reverse=True,
        )
        candidates = candidates[:max_facts]

        facts = [
            f"{e['source']} ({node_type.get(e['source'],'')}) "
            f"— {e['relation'].replace('_',' ')} — "
            f"{e['target']} ({node_type.get(e['target'],'')})"
            for e in candidates
        ]
        prompt = ("Claim:\n" + claim.text + "\n\nFacts:\n"
                  + "\n".join(f"[{i}] {f}" for i, f in enumerate(facts)))
        try:
            with attribute_to("verification", domain=domain):
                verdict = await complete_json(JUDGE_SYSTEM, prompt)
        except Exception as exc:
            claim.reason = f"Could not be checked: {exc}"
            continue

        decision = str(verdict.get("verdict", "unsupported")).lower()
        used = [i for i in (verdict.get("fact_indexes") or [])
                if isinstance(i, int) and 0 <= i < len(candidates)]
        claim.evidence = [
            {"source": candidates[i]["source"], "relation": candidates[i]["relation"],
             "target": candidates[i]["target"]}
            for i in used
        ]
        claim.reason = str(verdict.get("reason", ""))[:240]

        if decision == "supported" and claim.evidence:
            claim.status = ClaimStatus.SUPPORTED
        elif decision == "contradicted" and claim.evidence:
            # Only stand behind a contradiction when the relation is
            # single-valued across the whole graph; otherwise the graph simply
            # holds an additional fact, not a conflicting one.
            relation = claim.evidence[0]["relation"]
            if _single_valued(edges, relation):
                claim.status = ClaimStatus.CONTRADICTED
            else:
                claim.status = ClaimStatus.UNSUPPORTED
                claim.reason = (f"{claim.reason} Reported as unsupported rather than "
                                f"contradicted: the graph does not hold enough "
                                f"examples of '{relation.replace('_',' ')}' to treat "
                                f"a differing value as a conflict.")
        elif decision == "supported" and not claim.evidence:
            claim.status = ClaimStatus.UNSUPPORTED
            claim.reason = "Judged supported but no fact was cited."
        else:
            claim.status = ClaimStatus.UNSUPPORTED

    checkable = [c for c in claims if c.status is not ClaimStatus.NOT_CHECKABLE]
    supported = sum(1 for c in checkable if c.status is ClaimStatus.SUPPORTED)
    contradicted = sum(1 for c in checkable if c.status is ClaimStatus.CONTRADICTED)
    unsupported = len(checkable) - supported - contradicted
    score = round(supported / len(checkable), 4) if checkable else 1.0

    note = ""
    if not checkable:
        note = "The answer makes no verifiable factual claims."
    elif contradicted:
        note = ("At least one claim conflicts with a recorded fact. Treat this "
                "answer as unsafe until reviewed.")
    elif unsupported:
        note = ("Some claims could not be traced to a recorded fact. This may be a "
                "gap in the graph rather than an error in the answer.")

    return VerificationResult(claims, score, len(checkable), supported,
                              unsupported, contradicted, note)
