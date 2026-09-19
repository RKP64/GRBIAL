"""Deciding whether a newly extracted entity is one the graph already holds.

Without this the graph merges on exact id, so "Tata Steel", "Tata Steel Ltd."
and "TATA STEEL LIMITED" become three vendors. Nothing errors; an agent asked
which vendors serve a terminal simply returns a third of the answer with no
indication anything is missing. This is the step the field consistently reports
as where enterprise graph builds stall, and it is worse than extraction failure
because it is silent.

Three stages, cheapest first, and each only runs on what the previous one could
not settle:

  normalise  — casing, punctuation, and the legal suffixes that make the same
               company look like three. Free, and settles most of it.
  similarity — character-level comparison for typos and truncations. Free.
  adjudicate — a model decides the genuinely ambiguous ones. Costs a call, so it
               is reserved for the narrow band where the answer is not obvious.

The asymmetry that shapes every threshold here: leaving two records separate is
recoverable, merging two that are different is not. A merge silently attributes
one vendor's contracts to another, and nobody finds out. So the defaults are
deliberately conservative and the uncertain band is sent for review rather than
guessed.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..config import get_settings

log = logging.getLogger(__name__)

# Corporate and organisational suffixes that carry no distinguishing meaning.
# "Tata Steel" and "Tata Steel Ltd." are the same company.
SUFFIXES = {
    "ltd", "limited", "llp", "llc", "inc", "incorporated", "corp", "corporation",
    "co", "company", "plc", "pvt", "private", "gmbh", "sa", "nv", "bv", "ag",
    "pte", "sdn", "bhd", "srl", "spa", "oy", "ab", "as",
}

ARTICLES = {"the", "a", "an", "of", "and"}

# Above this, the same entity beyond reasonable doubt. Set high on purpose.
AUTO_MERGE = 0.94
# Below this, different. Between the two is the band worth a model's opinion.
CLEARLY_DIFFERENT = 0.80


def normalise(value: str) -> str:
    """Reduce a name to what actually distinguishes it."""
    text = re.sub(r"[^\w\s]", " ", str(value or "").lower())
    words = [w for w in text.split() if w and w not in SUFFIXES and w not in ARTICLES]
    return " ".join(words)


def _differs_only_by_number(a: str, b: str) -> bool:
    """True when two names are identical apart from their numbers.

    Gate A12 and Gate A13, Terminal 1 and Terminal 2, Section 45 and Section 47.
    These score highly on character overlap and are never the same thing, so
    without this every adjacent pair in a numbered series lands in the review
    queue — thirty-nine gates would generate dozens of pointless adjudications.
    """
    strip = lambda t: re.sub(r"\d+", "#", t)
    if strip(a) != strip(b):
        return False
    return re.findall(r"\d+", a) != re.findall(r"\d+", b)


def similarity(a: str, b: str) -> float:
    """0-1 on the normalised forms.

    Containment is treated as strong evidence, because truncation is the common
    real case — "Bangalore International Airport" against "Bangalore
    International Airport Limited" scores poorly on raw character overlap while
    plainly being the same thing.
    """
    na, nb = normalise(a), normalise(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    # Checked before containment: "Gate A1" is contained in "Gate A12" and would
    # otherwise score as a near-certain match.
    if _differs_only_by_number(na, nb):
        return 0.0
    if na in nb or nb in na:
        shorter, longer = sorted((len(na), len(nb)))
        return 0.90 + 0.09 * (shorter / longer)
    return SequenceMatcher(None, na, nb).ratio()


@dataclass
class Candidate:
    incoming: str
    existing: str
    entity_type: str
    score: float
    decision: str          # merge | separate | review
    reason: str = ""


@dataclass
class PendingMerge:
    """An ambiguous pair awaiting a person.

    Held rather than decided because an automatic answer here would be a guess,
    and the cost of guessing wrong is asymmetric.
    """
    id: str
    domain: str
    incoming: str
    existing: str
    entity_type: str
    score: float
    reason: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())


ADJUDICATE_SYSTEM = """You decide whether two names refer to the same real thing.

Return JSON only: {"same": true|false, "reason": "one short sentence"}

Say true only when they are plainly the same entity written differently —
abbreviation, legal suffix, punctuation, word order, a typo.

Say false when they might be related but distinct: a parent and its subsidiary,
two branches, two gates in the same terminal, sequential identifiers.

When genuinely uncertain, say false. Keeping two records apart can be corrected
later; merging two different ones silently attributes the wrong facts and is
usually never noticed.
"""


class Resolver:
    def __init__(self, domain: str) -> None:
        self.domain = domain
        self._queue = get_settings().data_dir / "resolution" / f"{domain}.json"
        self._queue.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- pending queue

    def pending(self) -> list[dict[str, Any]]:
        if not self._queue.exists():
            return []
        try:
            return json.loads(self._queue.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Resolution queue for %s is unreadable", self.domain)
            return []

    def _write(self, rows: list[dict[str, Any]]) -> None:
        self._queue.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def enqueue(self, merge: PendingMerge) -> None:
        rows = self.pending()
        # Same pair may be proposed by several chunks; keep one.
        if any(r["incoming"] == merge.incoming and r["existing"] == merge.existing
               for r in rows):
            return
        rows.append(asdict(merge))
        self._write(rows)

    def dismiss(self, merge_id: str) -> None:
        self._write([r for r in self.pending() if r["id"] != merge_id])

    # ---------------------------------------------------------- resolution

    async def resolve(self, incoming: str, entity_type: str,
                      existing: dict[str, dict], *,
                      adjudicate: bool = True) -> Candidate:
        """Decide what an incoming entity is, against what the graph already holds."""
        if incoming in existing:
            return Candidate(incoming, incoming, entity_type, 1.0, "merge",
                             "identical id")

        # Only same-type nodes are candidates. A Gate is never an Outlet, and
        # comparing across types produces confident nonsense.
        same_type = [nid for nid, node in existing.items()
                     if not entity_type or node.get("type") == entity_type]

        best, best_score = "", 0.0
        for nid in same_type:
            score = similarity(incoming, nid)
            if score > best_score:
                best, best_score = nid, score

        if not best or best_score < CLEARLY_DIFFERENT:
            return Candidate(incoming, "", entity_type, best_score, "separate",
                             "no close match")

        if best_score >= AUTO_MERGE:
            return Candidate(incoming, best, entity_type, best_score, "merge",
                             "names match after normalisation")

        if not adjudicate:
            return Candidate(incoming, best, entity_type, best_score, "review",
                             "close but not certain")

        same, reason = await self._adjudicate(incoming, best, entity_type)
        if same is None:
            return Candidate(incoming, best, entity_type, best_score, "review",
                             f"could not be judged: {reason}")
        return Candidate(incoming, best, entity_type, best_score,
                         "merge" if same else "separate", reason)

    async def _adjudicate(self, a: str, b: str,
                          entity_type: str) -> tuple[bool | None, str]:
        from ..services.llm import complete_json
        prompt = (f"Entity type: {entity_type or 'unknown'}\n"
                  f"Name A: {a}\nName B: {b}")
        try:
            raw = await complete_json(ADJUDICATE_SYSTEM, prompt, temperature=0.0)
        except Exception as exc:
            log.warning("Adjudication failed for %r / %r: %s", a, b, exc)
            return None, str(exc)
        return bool(raw.get("same")), str(raw.get("reason", "")).strip()


async def resolve_batch(domain: str, nodes: list, *, adjudicate: bool = True,
                        queue_reviews: bool = True) -> tuple[list, list[Candidate]]:
    """Rewrite a batch of extracted nodes onto their resolved identities.

    Returns the nodes with ids replaced where a merge was decided, and the
    decisions, so a caller can report what happened rather than the graph
    changing invisibly.
    """
    import uuid

    from ..stores import get_store

    graph = await get_store().export_json(domain)
    existing = {n["id"]: n for n in graph.get("nodes", [])}

    resolver = Resolver(domain)
    decisions: list[Candidate] = []
    out = []

    for node in nodes:
        decision = await resolver.resolve(node.id, node.type, existing,
                                          adjudicate=adjudicate)
        decisions.append(decision)

        if decision.decision == "merge" and decision.existing:
            # Point the node at the identity already in the graph, so the store's
            # own merge-by-id path handles the rest.
            merged = node.model_copy(update={"id": decision.existing})
            if node.id != decision.existing:
                merged.metadata = {**(node.metadata or {}),
                                   "also_seen_as": node.id}
            out.append(merged)
            continue

        if decision.decision == "review" and queue_reviews:
            resolver.enqueue(PendingMerge(
                id=uuid.uuid4().hex[:12], domain=domain,
                incoming=node.id, existing=decision.existing,
                entity_type=node.type, score=round(decision.score, 3),
                reason=decision.reason))
        # Kept separate either way. A pending review does not block ingestion —
        # the entity enters the graph under its own name and can be merged later.
        out.append(node)
        existing[node.id] = {"id": node.id, "type": node.type}

    return out, decisions
