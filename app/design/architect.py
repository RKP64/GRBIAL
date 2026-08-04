"""Proposing a design from documents and a stated goal.

The hard part of standing up a new domain is not choosing agents — it is
designing the ontology. That is where someone new stalls, because it asks them
to decide entity types, relationships and identifier conventions before they
have seen what the corpus actually contains.

This reverses that: show it a sample, say what you want to do, and it proposes
a schema, the agents that would use it, and how they would work together.

Two properties are deliberate and not negotiable:

  * It proposes; it never applies. A reviewed schema is the whole governance
    argument, and an agent quietly creating domains would hollow it out. The
    result is a draft the operator edits and saves through the normal path.
  * Every proposed entity type carries evidence from the sample. A reviewer can
    then check reasoning rather than accept a plausible-looking blob — and a
    type with no evidence is the clearest signal that the model invented it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..agent.registry import ALL_TOOLS
from ..ontology.registry import KEY_RE
from ..services.llm import complete_json
from ..services.parsing import chunk_file
from ..usage.recorder import attribute_to

log = logging.getLogger(__name__)

MAX_SAMPLE_CHARS = 24_000

SYSTEM = """You design knowledge-graph solutions. You are shown samples of a document
collection and told what someone wants to do with it. You propose a design.

You are NOT extracting data. You are deciding what KINDS of things this
collection contains and how they relate.

=== WHAT YOU ARE DESIGNING ===

1. An ONTOLOGY — the schema the extractor will be held to:
   - entity types (8-14; fewer broad types beat many narrow ones)
   - for each: a one-line definition, an identifier rule saying how instances
     are named, up to three examples taken from the sample, and a short verbatim
     quote from the sample that shows this type really occurs
   - relationships as (source type, relation, target type) triples, using
     snake_case verb phrases. Only relationships the sample actually evidences.
   - identifier rules as regular expressions where the collection has a naming
     convention worth enforcing (reference numbers, codes, formats)

2. AGENTS — one to four, each with a clear job, a plain-language instruction,
   and the tools it needs from: %(tools)s

3. A TEAM — only if the goal genuinely needs more than one agent working in
   sequence. Say which agent starts, and who may hand to whom and when.
   If one agent suffices, return an empty team. Do not invent handoffs.

=== RULES ===
- Propose only what the sample evidences. If you are unsure a type occurs, leave
  it out and note it under `uncertain`.
- Never invent example values. Take them from the sample or omit them.
- Identifier rules must be truthful to what you saw, not aspirational.
- Keys are lowercase with hyphens or underscores.

Return a single JSON object:
{
  "domain_key": "...", "domain_name": "...", "domain_description": "...",
  "reasoning": "two or three sentences on why this shape suits the goal",
  "entity_types": [
    {"name": "...", "definition": "...", "id_rule": "...",
     "examples": ["..."], "evidence": "short verbatim quote from the sample"}
  ],
  "relationships": [{"source": "...", "relation": "...", "target": "..."}],
  "id_transforms": [{"pattern": "...", "replace": "...", "case": "",
                     "note": "what this normalises"}],
  "normalization_rules": ["..."],
  "agents": [
    {"key": "...", "name": "...", "description": "...", "system_prompt": "...",
     "tools": ["..."], "starters": ["..."]}
  ],
  "team": {"key": "...", "name": "...", "description": "...", "entry": "...",
           "members": [{"agent": "...", "hands_off_to": ["..."], "when": "..."}]},
  "uncertain": ["things you could not tell from the sample"]
}
No prose outside the JSON.""" % {"tools": ", ".join(ALL_TOOLS)}


@dataclass
class Proposal:
    ontology: dict[str, Any]
    agents: list[dict[str, Any]]
    team: dict[str, Any] | None
    reasoning: str
    uncertain: list[str]
    warnings: list[str] = field(default_factory=list)
    sample: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"ontology": self.ontology, "agents": self.agents, "team": self.team,
                "reasoning": self.reasoning, "uncertain": self.uncertain,
                "warnings": self.warnings, "sample": self.sample,
                "applied": False,
                "note": ("This is a draft. Nothing has been created. Review it, "
                         "change what is wrong, then save it.")}


def build_sample(uploads: list[tuple[str, bytes]], *, rows_per_chunk: int,
                 chunk_size: int, overlap: int) -> tuple[str, dict[str, Any]]:
    """Take a spread across the files rather than the first N characters.

    The opening of a document is often a cover page or a header row, which tells
    a designer very little about what the collection contains.
    """
    per_file: list[tuple[str, list[str]]] = []
    for filename, data in uploads:
        try:
            chunks = chunk_file(filename, data, rows_per_chunk=rows_per_chunk,
                                chunk_size=chunk_size, overlap=overlap)
        except Exception as exc:
            log.warning("Sampling skipped %s: %s", filename, exc)
            continue
        if chunks:
            per_file.append((filename, [c.text for c in chunks]))

    if not per_file:
        raise ValueError("Nothing readable was found in those files.")

    budget = MAX_SAMPLE_CHARS // max(1, len(per_file))
    parts: list[str] = []
    stats = {"files": len(per_file), "chunks_seen": 0}
    for filename, chunks in per_file:
        stats["chunks_seen"] += len(chunks)
        picks = chunks if len(chunks) <= 3 else [
            chunks[0], chunks[len(chunks) // 2], chunks[-1]
        ]
        share = budget // len(picks)
        for i, text in enumerate(picks):
            parts.append(f"--- {filename} (extract {i + 1}) ---\n{text[:share]}")
    sample = "\n\n".join(parts)[:MAX_SAMPLE_CHARS]
    stats["sample_chars"] = len(sample)
    return sample, stats


async def propose(uploads: list[tuple[str, bytes]], goal: str, *,
                  rows_per_chunk: int = 50, chunk_size: int = 2000,
                  overlap: int = 200) -> Proposal:
    sample, stats = build_sample(uploads, rows_per_chunk=rows_per_chunk,
                                 chunk_size=chunk_size, overlap=overlap)
    prompt = (f"What they want to do:\n{goal.strip() or 'Not stated.'}\n\n"
              f"Sample of the collection:\n{sample}")
    with attribute_to("design"):
        raw = await complete_json(SYSTEM, prompt, temperature=0.2)

    warnings: list[str] = []
    entity_types: dict[str, Any] = {}
    evidence: dict[str, str] = {}
    for item in raw.get("entity_types", []) or []:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        entity_types[name] = {
            "id_rule": str(item.get("id_rule", "")).strip(),
            "examples": [str(e) for e in (item.get("examples") or [])][:3],
        }
        quote = str(item.get("evidence", "")).strip()
        evidence[name] = quote
        if not quote:
            warnings.append(
                f"'{name}' was proposed without a supporting quote from the sample. "
                f"Check that it really occurs before keeping it."
            )

    triples = []
    for t in raw.get("relationships", []) or []:
        source, relation, target = (str(t.get("source", "")).strip(),
                                    str(t.get("relation", "")).strip(),
                                    str(t.get("target", "")).strip())
        if not (source and relation and target):
            continue
        missing = [x for x in (source, target) if x not in entity_types]
        if missing:
            warnings.append(
                f"Relationship '{source} {relation} {target}' refers to "
                f"{', '.join(missing)}, which is not a proposed type. It has been "
                f"left out."
            )
            continue
        triples.append({"source": source, "relation": relation, "target": target})

    key = str(raw.get("domain_key", "")).strip().lower().replace(" ", "_")
    if not KEY_RE.fullmatch(key):
        key = "new_domain"
        warnings.append("The proposed key was not usable; 'new_domain' was "
                        "substituted. Rename it before saving.")

    transforms = []
    import re as _re
    for t in raw.get("id_transforms", []) or []:
        pattern = str(t.get("pattern", ""))
        try:
            _re.compile(pattern)
        except _re.error as exc:
            warnings.append(f"An identifier rule was dropped — /{pattern}/ is not "
                            f"a valid expression ({exc}).")
            continue
        transforms.append({"pattern": pattern, "replace": str(t.get("replace", "")),
                           "case": str(t.get("case", "")),
                           "note": str(t.get("note", ""))})

    ontology = {
        "key": key,
        "name": str(raw.get("domain_name") or key).strip(),
        "description": str(raw.get("domain_description", "")).strip(),
        "entity_types": entity_types,
        "allowed_triples": triples,
        "normalization_rules": [str(r) for r in (raw.get("normalization_rules") or [])],
        "id_transforms": transforms,
        "strip_type_prefixes": True,
        "collapse_whitespace": True,
        "open_relations": False,
        "custom_prompt": "",
        "evidence": evidence,
    }

    agents = []
    for a in raw.get("agents", []) or []:
        agent_key = str(a.get("key", "")).strip().lower().replace(" ", "-")
        if not KEY_RE.fullmatch(agent_key):
            continue
        tools = [t for t in (a.get("tools") or ALL_TOOLS) if t in ALL_TOOLS] or list(ALL_TOOLS)
        agents.append({
            "key": agent_key, "name": str(a.get("name") or agent_key).strip(),
            "description": str(a.get("description", "")).strip(),
            "domains": [key],
            "system_prompt": str(a.get("system_prompt", "")).strip(),
            "tools": tools, "max_steps": 6, "temperature": 0.2, "verify": True,
            "starters": [str(s) for s in (a.get("starters") or [])][:4],
        })

    team = None
    raw_team = raw.get("team") or {}
    members = [m for m in (raw_team.get("members") or [])
               if str(m.get("agent", "")) in {a["key"] for a in agents}]
    if len(members) > 1:
        agent_keys = {a["key"] for a in agents}
        cleaned = [{
            "agent": m["agent"],
            "hands_off_to": [t for t in (m.get("hands_off_to") or [])
                             if t in agent_keys and t != m["agent"]],
            "when": str(m.get("when", "")).strip(),
        } for m in members]
        entry = str(raw_team.get("entry") or cleaned[0]["agent"])
        if entry not in agent_keys:
            entry = cleaned[0]["agent"]
        team_key = str(raw_team.get("key", "")).strip().lower().replace(" ", "-")
        if not KEY_RE.fullmatch(team_key):
            team_key = f"{key.replace('_', '-')}-team"
        team = {"key": team_key,
                "name": str(raw_team.get("name") or team_key).strip(),
                "description": str(raw_team.get("description", "")).strip(),
                "entry": entry, "members": cleaned,
                "max_handoffs": 6, "max_steps_per_agent": 4, "verify": True,
                "starters": []}
    elif raw_team.get("members"):
        warnings.append("A team was proposed but only one agent survived review, "
                        "so it has been dropped. One agent is often enough.")

    if not entity_types:
        warnings.append("No entity types could be proposed from this sample. It may "
                        "be too small, or not contain recognisable records.")

    return Proposal(ontology=ontology, agents=agents, team=team,
                    reasoning=str(raw.get("reasoning", "")).strip(),
                    uncertain=[str(u) for u in (raw.get("uncertain") or [])],
                    warnings=warnings, sample=stats)
