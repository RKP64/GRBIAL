"""Read an ontology someone already wrote, in whatever shape they wrote it.

Business teams hand over schemas as spreadsheets far more often than as JSON —
a Node Types sheet, a Relationships sheet, columns named whatever made sense at
the time. Retyping fifty rows into a form is where these projects lose an
afternoon and gain a transcription error.

Two paths:

  JSON  — parsed directly. No model involved, so a file this platform exported
          always round-trips exactly.
  Other — the sheet or document is read to text and a model maps it onto the
          schema contract. Column names, sheet names and ordering are free.

Nothing here saves. The caller gets a draft to review in the editor first,
because a mapped schema is a proposal, not a fact.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..services.llm import complete_json
from ..services.parsing import chunk_file
from ..usage.recorder import attribute_to

MAX_SAMPLE_CHARS = 24_000
KEY_RE = re.compile(r"[^a-z0-9_-]")

SYSTEM = """You map a schema that a domain expert has already written onto a fixed contract.

They may have used a spreadsheet, a table, or prose. Column headings and sheet
names vary. Your job is to recognise which of their columns means what, not to
invent a schema of your own.

Return JSON only, in this shape:

{
  "key": "short-slug",
  "name": "Readable name",
  "description": "One or two sentences on what this domain covers.",
  "entity_types": [
    {"name": "TypeName", "id_rule": "", "examples": ["...", "..."]}
  ],
  "allowed_triples": [
    {"source": "TypeName", "relation": "relation_name", "target": "TypeName"}
  ],
  "normalization_rules": ["..."],
  "notes": ["anything you could not map, or had to guess at"]
}

Rules that matter:

- Entity type names keep the author's capitalisation: CaseLaw stays CaseLaw.
- Relation names become lower_snake_case: PART_OF becomes part_of.
- Every triple's source and target MUST be a name that appears in entity_types.
  Drop any triple that references a type they never defined, and say so in notes.
- Only include what is in their document. Do not add types or relations that
  seem like they belong. If a column is empty, leave the field empty.
- If they marked rows as planned, deferred or out of scope, include them but
  record that in notes.
- examples are illustrative values for that type, not descriptions of it.
"""


def _slug(value: str, fallback: str = "imported-domain") -> str:
    out = KEY_RE.sub("", (value or "").strip().lower().replace(" ", "-")).strip("-")
    return out[:49] or fallback


def _clean_relation(value: str) -> str:
    out = re.sub(r"[\s-]+", "_", (value or "").strip().lower())
    return re.sub(r"[^a-z0-9_]", "", out)


def normalise(raw: dict[str, Any], *, fallback_key: str) -> dict[str, Any]:
    """Turn a model's answer into the shape the ontology endpoint accepts.

    Applied to LLM output and to hand-written JSON alike, so an imported file
    that is slightly off-contract still lands rather than erroring.
    """
    notes: list[str] = [str(n) for n in (raw.get("notes") or []) if str(n).strip()]

    # entity_types arrives either as a list of objects or already as a mapping.
    entity_types: dict[str, dict[str, Any]] = {}
    incoming = raw.get("entity_types") or {}
    if isinstance(incoming, dict):
        for name, spec in incoming.items():
            name = str(name).strip()
            if not name:
                continue
            spec = spec if isinstance(spec, dict) else {}
            entity_types[name] = {
                "id_rule": str(spec.get("id_rule", "")).strip(),
                "examples": [str(e).strip() for e in (spec.get("examples") or [])
                             if str(e).strip()][:3],
            }
    else:
        for item in incoming:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            entity_types[name] = {
                "id_rule": str(item.get("id_rule", "")).strip(),
                "examples": [str(e).strip() for e in (item.get("examples") or [])
                             if str(e).strip()][:3],
            }

    # Triples must reference declared types, or extraction will reject them
    # later with a message that is hard to trace back to this import.
    triples: list[dict[str, str]] = []
    dropped: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw.get("allowed_triples") or []:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            source, relation, target = (str(x).strip() for x in item)
        elif isinstance(item, dict):
            source = str(item.get("source", "")).strip()
            relation = str(item.get("relation", "")).strip()
            target = str(item.get("target", "")).strip()
        else:
            continue
        relation = _clean_relation(relation)
        if not (source and relation and target):
            continue
        if source not in entity_types or target not in entity_types:
            dropped.append(f"{source} -[{relation}]-> {target}")
            continue
        key = (source, relation, target)
        if key in seen:
            continue
        seen.add(key)
        triples.append({"source": source, "relation": relation, "target": target})

    if dropped:
        shown = ", ".join(dropped[:5])
        more = f" and {len(dropped) - 5} more" if len(dropped) > 5 else ""
        notes.append(
            f"{len(dropped)} relationship(s) referenced a type that was never "
            f"defined and were left out: {shown}{more}."
        )

    return {
        "key": _slug(str(raw.get("key", "")), fallback_key),
        "name": str(raw.get("name", "")).strip() or fallback_key,
        "description": str(raw.get("description", "")).strip(),
        "entity_types": entity_types,
        "allowed_triples": triples,
        "normalization_rules": [str(r).strip() for r in
                                (raw.get("normalization_rules") or []) if str(r).strip()],
        "id_transforms": raw.get("id_transforms") or [],
        "strip_type_prefixes": bool(raw.get("strip_type_prefixes", True)),
        "collapse_whitespace": bool(raw.get("collapse_whitespace", True)),
        "open_relations": bool(raw.get("open_relations", False)),
        "custom_prompt": str(raw.get("custom_prompt", "")).strip(),
        "notes": notes,
    }


def _to_text(filename: str, data: bytes, *, rows_per_chunk: int,
             chunk_size: int, overlap: int) -> str:
    """Flatten a workbook or document to text the model can read.

    Sheets are read in order and the beginning of each is kept, since schema
    documents put the definitions at the top and examples further down.
    """
    chunks = chunk_file(filename, data, rows_per_chunk=rows_per_chunk,
                        chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return ""
    budget = MAX_SAMPLE_CHARS
    parts: list[str] = []
    for chunk in chunks:
        text = (chunk.text or "").strip()
        if not text:
            continue
        parts.append(text)
        budget -= len(text)
        if budget <= 0:
            break
    return "\n\n".join(parts)[:MAX_SAMPLE_CHARS]


async def import_ontology(filename: str, data: bytes, *, rows_per_chunk: int = 50,
                          chunk_size: int = 2000, overlap: int = 200) -> dict[str, Any]:
    """Produce a draft ontology from an uploaded file. Never saves."""
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    fallback_key = _slug(stem)
    lowered = filename.lower()

    if lowered.endswith(".json"):
        try:
            raw = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"That file is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("Expected a JSON object describing one ontology.")
        draft = normalise(raw, fallback_key=fallback_key)
        draft["source"] = "json"
        return draft

    text = _to_text(filename, data, rows_per_chunk=rows_per_chunk,
                    chunk_size=chunk_size, overlap=overlap)
    if not text.strip():
        raise ValueError("Nothing readable was found in that file.")

    prompt = (f"File name: {filename}\n\n"
              f"Their schema document:\n{text}")
    with attribute_to("design"):
        raw = await complete_json(SYSTEM, prompt, temperature=0.1)

    draft = normalise(raw, fallback_key=fallback_key)
    draft["source"] = "mapped"
    if not draft["entity_types"]:
        raise ValueError(
            "No entity types could be read from that file. Check that the sheet "
            "listing them is present, or import a JSON ontology instead."
        )
    return draft
