"""Training-data generation from the knowledge graph.

The graph is a generator, not just a store: every validated edge is a fact that
can be templated into a question and answer, every two-step path into a
reasoning example, and every *absence* into a refusal example. Because each pair
traces back to an edge that survived schema validation, the corpus carries
provenance the source text alone cannot give you.

Three deliberate choices:

* Question phrasings are varied per relation. A single template teaches the
  sentence shape as much as the fact, and the model then fails on paraphrases.
* Passages captured during ingestion are optionally included, so the model also
  learns the source's own wording, not only the ontology's abstraction.
* Volatile facts are excluded by rule. A model that memorises a value which
  changes weekly is confidently wrong the following week; those questions belong
  to retrieval at answer time, not to the weights.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..ontology import get_ontology
from ..retrieval import get_chunk_index
from ..stores import get_store

# Phrasings per relation direction. Two constraints shaped this set:
#
#   * They differ in structure, not just wording, so the model generalises past
#     a single question shape.
#   * None of them place an auxiliary verb before the relation. Relation names
#     come from the ontology already inflected ("applies_to", "concerns"), and
#     "What does X applies to?" would teach broken grammar. Where a relation has
#     to sit next to a noun that may be plural, it is quoted instead.
FORWARD_TEMPLATES = [
    "{source} {relation} what?",
    "Tell me what {source} {relation}.",
    "Which {target_type} does {source} link to under '{relation}'?",
    "For {source}, what is recorded under '{relation}'?",
]
INVERSE_TEMPLATES = [
    "Which {source_type} {relation} {target}?",
    "What {relation} {target}?",
    "Find what {relation} {target}.",
]
AGGREGATE_TEMPLATES = [
    "List everything that {relation} {target}.",
    "Which {source_type} records link to {target} under '{relation}'?",
    "Give me every {source_type} that {relation} {target}.",
]
REFUSAL_TEMPLATES = [
    "{source} {relation} what?",
    "For {source}, what is recorded under '{relation}'?",
    "Which {target_type} does {source} link to under '{relation}'?",
]

DEFAULT_SYSTEM = (
    "You are a domain assistant. Answer questions about {domain} accurately and "
    "concisely. If you do not have the information, say so plainly."
)


def _humanise(relation: str) -> str:
    return relation.replace("_", " ").strip()


@dataclass
class GenerationOptions:
    include_facts: bool = True
    include_inverse: bool = True
    include_multihop: bool = True
    include_refusals: bool = True
    include_passages: bool = False
    max_per_relation: int = 400
    refusal_fraction: float = 0.08
    validation_fraction: float = 0.05
    seed: int = 42
    system_prompt: str = ""
    exclude_types: list[str] = field(default_factory=list)


@dataclass
class Pair:
    question: str
    answer: str
    kind: str          # fact | inverse | multihop | refusal | passage

    def as_record(self, system: str) -> dict:
        return {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": self.question},
            {"role": "assistant", "content": self.answer},
        ]}


class DatasetBuilder:
    def __init__(self, data_dir: Path) -> None:
        self.dir = Path(data_dir) / "datasets"
        self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ generate
    async def generate(self, domain: str, options: GenerationOptions) -> dict[str, Any]:
        rng = random.Random(options.seed)
        graph = await get_store().export_json(domain)
        nodes = {n["id"]: n for n in graph["nodes"]
                 if n.get("type") not in options.exclude_types}
        edges = [e for e in graph["edges"]
                 if e["source"] in nodes and e["target"] in nodes]
        if not edges:
            raise ValueError(
                "This graph has no relationships to learn from yet. Extract more "
                "documents, or check that the ontology permits the relationships "
                "the source material contains."
            )

        ontology = get_ontology(domain)
        system = options.system_prompt.strip() or DEFAULT_SYSTEM.format(
            domain=ontology.name or domain
        )

        by_relation: dict[str, list[dict]] = defaultdict(list)
        for e in edges:
            by_relation[e["relation"]].append(e)

        pairs: list[Pair] = []
        if options.include_facts:
            pairs += self._facts(by_relation, nodes, rng, options.max_per_relation)
        if options.include_inverse:
            pairs += self._inverse(by_relation, nodes, rng, options.max_per_relation)
        if options.include_multihop:
            pairs += self._multihop(edges, nodes, rng, options.max_per_relation)
        if options.include_refusals:
            # Refusals must stay a small minority. Train on too many and the
            # model learns that declining is usually right, which is worse than
            # never having taught it at all.
            budget = max(3, int(len(pairs) * options.refusal_fraction))
            pairs += self._refusals(by_relation, nodes, rng, cap=budget)
        if options.include_passages:
            pairs += await self._passages(domain, rng)

        if not pairs:
            raise ValueError("No training pairs could be generated with these options.")

        rng.shuffle(pairs)
        n_val = max(10, int(len(pairs) * options.validation_fraction))
        n_val = min(n_val, len(pairs) // 4) or 1
        val, train = pairs[:n_val], pairs[n_val:]

        stamp = datetime.now(timezone.utc)
        dataset_id = f"{domain}-{stamp.strftime('%Y%m%d-%H%M%S')}"
        self._write(dataset_id, "train", train, system)
        self._write(dataset_id, "validation", val, system)

        breakdown: dict[str, int] = defaultdict(int)
        for p in pairs:
            breakdown[p.kind] += 1

        meta = {
            "id": dataset_id,
            "domain": domain,
            "created_at": stamp.isoformat(),
            "system_prompt": system,
            "train_examples": len(train),
            "validation_examples": len(val),
            "breakdown": dict(breakdown),
            "options": options.__dict__,
            "graph": {"nodes": len(nodes), "edges": len(edges)},
        }
        (self.dir / f"{dataset_id}.meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    # ------------------------------------------------------------ builders
    def _facts(self, by_relation, nodes, rng, cap) -> list[Pair]:
        out: list[Pair] = []
        for relation, group in by_relation.items():
            for e in group[:cap]:
                src, tgt = nodes[e["source"]], nodes[e["target"]]
                template = rng.choice(FORWARD_TEMPLATES)
                question = template.format(
                    source=src["id"], relation=_humanise(relation),
                    target_type=tgt.get("type", "entity"),
                )
                suffix = f" ({tgt['type']})" if tgt.get("type") else ""
                answer = f"{src['id']} {_humanise(relation)} {tgt['id']}{suffix}."
                out.append(Pair(question, answer, "fact"))
        return out

    def _inverse(self, by_relation, nodes, rng, cap) -> list[Pair]:
        out: list[Pair] = []
        for relation, group in by_relation.items():
            incoming: dict[str, list[str]] = defaultdict(list)
            for e in group:
                incoming[e["target"]].append(e["source"])
            for target, sources in list(incoming.items())[:cap]:
                if not sources:
                    continue
                tgt = nodes[target]
                src_type = nodes[sources[0]].get("type", "entity")
                templates = AGGREGATE_TEMPLATES if len(sources) > 1 else INVERSE_TEMPLATES
                question = rng.choice(templates).format(
                    source_type=src_type, relation=_humanise(relation), target=tgt["id"],
                )
                listed = sorted(set(sources))
                shown = listed[:12]
                extra = len(listed) - len(shown)
                more = f", and {extra} others" if extra > 0 else ""
                if len(listed) == 1:
                    answer = f"{shown[0]} {_humanise(relation)} {tgt['id']}."
                else:
                    # A list subject cannot take the relation's singular verb, so
                    # the relation is stated once as a label instead.
                    answer = (f"Under '{_humanise(relation)}', {tgt['id']} is linked "
                              f"from: {', '.join(shown)}{more}.")
                out.append(Pair(question, answer, "inverse"))
        return out

    def _multihop(self, edges, nodes, rng, cap) -> list[Pair]:
        out_edges: dict[str, list[dict]] = defaultdict(list)
        for e in edges:
            out_edges[e["source"]].append(e)
        pairs: list[Pair] = []
        for first in edges:
            for second in out_edges.get(first["target"], []):
                if second["target"] == first["source"]:
                    continue
                a, b, c = nodes[first["source"]], nodes[first["target"]], nodes[second["target"]]
                question = (f"{a['id']} {_humanise(first['relation'])} something that "
                            f"{_humanise(second['relation'])} what?")
                answer = (f"{a['id']} {_humanise(first['relation'])} {b['id']}, and "
                          f"{b['id']} {_humanise(second['relation'])} {c['id']}.")
                pairs.append(Pair(question, answer, "multihop"))
                if len(pairs) >= cap:
                    return pairs
        return pairs

    def _refusals(self, by_relation, nodes, rng, cap: int = 60) -> list[Pair]:
        """Ask about relation and entity-type combinations the graph never uses.

        Teaching the shape of "I don't know" is otherwise very hard: corpora
        contain answers, not absences. The safety condition is truthfulness — a
        refusal must be right. Asking a *type* that never participates in a
        relation is safe; asking a specific entity that simply has no edge yet
        is not, because the answer may exist in the source and merely be missing
        from this extraction.
        """
        observed: set[tuple[str, str]] = set()
        by_type: dict[str, list[str]] = defaultdict(list)
        target_type_for: dict[str, str] = {}
        for relation, group in by_relation.items():
            for e in group:
                src_type = nodes[e["source"]].get("type", "")
                observed.add((src_type, relation))
                target_type_for.setdefault(relation, nodes[e["target"]].get("type", "entity"))
        for node_id, node in nodes.items():
            by_type[node.get("type", "")].append(node_id)

        candidates = [
            (src_type, relation)
            for src_type in by_type
            for relation in by_relation
            if src_type and (src_type, relation) not in observed
        ]
        rng.shuffle(candidates)

        out: list[Pair] = []
        for src_type, relation in candidates:
            pool = by_type[src_type]
            if not pool:
                continue
            source = rng.choice(pool)
            question = rng.choice(REFUSAL_TEMPLATES).format(
                source=source, relation=_humanise(relation),
                target_type=target_type_for.get(relation, "entity"),
            )
            answer = (f"I don't have any information recorded for {source} under "
                      f"'{_humanise(relation)}'.")
            out.append(Pair(question, answer, "refusal"))
            if len(out) >= cap:
                break
        return out

    async def _passages(self, domain: str, rng) -> list[Pair]:
        store = get_chunk_index()
        if not store.exists(domain):
            return []
        meta = store._load_meta(domain)          # noqa: SLF001 — same package
        out: list[Pair] = []
        for p in meta["passages"][:500]:
            text = " ".join(p["text"].split())
            if len(text) < 120:
                continue
            first = re.split(r"(?<=[.!?])\s+", text)[0][:200]
            out.append(Pair(
                f"What does {p.get('source', 'the source document')} say about "
                f"{first[:60].rstrip('.')}?",
                text[:1200], "passage",
            ))
        return out

    # ------------------------------------------------------------ storage
    def _write(self, dataset_id: str, split: str, pairs: list[Pair], system: str) -> Path:
        path = self.dir / f"{dataset_id}.{split}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for p in pairs:
                fh.write(json.dumps(p.as_record(system), ensure_ascii=False) + "\n")
        return path

    def path(self, dataset_id: str, split: str) -> Path:
        return self.dir / f"{dataset_id}.{split}.jsonl"

    def list_datasets(self) -> list[dict]:
        out = []
        for meta_path in sorted(self.dir.glob("*.meta.json"), reverse=True):
            try:
                out.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return out

    def get(self, dataset_id: str) -> dict:
        path = self.dir / f"{dataset_id}.meta.json"
        if not path.exists():
            raise KeyError(f"No dataset '{dataset_id}'.")
        return json.loads(path.read_text(encoding="utf-8"))

    def preview(self, dataset_id: str, split: str = "train", limit: int = 8) -> list[dict]:
        path = self.path(dataset_id, split)
        if not path.exists():
            raise KeyError(f"No {split} split for '{dataset_id}'.")
        rows = []
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= limit:
                    break
                rows.append(json.loads(line))
        return rows

    def delete(self, dataset_id: str) -> None:
        found = False
        for path in self.dir.glob(f"{dataset_id}.*"):
            path.unlink()
            found = True
        if not found:
            raise KeyError(f"No dataset '{dataset_id}'.")
