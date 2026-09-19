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
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..services.llm import complete_json
from ..ontology import get_ontology
from ..retrieval import get_chunk_index
from ..stores import get_store

log = logging.getLogger(__name__)

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
# Capitalised words that begin the templates. They carry a capital but are not
# identifiers, so they must not be treated as text a rewrite has to preserve.
_REPHRASE_STOPWORDS = {
    "which", "what", "who", "where", "when", "tell", "find", "give", "list",
    "for", "the", "a", "an", "does", "do", "is", "are", "me", "under", "every",
    "everything", "records", "link", "links", "recorded", "entity", "and", "that",
}

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
    # Rewrite the templated questions into natural phrasing with a model.
    # Off by default: it costs one call per batch of questions, and a dataset
    # is often generated several times while the ontology is still settling.
    rephrase: bool = False
    rephrase_batch: int = 40
    # Replace every entity name with a typed placeholder, so the corpus teaches
    # vocabulary, question shapes and reasoning patterns without carrying the
    # facts themselves. This is what makes a dataset shareable outside the
    # organisation that produced it.
    abstract_entities: bool = False
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

        leaked: list[str] = []
        if options.abstract_entities:
            pairs = self._abstract(pairs, nodes)
            if not pairs:
                raise ValueError(
                    "Abstraction removed every pair. This happens when the graph "
                    "has no entities the templates actually name."
                )
            leaked = self._leaked(pairs, nodes)
            if leaked:
                log.error("Abstraction left %d entity name(s) in the corpus: %s",
                          len(leaked), ", ".join(leaked[:10]))

        if options.rephrase:
            pairs = await self._rephrase(pairs, options.rephrase_batch)
            if options.abstract_entities:
                # A rewrite could reintroduce a name the model inferred from
                # context, so the check is repeated after it.
                leaked = self._leaked(pairs, nodes)

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
            "abstracted": options.abstract_entities,
            "shareable": options.abstract_entities and not leaked,
            "leaked_entities": leaked[:20],
        }
        (self.dir / f"{dataset_id}.meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    # ------------------------------------------------------------ abstraction

    def _abstract(self, pairs: list[Pair], nodes: dict[str, dict]) -> list[Pair]:
        """Strip identifying names, keeping the shape of each pair.

            Gate A12 located in what?         ->  Gate <GATE_1> located in what?
            Gate A12 located in Terminal 1.   ->  Gate <GATE_1> located in <TERMINAL_1>.

        What survives is the vocabulary, the way the question is asked, and the
        reasoning it requires. What does not survive is which gate, which
        terminal, which vendor — so the corpus can leave the organisation that
        produced it without taking their operational detail along.

        Placeholders are consistent inside a pair and reset between pairs. A
        stable mapping across the whole corpus would be reversible: an attacker
        with a handful of known facts could unmask the rest by correlation.

        Passage pairs are dropped rather than abstracted. They carry verbatim
        source text, and scrubbing prose for every identifier is not something
        to be confident about — the safe move is to exclude them.
        """
        # Longest first: without this, "Terminal 1" inside "Terminal 10" would
        # be replaced first and leave a stray "0" behind.
        by_length = sorted(nodes.items(), key=lambda kv: len(kv[0]), reverse=True)

        out: list[Pair] = []
        for pair in pairs:
            if pair.kind == "passage":
                continue

            counters: dict[str, int] = defaultdict(int)
            assigned: dict[str, str] = {}
            question, answer = pair.question, pair.answer

            for node_id, node in by_length:
                if node_id not in question and node_id not in answer:
                    continue
                if node_id not in assigned:
                    node_type = (node.get("type") or "entity")
                    slug = re.sub(r"[^A-Z0-9]", "", node_type.upper()) or "ENTITY"
                    counters[slug] += 1
                    assigned[node_id] = f"<{slug}_{counters[slug]}>"
                token = assigned[node_id]
                question = question.replace(node_id, token)
                answer = answer.replace(node_id, token)

            # A pair where nothing was replaced still names something concrete,
            # or the identifier is spelled differently from the node id. Either
            # way it has not been shown to be safe, so it does not ship.
            if not assigned:
                continue

            out.append(Pair(question=question, answer=answer, kind=pair.kind))

        return out

    @staticmethod
    def _leaked(pairs: list[Pair], nodes: dict[str, dict]) -> list[str]:
        """Entity names still present after abstraction.

        Checked rather than assumed: the cost of a silent miss is shipping a
        competitor's operational detail inside model weights.
        """
        found = set()
        for pair in pairs:
            blob = f"{pair.question} {pair.answer}"
            for node_id in nodes:
                if len(node_id) > 2 and node_id in blob:
                    found.add(node_id)
        return sorted(found)

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

    # ------------------------------------------------------------ rephrasing

    REPHRASE_SYSTEM = (
        "You rewrite stiff, templated questions into the way a real person "
        "would ask them.\n\n"
        "Return JSON only: {\"questions\": [\"...\", \"...\"]} — one rewrite per "
        "input, in the same order, same count.\n\n"
        "Rules:\n"
        "- Keep every proper name, identifier, code and number exactly as given. "
        "Gate A12 stays Gate A12. Do not translate, expand or abbreviate them.\n"
        "- Keep the question asking for the same thing. Do not make it broader, "
        "narrower, or about something else.\n"
        "- Vary the phrasing across the batch. Do not start every question the "
        "same way.\n"
        "- Plain, direct language. No preamble, no politeness padding.\n"
        "- If a question is already natural, return it unchanged."
    )

    async def _rephrase(self, pairs: list[Pair], batch_size: int) -> list[Pair]:
        """Rewrite templated questions into natural phrasing.

        Only questions are touched. Answers come from validated graph edges and
        are the ground truth this corpus exists to teach — handing them to a
        model to reword would put generated text where verified fact should be.

        Every rewrite is checked before it is accepted: it must still contain
        the identifiers the original carried. A rewrite that drops "Gate A12"
        produces a question its answer no longer fits, which is worse than the
        stilted original.
        """
        if not pairs:
            return pairs

        # Passage questions are already prose, and refusal questions must keep
        # their exact shape for the absence to remain truthful.
        targets = [i for i, p in enumerate(pairs) if p.kind in
                   ("fact", "inverse", "multihop")]
        if not targets:
            return pairs

        rewritten = 0
        for start in range(0, len(targets), batch_size):
            chunk = targets[start:start + batch_size]
            numbered = "\n".join(f"{n + 1}. {pairs[i].question}"
                                 for n, i in enumerate(chunk))
            try:
                raw = await complete_json(self.REPHRASE_SYSTEM, numbered,
                                          temperature=0.7)
                candidates = raw.get("questions") or []
            except Exception as exc:  # a failed batch keeps its templates
                log.warning("Rephrase batch failed, keeping templates: %s", exc)
                continue

            if len(candidates) != len(chunk):
                log.warning("Rephrase returned %d for %d questions; batch skipped",
                            len(candidates), len(chunk))
                continue

            for idx, candidate in zip(chunk, candidates):
                text = str(candidate or "").strip()
                if self._safe_rewrite(pairs[idx].question, text):
                    pairs[idx].question = text
                    rewritten += 1

        log.info("Rephrased %d of %d questions", rewritten, len(targets))
        return pairs

    @staticmethod
    def _safe_rewrite(original: str, candidate: str) -> bool:
        """Accept a rewrite only if it still asks about the same things.

        The test is identifier retention. Anything in the original that looks
        like a name or code — capitalised words, alphanumerics with digits —
        must survive, or the answer no longer matches the question.
        """
        if not candidate or not candidate.endswith("?"):
            return False
        # Word count rather than character length: "Gate A12?" clears any
        # sensible character minimum while asking nothing. Four words is the
        # floor for a question that carries an interrogative and a subject.
        if not (4 <= len(candidate.split()) <= 60):
            return False

        identifiers = set(re.findall(r"\b(?:[A-Z][\w-]*|\w*\d[\w-]*)\b", original))
        # Relation words are lowercase and get reworded freely; only tokens that
        # carry a capital or a digit are treated as identifiers worth keeping.
        identifiers = {i for i in identifiers if len(i) > 1
                       and i.lower() not in _REPHRASE_STOPWORDS}
        if not identifiers:
            return True
        lowered = candidate.lower()
        return all(i.lower() in lowered for i in identifiers)

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
