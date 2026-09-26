from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ..config import get_settings
from .models import EdgeIn, NodeIn, RejectRecord, ValidationResult

BUILTIN_DIR = Path(__file__).parent / "domains"

_WS = re.compile(r"\s+")


class IdTransform:
    """One user-defined identifier rule.

    The platform ships no domain knowledge of its own — every canonicalisation
    (flight numbers, case numbers, SKUs, ticket ids) is expressed here by
    whoever owns the domain, and applies only to that domain.
    """

    def __init__(self, spec: dict[str, Any]) -> None:
        self.pattern_src: str = str(spec.get("pattern", ""))
        # An empty string is a valid replacement — it means "delete the match".
        # Only a missing key means "leave the text alone and just change case".
        raw_replace = spec.get("replace", None)
        self.replace: str | None = None if raw_replace is None else str(raw_replace)
        self.case: str = str(spec.get("case", "")).lower()   # upper | lower | title | ""
        self.note: str = str(spec.get("note", ""))
        try:
            self.pattern = re.compile(self.pattern_src) if self.pattern_src else None
            self.error = ""
        except re.error as exc:
            self.pattern = None
            self.error = str(exc)

    def apply(self, value: str) -> str:
        if self.pattern is None:
            return value
        if not self.pattern.search(value):
            return value
        out = self.pattern.sub(self._replacement, value) if self.replace is not None else value
        if self.case == "upper":
            out = out.upper()
        elif self.case == "lower":
            out = out.lower()
        elif self.case == "title":
            out = out.title()
        return out

    @property
    def _replacement(self) -> str:
        r"""Accept both `\1` and `$1` for capture groups.

        Replacement syntax differs by language, and `$1` is the common
        expectation coming from JavaScript and most regex tools. Emitting the
        literal text "$1-$2" as an identifier would be a confusing failure, so
        the alternative form is translated rather than rejected.
        """
        text = self.replace or ""
        return re.sub(r"\$(\d+)", r"\\\1", text)

    def to_spec(self) -> dict[str, str]:
        return {"pattern": self.pattern_src, "replace": self.replace or "",
                "case": self.case, "note": self.note}


def _actions_to_spec(actions) -> list:
    from .actions import as_spec
    return as_spec(actions)


class Ontology:
    """A closed schema for one domain.

    Built-in domains ship as YAML in ./domains. User-defined domains are written
    to <data_dir>/ontologies and are fully editable from the console, including
    a custom extraction prompt that overrides the generated one.
    """

    def __init__(self, spec: dict[str, Any], *, path: Path | None = None,
                 builtin: bool = False) -> None:
        self.path = path
        self.builtin = builtin
        self.key: str = spec["key"]
        self.name: str = spec.get("name", spec["key"])
        self.description: str = spec.get("description", "")
        self.entity_types: dict[str, dict] = spec.get("entity_types", {}) or {}
        self.allowed_triples: set[tuple[str, str, str]] = {
            (t[0], t[1], t[2]) for t in (spec.get("allowed_triples") or [])
        }
        norm = spec.get("normalization", {}) or {}
        self.strip_type_prefixes: bool = norm.get("strip_type_prefixes", True)
        self.collapse_whitespace: bool = norm.get("collapse_whitespace", True)
        self.id_transforms: list[IdTransform] = [
            IdTransform(t) for t in (norm.get("id_transforms") or [])
        ]
        self.normalization_rules: list[str] = norm.get("rules_text", []) or []
        # A non-empty custom prompt replaces the generated one verbatim.
        self.custom_prompt: str = (spec.get("custom_prompt") or "").strip()
        self.open_relations: bool = bool(spec.get("open_relations", False))
        # What may be done in this domain, not just what exists in it.
        from .actions import parse_actions
        self.actions = parse_actions(spec.get("actions"))
        self._rebuild_prefix_re()

    def _rebuild_prefix_re(self) -> None:
        names = [re.escape(t) for t in self.entity_types] or ["__none__"]
        self._prefix_re = re.compile(r"^(" + "|".join(names) + r")\s+(.+)$", re.I)

    # ------------------------------------------------------------- serialise
    def to_spec(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "entity_types": self.entity_types,
            "allowed_triples": [list(t) for t in sorted(self.allowed_triples)],
            "open_relations": self.open_relations,
            "normalization": {
                "strip_type_prefixes": self.strip_type_prefixes,
                "collapse_whitespace": self.collapse_whitespace,
                "id_transforms": [t.to_spec() for t in self.id_transforms],
                "rules_text": self.normalization_rules,
            },
            "custom_prompt": self.custom_prompt,
            # Without this, every save through the console would silently drop
            # the domain's declared actions.
            "actions": _actions_to_spec(self.actions),
        }

    def spine_of(self, entity_type: str | None) -> str | None:
        """The shared spine type this domain type extends, if it declares one."""
        from .spine import is_spine_type
        target = (self.entity_types.get(entity_type or "") or {}).get("extends")
        return target if is_spine_type(target) else None

    def summary(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "builtin": self.builtin,
            "entity_types": sorted(self.entity_types),
            "relationship_count": len(self.allowed_triples),
            "has_custom_prompt": bool(self.custom_prompt),
            "open_relations": self.open_relations,
            # Which shared spine type each domain type joins through.
            "spine_map": {t: self.spine_of(t) for t in self.entity_types
                          if self.spine_of(t)},
        }

    # ------------------------------------------------------------- normalise
    def normalize_id(self, raw: str) -> str:
        """Canonicalise an identifier using this domain's rules only.

        A domain's own transforms take precedence over the generic
        type-prefix stripper: if the raw value matches an explicit rule, that
        rule wins. Otherwise the value is tidied (prefix, whitespace) and the
        transforms get a second chance against the cleaned form.

        This ordering matters when an entity type shares a word with an
        identifier — "Terminal 1" in an airport domain is a Terminal id, not
        the word "Terminal" prefixed to the id "1".
        """
        s = str(raw).strip()
        if not s:
            return s

        explicit = self._apply_transforms(s)
        if explicit != s:
            return explicit.strip()

        if self.strip_type_prefixes:
            m = self._prefix_re.fullmatch(s)
            if m:
                s = m.group(2).strip()
        if self.collapse_whitespace:
            s = _WS.sub(" ", s).strip()
        return self._apply_transforms(s).strip()

    def _apply_transforms(self, value: str) -> str:
        for transform in self.id_transforms:
            value = transform.apply(value)
        return value

    # ------------------------------------------------------------- validate
    def validate(self, raw: dict[str, Any]) -> ValidationResult:
        result = ValidationResult()

        for item in raw.get("nodes", []) or []:
            try:
                node = NodeIn(**item)
            except Exception as exc:
                result.rejects.append(RejectRecord(kind="node",
                    reason=f"unparseable: {exc}", payload=_safe(item)))
                continue
            node.id = self.normalize_id(node.id)
            if not node.id:
                result.rejects.append(RejectRecord(kind="node",
                    reason="empty id", payload=_safe(item)))
                continue
            if node.type not in self.entity_types:
                result.rejects.append(RejectRecord(kind="node",
                    reason=f"type '{node.type}' not in ontology", payload=_safe(item)))
                continue
            result.nodes.append(node)

        typemap = {n.id: n.type for n in result.nodes}

        for item in raw.get("edges", []) or []:
            try:
                edge = EdgeIn(**item)
            except Exception as exc:
                result.rejects.append(RejectRecord(kind="edge",
                    reason=f"unparseable: {exc}", payload=_safe(item)))
                continue
            edge.source = self.normalize_id(edge.source)
            edge.target = self.normalize_id(edge.target)
            if not edge.type:
                result.rejects.append(RejectRecord(kind="edge",
                    reason="missing relation name", payload=_safe(item)))
                continue
            if edge.source not in typemap or edge.target not in typemap:
                result.rejects.append(RejectRecord(kind="edge",
                    reason="endpoint not among extracted nodes", payload=_safe(item)))
                continue
            triple = (typemap[edge.source], edge.type, typemap[edge.target])
            if not self.open_relations and triple not in self.allowed_triples:
                result.rejects.append(RejectRecord(kind="edge",
                    reason=f"triple not allowed: {triple[0]} -{triple[1]}-> {triple[2]}",
                    payload=_safe(item)))
                continue
            result.edges.append(edge)

        return result

    # ------------------------------------------------------------- prompt
    def extraction_prompt(self) -> str:
        if self.custom_prompt:
            return self.custom_prompt
        return self.generated_prompt()

    def generated_prompt(self) -> str:
        types_block = "\n".join(
            f"- {name}: id rule = {spec.get('id_rule','free text')}"
            + (f" (e.g. {', '.join(map(str, spec.get('examples', [])))})"
               if spec.get("examples") else "")
            for name, spec in self.entity_types.items()
        ) or "- (no entity types defined)"
        if self.open_relations:
            triples_block = ("Any relation name is permitted, but both endpoints must be "
                             "entity types listed above.")
        else:
            triples_block = "\n".join(
                f"{s} -{r}-> {t}" for s, r, t in sorted(self.allowed_triples)
            ) or "(no relationships defined)"
        rule_lines = list(self.normalization_rules)
        for t in self.id_transforms:
            if t.note:
                rule_lines.append(t.note)
        rules_block = "\n".join(f"- {r}" for r in rule_lines) or "- (none)"
        return f"""You are an information-extraction system that populates a knowledge graph with a FIXED schema. Operate as a strict parser, not a creative assistant.

DOMAIN: {self.name}. {self.description}

=== ENTITY TYPES (use EXACTLY these `type` values, nothing else) ===
{types_block}

=== ALLOWED RELATIONSHIPS (source_type -relation-> target_type) ===
{triples_block}

=== NORMALISATION (apply BEFORE emitting any id) ===
{rules_block}

=== RULES ===
1. Extract ONLY what the input explicitly supports. Do not infer.
2. Anything that fits NO entity type above: DROP IT. Never invent a type.
3. Every edge must reference ids present in your `nodes` array and match an allowed relationship row.
4. Every node needs: id (per its id rule), type, and metadata.evidence containing a verbatim snippet of under 15 words from the input.
5. If nothing is extractable, return {{"nodes": [], "edges": []}}.

Return a single JSON object: {{"nodes": [...], "edges": [...]}}. No prose, no code fences.
"""


def _safe(item: Any) -> dict:
    return item if isinstance(item, dict) else {"value": str(item)[:400]}


# ------------------------------------------------------------------ registry
def _custom_dir() -> Path:
    d = get_settings().data_dir / "ontologies"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_all() -> dict[str, Ontology]:
    """Read from disk on every call so console edits take effect immediately."""
    out: dict[str, Ontology] = {}
    for path in sorted(BUILTIN_DIR.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        out[spec["key"]] = Ontology(spec, path=path, builtin=True)
    for path in sorted(_custom_dir().glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        out[spec["key"]] = Ontology(spec, path=path, builtin=False)
    return out


KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,48}$")


def get_ontology(key: str) -> Ontology:
    ontologies = _load_all()
    if key not in ontologies:
        raise KeyError(f"unknown ontology '{key}'. available: {sorted(ontologies)}")
    return ontologies[key]


def list_ontologies() -> list[dict[str, Any]]:
    return [o.summary() for o in _load_all().values()]


def save_ontology(spec: dict[str, Any]) -> Ontology:
    """Create or update a user-defined ontology. Built-ins are copy-on-write."""
    key = str(spec.get("key", "")).strip().lower()
    if not KEY_RE.fullmatch(key):
        raise ValueError(
            "Key must be 2-49 characters: lowercase letters, digits, hyphen or underscore."
        )
    if not spec.get("entity_types"):
        raise ValueError("Define at least one entity type.")
    for t in (spec.get("normalization", {}) or {}).get("id_transforms", []) or []:
        probe = IdTransform(t)
        if probe.error:
            raise ValueError(f"Identifier rule /{probe.pattern_src}/ is not valid: {probe.error}")
    from .spine import validate_extends
    problems = validate_extends(spec.get("entity_types") or {})
    if problems:
        raise ValueError("; ".join(problems))
    spec["key"] = key
    ont = Ontology(spec, builtin=False)
    path = _custom_dir() / f"{key}.yaml"
    path.write_text(yaml.safe_dump(ont.to_spec(), sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    ont.path = path
    return ont


def delete_ontology(key: str) -> None:
    path = _custom_dir() / f"{key}.yaml"
    if not path.exists():
        raise KeyError(f"'{key}' is not a user-defined ontology and cannot be deleted.")
    path.unlink()
