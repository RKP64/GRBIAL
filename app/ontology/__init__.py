"""Ontology layer: schema definition, normalisation, validation, prompt building.

Design principle carried from the pilot: prompts constrain, code enforces,
logs reveal. The YAML in ./domains is the single source of truth for a domain;
the extraction prompt and the validator are both generated from it, so they can
never drift apart.
"""
from .models import (  # noqa: F401
    EdgeIn,
    Extraction,
    NodeIn,
    RejectRecord,
    ValidationResult,
)
from .registry import Ontology, get_ontology, list_ontologies  # noqa: F401
