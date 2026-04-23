"""
Drug interaction checker service.

Loads a curated JSON dataset of known drug-drug interactions and provides
a lookup function that checks all pairwise combinations of a given
medication list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DrugInteraction:
    """A single known drug-drug interaction."""

    drug1: str
    drug2: str
    severity: str  # HIGH | MODERATE | LOW
    description: str


# ---------------------------------------------------------------------------
# Load interaction database (once, at import time)
# ---------------------------------------------------------------------------

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "drug_interactions.json"

_INTERACTIONS: list[DrugInteraction] = []
_INTERACTION_INDEX: dict[tuple[str, str], DrugInteraction] = {}
_ALL_MEDICATIONS: list[str] = []


def _load_interactions() -> None:
    """Parse the JSON file and build the lookup index."""
    global _INTERACTIONS, _INTERACTION_INDEX, _ALL_MEDICATIONS  # noqa: PLW0603

    with open(_DATA_PATH, encoding="utf-8") as fh:
        raw: list[dict] = json.load(fh)

    medications: set[str] = set()

    for entry in raw:
        interaction = DrugInteraction(
            drug1=entry["drug1"],
            drug2=entry["drug2"],
            severity=entry["severity"],
            description=entry["description"],
        )
        _INTERACTIONS.append(interaction)

        # Index both orderings so lookup is O(1) regardless of order
        key_forward = (entry["drug1"].lower(), entry["drug2"].lower())
        key_reverse = (entry["drug2"].lower(), entry["drug1"].lower())
        _INTERACTION_INDEX[key_forward] = interaction
        _INTERACTION_INDEX[key_reverse] = interaction

        medications.add(entry["drug1"])
        medications.add(entry["drug2"])

    _ALL_MEDICATIONS = sorted(medications)


# Load on module import
_load_interactions()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_all_medications() -> list[str]:
    """Return a sorted list of all medication names in the database."""
    return list(_ALL_MEDICATIONS)


def check_interactions(medications: list[str]) -> list[DrugInteraction]:
    """Check all pairwise combinations of *medications* for known interactions.

    Parameters
    ----------
    medications:
        List of medication names (case-insensitive matching).

    Returns
    -------
    list[DrugInteraction]
        Found interactions sorted by severity (HIGH first) then alphabetically.
    """
    if len(medications) < 2:
        return []

    found: list[DrugInteraction] = []
    seen: set[tuple[str, str]] = set()

    for drug_a, drug_b in combinations(medications, 2):
        key = (drug_a.lower(), drug_b.lower())
        # Deduplicate (the index stores both orderings)
        canonical = tuple(sorted(key))
        if canonical in seen:
            continue
        seen.add(canonical)

        interaction = _INTERACTION_INDEX.get(key)
        if interaction is not None:
            found.append(interaction)

    # Sort: HIGH > MODERATE > LOW, then alphabetically by drug1
    severity_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    found.sort(key=lambda i: (severity_order.get(i.severity, 3), i.drug1, i.drug2))

    return found
