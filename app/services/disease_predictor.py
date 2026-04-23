"""
Disease prediction service using symptom-disease mapping with Jaccard similarity.

This module provides a rule-based prediction system that matches patient symptoms
against a curated database of disease-symptom associations. Prediction confidence
is calculated using Jaccard similarity (intersection over union of symptom sets).
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Symptom-disease knowledge base
# ---------------------------------------------------------------------------

DISEASE_SYMPTOM_MAP: dict[str, list[str]] = {
    "Common Cold": [
        "runny_nose", "sneezing", "sore_throat", "cough",
        "mild_fever", "headache",
    ],
    "Influenza": [
        "high_fever", "body_aches", "fatigue", "cough",
        "headache", "chills",
    ],
    "COVID-19": [
        "fever", "dry_cough", "fatigue", "loss_of_taste",
        "loss_of_smell", "difficulty_breathing",
    ],
    "Pneumonia": [
        "high_fever", "chest_pain", "cough", "difficulty_breathing",
        "fatigue", "chills",
    ],
    "Bronchitis": [
        "cough", "chest_discomfort", "fatigue",
        "shortness_of_breath", "mild_fever",
    ],
    "Asthma": [
        "wheezing", "shortness_of_breath", "chest_tightness", "cough",
    ],
    "Migraine": [
        "severe_headache", "nausea", "sensitivity_to_light",
        "sensitivity_to_sound", "visual_disturbances",
    ],
    "Tension Headache": [
        "mild_headache", "neck_pain", "pressure_in_forehead", "fatigue",
    ],
    "Hypertension": [
        "headache", "dizziness", "blurred_vision",
        "chest_pain", "shortness_of_breath",
    ],
    "Diabetes Type 2": [
        "frequent_urination", "increased_thirst", "blurred_vision",
        "fatigue", "slow_healing_wounds",
    ],
    "Gastritis": [
        "abdominal_pain", "nausea", "vomiting",
        "bloating", "loss_of_appetite",
    ],
    "Gastroesophageal Reflux": [
        "heartburn", "chest_pain", "difficulty_swallowing", "regurgitation",
    ],
    "Urinary Tract Infection": [
        "painful_urination", "frequent_urination",
        "lower_abdominal_pain", "cloudy_urine", "fever",
    ],
    "Anemia": [
        "fatigue", "weakness", "pale_skin", "dizziness",
        "cold_hands_and_feet", "shortness_of_breath",
    ],
    "Hypothyroidism": [
        "fatigue", "weight_gain", "cold_intolerance",
        "dry_skin", "constipation", "depression",
    ],
    "Allergic Rhinitis": [
        "sneezing", "runny_nose", "itchy_eyes",
        "nasal_congestion", "watery_eyes",
    ],
    "Sinusitis": [
        "facial_pain", "nasal_congestion", "headache",
        "thick_nasal_discharge", "fever",
    ],
    "Arthritis": [
        "joint_pain", "joint_stiffness", "swelling",
        "reduced_range_of_motion", "fatigue",
    ],
    "Depression": [
        "persistent_sadness", "loss_of_interest", "fatigue",
        "sleep_disturbances", "appetite_changes", "difficulty_concentrating",
    ],
    "Anxiety Disorder": [
        "excessive_worry", "restlessness", "fatigue",
        "difficulty_concentrating", "muscle_tension", "sleep_disturbances",
    ],
}


# ---------------------------------------------------------------------------
# Pre-computed sets for faster lookup
# ---------------------------------------------------------------------------

_DISEASE_SYMPTOM_SETS: dict[str, set[str]] = {
    disease: set(symptoms)
    for disease, symptoms in DISEASE_SYMPTOM_MAP.items()
}

ALL_SYMPTOMS: list[str] = sorted(
    {symptom for symptoms in DISEASE_SYMPTOM_MAP.values() for symptom in symptoms}
)


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PredictionResult:
    """A single disease prediction with its confidence score."""

    disease: str
    confidence: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_diseases(
    symptoms: list[str],
    *,
    top_k: int = 3,
    min_confidence: float = 0.0,
) -> list[PredictionResult]:
    """Return the top-k disease predictions for the given symptoms.

    Confidence is calculated as the **Jaccard similarity** between the input
    symptom set and each disease's known symptom set:

        J(A, B) = |A n B| / |A u B|

    Parameters
    ----------
    symptoms:
        List of symptom identifiers (e.g. ``["fever", "cough"]``).
    top_k:
        Maximum number of predictions to return.
    min_confidence:
        Minimum Jaccard score to include a disease in the results.

    Returns
    -------
    list[PredictionResult]
        Sorted descending by confidence.
    """
    if not symptoms:
        return []

    input_set = set(symptoms)
    scores: list[PredictionResult] = []

    for disease, disease_set in _DISEASE_SYMPTOM_SETS.items():
        intersection = input_set & disease_set
        if not intersection:
            continue

        union = input_set | disease_set
        jaccard = len(intersection) / len(union)

        if jaccard >= min_confidence:
            scores.append(
                PredictionResult(
                    disease=disease,
                    confidence=round(jaccard, 4),
                )
            )

    # Sort by confidence descending, then alphabetically for ties
    scores.sort(key=lambda r: (-r.confidence, r.disease))
    return scores[:top_k]
