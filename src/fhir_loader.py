"""Turn Synthea FHIR R4 bundles into patient-scoped documents for RAG.

Design decisions
----------------
* Every document carries the patient's ID and name, both in its text and in its
  metadata, so retrieval can be filtered by patient and answers never mix
  records from different people.
* Resources are grouped by meaning instead of one document per FHIR resource:
  one document for demographics, one for all conditions, one for all
  medications, and one per observation type (e.g. every "Body Mass Index"
  reading in date order). This keeps related facts together, so a question
  like "Which conditions does the patient have?" is answered from a single
  complete document instead of a handful of retrieved fragments.
"""

import json
import re
from collections import defaultdict

UNKNOWN = "Unknown"


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------
def load_fhir_bundle(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_resources(bundle, resource_type):
    resources = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == resource_type:
            resources.append(resource)
    return resources


def _concept_text(concept):
    """Human-readable label of a FHIR CodeableConcept."""
    if not concept:
        return UNKNOWN
    if concept.get("text"):
        return concept["text"]
    codings = concept.get("coding") or [{}]
    # Some concepts (e.g. clinicalStatus) only carry a code, no display
    return codings[0].get("display") or codings[0].get("code") or UNKNOWN


def _clean_name_part(part):
    """Synthea appends digits to names (e.g. 'Abdul218'); strip them for display."""
    return re.sub(r"\d+$", "", part)


def _date_only(value):
    """'2019-05-01T10:00:00+01:00' -> '2019-05-01'."""
    return value[:10] if value else UNKNOWN


# --------------------------------------------------------------------------
# Patient
# --------------------------------------------------------------------------
def get_patient_name(patient):
    names = patient.get("name") or [{}]
    official = next((n for n in names if n.get("use") == "official"), names[0])
    parts = [*official.get("given", []), official.get("family", "")]
    cleaned = [_clean_name_part(p) for p in parts if p]
    return " ".join(cleaned) or UNKNOWN


def patient_to_text(patient):
    return (
        f"Name: {get_patient_name(patient)} | Gender: {patient.get('gender', UNKNOWN)} | "
        f"Date of birth: {patient.get('birthDate', UNKNOWN)}"
    )


# --------------------------------------------------------------------------
# Clinical resources -> one line of text each
# --------------------------------------------------------------------------
def condition_to_text(condition):
    display = _concept_text(condition.get("code"))
    onset = _date_only(condition.get("onsetDateTime"))
    status = _concept_text(condition.get("clinicalStatus")).lower()
    line = f"{display} | onset: {onset} | status: {status}"
    if condition.get("abatementDateTime"):
        line += f" | resolved: {_date_only(condition['abatementDateTime'])}"
    return line


def _quantity_text(quantity):
    value = quantity.get("value")
    if value is None:
        return "?"
    if isinstance(value, float):
        value = round(value, 2)
    return f"{value} {quantity.get('unit', '')}".strip()


def observation_value_text(observation):
    """Handles single values, coded values and multi-part values (e.g. blood pressure)."""
    if "valueQuantity" in observation:
        return _quantity_text(observation["valueQuantity"])
    if "valueCodeableConcept" in observation:
        return _concept_text(observation["valueCodeableConcept"])
    if "valueString" in observation:
        return observation["valueString"]
    components = observation.get("component", [])
    if components:
        parts = []
        for comp in components:
            label = _concept_text(comp.get("code"))
            if "valueQuantity" in comp:
                parts.append(f"{label}: {_quantity_text(comp['valueQuantity'])}")
            elif "valueCodeableConcept" in comp:
                parts.append(f"{label}: {_concept_text(comp['valueCodeableConcept'])}")
        return "; ".join(parts) or "?"
    return "?"


def medication_to_text(med):
    display = _concept_text(med.get("medicationCodeableConcept"))
    authored = _date_only(med.get("authoredOn"))
    return f"{display} | prescribed: {authored} | status: {med.get('status', 'unknown')}"


# --------------------------------------------------------------------------
# Bundle -> patient-scoped documents
# --------------------------------------------------------------------------
def bundle_to_documents(filepath):
    """Return a list of {"text": str, "metadata": dict} documents for one patient bundle."""
    bundle = load_fhir_bundle(filepath)
    patients = extract_resources(bundle, "Patient")
    if not patients:
        return []

    patient = patients[0]
    patient_id = patient.get("id", UNKNOWN)
    patient_name = get_patient_name(patient)
    header = f"Patient: {patient_name} (ID {patient_id})"

    def make_doc(section, body_lines, extra_metadata=None):
        metadata = {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "section": section,
        }
        metadata.update(extra_metadata or {})
        text = f"{header}\nSection: {section}\n" + "\n".join(body_lines)
        return {"text": text, "metadata": metadata}

    documents = [make_doc("Demographics", [patient_to_text(patient)])]

    conditions = sorted(
        extract_resources(bundle, "Condition"),
        key=lambda c: c.get("onsetDateTime", ""),
    )
    if conditions:
        documents.append(make_doc("Conditions", [condition_to_text(c) for c in conditions]))

    medications = sorted(
        extract_resources(bundle, "MedicationRequest"),
        key=lambda m: m.get("authoredOn", ""),
    )
    if medications:
        documents.append(make_doc("Medications", [medication_to_text(m) for m in medications]))

    # One document per observation type, readings in date order
    readings = defaultdict(list)
    for obs in extract_resources(bundle, "Observation"):
        readings[_concept_text(obs.get("code"))].append(
            (obs.get("effectiveDateTime", ""), observation_value_text(obs))
        )
    for obs_type, values in sorted(readings.items()):
        lines = [f"{_date_only(date)}: {value}" for date, value in sorted(values)]
        documents.append(
            make_doc(f"Observation - {obs_type}", lines, {"observation_type": obs_type})
        )

    return documents


# --------------------------------------------------------------------------
# Helpers for the Streamlit dashboard
# --------------------------------------------------------------------------
def extract_medications(bundle):
    medications = []
    for med in extract_resources(bundle, "MedicationRequest"):
        medications.append({
            "name": _concept_text(med.get("medicationCodeableConcept")),
            "status": med.get("status", "unknown"),
            "date": _date_only(med.get("authoredOn")),
        })
    return medications


def get_risk_level(condition_name):
    """Keyword heuristic for UI highlighting only. NOT a clinical risk score."""
    high_risk = ["cancer", "diabetes", "stroke", "heart attack", "myocardial infarction", "covid"]
    medium_risk = ["sinusitis", "fracture", "otitis", "concussion"]

    name_lower = condition_name.lower()
    if any(word in name_lower for word in high_risk):
        return "high"
    if any(word in name_lower for word in medium_risk):
        return "medium"
    return "low"


def extract_timeline(bundle):
    events = []
    for condition in extract_resources(bundle, "Condition"):
        events.append({
            "date": condition.get("onsetDateTime", ""),
            "type": "Condition",
            "description": _concept_text(condition.get("code")),
        })
    for obs in extract_resources(bundle, "Observation"):
        events.append({
            "date": obs.get("effectiveDateTime", ""),
            "type": "Observation",
            "description": f"{_concept_text(obs.get('code'))}: {observation_value_text(obs)}",
        })
    for med in extract_resources(bundle, "MedicationRequest"):
        events.append({
            "date": med.get("authoredOn", ""),
            "type": "Medication",
            "description": _concept_text(med.get("medicationCodeableConcept")),
        })
    return sorted(events, key=lambda x: x["date"])
