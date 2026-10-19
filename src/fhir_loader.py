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

# Longest document body (characters). Keeps each document, header included, within
# the 256-token input of all-MiniLM-L6-v2, so it is never cut mid-line.
MAX_DOC_CHARS = 500


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


def reference_index(bundle):
    """Map 'urn:uuid:...' and 'Type/id' references to the resources they point to."""
    index = {}
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if entry.get("fullUrl"):
            index[entry["fullUrl"]] = resource
        if resource.get("resourceType") and resource.get("id"):
            index[f"{resource['resourceType']}/{resource['id']}"] = resource
    return index


def medication_name(med, references=None):
    """Drug name from medicationCodeableConcept, or from the Medication resource that
    medicationReference points to (newer Synthea versions use both)."""
    if med.get("medicationCodeableConcept"):
        return _concept_text(med["medicationCodeableConcept"])
    ref = (med.get("medicationReference") or {}).get("reference", "")
    if ref.startswith("#"):  # contained resource
        target = next((c for c in med.get("contained", []) if c.get("id") == ref[1:]), None)
    else:
        target = (references or {}).get(ref)
    if target:
        return _concept_text(target.get("code"))
    return (med.get("medicationReference") or {}).get("display", UNKNOWN)


def medication_to_text(med, references=None):
    display = medication_name(med, references)
    authored = _date_only(med.get("authoredOn"))
    return f"{display} | prescribed: {authored} | status: {med.get('status', 'unknown')}"


def summarise_medications(medications, references=None):
    """Group repeat prescriptions of the same drug into one line.

    Synthea creates a new MedicationRequest for every renewal, so the raw list
    repeats the same drug many times. One line per drug (first and last date,
    number of prescriptions, latest status) is shorter and easier to retrieve.
    """
    by_drug = defaultdict(list)
    for med in medications:
        name = medication_name(med, references)
        by_drug[name].append((med.get("authoredOn", ""), med.get("status", "unknown")))

    lines = []
    for name, entries in by_drug.items():
        entries.sort()
        first, last = _date_only(entries[0][0]), _date_only(entries[-1][0])
        period = first if first == last else f"{first} to {last}"
        count = len(entries)
        plural = "prescription" if count == 1 else "prescriptions"
        lines.append(f"{name} | {period} | {count} {plural} | latest status: {entries[-1][1]}")
    # Most recently prescribed first
    return sorted(lines, key=lambda line: line.split(" | ")[1][-10:], reverse=True)


def allergy_to_text(allergy):
    substance = _concept_text(allergy.get("code"))
    parts = [substance]
    if allergy.get("category"):
        parts.append(f"category: {', '.join(allergy['category'])}")
    if allergy.get("criticality"):
        parts.append(f"criticality: {allergy['criticality']}")
    reactions = [
        _concept_text(m)
        for r in allergy.get("reaction", [])
        for m in r.get("manifestation", [])
    ]
    if reactions:
        parts.append(f"reactions: {', '.join(reactions)}")
    recorded = allergy.get("recordedDate") or allergy.get("onsetDateTime")
    if recorded:
        parts.append(f"recorded: {_date_only(recorded)}")
    return " | ".join(parts)


# Synthea records social determinants of health (employment, education,
# housing, ...) as Condition resources. They are kept in their own document so
# that "Which diseases does the patient have?" is not answered with
# "full-time employment". Keyword list, not an official FHIR category.
SOCIAL_KEYWORDS = (
    "employment", "education", "labor force", "high school", "criminal record",
    "social isolation", "social contact", "housing", "homeless", "transport",
    "stress", "intimate partner", "violence", "refugee", "unemployed",
    "armed forces", "medication review due", "risk activity",
)


def is_social_finding(condition):
    name = _concept_text(condition.get("code")).lower()
    return any(keyword in name for keyword in SOCIAL_KEYWORDS)


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

    def make_docs(section, body_lines, extra_metadata=None):
        """One or more documents for a section, split between lines (never inside one)
        so every part stays under the embedding model's input limit and repeats the
        patient/section header that gives it context."""
        metadata = {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "section": section,
        }
        metadata.update(extra_metadata or {})
        groups, current, size = [], [], 0
        for line in body_lines:
            if current and size + len(line) > MAX_DOC_CHARS:
                groups.append(current)
                current, size = [], 0
            current.append(line)
            size += len(line) + 1
        if current:
            groups.append(current)
        docs = []
        for i, group in enumerate(groups, 1):
            title = section if len(groups) == 1 else f"{section} (part {i} of {len(groups)})"
            text = f"{header}\nSection: {title}\n" + "\n".join(group)
            docs.append({"text": text, "metadata": dict(metadata)})
        return docs

    references = reference_index(bundle)
    documents = make_docs("Demographics", [patient_to_text(patient)])

    allergy_list = extract_resources(bundle, "AllergyIntolerance")
    documents += make_docs(
        "Allergies",
        [allergy_to_text(a) for a in allergy_list] or ["No allergies recorded in this record"],
    )

    conditions = sorted(
        extract_resources(bundle, "Condition"),
        key=lambda c: c.get("onsetDateTime", ""),
    )
    clinical = [c for c in conditions if not is_social_finding(c)]
    social = [c for c in conditions if is_social_finding(c)]
    if clinical:
        documents += make_docs("Conditions", [condition_to_text(c) for c in clinical])
    if social:
        documents += make_docs("Social history", [condition_to_text(c) for c in social])

    medications = extract_resources(bundle, "MedicationRequest")
    if medications:
        documents += make_docs("Medications", summarise_medications(medications, references))

    # One document per observation type, readings in date order
    readings = defaultdict(list)
    for obs in extract_resources(bundle, "Observation"):
        readings[_concept_text(obs.get("code"))].append(
            (obs.get("effectiveDateTime", ""), observation_value_text(obs))
        )
    for obs_type, values in sorted(readings.items()):
        lines = [f"{_date_only(date)}: {value}" for date, value in sorted(values)]
        documents += make_docs(f"Observation - {obs_type}", lines, {"observation_type": obs_type})

    return documents


# --------------------------------------------------------------------------
# Helpers for the Streamlit dashboard
# --------------------------------------------------------------------------
def extract_medications(bundle):
    references = reference_index(bundle)
    medications = []
    for med in extract_resources(bundle, "MedicationRequest"):
        medications.append({
            "name": medication_name(med, references),
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
    references = reference_index(bundle)
    for med in extract_resources(bundle, "MedicationRequest"):
        events.append({
            "date": med.get("authoredOn", ""),
            "type": "Medication",
            "description": medication_name(med, references),
        })
    return sorted(events, key=lambda x: x["date"])
