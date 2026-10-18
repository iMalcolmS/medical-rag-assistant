"""Shared fixtures: small synthetic FHIR bundles, so tests never need the real data."""

import json

import pytest


def make_bundle(patient_id, given, family):
    """A minimal Synthea-style bundle with one resource of each kind we handle."""
    return {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {
                "resourceType": "Patient", "id": patient_id, "gender": "male",
                "birthDate": "1952-12-05",
                "name": [{"use": "official", "given": [given], "family": family}],
            }},
            {"resource": {
                "resourceType": "Condition",
                "code": {"text": "Coronary Heart Disease"},
                "onsetDateTime": "1992-04-10T00:00:00Z",
                "clinicalStatus": {"coding": [{"code": "active"}]},
            }},
            {"resource": {
                "resourceType": "Condition",
                "code": {"text": "Prediabetes (finding)"},
                "onsetDateTime": "1990-02-23T00:00:00Z",
                "clinicalStatus": {"coding": [{"code": "active"}]},
            }},
            {"resource": {
                "resourceType": "Condition",
                "code": {"text": "Full-time employment (finding)"},
                "onsetDateTime": "1971-01-29T00:00:00Z",
                "abatementDateTime": "1972-02-04T00:00:00Z",
                "clinicalStatus": {"coding": [{"code": "resolved"}]},
            }},
            {"resource": {
                "resourceType": "MedicationRequest",
                "medicationCodeableConcept": {"text": "Simvastatin 20 MG Oral Tablet"},
                "authoredOn": "1992-04-10T00:00:00Z", "status": "stopped",
            }},
            {"resource": {
                "resourceType": "MedicationRequest",
                "medicationCodeableConcept": {"text": "Simvastatin 20 MG Oral Tablet"},
                "authoredOn": "2001-12-21T00:00:00Z", "status": "stopped",
            }},
            {"resource": {
                "resourceType": "Observation",
                "code": {"text": "Blood Pressure"},
                "effectiveDateTime": "2001-01-01T00:00:00Z",
                "component": [
                    {"code": {"text": "Systolic"}, "valueQuantity": {"value": 128, "unit": "mm[Hg]"}},
                    {"code": {"text": "Diastolic"}, "valueQuantity": {"value": 82, "unit": "mm[Hg]"}},
                ],
            }},
            {"resource": {
                "resourceType": "Observation",
                "code": {"text": "Tobacco smoking status"},
                "effectiveDateTime": "2001-01-01T00:00:00Z",
                "valueCodeableConcept": {"text": "Never smoked tobacco"},
            }},
        ],
    }


@pytest.fixture
def fhir_dir(tmp_path):
    """A folder with two patients' bundles and a macOS .DS_Store file."""
    for filename, (pid, given, family) in {
        "Abdul218_Harris789.json": ("patient-a", "Abdul218", "Harris789"),
        "Zoe12_Smith44.json": ("patient-b", "Zoe12", "Smith44"),
    }.items():
        (tmp_path / filename).write_text(json.dumps(make_bundle(pid, given, family)))
    (tmp_path / ".DS_Store").write_bytes(b"\xc0\x00\x01")  # not JSON, must be skipped
    return tmp_path


@pytest.fixture
def bundle_path(fhir_dir):
    return str(fhir_dir / "Abdul218_Harris789.json")
