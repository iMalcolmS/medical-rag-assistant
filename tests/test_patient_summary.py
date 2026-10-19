from datetime import date

from src.fhir_loader import load_fhir_bundle
from src.patient_summary import (
    key_measures,
    medication_list,
    observation_series,
    patient_header,
    problem_list,
    timeline_events,
)


def test_patient_header(bundle_path):
    header = patient_header(load_fhir_bundle(bundle_path), today=date(2026, 1, 1))
    assert header["name"] == "Abdul Harris"
    assert header["age"] == 73  # born 1952-12-05
    assert header["deceased"] is None


def test_problem_list_puts_active_first_and_flags_social(bundle_path):
    problems = problem_list(load_fhir_bundle(bundle_path))
    assert problems[0]["status"] == "active"
    social = {p["problem"]: p["social"] for p in problems}
    assert social["Full-time employment"] is True
    assert social["Prediabetes"] is False  # "(finding)" suffix removed, still clinical


def test_medication_list_groups_prescriptions(bundle_path):
    meds = medication_list(load_fhir_bundle(bundle_path))
    assert meds == [{
        "medication": "Simvastatin 20 MG Oral Tablet", "status": "stopped",
        "first": "1992-04-10", "last": "2001-12-21", "prescriptions": 2,
    }]


def test_blood_pressure_is_split_into_components(bundle_path):
    series = observation_series(load_fhir_bundle(bundle_path))
    assert series["Systolic"][0]["value"] == 128
    assert series["Diastolic"][0]["value"] == 82


def test_key_measures_report_latest_value_and_change():
    series = {"Heart rate": [
        {"date": "2020-01-01", "value": 70.0, "unit": "/min"},
        {"date": "2021-01-01", "value": 76.0, "unit": "/min"},
    ]}
    (tile,) = key_measures(series)
    assert tile["label"] == "Heart rate"
    assert tile["value"] == 76.0
    assert tile["delta"] == 6.0
    assert tile["history"] == [70.0, 76.0]


def test_timeline_excludes_social_findings_and_is_sorted(bundle_path):
    events = timeline_events(load_fhir_bundle(bundle_path))
    assert all("employment" not in e["event"].lower() for e in events)
    assert [e["date"] for e in events] == sorted(e["date"] for e in events)
