from src.fhir_loader import bundle_to_documents, get_risk_level, load_fhir_bundle, extract_timeline


def docs_by_section(path):
    return {d["metadata"]["section"]: d for d in bundle_to_documents(path)}


def test_patient_name_is_cleaned(bundle_path):
    demographics = docs_by_section(bundle_path)["Demographics"]
    assert demographics["metadata"]["patient_name"] == "Abdul Harris"


def test_every_document_is_scoped_to_the_patient(bundle_path):
    docs = bundle_to_documents(bundle_path)
    assert docs
    for doc in docs:
        assert doc["metadata"]["patient_id"] == "patient-a"
        assert doc["text"].startswith("Patient: Abdul Harris (ID patient-a)")


def test_social_findings_are_separated_from_clinical_conditions(bundle_path):
    sections = docs_by_section(bundle_path)
    assert "Full-time employment" not in sections["Conditions"]["text"]
    assert "Full-time employment" in sections["Social history"]["text"]
    # Clinical findings must stay clinical even though Synthea labels them "(finding)"
    assert "Prediabetes" in sections["Conditions"]["text"]


def test_condition_status_is_included(bundle_path):
    text = docs_by_section(bundle_path)["Conditions"]["text"]
    assert "Coronary Heart Disease | onset: 1992-04-10 | status: active" in text


def test_repeat_prescriptions_are_grouped(bundle_path):
    text = docs_by_section(bundle_path)["Medications"]["text"]
    assert text.count("Simvastatin") == 1
    assert "1992-04-10 to 2001-12-21 | 2 prescriptions | latest status: stopped" in text


def test_multi_component_and_coded_observations(bundle_path):
    sections = docs_by_section(bundle_path)
    bp = sections["Observation - Blood Pressure"]
    assert "Systolic: 128 mm[Hg]; Diastolic: 82 mm[Hg]" in bp["text"]
    assert bp["metadata"]["observation_type"] == "Blood Pressure"
    assert "Never smoked tobacco" in sections["Observation - Tobacco smoking status"]["text"]


def test_risk_level_heuristic():
    assert get_risk_level("Diabetes mellitus type 2") == "high"
    assert get_risk_level("Viral sinusitis (disorder)") == "medium"
    assert get_risk_level("Seasonal allergy") == "low"


def test_timeline_is_sorted_by_date(bundle_path):
    dates = [event["date"] for event in extract_timeline(load_fhir_bundle(bundle_path))]
    assert dates == sorted(dates)
