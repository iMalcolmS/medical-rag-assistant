import json 
import os

def load_fhir_bundle(filepath):
    with open(filepath,"r") as f:
        return json.load(f)
    
def extract_resources(bundle, resource_type):
    resources = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == resource_type:
            resources.append(resource)
    return resources

def patient_to_text(patient):
    name = patient.get("name", [{}])[0]
    full_name ="".join(name.get("given",[] + [name.get("family","")]))
    gender = patient.get("gender","Unknown")
    birth_date = patient.get("birthDate", "Unknown")
    return f"Patient: {full_name} | Gender: {gender} | Date Birth: {birth_date}"

def extract_medications(bundle):
    medications = []
    for med in extract_resources(bundle, "MedicationRequest"):
        name = med.get("medicationCodeableConcept", {})
        display = name.get("text") or name.get("coding", [{}])[0].get("display", "Unknown")
        status = med.get("status", "unknown")
        medications.append({
            "name": display,
            "status": status
        })
    return medications

def condition_to_text(condition):
    code = condition.get("code",{})
    display = code.get("text") or code.get("coding", [{}])[0].get("display","Unknown")
    onset = condition.get("onsetDateTime","Unknown Date")
    return f"Condition: {display} | Start: {onset}"

def observation_to_text(observation):
    code = observation.get("code",{})
    display = code.get("text") or code.get("coding", [{}])[0].get("display","Unknown")
    value = observation.get("valueQuantity",{})
    val = f"{value.get('value','?')} {value.get('unit','')}"
    date = observation.get("effectiveDateTime","Unknown Date")
    return f"Observation: {display} | Value: {val} | Date: {date}"

def bundle_to_documents(filepath):
    bundle = load_fhir_bundle(filepath)
    documents = []

    for patient in extract_resources(bundle, "Patient"):
        documents.append(patient_to_text(patient))

    for condition in extract_resources(bundle, "Condition"):
        documents.append(condition_to_text(condition))

    for observation in extract_resources(bundle, "Observation"):
        documents.append(observation_to_text(observation))
    
    return documents 

def get_risk_level(condition_name):
    high_risk = ["cancer", "diabetes", "stroke", "heart attack", "covid"]
    medium_risk = ["sinusitis", "fracture", "otitis", "concussion"]
    
    name_lower = condition_name.lower()
    
    if any(word in name_lower for word in high_risk):
        return "high"
    elif any(word in name_lower for word in medium_risk):
        return "medium"
    else:
        return "low"

def extract_timeline(bundle):
    events = []
    
    for condition in extract_resources(bundle, "Condition"):
        code = condition.get("code", {})
        display = code.get("text") or code.get("coding", [{}])[0].get("display", "Unknown")
        date = condition.get("onsetDateTime", "")
        events.append({
            "date": date,
            "type": "Condition",
            "description": display
        })

    for obs in extract_resources(bundle, "Observation"):
        code = obs.get("code", {})
        display = code.get("text") or code.get("coding", [{}])[0].get("display", "Unknown")
        date = obs.get("effectiveDateTime", "")
        events.append({
            "date": date,
            "type": "Observation",
            "description": display
        })

    return sorted(events, key=lambda x: x["date"])