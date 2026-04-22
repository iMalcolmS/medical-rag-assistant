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