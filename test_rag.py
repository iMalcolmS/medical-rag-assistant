from src.fhir_loader import bundle_to_documents
from src.rag_pipeline import build_index, query_index
import os 

fhir_folder = "data/fhir"
files = os.listdir(fhir_folder)
filepath = os.path.join(fhir_folder,files[0])

documents = bundle_to_documents(filepath)
index = build_index(documents)
response = query_index(index, "What conditions does the patient have?")
print (response)