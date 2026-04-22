from src.fhir_loader import bundle_to_documents
import os

fhir_folder = "data/fhir"
files = os.listdir(fhir_folder)

for filename in files[:1]:  # testa só o primeiro ficheiro
    filepath = os.path.join(fhir_folder, filename)
    docs = bundle_to_documents(filepath)
    for doc in docs[:10]:  # mostra só os primeiros 10
        print(doc)
    print(f"\nTotal de documentos extraídos: {len(docs)}")