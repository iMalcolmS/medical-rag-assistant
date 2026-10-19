"""Create the small demo dataset that ships with the repository.

Copies the indexed patients' bundles from data/fhir to demo/fhir, keeping only
the FHIR resource types the app uses (Claims, ExplanationOfBenefit and
Provenance make up most of a Synthea bundle and are dropped), then copies the
pre-built vector index so the hosted demo starts without re-embedding.

Usage:  python -m scripts.make_demo_data
"""

import json
import os
import shutil

from src.rag_pipeline import list_bundle_files

SOURCE_DIR, SOURCE_INDEX = "data/fhir", "storage"
DEMO_DIR, DEMO_INDEX = os.path.join("demo", "fhir"), os.path.join("demo", "storage")
PATIENT_LIMIT = int(os.environ.get("PATIENT_LIMIT", 20))  # must match the index
KEEP = {"Patient", "Condition", "MedicationRequest", "Medication",
        "Observation", "AllergyIntolerance", "Encounter"}


def size_mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / 1e6
    return sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(path) for f in files) / 1e6


def main():
    if not os.path.isdir(SOURCE_DIR) or not os.path.isdir(SOURCE_INDEX):
        raise SystemExit("Needs data/fhir and a built storage/ index (run: python -m src.rag_pipeline).")

    shutil.rmtree(DEMO_DIR, ignore_errors=True)
    os.makedirs(DEMO_DIR)
    before = after = 0.0
    for path in list_bundle_files(SOURCE_DIR, limit=PATIENT_LIMIT):
        with open(path, encoding="utf-8") as f:
            bundle = json.load(f)
        bundle["entry"] = [e for e in bundle.get("entry", [])
                           if e.get("resource", {}).get("resourceType") in KEEP]
        out = os.path.join(DEMO_DIR, os.path.basename(path))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(bundle, f, separators=(",", ":"))
        before += size_mb(path)
        after += size_mb(out)

    shutil.rmtree(DEMO_INDEX, ignore_errors=True)
    shutil.copytree(SOURCE_INDEX, DEMO_INDEX)

    print(f"Demo bundles: {PATIENT_LIMIT} patients, {before:.1f} MB -> {after:.1f} MB in {DEMO_DIR}")
    print(f"Demo index:   {size_mb(DEMO_INDEX):.1f} MB in {DEMO_INDEX}")
    largest = max(os.listdir(DEMO_INDEX), key=lambda f: size_mb(os.path.join(DEMO_INDEX, f)))
    print(f"Largest file: {largest} ({size_mb(os.path.join(DEMO_INDEX, largest)):.1f} MB, GitHub limit is 100 MB)")


if __name__ == "__main__":
    main()
