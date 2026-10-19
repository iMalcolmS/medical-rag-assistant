"""Generate an evaluation set with known answers from the structured FHIR data.

Ground truth comes from the same bundles the index was built from, read with
src.patient_summary, so no answer is written by hand and the set can be
regenerated for any Synthea sample.

Usage:  python -m eval.build_eval_set --patients 8
Writes: eval/questions.json
"""

import argparse
import json
import os

from src.fhir_loader import load_fhir_bundle
from src.patient_summary import (
    allergies,
    key_measures,
    medication_list,
    observation_series,
    patient_header,
    problem_list,
)
from src.rag_pipeline import list_bundle_files

OUT_PATH = os.path.join("eval", "questions.json")
PATIENT_LIMIT = int(os.environ.get("PATIENT_LIMIT", 20))  # must match the indexed patients

# Things Synthea never records: the assistant must refuse rather than guess.
UNANSWERABLE = [
    "What is the patient's blood type?",
    "What is the name of the patient's GP?",
    "What is the patient's NHS number?",
]

STOPWORDS = {"allergy", "to", "of", "the", "and"}


# Dose forms and units that are not the drug's name
DOSE_WORDS = {
    "oral", "tablet", "tablets", "capsule", "capsules", "injector", "auto", "pack", "day", "hr",
    "extended", "release", "solution", "injection", "prefilled", "syringe", "topical", "cream",
    "inhaler", "inhalant", "metered", "dose", "actuat", "mucosal", "spray", "chewable", "unt",
    "suspension", "film", "patch", "transdermal", "nasal", "ophthalmic", "otic",
}


def drug_keyword(medication):
    """The first word that names the drug, skipping codes, doses and units.

    'Atorvastatin 40 MG Oral Tablet'                        -> 'atorvastatin'
    'NDA020800 0.3 ML Epinephrine 1 MG/ML Auto-Injector'   -> 'epinephrine'
    '24 HR Metformin hydrochloride 500 MG Extended Release' -> 'metformin'
    """
    for raw in medication.replace("/", " ").replace("-", " ").split():
        word = raw.strip("(),[]").lower()
        if word.isalpha() and len(word) > 3 and word not in DOSE_WORDS:
            return word
    return medication.split()[0].lower()


def allergy_keyword(substance):
    """'Penicillin V' -> 'penicillin', 'Allergy to mould' -> 'mould'."""
    words = [w.strip("(),").lower() for w in substance.split()]
    words = [w for w in words if w and w not in STOPWORDS]
    return words[0] if words else substance.lower()


def patient_questions(bundle, index, other_conditions):
    header = patient_header(bundle)
    pid, name = header["id"], header["name"]
    questions = []

    def add(category, question, check, expected):
        questions.append({
            "id": f"{pid[:8]}-{len(questions) + 1:02d}",
            "patient_id": pid,
            "patient_name": name,
            "category": category,
            "question": question,
            "check": check,
            "expected": expected,
        })

    clinical = [p for p in problem_list(bundle) if not p["social"]]
    names = [p["problem"].lower() for p in clinical]

    # 1. Onset of an active condition recorded only once (unambiguous date)
    unique_active = [p for p in clinical if p["status"] == "active" and names.count(p["problem"].lower()) == 1]
    if unique_active:
        p = unique_active[0]
        year = p["onset"][:4]
        add("diagnosis date", f"In which year was the patient first diagnosed with {p['problem'].lower()}?",
            {"type": "year", "year": year}, year)

    # 2. Active medication
    active = [m for m in medication_list(bundle) if m["status"] == "active"]
    if active:
        keywords = sorted({drug_keyword(m["medication"]) for m in active})
        add("active medication", "Which medications is the patient currently taking?",
            {"type": "keywords", "keywords": keywords}, ", ".join(keywords))

    # 3-4. Most recent vital signs
    for m in key_measures(observation_series(bundle)):
        if m["label"] == "Blood pressure" and "diastolic" in m:
            add("latest vital sign", "What was the patient's most recent blood pressure reading?",
                {"type": "numbers", "numbers": [m["value"], m["diastolic"]], "tol": 0.5},
                f"{m['value']:.0f}/{m['diastolic']:.0f} mmHg")
        elif m["label"] == "BMI":
            add("latest vital sign", "What was the patient's most recent BMI?",
                {"type": "numbers", "numbers": [round(m["value"], 1)], "tol": 0.15},
                f"{m['value']:.1f}")

    # 5. Allergies (only patients that have some)
    allergy_list = allergies(bundle)
    if allergy_list:
        keywords = sorted({allergy_keyword(a["substance"]) for a in allergy_list})
        add("allergies", "What allergies does the patient have?",
            {"type": "keywords", "keywords": keywords}, ", ".join(keywords))

    # 6. Leakage probe: a condition another indexed patient has, but this one does not
    absent = [c for c in other_conditions if c.lower() not in names]
    if absent:
        add("cross-patient leakage", f"Has the patient ever been diagnosed with {absent[index % len(absent)].lower()}?",
            {"type": "negative"}, "No / not in record")

    # 7. Information Synthea never records
    add("unanswerable", UNANSWERABLE[index % len(UNANSWERABLE)], {"type": "refuse"}, "Refusal")
    return questions


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--patients", type=int, default=8, help="number of indexed patients to use")
    args = parser.parse_args()

    bundles = [load_fhir_bundle(p) for p in list_bundle_files(limit=PATIENT_LIMIT)]
    all_conditions = sorted({p["problem"] for b in bundles for p in problem_list(b)
                             if not p["social"] and p["status"] == "active"})

    questions = []
    for i, bundle in enumerate(bundles[: args.patients]):
        questions += patient_questions(bundle, i, all_conditions)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    by_category = {}
    for q in questions:
        by_category[q["category"]] = by_category.get(q["category"], 0) + 1
    print(f"Wrote {len(questions)} questions for {min(args.patients, len(bundles))} patients to {OUT_PATH}")
    for category, n in sorted(by_category.items()):
        print(f"  {category:<22} {n}")


if __name__ == "__main__":
    main()
