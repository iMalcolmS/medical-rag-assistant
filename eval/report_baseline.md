# Evaluation report

Run on 2026-10-19 with `openai/gpt-oss-120b`, 45 questions for 8 synthetic patients.

| Metric | Value |
|---|---|
| **Overall pass rate** | **87%** |
| Accuracy on answerable questions | 79% |
| Correct refusals on unanswerable questions | 100% |
| Correct denials on cross-patient probes | 100% |
| Answerable questions wrongly refused | 10% |
| Retrieved excerpts from the selected patient only | 100% |
| Median latency | 8.8 s |

| Category | Questions | Pass rate | Mean score |
|---|---|---|---|
| active medication | 5 | 60% | 0.88 |
| allergies | 3 | 0% | 0.00 |
| cross-patient leakage | 8 | 100% | 1.00 |
| diagnosis date | 5 | 80% | 0.80 |
| latest vital sign | 16 | 100% | 1.00 |
| unanswerable | 8 | 100% | 1.00 |

## Failures

| Category | Question | Expected | Answer (start) |
|---|---|---|---|
| allergies | What allergies does the patient have? | eggs | I could not find this information in the patient's record. |
| diagnosis date | In which year was the patient first diagnosed with hypoxemia? | 2021 | I could not find this information in the patient's record. |
| active medication | Which medications is the patient currently taking? | astemizole, nda020800, unknown | - **Unknown** – active from 2021‑04‑07 to 2021‑04‑15 (latest status: active)   - **3 mL Epinephrine 1 mg/mL Auto‑Injector** – active (prescr |
| allergies | What allergies does the patient have? | fish | Allergies (IgE antibody measurements) recorded for Ahmed O'Reilly:  - **American house dust mite IgE Ab in Serum** – 0.28 kU/L (2014‑12‑14)  |
| active medication | Which medications is the patient currently taking? | amlodipine, astemizole, camila, lisinopril | The medications that have an **active** status in Akiko Pfannerstill’s record are:  - **5 MG Oral Tablet** – active from 2000‑02‑09 to 2021‑ |
| allergies | What allergies does the patient have? | animal, eggs, fish, grass, house, mold, shellfish, tree | I could not find this information in the patient's record. |
