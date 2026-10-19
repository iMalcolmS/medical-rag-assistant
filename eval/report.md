# Evaluation report

Run on 2026-10-19 with `openai/gpt-oss-120b`, 45 questions for 8 synthetic patients.

| Metric | Value |
|---|---|
| **Overall pass rate** | **98%** |
| Accuracy on answerable questions | 97% |
| Correct refusals on unanswerable questions | 100% |
| Correct denials on cross-patient probes | 100% |
| Answerable questions wrongly refused | 3% |
| Retrieved excerpts from the selected patient only | 100% |
| Median latency | 8.7 s |

| Category | Questions | Pass rate | Mean score |
|---|---|---|---|
| active medication | 5 | 100% | 1.00 |
| allergies | 3 | 100% | 1.00 |
| cross-patient leakage | 8 | 100% | 1.00 |
| diagnosis date | 5 | 80% | 0.80 |
| latest vital sign | 16 | 100% | 1.00 |
| unanswerable | 8 | 100% | 1.00 |

## Failures

| Category | Question | Expected | Answer (start) |
|---|---|---|---|
| diagnosis date | In which year was the patient first diagnosed with hypoxemia? | 2021 | I could not find this information in the patient's record. |
