"""Run the evaluation set against the RAG pipeline and report accuracy.

Usage:  python -m eval.run_eval [--sleep 3] [--limit N]
Reads:  eval/questions.json   (python -m eval.build_eval_set)
Writes: eval/results.csv      one row per question, with the full answer
        eval/report.md        summary tables for the README
"""

import argparse
import csv
import json
import os
import statistics
import time
from datetime import date

from eval.scoring import is_refusal, score
from src import rag_pipeline

QUESTIONS_PATH = os.path.join("eval", "questions.json")
RESULTS_PATH = os.path.join("eval", "results.csv")
REPORT_PATH = os.path.join("eval", "report.md")
PATIENT_LIMIT = int(os.environ.get("PATIENT_LIMIT", 20))


def ask(index, q, retries=4):
    """Query with retries, since the Groq free tier rate-limits bursts of requests."""
    for attempt in range(retries):
        try:
            start = time.perf_counter()
            response = rag_pipeline.query_patient(index, q["patient_id"], q["question"])
            return response, time.perf_counter() - start
        except Exception as error:  # noqa: BLE001 - report and retry any API error
            if attempt == retries - 1:
                raise
            wait = 20 * (attempt + 1)
            print(f"   ! {type(error).__name__}: retrying in {wait}s")
            time.sleep(wait)


def pct(values):
    return f"{100 * sum(values) / len(values):.0f}%" if values else "–"


def write_report(rows, model):
    categories = sorted({r["category"] for r in rows})
    answerable = [r for r in rows if r["category"] not in ("unanswerable", "cross-patient leakage")]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    leakage = [r for r in rows if r["category"] == "cross-patient leakage"]

    lines = [
        "# Evaluation report",
        "",
        f"Run on {date.today().isoformat()} with `{model}`, {len(rows)} questions "
        f"for {len({r['patient_id'] for r in rows})} synthetic patients.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Overall pass rate** | **{pct([r['passed'] for r in rows])}** |",
        f"| Accuracy on answerable questions | {pct([r['passed'] for r in answerable])} |",
        f"| Correct refusals on unanswerable questions | {pct([r['passed'] for r in unanswerable])} |",
        f"| Correct denials on cross-patient probes | {pct([r['passed'] for r in leakage])} |",
        f"| Answerable questions wrongly refused | {pct([r['refused'] for r in answerable])} |",
        f"| Retrieved excerpts from the selected patient only | {pct([r['patient_purity'] == 1.0 for r in rows])} |",
        f"| Median latency | {statistics.median(r['latency_s'] for r in rows):.1f} s |",
        "",
        "| Category | Questions | Pass rate | Mean score |",
        "|---|---|---|---|",
    ]
    for c in categories:
        group = [r for r in rows if r["category"] == c]
        lines.append(f"| {c} | {len(group)} | {pct([r['passed'] for r in group])} | "
                     f"{statistics.mean(r['score'] for r in group):.2f} |")
    failures = [r for r in rows if not r["passed"]]
    if failures:
        lines += ["", "## Failures", "", "| Category | Question | Expected | Answer (start) |", "|---|---|---|---|"]
        for r in failures:
            answer = r["answer"].replace("\n", " ").replace("|", "/")[:140]
            lines.append(f"| {r['category']} | {r['question']} | {r['expected']} | {answer} |")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sleep", type=float, default=3.0, help="seconds between questions (rate limits)")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    args = parser.parse_args()

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)[: args.limit]

    rag_pipeline.configure_settings()
    index = rag_pipeline.get_index(limit=PATIENT_LIMIT)
    model = getattr(rag_pipeline.Settings.llm, "model", "unknown model")

    rows = []
    for i, q in enumerate(questions, 1):
        response, latency = ask(index, q)
        answer = str(response).strip()
        passed, value = score(q["check"], answer)
        sources = response.source_nodes
        purity = (sum(n.metadata.get("patient_id") == q["patient_id"] for n in sources) / len(sources)) if sources else 1.0
        rows.append({
            "id": q["id"], "patient_id": q["patient_id"], "category": q["category"],
            "question": q["question"], "expected": q["expected"], "answer": answer,
            "passed": passed, "score": round(value, 2), "refused": is_refusal(answer),
            "patient_purity": purity, "latency_s": round(latency, 2), "n_sources": len(sources),
        })
        print(f"[{i:>2}/{len(questions)}] {'PASS' if passed else 'FAIL'}  {q['category']:<22} {q['question']}")
        time.sleep(args.sleep)

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(rows, model)

    print(f"\nPass rate: {pct([r['passed'] for r in rows])} on {len(rows)} questions")
    print(f"Details: {RESULTS_PATH} · Summary: {REPORT_PATH}")


if __name__ == "__main__":
    main()
