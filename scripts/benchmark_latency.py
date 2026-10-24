"""Measure how long the assistant takes to answer, per question type.

Run from the project root, with Ollama running and the knowledge base built:

    PYTHONPATH=. python scripts/benchmark_latency.py

Optional: number of repeats per question (default 3)

    PYTHONPATH=. python scripts/benchmark_latency.py --repeats 5

What it does
------------
1. Warms up both Ollama models (so model loading is not counted, just like
   the app does once at start-up). The warm-up time is reported separately.
2. Asks every question in QUESTIONS several times and times the full
   pipeline (plan -> retrieve -> answer) with answer_question().
3. Saves every run to evaluation/results/latency_results.csv and a summary
   per question type to evaluation/results/latency_summary.csv, and prints a
   table you can copy into the report.
"""
from __future__ import annotations

import argparse
import csv
import platform
import statistics
import time

from config import LLM_MODEL, RESULTS_DIR
from rag import answer_question, warm_up

# (question type, question). Edit or extend freely.
QUESTIONS = [
    ("Fee (single university)", "What are the fees for RMIT?"),
    ("Fee (single university)", "What is the international fee for Swinburne?"),
    ("Fee (single university)", "How much does the Master of Data Science cost at La Trobe?"),
    ("Duration / campus", "How long is the Deakin master of data science?"),
    ("Duration / campus", "Which campus hosts the University of Melbourne Master of Data Science?"),
    ("Comparison", "Compare the fees of RMIT and Monash"),
    ("Comparison", "Which university is the cheapest?"),
    ("Accommodation", "What accommodation does Monash offer?"),
    ("Work rights", "How many hours can I work per fortnight on a student visa?"),
    ("Open question (LLM)", "What will I study in the RMIT Master of Data Science?"),
    ("Open question (LLM)", "Tell me about the Monash Master of Data Science."),
    ("Out of domain", "What will the weather be in Melbourne tomorrow?"),
]


def time_one(question: str) -> dict:
    """Time one full answer. Streams LLM answers and records time to first text."""
    start = time.perf_counter()
    result = answer_question(question, stream=True)
    first_text = None
    if result.stream is not None:
        parts = []
        for piece in result.stream:
            if first_text is None and piece.strip():
                first_text = time.perf_counter() - start
            parts.append(piece)
        result.answer = "".join(parts)
    total = time.perf_counter() - start
    if first_text is None:
        first_text = total
    return {
        "route": result.route,
        "total_s": total,
        "first_text_s": first_text,
        "evidence": len(result.evidence),
        "answer_chars": len(result.answer),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    print("=" * 78)
    print(f"Latency benchmark  |  LLM: {LLM_MODEL}  |  {platform.platform()}")
    print("=" * 78)

    t = time.perf_counter()
    status = warm_up()
    print(f"Model warm-up: {time.perf_counter() - t:.1f}s  {status}\n")

    rows = []
    for qtype, question in QUESTIONS:
        for run in range(1, args.repeats + 1):
            measured = time_one(question)
            rows.append({"type": qtype, "question": question, "run": run, **measured})
            print(f"[{measured['route']:<11}] {measured['total_s']:6.2f}s  "
                  f"(first text {measured['first_text_s']:5.2f}s)  {question}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = RESULTS_DIR / "latency_results.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for qtype in dict.fromkeys(q for q, _ in QUESTIONS):
        group = [r for r in rows if r["type"] == qtype]
        totals = [r["total_s"] for r in group]
        firsts = [r["first_text_s"] for r in group]
        summary.append({
            "type": qtype,
            "route": "/".join(sorted({r["route"] for r in group})),
            "runs": len(group),
            "median_total_s": round(statistics.median(totals), 2),
            "max_total_s": round(max(totals), 2),
            "median_first_text_s": round(statistics.median(firsts), 2),
        })

    summary_path = RESULTS_DIR / "latency_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\n" + "-" * 78)
    print(f"{'Question type':<26}{'Route':<14}{'Median':>9}{'Max':>9}{'First text':>13}")
    print("-" * 78)
    for s in summary:
        print(f"{s['type']:<26}{s['route']:<14}{s['median_total_s']:>8.2f}s"
              f"{s['max_total_s']:>8.2f}s{s['median_first_text_s']:>12.2f}s")
    print("-" * 78)
    print(f"Detailed runs: {detail_path}")
    print(f"Summary:       {summary_path}")


if __name__ == "__main__":
    main()
