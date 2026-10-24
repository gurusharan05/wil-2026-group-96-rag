"""Evaluate retrieval using source-level and keyword-level metrics.

Metrics:
- Source_Hit@K: at least one retrieved result matches the expected university.
- Keyword_Hit@K: at least one individual retrieved chunk contains all expected keywords.
- Combined_Keyword_Hit@K: all expected keywords are found across the combined text
  of the top-K retrieved chunks.
- In-domain metrics: corresponding metrics for in-domain questions only.
- OutOfDomain_Threshold_Rejection: the top result for an out-of-domain question
  is beyond MAX_DISTANCE.

Keyword metrics are retrieval diagnostics, not a direct measure of answer factuality.
"""
from __future__ import annotations

import csv
import re

from config import EVALUATION_DIR, MAX_DISTANCE, RESULTS_DIR
from rag import retrieve


def load_questions():
    path = EVALUATION_DIR.parent / "data" / "structured" / "evaluation_questions.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_text(text: str) -> str:
    """Normalize punctuation and spacing for robust keyword comparisons."""
    text = str(text).lower()
    # Remove commas used as thousands separators: 57,984 -> 57984.
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"[^a-z0-9.%]+", " ", text)
    return " ".join(text.split())


def expected_terms(expected_keywords: str) -> list[str]:
    """Split semicolon-separated expected keywords and normalize each term."""
    return [
        normalize_text(term)
        for term in (expected_keywords or "").split(";")
        if normalize_text(term)
    ]


def result_text(result) -> str:
    """Combine a retrieved chunk's text and metadata for matching."""
    metadata = result.metadata or {}
    return normalize_text(
        " ".join([result.text or "", *[str(value) for value in metadata.values()]])
    )


def keyword_match(result, expected_keywords: str) -> bool:
    """Return True only if all expected terms occur in one retrieved result."""
    terms = expected_terms(expected_keywords)
    if not terms:
        return False
    haystack = result_text(result)
    return all(term in haystack for term in terms)


def keyword_match_across_results(results, expected_keywords: str) -> bool:
    """Return True if all expected terms occur across the combined retrieved chunks."""
    terms = expected_terms(expected_keywords)
    if not terms:
        return False
    haystack = " ".join(result_text(result) for result in results)
    return all(term in haystack for term in terms)


def source_match(result, expected_university: str) -> bool:
    """Check whether a result's university metadata matches the expected source."""
    if not expected_university:
        return False
    university = str((result.metadata or {}).get("university", "")).strip().lower()
    expected = expected_university.strip().lower()
    return university == expected


def metric_row(name: str, hits: int, total: int) -> dict:
    return {
        "metric": name,
        "value": round(hits / total, 4) if total else 0.0,
        "hits": hits,
        "total": total,
    }


def main():
    questions = load_questions()
    if not questions:
        raise ValueError("No evaluation questions were found.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detailed = []
    # Retrieve each question once, then reuse results for all metrics.
    retrieved_by_id = {}

    for row in questions:
        results = retrieve(row["question"], top_k=5)
        retrieved_by_id[row["id"]] = results

        for rank, result in enumerate(results, 1):
            metadata = result.metadata or {}
            detailed.append({
                "id": row["id"],
                "question": row["question"],
                "domain": row["domain"],
                "expected_keywords": row["expected_keywords"],
                "expected_university": row["expected_university"],
                "rank": rank,
                "distance": round(result.distance, 6),
                "source_file": metadata.get("source_file", ""),
                "university": metadata.get("university", ""),
                "category": metadata.get("category", ""),
                "authority": metadata.get("authority", ""),
                "section": metadata.get("section", ""),
                "source_match": source_match(result, row["expected_university"]),
                "keyword_hit": keyword_match(result, row["expected_keywords"]),
                "within_threshold": result.distance <= MAX_DISTANCE,
            })

    detail_path = RESULTS_DIR / "retrieval_results.csv"
    detail_fields = [
        "id", "question", "domain", "expected_keywords", "expected_university",
        "rank", "distance", "source_file", "university", "category", "authority",
        "section", "source_match", "keyword_hit", "within_threshold",
    ]
    with detail_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detailed)

    summary = []
    ids = [row["id"] for row in questions]

    # Source-level Hit@K: at least one result matches the expected university.
    for k in (1, 3, 5):
        hits = 0
        for qid in ids:
            rows = [item for item in detailed if item["id"] == qid and item["rank"] <= k]
            if any(item["source_match"] for item in rows):
                hits += 1
        summary.append(metric_row(f"Source_Hit@{k}", hits, len(ids)))

    # Strict per-chunk keyword Hit@K: one chunk must contain every expected term.
    for k in (1, 3, 5):
        hits = 0
        for qid in ids:
            rows = [item for item in detailed if item["id"] == qid and item["rank"] <= k]
            if any(item["keyword_hit"] for item in rows):
                hits += 1
        summary.append(metric_row(f"Keyword_Hit@{k}", hits, len(ids)))

    # Combined keyword Hit@K: expected terms may occur in different top-K chunks.
    for k in (1, 3, 5):
        hits = sum(
            keyword_match_across_results(retrieved_by_id[q["id"]][:k], q["expected_keywords"])
            for q in questions
        )
        summary.append(metric_row(f"Combined_Keyword_Hit@{k}", hits, len(ids)))

    in_domain = [row for row in questions if row["domain"] == "in-domain"]

    # In-domain source retrieval.
    for k in (1, 3, 5):
        hits = 0
        for question in in_domain:
            rows = [item for item in detailed if item["id"] == question["id"] and item["rank"] <= k]
            if any(item["source_match"] for item in rows):
                hits += 1
        summary.append(metric_row(f"InDomain_Source_Hit@{k}", hits, len(in_domain)))

    # In-domain strict per-chunk keyword retrieval.
    for k in (1, 3, 5):
        hits = 0
        for question in in_domain:
            rows = [item for item in detailed if item["id"] == question["id"] and item["rank"] <= k]
            if any(item["keyword_hit"] for item in rows):
                hits += 1
        summary.append(metric_row(f"InDomain_Keyword_Hit@{k}", hits, len(in_domain)))

    # In-domain combined keyword coverage across the top-K chunks.
    for k in (1, 3, 5):
        hits = sum(
            keyword_match_across_results(retrieved_by_id[q["id"]][:k], q["expected_keywords"])
            for q in in_domain
        )
        summary.append(metric_row(f"InDomain_Combined_Keyword_Hit@{k}", hits, len(in_domain)))

    # Out-of-domain rejection: top result must exceed the configured distance limit.
    out_of_domain = [row for row in questions if row["domain"] == "out-of-domain"]
    if out_of_domain:
        rejected = 0
        for question in out_of_domain:
            results = retrieved_by_id[question["id"]]
            if results and results[0].distance > MAX_DISTANCE:
                rejected += 1
        summary.append(metric_row("OutOfDomain_Threshold_Rejection", rejected, len(out_of_domain)))

    summary_path = RESULTS_DIR / "retrieval_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value", "hits", "total"])
        writer.writeheader()
        writer.writerows(summary)

    print("\nRetrieval evaluation")
    print("-" * 55)
    for row in summary:
        print(f"{row['metric']}: {row['value'] * 100:.1f}% ({row['hits']}/{row['total']})")

    print(f"\nDetailed results: {detail_path}")
    print(f"Summary:          {summary_path}")


if __name__ == "__main__":
    main()
