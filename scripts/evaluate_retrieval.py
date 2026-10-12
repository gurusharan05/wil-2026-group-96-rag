"""Evaluate retrieval using source-level and keyword-level metrics."""
from __future__ import annotations

import csv

from config import EVALUATION_DIR, MAX_DISTANCE, RESULTS_DIR
from rag import retrieve


def load_questions():
    path = EVALUATION_DIR.parent / "data" / "structured" / "evaluation_questions.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def keyword_match(result, expected_keywords: str) -> bool:
    terms = [x.strip().lower() for x in expected_keywords.split(";") if x.strip()]
    if not terms:
        return False

    m = result.metadata
    haystack = " ".join(
        [result.text, *[str(v) for v in m.values()]]
    ).lower()

    return all(term in haystack for term in terms)


def source_match(result, expected_university: str) -> bool:
    if not expected_university:
        return False

    university = str(result.metadata.get("university", "")).strip().lower()
    expected = expected_university.strip().lower()

    return university == expected


def main():
    questions = load_questions()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    detailed = []

    for row in questions:
        results = retrieve(row["question"], top_k=5)

        for rank, result in enumerate(results, 1):
            m = result.metadata

            detailed.append({
                "id": row["id"],
                "question": row["question"],
                "domain": row["domain"],
                "expected_keywords": row["expected_keywords"],
                "expected_university": row["expected_university"],
                "rank": rank,
                "distance": round(result.distance, 6),
                "source_file": m.get("source_file", ""),
                "university": m.get("university", ""),
                "category": m.get("category", ""),
                "authority": m.get("authority", ""),
                "section": m.get("section", ""),
                "source_match": source_match(
                    result,
                    row["expected_university"]
                ),
                "keyword_hit": keyword_match(
                    result,
                    row["expected_keywords"]
                ),
                "within_threshold": result.distance <= MAX_DISTANCE,
            })

    detail_path = RESULTS_DIR / "retrieval_results.csv"

    with detail_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=detailed[0].keys()
        )
        writer.writeheader()
        writer.writerows(detailed)

    summary = []

    ids = [r["id"] for r in questions]

    # Source-level Hit@K
    for k in (1, 3, 5):
        hits = 0

        for qid in ids:
            rows = [
                r for r in detailed
                if r["id"] == qid and r["rank"] <= k
            ]

            if any(r["source_match"] for r in rows):
                hits += 1

        summary.append({
            "metric": f"Source_Hit@{k}",
            "value": round(hits / len(ids), 4),
            "hits": hits,
            "total": len(ids),
        })

    # Keyword-level Hit@K
    for k in (1, 3, 5):
        hits = 0

        for qid in ids:
            rows = [
                r for r in detailed
                if r["id"] == qid and r["rank"] <= k
            ]

            if any(r["keyword_hit"] for r in rows):
                hits += 1

        summary.append({
            "metric": f"Keyword_Hit@{k}",
            "value": round(hits / len(ids), 4),
            "hits": hits,
            "total": len(ids),
        })

    # In-domain source retrieval
    in_domain = [
        r for r in questions
        if r["domain"] == "in-domain"
    ]

    for k in (1, 3, 5):
        hits = 0

        for q in in_domain:
            rows = [
                r for r in detailed
                if r["id"] == q["id"] and r["rank"] <= k
            ]

            if any(r["source_match"] for r in rows):
                hits += 1

        summary.append({
            "metric": f"InDomain_Source_Hit@{k}",
            "value": round(hits / len(in_domain), 4),
            "hits": hits,
            "total": len(in_domain),
        })

    # In-domain keyword retrieval
    for k in (1, 3, 5):
        hits = 0

        for q in in_domain:
            rows = [
                r for r in detailed
                if r["id"] == q["id"] and r["rank"] <= k
            ]

            if any(r["keyword_hit"] for r in rows):
                hits += 1

        summary.append({
            "metric": f"InDomain_Keyword_Hit@{k}",
            "value": round(hits / len(in_domain), 4),
            "hits": hits,
            "total": len(in_domain),
        })

    # Out-of-domain behaviour
    out_of_domain = [
        r for r in questions
        if r["domain"] == "out-of-domain"
    ]

    if out_of_domain:
        rejected = 0

        for q in out_of_domain:
            rows = [
                r for r in detailed
                if r["id"] == q["id"] and r["rank"] == 1
            ]

            if rows and not rows[0]["within_threshold"]:
                rejected += 1

        summary.append({
            "metric": "OutOfDomain_Threshold_Rejection",
            "value": round(rejected / len(out_of_domain), 4),
            "hits": rejected,
            "total": len(out_of_domain),
        })

    summary_path = RESULTS_DIR / "retrieval_summary.csv"

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary[0].keys()
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\nRetrieval evaluation")
    print("-" * 50)

    for row in summary:
        print(
            f"{row['metric']}: "
            f"{row['value'] * 100:.1f}% "
            f"({row['hits']}/{row['total']})"
        )

    print(f"\nDetailed results: {detail_path}")
    print(f"Summary:          {summary_path}")


if __name__ == "__main__":
    main()
