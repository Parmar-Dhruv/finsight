"""
================================================================================
FinSight - Task N/D-1: Systematic Error Analysis
================================================================================

Analyzes the output of Task J-5 (evaluation/results/j5_comparative_matrix.json)
across both Retrieval (Nilay) and Generation (Dhruv) modules, categorizing each
query outcome into:
  - success:                     Correct retrieval (top-5) + citation + numeric/semantic match
  - retrieval_miss:              Expected ticker/section not present in top-5 retrieved chunks
  - ranking_error:               Expected chunk in top-5 (rank 2-5), but not rank 1
  - numeric_extraction_error:    Expected chunk retrieved, but key gold number missing from answer
  - generation_hallucination:    Answer ungrounded / missing citation / low faithfulness
  - refusal:                     System returned canonical REFUSAL_MESSAGE

Produces per-company and per-question-type failure distributions and saves
the full diagnostic report to evaluation/results/nd1_error_analysis.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

J5_RESULTS_PATH = BASE_DIR / "evaluation" / "results" / "j5_comparative_matrix.json"
ND1_OUTPUT_PATH = BASE_DIR / "evaluation" / "results" / "nd1_error_analysis.json"


def classify_item_error(item: Dict[str, Any], use_rag: bool = True) -> str:
    """Categorizes a single question result into a diagnostic failure bucket."""
    if not use_rag:
        if item.get("is_refusal"):
            return "refusal"
        if item.get("numeric_match") is False:
            return "generation_hallucination_or_ungrounded"
        if not item.get("has_citation"):
            return "generation_hallucination_or_ungrounded"
        return "success"

    if item.get("is_refusal"):
        return "refusal"

    match_rank = item.get("match_rank")
    if match_rank is None:
        return "retrieval_miss"

    if not item.get("has_citation") or item.get("faithfulness", 0.0) < 0.5:
        return "generation_hallucination_or_ungrounded"

    num_match = item.get("numeric_match")
    if num_match is False:
        if match_rank > 1:
            return "ranking_error_causing_numeric_miss"
        return "numeric_extraction_error"

    if match_rank > 1 and item.get("token_f1", 0.0) < 0.25:
        return "ranking_error"

    return "success"


def analyze_configuration(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Computes full error taxonomy, company breakdown, and type breakdown."""
    summary = config_data["summary"]
    use_rag = summary.get("use_rag", True)
    items = config_data.get("detailed_results", [])

    overall_counts = Counter()
    by_company: Dict[str, Counter] = defaultdict(Counter)
    by_type: Dict[str, Counter] = defaultdict(Counter)
    sample_failures: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for item in items:
        cat = classify_item_error(item, use_rag=use_rag)
        overall_counts[cat] += 1
        by_company[item["ticker"]][cat] += 1
        by_type[item["answer_type"]][cat] += 1

        if cat != "success" and len(sample_failures[cat]) < 5:
            sample_failures[cat].append({
                "id": item["id"],
                "ticker": item["ticker"],
                "answer_type": item["answer_type"],
                "query": item["query"],
                "match_rank": item.get("match_rank"),
                "retrieved_top_sections": item.get("retrieved_top_sections", [])[:3],
                "answer": item.get("answer", "")[:220],
                "gold_answer": item.get("gold_answer", "")[:220],
            })

    total = len(items) or 1
    overall_pct = {k: round((v / total) * 100, 2) for k, v in overall_counts.items()}

    return {
        "config_id": summary["config_id"],
        "config_name": summary["config_name"],
        "total_questions": len(items),
        "error_counts": dict(overall_counts),
        "error_percentages": overall_pct,
        "by_company": {k: dict(v) for k, v in sorted(by_company.items())},
        "by_question_type": {k: dict(v) for k, v in sorted(by_type.items())},
        "sample_failures": dict(sample_failures),
    }


def main() -> None:
    if not J5_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"J-5 results not found at {J5_RESULTS_PATH}. Run eval/j5_comparative_eval.py first."
        )

    j5_data = json.loads(J5_RESULTS_PATH.read_text(encoding="utf-8"))
    configs = j5_data.get("configurations", {})

    analyses: Dict[str, Any] = {}
    for cfg_id, cfg_data in configs.items():
        analyses[cfg_id] = analyze_configuration(cfg_data)

    # Print diagnostic summary for Full FinSight System vs Naive RAG vs Domain Retrieval Only
    print("=" * 95)
    print("  FINSIGHT TASK N/D-1: ERROR ANALYSIS SUMMARY (100 QUESTIONS)")
    print("=" * 95)
    for cfg_id in ["naive_rag", "domain_retrieval_only", "full_finsight_system"]:
        if cfg_id not in analyses:
            continue
        a = analyses[cfg_id]
        print(f"\n[{a['config_name']}]")
        for cat, cnt in sorted(a["error_counts"].items(), key=lambda x: -x[1]):
            pct = a["error_percentages"][cat]
            print(f"  - {cat:<40} : {cnt:3d} ({pct:5.1f}%)")

        if cfg_id == "full_finsight_system":
            print("\n  Full System Failures by Question Type (non-success counts):")
            for qtype, counts in a["by_question_type"].items():
                fails = {k: v for k, v in counts.items() if k != "success"}
                if fails:
                    print(f"    * {qtype:<22}: {fails}")
            print("\n  Full System Failures by Company (non-success counts):")
            for comp, counts in a["by_company"].items():
                fails = {k: v for k, v in counts.items() if k != "success"}
                if fails:
                    print(f"    * {comp:<8}: {fails}")

    print("=" * 95)

    ND1_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ND1_OUTPUT_PATH.write_text(json.dumps(analyses, indent=2), encoding="utf-8")
    print(f"\n[OK] Task N/D-1 Error Analysis saved to: {ND1_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
