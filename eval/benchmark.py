"""
================================================================================
FinSight - Task N-2: Full 100-Question Retrieval Benchmark Evaluation
================================================================================

Task N-2 (docs/finsight-task-breakdown.md):
  "Import evaluation/dataset/questions.json from Jay's branch.
   Convert its 100 questions into the query->gold-chunk format your harness expects.
   Re-run evaluate_retrieval_quality() on this expanded set for both the naive baseline (N-1)
   and the current domain-adapted pipeline.
   Output: updated before/after table on the full 100-question set."
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not installed. Run tracking disabled. Install with: pip install mlflow")

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from retrieval.embed import EmbeddingEngine
from retrieval.search import FinSightRetriever
from eval.naive_baseline import (
    build_naive_corpus,
    build_naive_index,
    naive_search,
    PROCESSED_DIR,
    NAIVE_CHUNK_CHARS,
)

logger = logging.getLogger(__name__)

BENCHMARK_PATH = BASE_DIR / "evaluation" / "dataset" / "questions.json"
RESULTS_DIR = BASE_DIR / "evaluation" / "results"
MLFLOW_EXPERIMENT = "FinSight-Retrieval"
MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"


# ---------------------------------------------------------------------------
# Section Normalizer & Dataset Converter
# ---------------------------------------------------------------------------

def map_source_section_to_canonical(source_section: str) -> List[str]:
    """
    Translates descriptive source_section strings from the 100-question benchmark
    into canonical filing sections indexed by FinSight:
      - item_1  (Business, Human Capital, Overview)
      - item_1a (Risk Factors)
      - item_7  (Management's Discussion & Analysis, Segment, Operations)
      - item_8  (Financial Statements, Notes, Balance Sheets, Cash Flows)
    """
    s = source_section.lower().strip()
    sections: List[str] = []

    # Risk factors
    if "1a" in s or "risk" in s:
        sections.append("item_1a")

    # Item 1 Business
    if "item 1 " in s or "item 1 —" in s or "item 1 -" in s or "business overview" in s or "human capital" in s or "employee" in s:
        if "1a" not in s:
            sections.append("item_1")

    # Item 7 MD&A & Operations
    if any(k in s for k in [
        "item 7", "management's discussion", "md&a", "results of operations",
        "segment", "liquidity", "key metrics", "performance"
    ]):
        sections.append("item_7")

    # Item 8 Financial Statements & Notes
    if any(k in s for k in [
        "item 8", "balance sheet", "cash flow", "taxes",
        "business combination", "notes to financial"
    ]):
        sections.append("item_8")

    # Statements of operations / income are analyzed in MD&A (Item 7) & reported in Item 8
    if "operations" in s or "income" in s or "revenue" in s:
        if "item_7" not in sections:
            sections.append("item_7")
        if "item_8" not in sections:
            sections.append("item_8")

    if not sections:
        sections = ["item_7"]

    return list(dict.fromkeys(sections))


def load_and_convert_benchmark(path: Path = BENCHMARK_PATH) -> List[Dict[str, Any]]:
    """
    Loads Jay's 100-question JSON and converts each question into the format
    expected by the evaluation harness.
    """
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found at: {path}")

    raw_data = json.loads(path.read_text(encoding="utf-8"))
    questions = raw_data.get("questions", [])

    converted_suite: List[Dict[str, Any]] = []
    for q in questions:
        expected_sections = map_source_section_to_canonical(q.get("source_section", ""))
        converted_suite.append({
            "id": q.get("id"),
            "query": q.get("question"),
            "expected_ticker": q.get("source_company"),
            "expected_sections": expected_sections,
            "expected_section": expected_sections[0],  # primary section for backwards compat
            "answer_type": q.get("answer_type", "unknown"),
            "difficulty": q.get("difficulty", "medium"),
            "fiscal_year": q.get("fiscal_year", ""),
            "gold_answer": q.get("gold_answer", ""),
            "raw_source_section": q.get("source_section", ""),
        })

    return converted_suite


# ---------------------------------------------------------------------------
# Evaluator for Naive Baseline on 100 Questions
# ---------------------------------------------------------------------------

def run_naive_benchmark(
    test_suite: List[Dict[str, Any]],
    engine: EmbeddingEngine,
    embeddings: np.ndarray,
    corpus: List[Dict[str, Any]],
    top_k: int = 5,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Evaluates naive fixed-chunk retrieval across the benchmark suite.
    """
    total = len(test_suite)
    hits_at_1 = hits_at_3 = hits_at_5 = 0
    rr_list: List[float] = []
    item_results: List[Dict[str, Any]] = []

    print("\n" + "=" * 78)
    print(f"EVALUATING NAIVE BASELINE ON {total}-QUESTION BENCHMARK")
    print("=" * 78)

    for idx, test in enumerate(test_suite, 1):
        q = test["query"]
        exp_ticker = test["expected_ticker"]
        exp_sections = test["expected_sections"]

        results = naive_search(q, engine, embeddings, corpus, top_k=top_k)

        first_match_rank: Optional[int] = None
        for rank, r in enumerate(results, 1):
            if r["ticker"] == exp_ticker and r["section"] in exp_sections:
                first_match_rank = rank
                break

        if first_match_rank == 1:
            hits_at_1 += 1
        if first_match_rank and first_match_rank <= 3:
            hits_at_3 += 1
        if first_match_rank and first_match_rank <= 5:
            hits_at_5 += 1

        rr = (1.0 / first_match_rank) if first_match_rank else 0.0
        rr_list.append(rr)

        item_results.append({
            "id": test["id"],
            "query": q,
            "ticker": exp_ticker,
            "answer_type": test["answer_type"],
            "expected_sections": exp_sections,
            "match_rank": first_match_rank,
            "reciprocal_rank": rr,
            "retrieved_top_sections": [f"{r['ticker']}:{r['section']}" for r in results],
        })

        if idx % 20 == 0 or idx == total:
            print(f"  Processed {idx:3d}/{total} questions... Current HR@1: {(hits_at_1/idx)*100:5.1f}%, MRR: {sum(rr_list)/idx:.4f}")

    metrics = {
        "hit_rate_at_1": (hits_at_1 / total) * 100,
        "hit_rate_at_3": (hits_at_3 / total) * 100,
        "hit_rate_at_5": (hits_at_5 / total) * 100,
        "mrr": sum(rr_list) / total,
        "total_queries": total,
    }

    return metrics, item_results


# ---------------------------------------------------------------------------
# Evaluator for Domain-Adapted Pipeline on 100 Questions
# ---------------------------------------------------------------------------

def run_domain_adapted_benchmark(
    test_suite: List[Dict[str, Any]],
    retriever: FinSightRetriever,
    top_k: int = 5,
    hybrid: bool = False,
    rerank: bool = False,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Evaluates domain-adapted structure-aware retrieval on Qdrant across benchmark.
    Supports either Dense-Only or Hybrid (Dense + BM25 RRF).
    """
    total = len(test_suite)
    hits_at_1 = hits_at_3 = hits_at_5 = 0
    rr_list: List[float] = []
    item_results: List[Dict[str, Any]] = []

    mode_label = "HYBRID (Dense + BM25 RRF)" if hybrid else "DENSE-ONLY"
    if rerank:
        mode_label += " + RERANK"
    print("\n" + "=" * 78)
    print(f"EVALUATING DOMAIN-ADAPTED [{mode_label}] ON {total}-QUESTION BENCHMARK")
    print("=" * 78)

    for idx, test in enumerate(test_suite, 1):
        q = test["query"]
        exp_ticker = test["expected_ticker"]
        exp_sections = test["expected_sections"]

        # Global search (unconstrained) to stress-test semantic matching
        results = retriever.retrieve(query=q, k=top_k, hybrid=hybrid, rerank=rerank)

        first_match_rank: Optional[int] = None
        for rank, r in enumerate(results, 1):
            if r["ticker"] == exp_ticker and r["section"] in exp_sections:
                first_match_rank = rank
                break

        if first_match_rank == 1:
            hits_at_1 += 1
        if first_match_rank and first_match_rank <= 3:
            hits_at_3 += 1
        if first_match_rank and first_match_rank <= 5:
            hits_at_5 += 1

        rr = (1.0 / first_match_rank) if first_match_rank else 0.0
        rr_list.append(rr)

        item_results.append({
            "id": test["id"],
            "query": q,
            "ticker": exp_ticker,
            "answer_type": test["answer_type"],
            "expected_sections": exp_sections,
            "match_rank": first_match_rank,
            "reciprocal_rank": rr,
            "retrieved_top_sections": [f"{r['ticker']}:{r['section']}" for r in results],
        })

        if idx % 20 == 0 or idx == total:
            print(f"  Processed {idx:3d}/{total} questions... Current HR@1: {(hits_at_1/idx)*100:5.1f}%, MRR: {sum(rr_list)/idx:.4f}")

    metrics = {
        "hit_rate_at_1": (hits_at_1 / total) * 100,
        "hit_rate_at_3": (hits_at_3 / total) * 100,
        "hit_rate_at_5": (hits_at_5 / total) * 100,
        "mrr": sum(rr_list) / total,
        "total_queries": total,
    }

    return metrics, item_results


# ---------------------------------------------------------------------------
# Slice / Breakdown Analysis
# ---------------------------------------------------------------------------

def compute_breakdown_by_category(
    item_results: List[Dict[str, Any]],
    key: str,
) -> Dict[str, Dict[str, float]]:
    """
    Computes HR@1, HR@5, and MRR broken down by a specific field (e.g. answer_type, ticker).
    """
    groups = defaultdict(list)
    for res in item_results:
        groups[res[key]].append(res)

    breakdown = {}
    for grp_val, items in sorted(groups.items()):
        cnt = len(items)
        h1 = sum(1 for it in items if it["match_rank"] == 1)
        h5 = sum(1 for it in items if it["match_rank"] and it["match_rank"] <= 5)
        mrr = sum(it["reciprocal_rank"] for it in items) / cnt
        breakdown[grp_val] = {
            "count": cnt,
            "hr1": (h1 / cnt) * 100,
            "hr5": (h5 / cnt) * 100,
            "mrr": mrr,
        }
    return breakdown


# ---------------------------------------------------------------------------
# Output Formatter & Report Generator
# ---------------------------------------------------------------------------

def print_and_save_full_report(
    naive_metrics: Dict[str, float],
    dense_metrics: Dict[str, float],
    hybrid_metrics: Dict[str, float],
    naive_items: List[Dict[str, Any]],
    dense_items: List[Dict[str, Any]],
    hybrid_items: List[Dict[str, Any]],
    hybrid_rerank_metrics: Dict[str, float],
    hybrid_rerank_items: List[Dict[str, Any]],
    n_queries: int,
) -> str:
    d_dense_hr1 = dense_metrics["hit_rate_at_1"] - naive_metrics["hit_rate_at_1"]
    d_dense_hr5 = dense_metrics["hit_rate_at_5"] - naive_metrics["hit_rate_at_5"]
    d_dense_mrr = dense_metrics["mrr"] - naive_metrics["mrr"]

    d_hyb_hr1 = hybrid_metrics["hit_rate_at_1"] - naive_metrics["hit_rate_at_1"]
    d_hyb_hr3 = hybrid_metrics["hit_rate_at_3"] - naive_metrics["hit_rate_at_3"]
    d_hyb_hr5 = hybrid_metrics["hit_rate_at_5"] - naive_metrics["hit_rate_at_5"]
    d_hyb_mrr = hybrid_metrics["mrr"] - naive_metrics["mrr"]

    d_rr_hr1 = hybrid_rerank_metrics["hit_rate_at_1"] - naive_metrics["hit_rate_at_1"]
    d_rr_hr3 = hybrid_rerank_metrics["hit_rate_at_3"] - naive_metrics["hit_rate_at_3"]
    d_rr_hr5 = hybrid_rerank_metrics["hit_rate_at_5"] - naive_metrics["hit_rate_at_5"]
    d_rr_mrr = hybrid_rerank_metrics["mrr"] - naive_metrics["mrr"]

    report_lines = [
        "",
        "=" * 105,
        f"  FINSIGHT TASK N-4: 100-QUESTION RETRIEVAL BENCHMARK SCOREBOARD",
        f"  Full Benchmark: {n_queries} questions across AAPL, MSFT, GOOGL, AMZN, META",
        "=" * 105,
        f"  {'Metric':<15} {'Naive':>8} {'Dense-Only':>12} {'Hybrid (N-3)':>14} {'Hybrid+Rerank (N-4)':>20} {'Delta vs Naive':>16}",
        "-" * 105,
        f"  {'Hit Rate @ 1':<15} {naive_metrics['hit_rate_at_1']:7.1f}% {dense_metrics['hit_rate_at_1']:11.1f}% {hybrid_metrics['hit_rate_at_1']:13.1f}% {hybrid_rerank_metrics['hit_rate_at_1']:19.1f}% {f'(+{d_rr_hr1:.1f} pp)':>16}",
        f"  {'Hit Rate @ 3':<15} {naive_metrics['hit_rate_at_3']:7.1f}% {dense_metrics['hit_rate_at_3']:11.1f}% {hybrid_metrics['hit_rate_at_3']:13.1f}% {hybrid_rerank_metrics['hit_rate_at_3']:19.1f}% {f'(+{d_rr_hr3:.1f} pp)':>16}",
        f"  {'Hit Rate @ 5':<15} {naive_metrics['hit_rate_at_5']:7.1f}% {dense_metrics['hit_rate_at_5']:11.1f}% {hybrid_metrics['hit_rate_at_5']:13.1f}% {hybrid_rerank_metrics['hit_rate_at_5']:19.1f}% {f'(+{d_rr_hr5:.1f} pp)':>16}",
        f"  {'MRR':<15} {naive_metrics['mrr']:8.4f} {dense_metrics['mrr']:12.4f} {hybrid_metrics['mrr']:14.4f} {hybrid_rerank_metrics['mrr']:20.4f} {f'(+{d_rr_mrr:.4f})':>16}",
        "=" * 105,
        "",
        "Breakdown by Question Type (Hybrid+Rerank Search):",
        "-" * 105,
    ]

    type_breakdown = compute_breakdown_by_category(hybrid_rerank_items, "answer_type")
    dense_type_breakdown = compute_breakdown_by_category(dense_items, "answer_type")
    naive_type_breakdown = compute_breakdown_by_category(naive_items, "answer_type")

    report_lines.append(f"  {'Question Type':<24} {'Count':>6} {'Naive HR@5':>12} {'Dense HR@5':>12} {'Rerank HR@5':>13} {'Rerank MRR':>12}")
    report_lines.append("  " + "-" * 80)
    for q_type, stat in type_breakdown.items():
        n_stat = naive_type_breakdown.get(q_type, {"hr5": 0.0})
        d_stat = dense_type_breakdown.get(q_type, {"hr5": 0.0})
        report_lines.append(
            f"  {q_type:<24} {stat['count']:6d} {n_stat['hr5']:11.1f}% {d_stat['hr5']:11.1f}% {stat['hr5']:12.1f}% {stat['mrr']:12.4f}"
        )

    report_lines.append("")
    report_lines.append("Breakdown by Company (Hybrid+Rerank Search):")
    report_lines.append("-" * 105)
    comp_breakdown = compute_breakdown_by_category(hybrid_rerank_items, "ticker")
    dense_comp_breakdown = compute_breakdown_by_category(dense_items, "ticker")
    naive_comp_breakdown = compute_breakdown_by_category(naive_items, "ticker")
    report_lines.append(f"  {'Company':<24} {'Count':>6} {'Naive HR@5':>12} {'Dense HR@5':>12} {'Rerank HR@5':>13} {'Rerank MRR':>12}")
    report_lines.append("  " + "-" * 80)
    for comp, stat in comp_breakdown.items():
        n_stat = naive_comp_breakdown.get(comp, {"hr5": 0.0})
        d_stat = dense_comp_breakdown.get(comp, {"hr5": 0.0})
        report_lines.append(
            f"  {comp:<24} {stat['count']:6d} {n_stat['hr5']:11.1f}% {d_stat['hr5']:11.1f}% {stat['hr5']:12.1f}% {stat['mrr']:12.4f}"
        )
    report_lines.append("=" * 105)

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save to JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "summary": {
            "total_questions": n_queries,
            "naive_metrics": naive_metrics,
            "dense_metrics": dense_metrics,
            "hybrid_metrics": hybrid_metrics,
            "hybrid_rerank_metrics": hybrid_rerank_metrics,
            "delta_rerank_vs_naive": {
                "hit_rate_at_1": d_rr_hr1,
                "hit_rate_at_5": d_rr_hr5,
                "mrr": d_rr_mrr,
            },
        },
        "by_type": type_breakdown,
        "by_company": comp_breakdown,
        "hybrid_rerank_detailed_results": hybrid_rerank_items,
    }
    out_file = RESULTS_DIR / "n4_rerank_benchmark.json"
    out_file.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"\n[OK] Full benchmark output saved to: {out_file}")

    return report_text


# ---------------------------------------------------------------------------
# MLflow Logging Helper (J-3)
# ---------------------------------------------------------------------------

def log_run_to_mlflow(
    run_name: str,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    breakdown_by_type: Optional[Dict[str, Dict]] = None,
    breakdown_by_company: Optional[Dict[str, Dict]] = None,
    artifact_path: Optional[str] = None,
) -> None:
    """
    Logs a single benchmark run to MLflow (J-3).

    Each call creates one MLflow run under the FinSight-Retrieval experiment,
    logging:
      - Metrics  : hit_rate_at_1, hit_rate_at_3, hit_rate_at_5, mrr
      - Params   : retrieval_mode, top_k, total_queries, hybrid, rerank
      - Metrics  : per-question-type HR@5 & MRR (prefix: type_)
      - Metrics  : per-company HR@5 & MRR (prefix: co_)
      - Artifact : the benchmark result JSON file, if artifact_path is provided

    Parameters
    ----------
    run_name : str
        Display name for the MLflow run (e.g. "N4-Hybrid-BM25-Rerank").
    metrics : dict
        Core IR metrics (hit_rate_at_1, hit_rate_at_3, hit_rate_at_5, mrr).
    params : dict
        Run configuration (retrieval_mode, top_k, hybrid, rerank, total_queries).
    breakdown_by_type : dict, optional
        Per-question-type stats dict: {type_name: {"hr5": float, "mrr": float}}.
    breakdown_by_company : dict, optional
        Per-company stats dict: {ticker: {"hr5": float, "mrr": float}}.
    artifact_path : str, optional
        Filesystem path to a JSON result file to log as a run artifact.
    """
    if not MLFLOW_AVAILABLE:
        print(f"[MLflow] Skipping run logging (mlflow not installed): {run_name}")
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=run_name):
        # Core IR retrieval metrics
        mlflow.log_metric("hit_rate_at_1", metrics.get("hit_rate_at_1", 0.0))
        mlflow.log_metric("hit_rate_at_3", metrics.get("hit_rate_at_3", 0.0))
        mlflow.log_metric("hit_rate_at_5", metrics.get("hit_rate_at_5", 0.0))
        mlflow.log_metric("mrr",           metrics.get("mrr",           0.0))

        # Run configuration parameters
        mlflow.log_params(params)

        # Per-question-type HR@5 & MRR as metrics (prefixed with type_)
        if breakdown_by_type:
            for q_type, stat in breakdown_by_type.items():
                safe_key = q_type.replace(" ", "_").replace("/", "_").lower()
                mlflow.log_metric(f"type_{safe_key}_hr5", stat.get("hr5", 0.0))
                mlflow.log_metric(f"type_{safe_key}_mrr",  stat.get("mrr",  0.0))

        # Per-company HR@5 & MRR as metrics (prefixed with co_)
        if breakdown_by_company:
            for company, stat in breakdown_by_company.items():
                safe_key = company.replace(" ", "_").lower()
                mlflow.log_metric(f"co_{safe_key}_hr5", stat.get("hr5", 0.0))
                mlflow.log_metric(f"co_{safe_key}_mrr",  stat.get("mrr",  0.0))

        # Log the benchmark result JSON as a run artifact for full traceability
        if artifact_path:
            artifact_file = Path(artifact_path)
            if artifact_file.exists():
                mlflow.log_artifact(str(artifact_file), artifact_path="benchmark_results")
            else:
                logger.warning("[MLflow] Artifact file not found, skipping: %s", artifact_path)

        print(f"[MLflow] Logged run '{run_name}' → {MLFLOW_TRACKING_URI}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    print("=" * 85)
    print("FinSight Task N-3/N-4: Hybrid (Dense + BM25) vs Dense vs Naive Benchmark")
    print("=" * 85)

    if MLFLOW_AVAILABLE:
        print(f"[MLflow] Tracking enabled → experiment: '{MLFLOW_EXPERIMENT}' at {MLFLOW_TRACKING_URI}")
    else:
        print("[MLflow] Not installed — run tracking disabled.")

    # 1. Load and convert 100-question benchmark
    print(f"Loading benchmark from: {BENCHMARK_PATH}")
    test_suite = load_and_convert_benchmark(BENCHMARK_PATH)
    n_queries = len(test_suite)
    print(f"Successfully loaded and formatted {n_queries} benchmark questions.")

    # 2. Setup Naive Baseline
    print("\nPreparing Naive Baseline corpus and index...")
    corpus = build_naive_corpus(PROCESSED_DIR, chunk_chars=NAIVE_CHUNK_CHARS)
    engine = EmbeddingEngine()
    embeddings = build_naive_index(corpus, engine)

    # 3. Run Naive Baseline
    naive_metrics, naive_items = run_naive_benchmark(
        test_suite=test_suite,
        engine=engine,
        embeddings=embeddings,
        corpus=corpus,
        top_k=5,
    )
    naive_type_bd  = compute_breakdown_by_category(naive_items, "answer_type")
    naive_comp_bd  = compute_breakdown_by_category(naive_items, "ticker")
    log_run_to_mlflow(
        run_name="N1-Naive-Baseline",
        metrics=naive_metrics,
        params={"retrieval_mode": "naive", "top_k": 5, "hybrid": False, "rerank": False, "total_queries": n_queries},
        breakdown_by_type=naive_type_bd,
        breakdown_by_company=naive_comp_bd,
    )

    # 4. Setup Domain-Adapted Retriever
    print("\nConnecting to Domain-Adapted Retriever (Qdrant & BM25)...")
    retriever = FinSightRetriever.get_instance()

    # 5. Run Domain-Adapted Dense-Only Pipeline
    dense_metrics, dense_items = run_domain_adapted_benchmark(
        test_suite=test_suite,
        retriever=retriever,
        top_k=5,
        hybrid=False,
    )
    dense_type_bd = compute_breakdown_by_category(dense_items, "answer_type")
    dense_comp_bd = compute_breakdown_by_category(dense_items, "ticker")
    log_run_to_mlflow(
        run_name="N2-Dense-Only",
        metrics=dense_metrics,
        params={"retrieval_mode": "dense", "top_k": 5, "hybrid": False, "rerank": False, "total_queries": n_queries},
        breakdown_by_type=dense_type_bd,
        breakdown_by_company=dense_comp_bd,
    )

    # 6. Run Domain-Adapted Hybrid Pipeline (Dense + BM25 RRF)
    hybrid_metrics, hybrid_items = run_domain_adapted_benchmark(
        test_suite=test_suite,
        retriever=retriever,
        top_k=5,
        hybrid=True,
    )
    hybrid_type_bd = compute_breakdown_by_category(hybrid_items, "answer_type")
    hybrid_comp_bd = compute_breakdown_by_category(hybrid_items, "ticker")
    log_run_to_mlflow(
        run_name="N3-Hybrid-Dense-BM25-RRF",
        metrics=hybrid_metrics,
        params={"retrieval_mode": "hybrid", "top_k": 5, "hybrid": True, "rerank": False, "total_queries": n_queries},
        breakdown_by_type=hybrid_type_bd,
        breakdown_by_company=hybrid_comp_bd,
    )

    # 7. Run Domain-Adapted Hybrid Pipeline + Rerank
    print("\nConnecting to Domain-Adapted Retriever (Qdrant & BM25) + Rerank...")
    hybrid_rerank_metrics, hybrid_rerank_items = run_domain_adapted_benchmark(
        test_suite=test_suite,
        retriever=retriever,
        top_k=5,
        hybrid=True,
        rerank=True,
    )
    rerank_type_bd = compute_breakdown_by_category(hybrid_rerank_items, "answer_type")
    rerank_comp_bd = compute_breakdown_by_category(hybrid_rerank_items, "ticker")
    log_run_to_mlflow(
        run_name="N4-Hybrid-BM25-Rerank",
        metrics=hybrid_rerank_metrics,
        params={"retrieval_mode": "hybrid+rerank", "top_k": 5, "hybrid": True, "rerank": True, "total_queries": n_queries},
        breakdown_by_type=rerank_type_bd,
        breakdown_by_company=rerank_comp_bd,
    )

    # 8. Report and Save
    print_and_save_full_report(
        naive_metrics=naive_metrics,
        dense_metrics=dense_metrics,
        hybrid_metrics=hybrid_metrics,
        naive_items=naive_items,
        dense_items=dense_items,
        hybrid_items=hybrid_items,
        hybrid_rerank_metrics=hybrid_rerank_metrics,
        hybrid_rerank_items=hybrid_rerank_items,
        n_queries=n_queries,
    )

    if MLFLOW_AVAILABLE:
        print(f"\n[MLflow] All runs logged. View dashboard with:")
        print(f"         mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
        print(f"         Then open: http://localhost:5000")

    print(f"\nTask N-3/N-4 completed in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
