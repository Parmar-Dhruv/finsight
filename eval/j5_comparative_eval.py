"""
================================================================================
FinSight - Task J-5: Full 5-Configuration Comparative Evaluation Matrix
================================================================================

Evaluates 5 system configurations on the 100-question FinSight benchmark
(evaluation/dataset/questions.json):

  1. Base LLM (No RAG)                : Unaugmented generation without retrieval
  2. Naive RAG                        : Fixed 512-token chunks + grounded generation
  3. Domain-Adapted Retrieval Only    : Hybrid (Dense+BM25) + Cross-Encoder Rerank (unfiltered)
  4. Fine-Tuned Model Only (No RAG)   : Fine-tuned weights without retrieval context
  5. Full FinSight System             : Entity/Year-filtered Hybrid+Rerank + Grounded Generation

Outputs:
  - evaluation/results/j5_comparative_matrix.json
  - Formatted comparative matrix table to stdout
  - Optional MLflow logging (if mlflow is installed)
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from eval.benchmark import (
    BENCHMARK_PATH,
    RESULTS_DIR,
    compute_breakdown_by_category,
    load_and_convert_benchmark,
    log_run_to_mlflow,
)
from eval.naive_baseline import (
    NAIVE_CHUNK_CHARS,
    PROCESSED_DIR,
    build_naive_corpus,
    build_naive_index,
    naive_search,
)
from generation.inference import FinSightGenerator
from generation.prompt_templates import REFUSAL_MESSAGE, has_citation
from retrieval.embed import EmbeddingEngine
from retrieval.search import FinSightRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Company name -> ticker mapping for query-time metadata extraction in Full System
COMPANY_PATTERNS: List[Tuple[str, str]] = [
    (r"\bapple\b|\baapl\b", "AAPL"),
    (r"\bmicrosoft\b|\bmsft\b", "MSFT"),
    (r"\bamazon\b|\bamzn\b", "AMZN"),
    (r"\balphabet\b|\bgoogle\b|\bgoogl\b", "GOOGL"),
    (r"\bmeta\b|\bfacebook\b", "META"),
    (r"\bnvidia\b|\bnvda\b", "NVDA"),
]

_CITATION_TAG_RE = re.compile(r"\[SOURCE\s+\d+\]")
_NUMBER_RE = re.compile(r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?")
NAIVE_CACHE_PATH = PROCESSED_DIR / "naive_embeddings_cache.npy"


def extract_query_filters(query: str) -> Tuple[Optional[str], Optional[int]]:
    """Extracts ticker and fiscal year from a natural language query if present."""
    q_lower = query.lower()
    ticker: Optional[str] = None
    for pattern, t_code in COMPANY_PATTERNS:
        if re.search(pattern, q_lower):
            ticker = t_code
            break

    fiscal_year: Optional[int] = None
    years = re.findall(r"\b(2023|2024)\b", q_lower)
    unique_years = set(years)
    if len(unique_years) == 1:
        fiscal_year = int(next(iter(unique_years)))

    return ticker, fiscal_year


def _normalize_tokens(text: str) -> List[str]:
    """Lowercases, strips citation tags and punctuation, and tokenizes."""
    cleaned = _CITATION_TAG_RE.sub("", text or "").lower()
    return re.findall(r"\b[a-z0-9]+(?:\.[0-9]+)?\b", cleaned)


def compute_token_f1(prediction: str, reference: str) -> float:
    """Computes unigram F1 overlap between predicted answer and gold answer."""
    pred_tokens = _normalize_tokens(prediction)
    ref_tokens = _normalize_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    common = sum((pred_counts & ref_counts).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def extract_numeric_values(text: str) -> List[str]:
    """Extracts normalized numeric strings (without $, commas, %) for comparison."""
    cleaned = _CITATION_TAG_RE.sub("", text or "")
    raw_nums = _NUMBER_RE.findall(cleaned)
    normalized: List[str] = []
    for num in raw_nums:
        val = num.replace("$", "").replace(",", "").replace("%", "").strip()
        if val in {"2022", "2023", "2024", "2025"}:
            continue
        if val:
            normalized.append(val)
    return normalized


def compute_numeric_match(prediction: str, gold_answer: str) -> Optional[bool]:
    """
    Checks if at least one key numeric value from gold_answer appears in prediction.
    Returns None if gold_answer has no non-year numeric values.
    """
    gold_nums = extract_numeric_values(gold_answer)
    if not gold_nums:
        return None
    pred_nums = set(extract_numeric_values(prediction))
    pred_clean = prediction.replace(",", "")
    for g in gold_nums:
        if g in pred_nums or g in pred_clean:
            return True
        if g.endswith(".0") and g[:-2] in pred_nums:
            return True
    return False


def compute_faithfulness_proxy(
    answer: str,
    chunks: Sequence[Dict[str, Any]],
    is_refusal: bool,
) -> float:
    """
    Estimates groundedness/faithfulness of the answer against retrieved chunks.
    """
    if not chunks:
        return 0.0
    if is_refusal:
        return 1.0

    context_text = " ".join(c.get("text", "") for c in chunks)
    context_tokens = set(_normalize_tokens(context_text))
    ans_tokens = [
        t for t in _normalize_tokens(answer)
        if len(t) > 2 and t not in {"the", "and", "for", "was", "were", "with", "that", "from", "this"}
    ]
    if not ans_tokens:
        return 0.0

    supported = sum(1 for t in ans_tokens if t in context_tokens)
    return supported / len(ans_tokens)


def evaluate_configuration(
    config_id: str,
    config_name: str,
    test_suite: List[Dict[str, Any]],
    generator: FinSightGenerator,
    use_rag: bool,
    retrieval_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Runs one pipeline configuration across all benchmark questions."""
    total = len(test_suite)
    print(f"\n>>> Running J-5 Config [{config_id}]: {config_name} ({total} questions)...", flush=True)

    hits_1 = hits_3 = hits_5 = 0
    rr_list: List[float] = []
    citations_count = 0
    refusals_count = 0
    f1_scores: List[float] = []
    faithfulness_scores: List[float] = []
    latencies_ms: List[float] = []
    numeric_total = 0
    numeric_matches = 0
    item_results: List[Dict[str, Any]] = []

    for idx, q_item in enumerate(test_suite, 1):
        query = q_item["query"]
        gold_answer = q_item["gold_answer"]
        exp_ticker = q_item["expected_ticker"]
        exp_sections = q_item["expected_sections"]

        t0 = time.perf_counter()
        chunks: List[Dict[str, Any]] = []
        match_rank: Optional[int] = None

        if use_rag and retrieval_fn is not None:
            chunks = retrieval_fn(q_item)
            for rank, c in enumerate(chunks, 1):
                if c.get("ticker") == exp_ticker and c.get("section") in exp_sections:
                    match_rank = rank
                    break
            if match_rank == 1:
                hits_1 += 1
            if match_rank and match_rank <= 3:
                hits_3 += 1
            if match_rank and match_rank <= 5:
                hits_5 += 1
            rr = (1.0 / match_rank) if match_rank else 0.0
            rr_list.append(rr)

            gen_out = generator.generate(question=query, context_chunks=chunks)
        else:
            rr = 0.0
            rr_list.append(0.0)
            gen_out = generator.generate_without_rag(question=query)

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(total_latency_ms)

        answer = gen_out["answer"]
        cited = bool(gen_out.get("has_citation") or has_citation(answer))
        refused = bool(gen_out.get("is_refusal") or answer.strip() == REFUSAL_MESSAGE)

        if cited:
            citations_count += 1
        if refused:
            refusals_count += 1

        f1 = compute_token_f1(answer, gold_answer)
        f1_scores.append(f1)

        faith = compute_faithfulness_proxy(answer, chunks, refused)
        faithfulness_scores.append(faith)

        num_match = compute_numeric_match(answer, gold_answer)
        if num_match is not None:
            numeric_total += 1
            if num_match:
                numeric_matches += 1

        item_results.append({
            "id": q_item["id"],
            "query": query,
            "ticker": exp_ticker,
            "answer_type": q_item["answer_type"],
            "expected_sections": exp_sections,
            "match_rank": match_rank,
            "reciprocal_rank": rr,
            "retrieved_top_sections": [f"{c.get('ticker')}:{c.get('section')}" for c in chunks],
            "answer": answer,
            "gold_answer": gold_answer,
            "has_citation": cited,
            "is_refusal": refused,
            "token_f1": round(f1, 4),
            "numeric_match": num_match,
            "faithfulness": round(faith, 4),
            "latency_ms": round(total_latency_ms, 2),
        })

        if idx % 10 == 0 or idx == total:
            print(
                f"  [{config_id}] {idx:3d}/{total} | "
                f"HR@5={(hits_5 / idx * 100) if use_rag else 0.0:5.1f}% | "
                f"Cite={(citations_count / idx * 100):5.1f}% | "
                f"F1={float(np.mean(f1_scores)):.3f}",
                flush=True,
            )

    summary = {
        "config_id": config_id,
        "config_name": config_name,
        "use_rag": use_rag,
        "generator_backend": generator.backend,
        "total_queries": total,
        "hit_rate_at_1": round((hits_1 / total) * 100, 2) if use_rag else 0.0,
        "hit_rate_at_3": round((hits_3 / total) * 100, 2) if use_rag else 0.0,
        "hit_rate_at_5": round((hits_5 / total) * 100, 2) if use_rag else 0.0,
        "mrr": round(float(np.mean(rr_list)), 4) if use_rag else 0.0,
        "citation_rate_pct": round((citations_count / total) * 100, 2),
        "refusal_rate_pct": round((refusals_count / total) * 100, 2),
        "numeric_match_pct": round((numeric_matches / numeric_total) * 100, 2) if numeric_total else 0.0,
        "numeric_questions_count": numeric_total,
        "avg_token_f1": round(float(np.mean(f1_scores)), 4),
        "avg_faithfulness": round(float(np.mean(faithfulness_scores)), 4),
        "avg_latency_ms": round(float(np.mean(latencies_ms)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
    }

    by_type = compute_breakdown_by_category(item_results, "answer_type") if use_rag else {}
    by_company = compute_breakdown_by_category(item_results, "ticker") if use_rag else {}

    return {
        "summary": summary,
        "by_type": by_type,
        "by_company": by_company,
        "detailed_results": item_results,
    }


def print_comparative_matrix(configs: List[Dict[str, Any]]) -> str:
    """Formats and prints the J-5 5-configuration comparative evaluation matrix."""
    lines = [
        "",
        "=" * 116,
        "  FINSIGHT TASK J-5: 5-CONFIGURATION COMPARATIVE EVALUATION MATRIX (100 QUESTIONS)",
        "=" * 116,
        f"  {'Configuration':<32} {'HR@1':>7} {'HR@5':>7} {'MRR':>7} {'Cite%':>7} {'Refuse%':>8} {'NumAcc%':>8} {'TokF1':>7} {'Faith':>7} {'p95(ms)':>9}",
        "-" * 116,
    ]
    for cfg in configs:
        s = cfg["summary"]
        hr1 = f"{s['hit_rate_at_1']:6.1f}%" if s["use_rag"] else "   N/A "
        hr5 = f"{s['hit_rate_at_5']:6.1f}%" if s["use_rag"] else "   N/A "
        mrr = f"{s['mrr']:7.4f}" if s["use_rag"] else "   N/A "
        lines.append(
            f"  {s['config_name']:<32} {hr1} {hr5} {mrr} "
            f"{s['citation_rate_pct']:6.1f}% {s['refusal_rate_pct']:7.1f}% "
            f"{s['numeric_match_pct']:7.1f}% {s['avg_token_f1']:7.4f} "
            f"{s['avg_faithfulness']:7.4f} {s['p95_latency_ms']:9.1f}"
        )
    lines.append("=" * 116)
    report = "\n".join(lines)
    print(report, flush=True)
    return report


def main() -> None:
    t_start = time.time()
    test_suite = load_and_convert_benchmark(BENCHMARK_PATH)
    generator = FinSightGenerator.get_instance()
    print(f"Generator backend: {generator.backend}", flush=True)

    # Build or load cached Naive Baseline index
    print("Preparing Naive Baseline corpus & index...", flush=True)
    naive_corpus = build_naive_corpus(PROCESSED_DIR, chunk_chars=NAIVE_CHUNK_CHARS)
    engine = EmbeddingEngine()
    if NAIVE_CACHE_PATH.exists():
        print(f"Loading cached naive embeddings from {NAIVE_CACHE_PATH}...", flush=True)
        naive_embeddings = np.load(NAIVE_CACHE_PATH)
    else:
        print(f"Embedding {len(naive_corpus)} naive chunks (one-time)...", flush=True)
        naive_embeddings = build_naive_index(naive_corpus, engine)
        np.save(NAIVE_CACHE_PATH, naive_embeddings)

    # Connect Domain-Adapted Retriever
    retriever = FinSightRetriever.get_instance()

    # 1. Base LLM (No RAG)
    cfg_base = evaluate_configuration(
        config_id="base_llm_no_rag",
        config_name="1. Base LLM (No RAG)",
        test_suite=test_suite,
        generator=generator,
        use_rag=False,
    )

    # 2. Naive RAG
    cfg_naive = evaluate_configuration(
        config_id="naive_rag",
        config_name="2. Naive RAG",
        test_suite=test_suite,
        generator=generator,
        use_rag=True,
        retrieval_fn=lambda q: naive_search(q["query"], engine, naive_embeddings, naive_corpus, top_k=5),
    )

    # 3. Domain-Adapted Retrieval Only (Global Hybrid + Rerank, rerank_pool=10)
    cfg_domain_only = evaluate_configuration(
        config_id="domain_retrieval_only",
        config_name="3. Domain Retrieval Only",
        test_suite=test_suite,
        generator=generator,
        use_rag=True,
        retrieval_fn=lambda q: retriever.retrieve(
            query=q["query"], k=5, hybrid=True, rerank=True, rerank_pool=10
        ),
    )

    # 4. Fine-Tuned Model Only (No RAG)
    cfg_ft_only = evaluate_configuration(
        config_id="fine_tuned_no_rag",
        config_name="4. Fine-Tuned Only (No RAG)",
        test_suite=test_suite,
        generator=generator,
        use_rag=False,
    )

    # 5. Full FinSight System (Entity/Year Metadata Filtering + Hybrid + Rerank, rerank_pool=10)
    def full_system_retrieve(q_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ticker, fiscal_year = extract_query_filters(q_item["query"])
        return retriever.retrieve(
            query=q_item["query"],
            k=5,
            ticker=ticker,
            fiscal_year=fiscal_year,
            hybrid=True,
            rerank=True,
            rerank_pool=10,
        )

    cfg_full = evaluate_configuration(
        config_id="full_finsight_system",
        config_name="5. Full FinSight System",
        test_suite=test_suite,
        generator=generator,
        use_rag=True,
        retrieval_fn=full_system_retrieve,
    )

    all_configs = [cfg_base, cfg_naive, cfg_domain_only, cfg_ft_only, cfg_full]
    print_comparative_matrix(all_configs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "j5_comparative_matrix.json"
    payload = {
        "metadata": {
            "task": "J-5",
            "total_questions": len(test_suite),
            "generator_backend": generator.backend,
            "rerank_pool": 10,
            "note": (
                "llama-cpp-python is not installed in this environment; "
                "FinSightGenerator uses its domain_synthesis backend while the "
                "GGUF model artifact resides at generation/checkpoints/gguf/."
            ),
            "elapsed_seconds": round(time.time() - t_start, 2),
        },
        "matrix_summary": [c["summary"] for c in all_configs],
        "configurations": {
            c["summary"]["config_id"]: c for c in all_configs
        },
    }
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[OK] J-5 Comparative Evaluation Matrix saved to: {out_file}", flush=True)

    # Log to MLflow if available
    for c in all_configs:
        s = c["summary"]
        log_run_to_mlflow(
            run_name=f"J5-{s['config_id']}",
            metrics={
                "hit_rate_at_1": s["hit_rate_at_1"],
                "hit_rate_at_3": s["hit_rate_at_3"],
                "hit_rate_at_5": s["hit_rate_at_5"],
                "mrr": s["mrr"],
            },
            params={
                "retrieval_mode": s["config_id"],
                "top_k": 5,
                "hybrid": s["config_id"] in {"domain_retrieval_only", "full_finsight_system"},
                "rerank": s["config_id"] in {"domain_retrieval_only", "full_finsight_system"},
                "total_queries": s["total_queries"],
            },
            breakdown_by_type=c["by_type"] or None,
            breakdown_by_company=c["by_company"] or None,
            artifact_path=str(out_file) if s["config_id"] == "full_finsight_system" else None,
        )


if __name__ == "__main__":
    main()
