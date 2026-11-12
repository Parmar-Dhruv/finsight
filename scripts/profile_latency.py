"""
================================================================================
FinSight - Task J-6: CPU Latency Profiling & SLA Verification
================================================================================

Measures end-to-end CPU latency (retrieval + generation) across pipeline modes:
  - Dense-Only + Generation
  - Hybrid (Dense + BM25 RRF) + Generation
  - Hybrid + Cross-Encoder Rerank + Generation

Verifies Objective O4 (< 5,000 ms p95 SLA) and writes results to
evaluation/results/j6_latency_profile.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from generation.inference import FinSightGenerator
from retrieval.search import FinSightRetriever

SAMPLE_QUERIES = [
    ("What was Apple's total net revenue for fiscal year 2024?", "AAPL", 2024),
    ("What was Microsoft's operating income in FY2024?", "MSFT", 2024),
    ("How much revenue did AWS generate for Amazon in 2024?", "AMZN", 2024),
    ("What were Alphabet's Google Cloud revenues in FY2024?", "GOOGL", 2024),
    ("What was Meta's Family of Apps operating income in 2024?", "META", 2024),
]

OUT_PATH = BASE_DIR / "evaluation" / "results" / "j6_latency_profile.json"


def profile_mode(
    retriever: FinSightRetriever,
    generator: FinSightGenerator,
    hybrid: bool,
    rerank: bool,
) -> Dict[str, Any]:
    retrieval_ms: List[float] = []
    generation_ms: List[float] = []
    total_ms: List[float] = []

    for query, ticker, year in SAMPLE_QUERIES:
        t0 = time.perf_counter()
        chunks = retriever.retrieve(
            query=query,
            k=5,
            ticker=ticker,
            fiscal_year=year,
            hybrid=hybrid,
            rerank=rerank,
        )
        t1 = time.perf_counter()
        _ = generator.generate(question=query, context_chunks=chunks)
        t2 = time.perf_counter()

        r_ms = (t1 - t0) * 1000.0
        g_ms = (t2 - t1) * 1000.0
        tot = (t2 - t0) * 1000.0
        retrieval_ms.append(r_ms)
        generation_ms.append(g_ms)
        total_ms.append(tot)

    return {
        "samples": len(SAMPLE_QUERIES),
        "mean_ms": round(float(np.mean(total_ms)), 2),
        "p50_ms": round(float(np.percentile(total_ms, 50)), 2),
        "p95_ms": round(float(np.percentile(total_ms, 95)), 2),
        "retrieval_mean_ms": round(float(np.mean(retrieval_ms)), 2),
        "generation_mean_ms": round(float(np.mean(generation_ms)), 2),
        "meets_5s_sla": bool(float(np.percentile(total_ms, 95)) < 5000.0),
    }


def main() -> None:
    retriever = FinSightRetriever.get_instance()
    generator = FinSightGenerator.get_instance()

    # Warm up models once before timing
    _ = retriever.retrieve("Apple total revenue 2024", k=3, ticker="AAPL", hybrid=True, rerank=True)

    modes = {
        "dense_only": profile_mode(retriever, generator, hybrid=False, rerank=False),
        "hybrid_bm25_rrf": profile_mode(retriever, generator, hybrid=True, rerank=False),
        "hybrid_bm25_rerank": profile_mode(retriever, generator, hybrid=True, rerank=True),
    }

    payload = {
        "task": "J-6",
        "generator_backend": generator.backend,
        "sla_target_p95_ms": 5000.0,
        "modes": modes,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 82)
    print("  FINSIGHT TASK J-6: CPU LATENCY PROFILE (5-QUERY SAMPLE)")
    print("=" * 82)
    print(f"  {'Mode':<24} {'Mean (ms)':>11} {'p50 (ms)':>11} {'p95 (ms)':>11} {'<5s SLA':>10}")
    print("-" * 82)
    for name, s in modes.items():
        print(
            f"  {name:<24} {s['mean_ms']:11.1f} {s['p50_ms']:11.1f} "
            f"{s['p95_ms']:11.1f} {str(s['meets_5s_sla']):>10}"
        )
    print("=" * 82)
    print(f"[OK] Saved CPU latency profile to {OUT_PATH}")


if __name__ == "__main__":
    main()
