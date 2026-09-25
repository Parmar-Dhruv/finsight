"""
================================================================================
FinSight - N-1: Naive Baseline Retrieval Pipeline
================================================================================

Task N-1 (finsight-task-breakdown.md):
  "Build a naive baseline pipeline - fixed 512-token chunks, no
   structure-awareness, no metadata filtering, same embedding model.
   Run it through the existing evaluate_retrieval_quality() function.
   Output: one table with Hit Rate@1/3/5 and MRR for the naive pipeline,
   to sit next to the existing domain-adapted numbers."

Design: what makes this "naive"
---------------------------------
Dimension           | Domain-Adapted Pipeline              | Naive Baseline (THIS FILE)
--------------------|--------------------------------------|---------------------------------
Chunking            | Recursive, structure-aware,          | Fixed sliding window,
                    | paragraph/table-boundary-respecting, | 2048 chars (~512 tokens),
                    | 1500-char target (~375 tokens)       | NO overlap, NO structure sense
Section handling    | Per-section chunking with provenance | Full filing concatenated; section
                    | metadata preserved                   | label guessed from header text
Vector index        | Qdrant HNSW (local embedded)         | Flat NumPy dot-product (brute)
Metadata filtering  | ticker / section / year filters      | NONE - global search only
Embedding model     | BAAI/bge-small-en-v1.5               | BAAI/bge-small-en-v1.5 (SAME)

The embedding model is kept identical so the only variable is chunking strategy
and the absence of metadata filtering.

Usage
-----
    python eval/naive_baseline.py

Output
------
Side-by-side table:  Naive Baseline  vs  Domain-Adapted  (Hit Rate@1/3/5, MRR)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Project root on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from retrieval.embed import EmbeddingEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROCESSED_DIR = BASE_DIR / "data" / "processed"

# 512 tokens x 4 chars/token = 2048 chars per chunk (fixed, no overlap)
NAIVE_CHUNK_CHARS: int = 2048

# Crude section-label guesser: scan for the FIRST occurrence of a known header.
# Order matters - check item_8 before item_1 to avoid false positives.
SECTION_HEADER_PATTERNS: List[Tuple[str, str]] = [
    ("item_8",  "Item 8"),
    ("item_7",  "Item 7"),
    ("item_1a", "Item 1A"),
    ("item_1",  "Item 1"),
]


# ---------------------------------------------------------------------------
# Part 1: Naive Fixed-Window Chunker
# ---------------------------------------------------------------------------

def naive_fixed_chunk(text: str, chunk_chars: int = NAIVE_CHUNK_CHARS) -> List[str]:
    """
    Splits text into non-overlapping fixed-size windows of chunk_chars characters.
    Deliberately ignores paragraph / sentence / table boundaries.
    Zero overlap between adjacent chunks - classic "naive" behaviour.
    """
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        piece = text[start : start + chunk_chars].strip()
        if piece:
            chunks.append(piece)
        start += chunk_chars
    return chunks


def _guess_section(chunk_text: str) -> str:
    """
    Naively guesses the 10-K section label by scanning for the first known
    header string in the chunk.  Returns 'unknown' when none matches.
    """
    for section_key, header in SECTION_HEADER_PATTERNS:
        if header.lower() in chunk_text.lower():
            return section_key
    return "unknown"


def build_naive_corpus(
    processed_dir: Path = PROCESSED_DIR,
    chunk_chars: int = NAIVE_CHUNK_CHARS,
) -> List[Dict[str, Any]]:
    """
    Loads every *_parsed.json filing, concatenates all section texts into one
    flat blob (ignoring section boundaries), then applies fixed-window chunking.

    Returns a list of minimal dicts:
        text | ticker | fiscal_year | section | chunk_index
    """
    parsed_files = sorted(processed_dir.glob("*_parsed.json"))
    if not parsed_files:
        raise FileNotFoundError(
            f"No *_parsed.json files in {processed_dir}. Run retrieval/parse.py first."
        )

    print(f"\n[Naive Baseline] Loading {len(parsed_files)} parsed filing(s)...")
    corpus: List[Dict[str, Any]] = []

    for fp in parsed_files:
        with open(fp, "r", encoding="utf-8") as f:
            doc = json.load(f)

        ticker: str = doc["ticker"]
        fiscal_year: int = int(doc["fiscal_year"])
        sections: Dict[str, str] = doc.get("sections", {})

        # Concatenate ALL section texts - a naive pipeline would not separate sections.
        full_text = "\n".join(v for v in sections.values() if v)
        raw_chunks = naive_fixed_chunk(full_text, chunk_chars)

        for idx, text in enumerate(raw_chunks):
            corpus.append(
                {
                    "text": text,
                    "ticker": ticker,
                    "fiscal_year": fiscal_year,
                    "section": _guess_section(text),
                    "chunk_index": idx,
                }
            )

        print(f"  [{ticker} FY{fiscal_year}]  {fp.name:30s}  ->  {len(raw_chunks):4d} chunks")

    total = len(corpus)
    print(f"\n[Naive Baseline] Corpus: {total:,} fixed-window chunks ({chunk_chars} chars, 0 overlap)")
    return corpus


# ---------------------------------------------------------------------------
# Part 2: In-Memory Embedding Index
# ---------------------------------------------------------------------------

def build_naive_index(
    corpus: List[Dict[str, Any]],
    engine: EmbeddingEngine,
) -> np.ndarray:
    """
    Encodes every chunk with BAAI/bge-small-en-v1.5 (same model as domain pipeline).
    Returns an (N, 384) float32 matrix for brute-force cosine search.
    No HNSW, no Qdrant - intentionally flat.
    """
    print("\n[Naive Baseline] Encoding chunks with BAAI/bge-small-en-v1.5 ...")
    texts = [c["text"] for c in corpus]
    embeddings = engine.encode_passages(texts, batch_size=64, show_progress_bar=True)
    print(f"[Naive Baseline] Done. Embedding matrix shape: {embeddings.shape}")
    return embeddings


def naive_search(
    query: str,
    engine: EmbeddingEngine,
    embeddings: np.ndarray,
    corpus: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Brute-force dot-product cosine search with NO metadata filter of any kind.
    Result dicts mirror the shape of retrieval.index.search_index() output.
    """
    query_vec = engine.encode_queries(query, show_progress_bar=False)   # (1, 384)
    scores = np.dot(embeddings, query_vec.T).flatten()                   # (N,)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        chunk = corpus[idx]
        results.append(
            {
                "score": float(round(scores[idx], 4)),
                "chunk_id": f"{chunk['ticker']}_{chunk['fiscal_year']}_naive_{chunk['chunk_index']:04d}",
                "ticker": chunk["ticker"],
                "fiscal_year": chunk["fiscal_year"],
                "section": chunk["section"],
                "text": chunk["text"],
                "token_count": len(chunk["text"]) // 4,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Part 3: Evaluation - same 8-query test suite as search.py
# ---------------------------------------------------------------------------

# Mirrors evaluate_retrieval_quality() default test_suite exactly
EVAL_TEST_SUITE: List[Dict[str, Any]] = [
    {
        "query": "What were Apple's annual net sales and iPhone product revenue in 2024?",
        "expected_ticker": "AAPL",
        "expected_section": "item_7",
    },
    {
        "query": "What are the primary cloud computing and cybersecurity risk factors for Microsoft?",
        "expected_ticker": "MSFT",
        "expected_section": "item_1a",
    },
    {
        "query": "What drove Microsoft's Azure cloud revenue expansion and server growth?",
        "expected_ticker": "MSFT",
        "expected_section": "item_7",
    },
    {
        "query": "What are the key competition and antitrust legal risks facing Alphabet Google?",
        "expected_ticker": "GOOGL",
        "expected_section": "item_1a",
    },
    {
        "query": "What was Amazon's total net sales breakdown across retail, AWS, and advertising?",
        "expected_ticker": "AMZN",
        "expected_section": "item_7",
    },
    {
        "query": "How does Meta describe the risks of AI model safety and regulatory compliance?",
        "expected_ticker": "META",
        "expected_section": "item_1a",
    },
    {
        "query": "What was Apple's total research and development expense and operating budget?",
        "expected_ticker": "AAPL",
        "expected_section": "item_7",
    },
    {
        "query": "What are Amazon's primary logistics, fulfillment, and supply chain operational risks?",
        "expected_ticker": "AMZN",
        "expected_section": "item_1a",
    },
]


def evaluate_naive_pipeline(
    engine: EmbeddingEngine,
    embeddings: np.ndarray,
    corpus: List[Dict[str, Any]],
    test_suite: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Runs the same IR evaluation loop as evaluate_retrieval_quality() in
    retrieval/search.py but wired to the naive in-memory search instead of Qdrant.

    A result is a HIT only if BOTH ticker AND section label match - same criterion
    as the domain-adapted evaluator.  Because many naive chunks land in 'unknown'
    section, they can never satisfy the section criterion even when the text is
    semantically relevant.  This is an honest penalty for structural blindness.
    """
    if test_suite is None:
        test_suite = EVAL_TEST_SUITE

    total_queries = len(test_suite)
    hits_at_1 = hits_at_3 = hits_at_5 = 0
    reciprocal_ranks: List[float] = []

    print("\n" + "=" * 75)
    print(f"NAIVE BASELINE RETRIEVAL EVALUATION  (Benchmark: {total_queries} queries)")
    print("=" * 75)

    for idx, test in enumerate(test_suite, 1):
        q            = test["query"]
        exp_ticker   = test["expected_ticker"]
        exp_section  = test["expected_section"]

        results = naive_search(q, engine, embeddings, corpus, top_k=top_k)

        first_match_rank: Optional[int] = None
        for rank, r in enumerate(results, 1):
            if r["ticker"] == exp_ticker and r["section"] == exp_section:
                first_match_rank = rank
                break

        if first_match_rank == 1:
            hits_at_1 += 1
        if first_match_rank and first_match_rank <= 3:
            hits_at_3 += 1
        if first_match_rank and first_match_rank <= 5:
            hits_at_5 += 1

        rr = (1.0 / first_match_rank) if first_match_rank else 0.0
        reciprocal_ranks.append(rr)

        status_tag = f"[MATCH Rank {first_match_rank}]" if first_match_rank else "[MISS  top-5]"
        print(f"[{idx:2d}/{total_queries}] {status_tag:<18} | Target: [{exp_ticker} {exp_section}]")
        print(f'     Query: "{q[:65]}..."')

    hit_rate_1 = (hits_at_1 / total_queries) * 100
    hit_rate_3 = (hits_at_3 / total_queries) * 100
    hit_rate_5 = (hits_at_5 / total_queries) * 100
    mrr        = sum(reciprocal_ranks) / total_queries

    print("-" * 75)
    print("NAIVE BASELINE SCOREBOARD")
    print("-" * 75)
    print(f"  Hit Rate @ 1:               {hit_rate_1:5.1f}%  ({hits_at_1}/{total_queries})")
    print(f"  Hit Rate @ 3:               {hit_rate_3:5.1f}%  ({hits_at_3}/{total_queries})")
    print(f"  Hit Rate @ 5:               {hit_rate_5:5.1f}%  ({hits_at_5}/{total_queries})")
    print(f"  MRR (Mean Reciprocal Rank):  {mrr:.4f}")
    print("=" * 75)

    return {
        "hit_rate_at_1": hit_rate_1,
        "hit_rate_at_3": hit_rate_3,
        "hit_rate_at_5": hit_rate_5,
        "mrr": mrr,
    }


def evaluate_domain_adapted_pipeline(
    test_suite: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Calls the existing evaluate_retrieval_quality() from retrieval/search.py
    using the same 8-query test suite so both results are collected in one run.
    """
    from retrieval.search import evaluate_retrieval_quality, FinSightRetriever

    metrics = evaluate_retrieval_quality(test_suite=test_suite, top_k=top_k)

    # Clean up singleton so re-runs don't reuse a stale connection
    try:
        FinSightRetriever.get_instance().close()
        FinSightRetriever._instance = None
    except Exception:
        pass

    return metrics


# ---------------------------------------------------------------------------
# Part 4: Comparison Table (N-1 deliverable)
# ---------------------------------------------------------------------------

def print_comparison_table(
    naive: Dict[str, float],
    adapted: Dict[str, float],
    n_queries: int = 8,
) -> None:
    """
    Prints the side-by-side N-1 result table requested in the task breakdown.
    Delta columns show absolute percentage-point improvement of domain-adapted over naive.
    """
    def pct(v: float) -> str:
        return f"{v:5.1f}%"

    def delta(dom: float, nav: float) -> str:
        d = dom - nav
        return f"({'+'if d>=0 else ''}{d:.1f} pp)"

    W = 80
    print("\n")
    print("=" * W)
    print("  FINSIGHT  -  N-1 RETRIEVAL QUALITY COMPARISON")
    print(f"  Benchmark: {n_queries} queries  |  top-5  |  same embedding model in both pipelines")
    print("=" * W)
    print(
        f"  {'Metric':<22}"
        f"{'Naive Baseline':>16}"
        f"{'Domain-Adapted':>18}"
        f"{'Delta':>18}"
    )
    print("  " + "-" * 74)

    rows = [
        ("Hit Rate @ 1",
            pct(naive["hit_rate_at_1"]),
            pct(adapted["hit_rate_at_1"]),
            delta(adapted["hit_rate_at_1"], naive["hit_rate_at_1"])),
        ("Hit Rate @ 3",
            pct(naive["hit_rate_at_3"]),
            pct(adapted["hit_rate_at_3"]),
            delta(adapted["hit_rate_at_3"], naive["hit_rate_at_3"])),
        ("Hit Rate @ 5",
            pct(naive["hit_rate_at_5"]),
            pct(adapted["hit_rate_at_5"]),
            delta(adapted["hit_rate_at_5"], naive["hit_rate_at_5"])),
        ("MRR",
            f"{naive['mrr']:.4f}",
            f"{adapted['mrr']:.4f}",
            delta(adapted["mrr"] * 100, naive["mrr"] * 100).replace("pp", "MRR-pp")),
    ]

    for label, naive_val, domain_val, d in rows:
        print(
            f"  {label:<22}"
            f"{naive_val:>16}"
            f"{domain_val:>18}"
            f"{d:>18}"
        )

    print("=" * W)
    print("  Naive baseline  : fixed 2048-char windows (~512 tok), no structural awareness,")
    print("                    no metadata filter, brute-force cosine similarity (NumPy).")
    print("  Domain-adapted  : recursive structure-aware chunking (~375-tok target),")
    print("                    Qdrant HNSW index. Same unconstrained global search here.")
    print("  Embedding model : BAAI/bge-small-en-v1.5  (identical in both pipelines).")
    print("  Delta           : percentage-point difference (domain-adapted minus naive).")
    print("=" * W)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    print("=" * 75)
    print("FinSight  N-1  -  Naive Baseline vs Domain-Adapted Retrieval")
    print("=" * 75)

    # Step 1: Build naive fixed-window corpus from parsed filings
    corpus = build_naive_corpus(PROCESSED_DIR, chunk_chars=NAIVE_CHUNK_CHARS)

    # Step 2: Embed corpus with the SAME model as domain-adapted pipeline
    engine = EmbeddingEngine()           # BAAI/bge-small-en-v1.5
    embeddings = build_naive_index(corpus, engine)

    # Step 3: Run naive evaluation (mirrors evaluate_retrieval_quality)
    naive_metrics = evaluate_naive_pipeline(
        engine=engine,
        embeddings=embeddings,
        corpus=corpus,
        test_suite=EVAL_TEST_SUITE,
        top_k=5,
    )

    # Step 4: Run domain-adapted evaluation via existing evaluate_retrieval_quality()
    print("\n[Domain-Adapted] Running existing evaluate_retrieval_quality()...")
    adapted_metrics = evaluate_domain_adapted_pipeline(
        test_suite=EVAL_TEST_SUITE,
        top_k=5,
    )

    # Step 5: Print comparison table (the N-1 deliverable)
    print_comparison_table(naive_metrics, adapted_metrics, n_queries=len(EVAL_TEST_SUITE))

    print(f"\n[OK] N-1 completed in {time.time() - t_start:.1f}s")
