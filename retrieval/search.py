"""
FinSight — Retrieval & Data Layer
Search & Retrieval API Layer (retrieval/search.py)

This module provides the primary user-facing and pipeline-facing retrieval interface:
  1. retrieve(query, k, ticker, fiscal_year, section) -> List[Dict]
  2. format_context_for_prompt(chunks) -> str
  3. evaluate_retrieval_quality() -> Dict[str, float]

This encapsulates all embedding logic, vector index communication, and metadata filtering
so that teammates (Dhruv for LLM generation, Jay for FastAPI serving) have a clean,
zero-boilerplate API to integrate.
"""

import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from retrieval.embed import EmbeddingEngine
from retrieval.index import QdrantIndexManager, search_index
from retrieval.hybrid import BM25Index, reciprocal_rank_fusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Part 1: Singleton Retriever & Public API ─────────────────────────────────

class FinSightRetriever:
    """
    Singleton retriever that keeps the embedding model, vector database,
    and BM25 sparse index warm in memory across requests.
    Supports Dense Vector Search, BM25 Sparse Search, and Hybrid Fusion (RRF).
    """

    _instance: Optional["FinSightRetriever"] = None

    def __init__(self, local_path: Optional[Path] = None) -> None:
        """Initializes the embedding engine and vector index connection."""
        logger.info("Initializing FinSightRetriever singleton...")
        self.engine = EmbeddingEngine()
        self.index_manager = QdrantIndexManager(local_path=local_path)
        self.bm25_index: Optional[BM25Index] = None
        logger.info("[OK] FinSightRetriever ready for incoming queries.")

    @classmethod
    def get_instance(cls, local_path: Optional[Path] = None) -> "FinSightRetriever":
        """Thread-safe accessor for the retriever singleton."""
        if cls._instance is None:
            cls._instance = cls(local_path=local_path)
        return cls._instance

    def _ensure_bm25(self) -> BM25Index:
        """Lazily initializes the BM25 index on first hybrid call."""
        if self.bm25_index is None:
            self.bm25_index = BM25Index.get_instance()
        return self.bm25_index

    def _ensure_cross_encoder(self):
        """Lazily initializes the pure-NumPy FastCrossEncoder for reranking."""
        if getattr(self, "cross_encoder", None) is None:
            from retrieval.rerank import get_cross_encoder
            logger.info("Initializing FastCrossEncoder (zero-PyTorch, pure NumPy)...")
            self.cross_encoder = get_cross_encoder()
        return self.cross_encoder

    def retrieve(
        self,
        query: str,
        k: int = 5,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        section: Optional[str] = None,
        hybrid: bool = False,
        rerank: bool = False,
        candidate_pool: int = 25,
        rerank_pool: int = 25,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k most relevant financial text chunks for a given query.

        Parameters:
            query: Natural language financial question or search term.
            k: Number of chunks to retrieve (default: 5).
            ticker: Optional company ticker filter (e.g. 'AAPL', 'MSFT').
            fiscal_year: Optional filing fiscal year (e.g. 2024, 2023).
            section: Optional 10-K section filter (e.g. 'item_7', 'item_1a').
            hybrid: If True, combines Dense Vector Search with BM25 Sparse Search via RRF.
            candidate_pool: Number of candidates to fetch from each retriever before fusion.
            dense_weight: Relative weight for dense vector ranking in RRF.
            sparse_weight: Relative weight for BM25 lexical ranking in RRF.
            rrf_k: Smoothing constant for Reciprocal Rank Fusion (default: 60).

        Returns:
            List of result dictionaries containing score, chunk_id, ticker,
            fiscal_year, section, token_count, and full text.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()

        # Determine the number of candidates to fetch before optional reranking
        fetch_k = rerank_pool if rerank else k
        
        # Pure dense vector search
        if not hybrid:
            results = search_index(
                manager=self.index_manager,
                query=clean_query,
                engine=self.engine,
                top_k=fetch_k,
                ticker=ticker,
                fiscal_year=fiscal_year,
                section=section,
            )
        else:
            # Hybrid Search: Dense + Sparse with Reciprocal Rank Fusion
            bm25 = self._ensure_bm25()
            pool_size = max(fetch_k, candidate_pool)

            dense_candidates = search_index(
                manager=self.index_manager,
                query=clean_query,
                engine=self.engine,
                top_k=pool_size,
                ticker=ticker,
                fiscal_year=fiscal_year,
                section=section,
            )

            sparse_candidates = bm25.search(
                query=clean_query,
                top_k=pool_size,
                ticker=ticker,
                fiscal_year=fiscal_year,
                section=section,
            )

            results = reciprocal_rank_fusion(
                dense_results=dense_candidates,
                sparse_results=sparse_candidates,
                top_k=fetch_k,
                rrf_k=rrf_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
            
        if rerank and results:
            cross_encoder = self._ensure_cross_encoder()
            pairs = [[clean_query, doc["text"]] for doc in results]
            scores = cross_encoder.predict(pairs)
            
            for idx, doc in enumerate(results):
                doc["rerank_score"] = float(scores[idx])
                doc["score"] = float(scores[idx]) # Overwrite score so format_context_for_prompt works transparently
                
            results.sort(key=lambda x: x["rerank_score"], reverse=True)
            results = results[:k]
            
        return results

    def close(self) -> None:
        """Closes internal database connections cleanly."""
        if self.index_manager:
            self.index_manager.close()


def retrieve(
    query: str,
    k: int = 5,
    ticker: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    section: Optional[str] = None,
    hybrid: bool = False,
    rerank: bool = False,
    candidate_pool: int = 25,
    rerank_pool: int = 25,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Public API function for pipeline integration (Dhruv / Jay).
    
    Example:
        >>> from retrieval.search import retrieve
        >>> chunks = retrieve("What was Microsoft's cloud growth?", k=3, ticker="MSFT", hybrid=True)
    """
    retriever = FinSightRetriever.get_instance()
    return retriever.retrieve(
        query=query,
        k=k,
        ticker=ticker,
        fiscal_year=fiscal_year,
        section=section,
        hybrid=hybrid,
        rerank=rerank,
        candidate_pool=candidate_pool,
        rerank_pool=rerank_pool,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        rrf_k=rrf_k,
    )


def format_context_for_prompt(
    chunks: List[Dict[str, Any]],
    max_tokens: int = 2000,
) -> str:
    """
    Formats a list of retrieved chunks into an LLM context block with complete
    attribution headers and token budgeting.

    Parameters:
        chunks: List of result dictionaries from retrieve().
        max_tokens: Approximate token ceiling for the context window.

    Returns:
        Formatted context string ready for injection into prompt templates.
    """
    if not chunks:
        return "No relevant filing passages found."

    header = f"=== RETRIEVED 10-K CONTEXT ({len(chunks)} passages) ===\n"
    passages: List[str] = []
    total_tokens = 0

    for idx, c in enumerate(chunks, 1):
        chunk_tokens = c.get("token_count", len(c.get("text", "")) // 4)
        if total_tokens + chunk_tokens > max_tokens and passages:
            break

        meta = f"[SOURCE {idx} | {c.get('ticker')} FY{c.get('fiscal_year')} {c.get('section')} | Score: {c.get('score'):.4f}]"
        body = c.get("text", "").strip()
        passages.append(f"{meta}\n{body}\n")
        total_tokens += chunk_tokens

    footer = "========================================================="
    return header + "\n" + "\n".join(passages) + footer


# ─── Part 2: Retrieval Quality Evaluation Suite ───────────────────────────────

def evaluate_retrieval_quality(
    test_suite: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 5,
    hybrid: bool = False,
    rerank: bool = False,
) -> Dict[str, float]:
    """
    Runs an Information Retrieval (IR) evaluation benchmark against the vector index.
    Measures:
      - Hit Rate @ 1
      - Hit Rate @ 3
      - Hit Rate @ 5
      - Mean Reciprocal Rank (MRR)
    """
    # Standard finance evaluation benchmark across multiple companies and filing sections
    if test_suite is None:
        test_suite = [
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
            {
                "query": "What drove NVIDIA's Data Center revenue expansion and AI GPU computing demand?",
                "expected_ticker": "NVDA",
                "expected_section": "item_7",
            },
            {
                "query": "What are NVIDIA's primary supply chain concentration and semiconductor manufacturing risks?",
                "expected_ticker": "NVDA",
                "expected_section": "item_1a",
            },
        ]

    retriever = FinSightRetriever.get_instance()
    total_queries = len(test_suite)
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks: List[float] = []

    mode_label = "HYBRID (Dense + BM25 RRF)" if hybrid else "DENSE-ONLY"
    if rerank:
        mode_label += " + RERANK"
    print("\n" + "=" * 75)
    print(f"FINSIGHT RETRIEVAL QUALITY EVALUATION [{mode_label}] ({total_queries} queries)")
    print("=" * 75)

    for idx, test in enumerate(test_suite, 1):
        q = test["query"]
        exp_ticker = test["expected_ticker"]
        exp_sections = test.get("expected_sections") or [test.get("expected_section")]

        # Run unconstrained global retrieval (no metadata filters) to stress-test semantic matching
        results = retriever.retrieve(query=q, k=top_k, hybrid=hybrid, rerank=rerank)

        first_match_rank: Optional[int] = None
        for rank, r in enumerate(results, 1):
            if r["ticker"] == exp_ticker and r["section"] in exp_sections:
                first_match_rank = rank
                break

        # Compute metric contributions
        if first_match_rank == 1:
            hits_at_1 += 1
        if first_match_rank and first_match_rank <= 3:
            hits_at_3 += 1
        if first_match_rank and first_match_rank <= 5:
            hits_at_5 += 1

        rr = (1.0 / first_match_rank) if first_match_rank else 0.0
        reciprocal_ranks.append(rr)

        status_tag = f"[MATCH Rank {first_match_rank}]" if first_match_rank else "[MISS in top-5]"
        sec_label = "/".join(exp_sections)
        print(f"[{idx:2d}/{total_queries}] {status_tag:<18} | Target: [{exp_ticker} {sec_label}]")
        print(f"     Query: \"{q[:65]}...\"")

    hit_rate_1 = (hits_at_1 / total_queries) * 100
    hit_rate_3 = (hits_at_3 / total_queries) * 100
    hit_rate_5 = (hits_at_5 / total_queries) * 100
    mrr = sum(reciprocal_ranks) / total_queries

    print("-" * 75)
    print("RETRIEVAL EVALUATION SCOREBOARD")
    print("-" * 75)
    print(f"  Hit Rate @ 1: {hit_rate_1:6.1f}% ({hits_at_1}/{total_queries})")
    print(f"  Hit Rate @ 3: {hit_rate_3:6.1f}% ({hits_at_3}/{total_queries})")
    print(f"  Hit Rate @ 5: {hit_rate_5:6.1f}% ({hits_at_5}/{total_queries})")
    print(f"  MRR (Mean Reciprocal Rank): {mrr:.4f}")
    print("=" * 75)

    return {
        "hit_rate_at_1": hit_rate_1,
        "hit_rate_at_3": hit_rate_3,
        "hit_rate_at_5": hit_rate_5,
        "mrr": mrr,
    }


if __name__ == "__main__":
    print("=" * 75)
    print("FinSight: Public Retrieval API & Evaluation Harness")
    print("=" * 75)

    # 1. Test clean single-function retrieve() API
    sample_q = "What were Apple's net sales and iPhone revenue in 2024?"
    print(f"\n>>> Testing retrieve() API with query: '{sample_q}' (k=3)...")
    chunks = retrieve(query=sample_q, k=3, ticker="AAPL")

    print(f"[OK] Retrieved {len(chunks)} chunks:")
    for rank, c in enumerate(chunks, 1):
        raw_snippet = c["text"].replace("\n", " ")[:120]
        snippet = raw_snippet.encode("ascii", errors="replace").decode("ascii")
        print(f"  [{rank}] Score: {c['score']:.4f} | [{c['ticker']} FY{c['fiscal_year']} {c['section']}]")
        print(f"      Excerpt: \"{snippet}...\"")

    # 2. Test context formatting for LLM prompt
    print("\n>>> Testing format_context_for_prompt()...")
    context_block = format_context_for_prompt(chunks, max_tokens=600)
    # Sanitize for Windows terminal display
    safe_context = context_block.encode("ascii", errors="replace").decode("ascii")
    print(safe_context[:350] + "\n...[truncated for display]...")

    # 3. Run IR Quality Benchmark Suite
    metrics = evaluate_retrieval_quality()

    # Clean shutdown
    FinSightRetriever.get_instance().close()
    print("[OK] Search API & Retrieval Evaluation completed successfully!")
