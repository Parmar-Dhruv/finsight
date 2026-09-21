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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Part 1: Singleton Retriever & Public API ─────────────────────────────────

class FinSightRetriever:
    """
    Singleton retriever that keeps the embedding model and vector database
    connection warm in memory across requests.
    """

    _instance: Optional["FinSightRetriever"] = None

    def __init__(self, local_path: Optional[Path] = None) -> None:
        """Initializes the embedding engine and vector index connection."""
        logger.info("Initializing FinSightRetriever singleton...")
        self.engine = EmbeddingEngine()
        self.index_manager = QdrantIndexManager(local_path=local_path)
        logger.info("[OK] FinSightRetriever ready for incoming queries.")

    @classmethod
    def get_instance(cls, local_path: Optional[Path] = None) -> "FinSightRetriever":
        """Thread-safe accessor for the retriever singleton."""
        if cls._instance is None:
            cls._instance = cls(local_path=local_path)
        return cls._instance

    def retrieve(
        self,
        query: str,
        k: int = 5,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        section: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k most relevant financial text chunks for a given query.

        Parameters:
            query: Natural language financial question or search term.
            k: Number of chunks to retrieve (default: 5).
            ticker: Optional company ticker filter (e.g. 'AAPL', 'MSFT').
            fiscal_year: Optional filing fiscal year (e.g. 2024, 2023).
            section: Optional 10-K section filter (e.g. 'item_7', 'item_1a').

        Returns:
            List of result dictionaries containing score, chunk_id, ticker,
            fiscal_year, section, token_count, and full text.
        """
        if not query or not query.strip():
            return []

        return search_index(
            manager=self.index_manager,
            query=query.strip(),
            engine=self.engine,
            top_k=k,
            ticker=ticker,
            fiscal_year=fiscal_year,
            section=section,
        )

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
) -> List[Dict[str, Any]]:
    """
    Public API function for pipeline integration (Dhruv / Jay).
    
    Example:
        >>> from retrieval.search import retrieve
        >>> chunks = retrieve("What was Microsoft's cloud growth?", k=3, ticker="MSFT")
    """
    retriever = FinSightRetriever.get_instance()
    return retriever.retrieve(
        query=query,
        k=k,
        ticker=ticker,
        fiscal_year=fiscal_year,
        section=section,
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
        ]

    retriever = FinSightRetriever.get_instance()
    total_queries = len(test_suite)
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks: List[float] = []

    print("\n" + "=" * 75)
    print(f"FINSIGHT RETRIEVAL QUALITY EVALUATION (Benchmark: {total_queries} queries)")
    print("=" * 75)

    for idx, test in enumerate(test_suite, 1):
        q = test["query"]
        exp_ticker = test["expected_ticker"]
        exp_section = test["expected_section"]

        # Run unconstrained global retrieval (no metadata filters) to stress-test semantic matching
        results = retriever.retrieve(query=q, k=top_k)

        first_match_rank: Optional[int] = None
        for rank, r in enumerate(results, 1):
            if r["ticker"] == exp_ticker and r["section"] == exp_section:
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
        print(f"[{idx:2d}/{total_queries}] {status_tag:<18} | Target: [{exp_ticker} {exp_section}]")
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
