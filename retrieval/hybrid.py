"""
================================================================================
FinSight — Retrieval & Data Layer
BM25 Sparse Retrieval & Reciprocal Rank Fusion (retrieval/hybrid.py)
================================================================================

Task N-3 (docs/finsight-task-breakdown.md):
  "Implement BM25 sparse retrieval over the existing chunk corpus using rank-bm25.
   Fuse with dense retrieval scores using reciprocal rank fusion.
   Re-run evaluation with hybrid search on vs. off."

Architecture:
  - BM25Index: In-memory lexical index over the 3,504 structure-aware chunks.
  - Tokenizer: Finance-aware tokenization preserving numbers, currencies, tickers.
  - Reciprocal Rank Fusion (RRF): Combines ranks from Dense Vector Search (Qdrant)
    and Sparse Lexical Search (BM25) into a unified, high-precision ranking.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"


# ---------------------------------------------------------------------------
# Finance-Aware Tokenizer
# ---------------------------------------------------------------------------

TOKEN_PATTERN = re.compile(r"\b[A-Za-z0-9_$%.-]+\b")


def tokenize_finance_text(text: str) -> List[str]:
    """
    Tokenizes financial text while preserving symbols, numbers, tickers,
    and decimal points (e.g. '$391.0', '10-k', 'fy2024', 'aws').
    """
    if not text:
        return []
    # Lowercase and extract alphanumeric words plus financial symbols
    tokens = TOKEN_PATTERN.findall(text.lower())
    # Strip trailing punctuation like '.' at the end of a sentence
    return [t.strip(".-") for t in tokens if t.strip(".-")]


# ---------------------------------------------------------------------------
# BM25 Lexical Indexer
# ---------------------------------------------------------------------------

class BM25Index:
    """
    In-memory BM25 index over the 3,504 structure-aware financial chunks.
    Supports exact keyword, ticker, numerical, and segment matching with
    optional metadata pre-filtering.
    """

    _instance: Optional["BM25Index"] = None

    def __init__(self, chunks_path: Path = CHUNKS_PATH) -> None:
        logger.info(f"Initializing BM25Index from: {chunks_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at: {chunks_path}")

        raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.chunks: List[Dict[str, Any]] = raw_chunks
        self.chunk_id_map: Dict[str, Dict[str, Any]] = {
            c["chunk_id"]: c for c in self.chunks
        }

        # Tokenize corpus for BM25Okapi
        logger.info(f"Tokenizing {len(self.chunks)} chunks for BM25...")
        self.tokenized_corpus: List[List[str]] = [
            tokenize_finance_text(c["text"]) for c in self.chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"[OK] BM25Index ready with {len(self.chunks)} documents.")

    @classmethod
    def get_instance(cls, chunks_path: Path = CHUNKS_PATH) -> "BM25Index":
        """Singleton accessor for BM25 index."""
        if cls._instance is None:
            cls._instance = cls(chunks_path=chunks_path)
        return cls._instance

    def search(
        self,
        query: str,
        top_k: int = 20,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        section: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes BM25 lexical search against the corpus with optional metadata filters.
        """
        if not query or not query.strip():
            return []

        tokens = tokenize_finance_text(query)
        if not tokens:
            return []

        doc_scores = self.bm25.get_scores(tokens)

        # Apply metadata filters if supplied
        candidate_indices = []
        for idx, chunk in enumerate(self.chunks):
            if ticker and chunk.get("ticker") != ticker:
                continue
            if fiscal_year and chunk.get("fiscal_year") != fiscal_year:
                continue
            if section and chunk.get("section") != section:
                continue
            candidate_indices.append(idx)

        if not candidate_indices:
            return []

        # Sort candidate indices by score descending
        filtered_scores = [(idx, doc_scores[idx]) for idx in candidate_indices]
        filtered_scores.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for idx, score in filtered_scores[:top_k]:
            c = self.chunks[idx]
            results.append({
                "chunk_id": c["chunk_id"],
                "ticker": c["ticker"],
                "fiscal_year": c["fiscal_year"],
                "section": c["section"],
                "text": c["text"],
                "token_count": c.get("token_count_est", len(c["text"]) // 4),
                "bm25_score": float(score),
                "score": float(score),
            })

        return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    top_k: int = 5,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Fuses two ranked lists using Reciprocal Rank Fusion (RRF):
        RRF_Score(d) = dense_weight / (rrf_k + rank_dense) + sparse_weight / (rrf_k + rank_sparse)

    Parameters:
        dense_results: Ranked candidates from dense vector search (Qdrant).
        sparse_results: Ranked candidates from BM25 lexical search.
        top_k: Number of final fused candidates to return.
        rrf_k: Smoothing constant to mitigate high-rank dominance (standard: 60).
        dense_weight: Multiplier weight for dense vector rank.
        sparse_weight: Multiplier weight for BM25 lexical rank.

    Returns:
        Unified list of top_k chunks sorted by descending RRF score.
    """
    # Track scores and metadata by chunk_id
    fused_scores: Dict[str, float] = {}
    chunk_data: Dict[str, Dict[str, Any]] = {}
    dense_ranks: Dict[str, int] = {}
    sparse_ranks: Dict[str, int] = {}

    # 1. Process Dense Rankings (1-indexed)
    for rank, item in enumerate(dense_results, 1):
        cid = item["chunk_id"]
        dense_ranks[cid] = rank
        rrf_contribution = dense_weight / (rrf_k + rank)
        fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contribution
        chunk_data[cid] = item

    # 2. Process Sparse Rankings (1-indexed)
    for rank, item in enumerate(sparse_results, 1):
        cid = item["chunk_id"]
        sparse_ranks[cid] = rank
        rrf_contribution = sparse_weight / (rrf_k + rank)
        fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contribution
        if cid not in chunk_data:
            chunk_data[cid] = item

    # 3. Sort by fused RRF score descending
    sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    # 4. Construct final top_k result list
    final_results: List[Dict[str, Any]] = []
    for cid, score in sorted_chunks[:top_k]:
        item = chunk_data[cid].copy()
        item["score"] = float(score)
        item["rrf_score"] = float(score)
        item["dense_rank"] = dense_ranks.get(cid)
        item["sparse_rank"] = sparse_ranks.get(cid)
        final_results.append(item)

    return final_results
