"""
================================================================================
FinSight — Production API Layer (api/main.py)
Task J-4: Full Retrieval & Generation Pipeline Integration
================================================================================

Replaces the J-1 stub endpoints with real end-to-end calls to:
  1. retrieval.search.retrieve() (Nilay's structure-aware, hybrid search + reranking)
  2. generation.inference.FinSightGenerator (Dhruv's citation-enforced generation)

Endpoints:
  - GET  /health   : Health check with subsystem readiness flags.
  - GET  /metadata : Supported companies, fiscal years, sections, and pipeline config.
  - POST /retrieve : Dense / Hybrid / Reranked chunk retrieval with metadata filtering.
  - POST /query    : End-to-end RAG question answering with citation tags [SOURCE N].
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from retrieval.search import retrieve
from generation.inference import FinSightGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinSight Financial RAG API",
    version="1.0.0",
    description="Domain-Adapted RAG Service for SEC 10-K Question Answering",
)


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Financial search query or question")
    top_k: Optional[int] = Field(5, ge=1, le=25, description="Number of passages to retrieve")
    ticker: Optional[str] = Field(None, description="Optional company filter (e.g. AAPL, MSFT)")
    fiscal_year: Optional[int] = Field(None, description="Optional fiscal year filter (e.g. 2024, 2023)")
    section: Optional[str] = Field(None, description="Optional 10-K section filter (e.g. item_7, item_1a)")
    hybrid: Optional[bool] = Field(True, description="Enable BM25 sparse + dense RRF hybrid search")
    rerank: Optional[bool] = Field(True, description="Enable Cross-Encoder neural reranking")


class ChunkItem(BaseModel):
    chunk_id: str
    ticker: str
    fiscal_year: int
    section: str
    score: float
    text: str
    token_count: Optional[int] = 0


class RetrieveResponse(BaseModel):
    query: str
    total_results: int
    retrieval_mode: str
    results: List[ChunkItem]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Financial question to answer")
    top_k: Optional[int] = Field(5, ge=1, le=25, description="Context retrieval pool size")
    ticker: Optional[str] = Field(None, description="Company filter (e.g. AAPL, NVDA)")
    fiscal_year: Optional[int] = Field(None, description="Fiscal year filter (e.g. 2024, 2023)")
    section: Optional[str] = Field(None, description="Filing section filter (e.g. item_7)")
    hybrid: Optional[bool] = Field(True, description="Use hybrid dense+BM25 search")
    rerank: Optional[bool] = Field(True, description="Use cross-encoder reranking")
    temperature: Optional[float] = Field(0.1, ge=0.0, le=1.0, description="Sampling temperature")


class SourceAttribution(BaseModel):
    index: int
    chunk_id: str
    ticker: str
    fiscal_year: int
    section: str
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceAttribution]
    citations: List[str]
    has_citation: bool
    is_refusal: bool
    latency_ms: float
    pipeline: Dict[str, Any]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Returns service health status and subsystem readiness flags."""
    generator = FinSightGenerator.get_instance()
    return {
        "status": "ok",
        "service": "FinSight Financial RAG API",
        "version": "1.0.0",
        "retrieval_ready": True,
        "generator_backend": generator.backend,
        "timestamp": time.time(),
    }


@app.get("/metadata", tags=["System"])
async def get_metadata():
    """Returns supported companies, filing years, sections, and architecture metadata."""
    return {
        "supported_companies": [
            {"ticker": "AAPL", "name": "Apple Inc."},
            {"ticker": "MSFT", "name": "Microsoft Corporation"},
            {"ticker": "AMZN", "name": "Amazon.com Inc."},
            {"ticker": "GOOGL", "name": "Alphabet Inc."},
            {"ticker": "META", "name": "Meta Platforms Inc."},
            {"ticker": "NVDA", "name": "NVIDIA Corporation"},
        ],
        "fiscal_years": [2023, 2024],
        "filing_sections": {
            "item_1": "Business Description & Human Capital",
            "item_1a": "Risk Factors",
            "item_7": "Management's Discussion & Analysis (MD&A)",
            "item_8": "Financial Statements & Supplementary Data",
        },
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "generation_model": "Llama-3.2-3B-Instruct (QLoRA Fine-Tuned / GGUF Q4_K_M)",
    }


@app.post("/retrieve", response_model=RetrieveResponse, tags=["Retrieval"])
async def retrieve_endpoint(request: RetrieveRequest):
    """
    Retrieves the top-k relevant 10-K filing chunks with optional metadata filtering,
    BM25 lexical fusion, and cross-encoder reranking.
    """
    clean_query = request.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty or whitespace.",
        )

    try:
        chunks = retrieve(
            query=clean_query,
            k=request.top_k or 5,
            ticker=request.ticker.strip().upper() if request.ticker else None,
            fiscal_year=request.fiscal_year,
            section=request.section.strip().lower() if request.section else None,
            hybrid=bool(request.hybrid),
            rerank=bool(request.rerank),
        )
    except Exception as exc:
        logger.error(f"Retrieval error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval pipeline failed: {str(exc)}",
        )

    # Determine mode label for reporting
    mode_parts = ["dense"]
    if request.hybrid:
        mode_parts.append("bm25")
    if request.rerank:
        mode_parts.append("rerank")
    mode_label = "+".join(mode_parts)

    chunk_items = [
        ChunkItem(
            chunk_id=c.get("chunk_id", ""),
            ticker=c.get("ticker", ""),
            fiscal_year=c.get("fiscal_year", 0),
            section=c.get("section", ""),
            score=c.get("score", 0.0),
            text=c.get("text", ""),
            token_count=c.get("token_count", 0),
        )
        for c in chunks
    ]

    return RetrieveResponse(
        query=clean_query,
        total_results=len(chunk_items),
        retrieval_mode=mode_label,
        results=chunk_items,
    )


@app.post("/query", response_model=QueryResponse, tags=["Generation"])
async def query_endpoint(request: QueryRequest):
    """
    Generates a citation-backed, grounded financial answer to a natural language query
    using the full FinSight domain-adapted RAG pipeline.
    """
    clean_query = request.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty or whitespace.",
        )

    t_start = time.perf_counter()

    # 1. Retrieve relevant 10-K passages
    try:
        chunks = retrieve(
            query=clean_query,
            k=request.top_k or 5,
            ticker=request.ticker.strip().upper() if request.ticker else None,
            fiscal_year=request.fiscal_year,
            section=request.section.strip().lower() if request.section else None,
            hybrid=bool(request.hybrid),
            rerank=bool(request.rerank),
        )
    except Exception as exc:
        logger.error(f"Retrieval stage failed in /query: {exc}", exc_info=True)
        chunks = []

    # 2. Generate citation-enforced answer
    generator = FinSightGenerator.get_instance()
    gen_result = generator.generate(
        question=clean_query,
        context_chunks=chunks,
        temperature=request.temperature or 0.1,
    )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    sources = [
        SourceAttribution(
            index=s["index"],
            chunk_id=s["chunk_id"],
            ticker=s["ticker"],
            fiscal_year=s["fiscal_year"],
            section=s["section"],
            score=s["score"],
            excerpt=s["excerpt"],
        )
        for s in gen_result.get("sources", [])
    ]

    mode_label = "dense" + ("+bm25" if request.hybrid else "") + ("+rerank" if request.rerank else "")

    return QueryResponse(
        query=clean_query,
        answer=gen_result["answer"],
        sources=sources,
        citations=gen_result.get("citations", []),
        has_citation=gen_result.get("has_citation", False),
        is_refusal=gen_result.get("is_refusal", False),
        latency_ms=round(elapsed_ms, 2),
        pipeline={
            "retrieval_mode": mode_label,
            "generator_backend": gen_result.get("backend", "unknown"),
            "chunks_retrieved": len(chunks),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
