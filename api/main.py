"""
================================================================================
FinSight — Production API Layer (api/main.py)
Task J-4: Full Retrieval & Generation Pipeline Integration
Task J-7: Request Telemetry & Monitoring (/metrics)
================================================================================

Replaces the J-1 stub endpoints with real end-to-end calls to:
  1. retrieval.search.retrieve() (Nilay's structure-aware, hybrid search + reranking)
  2. generation.inference.FinSightGenerator (Dhruv's citation-enforced generation)

Endpoints:
  - GET  /health   : Health check with subsystem readiness flags.
  - GET  /metadata : Supported companies, fiscal years, sections, and pipeline config.
  - GET  /metrics  : Live request latency (p50/p95/avg), error rate, and RAG telemetry.
  - POST /retrieve : Dense / Hybrid / Reranked chunk retrieval with metadata filtering.
  - POST /query    : End-to-end RAG question answering with citation tags [SOURCE N].
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from retrieval.search import retrieve
from generation.inference import FinSightGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── J-7: Request Telemetry & Monitoring Tracker ──────────────────────────────

class RequestMonitor:
    """Thread-safe in-memory telemetry tracker for latency, errors, and RAG metrics."""

    def __init__(self, max_history: int = 500) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.total_requests = 0
        self.total_errors = 0
        self.latencies_ms: Deque[float] = deque(maxlen=max_history)
        self.endpoint_counts: Dict[str, int] = {}
        self.endpoint_errors: Dict[str, int] = {}
        self.query_citations = 0
        self.query_refusals = 0
        self.query_count = 0
        self.recent_requests: Deque[Dict[str, Any]] = deque(maxlen=50)

    def record_http(self, path: str, method: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self.total_requests += 1
            self.latencies_ms.append(latency_ms)
            self.endpoint_counts[path] = self.endpoint_counts.get(path, 0) + 1
            if status_code >= 400:
                self.total_errors += 1
                self.endpoint_errors[path] = self.endpoint_errors.get(path, 0) + 1
            self.recent_requests.append({
                "timestamp": round(time.time(), 2),
                "method": method,
                "path": path,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 2),
            })

    def record_query_outcome(self, has_citation: bool, is_refusal: bool) -> None:
        with self._lock:
            self.query_count += 1
            if has_citation:
                self.query_citations += 1
            if is_refusal:
                self.query_refusals += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            lats = list(self.latencies_ms)
            err_rate = (self.total_errors / self.total_requests * 100.0) if self.total_requests else 0.0
            cite_rate = (self.query_citations / self.query_count * 100.0) if self.query_count else 0.0
            refusal_rate = (self.query_refusals / self.query_count * 100.0) if self.query_count else 0.0
            return {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "error_rate_pct": round(err_rate, 2),
                "latency_ms": {
                    "avg": round(float(np.mean(lats)), 2) if lats else 0.0,
                    "p50": round(float(np.percentile(lats, 50)), 2) if lats else 0.0,
                    "p95": round(float(np.percentile(lats, 95)), 2) if lats else 0.0,
                    "max": round(float(np.max(lats)), 2) if lats else 0.0,
                },
                "rag_quality": {
                    "total_queries": self.query_count,
                    "citation_rate_pct": round(cite_rate, 2),
                    "refusal_rate_pct": round(refusal_rate, 2),
                },
                "endpoint_counts": dict(self.endpoint_counts),
                "endpoint_errors": dict(self.endpoint_errors),
                "recent_requests": list(self.recent_requests),
            }


monitor = RequestMonitor()

app = FastAPI(
    title="FinSight Financial RAG API",
    version="1.0.0",
    description="Domain-Adapted RAG Service for SEC 10-K Question Answering",
)


@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if request.url.path != "/metrics":
        monitor.record_http(
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=elapsed_ms,
        )
    return response


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


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Returns live request latency, error rate, and RAG citation/refusal telemetry (Task J-7)."""
    return monitor.snapshot()


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
    has_cit = gen_result.get("has_citation", False)
    is_ref = gen_result.get("is_refusal", False)
    monitor.record_query_outcome(has_citation=has_cit, is_refusal=is_ref)

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
        has_citation=has_cit,
        is_refusal=is_ref,
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
