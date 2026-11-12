"""
================================================================================
FinSight - Task J-4: API Integration Unit & Integration Tests (tests/test_api.py)
================================================================================

Tests the production FastAPI endpoints:
  - GET  /health
  - GET  /metadata
  - POST /retrieve
  - POST /query

Run:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# ── Force BM25-only mode before any retrieval module is imported ──────────────
# This prevents Qdrant Cloud connection attempts (which hang or time out in
# sandboxed / offline environments where the index has not been seeded).
os.environ.pop("QDRANT_URL", None)
os.environ.pop("QDRANT_API_KEY", None)

# Reset singletons in case a previous test session left them initialised
# with cloud credentials.
def _reset_singletons() -> None:
    """Clear cached singletons so tests always start with a clean state."""
    try:
        from retrieval.search import FinSightRetriever
        FinSightRetriever._instance = None
    except Exception:
        pass
    try:
        from retrieval.hybrid import BM25Index
        BM25Index._instance = None
    except Exception:
        pass
    try:
        from generation.inference import FinSightGenerator
        FinSightGenerator._instance = None
    except Exception:
        pass

_reset_singletons()

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from api.main import app
from generation.prompt_templates import REFUSAL_MESSAGE


@pytest.fixture(scope="module")
def client():
    """Shared TestClient instance across all API tests (warm once per session)."""
    with TestClient(app) as test_client:
        yield test_client


# ─── System Endpoints ─────────────────────────────────────────────────────────

class TestSystemEndpoints:
    def test_health_check_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "FinSight" in data["service"]
        assert data["retrieval_ready"] is True
        assert "generator_backend" in data

    def test_metadata_returns_all_six_companies(self, client):
        response = client.get("/metadata")
        assert response.status_code == 200
        data = response.json()
        tickers = [c["ticker"] for c in data["supported_companies"]]
        assert sorted(tickers) == ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA"]
        assert 2024 in data["fiscal_years"]
        assert 2023 in data["fiscal_years"]
        assert "item_7" in data["filing_sections"]
        assert "item_1a" in data["filing_sections"]


# ─── Retrieve Endpoint ────────────────────────────────────────────────────────

class TestRetrieveEndpoint:
    def test_retrieve_basic_query(self, client):
        payload = {
            "query": "What were Apple's net sales in 2024?",
            "top_k": 3,
            "hybrid": True,
            "rerank": False,
        }
        response = client.post("/retrieve", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == payload["query"]
        assert len(data["results"]) <= 3
        assert len(data["results"]) > 0
        first = data["results"][0]
        assert "ticker" in first
        assert "score" in first
        assert "text" in first
        assert "section" in first

    def test_retrieve_with_ticker_filter(self, client):
        payload = {
            "query": "cloud computing revenue and server growth",
            "top_k": 3,
            "ticker": "MSFT",
            "hybrid": True,
            "rerank": False,
        }
        response = client.post("/retrieve", json=payload)
        assert response.status_code == 200
        data = response.json()
        for chunk in data["results"]:
            assert chunk["ticker"] == "MSFT"

    def test_retrieve_with_year_filter(self, client):
        payload = {
            "query": "net sales and gross margin",
            "top_k": 3,
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "hybrid": True,
            "rerank": False,
        }
        response = client.post("/retrieve", json=payload)
        assert response.status_code == 200
        data = response.json()
        for chunk in data["results"]:
            assert chunk["fiscal_year"] == 2024

    def test_retrieve_rejects_empty_query(self, client):
        response = client.post("/retrieve", json={"query": ""})
        assert response.status_code in (400, 422)

    def test_retrieve_rejects_whitespace_query(self, client):
        response = client.post("/retrieve", json={"query": "   "})
        assert response.status_code in (400, 422)


# ─── Query Endpoint (End-to-End RAG) ──────────────────────────────────────────

class TestQueryEndpoint:
    def test_query_generates_answer_with_citations(self, client):
        payload = {
            "query": "What were Apple's total net sales in fiscal 2024?",
            "top_k": 3,
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "hybrid": True,
            "rerank": False,
        }
        response = client.post("/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == payload["query"]
        assert len(data["answer"]) > 0
        assert data["latency_ms"] > 0
        assert "sources" in data
        assert len(data["sources"]) > 0
        assert data["has_citation"] is True
        assert any("[SOURCE" in c for c in data["citations"])
        assert data["is_refusal"] is False

    def test_query_handles_empty_query_validation(self, client):
        response = client.post("/query", json={"query": ""})
        assert response.status_code in (400, 422)

    def test_query_unknown_topic_refusal(self, client):
        # Query about non-existent company that yields zero chunks
        payload = {
            "query": "What was QuantumDynamic Corp widget manufacturing sales in 1992?",
            "top_k": 3,
            "ticker": "NONEXISTENT",
        }
        response = client.post("/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        # Should cleanly produce the canonical refusal message
        assert data["is_refusal"] is True
        assert REFUSAL_MESSAGE in data["answer"]
        assert len(data["sources"]) == 0

    def test_query_pipeline_telemetry(self, client):
        payload = {
            "query": "What are Microsoft's primary risk factors?",
            "top_k": 2,
            "ticker": "MSFT",
            "section": "item_1a",
            "hybrid": True,
            "rerank": False,
        }
        response = client.post("/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        pipeline = data["pipeline"]
        assert "retrieval_mode" in pipeline
        assert "generator_backend" in pipeline
        assert pipeline["chunks_retrieved"] <= 2
