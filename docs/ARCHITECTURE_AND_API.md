# FinSight — Setup, Architecture & API Documentation (Task A-1)

## 1. System Architecture

FinSight is a domain-adapted Retrieval-Augmented Generation (RAG) system built for SEC Form 10-K financial question answering across six US Public Tech companies (`AAPL`, `MSFT`, `AMZN`, `GOOGL`, `META`, `NVDA`) for `FY2023–FY2024`.

```text
+------------------------------------------------------------------------------+
| 1. SEC EDGAR INGESTION & STRUCTURE-AWARE PARSING (retrieval/download, parse) |
|    - Downloads 10-K filings (FY2023-2024)                                    |
|    - Converts HTML financial tables -> Markdown tables                       |
|    - Normalizes sections: item_1, item_1a, item_7, item_8                    |
+--------------------------------------+---------------------------------------+
                                       |
                                       v
+------------------------------------------------------------------------------+
| 2. CHUNKING & EMBEDDING (retrieval/chunk.py, retrieval/embed.py)             |
|    - Boundary-respecting recursive chunking (~375 tokens, 3,504 chunks)      |
|    - Pure-NumPy FastBertEmbedder (BAAI/bge-small-en-v1.5, 384-dim L2-norm)   |
+--------------------------------------+---------------------------------------+
                                       |
                                       v
+------------------------------------------------------------------------------+
| 3. HYBRID VECTOR + LEXICAL RETRIEVAL & RERANKING                             |
|    - Qdrant Embedded / Cloud HNSW (single-stage metadata filtering)          |
|    - BM25 sparse index (rank-bm25) + Reciprocal Rank Fusion (RRF)            |
|    - Pure-NumPy FastCrossEncoder (ms-marco-MiniLM-L-6-v2)                    |
+--------------------------------------+---------------------------------------+
                                       |
                                       v
+------------------------------------------------------------------------------+
| 4. CITATION-ENFORCED GENERATION (generation/)                                |
|    - QLoRA Fine-tuned Llama-3.2-3B-Instruct (GGUF Q4_K_M, 2.02 GB)           |
|    - Grounded Domain Synthesis fallback when llama-cpp-python is absent      |
|    - Mandatory [SOURCE N] citations + canonical refusal on missing evidence  |
+--------------------------------------+---------------------------------------+
                                       |
                                       v
+------------------------------------------------------------------------------+
| 5. SERVING, MONITORING & EXPERIMENT TRACKING (api/, demo/, eval/)            |
|    - FastAPI (/health, /metadata, /metrics, /retrieve, /query)               |
|    - Streamlit Q&A & Monitoring Dashboard (demo/app.py)                      |
|    - MLflow experiment tracking + Docker Compose (api + mlflow services)     |
+------------------------------------------------------------------------------+
```

---

## 2. Local Setup & Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.14 on Windows and Python 3.10 in Docker)
- Git

### Installation
```bash
git clone https://github.com/Parmar-Dhruv/finsight.git
cd finsight
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### Running the Services

> **Note on Local Qdrant Embedded Lock:** The local embedded Qdrant database (`data/processed/qdrant_db/`) uses an exclusive file lock. Run either the FastAPI server OR the Streamlit UI at a time when using local embedded mode.

1. **FastAPI Production Server:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   Open interactive Swagger docs at `http://localhost:8000/docs`.

2. **Streamlit Interactive Q&A & Monitoring Dashboard:**
   ```bash
   streamlit run demo/app.py
   ```
   Open `http://localhost:8501`.

3. **Docker Compose (FastAPI + MLflow):**
   ```bash
   docker compose up --build
   ```
   - FastAPI API: `http://localhost:8000`
   - MLflow Tracking UI: `http://localhost:5000`

4. **Free-Tier Cloud Deployment (Hugging Face Spaces / Streamlit Cloud):**
   - **Streamlit Cloud:** Point the app entrypoint to `demo/app.py` with Python 3.10+ and `requirements.txt`.
   - **Hugging Face Spaces (Docker SDK):** Use `api/Dockerfile` (exposes port `8000` with `HEALTHCHECK` on `/health`).

5. **Run Test Suite:**
   ```bash
   pytest
   ```

---

## 3. REST API Reference (`api/main.py`)

### `GET /health`
Returns service health status and active generator backend (`llama_cpp` or `domain_synthesis`).

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "service": "FinSight Financial RAG API",
  "version": "1.0.0",
  "retrieval_ready": true,
  "generator_backend": "domain_synthesis",
  "timestamp": 1731400000.0
}
```

### `GET /metadata`
Returns supported tickers (`AAPL`, `MSFT`, `AMZN`, `GOOGL`, `META`, `NVDA`), fiscal years (`2023`, `2024`), 10-K sections (`item_1`, `item_1a`, `item_7`, `item_8`), and model identifiers.

### `GET /metrics`
Returns live operational telemetry collected by `RequestMonitor`:
- `uptime_seconds`, `total_requests`, `total_errors`, `error_rate_pct`
- `latency_ms`: `avg`, `p50`, `p95`, `max`
- `rag_quality`: `total_queries`, `citation_rate_pct`, `refusal_rate_pct`
- `endpoint_counts`, `endpoint_errors`, `recent_requests`

### `POST /retrieve`
Retrieves top-$k$ 10-K passages with optional metadata filtering, BM25 hybrid fusion, and cross-encoder reranking.

**Request Body:**
```json
{
  "query": "What were Apple's total net sales in fiscal 2024?",
  "top_k": 5,
  "ticker": "AAPL",
  "fiscal_year": 2024,
  "section": "item_7",
  "hybrid": true,
  "rerank": true
}
```

### `POST /query`
End-to-end RAG endpoint returning a citation-enforced answer with `[SOURCE N]` tags, provenance metadata, refusal flag, and end-to-end latency.

**Request Body:**
```json
{
  "query": "What were Apple's total net sales in fiscal 2024?",
  "top_k": 5,
  "ticker": "AAPL",
  "fiscal_year": 2024,
  "hybrid": true,
  "rerank": true,
  "temperature": 0.1
}
```

**Response (`200 OK`):**
```json
{
  "query": "What were Apple's total net sales in fiscal 2024?",
  "answer": "Total net sales increased 2% or $7.8 billion during 2024 compared to 2023, driven primarily by Services [SOURCE 1].",
  "sources": [
    {
      "index": 1,
      "chunk_id": "AAPL_2024_item_7_0012",
      "ticker": "AAPL",
      "fiscal_year": 2024,
      "section": "item_7",
      "score": 0.8912,
      "excerpt": "..."
    }
  ],
  "citations": ["[SOURCE 1]"],
  "has_citation": true,
  "is_refusal": false,
  "latency_ms": 1420.5,
  "pipeline": {
    "retrieval_mode": "dense+bm25+rerank",
    "generator_backend": "domain_synthesis",
    "chunks_retrieved": 5
  }
}
```

