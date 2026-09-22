# FinSight — Retrieval Layer Handoff to Dhruv (Generation & LoRA Layer)

**From:** Nilay (Retrieval & Data Layer)  
**To:** Dhruv (Model, Generation & Pipeline Integration Layer)  
**Component:** SEC 10-K Retrieval Engine & Prompt Context Formatter  
**Branch:** `nilay`  

---

## 1. Quick Start

To integrate the retrieval layer into your environment, pull the latest branch and install dependencies:

```bash
git fetch origin
git checkout nilay
git pull origin nilay
pip install -r requirements.txt
```

Set up your environment variables:
```bash
# Copy example environment configuration
cp .env.example .env          # Linux/macOS
Copy-Item .env.example .env   # Windows PowerShell
```
> [!NOTE]
> Ping Nilay for the shared cloud `QDRANT_API_KEY` and `QDRANT_URL`, and paste them into your local `.env`.

Verify everything is working on your machine:
```bash
python retrieval/search.py
```


---

## 2. The Retrieval API Contract

You do not need to deal with embedding dimensions, Qdrant client connection lifecycles, or chunking math. Everything is exposed via two high-level functions in [`retrieval/search.py`](file:///d:/College/3rd_Year/5th_Sem/SGP-II(CEUP301)/finsight/retrieval/search.py):

```python
from retrieval.search import retrieve, format_context_for_prompt
```

### Complete End-to-End Pipeline Example:

```python
from retrieval.search import retrieve, format_context_for_prompt

# 1. Retrieve the top-K relevant 10-K filing chunks
query = "What were Apple's total annual revenues and iPhone sales in 2024?"

chunks = retrieve(
    query=query,
    k=3,                   # Number of passages to return (default: 5)
    ticker="AAPL",         # Optional filter: 'AAPL', 'MSFT', 'AMZN', 'META', 'GOOGL'
    fiscal_year=2024,      # Optional filter: 2024, 2023
    section="item_7",      # Optional filter: 'item_7' (MD&A), 'item_1a' (Risks), 'item_8' (Financials)
)

# 2. Format chunks into a clean, attributed context block for your prompt
context_block = format_context_for_prompt(chunks, max_tokens=1500)

# 3. Construct your LoRA / Base LLM Prompt
prompt = f"""You are a specialized financial question-answering assistant.
Answer the question accurately using ONLY the provided SEC 10-K context.
Always cite the source filing and section when stating figures.

{context_block}

Question: {query}
Answer:"""

print(prompt)
# 4. Pass 'prompt' into your model:
# output = model.generate(prompt)
```

---

## 3. API Reference

### `retrieve()`
```python
def retrieve(
    query: str,
    k: int = 5,
    ticker: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    section: Optional[str] = None,
) -> List[Dict[str, Any]]
```

#### Parameters:
* **`query`** *(str, required)*: The natural language search question.
* **`k`** *(int, optional)*: Number of nearest neighbor chunks to return (default `5`).
* **`ticker`** *(str, optional)*: Filter by company ticker symbol (`'AAPL'`, `'MSFT'`, `'AMZN'`, `'META'`, `'GOOGL'`). Case-insensitive.
* **`fiscal_year`** *(int, optional)*: Filter by fiscal year (`2024` or `2023`).
* **`section`** *(str, optional)*: Filter by 10-K section (`'item_1'`, `'item_1a'`, `'item_7'`, `'item_8'`).

#### Return Value:
Returns a list of dictionaries with the following schema:
```python
[
    {
        "score": 0.8255,                       # Cosine similarity score (higher is more relevant)
        "point_id": 42,                         # Vector database point ID
        "chunk_id": "AAPL_2024_item_7_chunk_005",
        "ticker": "AAPL",
        "fiscal_year": 2024,
        "section": "item_7",
        "token_count": 312,                     # Estimated token length
        "text": "Total net sales increased 2% or $7.8 billion during 2024..."
    },
    ...
]
```

---

### `format_context_for_prompt()`
```python
def format_context_for_prompt(
    chunks: List[Dict[str, Any]],
    max_tokens: int = 2000,
) -> str
```
Automatically wraps chunks in clear attribution headers with token budgeting so your model learns where each fact comes from:

```text
=== RETRIEVED 10-K CONTEXT (3 passages) ===

[SOURCE 1 | AAPL FY2024 item_7 | Score: 0.8255]
Total net sales increased 2% or $7.8 billion during 2024 compared to 2023...

[SOURCE 2 | AAPL FY2024 item_8 | Score: 0.7810]
Wearables, Home and Accessories: $37,005M... Services: $96,169M...
=========================================================
```

---

## 4. Database Modes: Local Embedded vs. Qdrant Cloud

The retrieval engine automatically supports two execution modes:

1. **Local Embedded Mode (Default)**:
   - Stored on disk in `data/processed/qdrant_db/`.
   - Runs in-process via embedded Rust engine.
   - **Zero Docker, zero network calls, zero configuration required.**
   - Ideal for local training and debugging.

2. **Qdrant Cloud Mode (Optional)**:
   - If you set `QDRANT_URL` and `QDRANT_API_KEY` in your `.env` file, the retriever will automatically query our shared live AWS cluster in `us-east-2`.
   - All 2,816 vectors are already synced to the cloud.

---

## 5. Retrieval Quality Benchmark (Empirical Baseline)

Tested against 8 multi-company financial benchmark queries without metadata filters:

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Hit Rate @ 1** | **62.5%** | Over 60% of questions find the exact target section at Rank 1 |
| **Hit Rate @ 3** | **75.0%** | 3 out of 4 queries contain the right section in top-3 |
| **Hit Rate @ 5** | **87.5%** | 7 out of 8 queries contain the right section in top-5 |
| **MRR** | **0.6917** | Mean Reciprocal Rank |
| **Filtered Search Precision** | **100.0%** | When `ticker` is passed, 100% of chunks are strictly from that company |

---

## 6. Architecture & Understanding Documentation

For deep technical questions, viva preparation, or interview defense, see the companion walkthroughs in the repository:
* `understandings/retrieval.chunk_walkthrough`: Recursive paragraph splitting & strategy comparison.
* `understandings/retrieval.embed_walkthrough`: Mathematical derivation of L2 normalization & BGE asymmetric instructions.
* `understandings/retrieval.index_walkthrough`: HNSW graph traversal & single-stage payload filtering.
* `understandings/retrieval.search_walkthrough`: Software design contracts & IR metrics.
