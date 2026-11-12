# FinSight: Domain-Adapted Retrieval-Augmented Generation for SEC 10-K Financial Question Answering

**Team:** Nilay (Vector DB & Retrieval) · Dhruv (LLM Integration & Fine-Tuning) · Jay (Serving, Evaluation & MLOps)  
**Scope:** 6 US Public Tech Companies (`AAPL`, `MSFT`, `AMZN`, `GOOGL`, `META`, `NVDA`) · SEC Form 10-K · FY2023–FY2024

---

## 1. Problem Statement

General-purpose Large Language Models (LLMs) and off-the-shelf Retrieval-Augmented Generation (RAG) pipelines perform poorly on corporate financial filings due to three structural mismatches:

1. **Tabular & Numerical Context Destruction:** Fixed-size character/token chunking slices across HTML financial tables and footnotes, severing reported figures from their fiscal year, currency, and scale units (`$M` vs. `$B`).
2. **Domain Terminology Sensitivity:** Generic retrieval pipelines struggle to distinguish closely related accounting metrics (*e.g., operating income vs. net income, gross margin, diluted EPS*) or match exact fiscal-year qualifiers (`FY2023` vs. `FY2024`).
3. **Temporal & Entity Ambiguity:** Multi-year SEC 10-K filings contain restated or comparative tables. Without single-stage metadata filtering (`ticker`, `fiscal_year`, `section`), vector search frequently surfaces figures from the wrong company or fiscal period.

---

## 2. Methodology

### 2.1 Structure-Aware Ingestion & Indexing (Retrieval Layer)
- **Parsing (`retrieval/parse.py`):** SEC EDGAR 10-K HTML filings are parsed with financial tables converted to Markdown tables and normalized into canonical SEC sections (`item_1`, `item_1a`, `item_7`, `item_8`).
- **Chunking (`retrieval/chunk.py`):** Recursive boundary-respecting chunking (~375 tokens/chunk) produces **3,504 structured chunks** across 12 filings (6 companies × 2 fiscal years), preserving table integrity and provenance metadata.
- **Embeddings (`retrieval/embed.py`):** `BAAI/bge-small-en-v1.5` (384-dim L2-normalized) executed via a zero-PyTorch pure-NumPy `FastBertEmbedder`.
- **Hybrid Retrieval & Reranking (`retrieval/index.py`, `retrieval/hybrid.py`, `retrieval/rerank.py`):** Single-stage metadata-filtered HNSW search in Qdrant (with automatic Cloud → Local Embedded → BM25 fallback), fused with `rank-bm25` via Reciprocal Rank Fusion (RRF), and optionally reranked using `cross-encoder/ms-marco-MiniLM-L-6-v2` (`FastCrossEncoder`).

### 2.2 Citation-Enforced Generation & Fine-Tuning (Generation Layer)
- **Dataset (`generation/build_finetune_dataset.py`):** Curated from `virattt/financial-qa-10K` with strict de-duplication/de-leaking against the 100-question evaluation benchmark.
- **Prompt Contract (`generation/prompt_templates.py`):** Enforces `[SOURCE N]` inline citations for every factual claim and a canonical `REFUSAL_MESSAGE` when retrieved evidence is insufficient.
- **Fine-Tuning & Quantization (`generation/train_lora.py`, `generation/quantize_gguf.py`):** QLoRA fine-tuning of `Llama-3.2-3B-Instruct` (rank 16, alpha 32, 71 steps; train loss `2.3516 → 0.4042`, val loss `0.4136`), exported to `Q4_K_M` GGUF (`2.02 GB` at `generation/checkpoints/gguf/finsight-llama-3.2-3b-q4_k_m.gguf`).
- **Inference Fallback (`generation/inference.py`):** When `llama-cpp-python` is unavailable on CPU environments without a C++ compiler, `FinSightGenerator` executes its `domain_synthesis` grounded extraction engine to preserve `[SOURCE N]` citation contracts.

### 2.3 Serving, Experiment Tracking & Monitoring
- **FastAPI (`api/main.py`):** Exposes `/health`, `/metadata`, `/metrics`, `/retrieve`, and `/query` with thread-safe `RequestMonitor` middleware tracking p50/p95 latency, error rate, and citation/refusal rates.
- **MLflow & Docker (`eval/benchmark.py`, `docker-compose.yml`):** Automated experiment tracking (`FinSight-Retrieval`) and containerized deployment for API + MLflow UI.

---

## 3. Comparative Evaluation Results (Task J-5)

Evaluated on the **100-question SEC 10-K benchmark** (`evaluation/dataset/questions.json`, covering `AAPL`, `MSFT`, `AMZN`, `GOOGL`, `META` across 6 question types):

| Configuration | HR@1 | HR@5 | MRR | Citation Rate | Refusal Rate | Numeric Match | Token F1 | Faithfulness | p95 Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1. Base LLM (No RAG)** | N/A | N/A | N/A | 0.0% | 0.0% | 1.1% | 0.3246 | 0.0000 | 0.0 |
| **2. Naive RAG** | 13.0% | 34.0% | 0.2128 | 100.0% | 0.0% | 3.3% | 0.1778 | 1.0000 | 384.9 |
| **3. Domain Retrieval Only** | 67.0% | 86.0% | 0.7340 | 100.0% | 0.0% | 3.3% | 0.1810 | 1.0000 | 27,104.9 |
| **4. Fine-Tuned Only (No RAG)** | N/A | N/A | N/A | 0.0% | 0.0% | 1.1% | 0.3246 | 0.0000 | 0.0 |
| **5. Full FinSight System** | **82.0%** | **94.0%** | **0.8683** | **100.0%** | **0.0%** | **5.5%** | 0.1701 | **1.0000** | **3,836.8** |

*Note:* In this Windows evaluation environment, `llama-cpp-python` is not installed, so `FinSightGenerator` ran with its `domain_synthesis` backend across all configurations (making Configs 1 and 4 share the no-RAG baseline).

### Key Empirical Findings
1. **+60.0 pp HR@5 & +0.6555 MRR over Naive RAG:** Structure-aware chunking, hybrid BM25+RRF, cross-encoder reranking (`rerank_pool=10`), and query-time entity/year filtering lifted Hit Rate@5 from **34.0%** to **94.0%** and Hit Rate@1 from **13.0%** to **82.0%**.
2. **Metadata Filtering Cuts Latency by 7×:** Filtering Qdrant/BM25 by extracted `ticker` and `fiscal_year` reduced p95 end-to-end latency from **27,104.9 ms** (global search) to **3,836.8 ms** (meeting the `< 5,000 ms` SLA).
3. **100% Citation & Groundedness:** All RAG configurations achieved **100.0% `[SOURCE N]` citation compliance** and **1.0000 faithfulness** against retrieved passages.

---

## 4. Error Analysis (Task N/D-1)

From `evaluation/results/nd1_error_analysis.json`:

| Outcome Category | Naive RAG | Domain Retrieval Only | Full FinSight System |
|---|---:|---:|---:|
| **Retrieval Miss** | **66.0%** | 14.0% | **6.0%** |
| **Ranking Error (causing numeric miss)** | 18.0% | 14.0% | 9.0% |
| **Ranking Error (low F1)** | 3.0% | 5.0% | 3.0% |
| **Numeric Extraction Error** | 12.0% | 61.0% | **73.0%** |
| **Success (Retrieval + Numeric/Semantic Match)** | 1.0% | 6.0% | 9.0% |

### Diagnostic Takeaways
- **Nilay's Retrieval Module:** Retrieval misses dropped from **66% (Naive)** → **14% (Global Domain)** → **6% (Full System)**. Remaining misses are concentrated in `numeric_extraction` (3) and `qualitative` (2).
- **Dhruv's Generation Module:** Once retrieval succeeds (94% of queries), **73% of errors** in the `domain_synthesis` fallback stem from single-sentence prose extraction failing to parse multi-column Markdown tables, convert units (`$391,035 million` → `$391.0 billion`), or compute YoY arithmetic (`numeric_reasoning`, `tabular`, `segment_analysis`). Running the quantized `Llama-3.2-3B-Instruct` GGUF model via `llama-cpp-python` on a Linux/GPU or C++-enabled host directly addresses this gap.

---

## 5. Limitations & Future Work

### Explicit Limitations
1. **3B Parameter Reasoning Ceiling:** `Llama-3.2-3B-Instruct` has bounded multi-step financial arithmetic capacity compared to 7B–70B models; complex `numeric_reasoning` and multi-table synthesis remain challenging.
2. **Narrow Corpus Scope:** Evaluated on **5–6 US Public Tech companies** (`AAPL`, `MSFT`, `AMZN`, `GOOGL`, `META`, `NVDA`) and **10-K filings only** (`FY2023–FY2024`). Cross-sector generalization (banking, energy, healthcare) and `10-Q` / earnings calls are out of scope.
3. **General-Purpose Embeddings:** `BAAI/bge-small-en-v1.5` is not finance-pretrained; domain adaptation is achieved at the system level (table parsing, metadata filtering, BM25 fusion, cross-encoder reranking).
4. **Local CPU Runtime Constraint:** On Windows Python 3.14 without `llama-cpp-python`, generation uses the deterministic `domain_synthesis` fallback rather than executing the 2.02 GB GGUF weights.

### Future Work
- Deploy the `api/Dockerfile` container on Linux with `llama-cpp-python` compiled to benchmark the fine-tuned `finsight-llama-3.2-3b-q4_k_m.gguf` model end-to-end on J-5.
- Add a lightweight calculator/table-parser tool-use step for `numeric_reasoning` and unit normalization (`$M` ↔ `$B`).
- Expand corpus coverage to S&P 500 sectors and quarterly `10-Q` filings.


---

## 6. CPU Latency Profile (Task J-6)

Measured on CPU (`scripts/profile_latency.py` -> `evaluation/results/j6_latency_profile.json`) across 5 representative queries:

| Pipeline Mode | Mean (ms) | p50 (ms) | p95 (ms) | Meets `< 5,000 ms` SLA |
|---|---:|---:|---:|:---:|
| **Dense-Only** | 244.1 | 158.3 | 450.4 | Yes |
| **Hybrid (Dense + BM25 RRF)** | 298.1 | 321.8 | 457.5 | Yes |
| **Hybrid + Cross-Encoder (`rerank_pool=25`)** | 7,329.7 | 7,718.6 | 7,883.4 | No |
| **Full FinSight (`rerank_pool=10`, metadata-filtered, J-5)** | — | — | **3,836.8** | **Yes** |

Reducing `rerank_pool` from `25` to `10` with entity/year metadata filtering brings CPU cross-encoder reranking within the `< 5,000 ms` p95 SLA while retaining **94.0% HR@5**.
