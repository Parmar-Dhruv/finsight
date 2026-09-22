# FinSight

> **Domain-Adapted Retrieval-Augmented Generation (RAG) for SEC 10-K Financial Disclosures**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Vector DB](https://img.shields.io/badge/Vector_DB-Qdrant-DC2626?style=flat-square&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Embeddings](https://img.shields.io/badge/Embeddings-BAAI%2Fbge--small--en--v1.5-blue?style=flat-square)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![Base LLM](https://img.shields.io/badge/LLM-Llama_3.2_3B_Instruct-purple?style=flat-square&logo=meta&logoColor=white)](https://ai.meta.com/llama/)
[![API](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 1. Project Overview

**FinSight** is a domain-specialized question-answering system designed to extract accurate, citation-grounded insights from complex SEC Form 10-K annual reports. Unlike generic chatbots, FinSight couples structure-preserving financial document parsing with dense vector indexing and strict attribution prompting to eliminate hallucinations and preserve numerical integrity across public corporate disclosures.

* **Target Sector:** US Public Technology
* **Coverage:** Apple (`AAPL`), Microsoft (`MSFT`), Amazon (`AMZN`), Alphabet (`GOOGL`), and Meta (`META`)
* **Time Horizon:** Fiscal Years 2023 and 2024

---

## 2. Problem Statement

General-purpose Large Language Models (LLMs) and off-the-shelf RAG implementations struggle with corporate financial filings due to three core failure modes:

1. **Destruction of Tabular Context:** Naive character or token chunking slices across financial tables and footnotes, separating balance sheet figures from their fiscal years, currencies, and scale factors ($M vs. $B).
2. **Domain Terminology Gap:** Off-the-shelf embeddings fail to differentiate subtle regulatory and accounting terminology (*e.g., operating margin vs. net margin, diluted EPS, impairment losses*).
3. **Temporal & Entity Ambiguity:** Financial figures are frequently restated across successive annual filings. Without strict single-stage metadata filtering, models retrieve conflicting figures from wrong fiscal years.

---

## 3. Objectives

* **O1: Structure Preservation:** Retain tabular relationships by translating raw HTML filing tables into Markdown before text chunking.
* **O2: High-Precision Retrieval:** Achieve >85% Hit Rate @ 5 and >0.65 MRR on financial extraction benchmarks.
* **O3: Attribution & Grounding:** Ensure 100% of generated responses cite specific filing provenance (company ticker, fiscal year, 10-K Item section).
* **O4: Zero-Friction Reproducibility:** Support dual execution environments (managed Qdrant Cloud and zero-dependency local embedded storage).

---

## 4. Key Features

* **Table-to-Markdown Ingestion:** HTML cleaning pipeline preserves multi-column tabular relationships and footnotes.
* **Recursive Document Chunking:** Splits filing text along structural paragraph, table, and sentence boundaries (~375 tokens) to maintain semantic coherence.
* **Dense Embedding Optimization:** Employs `BAAI/bge-small-en-v1.5` with L2 normalization for query-passage similarity.
* **Single-Stage Metadata Filtering:** Directly filters by `ticker`, `fiscal_year`, and `section` during vector search to prevent cross-company noise.
* **Dual Database Modes:** Operates seamlessly on Qdrant Cloud or via an in-process local embedded Rust engine.
* **Attributed Context Formatting:** Automatically compiles retrieved passages with token budgets and provenance tags for downstream LLMs.

---

## 5. System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        1. DATA ACQUISITION & PARSING                   │
│   SEC EDGAR 10-K Filings ──> Table Extraction ──> Section Normalization│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Structured Sections (Items 1, 1A, 7, 8)
┌───────────────────────────────────▼────────────────────────────────────┐
│                    2. CHUNKING & EMBEDDING PIPELINE                    │
│   Recursive Boundary Chunking ──> BAAI/bge-small-en-v1.5 (384-dim)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Dense Vectors + Metadata Payloads
┌───────────────────────────────────▼────────────────────────────────────┐
│                    3. VECTOR DATABASE & RETRIEVAL                      │
│   Qdrant (Cloud / Embedded) ──> HNSW Graph + Single-Stage Filtering    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Top-K Attributed Chunks
┌───────────────────────────────────▼────────────────────────────────────┐
│                    4. GENERATION & REASONING (LLM)                     │
│   Token Budgeting ──> Llama 3.2 3B Instruct (QLoRA Fine-tuned)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Grounded Answer + Source Citations
┌───────────────────────────────────▼────────────────────────────────────┐
│                    5. SERVING & EVALUATION LAYER                       │
│   FastAPI Endpoints ──> Docker ──> IR Benchmarks & RAGAS Groundedness  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. RAG Workflow

1. **Query Processing:** The incoming financial question is prefixed with an asymmetric search instruction and encoded into a 384-dimensional dense vector.
2. **Filtered HNSW Search:** Qdrant evaluates cosine distance while enforcing metadata filter predicates (`ticker`, `fiscal_year`, `section`) directly within the index graph.
3. **Context Assembly:** The top-$k$ retrieved chunks are formatted with strict source attributions (`[SOURCE X | TICKER FY SECTION]`) within a specified token budget.
4. **Grounded Generation:** The LLM receives the system instruction, retrieved context, and user question, producing an answer with direct citations.

---

## 7. Technology Stack

| Layer | Tools / Technologies | Function |
|:---|:---|:---|
| **Data Ingestion** | BeautifulSoup4, Requests | SEC EDGAR extraction & HTML table parsing |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | Dense semantic encoding (384 dimensions) |
| **Vector Store** | Qdrant (Cloud & Embedded) | Cosine similarity search with payload indices |
| **LLM Reasoning** | Llama 3.2 3B Instruct | QLoRA domain fine-tuning & prompt generation |
| **Serving Layer** | FastAPI, Uvicorn, Docker | Asynchronous REST endpoints & orchestration |
| **Evaluation** | Custom IR Benchmarking, RAGAS | Retrieval Hit Rate, MRR, and faithfulness |

---

## 8. Project Structure

```text
finsight/
├── data/
│   ├── raw/                  # Downloaded SEC EDGAR 10-K HTML filings
│   └── processed/            # Parsed section JSONs, chunks.json, embeddings.npy
├── retrieval/                # Vector Database & Retrieval Engine
│   ├── parse.py              # HTML cleaning & table-to-markdown extraction
│   ├── chunk.py              # Recursive structure-preserving chunker
│   ├── embed.py              # Dense vector embedding generator
│   ├── index.py              # Qdrant client manager (Cloud + Local)
│   └── search.py             # Public retrieve() API & context formatter
├── generation/               # LLM Generation & Fine-Tuning
│   ├── prompt_templates.py   # Citation-enforcing prompt templates
│   └── train_lora.py         # QLoRA fine-tuning scripts
├── api/                      # REST Service & Serving
│   └── main.py               # FastAPI application endpoints
├── eval/                     # Evaluation & Benchmarks
│   └── benchmark/            # Curated financial QA evaluation sets
├── .env.example              # Template for required environment variables
├── requirements.txt          # Python dependency specifications
└── README.md                 # Project documentation
```

---

## 9. Data Sources

* **Source:** Official SEC EDGAR public disclosure system.
* **Form Type:** Form 10-K (Annual Comprehensive Filing).
* **Target Companies:**
  * Apple Inc. (`AAPL`)
  * Microsoft Corporation (`MSFT`)
  * Amazon.com, Inc. (`AMZN`)
  * Alphabet Inc. (`GOOGL`)
  * Meta Platforms, Inc. (`META`)
* **Extracted Sections:**
  * **Item 1:** Business Overview
  * **Item 1A:** Risk Factors
  * **Item 7:** Management's Discussion and Analysis (MD&A)
  * **Item 8:** Financial Statements & Supplementary Data

---

## 10. Retrieval Pipeline

The ingestion pipeline transforms raw SEC disclosures into query-ready vector collections:

1. **HTML Parsing:** `retrieval/parse.py` isolates target 10-K Items, discards navigational boilerplate, and translates table cells into Markdown rows.
2. **Recursive Chunking:** `retrieval/chunk.py` splits text using a 4-tier separator hierarchy (`\n\n` $\rightarrow$ `\n` $\rightarrow$ `. ` $\rightarrow$ ` `), maintaining an average chunk size of ~375 tokens with a 200-character overlap.
3. **Embedding Generation:** `retrieval/embed.py` generates normalized dense vectors using `BAAI/bge-small-en-v1.5` on CPU or CUDA.

---

## 11. Vector Database & Search

FinSight uses **Qdrant** configured with:
* **Distance Metric:** Cosine Similarity
* **Vector Dimension:** 384
* **Payload Indices:** `ticker` (keyword), `fiscal_year` (integer), `section` (keyword)
* **Single-Stage Filtering:** Metadata constraints filter candidates directly during graph traversal, avoiding post-filtering recall loss.

**Execution Modes:**
* **Cloud Mode:** Connects to managed AWS cluster (`us-east-2`) housing all pre-indexed vectors.
* **Local Mode:** Automatically falls back to an embedded in-process disk database at `data/processed/qdrant_db/` if cloud credentials are not supplied.

---

## 12. Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Parmar-Dhruv/finsight.git
cd finsight
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 13. Environment Variables

Create your local `.env` file from the provided template:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

Populate the configuration keys in `.env`:

```dotenv
# Required by SEC EDGAR Fair Access Policy
SEC_EDGAR_USER_AGENT="FinSight Research Team (contact: your_email@example.com)"

# Qdrant Vector DB Configuration
# Provide cluster credentials for Cloud Mode, or leave blank for Local Embedded Mode
QDRANT_URL="https://your-cluster-id.cloud.qdrant.io"
QDRANT_API_KEY="your-api-key-here"
```

---

## 14. Running the Project

### Smoke Test & Retrieval Verification
Run the verification script to confirm embedding initialization and vector search:

```bash
python retrieval/search.py
```

### Rebuilding Ingestion from Scratch *(Optional)*
```bash
python retrieval/parse.py   # Parse raw filings
python retrieval/chunk.py   # Chunk document text
python retrieval/embed.py   # Generate vector embeddings
python retrieval/index.py   # Index vectors into Qdrant
```

---

## 15. API Documentation

### High-Level Python Interface
```python
from retrieval.search import retrieve, format_context_for_prompt

# 1. Retrieve top-k passages with metadata filters
chunks = retrieve(
    query="What were Apple's iPhone net sales in fiscal 2024?",
    k=3,
    ticker="AAPL",
    fiscal_year=2024,
    section="item_7"
)

# 2. Format into attributed prompt context
context_block = format_context_for_prompt(chunks, max_tokens=1000)
```

### Chunk Payload Schema
```json
{
  "score": 0.8124,
  "chunk_id": "AAPL_2024_item_7_chunk_005",
  "ticker": "AAPL",
  "fiscal_year": 2024,
  "section": "item_7",
  "token_count": 312,
  "text": "Total net sales increased 2% or $7.8 billion during 2024 compared to 2023..."
}
```

---

## 16. Example Queries

* *"What was Microsoft's Intelligent Cloud revenue growth in fiscal year 2024?"*
* *"What are Amazon's primary risk factors regarding AWS infrastructure?"*
* *"Compare Alphabet's Google Services revenue between 2023 and 2024."*
* *"What capital expenditures did Meta disclose for AI data centers in 2024?"*

---

## 17. Evaluation

Evaluated against an empirical multi-company financial query test set:

| Evaluation Metric | Measured Score | Description |
|:---|:---|:---|
| **Hit Rate @ 1** | **62.5%** | Correct section and company retrieved at Rank 1 without filters |
| **Hit Rate @ 3** | **75.0%** | Target financial passage present within top-3 results |
| **Hit Rate @ 5** | **87.5%** | Target financial passage present within top-5 results |
| **MRR** | **0.6917** | Mean Reciprocal Rank across benchmark queries |
| **Filter Precision** | **100.0%** | Zero cross-company leakage when metadata filters are applied |

---

## 18. Limitations

* **Model Capacity Ceiling:** Llama 3.2 3B has a lower natural reasoning ceiling compared to 70B+ frontier models; complex multi-hop calculations require prompt-level guidance.
* **Corpus Scope:** Current implementation focuses on US Public Tech sector 10-K filings; does not cover foreign filings (Form 20-F) or quarterly updates (Form 10-Q).
* **Static Snapshot:** Operates over historical filed annual disclosures; does not incorporate real-time market or tick data.

---

## 19. Future Scope

* **Hybrid Lexical Search:** Integrating BM25 sparse retrieval alongside dense BGE vectors to improve exact ticker and numeric matching.
* **Cross-Encoder Reranking:** Adding a secondary `bge-reranker-large` stage to refine top-5 passages before generation.
* **Structured Financial Extraction:** Expanding table parsing with tabular semantic models (e.g., Table-Transformer) for multi-level nested tables.
* **Interactive Frontend:** Deploying an attributed Streamlit/Gradio web dashboard with interactive PDF citation highlights.

---

## 20. Team & Responsibilities

| Team Member | Project Role | Git Branch | Core Responsibilities |
|:---|:---|:---|:---|
| **Nilay Patel** | **Vector DB & Retrieval** | `nilay` | SEC acquisition, table parser, chunking, BGE embeddings, Qdrant index & retrieval API |
| **Dhruv Parmar** | **LLM & Fine-Tuning** | `dhruv` | QLoRA fine-tuning (Llama 3.2 3B), citation prompt engineering, GGUF quantization |
| **Jay Patel** | **Serving & MLOps** | `jay` | FastAPI microservice, Docker deployment, CI/CD pipeline, RAGAS automated eval |

---

## 21. License

Distributed under the **MIT License**. See `LICENSE` for details.

---

## 22. Acknowledgements / References

* **SEC EDGAR System:** Source of corporate 10-K filings under fair access standards.
* **BAAI:** `bge-small-en-v1.5` dense embedding model.
* **Qdrant:** Vector Search Engine & Cloud Cluster infrastructure.
* **Meta AI:** Llama 3.2 open-weight foundational model architecture.

```bibtex
@misc{finsight2026,
  author = {Nilay Patel and Dhruv Parmar and Jay Patel},
  title = {FinSight: Domain-Adapted RAG System for Financial Document Q&A},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Parmar-Dhruv/finsight}}
}
```
