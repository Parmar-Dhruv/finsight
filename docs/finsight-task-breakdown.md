
# FinSight — Problem Statement & Phase Task Breakdown

**Team:** Nilay (Vector DB & Retrieval) · Dhruv (LLM Integration & Fine-tuning) · Jay (Deployment & MLOps)
**Ratified scope:** 6 companies (AAPL, MSFT, AMZN, GOOGL, META, NVDA), US Public Tech sector, 10-K only, FY2023-2024

This document replaces open-ended role ownership with **small, atomic, assigned tasks**. Each task is scoped so the person executes it, rather than makes a design decision on your behalf. Hand these out incrementally rather than all at once.

---

## 1. Detailed Problem Statement

### 1.1 The core failure this project addresses

General-purpose LLMs and off-the-shelf RAG pipelines perform poorly on financial question-answering. This is not primarily a data-availability problem — it is a set of structural mismatches between how financial documents are built and how generic RAG systems process text:

1. **Numerical and tabular reasoning failure.** SEC 10-K filings are dense with tables, footnotes, and cross-references — revenue broken out by segment, margins calculated across multiple line items, figures qualified by fiscal year and reporting currency. Naive fixed-size text chunking cuts through table structure indiscriminately, severing a number from the unit, fiscal period, or qualifier that gives it meaning. The result is a system that retrieves a number with total confidence and no way to verify it is the *right* number.
2. **Terminology and context sensitivity.** Terms like "EBITDA," "diluted EPS," "impairment," or "segment operating margin" carry precise, regulation-bound meanings that differ from casual usage. A generic embedding model treats these as ordinary English phrases and does not weight them the way a reader with financial literacy would, which degrades retrieval precision on exactly the queries where precision matters most.
3. **Temporal and source ambiguity.** The same company publishes filings across multiple fiscal years, sometimes with restated figures. A retrieval system with no metadata awareness of company, fiscal year, and filing section risks silently pulling a stale or contradictory number and presenting it with the same confidence as a current, correct one.

### 1.2 Project goal

Build a retrieval-augmented generation system that retrieves **financially accurate, source-grounded, citation-backed** answers from a corpus of SEC 10-K filings, and **demonstrably outperforms a naive RAG baseline** on a financial QA benchmark — not merely "produces answers that sound plausible."

### 1.3 Explicitly out of scope

- General-purpose chatbot behavior
- Real-time market data or price feeds
- Trading signal generation or investment advice
- Filing types other than 10-K (no 10-Q, no earnings call transcripts, no Form 20-F)
- Sectors outside US Public Tech
- Companies outside the ratified list of 5

### 1.4 Honest framing for the report

A 3B parameter model (Llama 3.2 3B Instruct — locked, see prior decisions) has a real reasoning ceiling relative to 7-8B+ models. The project's contribution should be framed as: *domain-adapted retrieval plus targeted fine-tuning closes most of the performance gap for a small model on financial QA* — not as a claim of matching large-model quality outright. This is a legitimate, interesting result on its own; overstating it will not survive scrutiny in grading or an interview.

---

## 2. Measurable Objectives

| # | Objective | Success Criterion |
|---|---|---|
| O1 | Domain-adapted retrieval beats naive baseline | Precision@5 / Recall@5 improvement over fixed-chunk + generic-embedding baseline, on the shared 100-question benchmark |
| O2 | Grounded generation | ≥90% of generated answers cite the correct source chunk |
| O3 | Reduced hallucination | Lower hallucination rate than base LLM (no-RAG), measured via groundedness (e.g., RAGAS faithfulness) |
| O4 | Deployable system | End-to-end latency under a defined SLA (e.g., <5s p95), working API, basic monitoring |
| O5 | Reproducibility | Documented, containerized, one-command setup |

---

## 3. Current Implementation Status (as of last check)

| Person | Branch | What exists |
|---|---|---|
| Nilay | `nilay` | Full retrieval pipeline for all 6 companies (AAPL, MSFT, AMZN, GOOGL, META, NVDA): parsing, structure-aware chunking (3,504 chunks), BGE embeddings, Qdrant Cloud index, metadata filtering, `retrieve()`/`format_context_for_prompt()` API. Evaluated Naive baseline (N-1), 100-question benchmark harness (N-2), BM25 + RRF Hybrid Search (N-3), and Cross-Encoder Reranking (N-4) with full metrics recorded. |
| Jay | `jay` | A 100-question gold-answer benchmark (`evaluation/dataset/questions.json`) covering 5 companies, both fiscal years, 6 question types. FastAPI service with stub endpoints `/query`, `/retrieve`, `/health` (J-1) and Docker containerization (`Dockerfile` + `docker-compose.yml`) (J-2). |
| Dhruv | `dhruv` | Generation layer complete as harnesses: dataset builder (D-1), citation-enforced prompt templates (D-2), QLoRA training harness (D-3), and GGUF export harness (D-4) all in `generation/`, with unit tests (40 passing). Remaining: **execute** D-3 on free-tier T4, then D-4 export. |

**Coordination update:** Tasks N-1 through N-5, J-1, J-2, and D-1 through D-4 (harnesses) are completed. Retrieval architecture is locked with comprehensive benchmarks across Naive, Dense, Hybrid, and Cross-Encoder Reranking pipelines. The full generation layer now exists in `generation/` (dataset builder, prompt templates, QLoRA training, GGUF export). Next up: **execute** D-3 training on a free-tier T4, then D-4 export; in parallel Jay can proceed with J-3 (MLflow) against the locked retrieval harness.

---

## 4. Phase Task Breakdown — Small, Atomic Tasks

### Phase 1 — Data & Baseline *(retroactive completion)*

**Nilay**
- **N-1 (COMPLETED):** Build a naive baseline pipeline — fixed 512-token chunks (1,641 chunks across all 6 companies), no structure-awareness, no metadata filtering, same embedding model (`BAAI/bge-small-en-v1.5`). Run through `evaluate_retrieval_quality()` on all 6 companies. Output recorded below:
  
  | Metric | Naive Baseline | Domain-Adapted | Delta |
  |---|---|---|---|
  | **Hit Rate @ 1** | 20.0% | **70.0%** | **+50.0 pp** |
  | **Hit Rate @ 3** | 30.0% | **80.0%** | **+50.0 pp** |
  | **Hit Rate @ 5** | 40.0% | **80.0%** | **+40.0 pp** |
  | **MRR** | 0.2750 | **0.7333** | **+0.4583** |

**Jay**
- **J-1 (COMPLETED):** Scaffold a FastAPI service with three stub endpoints: `/query`, `/retrieve`, `/health`. They can return hardcoded/mock responses for now — the goal is a running service Dhruv and Nilay can point real logic at later, not full functionality yet.
- **J-2 (COMPLETED):** Add a `Dockerfile` and minimal `docker-compose.yml` so the stub API runs in a container.

---

### Phase 2 — Domain Adaptation *(in progress)*

**Nilay**
- **N-2 (COMPLETED):** Import `evaluation/dataset/questions.json` from Jay's `jay` branch. Convert its 100 questions into the query→gold-chunk format your harness expects. Re-run `evaluate_retrieval_quality()` on this expanded set for both the naive baseline (N-1) and the current domain-adapted pipeline (`eval/benchmark.py`). Output recorded below:

  | Metric | Naive Baseline (100 Qs) | Domain-Adapted (100 Qs) | Delta |
  |---|---|---|---|
  | **Hit Rate @ 1** | 13.0% | **69.0%** | **+56.0 pp** |
  | **Hit Rate @ 3** | 28.0% | **82.0%** | **+54.0 pp** |
  | **Hit Rate @ 5** | 34.0% | **91.0%** | **+57.0 pp** |
  | **MRR** | 0.2128 | **0.7633** | **+0.5505** |

  *Breakdown by question type (Domain-Adapted HR@5):* `tabular` (100.0%), `segment_analysis` (100.0%), `numeric_reasoning` (92.9%), `numeric_extraction` (91.8%), `qualitative` (88.9%), `temporal_comparison` (80.0%). Detailed results archived in `evaluation/results/n2_retrieval_benchmark.json`.
- **N-3 (COMPLETED):** Implement BM25 sparse retrieval over the existing chunk corpus using `rank-bm25` (`retrieval/hybrid.py`). Fuse with dense retrieval scores using reciprocal rank fusion (RRF). Re-ran evaluation on the full 100-question benchmark with hybrid search on vs. off (`eval/benchmark.py`). Output recorded below:

  | Metric | Naive Baseline (100 Qs) | Dense-Only (100 Qs) | Hybrid (Dense+BM25) | Delta vs Naive |
  |---|---|---|---|---|
  | **Hit Rate @ 1** | 13.0% | **69.0%** | 55.0% | **+42.0 pp** |
  | **Hit Rate @ 3** | 28.0% | **82.0%** | 68.0% | **+40.0 pp** |
  | **Hit Rate @ 5** | 34.0% | **91.0%** | 77.0% | **+43.0 pp** |
  | **MRR** | 0.2128 | **0.7633** | 0.6277 | **+0.4148** |

  *Key Empirical Insight:* Hybrid retrieval yielded a **+6.7 pp gain in Temporal Comparison queries** (86.7% vs 80.0% HR@5, MRR 0.6944 vs 0.6833), where BM25 exact-token matching eliminated "number blindness" between FY2023 and FY2024. Tabular and Segment Analysis maintained **100% HR@5**. For broad questions, dense vector search remained superior due to financial term repetition across 10-K boilerplate. Detailed results archived in `evaluation/results/n3_hybrid_benchmark.json`.
- **N-4 (COMPLETED):** Implemented cross-encoder reranking over top candidates retrieved by `retrieve()`, using `cross-encoder/ms-marco-MiniLM-L-6-v2` (`retrieval/rerank.py` via `FastCrossEncoder` pure NumPy / safetensors execution). Integrated `rerank=True` option into `retrieve()`. Re-ran evaluation on the full 100-question benchmark (`eval/benchmark.py`) comparing Naive Baseline, Dense-Only, Hybrid (N-3), and Hybrid+Rerank (N-4). Output recorded below:

  | Metric | Naive Baseline | Dense-Only | Hybrid (Dense+BM25) | Hybrid + Rerank (N-4) | Delta vs Naive |
  |---|---|---|---|---|---|
  | **Hit Rate @ 1** | 13.0% | 69.0% | 55.0% | **63.0%** | **+50.0 pp** |
  | **Hit Rate @ 3** | 28.0% | 82.0% | 68.0% | **77.0%** | **+49.0 pp** |
  | **Hit Rate @ 5** | 34.0% | 91.0% | 77.0% | **81.0%** | **+47.0 pp** |
  | **MRR** | 0.2128 | 0.7633 | 0.6277 | **0.7002** | **+0.4873** |

  *Key Empirical Insight:* Cross-encoder reranking boosted top-1 precision significantly over standard hybrid search (**+8.0 pp Hit Rate @ 1**, from 55.0% to 63.0%, and **+0.0725 MRR**, from 0.6277 to 0.7002). Joint query-document cross-attention effectively re-ranked true answers from positions #3-#5 to position #1. Achieved **100.0% HR@5 and 1.0000 MRR on Segment Analysis**. Results archived in `evaluation/results/n4_rerank_benchmark.json`. Walkthrough: `understandings/retrieval.rerank_walkthrough`.
- **N-5 (COMPLETED):** Added explicit architectural notes and disclaimers to `README.md` (Sections 10 and 18) stating plainly that `BAAI/bge-small-en-v1.5` is a general-purpose embedding model rather than a finance-pretrained embedding model, and clearly articulating that FinSight's domain adaptation is implemented at the system level (10-K table-to-markdown parsing, SEC Item metadata filtering, BM25 exact lexical matching for fiscal years/numbers, and cross-encoder reranking). Also updated the official 100-question retrieval benchmark scoreboard and system limitations in `README.md`.

**Dhruv (you)**
- **D-1 (COMPLETED):** Built `generation/build_finetune_dataset.py`. Primary source is `virattt/financial-qa-10K` (`scripts/download_virattt_dataset.py`, now fixed so it no longer deletes the eval benchmark `questions.json`), with a synthetic extractive-QA fallback from `data/processed/chunks.json`. Filters to the 6 ratified tickers and FY2023/FY2024, **de-leaks against Jay's 100-question benchmark** via exact-match + token-Jaccard similarity, renders citation-enforced Llama 3.2 chat examples, and writes a deterministic 95/5 train/val split plus `manifest.json` to `generation/data/`. Covered by `tests/test_build_finetune_dataset.py`.
- **D-2 (COMPLETED):** Built `generation/prompt_templates.py`. Llama 3.2 Instruct chat rendering with a citation-enforced system prompt (every claim tagged `[SOURCE N]`), an explicit low-confidence refusal message, and a canonical `[SOURCE N | TICKER FYyyyy section | Score]` formatter kept in sync with `retrieval.search.format_context_for_prompt()`. Pure/import-light so it is unit-testable without a GPU or vector DB. Covered by `tests/test_prompt_templates.py`.
- **D-3 (COMPLETED):** Executed QLoRA fine-tuning of Llama 3.2 3B Instruct on Colab Tesla T4 GPU via Unsloth. Training config: rank 16, alpha 32, learning rate 2e-4, 1 epoch, 71 total steps (batch size 2 × grad accum 4 = 8), 24,313,856 trainable parameters (0.75%). Training loss converged from 2.3516 to 0.4042; validation loss reached 0.4136. LoRA adapter weights saved at `generation/checkpoints/lora_adapter/`.
- **D-4 (HARNESS COMPLETE; ADAPTER READY):** Built `generation/quantize_gguf.py` — merges the D-3 LoRA adapter into the base model (`merge_and_unload`), then drives llama.cpp's `convert_hf_to_gguf.py` (f16) and `llama-quantize` (Q4_K_M), with `--dry-run` and `--merge-only` modes and tool auto-discovery under `--llama-cpp-dir`. LoRA adapter is in place; ready for merge and quantize to GGUF format for CPU inference. Covered by `tests/test_quantize_gguf.py`.

**Jay**
- **J-3:** Wire experiment tracking (MLflow, self-hosted and free) into Nilay's evaluation harness so each of N-2 through N-4's runs gets logged automatically instead of printed to console only.

---

### Phase 3 — Integration & Evaluation

**Jay**
- **J-4:** Replace the stub endpoints from J-1 with real calls to `retrieve()` (Nilay) and the fine-tuned model (Dhruv) once both are ready.
- **J-5:** Run the full comparative evaluation matrix on the 100-question benchmark: base LLM (no RAG) vs. naive RAG vs. domain-adapted-retrieval-only vs. fine-tuned-only vs. full system.

**Nilay + Dhruv**
- **N/D-1:** Each does error analysis on their own module using J-5's output — categorize failures as retrieval miss vs. generation hallucination vs. numeric error.

---

### Phase 4 — Deployment & Polish

**Jay**
- **J-6:** Finalize containerized deployment to free-tier hosting (HF Spaces or Streamlit Cloud), confirm the GGUF 4-bit model runs within acceptable latency on CPU.
- **J-7:** Add a monitoring/logging dashboard for request latency and error rate.

**All**
- **A-1:** Write setup instructions, architecture diagram, and API docs.

---

### Phase 5 — Report & Presentation

**All**
- **A-2:** Academic report — problem statement, methodology, results (using J-5's comparative matrix), limitations (state the 3B model ceiling and the 5-company/single-sector scope explicitly), future work.
- **A-3:** Portfolio README — results table, architecture diagram, demo link, individual contribution breakdown.

---

## 5. Sequencing Rule of Thumb

Hand out tasks in this order so nobody blocks on incomplete work unnecessarily:

1. **N-1, N-2, J-1, J-2, D-1, D-2** can all start immediately and run in parallel — none depend on anything unfinished.
2. **N-3, N-4** should happen next — they change retrieval behavior.
3. **D-3 (the actual training run)** waits until N-3 and N-4 land, so Dhruv doesn't burn free-tier GPU time retraining against retrieval behavior that's about to change.
4. **J-4, J-5** wait until Nilay's retrieval and Dhruv's fine-tuned model both exist in their final form.
