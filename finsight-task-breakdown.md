# FinSight — Problem Statement & Phase Task Breakdown

**Team:** Nilay (Vector DB & Retrieval) · Dhruv (LLM Integration & Fine-tuning) · Jay (Deployment & MLOps)
**Ratified scope:** 5 companies (AAPL, MSFT, AMZN, GOOGL, META), US Public Tech sector, 10-K only, FY2023-2024

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
| Nilay | `nilay` | Full retrieval pipeline: parsing, structure-aware chunking, BGE embeddings, Qdrant index, metadata filtering, `retrieve()`/`format_context_for_prompt()` API. Evaluated on only 8 queries. No naive baseline recorded. No hybrid/BM25. No reranking |
| Jay | `jay` | A 100-question gold-answer benchmark (`evaluation/dataset/questions.json`) covering all 5 companies, both fiscal years, 6 question types. **No API, Docker, or CI work yet** |
| Dhruv | `dhruv` | Branch exists, no commits beyond initial scaffold. Nothing started |

**Immediate coordination fix needed:** Nilay's evaluation harness and Jay's 100-question benchmark are not connected. Task N-2 below wires them together — this closes the "eval set too small" gap without anyone having to build a new dataset.

---

## 4. Phase Task Breakdown — Small, Atomic Tasks

### Phase 1 — Data & Baseline *(retroactive completion)*

**Nilay**
- **N-1:** Build a naive baseline pipeline — fixed 512-token chunks, no structure-awareness, no metadata filtering, same embedding model. Run it through the existing `evaluate_retrieval_quality()` function. Output: one table with Hit Rate@1/3/5 and MRR for the naive pipeline, to sit next to the existing domain-adapted numbers.

**Jay**
- **J-1:** Scaffold a FastAPI service with three stub endpoints: `/query`, `/retrieve`, `/health`. They can return hardcoded/mock responses for now — the goal is a running service Dhruv and Nilay can point real logic at later, not full functionality yet.
- **J-2:** Add a `Dockerfile` and minimal `docker-compose.yml` so the stub API runs in a container.

---

### Phase 2 — Domain Adaptation *(in progress)*

**Nilay**
- **N-2:** Import `evaluation/dataset/questions.json` from Jay's `jay` branch. Convert its 100 questions into the query→gold-chunk format your harness expects. Re-run `evaluate_retrieval_quality()` on this expanded set for both the naive baseline (N-1) and the current domain-adapted pipeline. Output: updated before/after table on the full 100-question set.
- **N-3:** Implement BM25 sparse retrieval over the existing chunk corpus using `rank-bm25` (already a dependency, currently unused). Fuse with dense retrieval scores using reciprocal rank fusion. Re-run evaluation with hybrid search on vs. off.
- **N-4:** Add a cross-encoder reranking step on top-k candidates before they're returned from `retrieve()`, using `sentence-transformers` (already a dependency). Re-run evaluation with reranking on vs. off.
- **N-5:** Add one paragraph to the README stating plainly that `BAAI/bge-small-en-v1.5` is a general-purpose embedding model, not a finance-tuned one — so this doesn't silently read as "domain-tuned" to a grader.

**Dhruv (you)**
- **D-1:** Build the fine-tuning QA dataset from the same 5-company corpus Nilay indexed. **Do not reuse any of Jay's 100 benchmark questions as training data** — they must stay held out for evaluation, or your eval numbers will be inflated by data leakage.
- **D-2:** Write the citation-enforced prompt template, using the `[SOURCE N | TICKER FYyyyy section | Score]` format already produced by `format_context_for_prompt()`. Include an explicit refusal/low-confidence instruction for when retrieved context doesn't support an answer.
- **D-3:** Run QLoRA fine-tuning via Unsloth on Llama 3.2 3B Instruct, free-tier Colab/Kaggle T4. Record the training config (rank, alpha, learning rate, steps) for the report.
  - **Hold this task until N-3 and N-4 are done** — training against retrieval behavior that's about to change wastes your compute budget.
- **D-4:** Quantize the fine-tuned model to GGUF (4-bit) via `llama.cpp`. Confirm the output format with Jay before handoff.

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
