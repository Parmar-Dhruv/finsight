# 🧠 FinSight — Brain

> **Living document.** Updated after every significant change.
> Single source of truth for project context, status, architecture, and decisions.

---

## 📌 Project Overview

| Field | Value |
|-------|-------|
| **Name** | FinSight |
| **Type** | Academic Capstone (SGP-II / CEUP301) + Portfolio Deliverable |
| **Timeline** | 12 weeks |
| **Team** | Nilay (Vector DB & Retrieval) · Dhruv (LLM & Fine-tuning) · Jay (Deployment & MLOps) |

**One-line summary:**
A domain-adapted RAG system that answers questions over SEC 10-K financial filings with grounded, citation-backed, and numerically accurate responses — measurably outperforming naive RAG and base-LLM baselines.

---

## 🎯 Problem Being Solved

General-purpose LLMs and off-the-shelf RAG pipelines fail at financial Q&A for three structural reasons:

1. **Numerical/tabular reasoning failure** — naive fixed-size chunking destroys table structure; numbers lose their units, fiscal periods, and qualifiers.
2. **Domain terminology gap** — generic embeddings don't weight financial terms (EBITDA, diluted EPS, impairment) correctly, hurting retrieval precision.
3. **Temporal/source ambiguity** — same company files multiple restated filings; without metadata filtering, stale or contradictory figures get retrieved.

**Project goal:** Build a RAG system that is financially accurate, source-grounded, and citation-backed — and demonstrably better than a naive baseline on a financial QA benchmark.

**Explicitly NOT in scope:** General chatbot, real-time market prediction, trading signals.

---

## 🔒 Locked Project Decisions

These are fixed. Not up for debate mid-project to avoid scope drift and hardware mismatches.

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Corpus scope | **10-K filings, 6 companies (Apple, Microsoft, Alphabet, Amazon, Meta, NVIDIA), Tech sector, FY2023 + FY2024** | Official SEC EDGAR source; Big Tech / AI leaders = consistent 10-K structure; 2 fiscal years gives temporal comparison |
| Fine-tuning compute | Free-tier Colab/Kaggle (T4 GPU, ~16 GB VRAM) | No paid compute budget; QLoRA on 3B is feasible here |
| Deployment target | Free-tier CPU-only (HF Spaces / Streamlit Cloud) | No paid hosting budget |
| Base LLM | **Llama 3.2 3B Instruct** | Small enough for CPU inference; Unsloth VRAM/speed gains; large community base |
| Fine-tuning method | **QLoRA via Unsloth + `peft`** | Only feasible method on available compute; full fine-tuning is explicitly out of scope |
| Deployment model format | **GGUF 4-bit quantized** (via `llama.cpp`) | Required for reasonable latency on CPU-only free hosting |

**Honest trade-off (to state in the report):** A 3B model has a real reasoning ceiling vs. 7–8B models. The project's core claim is: *"domain-adapted retrieval + targeted fine-tuning closes most of the performance gap for a small model on financial QA"* — not that it matches large-model quality outright.

---

## 🏗️ System Architecture

```
                     ┌──────────────────────────────────────────┐
                     │              Data Layer                   │
                     │  SEC EDGAR 10-K filings                   │
                     │  Parsing → chunking → metadata tagging    │
                     └────────────────────┬─────────────────────┘
                                          │
                     ┌────────────────────▼─────────────────────┐
                     │      Vector DB & Retrieval  (Nilay)       │
                     │  Domain-tuned embedding model → Index     │
                     │  Hybrid search: dense + BM25/sparse       │
                     │  Metadata filters: ticker, year, doc type │
                     │  Reranking layer (cross-encoder)          │
                     └────────────────────┬─────────────────────┘
                                          │ top-k chunks
                     ┌────────────────────▼─────────────────────┐
                     │   LLM Integration & Fine-tuning  (Dhruv)  │
                     │  Prompt/context construction              │
                     │  Llama 3.2 3B Instruct (QLoRA fine-tuned) │
                     │  Citation-enforced generation             │
                     └────────────────────┬─────────────────────┘
                                          │ answer + citations
                     ┌────────────────────▼─────────────────────┐
                     │        Deployment & MLOps  (Jay)          │
                     │  FastAPI → Docker → CI/CD (GH Actions)    │
                     │  Monitoring, logging, eval pipeline       │
                     │  Streamlit/Gradio demo frontend           │
                     └───────────────────────────────────────────┘
```

---

## 👥 Team Roles & Responsibilities

### Nilay — Vector Database & Retrieval
- Corpus acquisition via SEC EDGAR API / Kaggle financial datasets
- PDF/HTML parsing with table-aware handling (`unstructured`, `pdfplumber`, `camelot`)
- Chunking strategy: evaluate fixed-size vs. semantic vs. structure-aware
- Embedding model: compare general (`bge-large-en`, `e5-large-v2`) vs. finance-tuned (FinBERT-derived)
- Vector DB: FAISS (local/academic) or Qdrant (free self-hosted, better filtering)
- Hybrid retrieval: dense + sparse (BM25) fusion — exact numeric/ticker matches need lexical search
- Metadata schema & filtering (company, fiscal year, filing type, section)
- Reranking layer (`bge-reranker` or cross-encoder from `sentence-transformers`)
- Retrieval evaluation: Precision@k, Recall@k, MRR

### Dhruv — LLM Integration & Fine-tuning
- Base model: **Llama 3.2 3B Instruct** (locked)
- QLoRA fine-tuning via Unsloth + `peft` on free-tier Colab/Kaggle T4
- Dataset: FinQA, TAT-QA, ConvFinQA (license check required) or synthetic QA from the same corpus
- Prompt engineering: citation enforcement, refuse when retrieval confidence is low
- Context window management for retrieved chunks
- Quantize merged model → GGUF 4-bit (coordinate handoff with Jay early)
- Hallucination mitigation: answer verification against retrieved context, confidence thresholds
- Evaluation comparison: base LLM → naive RAG → domain-adapted RAG → fine-tuned + domain-adapted RAG

### Jay — Deployment & MLOps
- FastAPI endpoints: query, retrieval-only, health check
- Docker + docker-compose for reproducible local multi-service setup
- CI/CD via GitHub Actions (lint, test, build)
- Experiment tracking: MLflow or W&B (free tier)
- Monitoring: request latency, retrieval hit rate, error rates; structured logging
- Eval pipeline automation (must be runnable by all three members)
- Optional: Streamlit/Gradio demo frontend — high portfolio value
- Cost/resource tracking for any paid APIs or cloud compute

> ⚠️ **Dependency risk:** Jay is largely blocked on Nilay's and Dhruv's outputs. Jay must build API scaffolding early against a mock/stub interface so all three can run in parallel from Week 2.

---

## 📅 Development Phases & Status

| Phase | Weeks | Primary Owner | Status |
|-------|-------|---------------|---------|
| Phase 0 — Setup & Planning | 1 | All (joint) | ✅ Done |
| Phase 1 — Data & Baseline | 2–3 | Nilay / Jay | ✅ Done (N-1, J-1, J-2 complete) |
| Phase 2 — Domain Adaptation | 4–6 | Nilay + Dhruv | 🔄 In Progress (N-2..N-5 ✅ done; D-1..D-4 harnesses ✅ done; D-3 training run pending free-tier T4) |
| Phase 3 — Integration & Evaluation | 7–8 | Jay / All | ⬜ Not Started |
| Phase 4 — Deployment & Polish | 9–10 | Jay | ⬜ Not Started |
| Phase 5 — Report & Presentation | 11–12 | All (joint) | ⬜ Not Started |

### Phase 0 — Setup & Planning (Week 1)
- [x] Finalize scope: single domain/sector, specific filing types, date range (6 Big Tech companies, FY2023-2024 10-K filings)
- [x] Define evaluation benchmark **before** building the system (100 QA pairs + gold answers in `evaluation/dataset/questions.json`)
- [x] Set up shared repo, Python env, dependency manager, communication cadence
- [x] Agree on inter-component interface contracts (JSON schemas: retrieval → LLM → API)
- **Deliverable:** Project charter, repo scaffold, interface contracts

### Phase 1 — Data & Baseline (Weeks 2–3)
- [x] Nilay: corpus acquisition + parsing; naive baseline (fixed-chunk + generic embedding + brute-force NumPy cosine)
- [ ] Dhruv: baseline generation (base LLM + naive RAG context; no fine-tuning yet)
- [x] Jay: scaffold FastAPI service calling naive pipeline end-to-end; Docker + docker-compose (J-1, J-2)
- **Deliverable:** Working naive RAG baseline, end-to-end, with baseline metrics recorded

### Phase 2 — Domain Adaptation (Weeks 4–6)
- [x] Nilay: structure-aware chunking, BGE embeddings, hybrid retrieval (BM25+RRF), metadata filtering, cross-encoder reranking, BGE transparency note (N-2..N-5 completed & benchmarked)
- [x] Dhruv: build/curate fine-tuning dataset (D-1 ✅); citation-enforced prompting + refusal template (D-2 ✅)
- [x] Dhruv: QLoRA training harness (D-3 ✅ `generation/train_lora.py`); GGUF merge+export harness (D-4 ✅ `generation/quantize_gguf.py`)
- [ ] Dhruv: **run** D-3 training on free-tier T4, then D-4 export (pending compute)
- [ ] Jay: integrate experiment tracking (MLflow J-3); automate eval pipeline per iteration
- **Deliverable:** Domain-adapted retrieval + fine-tuned generation, each independently benchmarked vs. Phase 1

### Phase 3 — Integration & Evaluation (Weeks 7–8)
- Jay: full pipeline integration (adapted retrieval + fine-tuned generation + API)
- All: comparative evaluation matrix — base LLM / naive RAG / adapted-retrieval-only / fine-tuned-only / full system
- Nilay + Dhruv: error analysis on own modules (retrieval miss vs. generation hallucination vs. numeric error)
- **Deliverable:** Full metrics table, error analysis writeup

### Phase 4 — Deployment & Polish (Weeks 9–10)
- Jay: finalize containerized deployment (one-command setup), monitoring dashboard
- Optional: Streamlit/Gradio demo frontend with citations shown
- All: architecture diagram, setup instructions, API docs
- **Deliverable:** Deployed/deployable system, demo video or live demo, full documentation

### Phase 5 — Report & Presentation (Weeks 11–12)
- Academic report: problem → related work → methodology → results → limitations → future work
- Portfolio artifacts: README with results table, architecture diagram, demo link/video, individual contributions
- **Deliverable:** Final report, presentation slides, polished GitHub repo

---

## 📊 Measurable Objectives

| # | Objective | Success Criterion |
|---|-----------|-------------------|
| O1 | Domain-adapted retrieval | Precision@5 / Recall@5 improves over naive baseline |
| O2 | Grounded generation | ≥90% of generated answers cite the correct source chunk |
| O3 | Reduced hallucination | Lower hallucination rate than base LLM (no-RAG), via RAGAS faithfulness |
| O4 | Deployable system | End-to-end latency < 5s p95; working API + monitoring |
| O5 | Reproducibility | Documented, containerized, one-command setup |

---

## 🛠️ Tech Stack

| Layer | Tool/Library | Status |
|-------|-------------|--------|
| Data source | SEC EDGAR HTML 10-K filings (6 companies × 2 FY = 12 filings) | ✅ Acquired (`retrieval/download.py`) |
| Parsing | `beautifulsoup4` — HTML cleaning, table→Markdown, section extraction + Item 8 cross-ref | ✅ Done (`retrieval/parse.py`) |
| Chunking | Recursive structure-aware splitter (1500-char target, 200-char overlap, 3,504 chunks) | ✅ Done (`retrieval/chunk.py`) |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim, FastBertEmbedder pure NumPy execution) | ✅ Done (`retrieval/embed.py`) |
| Vector DB | **Qdrant Cloud** (3,504 vectors across 6 companies, payload indices: ticker/year/section) | ✅ Indexed (`retrieval/index.py`) |
| Retrieval API | `retrieve()` + `format_context_for_prompt()` | ✅ Done (`retrieval/search.py`) |
| Sparse retrieval | `rank-bm25` (hybrid BM25Okapi + Reciprocal Rank Fusion) | ✅ Done (`retrieval/hybrid.py`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (`FastCrossEncoder` NumPy/safetensors) | ✅ Done (`retrieval/rerank.py`) |
| Naive baseline | Fixed 2048-char windows (1,641 chunks), brute-force NumPy cosine, benchmarked on 10 queries | ✅ Done (`eval/naive_baseline.py`) |
| LLM prompt layer | `generation/prompt_templates.py` (citation-enforced + refusal, Llama 3.2 chat format) | ✅ Done (`generation/`) |
| Fine-tuning dataset | `generation/build_finetune_dataset.py` (virattt/financial-qa-10K primary + synthetic fallback, leak-guarded) | ✅ Done (`generation/`) |
| LLM | Llama 3.2 3B Instruct | ✅ Locked (`unsloth/Llama-3.2-3B-Instruct`) |
| Fine-tuning | QLoRA via Unsloth + `peft` | ✅ Trained on Colab T4 (rank 16, alpha 32, lr 2e-4, 71 steps, final eval loss 0.4136); adapter saved at `generation/checkpoints/lora_adapter` |
| Deployment inference | GGUF 4-bit via `llama.cpp` | ✅ Script ready (`generation/quantize_gguf.py`); adapter ready |
| API | FastAPI (stub endpoints / real integration) | ✅ Done (Jay) |
| Experiment tracking | MLflow self-hosted (pending J-3) | ⬜ Jay |
| Containerization | Docker + docker-compose | ✅ Done (Jay) |
| CI/CD | GitHub Actions | ⬜ Jay |
| Demo UI | Minimal Streamlit test UI (`demo/app.py`) | ✅ Base test UI (frontend base) |

---

## 📏 Evaluation Metrics

**Retrieval:**
- Precision@k, Recall@k, MRR — against labeled query → relevant-chunk mapping

**Generation:**
- RAGAS faithfulness/groundedness
- Answer correctness: exact match for numeric answers; semantic similarity for descriptive
- Citation accuracy: does the cited source actually contain the claimed fact?

**System:**
- End-to-end latency: p50 / p95
- Cost per query (if any paid APIs used)

> ⚠️ Benchmark set (100–200 QA pairs minimum) must be defined **before Phase 2 begins** — not retroactively — or evaluation criteria will be shaped to fit whatever the system produces.

---

## ⚠️ Known Risks & Constraints

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Compute cap (3B, free GPU) | Caps reasoning ability | Frame contribution as "closing the gap," not "matching large models"; state plainly in report |
| Licensing (FinQA, TAT-QA) | Legal/portfolio risk | Check licenses before redistribution or commercial framing |
| Small benchmark (<50 QA pairs) | Unreliable precision/recall diffs | Budget real time for benchmark construction |
| Non-zero hallucination rate | Credibility of results | Report honestly; don't cherry-pick examples |
| Scope creep | Timeline overrun | Narrow aggressively to one sector in Phase 0 |
| Jay blocked on others | Idle time | Jay builds API scaffold early against mock interfaces |

---

## 📦 Deliverables Checklist

**Phase 0**
- [x] Project charter & interface contracts
- [x] Repo scaffold, Python env (`Python 3.14.6`, `.venv`), full folder structure
- [x] `.env.example` template + `.gitignore`

**Phase 1 (Nilay)**
- [x] Corpus acquired — 12 × 10-K filings (6 companies: AAPL, AMZN, GOOGL, META, MSFT, NVDA × FY2023 + FY2024) via `retrieval/download.py`
- [x] Parsing pipeline — `retrieval/parse.py` (HTML → Markdown tables → section extraction + Item 8 cross-ref)
- [x] Structure-aware chunking — `retrieval/chunk.py` (recursive, 1500-char target, 3,504 total chunks)
- [x] BGE embedding engine — `retrieval/embed.py` (`BAAI/bge-small-en-v1.5`, FastBertEmbedder pure NumPy execution, zero DLL locks)
- [x] Qdrant Cloud index — `retrieval/index.py` (3,504 points: AAPL 354, AMZN 451, GOOGL 593, META 842, MSFT 576, NVDA 688)
- [x] Public retrieval API — `retrieval/search.py` (`retrieve()`, `format_context_for_prompt()`, `evaluate_retrieval_quality()`)
- [x] **N-2: Expand eval to 100-question set** — Imported `evaluation/dataset/questions.json` from Jay's `jay` branch. Built `eval/benchmark.py` with section normalization and evaluated both Naive Baseline and Domain-Adapted Retrieval across all 100 questions. Results: Hit Rate@1: 13.0% vs 69.0% (+56 pp) / Hit Rate@3: 28.0% vs 82.0% (+54 pp) / Hit Rate@5: 34.0% vs 91.0% (+57 pp) / MRR: 0.2128 vs 0.7633 (+0.5505). Granular results saved in `evaluation/results/n2_retrieval_benchmark.json`. Walkthrough: `understandings/eval.benchmark_walkthrough`.
- [x] **N-3: BM25 + reciprocal rank fusion hybrid search** — Built `retrieval/hybrid.py` (`BM25Index` + `reciprocal_rank_fusion`), integrated into `retrieve(hybrid=True)`. Evaluated on full 100-question benchmark: Hybrid HR@5: 77.0% (+43 pp over naive); achieved **+6.7 pp gain in Temporal Comparison** (86.7% vs 80.0% HR@5, MRR 0.6944 vs 0.6833) overcoming neural number blindness on fiscal years. Tabular and Segment analysis maintained 100% HR@5. Results archived in `evaluation/results/n3_hybrid_benchmark.json`. Walkthrough: `understandings/retrieval.hybrid_walkthrough`.
- [x] **N-4: Cross-encoder reranking** — Built `retrieval/rerank.py` (`FastCrossEncoder` pure NumPy/safetensors forward pass of `cross-encoder/ms-marco-MiniLM-L-6-v2`), integrated into `retrieve(rerank=True)`. Evaluated on full 100-question benchmark: HR@1 boosted to 63.0% (+8.0 pp over hybrid), MRR improved to 0.7002 (+11.5% gain over hybrid), Segment Analysis reached 100% HR@5 and 1.0000 MRR. Results archived in `evaluation/results/n4_rerank_benchmark.json`. Walkthrough: `understandings/retrieval.rerank_walkthrough`.
- [x] **N-5: README note on BGE-small not being finance-tuned** — Added transparent documentation and architecture callout in `README.md` (Sections 10 and 18) stating that `BAAI/bge-small-en-v1.5` is a general-purpose embedding model and explaining how domain adaptation is achieved structurally. Updated official 100-question retrieval benchmark scoreboard in `README.md`.

**Phase 1 (Jay)**
- [x] J-1: FastAPI stub endpoints (`/query`, `/retrieve`, `/health`)
- [x] J-2: Dockerfile + docker-compose

**Phase 2 onwards**
- [x] **D-1:** Fine-tuning dataset builder — `generation/build_finetune_dataset.py` (virattt/financial-qa-10K primary + synthetic corpus fallback; 6-ticker/FY23-24 filter; benchmark leak guard; deterministic train/val split + manifest)
- [x] **D-2:** Citation-enforced prompt templates — `generation/prompt_templates.py` (Llama 3.2 chat format, `[SOURCE N]` enforcement, explicit refusal, shared source-header contract)
- [x] **D-3:** QLoRA training harness — `generation/train_lora.py` (Unsloth primary + transformers/peft fallback, 4-bit, dry-run mode, run-config logging)
- [x] **D-4:** GGUF export harness — `generation/quantize_gguf.py` (LoRA merge + llama.cpp f16→Q4_K_M, dry-run mode)
- [ ] **D-3 run:** Execute QLoRA fine-tuning on free-tier T4 (record config/metrics)
- [ ] **D-4 run:** Merge + export GGUF Q4_K_M after adapter exists
- [ ] Fine-tuned generation module + benchmark (Dhruv)
- [ ] Full integrated system + comparative evaluation matrix (Phase 3)
- [ ] Error analysis report (Phase 3)
- [ ] Deployed/deployable containerized system (Phase 4)
- [ ] Demo — video or live (Phase 4)
- [ ] Final academic report (Phase 5)
- [ ] Portfolio-ready GitHub repo (Phase 5)

---

## 📁 Repository Structure (Planned)

```
finsight/
├── data/                    # Raw + processed SEC filings
├── retrieval/               # Nilay: chunking, embeddings, vector DB, hybrid search, reranking
├── generation/              # Dhruv: prompt templates, fine-tuning scripts, model checkpoints
├── api/                     # Jay: FastAPI app, Docker configs, health checks
├── eval/                    # Evaluation harness, benchmark QA pairs, metric scripts
├── experiments/             # MLflow/W&B experiment logs
├── demo/                    # Streamlit/Gradio frontend
├── docs/                    # Architecture diagrams, API docs, setup guides, handoff contracts
├── .github/workflows/       # CI/CD pipelines (GitHub Actions)
├── .env.example             # Template for required environment variables (Qdrant, SEC EDGAR)
├── docker-compose.yml
├── requirements.txt
├── README.md
└── brain.md                 # ← This file
```

---

## 📋 Month 1–2 Learning Checklist (All Members — Independently)

**Rule:** Each person completes this independently. No splitting fundamentals — interviewers ask individuals.

**Primary resource:** Andrej Karpathy — *Neural Networks: Zero to Hero* (YouTube)

**Dhruv / Nilay / Jay — each independently:**
- [ ] Micrograd: watch, implement, rebuild from memory, push to GitHub
- [ ] Makemore Parts 1–2 (bigram + MLP language model) — implement + extend on finance text
- [ ] Makemore Parts 3–4 (activations/batchnorm + manual backprop) — do backprop by hand on paper first
- [ ] Makemore Part 5 (WaveNet) — implement
- [ ] Checkpoint: NumPy/PyTorch NN library, MNIST >97%, no copying
- [ ] Let's Build GPT — self-attention, MHA, positional encoding, LN, residuals; rebuild from memory
- [ ] LoRA: read paper, implement rank decomposition yourself, fine-tune on finance corpus
- [ ] Self-test: whiteboard backprop for 2-layer MLP + explain attention — cold, no notes

**Weekly anti-bluffing check:** Each person explains what they built to the other two — cold, no notes.

---

## 🚦 Final Gate (Per Person — Before Calling Done)

- [ ] Derive backprop by hand, no notes
- [ ] Explain self-attention and positional encoding without lookups
- [ ] Explain what LoRA does **mathematically** (not just "it's cheaper")
- [ ] Explain all three Month 3 components (retrieval, generation/quantization, eval/deployment) — not just your own
- [ ] Every project on GitHub with a README in your own words

---

## 📝 Change Log

| Date | Change | By |
|------|--------|----|
| 2026-09-16 | Repo initialized; `.gitignore` created | Nilay |
| 2026-09-16 | `brain.md` created | Antigravity |
| 2026-09-16 | **Phase 0 complete:** Python 3.14.6, `.venv`, full folder scaffold, `requirements.txt`, `.gitignore` | Nilay + Antigravity |
| 2026-09-21 | BGE embedding engine — `retrieval/embed.py` (batch encoding, L2 normalisation) | Nilay + Antigravity |
| 2026-09-21 | Qdrant indexing — `retrieval/index.py` (local + cloud, HNSW, payload filters) | Nilay + Antigravity |
| 2026-09-22 | Public retrieve API + context formatter — `retrieval/search.py`; IR eval harness `evaluate_retrieval_quality()` | Nilay + Antigravity |
| 2026-09-22 | Cloud ingestion to Qdrant Cloud cluster; `docs/HANDOFF_DHRUV.md` | Nilay + Antigravity |
| 2026-09-22 | `.env.example` template; env setup docs | Nilay + Antigravity |
| 2026-09-25 | **N-1 complete:** `eval/naive_baseline.py` — naive baseline pipeline (fixed 2048-char chunks, no structure, no metadata filter, NumPy cosine search). Results vs domain-adapted on 8-query benchmark: Hit Rate@1: 25% vs 62.5% (+37.5 pp) / Hit Rate@3: 25% vs 75% (+50 pp) / Hit Rate@5: 37.5% vs 87.5% (+50 pp) / MRR: 0.2812 vs 0.6917 (+41 MRR-pp). Pushed to `nilay` branch (commit `efa44a0`). | Nilay + Antigravity |
| 2026-09-25 | `docs/finsight-task-breakdown.md` added — atomic task reference for all three members | Nilay + Antigravity |
| 2026-09-25 | `brain.md` updated to reflect current actual status | Nilay + Antigravity |
| 2026-10-04 | **Corpus expanded to 6 companies:** Added NVIDIA (NVDA FY2023 & FY2024 10-Ks); built automated SEC EDGAR downloader `retrieval/download.py`; updated `retrieval/parse.py` with Item 8 consolidated financial statement cross-referencing. | Nilay + Antigravity |
| 2026-10-04 | Upgraded `retrieval/embed.py` with pure NumPy `FastBertEmbedder` over cached `bge-small-en-v1.5` safetensors to bypass Windows PyTorch DLL locks and achieve instant sub-second cold starts. | Nilay + Antigravity |
| 2026-10-04 | Qdrant Cloud refreshed with all 3,504 vectors & payloads across all 6 companies (NVDA: 688 points). | Nilay + Antigravity |
| 2026-10-04 | **N-1 re-benchmarked across all 6 companies:** `eval/naive_baseline.py` evaluated across 1,641 chunks on 10 queries (AAPL, MSFT, AMZN, GOOGL, META, NVDA): Hit Rate@1: 20% vs 70% (+50 pp) / Hit Rate@3: 30% vs 80% (+50 pp) / Hit Rate@5: 40% vs 80% (+40 pp) / MRR: 0.2750 vs 0.7333 (+45.8 MRR-pp). | Nilay + Antigravity |
| 2026-10-04 | **N-2 complete:** Imported `evaluation/dataset/questions.json` from `jay` branch. Built `eval/benchmark.py` evaluating Naive Baseline vs Domain-Adapted Retrieval across all 100 questions. Hit Rate@1: 13.0% vs 69.0% (+56 pp) / Hit Rate@3: 28.0% vs 82.0% (+54 pp) / Hit Rate@5: 34.0% vs 91.0% (+57 pp) / MRR: 0.2128 vs 0.7633 (+0.5505). Full results archived in `evaluation/results/n2_retrieval_benchmark.json`. Walkthrough: `understandings/eval.benchmark_walkthrough`. | Nilay + Antigravity |
| 2026-10-04 | **N-3 complete:** Implemented BM25 sparse lexical retrieval (`retrieval/hybrid.py`) and Reciprocal Rank Fusion. Integrated `retrieve(hybrid=True)`. Benchmarked 100 questions: Hybrid HR@5 77.0% (+43 pp over naive); +6.7 pp gain on Temporal Comparisons (86.7% vs 80.0%) fixing neural number blindness between fiscal years. Archived in `evaluation/results/n3_hybrid_benchmark.json`. Walkthrough: `understandings/retrieval.hybrid_walkthrough`. | Nilay + Antigravity |
| 2026-10-05 | **D-2 complete:** Added `generation/prompt_templates.py` — Llama 3.2 Instruct chat rendering, citation-enforced system prompt (`[SOURCE N]`), explicit low-confidence refusal, and a canonical `[SOURCE N | TICKER FYyyyy section | Score]` formatter kept in sync with `retrieval.search.format_context_for_prompt()`. | Dhruv |
| 2026-10-05 | **D-1 complete:** Added `generation/build_finetune_dataset.py` — virattt/financial-qa-10K primary source with synthetic extractive-QA fallback from the indexed corpus; filters to the 6 ratified tickers and FY2023/FY2024; de-leaks against the 100-question benchmark (exact + token-Jaccard); renders chat training examples; deterministic 95/5 train/val split + `manifest.json`. Fixed `scripts/download_virattt_dataset.py` to stop deleting `questions.json` (the eval benchmark) and exposed `download_dataset()`. Added `tests/` suite. | Dhruv |
| 2026-10-05 | **D-3/D-4 harnesses complete:** Added `generation/train_lora.py` (QLoRA via Unsloth with transformers+peft fallback, 4-bit, `--dry-run`, run-config export) and `generation/quantize_gguf.py` (LoRA merge + llama.cpp f16→Q4_K_M, `--dry-run`/`--merge-only`). Both validated via dry-run; 40 unit tests passing. Actual training/export runs pending free-tier T4 compute. | Dhruv |
| 2026-10-05 | **Repo hygiene + base UI:** Added `LICENSE` (MIT), GitHub Actions CI (`.github/workflows/ci.yml` — unit tests + byte-compile), minimal Streamlit test UI (`demo/app.py`), `tests/` suite + `conftest.py`, and `MASTER_PROMPT.md` runbook for the GPU execution handoff. Aligned README to the 6-company corpus scope and corrected stale paths/vector counts in `docs/HANDOFF_DHRUV.md`. | Dhruv |
| 2026-10-05 | **D-3 QLoRA fine-tuning run completed:** Trained Llama 3.2 3B Instruct on Colab Tesla T4 GPU via Unsloth (r=16, alpha=32, lr=2e-4, 1 epoch, 71 steps, batch size 2 x 4 grad accum = 8, 24.3M trainable params). Final training loss dropped from 2.3516 to 0.4042; validation eval_loss reached 0.4136. Trained LoRA adapter extracted and placed at `generation/checkpoints/lora_adapter/`. All 40 unit tests passing. | Dhruv + Nilay |



