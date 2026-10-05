# MASTER PROMPT — FinSight Phase 2 Execution (Dhruv side)

> **⚠️ DELETE THIS FILE AFTER THE ENTIRE PHASE IS COMPLETED.**
> This is a temporary hand-off/runbook, not a permanent project document.
> Once D-1, D-3 and D-4 have been executed and their artifacts + doc updates
> are committed, remove `MASTER_PROMPT.md` and its `.gitignore` exception so the
> repository stays clean. Do not leave it in `main`.

---

Paste everything in the fenced block below into the agent running on the GPU machine.

```text
# =============================================================================
# MASTER PROMPT — FinSight: Complete Dhruv's Phase 2 (D-1 + D-3 + D-4)
# =============================================================================
# ROLE
You are an autonomous engineering agent working on the "FinSight" capstone repo.
You own the GENERATION / FINE-TUNING side (author: Dhruv). Your job is to
actually EXECUTE Phase 2 end-to-end on this machine and report honest results.
The code already exists; you are running it, not redesigning it.

# REPO
- Remote: https://github.com/Parmar-Dhruv/finsight.git
- Work on a branch named `dhruv` (or `generation` if it exists).
- Project root contains: retrieval/, generation/, eval/, evaluation/, scripts/,
  api/, demo/, docs/, brain.md, README.md, requirements.txt, .env.example

# OBJECTIVE (finish all of these)
D-1  Build the fine-tuning dataset (primary: virattt/financial-qa-10K;
     synthetic fallback from the indexed corpus).
D-3  Run QLoRA fine-tuning of Llama 3.2 3B Instruct, record the run config.
D-4  Merge the LoRA adapter and export a GGUF Q4_K_M model via llama.cpp.

# HARD FENCE — DO NOT CROSS
- NEVER use evaluation/dataset/questions.json contents as training data.
  The D-1 builder already de-leaks against it; do not disable that guard.
- NEVER commit .env, model weights, adapters (.safetensors/.bin/.gguf), or
  generation/data/. They are gitignored on purpose. Commit code + docs only.
- Do not modify retrieval/ or evaluation/dataset/questions.json.
- Do not claim training/export succeeded unless the artifact exists and loads.

# =============================================================================
# STEP 0 — PRE-FLIGHT
# =============================================================================
0.1 Confirm these files exist (if any are missing, tell the user the Phase-2
    code has not been pushed yet and STOP):
      generation/__init__.py
      generation/prompt_templates.py
      generation/build_finetune_dataset.py
      generation/train_lora.py
      generation/quantize_gguf.py
      generation/requirements.txt
      scripts/download_virattt_dataset.py
0.2 Confirm you are on the correct branch and pull latest:
      git fetch origin
      git checkout dhruv   # or: git checkout -b dhruv origin/dhruv
      git pull
0.3 Record hardware:
      nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
    Decision rule:
      VRAM >= 8 GB  -> local training OK
      VRAM <  8 GB  -> use free Colab/Kaggle T4 (16 GB) instead of this laptop
0.4 Read generation/train_lora.py and generation/quantize_gguf.py so you know
    the CLI flags. Do not rewrite them unless a real bug blocks the run.

# =============================================================================
# STEP 1 — ENVIRONMENT
# =============================================================================
1.1 Create + activate a virtual environment:
      Windows PowerShell:
        python -m venv .venv
        .\.venv\Scripts\Activate.ps1
      Linux/macOS:
        python -m venv .venv && source .venv/bin/activate
1.2 Install dataset/build deps:
      pip install --upgrade pip
      pip install datasets pandas pyarrow requests python-dotenv pytest
1.3 Install training deps (match CUDA; RTX 20/30/40 -> cu121 is fine):
      pip install -r generation/requirements.txt
      pip install torch --index-url https://download.pytorch.org/whl/cu121
    NOTE ON WINDOWS:
      - Unsloth is Linux/WSL only. On native Windows, train_lora.py auto-falls
        back to transformers + peft + trl. That fallback is expected and fine.
      - If bitsandbytes errors on Windows, prefer WSL2 (Ubuntu) or use Colab.
1.4 (Optional, only for retrieval-based eval later) copy credentials:
      copy .env.example .env   (Windows)   /   cp .env.example .env
      Paste QDRANT_URL + QDRANT_API_KEY from Nilay. Do NOT commit .env.
1.5 Run the unit tests to confirm a healthy checkout:
      python -m pytest tests -q
    Expect: 40 passed. If not, STOP and report.

# =============================================================================
# STEP 2 — D-1: BUILD THE FINE-TUNING DATASET
# =============================================================================
2.1 Download the primary dataset:
      python scripts/download_virattt_dataset.py
    Expect: writes evaluation/dataset/financial_qa_10k.json and .jsonl.
    Verify it did NOT delete evaluation/dataset/questions.json.
2.2 Build the training set:
      python generation/build_finetune_dataset.py --source auto
    Expect a summary table and files:
      generation/data/train.jsonl
      generation/data/val.jsonl
      generation/data/manifest.json
2.3 Validate the manifest:
      - source == "virattt/financial-qa-10K"
      - counts.leaks_removed >= 0 (report the number)
      - counts.train + counts.val == counts.kept > 0
      - config.tickers == 6 tickers; config.years == [2023, 2024]
    If kept == 0, fall back:
      python generation/build_finetune_dataset.py --source synthetic
      (and report that the primary download failed).
2.4 Spot-check 3 random rows in train.jsonl:
      - each has a "text" field in Llama 3.2 chat format
      - the assistant turn ends with a [SOURCE n] citation
      - the question does NOT appear in questions.json (leak check)

# =============================================================================
# STEP 3 — D-3: QLoRA FINE-TUNING
# =============================================================================
3.1 ALWAYS dry-run first (no GPU needed, validates data + config):
      python generation/train_lora.py --dry-run
    Expect exit 0 and generation/checkpoints/train_run_config.json written.
3.2 Train. Start conservative for <=8 GB VRAM:
      python generation/train_lora.py --rank 8 --alpha 16 --epochs 1
    For 12–16 GB VRAM you may raise to --rank 16 --alpha 32.
    If you hit OOM: reduce max_seq_length via --max-seq-length 1024 (or 512),
    rank to 8, and epochs to 1. Retry; do not silently swallow OOM.
3.3 Expect the LoRA adapter at:
      generation/checkpoints/lora_adapter/   (adapter_config.json + weights)
    And the report record at:
      generation/checkpoints/train_run_config.json
3.4 Capture for the report: backend (unsloth | transformers+peft), LoRA rank,
    alpha, learning rate, epochs, steps, final train/eval loss, wall-clock time.
3.5 If local training is impossible after honest attempts (no GPU / persistent
    OOM / bitsandbytes failure), STOP and hand off to Colab:
      - New Colab notebook, Runtime = T4 GPU
      - git clone the repo, checkout the branch
      - pip install -r generation/requirements.txt
      - run Steps 2.2 then 3.2 there
      - download generation/checkpoints/ back to the laptop
    Report that training was done on Colab, with the same config numbers.

# =============================================================================
# STEP 4 — D-4: MERGE + GGUF EXPORT
# =============================================================================
4.1 Merge the adapter first (works on CPU/GPU; needs ~6–8 GB RAM):
      python generation/quantize_gguf.py --merge-only
    Expect generation/checkpoints/merged_fp16/ with config.json + weights.
4.2 Get llama.cpp (needed for GGUF):
      git clone https://github.com/ggerganov/llama.cpp
      cd llama.cpp
      # Windows: use a prebuilt release OR build with cmake:
      cmake -B build
      cmake --build build --config Release
      cd ..
4.3 Preview the exact conversion commands, then run them:
      python generation/quantize_gguf.py --dry-run --llama-cpp-dir ./llama.cpp
      python generation/quantize_gguf.py --llama-cpp-dir ./llama.cpp
    Expect generation/checkpoints/gguf/finsight-llama-3.2-3b-q4_k_m.gguf
4.4 Sanity-check the GGUF loads (optional but recommended):
      pip install llama-cpp-python
      python -c "from llama_cpp import Llama; m=Llama(model_path='generation/checkpoints/gguf/finsight-llama-3.2-3b-q4_k_m.gguf', n_ctx=512); print(m('Q: What is FinSight? A:', max_tokens=32))"
    Confirm the file size is reasonable for Q4_K_M of a 3B model (~2 GB).
4.5 If quantization cannot build on this OS, do the convert/quantize step in
    WSL2 or copy merged_fp16/ to a Linux box; report where it was produced.

# =============================================================================
# STEP 5 — VERIFY & DOCUMENT
# =============================================================================
5.1 Re-run tests: python -m pytest tests -q   (expect 40 passed).
5.2 Confirm no junk is staged:
      git status --short
    generation/data/, checkpoints/, .env, *.gguf, *.safetensors MUST NOT appear.
5.3 Update documentation to reflect the ACTUAL runs:
      - brain.md: Phase 2 status; change "D-3 training run pending" to done;
        add a Change Log row with the real config + adapter path + GGUF path.
      - docs/finsight-task-breakdown.md: mark D-3 and D-4 COMPLETED with the
        actual backend, rank/alpha/lr, steps, and artifact locations.
5.4 Delete this runbook once the phase is done (keep the repo clean):
      git rm MASTER_PROMPT.md
      # also remove the "!MASTER_PROMPT.md" exception from .gitignore
5.5 Commit ONLY code + docs (never artifacts/secrets):
      git add generation/ tests/ scripts/ docs/ brain.md .gitignore
      git commit -m "feat(generation): run D-1 dataset build, D-3 QLoRA, D-4 GGUF export"
      git push origin dhruv

# =============================================================================
# FALLBACK / TROUBLESHOOTING
# =============================================================================
- ModuleNotFoundError trl/unsloth: `pip install -r generation/requirements.txt`;
  on Windows the fallback path is intentional.
- CUDA OOM: lower --max-seq-length (1024 -> 512), --rank 8, batch stays 1.
- bitsandbytes CUDA error on Windows: use WSL2 or Colab T4.
- llama-quantize not found: build llama.cpp Release, or use a prebuilt release.
- Download fails: use --source synthetic (needs data/processed/chunks.json).
  If that file is missing too, ask the user to run retrieval/chunk.py first.
- Empty/zero dataset: report which source was attempted and the exact error.

# =============================================================================
# FINAL REPORT (return this to the user)
# =============================================================================
1. Hardware used (GPU, VRAM) and whether training ran locally or on Colab.
2. D-1: source used, train/val counts, leaks_removed, manifest path.
3. D-3: backend, rank/alpha/lr/epochs, steps, loss, wall-clock, adapter path.
4. D-4: merge path, GGUF filename + size, llama.cpp build method.
5. Test result (pytest count).
6. Any task NOT completed, with the exact blocker and what was tried.
Be honest: if something failed or was only partially done, say so explicitly.
```
