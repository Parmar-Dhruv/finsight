"""
================================================================================
FinSight — Generation & Fine-Tuning Layer
D-1: Fine-Tuning Dataset Builder (generation/build_finetune_dataset.py)
================================================================================

Task D-1 (docs/finsight-task-breakdown.md):
  "Build the fine-tuning QA dataset from the same 5-company corpus Nilay
   indexed. Do NOT reuse any of Jay's 100 benchmark questions as training data
   -- they must stay held out for evaluation, or your eval numbers will be
   inflated by data leakage."

Data sources (in priority order)
--------------------------------
1. PRIMARY : `evaluation/dataset/financial_qa_10k.jsonl` (or `.json`), the
   virattt/financial-qa-10K dataset downloaded by
   scripts/download_virattt_dataset.py. Fields: question, answer, context,
   ticker, filing.
2. FALLBACK: synthetic, extractive QA templated from the indexed corpus at
   `data/processed/chunks.json` (produced by retrieval/chunk.py). Lower
   fidelity, used only when the primary dataset is unavailable.

Both paths are:
  * filtered to the ratified tickers and FY2023/FY2024 10-K filings,
  * de-leaked against the 100-question benchmark (`questions.json`),
  * rendered into the citation-enforced Llama 3.2 chat format from
    generation/prompt_templates.py,
  * split deterministically into train/val and saved with a manifest.

Outputs (generation/data/):
  train.jsonl, val.jsonl, manifest.json

Usage:
    python generation/build_finetune_dataset.py --source auto
    python generation/build_finetune_dataset.py --source primary
    python generation/build_finetune_dataset.py --source synthetic --max-synthetic 1500
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure project root is importable when run as a script.
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from generation.prompt_templates import (  # noqa: E402
    build_training_example,
    format_source_block,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths & ratified scope ───────────────────────────────────────────────────

DATASET_DIR = BASE_DIR / "evaluation" / "dataset"
BENCHMARK_PATH = DATASET_DIR / "questions.json"
PRIMARY_JSONL = DATASET_DIR / "financial_qa_10k.jsonl"
PRIMARY_JSON = DATASET_DIR / "financial_qa_10k.json"
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
OUT_DIR = BASE_DIR / "generation" / "data"

RATIFIED_TICKERS = ("AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA")
RATIFIED_YEARS = (2023, 2024)

YEAR_RE = re.compile(r"(19|20)\d{2}")
WORD_RE = re.compile(r"[a-z0-9]+")


# ─── Part 1: Text normalization & leak detection ──────────────────────────────

def normalize_text(value: str) -> str:
    """Lowercases and collapses to space-separated alphanumeric tokens."""
    return " ".join(WORD_RE.findall((value or "").lower()))


def token_set(value: str) -> set:
    """Returns the set of normalized tokens in a string."""
    return set(WORD_RE.findall((value or "").lower()))


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two token sets (0.0 when both empty)."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class LeakIndex:
    """Holds the held-out benchmark and decides whether a record leaks."""

    def __init__(self, benchmark: Sequence[Dict[str, Any]], threshold: float = 0.8) -> None:
        self.threshold = threshold
        self.question_norms: set = set()
        self.question_tokens: List[set] = []
        self.answer_norms: set = set()

        for q in benchmark:
            question = q.get("question", "")
            answer = q.get("gold_answer", "")
            self.question_norms.add(normalize_text(question))
            self.question_tokens.append(token_set(question))
            if answer and len(normalize_text(answer)) >= 20:
                self.answer_norms.add(normalize_text(answer))

    def is_leak(self, question: str, answer: str = "") -> bool:
        norm_q = normalize_text(question)
        if not norm_q:
            return False
        if norm_q in self.question_norms:
            return True

        q_tokens = token_set(question)
        for benchmark_tokens in self.question_tokens:
            if jaccard(q_tokens, benchmark_tokens) >= self.threshold:
                return True

        norm_a = normalize_text(answer)
        if norm_a and len(norm_a) >= 20 and norm_a in self.answer_norms:
            return True
        return False


def load_benchmark(path: Path = BENCHMARK_PATH) -> List[Dict[str, Any]]:
    """Loads the held-out 100-question benchmark (empty list if absent)."""
    if not path.exists():
        logger.warning("Benchmark not found at %s; leak detection disabled.", path)
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("questions", [])


# ─── Part 2: Filing-year normalization ────────────────────────────────────────

def parse_filing_year(record: Dict[str, Any]) -> Optional[int]:
    """Extracts a 4-digit fiscal year from the record's filing/fiscal_year field."""
    for key in ("fiscal_year", "filing", "year", "source_doc"):
        raw = record.get(key)
        if raw is None:
            continue
        if isinstance(raw, int):
            if 2000 <= raw <= 2099:
                return raw
            continue
        match = YEAR_RE.search(str(raw))
        if match:
            return int(match.group(0))
    return None


def record_section(record: Dict[str, Any]) -> str:
    """Returns the filing section label, defaulting to a generic 10-K label."""
    section = record.get("section") or record.get("source_section")
    return str(section).strip() if section else "10-K"


# ─── Part 3: Primary (virattt/financial-qa-10K) source ────────────────────────

def load_primary_records() -> List[Dict[str, Any]]:
    """Loads the primary dataset from disk (JSONL preferred, then JSON)."""
    if PRIMARY_JSONL.exists():
        records: List[Dict[str, Any]] = []
        with PRIMARY_JSONL.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.info("Loaded %d primary records from %s", len(records), PRIMARY_JSONL)
        return records

    if PRIMARY_JSON.exists():
        records = json.loads(PRIMARY_JSON.read_text(encoding="utf-8"))
        logger.info("Loaded %d primary records from %s", len(records), PRIMARY_JSON)
        return records

    return []


def download_primary_records() -> List[Dict[str, Any]]:
    """Attempts a network download via the shared downloader script."""
    try:
        from scripts.download_virattt_dataset import download_dataset

        records = download_dataset()
        logger.info("Downloaded %d primary records from Hugging Face.", len(records))
        return records
    except Exception as exc:  # pragma: no cover - network/environment dependent
        logger.warning("Primary dataset download unavailable: %s", exc)
        return []


def normalize_primary_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Filters + normalizes one primary record; returns None to drop it."""
    ticker = str(record.get("ticker", "")).strip().upper()
    if ticker not in RATIFIED_TICKERS:
        return None

    year = parse_filing_year(record)
    if year not in RATIFIED_YEARS:
        return None

    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    context = str(record.get("context", "")).strip()
    if not question or not answer or not context:
        return None

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "ticker": ticker,
        "fiscal_year": year,
        "section": record_section(record),
        "source": "virattt/financial-qa-10K",
    }


# ─── Part 4: Synthetic fallback (templated extractive QA from corpus) ─────────

_NUMERIC_FACT_RE = re.compile(
    r"(?P<label>[A-Z][A-Za-z0-9 ,\-']{4,70}?)\s+"
    r"(?:was|were|totaled|totalled|reached|increased|decreased|grew|of|by)\s+"
    r"(?P<value>\$?\d[\d,\.]*\s*(?:billion|million|trillion|%)?)",
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _extractive_question(label: str, ticker: str, year: int) -> str:
    label = re.sub(r"\s+", " ", label).strip(" ,-")
    return f"What does {ticker} report regarding {label} in its FY{year} 10-K?"


def synthesize_from_chunks(
    chunks_path: Path = CHUNKS_PATH,
    max_records: int = 1500,
) -> List[Dict[str, Any]]:
    """
    Builds low-fidelity extractive QA pairs from the indexed chunk corpus.

    This is a safety net, not the primary training data: the benchmark's gold
    answers and the fine-tuning questions must stay held out from one another,
    and templated pairs are noisier than the virattt dataset.
    """
    if not chunks_path.exists():
        logger.warning("Chunk corpus not found at %s; run retrieval/chunk.py first.", chunks_path)
        return []

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = []
    seen_questions: set = set()

    for chunk in chunks:
        ticker = str(chunk.get("ticker", "")).strip().upper()
        year = chunk.get("fiscal_year")
        if ticker not in RATIFIED_TICKERS or year not in RATIFIED_YEARS:
            continue

        text = chunk.get("text", "")
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            match = _NUMERIC_FACT_RE.search(sentence)
            if not match:
                continue

            label = match.group("label")
            question = _extractive_question(label, ticker, int(year))
            norm = normalize_text(question)
            if not norm or norm in seen_questions:
                continue
            seen_questions.add(norm)

            records.append(
                {
                    "question": question,
                    "answer": sentence.strip(),
                    "context": sentence.strip(),
                    "ticker": ticker,
                    "fiscal_year": int(year),
                    "section": record_section(chunk),
                    "source": "synthetic-corpus",
                }
            )
            if len(records) >= max_records:
                logger.info("Synthetic fallback capped at %d records.", max_records)
                return records

    logger.info("Synthesized %d records from the chunk corpus.", len(records))
    return records


# ─── Part 5: Example rendering, leak filtering, splitting & persistence ───────

def build_context_block(record: Dict[str, Any]) -> str:
    """Wraps a single training context using the canonical source-header format."""
    body = format_source_block(
        ticker=record["ticker"],
        fiscal_year=record["fiscal_year"],
        section=record["section"],
        text=record["context"],
        score=1.0,
        index=1,
    )
    return (
        "=== RETRIEVED 10-K CONTEXT (1 passage) ===\n\n"
        f"{body}"
        "========================================================="
    )


def build_examples(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Renders normalized records into Llama 3.2 chat training examples."""
    examples: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        example = build_training_example(
            question=record["question"],
            context_block=build_context_block(record),
            answer=record["answer"],
            source_indices=(1,),
        )
        examples.append(
            {
                "id": f"ft_{idx:06d}",
                "ticker": record["ticker"],
                "fiscal_year": record["fiscal_year"],
                "section": record["section"],
                "source": record["source"],
                "question": record["question"],
                "answer": record["answer"],
                "context": record["context"],
                "messages": example["messages"],
                "text": example["text"],
            }
        )
    return examples


def filter_leaks(
    examples: Sequence[Dict[str, Any]],
    leak_index: LeakIndex,
) -> Tuple[List[Dict[str, Any]], int]:
    """Drops any example whose question/answer collides with the benchmark."""
    kept: List[Dict[str, Any]] = []
    removed = 0
    for example in examples:
        if leak_index.is_leak(example["question"], example["answer"]):
            removed += 1
            continue
        kept.append(example)
    return kept, removed


def split_and_save(
    examples: Sequence[Dict[str, Any]],
    val_fraction: float,
    seed: int,
    out_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministically shuffles and splits into train/val, writing JSONL."""
    rows = list(examples)
    random.Random(seed).shuffle(rows)

    if val_fraction <= 0:
        val_count = 0
    else:
        val_count = max(1, int(round(len(rows) * val_fraction))) if rows else 0
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "val.jsonl", val_rows)
    return train_rows, val_rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), path)


# ─── Part 6: Orchestration ────────────────────────────────────────────────────

def collect_records(
    source: str,
    allow_download: bool,
    max_synthetic: int,
) -> Tuple[List[Dict[str, Any]], str]:
    """Resolves the raw records according to --source and availability."""
    if source == "primary":
        records = load_primary_records()
        if not records and allow_download:
            records = download_primary_records()
        return records, "virattt/financial-qa-10K"

    if source == "synthetic":
        return synthesize_from_chunks(max_records=max_synthetic), "synthetic-corpus"

    # auto: prefer primary, fall back to synthetic
    records = load_primary_records()
    if not records and allow_download:
        records = download_primary_records()
    if records:
        return records, "virattt/financial-qa-10K"
    logger.warning("Primary dataset unavailable; falling back to synthetic corpus QA.")
    return synthesize_from_chunks(max_records=max_synthetic), "synthetic-corpus"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the FinSight D-1 fine-tuning dataset.")
    parser.add_argument(
        "--source",
        choices=("auto", "primary", "synthetic"),
        default="auto",
        help="Which data source to use (default: auto = primary with synthetic fallback).",
    )
    parser.add_argument(
        "--max-synthetic",
        type=int,
        default=1500,
        help="Maximum synthetic fallback records to generate.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Fraction of examples held out for validation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle/split random seed.")
    parser.add_argument(
        "--leak-threshold",
        type=float,
        default=0.8,
        help="Jaccard similarity above which a question is treated as a benchmark leak.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Never attempt a network download; use on-disk data only.",
    )
    args = parser.parse_args()

    raw_records, source_label = collect_records(
        source=args.source,
        allow_download=not args.no_download,
        max_synthetic=args.max_synthetic,
    )

    normalized = [
        record
        for record in (normalize_primary_record(r) if source_label.startswith("virattt") else r
                       for r in raw_records)
        if record is not None
    ]
    logger.info("Kept %d records after scope filtering (from %d raw).", len(normalized), len(raw_records))

    if not normalized:
        logger.error(
            "No usable records found. Populate %s via scripts/download_virattt_dataset.py, "
            "or build the corpus (retrieval/chunk.py) for the synthetic fallback.",
            PRIMARY_JSONL,
        )
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": source_label,
            "counts": {"raw": len(raw_records), "after_filter": 0, "leaks_removed": 0, "kept": 0,
                       "train": 0, "val": 0},
            "config": {
                "tickers": list(RATIFIED_TICKERS),
                "years": list(RATIFIED_YEARS),
                "val_fraction": args.val_fraction,
                "seed": args.seed,
                "leak_threshold": args.leak_threshold,
            },
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return 1

    leak_index = LeakIndex(load_benchmark(), threshold=args.leak_threshold)
    examples = build_examples(normalized)
    examples, leaks_removed = filter_leaks(examples, leak_index)
    logger.info("Removed %d benchmark-leaking examples; %d remain.", leaks_removed, len(examples))

    train_rows, val_rows = split_and_save(
        examples, val_fraction=args.val_fraction, seed=args.seed, out_dir=OUT_DIR
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source_label,
        "counts": {
            "raw": len(raw_records),
            "after_filter": len(normalized),
            "leaks_removed": leaks_removed,
            "kept": len(examples),
            "train": len(train_rows),
            "val": len(val_rows),
        },
        "config": {
            "tickers": list(RATIFIED_TICKERS),
            "years": list(RATIFIED_YEARS),
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "leak_threshold": args.leak_threshold,
            "max_synthetic": args.max_synthetic,
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("  FINSIGHT D-1: FINE-TUNING DATASET BUILD COMPLETE")
    print("=" * 72)
    print(f"  Source          : {source_label}")
    print(f"  Raw records     : {len(raw_records)}")
    print(f"  After filtering : {len(normalized)}")
    print(f"  Leaks removed   : {leaks_removed}")
    print(f"  Kept examples   : {len(examples)}")
    print(f"  Train / Val     : {len(train_rows)} / {len(val_rows)}")
    print(f"  Output dir      : {OUT_DIR}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
