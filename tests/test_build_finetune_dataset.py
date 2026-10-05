"""Unit tests for D-1: fine-tuning dataset builder (pure logic only)."""

import json

from generation import build_finetune_dataset as bfd


# ─── Normalization & leak detection ───────────────────────────────────────────

def test_normalize_text_strips_punctuation_and_case():
    assert bfd.normalize_text("Apple's Net Sales, $391.0 billion!") == (
        "apple s net sales 391 0 billion"
    )


def test_jaccard_bounds():
    assert bfd.jaccard(set(), set()) == 0.0
    assert bfd.jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert bfd.jaccard({"a", "b"}, {"b", "c"}) == 1 / 3


def test_parse_filing_year_variants():
    assert bfd.parse_filing_year({"filing": "2024_10K"}) == 2024
    assert bfd.parse_filing_year({"fiscal_year": 2023}) == 2023
    assert bfd.parse_filing_year({"fiscal_year": "FY2024"}) == 2024
    assert bfd.parse_filing_year({"filing": "no year here"}) is None


def test_leak_index_exact_match():
    index = bfd.LeakIndex([{"question": "What was Apple's total revenue?", "gold_answer": ""}])
    assert index.is_leak("What was Apple's total revenue?")
    assert not index.is_leak("What was Microsoft's operating margin?")


def test_leak_index_jaccard_near_duplicate():
    index = bfd.LeakIndex(
        [{"question": "What was Apple total net revenue fiscal 2024?", "gold_answer": ""}]
    )
    assert index.is_leak("What was Apple's total net revenue for fiscal 2024?")


def test_leak_index_answer_match():
    index = bfd.LeakIndex(
        [{"question": "unrelated", "gold_answer": "Apple generated $96.2 billion from Services."}]
    )
    assert index.is_leak("Something else", "Apple generated $96.2 billion from Services.")


# ─── Primary record normalization ─────────────────────────────────────────────

def _valid_record():
    return {
        "question": "What was the GPU invention year?",
        "answer": "NVIDIA invented the GPU in 1999.",
        "context": "Our invention of the GPU in 1999 defined modern computer graphics.",
        "ticker": "nvda",
        "filing": "2023_10K",
    }


def test_normalize_primary_record_keeps_valid():
    record = bfd.normalize_primary_record(_valid_record())
    assert record is not None
    assert record["ticker"] == "NVDA"
    assert record["fiscal_year"] == 2023


def test_normalize_primary_record_drops_wrong_ticker():
    bad = _valid_record()
    bad["ticker"] = "TSLA"
    assert bfd.normalize_primary_record(bad) is None


def test_normalize_primary_record_drops_wrong_year():
    bad = _valid_record()
    bad["filing"] = "2020_10K"
    assert bfd.normalize_primary_record(bad) is None


def test_normalize_primary_record_drops_incomplete():
    bad = _valid_record()
    bad["answer"] = ""
    assert bfd.normalize_primary_record(bad) is None


# ─── Example rendering & leak filtering ───────────────────────────────────────

def test_build_context_block_uses_retrieval_headers():
    record = bfd.normalize_primary_record(_valid_record())
    block = bfd.build_context_block(record)
    assert "[SOURCE 1 | NVDA FY2023" in block
    assert "RETRIEVED 10-K CONTEXT" in block


def test_build_examples_and_filter_leaks():
    records = [bfd.normalize_primary_record(_valid_record())]
    examples = bfd.build_examples(records)
    assert len(examples) == 1
    assert "[SOURCE 1]" in examples[0]["messages"][-1]["content"]

    leaky_index = bfd.LeakIndex(
        [{"question": "What was the GPU invention year?", "gold_answer": ""}]
    )
    kept, removed = bfd.filter_leaks(examples, leaky_index)
    assert removed == 1
    assert kept == []


# ─── Synthetic fallback ───────────────────────────────────────────────────────

def test_synthesize_from_chunks(tmp_path):
    chunks = [
        {
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "section": "item_7",
            "chunk_id": "AAPL_2024_item_7_chunk_000",
            "text": "Total net sales was 391.0 billion in fiscal 2024.",
        },
        {
            "ticker": "TSLA",
            "fiscal_year": 2024,
            "section": "item_7",
            "text": "Irrelevant company should be filtered out.",
        },
    ]
    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text(json.dumps(chunks), encoding="utf-8")

    records = bfd.synthesize_from_chunks(chunks_path=chunks_file, max_records=10)
    assert len(records) == 1
    assert records[0]["ticker"] == "AAPL"
    assert records[0]["source"] == "synthetic-corpus"
    assert records[0]["context"].startswith("Total net sales")


def test_synthesize_missing_corpus_returns_empty(tmp_path):
    assert bfd.synthesize_from_chunks(chunks_path=tmp_path / "nope.json") == []


# ─── Deterministic split & persistence ────────────────────────────────────────

def test_split_and_save_deterministic(tmp_path):
    examples = [{"id": f"ft_{i:06d}", "question": f"q{i}"} for i in range(20)]
    train_a, val_a = bfd.split_and_save(examples, 0.2, 42, tmp_path)
    train_b, val_b = bfd.split_and_save(examples, 0.2, 42, tmp_path)

    assert [e["id"] for e in train_a] == [e["id"] for e in train_b]
    assert [e["id"] for e in val_a] == [e["id"] for e in val_b]
    assert len(val_a) == 4
    assert len(train_a) == 16
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "val.jsonl").exists()
