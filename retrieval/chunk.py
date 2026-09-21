"""
================================================================================
FinSight: Financial Document Chunking Module (Parts 1, 2, 3 & 4)
Component: Retrieval & Data Layer (Nilay)
================================================================================

This module transforms parsed SEC 10-K filings into retrieval-ready text chunks.

Why Recursive Chunking?
-----------------------
Raw 10-K sections are tens of thousands of words long. Embedding models
(e.g., BGE, MiniLM) have maximum sequence lengths (usually 512 tokens).
If chunks are cut arbitrarily (e.g., at fixed character counts), sentences
and markdown tables get sliced in half, destroying semantic meaning.

Recursive splitting respects structural document hierarchy:
  1. Paragraph boundaries ("\\n\\n") — preserve full conceptual paragraphs
  2. Line boundaries ("\\n") — preserve markdown table rows or lists
  3. Sentence boundaries (". ") — preserve complete grammatical ideas
  4. Word boundaries (" ") — absolute fallback to avoid cutting words
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─── Part 1: Chunk Data Model ──────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    Represents a single retrieval chunk along with complete provenance metadata.

    Attributes:
        chunk_id: Unique identifier, e.g. 'AAPL_2023_item_7_chunk_004'.
        text: The actual textual content of the chunk (ready for embedding).
        ticker: Company stock ticker symbol (e.g., 'AAPL', 'MSFT').
        fiscal_year: Filing year (e.g., 2023, 2024).
        section: 10-K section key (e.g., 'item_1', 'item_1a', 'item_7').
        chunk_index: 0-indexed position within the section.
        char_count: Length of the chunk text in characters.
        token_count_est: Approximate token count (~4 characters per token).
    """
    chunk_id: str
    text: str
    ticker: str
    fiscal_year: int
    section: str
    chunk_index: int
    char_count: int
    token_count_est: int

    def to_dict(self) -> dict[str, Any]:
        """Converts chunk dataclass to dictionary for JSON serialization."""
        return asdict(self)


# ─── Part 2: Recursive Text Splitter ──────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Quick estimation of token count.
    As a rule of thumb for English text, 1 token is roughly 4 characters (or ~0.75 words).
    """
    return max(1, len(text) // 4)


def split_text_recursive(
    text: str,
    max_chunk_size: int = 1500,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Splits text recursively using hierarchical separators to stay below max_chunk_size
    while maintaining context via chunk_overlap.

    Parameters:
        text: The raw input string to split.
        max_chunk_size: Target maximum characters per chunk (default: 1500 chars ~ 375 tokens).
        chunk_overlap: Number of characters to overlap between consecutive chunks (default: 200 chars ~ 50 tokens).
        separators: Priority list of delimiters. Defaults to ["\\n\\n", "\\n", ". ", " "].

    Returns:
        A list of string chunks, each roughly <= max_chunk_size.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    text = text.strip()
    if not text:
        return []

    # Base case: text is already within size limit
    if len(text) <= max_chunk_size:
        return [text]

    # Find the highest-priority separator present in the text
    chosen_separator = ""
    sub_separators = []
    for i, sep in enumerate(separators):
        if sep in text:
            chosen_separator = sep
            # Remaining separators for deeper splitting if pieces are still too large
            sub_separators = separators[i + 1:]
            break

    # If no separator was found (e.g. single long string without spaces), hard split
    if not chosen_separator:
        chunks = []
        start = 0
        step = max(1, max_chunk_size - chunk_overlap)
        while start < len(text):
            chunks.append(text[start : start + max_chunk_size])
            start += step
        return chunks

    # Split into raw segments by the chosen separator
    raw_segments = text.split(chosen_separator)
    
    # Merge segments into chunks <= max_chunk_size, carrying overlap forward
    merged_chunks: List[str] = []
    current_piece = ""

    for segment in raw_segments:
        segment = segment.strip()
        if not segment:
            continue

        # If an individual segment exceeds max_chunk_size, recursively split it
        # using the remaining finer separators
        if len(segment) > max_chunk_size:
            if current_piece:
                merged_chunks.append(current_piece.strip())
                current_piece = ""
            
            sub_chunks = split_text_recursive(
                segment,
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
                separators=sub_separators if sub_separators else [" "],
            )
            merged_chunks.extend(sub_chunks)
            continue

        # Calculate candidate text length if we add this segment
        candidate = (
            f"{current_piece}{chosen_separator}{segment}"
            if current_piece
            else segment
        )

        if len(candidate) <= max_chunk_size:
            current_piece = candidate
        else:
            # Current piece is full, commit it
            if current_piece:
                merged_chunks.append(current_piece.strip())
            
            # Start new piece with overlap from the end of current_piece
            if chunk_overlap > 0 and current_piece:
                # Take the trailing chunk_overlap characters from current_piece
                overlap_text = current_piece[-chunk_overlap:]
                # Try to clean boundary so we don't start midway through a word
                space_idx = overlap_text.find(" ")
                if space_idx != -1 and space_idx < len(overlap_text) - 1:
                    overlap_text = overlap_text[space_idx + 1:]
                current_piece = f"{overlap_text}{chosen_separator}{segment}"
            else:
                current_piece = segment

    # Append any remaining piece
    if current_piece and current_piece.strip():
        merged_chunks.append(current_piece.strip())

    return merged_chunks


# ─── Part 3: Document & Section Chunker ────────────────────────────────────────

def chunk_section(
    text: str,
    ticker: str,
    fiscal_year: int,
    section: str,
    max_chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    """
    Chunks a single 10-K section and wraps each text piece in a Chunk dataclass
    with complete provenance metadata.

    Parameters:
        text: Raw section text extracted from filing.
        ticker: Company ticker (e.g., 'AAPL').
        fiscal_year: Filing year (e.g., 2023).
        section: Section name (e.g., 'item_7').
        max_chunk_size: Character limit per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        A list of tagged Chunk objects.
    """
    raw_pieces = split_text_recursive(
        text=text,
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: List[Chunk] = []
    for idx, piece in enumerate(raw_pieces):
        chunk_id = f"{ticker}_{fiscal_year}_{section}_chunk_{idx:03d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=piece,
                ticker=ticker,
                fiscal_year=fiscal_year,
                section=section,
                chunk_index=idx,
                char_count=len(piece),
                token_count_est=estimate_tokens(piece),
            )
        )
    return chunks


def chunk_document(
    doc_data: dict[str, Any],
    max_chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    """
    Chunks an entire parsed filing JSON dictionary across all extracted sections.

    Parameters:
        doc_data: Dictionary loaded from parsed JSON (contains 'ticker', 'fiscal_year', 'sections').
        max_chunk_size: Character limit per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        Combined list of Chunk objects across all sections in the document.
    """
    ticker = doc_data["ticker"]
    fiscal_year = int(doc_data["fiscal_year"])
    sections = doc_data.get("sections", {})

    doc_chunks: List[Chunk] = []
    for section_name, section_text in sections.items():
        if not section_text or not section_text.strip():
            continue
        section_chunks = chunk_section(
            text=section_text,
            ticker=ticker,
            fiscal_year=fiscal_year,
            section=section_name,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        doc_chunks.extend(section_chunks)

    return doc_chunks


# ─── Part 4: Batch Pipeline & Strategy Comparison ─────────────────────────────

def chunk_all_filings(
    processed_dir: Path,
    output_file: Optional[Path] = None,
    max_chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    """
    Loads all parsed filing JSON files from processed_dir, chunks them,
    and optionally writes the list of Chunk dictionaries to output_file.

    Parameters:
        processed_dir: Directory containing '*_parsed.json' files.
        output_file: Path to write the aggregated chunks JSON (optional).
        max_chunk_size: Character limit per chunk.
        chunk_overlap: Overlap characters between chunks.

    Returns:
        Aggregated list of all Chunk objects across all documents.
    """
    processed_files = sorted(list(processed_dir.glob("*_parsed.json")))
    if not processed_files:
        print(f"[!] No parsed JSON filings found in {processed_dir}")
        return []

    all_chunks: List[Chunk] = []
    for file_path in processed_files:
        with open(file_path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)
        chunks = chunk_document(
            doc_data,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)
        print(f"  [+] {file_path.name:25s} -> {len(chunks):4d} chunks")

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        chunks_dict_list = [c.to_dict() for c in all_chunks]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks_dict_list, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Successfully saved {len(all_chunks)} chunks to: {output_file}")

    return all_chunks


def compute_chunk_statistics(chunks: List[Chunk]) -> dict[str, Any]:
    """Computes summary statistics for a set of chunks."""
    if not chunks:
        return {}
    char_counts = [c.char_count for c in chunks]
    token_counts = [c.token_count_est for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_chars": round(sum(char_counts) / len(chunks), 1),
        "min_chars": min(char_counts),
        "max_chars": max(char_counts),
        "avg_tokens": round(sum(token_counts) / len(chunks), 1),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
    }


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    OUTPUT_FILE = PROCESSED_DIR / "chunks.json"

    print("\n" + "=" * 70)
    print("FinSight: 10-K Document Chunking Pipeline & Strategy Comparison")
    print("=" * 70)

    # ── Strategy A: Standard / Context-Rich (1500 chars ~ 375 tokens, overlap 200)
    print("\n>>> Running Strategy A (Standard Context: 1500 chars, overlap 200)...")
    strategy_a_chunks = chunk_all_filings(
        processed_dir=PROCESSED_DIR,
        output_file=OUTPUT_FILE,  # Strategy A will be saved as primary dataset
        max_chunk_size=1500,
        chunk_overlap=200,
    )
    stats_a = compute_chunk_statistics(strategy_a_chunks)

    # ── Strategy B: Compact / High-Precision (800 chars ~ 200 tokens, overlap 120)
    print("\n>>> Running Strategy B (Compact/Targeted: 800 chars, overlap 120)...")
    strategy_b_chunks = chunk_all_filings(
        processed_dir=PROCESSED_DIR,
        output_file=None,  # Only compute for comparison
        max_chunk_size=800,
        chunk_overlap=120,
    )
    stats_b = compute_chunk_statistics(strategy_b_chunks)

    # ── Comparison Summary Table
    min_max_chars_a = f"{stats_a['min_chars']} / {stats_a['max_chars']}"
    min_max_chars_b = f"{stats_b['min_chars']} / {stats_b['max_chars']}"
    min_max_tokens_a = f"{stats_a['min_tokens']} / {stats_a['max_tokens']}"
    min_max_tokens_b = f"{stats_b['min_tokens']} / {stats_b['max_tokens']}"

    print("\n" + "=" * 70)
    print("CHUNK STRATEGY COMPARISON REPORT")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Strategy A (1500/200)':<20} | {'Strategy B (800/120)':<20}")
    print("-" * 70)
    print(f"{'Total Chunks':<25} | {stats_a['total_chunks']:<20} | {stats_b['total_chunks']:<20}")
    print(f"{'Avg Characters':<25} | {stats_a['avg_chars']:<20} | {stats_b['avg_chars']:<20}")
    print(f"{'Min / Max Characters':<25} | {min_max_chars_a:<20} | {min_max_chars_b:<20}")
    print(f"{'Avg Estimated Tokens':<25} | {stats_a['avg_tokens']:<20} | {stats_b['avg_tokens']:<20}")
    print(f"{'Min / Max Tokens':<25} | {min_max_tokens_a:<20} | {min_max_tokens_b:<20}")
    print("=" * 70)

