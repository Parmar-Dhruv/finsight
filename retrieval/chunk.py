"""
================================================================================
FinSight: Financial Document Chunking Module (Part 1 & Part 2)
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

import re
from dataclasses import asdict, dataclass
from typing import Any, List, Optional


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
