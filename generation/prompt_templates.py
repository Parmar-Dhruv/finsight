"""
================================================================================
FinSight — Generation & Fine-Tuning Layer
Citation-Enforced Prompt Templates (generation/prompt_templates.py)
================================================================================

Task D-2 (docs/finsight-task-breakdown.md):
  "Write the citation-enforced prompt template, using the
   [SOURCE N | TICKER FYyyyy section | Score] format already produced by
   format_context_for_prompt(). Include an explicit refusal/low-confidence
   instruction for when retrieved context doesn't support an answer."

Design notes
------------
* This module is deliberately PURE: it has no heavy dependencies (no torch, no
  transformers, no qdrant) so it can be unit-tested in CI without a GPU or a
  live vector database.
* The source-header format here is the canonical contract shared with the
  retrieval layer (retrieval/search.py::format_context_for_prompt). Keeping the
  formatter in one place lets us assert it stays in sync via tests.
* Chat rendering follows the Llama 3.2 Instruct chat template
  (<|begin_of_text|> / <|start_header_id|> ... <|eot_id|>).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

# ─── Llama 3.2 Instruct chat special tokens ───────────────────────────────────

BEGIN_OF_TEXT = "<|begin_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
EOT = "<|eot_id|>"

# ─── Refusal & system instruction ─────────────────────────────────────────────

REFUSAL_MESSAGE = (
    "I could not find sufficient information in the retrieved 10-K passages "
    "to answer this question."
)

SYSTEM_PROMPT = (
    "You are FinSight, a financial question-answering assistant specialised in "
    "SEC 10-K annual filings.\n"
    "Answer the user's question using ONLY the information contained in the "
    "retrieved 10-K context provided below.\n"
    "Rules:\n"
    "1. Every factual claim or figure MUST be followed by a citation to the "
    "source it came from, using the format [SOURCE N], where N is the numbered "
    "source label from the context block.\n"
    "2. Preserve exact figures, units ($M / $B), fiscal years, and accounting "
    "terminology. Do not round or restate numbers unless asked.\n"
    "3. If the retrieved context does not contain enough information to answer "
    f'the question, reply exactly: "{REFUSAL_MESSAGE}"\n'
    "4. Never use outside knowledge. Never guess, estimate, or fabricate a "
    "number. A refusal is always better than an unsupported answer."
)

# Matches an existing citation such as [SOURCE 1] or [SOURCE 12].
_CITATION_RE = re.compile(r"\[SOURCE\s+\d+\]")


# ─── Part 1: Source-header formatting (shared retrieval contract) ─────────────

def format_source_header(
    index: int,
    ticker: str,
    fiscal_year: Any,
    section: str,
    score: float,
) -> str:
    """Renders a single attribution header identical to the retrieval layer."""
    return f"[SOURCE {index} | {ticker} FY{fiscal_year} {section} | Score: {score:.4f}]"


def format_source_block(
    ticker: str,
    fiscal_year: Any,
    section: str,
    text: str,
    score: float = 1.0,
    index: int = 1,
) -> str:
    """Renders one attributed passage (header + body + trailing newline)."""
    header = format_source_header(index, ticker, fiscal_year, section, score)
    return f"{header}\n{text.strip()}\n"


def format_context_block(chunks: Sequence[Dict[str, Any]], max_tokens: int = 2000) -> str:
    """
    Thin wrapper over the retrieval layer's canonical formatter.

    The import is deferred so this module stays import-light for tests; the
    retrieval layer is only required when formatting live search results.
    """
    from retrieval.search import format_context_for_prompt  # noqa: WPS433 (lazy)

    return format_context_for_prompt(list(chunks), max_tokens=max_tokens)


# ─── Part 2: User turn & chat assembly ────────────────────────────────────────

def build_user_content(question: str, context_block: str) -> str:
    """Composes the user turn from retrieved context + the question."""
    return (
        f"{context_block.strip()}\n\n"
        f"Question: {question.strip()}\n"
        f"Answer:"
    )


def build_messages(question: str, context_block: str) -> List[Dict[str, str]]:
    """Returns the [system, user] message list for a RAG query."""
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(question, context_block)},
    ]


def render_chat(
    messages: Sequence[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    """
    Renders a message list into the Llama 3.2 Instruct chat string.

    When add_generation_prompt is True, the string ends with an open assistant
    header so the model continues the answer.
    """
    parts: List[str] = [BEGIN_OF_TEXT]
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "").strip()
        parts.append(f"{START_HEADER}{role}{END_HEADER}\n\n{content}{EOT}")

    if add_generation_prompt:
        parts.append(f"{START_HEADER}assistant{END_HEADER}\n\n")

    return "".join(parts)


def build_prompt(question: str, context_block: str) -> str:
    """Full inference prompt ending at the open assistant turn."""
    return render_chat(build_messages(question, context_block), add_generation_prompt=True)


# ─── Part 3: Training-target construction (D-1 uses these) ────────────────────

def has_citation(text: str) -> bool:
    """True if the text already contains at least one [SOURCE N] citation."""
    return bool(_CITATION_RE.search(text or ""))


def build_training_target(
    answer: str,
    source_indices: Sequence[int] = (1,),
) -> str:
    """
    Produces the assistant target for a training example.

    * Empty answers collapse to the canonical refusal.
    * Answers that already carry citations are preserved verbatim.
    * Otherwise the source citation(s) are appended so the model learns the
      attribution contract.
    """
    answer = (answer or "").strip()
    if not answer:
        return REFUSAL_MESSAGE
    if has_citation(answer):
        return answer

    tags = " ".join(f"[SOURCE {i}]" for i in source_indices)
    return f"{answer} {tags}".strip()


def build_training_example(
    question: str,
    context_block: str,
    answer: str,
    source_indices: Sequence[int] = (1,),
) -> Dict[str, Any]:
    """
    Builds a single supervised fine-tuning example:
        messages : [system, user, assistant]
        text     : fully rendered Llama 3.2 chat string
    """
    messages = build_messages(question, context_block)
    target = build_training_target(answer, source_indices=source_indices)
    messages.append({"role": "assistant", "content": target})
    return {"messages": messages, "text": render_chat(messages)}


if __name__ == "__main__":
    demo_chunks = [
        {
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "section": "item_7",
            "score": 0.8255,
            "text": (
                "Total net sales increased 2% or $7.8 billion during 2024 "
                "compared to 2023, driven primarily by Services."
            ),
        }
    ]
    demo_context = (
        "=== RETRIEVED 10-K CONTEXT (1 passages) ===\n\n"
        + format_source_block(
            demo_chunks[0]["ticker"],
            demo_chunks[0]["fiscal_year"],
            demo_chunks[0]["section"],
            demo_chunks[0]["text"],
            score=demo_chunks[0]["score"],
        )
        + "========================================================="
    )

    print("=" * 78)
    print("FinSight Prompt Template Smoke Test (D-2)")
    print("=" * 78)
    print("\n--- System prompt ---\n" + SYSTEM_PROMPT)
    print("\n--- Rendered inference prompt ---\n")
    print(build_prompt("What were Apple's total net sales in FY2024?", demo_context))
    print("\n--- Rendered training example ---\n")
    print(
        build_training_example(
            "What were Apple's total net sales in FY2024?",
            demo_context,
            "Apple's total net sales in fiscal year 2024 were $391.0 billion.",
        )["text"]
    )
