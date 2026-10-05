"""
FinSight — Generation & Fine-Tuning Layer (Dhruv)

This package contains the LLM-side of the FinSight RAG pipeline:
  - prompt_templates.py       : citation-enforced prompt construction (D-2)
  - build_finetune_dataset.py : QLoRA fine-tuning dataset builder (D-1)

The retrieval contract consumed here lives in retrieval/search.py:
  retrieve() -> chunks, format_context_for_prompt() -> attributed context block.
"""

__all__ = ["prompt_templates", "build_finetune_dataset"]
