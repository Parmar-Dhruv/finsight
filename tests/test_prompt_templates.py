"""Unit tests for D-2: citation-enforced prompt templates."""

import pytest

from generation import prompt_templates as pt


def test_system_prompt_contains_refusal_instruction():
    assert pt.REFUSAL_MESSAGE in pt.SYSTEM_PROMPT
    assert "[SOURCE N]" in pt.SYSTEM_PROMPT


def test_format_source_block_matches_retrieval_contract():
    block = pt.format_source_block("AAPL", 2024, "item_7", "Total net sales...", score=0.8255)
    assert block.startswith("[SOURCE 1 | AAPL FY2024 item_7 | Score: 0.8255]")
    assert block.endswith("Total net sales...\n")


def test_build_messages_roles():
    messages = pt.build_messages("Q?", "CONTEXT")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Q?" in messages[1]["content"]
    assert "CONTEXT" in messages[1]["content"]


def test_build_messages_rejects_empty_question():
    with pytest.raises(ValueError):
        pt.build_messages("   ", "CONTEXT")


def test_render_chat_uses_llama_tokens():
    rendered = pt.render_chat([{"role": "user", "content": "hi"}])
    assert rendered.startswith(pt.BEGIN_OF_TEXT)
    assert f"{pt.START_HEADER}user{pt.END_HEADER}" in rendered
    assert rendered.endswith(pt.EOT)


def test_build_prompt_opens_assistant_turn():
    rendered = pt.build_prompt("Q?", "CONTEXT")
    assert rendered.endswith(f"{pt.START_HEADER}assistant{pt.END_HEADER}\n\n")


def test_training_target_appends_citation():
    target = pt.build_training_target("Revenue was $391.0 billion.", source_indices=(1,))
    assert target == "Revenue was $391.0 billion. [SOURCE 1]"


def test_training_target_preserves_existing_citation():
    original = "Revenue was $391.0 billion [SOURCE 2]."
    assert pt.build_training_target(original) == original


def test_training_target_empty_becomes_refusal():
    assert pt.build_training_target("   ") == pt.REFUSAL_MESSAGE


def test_has_citation():
    assert pt.has_citation("fact [SOURCE 12]")
    assert not pt.has_citation("no citation here")


def test_build_training_example_has_three_turns():
    example = pt.build_training_example("Q?", "CONTEXT", "Answer.")
    assert [m["role"] for m in example["messages"]] == ["system", "user", "assistant"]
    assert "[SOURCE 1]" in example["messages"][-1]["content"]
    assert example["text"].startswith(pt.BEGIN_OF_TEXT)
