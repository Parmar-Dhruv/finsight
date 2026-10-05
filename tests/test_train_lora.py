"""Unit tests for D-3: QLoRA training config + data validation (no heavy deps)."""

import json

import pytest

from generation import train_lora as tl


def test_config_defaults_and_serialization():
    config = tl.LoraTrainingConfig()
    record = config.to_dict()
    assert record["lora_r"] == 16
    assert record["lora_alpha"] == 32
    assert record["learning_rate"] == 2e-4
    assert record["load_in_4bit"] is True
    assert "q_proj" in record["lora_target_modules"]


def test_load_jsonl(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"text": "a"}) + "\n" + json.dumps({"text": "b"}) + "\n",
        encoding="utf-8",
    )
    rows = tl.load_jsonl(path)
    assert [r["text"] for r in rows] == ["a", "b"]


def test_load_jsonl_missing_returns_empty(tmp_path):
    assert tl.load_jsonl(tmp_path / "nope.jsonl") == []


def test_validate_missing_train_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        tl.validate_training_data(tmp_path / "train.jsonl", tmp_path / "val.jsonl")


def test_validate_empty_train_raises(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        tl.validate_training_data(train, tmp_path / "val.jsonl")


def test_validate_missing_text_field_raises(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"question": "no text"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        tl.validate_training_data(train, tmp_path / "val.jsonl")


def test_validate_success_summary(tmp_path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text(json.dumps({"text": "hello world"}) + "\n", encoding="utf-8")
    val.write_text(json.dumps({"text": "val"}) + "\n", encoding="utf-8")

    summary = tl.validate_training_data(train, val)
    assert summary["train_rows"] == 1
    assert summary["val_rows"] == 1
    assert summary["sample_chars"] == len("hello world")


def test_write_run_config(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "OUTPUT_DIR", tmp_path)
    config = tl.LoraTrainingConfig()
    path = tl.write_run_config(config, {"train_rows": 10, "backend": "dry-run"})
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["task"] == "D-3"
    assert payload["data"]["train_rows"] == 10
    assert payload["config"]["lora_r"] == 16
