"""
================================================================================
FinSight — Generation & Fine-Tuning Layer
D-3: QLoRA Fine-Tuning of Llama 3.2 3B Instruct (generation/train_lora.py)
================================================================================

Task D-3 (docs/finsight-task-breakdown.md):
  "Run QLoRA fine-tuning via Unsloth on Llama 3.2 3B Instruct, free-tier
   Colab/Kaggle T4. Record the training config (rank, alpha, learning rate,
   steps) for the report."

This script consumes the D-1 artifact (`generation/data/train.jsonl` /
`val.jsonl`), whose records already contain a fully rendered Llama 3.2 chat
string under the `text` field (see generation/prompt_templates.py). It writes
a LoRA adapter (and a machine-readable run config for the report) to
`generation/checkpoints/`.

Compute
-------
Intended to run on a free-tier T4 (Unsloth QLoRA, 4-bit base weights). The base
model is loaded 4-bit; only LoRA adapters are trained. Heavy ML imports are
deferred so this module can be imported and dry-run on any machine.

Usage
-----
    # Validate config + data without a GPU (safe everywhere):
    python generation/train_lora.py --dry-run

    # Actual training (Colab/Kaggle T4):
    python generation/train_lora.py

    # Custom run:
    python generation/train_lora.py --rank 32 --alpha 64 --epochs 2 --lr 2e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / "generation" / "data"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"
OUTPUT_DIR = BASE_DIR / "generation" / "checkpoints"


# ─── Part 1: Training configuration ───────────────────────────────────────────

@dataclass
class LoraTrainingConfig:
    """All hyperparameters recorded for the report and reproducibility."""

    base_model: str = "unsloth/Llama-3.2-3B-Instruct"
    output_dir: str = str(OUTPUT_DIR / "lora_adapter")
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # LoRA adapter geometry
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # Optimisation
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    seed: int = 42

    # Evaluation / checkpointing
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    logging_steps: int = 10
    report_to: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─── Part 2: Data loading (pure / testable) ───────────────────────────────────

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dicts (empty list if missing)."""
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_training_data(
    train_path: Path = TRAIN_PATH,
    val_path: Path = VAL_PATH,
) -> Dict[str, Any]:
    """
    Verifies that the D-1 artifacts exist and carry the `text` training field.

    Returns a summary dict; raises FileNotFoundError/ValueError on hard errors.
    """
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}. Run "
            f"generation/build_finetune_dataset.py (D-1) first."
        )

    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path)

    if not train_rows:
        raise ValueError(f"Training file is empty: {train_path}")

    missing = [i for i, row in enumerate(train_rows) if not str(row.get("text", "")).strip()]
    if missing:
        raise ValueError(
            f"{len(missing)} training rows are missing a non-empty 'text' field "
            f"(first at index {missing[0]}). Rebuild the D-1 dataset."
        )

    return {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "sample_chars": len(train_rows[0]["text"]),
    }


# ─── Part 3: Model + trainer construction (heavy imports deferred) ────────────

def _load_text_dataset(train_path: Path, val_path: Path):
    from datasets import Dataset, DatasetDict

    def _to_dataset(rows: List[Dict[str, Any]]):
        return Dataset.from_list([{"text": row["text"]} for row in rows])

    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path)
    splits: Dict[str, Any] = {"train": _to_dataset(train_rows)}
    if val_rows:
        splits["validation"] = _to_dataset(val_rows)
    return DatasetDict(splits)


def train(config: LoraTrainingConfig) -> Dict[str, Any]:
    """
    Runs QLoRA fine-tuning and writes the adapter + run config.

    Prefers Unsloth (as ratified in brain.md); falls back to a plain
    transformers + peft + trl SFTTrainer stack when Unsloth is unavailable.
    """
    validate_training_data(TRAIN_PATH, VAL_PATH)
    dataset = _load_text_dataset(TRAIN_PATH, VAL_PATH)

    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig  # noqa: F401  (ensures installed)

    try:
        return _train_with_unsloth(config, dataset)
    except ImportError as exc:
        logger.warning("Unsloth unavailable (%s); falling back to transformers+peft.", exc)
        return _train_with_transformers(config, dataset)


def _train_with_unsloth(config: LoraTrainingConfig, dataset) -> Dict[str, Any]:
    from unsloth import FastLanguageModel  # type: ignore
    from trl import SFTTrainer  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        dtype=None,               # auto: bf16 on Ampere+, else fp16
        load_in_4bit=config.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.lora_target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )

    training_args = _build_training_args(config)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=training_args,
    )
    trainer.train()
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    return {"backend": "unsloth", "output_dir": config.output_dir}


def _train_with_transformers(config: LoraTrainingConfig, dataset) -> Dict[str, Any]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTTrainer  # type: ignore

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    ) if config.load_in_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=_build_training_args(config),
    )
    trainer.train()
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    return {"backend": "transformers+peft", "output_dir": config.output_dir}


def _build_training_args(config: LoraTrainingConfig):
    try:
        from trl import SFTConfig  # type: ignore

        return SFTConfig(
            output_dir=config.output_dir,
            max_seq_length=config.max_seq_length,
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            optim=config.optim,
            seed=config.seed,
            eval_strategy=config.eval_strategy,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            logging_steps=config.logging_steps,
            report_to=config.report_to,
        )
    except ImportError:
        from transformers import TrainingArguments  # type: ignore

        return TrainingArguments(
            output_dir=config.output_dir,
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            optim=config.optim,
            seed=config.seed,
            evaluation_strategy=config.eval_strategy,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            logging_steps=config.logging_steps,
            report_to=config.report_to,
        )


# ─── Part 4: Run-record persistence ───────────────────────────────────────────

def write_run_config(config: LoraTrainingConfig, summary: Dict[str, Any]) -> Path:
    """Persists the exact config + data summary used for this run (for the report)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "D-3",
        "config": config.to_dict(),
        "data": summary,
        "backend": summary.get("backend"),
    }
    path = OUTPUT_DIR / "train_run_config.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("Wrote run config to %s", path)
    return path


# ─── Part 5: CLI ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinSight D-3 QLoRA fine-tuning.")
    parser.add_argument("--base-model", default=LoraTrainingConfig.base_model)
    parser.add_argument("--rank", type=int, default=LoraTrainingConfig.lora_r)
    parser.add_argument("--alpha", type=int, default=LoraTrainingConfig.lora_alpha)
    parser.add_argument("--dropout", type=float, default=LoraTrainingConfig.lora_dropout)
    parser.add_argument("--lr", type=float, default=LoraTrainingConfig.learning_rate)
    parser.add_argument("--epochs", type=int, default=LoraTrainingConfig.num_train_epochs)
    parser.add_argument("--max-seq-length", type=int, default=LoraTrainingConfig.max_seq_length)
    parser.add_argument("--output-dir", default=LoraTrainingConfig.output_dir)
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (not recommended on 16GB T4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data + print the resolved config without loading a model.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = LoraTrainingConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        lora_r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
    )

    print("=" * 78)
    print("FinSight D-3: QLoRA Fine-Tuning (Llama 3.2 3B Instruct)")
    print("=" * 78)

    try:
        summary = validate_training_data(TRAIN_PATH, VAL_PATH)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    print(f"  Train rows     : {summary['train_rows']}")
    print(f"  Val rows       : {summary['val_rows']}")
    print(f"  LoRA rank/alpha: {config.lora_r}/{config.lora_alpha}")
    print(f"  Learning rate  : {config.learning_rate}")
    print(f"  Epochs         : {config.num_train_epochs}")
    print(f"  Max seq length : {config.max_seq_length}")

    if args.dry_run:
        print("\n[DRY RUN] Data validated; no model loaded. Config is ready to train.")
        write_run_config(config, {**summary, "backend": "dry-run"})
        return 0

    result = train(config)
    merged = {**summary, **result}
    write_run_config(config, merged)
    print(f"\n[OK] Adapter saved to: {merged.get('output_dir')} (backend: {merged.get('backend')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
