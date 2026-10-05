"""
================================================================================
FinSight — Generation & Fine-Tuning Layer
D-4: Merge LoRA + Export GGUF 4-bit (generation/quantize_gguf.py)
================================================================================

Task D-4 (docs/finsight-task-breakdown.md):
  "Quantize the fine-tuned model to GGUF (4-bit) via llama.cpp. Confirm the
   output format with Jay before handoff."

Pipeline
--------
  1. MERGE  : load the D-3 LoRA adapter on top of the base model (fp16/CPU),
              `merge_and_unload()`, and write a standard HF model directory.
  2. CONVERT: run llama.cpp's `convert_hf_to_gguf.py` -> f16 GGUF.
  3. QUANT  : run llama.cpp's `llama-quantize` -> Q4_K_M GGUF (the deployment
              format for free-tier CPU hosting, per brain.md).

llama.cpp binaries are not bundled; point `--llama-cpp-dir` at a local build
(see https://github.com/ggerganov/llama.cpp#build). The script locates the
converter (`convert_hf_to_gguf.py`) and quantizer (`llama-quantize`/
`llama-quantize.exe`) inside that directory.

Usage
-----
    # Show the exact commands without doing any work:
    python generation/quantize_gguf.py --dry-run

    # Merge only (skip llama.cpp steps):
    python generation/quantize_gguf.py --merge-only

    # Full pipeline:
    python generation/quantize_gguf.py --llama-cpp-dir C:\\tools\\llama.cpp
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = BASE_DIR / "generation" / "checkpoints"
DEFAULT_ADAPTER = CHECKPOINTS_DIR / "lora_adapter"
DEFAULT_MERGED = CHECKPOINTS_DIR / "merged_fp16"
DEFAULT_EXPORT = CHECKPOINTS_DIR / "gguf"


# ─── Part 1: Configuration & command construction (pure / testable) ───────────

@dataclass
class QuantizeConfig:
    base_model: str = "unsloth/Llama-3.2-3B-Instruct"
    adapter_dir: str = str(DEFAULT_ADAPTER)
    merged_dir: str = str(DEFAULT_MERGED)
    export_dir: str = str(DEFAULT_EXPORT)
    llama_cpp_dir: Optional[str] = None
    quant_type: str = "Q4_K_M"
    model_basename: str = "finsight-llama-3.2-3b"

    @property
    def f16_gguf(self) -> str:
        return str(Path(self.export_dir) / f"{self.model_basename}-f16.gguf")

    @property
    def quant_gguf(self) -> str:
        return str(Path(self.export_dir) / f"{self.model_basename}-{self.quant_type.lower()}.gguf")


def find_llama_cpp_binaries(llama_cpp_dir: Optional[str]) -> dict:
    """
    Locates the llama.cpp converter script and quantizer executable.

    Raises FileNotFoundError with guidance if the directory is missing or the
    expected tools cannot be found.
    """
    if not llama_cpp_dir:
        raise FileNotFoundError(
            "llama.cpp directory not provided. Pass --llama-cpp-dir pointing to a "
            "local build (clone https://github.com/ggerganov/llama.cpp and build it)."
        )

    root = Path(llama_cpp_dir)
    if not root.exists():
        raise FileNotFoundError(f"llama.cpp directory does not exist: {root}")

    converter = root / "convert_hf_to_gguf.py"
    if not converter.exists():
        # Older llama.cpp layouts keep the converter at the repo root as well.
        alt = root / "convert-hf-to-gguf.py"
        converter = alt if alt.exists() else converter

    candidates = [
        root / "llama-quantize",
        root / "llama-quantize.exe",
        root / "build" / "bin" / "llama-quantize",
        root / "build" / "bin" / "Release" / "llama-quantize.exe",
        root / "build" / "bin" / "llama-quantize.exe",
        root / "quantize",
        root / "quantize.exe",
    ]
    quantizer = next((c for c in candidates if c.exists()), None)

    if not converter.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found under {root}")
    if quantizer is None:
        raise FileNotFoundError(
            f"llama-quantize executable not found under {root}. Build llama.cpp first."
        )

    return {"converter": str(converter), "quantizer": str(quantizer)}


def build_conversion_commands(config: QuantizeConfig, tools: dict) -> List[List[str]]:
    """Returns the [convert, quantize] command lists for the GGUF pipeline."""
    Path(config.export_dir).mkdir(parents=True, exist_ok=True)
    convert_cmd = [
        sys.executable,
        tools["converter"],
        config.merged_dir,
        "--outfile",
        config.f16_gguf,
        "--outtype",
        "f16",
    ]
    quantize_cmd = [
        tools["quantizer"],
        config.f16_gguf,
        config.quant_gguf,
        config.quant_type,
    ]
    return [convert_cmd, quantize_cmd]


# ─── Part 2: LoRA merge (heavy imports deferred) ──────────────────────────────

def merge_adapter(config: QuantizeConfig) -> None:
    """
    Loads the base model + PEFT adapter, merges the LoRA weights, and saves a
    standard HF directory ready for GGUF conversion.

    Note: merging runs in fp16 on CPU/GPU and needs ~6-8 GB of free RAM/VRAM for
    a 3B model. It is intentionally NOT 4-bit (quantization happens later in GGUF).
    """
    adapter_path = Path(config.adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found: {adapter_path}. Run generation/train_lora.py (D-3) first."
        )

    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    logger.info("Loading base model %s for merge...", config.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Attaching LoRA adapter from %s...", adapter_path)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()

    merged_path = Path(config.merged_dir)
    merged_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_path), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_path))
    logger.info("Merged model saved to %s", merged_path)


# ─── Part 3: llama.cpp conversion & quantization ──────────────────────────────

def run_commands(commands: List[List[str]]) -> None:
    """Executes each command in sequence, stopping on the first failure."""
    for cmd in commands:
        printable = " ".join(str(part) for part in cmd)
        logger.info("Running: %s", printable)
        subprocess.run(cmd, check=True)


# ─── Part 4: CLI ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinSight D-4 GGUF quantization.")
    parser.add_argument("--base-model", default=QuantizeConfig.base_model)
    parser.add_argument("--adapter-dir", default=QuantizeConfig.adapter_dir)
    parser.add_argument("--merged-dir", default=QuantizeConfig.merged_dir)
    parser.add_argument("--export-dir", default=QuantizeConfig.export_dir)
    parser.add_argument("--llama-cpp-dir", default=None)
    parser.add_argument("--quant-type", default=QuantizeConfig.quant_type)
    parser.add_argument("--model-basename", default=QuantizeConfig.model_basename)
    parser.add_argument("--merge-only", action="store_true", help="Stop after merging the adapter.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = QuantizeConfig(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        merged_dir=args.merged_dir,
        export_dir=args.export_dir,
        llama_cpp_dir=args.llama_cpp_dir,
        quant_type=args.quant_type,
        model_basename=args.model_basename,
    )

    print("=" * 78)
    print("FinSight D-4: Merge LoRA + Export GGUF (llama.cpp)")
    print("=" * 78)
    print(f"  Base model   : {config.base_model}")
    print(f"  Adapter dir  : {config.adapter_dir}")
    print(f"  Merged dir   : {config.merged_dir}")
    print(f"  Export dir   : {config.export_dir}")
    print(f"  Quant type   : {config.quant_type}")

    if args.dry_run:
        print("\n[DRY RUN] Planned pipeline:")
        print(f"  1. merge  -> {config.merged_dir}")
        if not args.merge_only:
            try:
                tools = find_llama_cpp_binaries(config.llama_cpp_dir)
                for cmd in build_conversion_commands(config, tools):
                    print("  2/3. " + " ".join(str(part) for part in cmd))
            except FileNotFoundError as exc:
                print(f"  [llama.cpp not available] {exc}")
        return 0

    try:
        merge_adapter(config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if args.merge_only:
        print(f"\n[OK] Merge complete: {config.merged_dir} (skipped GGUF export).")
        return 0

    try:
        tools = find_llama_cpp_binaries(config.llama_cpp_dir)
        commands = build_conversion_commands(config, tools)
        run_commands(commands)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except subprocess.CalledProcessError as exc:
        logger.error("llama.cpp command failed (exit %s).", exc.returncode)
        return 1

    print(f"\n[OK] GGUF ready: {config.quant_gguf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
