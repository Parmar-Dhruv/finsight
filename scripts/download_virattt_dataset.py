"""
Download the virattt/financial-qa-10K dataset from Hugging Face.

The dataset is the primary source for the D-1 fine-tuning set
(see generation/build_finetune_dataset.py and evaluation/dataset/README.md).

The downloader intentionally NEVER touches `questions.json`: that file is the
held-out 100-question evaluation benchmark, and deleting it would silently
destroy the gold set used to measure retrieval/generation quality.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

# Legacy scratch files that may linger from earlier dataset experiments.
# NOTE: questions.json is deliberately EXCLUDED - it is the eval benchmark.
LEGACY_FILES = [
    "train.json",
    "dev.json",
    "test.json",
    "sec_10k_target_companies_qa.json",
]


def download_dataset(dataset_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetches virattt/financial-qa-10K and persists it to `dataset_dir`.

    Returns the list of records. Raises on unrecoverable download failure so
    callers (e.g. build_finetune_dataset.py) can fall back gracefully.
    """
    dataset_dir = dataset_dir or os.path.join("evaluation", "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    print("=" * 60)
    print("DOWNLOADING virattt/financial-qa-10K FROM HUGGING FACE")
    print("=" * 60)

    records: List[Dict[str, Any]]
    try:
        from datasets import load_dataset

        print("Loading dataset via Hugging Face 'datasets' library...")
        ds = load_dataset("virattt/financial-qa-10K")
        split_data = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
        records = [dict(row) for row in split_data]
        print(f"Successfully loaded {len(records)} records!")

    except Exception as exc:
        print(f"Hugging Face datasets library not available or error: {exc}")
        print("Falling back to pandas / pyarrow direct download...")
        try:
            import pandas as pd

            url = (
                "https://huggingface.co/datasets/virattt/financial-qa-10K/"
                "resolve/main/data/train-00000-of-00001.parquet"
            )
            print(f"Downloading parquet from {url}...")
            df = pd.read_parquet(url)
            records = df.to_dict(orient="records")
            print(f"Successfully loaded {len(records)} records via pandas!")
        except Exception as exc2:
            print(f"Pandas download failed: {exc2}")
            raise

    output_json = os.path.join(dataset_dir, "financial_qa_10k.json")
    output_jsonl = os.path.join(dataset_dir, "financial_qa_10k.jsonl")

    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    with open(output_jsonl, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nSaved master dataset to:")
    print(f" - {output_json}")
    print(f" - {output_jsonl}")

    # Clean up legacy scratch files only (questions.json is protected).
    for old_name in LEGACY_FILES:
        old_path = os.path.join(dataset_dir, old_name)
        if os.path.exists(old_path):
            os.remove(old_path)
            print(f"Removed legacy dataset file: {old_name}")

    print("\nSUCCESS: Dataset written to evaluation/dataset/ (benchmark preserved).")
    return records


def main() -> None:
    try:
        download_dataset()
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
