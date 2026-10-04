import json
import os
import sys

def main():
    dataset_dir = os.path.join("evaluation", "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    print("=" * 60)
    print("DOWNLOADING virattt/financial-qa-10K FROM HUGGING FACE")
    print("=" * 60)

    try:
        from datasets import load_dataset
        print("Loading dataset via Hugging Face 'datasets' library...")
        ds = load_dataset("virattt/financial-qa-10K")
        
        # Convert to list of dicts
        split_data = ds['train'] if 'train' in ds else ds[list(ds.keys())[0]]
        records = [dict(row) for row in split_data]
        print(f"Successfully loaded {len(records)} records!")

    except Exception as e:
        print(f"Hugging Face datasets library not available or error: {e}")
        print("Falling back to pandas / pyarrow or direct HuggingFace download...")
        
        try:
            import pandas as pd
            url = "https://huggingface.co/datasets/virattt/financial-qa-10K/resolve/main/data/train-00000-of-00001.parquet"
            print(f"Downloading parquet from {url}...")
            df = pd.read_parquet(url)
            records = df.to_dict(orient="records")
            print(f"Successfully loaded {len(records)} records via pandas!")
        except Exception as e2:
            print(f"Pandas download failed: {e2}")
            sys.exit(1)

    # Output paths
    output_json = os.path.join(dataset_dir, "financial_qa_10k.json")
    output_jsonl = os.path.join(dataset_dir, "financial_qa_10k.jsonl")

    # 1. Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # 2. Save JSONL
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved master dataset to:")
    print(f" - {output_json}")
    print(f" - {output_jsonl}")

    # Remove old dataset files as requested
    old_files = ["train.json", "dev.json", "test.json", "sec_10k_target_companies_qa.json", "questions.json"]
    for old_f in old_files:
        old_path = os.path.join(dataset_dir, old_f)
        if os.path.exists(old_path):
            os.remove(old_path)
            print(f"Removed old dataset: {old_f}")

    print("\nSUCCESS: Dataset replaced with virattt/financial-qa-10K!")

if __name__ == "__main__":
    main()
