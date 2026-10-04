# Financial 10-K QA Dataset (virattt/financial-qa-10K)

**Source:** [virattt/financial-qa-10K on Hugging Face](https://huggingface.co/datasets/virattt/financial-qa-10K)  
**Total Records:** **7,000 real 10-K filing Question-Answer pairs**  
**Data Format:** JSON & JSONL (`financial_qa_10k.json`, `financial_qa_10k.jsonl`)

---

## 📑 Dataset Overview

This dataset contains **7,000 real-world question-answer pairs** extracted directly from SEC 10-K annual filings of major companies (including NVDA, AAPL, MSFT, AMZN, GOOGL, META, and other tech/S&P 500 leaders).

Each record provides:
- **`question`**: A clear financial or operational query.
- **`answer`**: The precise, ground-truth answer.
- **`context`**: The exact SEC 10-K text excerpt / filing snippet supporting the answer.
- **`ticker`**: Company stock symbol (e.g., `NVDA`, `AAPL`, `MSFT`).
- **`filing`**: Fiscal year and filing type (e.g., `2023_10K`).

---

## 🔍 Record Structure Example

```json
{
  "question": "What significant invention did NVIDIA create in 1999?",
  "answer": "NVIDIA invented the GPU in 1999.",
  "context": "Our invention of the GPU in 1999 defined modern computer graphics and established NVIDIA as the leader in computer graphics.",
  "ticker": "NVDA",
  "filing": "2023_10K"
}
```

---

## 🎯 Primary Usage in FinSight

- **Fine-Tuning (Task D-1)**: Used as the master instruction fine-tuning dataset for QLoRA fine-tuning of Llama 3.2 3B / Qwen-2.5 3B.
- **Retrieval Augmented Generation (Task B-1 / C-1)**: Serves as ground-truth evaluation pairs for RAG context extraction and accuracy evaluation.
