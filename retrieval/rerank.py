"""
================================================================================
FinSight — Retrieval & Data Layer
Cross-Encoder Reranker (retrieval/rerank.py)
================================================================================

Task N-4 (docs/finsight-task-breakdown.md):
  "Add a cross-encoder reranking step on top-k candidates before they're
   returned from retrieve(), using sentence-transformers (already a dependency).
   Re-run evaluation with reranking on vs. off."

Implementation Note:
  The standard sentence_transformers CrossEncoder import triggers PyTorch,
  which is blocked by Windows Application Control policy on this machine
  (same issue that led to FastBertEmbedder in embed.py).
  This module implements FastCrossEncoder — a pure NumPy / safetensors forward
  pass of cross-encoder/ms-marco-MiniLM-L-6-v2, with zero PyTorch dependency.

Architecture (MiniLM-L-6-v2 cross-encoder):
  - Vocab size: 30522  |  Hidden: 384  |  Heads: 12  |  Layers: 6
  - Intermediate (FFN): 1536
  - Classifier head: Linear(384, 1)
  - Input: [CLS] query [SEP] passage [SEP]  -> scalar relevance score
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _find_snapshot_dir(model_name: str) -> Path:
    """Resolves the HuggingFace hub snapshot directory for a model."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_slug = "models--" + model_name.replace("/", "--")
    model_dir = hf_cache / model_slug
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model '{model_name}' not found in HuggingFace cache at {model_dir}.\n"
            f"Download it with: huggingface_hub.snapshot_download('{model_name}')"
        )
    snapshots = list((model_dir / "snapshots").iterdir())
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found for model '{model_name}'")
    return snapshots[0]


class FastCrossEncoder:
    """
    Pure NumPy cross-encoder for relevance scoring.

    Loads cross-encoder/ms-marco-MiniLM-L-6-v2 weights directly from
    HuggingFace safetensors cache and runs the forward pass via NumPy --
    zero PyTorch / DLL dependency.

    Usage:
        ce = FastCrossEncoder()
        scores = ce.predict([["query text", "passage text"], ...])
        # scores: np.ndarray of shape (N,), higher = more relevant
    """

    NUM_LAYERS = 6
    HIDDEN = 384
    NUM_HEADS = 12
    HEAD_DIM = 32       # HIDDEN // NUM_HEADS
    INTERMEDIATE = 1536
    MAX_LENGTH = 512

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        logger.info(f"Initializing FastCrossEncoder from '{model_name}'...")
        snap = _find_snapshot_dir(model_name)

        from safetensors import safe_open
        from tokenizers import Tokenizer

        # Load weights
        st_path = snap / "model.safetensors"
        if not st_path.exists():
            raise FileNotFoundError(
                f"model.safetensors not found at {st_path}. "
                "Ensure the model was downloaded with safetensors files."
            )

        self.w: Dict[str, np.ndarray] = {}
        with safe_open(str(st_path), framework="numpy") as f:
            for k in f.keys():
                self.w[k] = f.get_tensor(k)

        # Load tokenizer
        tok_path = snap / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=None)
        self.tokenizer.enable_truncation(max_length=self.MAX_LENGTH)

        self._scale = 1.0 / math.sqrt(self.HEAD_DIM)
        logger.info("[OK] FastCrossEncoder ready.")

    # -- Activations ----------------------------------------------------------

    @staticmethod
    def _layer_norm(
        x: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return weight * (x - mean) / np.sqrt(var + eps) + bias

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    # -- Forward pass ---------------------------------------------------------

    def _forward(
        self,
        input_ids: np.ndarray,
        attn_mask: np.ndarray,
        token_type_ids: np.ndarray,
    ) -> np.ndarray:
        """Returns CLS hidden state of shape (B, HIDDEN)."""
        B, T = input_ids.shape
        pos_ids = np.arange(T, dtype=np.int64)[None, :]

        # Embedding layer (all keys prefixed with 'bert.')
        emb = "bert.embeddings"
        x = (
            self.w[f"{emb}.word_embeddings.weight"][input_ids]
            + self.w[f"{emb}.position_embeddings.weight"][pos_ids]
            + self.w[f"{emb}.token_type_embeddings.weight"][token_type_ids]
        )
        x = self._layer_norm(
            x,
            self.w[f"{emb}.LayerNorm.weight"],
            self.w[f"{emb}.LayerNorm.bias"],
        )

        # Attention mask broadcast: (B, 1, 1, T)
        mask_4d = (1.0 - attn_mask[:, None, None, :].astype(np.float32)) * -10000.0

        # Encoder layers
        for l in range(self.NUM_LAYERS):
            pfx = f"bert.encoder.layer.{l}.attention.self"
            Q = (x @ self.w[f"{pfx}.query.weight"].T + self.w[f"{pfx}.query.bias"]).reshape(B, T, self.NUM_HEADS, self.HEAD_DIM).transpose(0, 2, 1, 3)
            K = (x @ self.w[f"{pfx}.key.weight"].T   + self.w[f"{pfx}.key.bias"]).reshape(B, T, self.NUM_HEADS, self.HEAD_DIM).transpose(0, 2, 1, 3)
            V = (x @ self.w[f"{pfx}.value.weight"].T + self.w[f"{pfx}.value.bias"]).reshape(B, T, self.NUM_HEADS, self.HEAD_DIM).transpose(0, 2, 1, 3)

            scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self._scale + mask_4d
            attn = self._softmax(scores)
            ctx = np.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, T, self.HIDDEN)

            out_pfx = f"bert.encoder.layer.{l}.attention.output"
            ctx = ctx @ self.w[f"{out_pfx}.dense.weight"].T + self.w[f"{out_pfx}.dense.bias"]
            x = self._layer_norm(
                x + ctx,
                self.w[f"{out_pfx}.LayerNorm.weight"],
                self.w[f"{out_pfx}.LayerNorm.bias"],
            )

            ffn = f"bert.encoder.layer.{l}"
            h = self._gelu(x @ self.w[f"{ffn}.intermediate.dense.weight"].T + self.w[f"{ffn}.intermediate.dense.bias"])
            h = h @ self.w[f"{ffn}.output.dense.weight"].T + self.w[f"{ffn}.output.dense.bias"]
            x = self._layer_norm(
                x + h,
                self.w[f"{ffn}.output.LayerNorm.weight"],
                self.w[f"{ffn}.output.LayerNorm.bias"],
            )

        return x[:, 0, :]  # CLS token: (B, HIDDEN)

    def predict(
        self,
        pairs: List[List[str]],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Score a list of [query, passage] pairs.

        Parameters:
            pairs: List of [query_str, passage_str] pairs.
            batch_size: Mini-batch size for NumPy forward pass.

        Returns:
            np.ndarray of shape (N,) -- raw logit scores (higher = more relevant).
        """
        if not pairs:
            return np.array([], dtype=np.float32)

        all_scores: List[float] = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]

            # Tokenize as sentence-pair: [CLS] query [SEP] passage [SEP]
            encodings = self.tokenizer.encode_batch(
                [(q, p) for q, p in batch]
            )

            input_ids    = np.array([e.ids for e in encodings], dtype=np.int64)
            attn_mask    = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            type_ids     = np.array([e.type_ids for e in encodings], dtype=np.int64)

            cls_hidden = self._forward(input_ids, attn_mask, type_ids)  # (B, HIDDEN)

            # Classifier head: Linear(HIDDEN -> 1)
            if "classifier.weight" in self.w:
                logits = cls_hidden @ self.w["classifier.weight"].T + self.w["classifier.bias"]
            else:
                logits = cls_hidden @ self.w["linear.weight"].T + self.w["linear.bias"]

            all_scores.extend(logits[:, 0].tolist())

        return np.array(all_scores, dtype=np.float32)


# -- Singleton accessor -------------------------------------------------------

_INSTANCE: Optional[FastCrossEncoder] = None


def get_cross_encoder(model_name: str = MODEL_NAME) -> FastCrossEncoder:
    """Singleton accessor -- initializes once, reuses across calls."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = FastCrossEncoder(model_name=model_name)
    return _INSTANCE


# -- Standalone smoke test ----------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("FastCrossEncoder -- Smoke Test")
    print("=" * 60)

    ce = get_cross_encoder()

    test_pairs = [
        ["What was Apple's total revenue in 2024?",
         "Apple's net sales for fiscal year 2024 were $391.0 billion."],
        ["What was Apple's total revenue in 2024?",
         "Microsoft reported cloud revenue growth of 22% in fiscal 2024."],
        ["What drove NVIDIA's data center revenue growth?",
         "NVIDIA's Data Center revenue grew 217% to $47.5 billion driven by AI GPU demand."],
        ["What drove NVIDIA's data center revenue growth?",
         "Amazon's AWS segment reported net sales of $107.6 billion in 2024."],
    ]

    scores = ce.predict(test_pairs)

    for (q, p), s in zip(test_pairs, scores):
        print(f"\nScore: {s:+.4f}")
        print(f"  Q: {q[:60]}")
        print(f"  P: {p[:70]}...")

    print("\n[OK] FastCrossEncoder smoke test complete.")
