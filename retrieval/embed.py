"""
FinSight — Retrieval & Data Layer
Part 1: Text Embedding Engine

This module defines the EmbeddingEngine class responsible for transforming financial
text chunks and search queries into dense vector representations.

Model: BAAI/bge-small-en-v1.5
Dimensions: 384
Metric: Cosine Similarity via L2 Normalized Inner Product (u . v)
Query Instruction: "Represent this sentence for searching relevant passages: "
"""

import logging
from typing import List, Optional, Union

import numpy as np

# Set up logging for clean operational feedback
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)



class FastBertEmbedder:
    """
    Lightweight, pure NumPy BERT forward pass for BGE-small-en-v1.5.
    Loads cached safetensors weights directly with zero PyTorch / DLL dependencies.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        import math
        from pathlib import Path
        from tokenizers import Tokenizer
        from safetensors import safe_open

        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        model_slug = "models--" + model_name.replace("/", "--")
        model_dir = hf_cache / model_slug
        snapshot_dir = next((model_dir / "snapshots").iterdir())
        safetensors_path = snapshot_dir / "model.safetensors"
        tokenizer_path = snapshot_dir / "tokenizer.json"

        self.w = {}
        with safe_open(str(safetensors_path), framework="numpy") as f:
            for k in f.keys():
                self.w[k] = f.get_tensor(k)

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=512)
        self.scale = 1.0 / math.sqrt(32)

    def _layer_norm(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return weight * (x - mean) / np.sqrt(var + eps) + bias

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        import math
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False) -> np.ndarray:
        all_vecs = []
        total = len(texts)
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer.encode_batch(batch)
            input_ids = np.array([e.ids for e in enc], dtype=np.int64)
            attn_mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            seq_len = input_ids.shape[1]
            pos_ids = np.arange(seq_len, dtype=np.int64)[None, :]

            x = (
                self.w["embeddings.word_embeddings.weight"][input_ids]
                + self.w["embeddings.position_embeddings.weight"][pos_ids]
                + self.w["embeddings.token_type_embeddings.weight"][np.zeros_like(input_ids)]
            )
            x = self._layer_norm(x, self.w["embeddings.LayerNorm.weight"], self.w["embeddings.LayerNorm.bias"])

            B, T, H = x.shape
            for l in range(12):
                pfx = f"encoder.layer.{l}.attention.self"
                Q = (x @ self.w[f"{pfx}.query.weight"].T + self.w[f"{pfx}.query.bias"]).reshape(B, T, 12, 32).transpose(0, 2, 1, 3)
                K = (x @ self.w[f"{pfx}.key.weight"].T + self.w[f"{pfx}.key.bias"]).reshape(B, T, 12, 32).transpose(0, 2, 1, 3)
                V = (x @ self.w[f"{pfx}.value.weight"].T + self.w[f"{pfx}.value.bias"]).reshape(B, T, 12, 32).transpose(0, 2, 1, 3)

                scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
                mask = (1.0 - attn_mask[:, None, None, :].astype(np.float32)) * -10000.0
                attn = self._softmax(scores + mask)
                context = np.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, T, H)

                out_pfx = f"encoder.layer.{l}.attention.output"
                context = context @ self.w[f"{out_pfx}.dense.weight"].T + self.w[f"{out_pfx}.dense.bias"]
                x = self._layer_norm(x + context, self.w[f"{out_pfx}.LayerNorm.weight"], self.w[f"{out_pfx}.LayerNorm.bias"])

                ffn_pfx = f"encoder.layer.{l}"
                h = self._gelu(x @ self.w[f"{ffn_pfx}.intermediate.dense.weight"].T + self.w[f"{ffn_pfx}.intermediate.dense.bias"])
                h = h @ self.w[f"{ffn_pfx}.output.dense.weight"].T + self.w[f"{ffn_pfx}.output.dense.bias"]
                x = self._layer_norm(x + h, self.w[f"{ffn_pfx}.output.LayerNorm.weight"], self.w[f"{ffn_pfx}.output.LayerNorm.bias"])

            cls_emb = x[:, 0, :]
            norms = np.linalg.norm(cls_emb, axis=-1, keepdims=True)
            all_vecs.append((cls_emb / np.maximum(norms, 1e-8)).astype(np.float32))

            if show_progress_bar and total > batch_size:
                print(f"  [Embedder] Encoded {min(i + batch_size, total)}/{total} chunks", end="\r", flush=True)

        if show_progress_bar and total > batch_size:
            print()
        return np.vstack(all_vecs) if all_vecs else np.empty((0, 384), dtype=np.float32)


class EmbeddingEngine:
    """
    High-performance embedding engine for FinSight.
    Uses cached weights and pure NumPy execution for zero-DLL-lock stability on Windows.
    """

    DEFAULT_MODEL: str = "BAAI/bge-small-en-v1.5"
    BGE_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or "cpu"
        self.embedding_dim = 384

        try:
            logger.info(f"Initializing FastBertEmbedder for '{model_name}'...")
            self.model = FastBertEmbedder(model_name=model_name)
            logger.info(f"[OK] EmbeddingEngine loaded via pure NumPy engine (vector dim={self.embedding_dim})")
        except Exception as e:
            logger.warning(f"FastBertEmbedder unavailable ({e}), falling back to sentence_transformers...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=self.device)

    def encode_passages(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        if isinstance(self.model, FastBertEmbedder):
            return self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)

        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def encode_queries(
        self,
        queries: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        if isinstance(queries, str):
            queries = [queries]

        if not queries:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        is_bge = "bge" in self.model_name.lower()
        if is_bge:
            prefixed = [f"{self.BGE_QUERY_PREFIX}{q.strip()}" for q in queries]
        else:
            prefixed = [q.strip() for q in queries]

        if isinstance(self.model, FastBertEmbedder):
            return self.model.encode(prefixed, batch_size=batch_size, show_progress_bar=show_progress_bar)

        return self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)



# ─── Part 2: Batch Pipeline for Corpus Chunks ─────────────────────────────────

import json
from pathlib import Path


def embed_all_chunks(
    chunks_file: Path,
    output_embeddings: Path,
    output_ids: Path,
    engine: Optional[EmbeddingEngine] = None,
    batch_size: int = 64,
) -> tuple[np.ndarray, List[str]]:
    """
    Loads all chunk records from chunks_file, encodes their text with EmbeddingEngine,
    and persists the resulting dense vectors (.npy) and chunk IDs (.json).

    Parameters:
        chunks_file: Path to processed chunks.json.
        output_embeddings: Path to save the NumPy binary array (.npy).
        output_ids: Path to save the ordered chunk ID mapping (.json).
        engine: An existing EmbeddingEngine instance (instantiates default if None).
        batch_size: Mini-batch size for transformer forward pass.

    Returns:
        tuple[np.ndarray, List[str]]: (embeddings_matrix, ordered_chunk_ids)
    """
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found at: {chunks_file}")

    logger.info(f"Loading chunk dataset from {chunks_file}...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    total_chunks = len(chunks_data)
    logger.info(f"Total chunks to embed: {total_chunks}")

    chunk_ids = [c["chunk_id"] for c in chunks_data]
    texts = [c["text"] for c in chunks_data]

    if engine is None:
        engine = EmbeddingEngine()

    logger.info(f"Generating dense embeddings using '{engine.model_name}' (batch_size={batch_size})...")
    embeddings = engine.encode_passages(
        texts=texts,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    # Persist the embedding matrix
    output_embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_embeddings, embeddings)
    logger.info(f"[OK] Saved embedding matrix to {output_embeddings} (shape: {embeddings.shape})")

    # Persist the 1:1 mapped chunk IDs
    with open(output_ids, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, indent=2)
    logger.info(f"[OK] Saved chunk ID index to {output_ids}")

    return embeddings, chunk_ids


# ─── Part 3: Verification & Sanity Search ─────────────────────────────────────

def verify_embedding_dataset(
    embeddings_file: Path,
    output_ids_file: Path,
    chunks_file: Path,
) -> None:
    """
    Validates integrity of generated embeddings:
    1. Shape matching total chunks
    2. Data type is float32
    3. L2 norm of all vectors is 1.0 +- 1e-4
    4. 1:1 parity between vectors, chunk IDs, and chunks.json
    """
    print("\n" + "=" * 70)
    print("VERIFICATION & INTEGRITY REPORT")
    print("=" * 70)

    embeddings = np.load(embeddings_file)
    with open(output_ids_file, "r", encoding="utf-8") as f:
        chunk_ids = json.load(f)
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    num_vectors, dim = embeddings.shape
    file_size_mb = embeddings_file.stat().st_size / (1024 * 1024)

    print(f"Total Vector Rows:       {num_vectors}")
    print(f"Vector Dimensions:       {dim}")
    print(f"Data Type:               {embeddings.dtype}")
    print(f"File Size on Disk:       {file_size_mb:.2f} MB")
    print(f"Total Chunk IDs:         {len(chunk_ids)}")
    print(f"Total Source Chunks:     {len(chunks_data)}")

    # Check alignment
    assert num_vectors == len(chunk_ids) == len(chunks_data), "Mismatched record counts!"
    print("[OK] 1:1 Vector-to-Chunk alignment confirmed.")

    # Check L2 Norms
    norms = np.linalg.norm(embeddings, axis=1)
    norm_min, norm_max, norm_avg = float(norms.min()), float(norms.max()), float(norms.mean())
    print(f"Vector L2 Norms:         min={norm_min:.6f}, max={norm_max:.6f}, avg={norm_avg:.6f}")
    assert np.allclose(norms, 1.0, atol=1e-4), "Vectors are not properly L2 normalized!"
    print("[OK] Unit normalization verified (all ||v||_2 = 1.0).")
    print("=" * 70)


def raw_cosine_similarity_search(
    query: str,
    engine: EmbeddingEngine,
    embeddings: np.ndarray,
    chunk_ids: List[str],
    chunks_lookup: dict[str, dict],
    top_k: int = 3,
) -> None:
    """
    Demonstrates pure NumPy dot product retrieval (cosine similarity) without FAISS.
    Used for sanity checking retrieval relevance before building the vector index.
    """
    print(f"\nQuery: \"{query}\"")
    print("-" * 70)

    query_vec = engine.encode_queries(query, show_progress_bar=False)  # shape: (1, 384)
    # Cosine similarity is simply the dot product with normalized vectors
    scores = np.dot(embeddings, query_vec.T).flatten()  # shape: (N,)

    top_indices = np.argsort(scores)[::-1][:top_k]

    for rank, idx in enumerate(top_indices, 1):
        cid = chunk_ids[idx]
        score = scores[idx]
        chunk_info = chunks_lookup.get(cid, {})
        ticker = chunk_info.get("ticker", "UNKNOWN")
        year = chunk_info.get("fiscal_year", "UNKNOWN")
        section = chunk_info.get("section", "UNKNOWN")
        snippet = chunk_info.get("text", "").replace("\n", " ")[:140]

        print(f" Rank {rank} | Score: {score:.4f} | [{ticker} FY{year} {section}]")
        print(f"   Chunk ID: {cid}")
        print(f"   Snippet:  \"{snippet}...\"\n")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    CHUNKS_FILE = PROCESSED_DIR / "chunks.json"
    EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.npy"
    CHUNK_IDS_FILE = PROCESSED_DIR / "chunk_ids.json"

    print("=" * 70)
    print("FinSight: 10-K Chunk Embedding Pipeline (retrieval/embed.py)")
    print("=" * 70)

    # 1. Initialize Engine
    engine = EmbeddingEngine()

    # 2. Batch embed all chunks
    embeddings, chunk_ids = embed_all_chunks(
        chunks_file=CHUNKS_FILE,
        output_embeddings=EMBEDDINGS_FILE,
        output_ids=CHUNK_IDS_FILE,
        engine=engine,
        batch_size=64,
    )

    # 3. Verification Report
    verify_embedding_dataset(EMBEDDINGS_FILE, CHUNK_IDS_FILE, CHUNKS_FILE)

    # 4. Sanity Semantic Search Test
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks_raw = json.load(f)
    chunks_lookup = {c["chunk_id"]: c for c in chunks_raw}

    test_queries = [
        "What was Apple's total revenue and iPhone sales?",
        "What are the major cloud and cybersecurity risks for Microsoft?",
    ]

    print("\n" + "=" * 70)
    print("SANITY RETRIEVAL TEST (Raw NumPy Cosine Similarity)")
    print("=" * 70)
    for q in test_queries:
        raw_cosine_similarity_search(
            query=q,
            engine=engine,
            embeddings=embeddings,
            chunk_ids=chunk_ids,
            chunks_lookup=chunks_lookup,
            top_k=3,
        )
