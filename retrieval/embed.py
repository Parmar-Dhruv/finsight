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


class EmbeddingEngine:
    """
    Wrapper around Hugging Face / Sentence-Transformers embedding models.
    
    Provides specialized methods for encoding document passages and search queries
    with proper normalization and query instruction prefixes.
    """

    DEFAULT_MODEL: str = "BAAI/bge-small-en-v1.5"
    BGE_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        """
        Initializes the embedding model on the specified device.

        Parameters:
            model_name: HuggingFace model identifier (default: 'BAAI/bge-small-en-v1.5').
            device: 'cuda', 'cpu', or None (auto-detects CUDA if available).
        """
        # Lazy import of torch and SentenceTransformer to allow fast module import
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading embedding model '{model_name}' onto device: {self.device}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        # SentenceTransformers >= 3.0 renamed get_sentence_embedding_dimension to get_embedding_dimension
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded successfully. Vector dimension: {self.embedding_dim}")

    def encode_passages(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """
        Encodes a list of document passage texts into dense vectors.
        
        Passages are encoded directly without query instruction prefixes,
        and vectors are L2-normalized so that cosine similarity equals dot product.

        Parameters:
            texts: List of document text chunks to encode.
            batch_size: Number of texts processed in each forward pass.
            show_progress_bar: Whether to display a progress indicator.

        Returns:
            np.ndarray: Matrix of shape (len(texts), embedding_dim) as float32.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,  # Crucial: L2 norm = 1.0 for fast dot-product cosine similarity
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_queries(
        self,
        queries: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encodes user queries into dense vectors using BGE's asymmetric query instruction.

        Parameters:
            queries: A single query string or a list of query strings.
            batch_size: Batch size for encoding.
            show_progress_bar: Whether to display a progress bar.

        Returns:
            np.ndarray: Matrix of shape (num_queries, embedding_dim) as float32.
        """
        if isinstance(queries, str):
            queries = [queries]

        if not queries:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        # Apply BGE query instruction prefix for asymmetric search
        is_bge = "bge" in self.model_name.lower()
        if is_bge:
            prefixed_queries = [f"{self.BGE_QUERY_PREFIX}{q.strip()}" for q in queries]
        else:
            prefixed_queries = [q.strip() for q in queries]

        embeddings = self.model.encode(
            prefixed_queries,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)



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
