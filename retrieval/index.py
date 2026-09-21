"""
FinSight — Retrieval & Data Layer
Part 1: Vector Database Index Manager (Qdrant)

This module implements the QdrantIndexManager class, providing a unified interface
for vector database operations supporting both:
  1. Local embedded mode (zero-Docker, stored on disk in data/processed/qdrant_db)
  2. Cloud mode (Qdrant Cloud managed cluster via URL & API key)

Collection: finsight_10k
Vector Dimension: 384 (BAAI/bge-small-en-v1.5)
Distance Metric: Cosine (equivalent to normalized Dot Product)
Payload Indices: ticker, fiscal_year, section (for single-stage filtered search)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class QdrantIndexManager:
    """
    Manages the lifecycle, schema, and configuration of the FinSight vector index.
    Supports seamless switching between local disk storage and Qdrant Cloud.
    """

    DEFAULT_COLLECTION: str = "finsight_10k"
    VECTOR_DIMENSION: int = 384  # BGE-small embedding dimension

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        local_path: Optional[Path] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initializes the Qdrant client in either local embedded mode or cloud mode.

        Parameters:
            collection_name: Name of the vector collection.
            local_path: Path for local embedded storage. If provided (or if cloud env
                        vars are missing), Qdrant runs embedded on disk with no Docker.
            url: Qdrant Cloud cluster URL (e.g., https://xyz.cloud.qdrant.io).
            api_key: Qdrant Cloud API key.
        """
        self.collection_name = collection_name

        # Priority 1: Cloud credentials (passed explicitly or via environment variables)
        cloud_url = url or os.getenv("QDRANT_URL")
        cloud_key = api_key or os.getenv("QDRANT_API_KEY")

        if cloud_url and cloud_key:
            logger.info(f"Connecting to Qdrant Cloud at: {cloud_url}")
            self.client = QdrantClient(url=cloud_url, api_key=cloud_key)
            self.mode = "cloud"
        else:
            # Priority 2: Local embedded on disk (ideal for development & teammate handoff)
            if local_path is None:
                base_dir = Path(__file__).resolve().parent.parent
                local_path = base_dir / "data" / "processed" / "qdrant_db"

            local_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing local embedded Qdrant at: {local_path}")
            self.client = QdrantClient(path=str(local_path))
            self.mode = "local"

    def init_collection(self, recreate: bool = False) -> None:
        """
        Creates or validates the collection with proper vector dimension and payload indices.

        Parameters:
            recreate: If True, deletes any existing collection with the same name.
        """
        collections_response = self.client.get_collections()
        existing_names = [c.name for c in collections_response.collections]

        if self.collection_name in existing_names:
            if recreate:
                logger.warning(f"Recreating collection '{self.collection_name}'...")
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(f"Collection '{self.collection_name}' already exists. Skipping recreation.")
                return

        logger.info(f"Creating collection '{self.collection_name}' (dim={self.VECTOR_DIMENSION}, metric=COSINE)...")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.VECTOR_DIMENSION,
                distance=models.Distance.COSINE,
            ),
            # Optimize HNSW parameters for high recall in finance prose
            hnsw_config=models.HnswConfigDiff(
                m=16,               # Number of edges per node in HNSW graph
                ef_construct=100,   # Exploration depth during index build
            ),
        )

        # Create payload field indexes in cloud mode for accelerated filtered searching
        if self.mode == "cloud":
            self._create_payload_indexes()
        logger.info(f"[OK] Collection '{self.collection_name}' initialized successfully.")

    def _create_payload_indexes(self) -> None:
        """Sets up index structures on payload fields for single-stage metadata filtering in cloud mode."""
        fields_to_index = [
            ("ticker", models.PayloadSchemaType.KEYWORD),
            ("fiscal_year", models.PayloadSchemaType.INTEGER),
            ("section", models.PayloadSchemaType.KEYWORD),
        ]
        for field_name, schema_type in fields_to_index:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                logger.info(f"  [+] Created payload index for field: '{field_name}' ({schema_type})")
            except Exception as e:
                logger.warning(f"  [!] Note on payload index for '{field_name}': {e}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Returns metadata about the active collection (points count, status, vector size)."""
        info = self.client.get_collection(self.collection_name)
        return {
            "status": info.status.name if hasattr(info.status, "name") else str(info.status),
            "points_count": info.points_count,
            "vectors_count": getattr(info, "vectors_count", info.points_count),
        }

    def close(self) -> None:
        """Closes client connections cleanly."""
        try:
            self.client.close()
        except Exception:
            pass


# ─── Part 2: Batch Ingestion Pipeline ─────────────────────────────────────────

import json
import sys
import numpy as np

# Ensure project root is in sys.path for local module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.embed import EmbeddingEngine


def index_chunks(
    manager: QdrantIndexManager,
    chunks_file: Path,
    embeddings_file: Path,
    batch_size: int = 128,
) -> int:
    """
    Ingests all precomputed chunk vectors and metadata payloads into the Qdrant collection.

    Parameters:
        manager: Initialized QdrantIndexManager instance.
        chunks_file: Path to processed chunks.json.
        embeddings_file: Path to precomputed embeddings.npy.
        batch_size: Number of points uploaded per batch.

    Returns:
        int: Total number of points indexed.
    """
    logger.info(f"Loading embeddings from {embeddings_file}...")
    embeddings = np.load(embeddings_file)

    logger.info(f"Loading chunks from {chunks_file}...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    total_chunks = len(chunks_data)
    if len(embeddings) != total_chunks:
        raise ValueError(
            f"Embeddings count ({len(embeddings)}) does not match chunks count ({total_chunks})!"
        )

    logger.info(f"Upserting {total_chunks} points into collection '{manager.collection_name}' (batch_size={batch_size})...")

    # Ingest in batches to avoid overwhelming memory and HTTP buffer limits
    for start_idx in range(0, total_chunks, batch_size):
        end_idx = min(start_idx + batch_size, total_chunks)
        batch_points: List[models.PointStruct] = []

        for idx in range(start_idx, end_idx):
            chunk = chunks_data[idx]
            vector = embeddings[idx].tolist()

            # The payload contains all text and filing provenance
            payload = {
                "chunk_id": chunk["chunk_id"],
                "ticker": chunk["ticker"],
                "fiscal_year": int(chunk["fiscal_year"]),
                "section": chunk["section"],
                "chunk_index": int(chunk.get("chunk_index", 0)),
                "char_count": int(chunk.get("char_count", len(chunk["text"]))),
                "token_count_est": int(chunk.get("token_count_est", 0)),
                "text": chunk["text"],
            }

            batch_points.append(
                models.PointStruct(
                    id=idx,
                    vector=vector,
                    payload=payload,
                )
            )

        manager.client.upsert(
            collection_name=manager.collection_name,
            points=batch_points,
            wait=True,
        )
        logger.info(f"  [+] Ingested points [{start_idx:4d} -> {end_idx:4d}] / {total_chunks}")

    info = manager.get_collection_info()
    logger.info(f"[OK] Ingestion complete. Total points in collection: {info['points_count']}")
    return total_chunks


# ─── Part 3: Filtered Vector Search API ───────────────────────────────────────

def search_index(
    manager: QdrantIndexManager,
    query: str,
    engine: EmbeddingEngine,
    top_k: int = 5,
    ticker: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    section: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Executes an approximate nearest neighbor search with optional single-stage
    metadata filtering by company ticker, fiscal year, and/or 10-K section.

    Parameters:
        manager: Initialized QdrantIndexManager instance.
        query: Natural language query string.
        engine: EmbeddingEngine to encode the query.
        top_k: Number of most relevant chunks to return.
        ticker: Optional company ticker filter (e.g. 'AAPL', 'MSFT').
        fiscal_year: Optional filing year filter (e.g. 2024, 2023).
        section: Optional 10-K section filter (e.g. 'item_7', 'item_1a').

    Returns:
        List of result dictionaries containing score, chunk_id, metadata, and text.
    """
    # 1. Encode query with BGE asymmetric instruction
    query_vector = engine.encode_queries(query, show_progress_bar=False)[0].tolist()

    # 2. Build single-stage payload filter conditions
    filter_conditions: List[models.FieldCondition] = []

    if ticker:
        filter_conditions.append(
            models.FieldCondition(
                key="ticker",
                match=models.MatchValue(value=ticker.strip().upper()),
            )
        )
    if fiscal_year:
        filter_conditions.append(
            models.FieldCondition(
                key="fiscal_year",
                match=models.MatchValue(value=int(fiscal_year)),
            )
        )
    if section:
        filter_conditions.append(
            models.FieldCondition(
                key="section",
                match=models.MatchValue(value=section.strip().lower()),
            )
        )

    query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

    # 3. Perform vector search (using modern query_points API)
    search_response = manager.client.query_points(
        collection_name=manager.collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    results: List[Dict[str, Any]] = []
    for hit in search_response.points:
        payload = hit.payload or {}
        results.append(
            {
                "score": round(hit.score, 4),
                "point_id": hit.id,
                "chunk_id": payload.get("chunk_id", ""),
                "ticker": payload.get("ticker", ""),
                "fiscal_year": payload.get("fiscal_year", 0),
                "section": payload.get("section", ""),
                "text": payload.get("text", ""),
                "token_count": payload.get("token_count_est", 0),
            }
        )

    return results


def print_search_results(query: str, results: List[Dict[str, Any]], filter_label: str = "None") -> None:
    """Pretty prints search results for inspection and validation."""
    print("\n" + "-" * 75)
    print(f"Query:   \"{query}\"")
    print(f"Filter:  {filter_label}")
    print(f"Results: {len(results)} chunk(s) returned")
    print("-" * 75)

    for rank, r in enumerate(results, 1):
        # Sanitize text for Windows cp1252 terminal compatibility
        raw_snippet = r["text"].replace("\n", " ")[:135]
        snippet = raw_snippet.encode("ascii", errors="replace").decode("ascii")
        print(f" Rank {rank} | Score: {r['score']:.4f} | [{r['ticker']} FY{r['fiscal_year']} {r['section']}]")
        print(f"   Chunk ID: {r['chunk_id']}")
        print(f"   Excerpt:  \"{snippet}...\"\n")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    CHUNKS_FILE = PROCESSED_DIR / "chunks.json"
    EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.npy"

    print("=" * 75)
    print("FinSight: Qdrant Vector Database Pipeline (retrieval/index.py)")
    print("=" * 75)

    # 1. Initialize manager and recreate collection with proper schema
    manager = QdrantIndexManager()
    manager.init_collection(recreate=True)

    # 2. Ingest all 2,816 precomputed vectors + payloads into Qdrant
    index_chunks(
        manager=manager,
        chunks_file=CHUNKS_FILE,
        embeddings_file=EMBEDDINGS_FILE,
        batch_size=256,
    )

    # 3. Initialize embedding engine for real-time query encoding
    engine = EmbeddingEngine()

    print("\n" + "=" * 75)
    print("SEARCH TEST SUITE (Global vs. Filtered Single-Stage Search)")
    print("=" * 75)

    # Test 1: Global semantic search across all companies
    q1 = "What was the total net sales and annual revenue growth?"
    res1 = search_index(manager=manager, query=q1, engine=engine, top_k=3)
    print_search_results(q1, res1, filter_label="None (Global across all 10-K filings)")

    # Test 2: Filtered search strictly for Microsoft FY2023 Risk Factors
    q2 = "What are the primary risks regarding cybersecurity and cloud infrastructure?"
    res2 = search_index(
        manager=manager,
        query=q2,
        engine=engine,
        top_k=3,
        ticker="MSFT",
        fiscal_year=2023,
        section="item_1a",
    )
    print_search_results(q2, res2, filter_label="ticker='MSFT', fiscal_year=2023, section='item_1a'")

    # Test 3: Filtered search strictly for Apple FY2024 MD&A (Management Discussion)
    q3 = "What were the drivers for research and development expenses?"
    res3 = search_index(
        manager=manager,
        query=q3,
        engine=engine,
        top_k=3,
        ticker="AAPL",
        fiscal_year=2024,
        section="item_7",
    )
    print_search_results(q3, res3, filter_label="ticker='AAPL', fiscal_year=2024, section='item_7'")

    manager.close()
    print("[OK] Vector Database Pipeline & Filtered Search verified successfully!")

