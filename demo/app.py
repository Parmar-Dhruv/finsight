"""
FinSight - Minimal Streamlit Test UI (demo/app.py)

A deliberately small interface to exercise the existing retrieval pipeline:
  * query input + metadata filters (ticker / fiscal_year / section)
  * dense-only vs. hybrid (BM25+RRF) vs. + cross-encoder rerank
  * retrieved passages with scores and provenance
  * the attributed LLM context block produced for generation

This is a base for manual testing. The full frontend (attributed answers,
citation highlights, GGUF inference) is planned for a later phase.

Run:
    pip install -r demo/requirements.txt
    streamlit run demo/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="FinSight - Retrieval Test UI", layout="wide")

TICKERS = ["(any)", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA"]
YEARS = ["(any)", 2023, 2024]
SECTIONS = ["(any)", "item_1", "item_1a", "item_7", "item_8"]


@st.cache_resource(show_spinner="Loading retrieval engine (embeddings + Qdrant)...")
def get_retriever():
    """Warms the FinSightRetriever singleton once per Streamlit session."""
    from retrieval.search import FinSightRetriever

    return FinSightRetriever.get_instance()


def main() -> None:
    st.title("FinSight - Retrieval Test UI")
    st.caption(
        "Minimal harness over retrieval/search.py. Use it to verify retrieval "
        "quality and the attributed context block before generation."
    )

    with st.sidebar:
        st.header("Filters")
        ticker = st.selectbox("Ticker", TICKERS, index=0)
        fiscal_year = st.selectbox("Fiscal year", YEARS, index=0)
        section = st.selectbox("Section", SECTIONS, index=0)

        st.header("Retrieval mode")
        top_k = st.slider("Top-K", min_value=1, max_value=10, value=5)
        hybrid = st.checkbox("Hybrid (dense + BM25 RRF)", value=False)
        rerank = st.checkbox("Cross-encoder rerank", value=False)

    query = st.text_input(
        "Question",
        value="What were Apple's total net sales in fiscal 2024?",
    )

    if st.button("Search", type="primary") and query.strip():
        try:
            retriever = get_retriever()
            chunks = retriever.retrieve(
                query=query,
                k=top_k,
                ticker=None if ticker == "(any)" else ticker,
                fiscal_year=None if fiscal_year == "(any)" else int(fiscal_year),
                section=None if section == "(any)" else section,
                hybrid=hybrid,
                rerank=rerank,
            )
        except Exception as exc:  # surface setup errors clearly
            st.error(f"Retrieval failed: {exc}")
            st.info(
                "Check that `.env` has valid QDRANT_URL / QDRANT_API_KEY (or that a "
                "local index exists at data/processed/qdrant_db/), and that the "
                "retrieval dependencies from requirements.txt are installed."
            )
            return

        mode = "dense"
        if hybrid:
            mode += " + BM25"
        if rerank:
            mode += " + rerank"
        st.success(f"Retrieved {len(chunks)} passage(s)  |  mode: {mode}")

        if not chunks:
            st.warning("No passages returned. Try removing filters or rephrasing.")
            return

        for rank, chunk in enumerate(chunks, 1):
            header = (
                f"[{rank}] {chunk['ticker']} FY{chunk['fiscal_year']} "
                f"{chunk['section']}  |  score {chunk['score']:.4f}"
            )
            with st.expander(header, expanded=rank == 1):
                st.caption(chunk.get("chunk_id", ""))
                st.write(chunk["text"])

        with st.expander("Attributed LLM context block"):
            from retrieval.search import format_context_for_prompt

            st.code(format_context_for_prompt(chunks, max_tokens=2000), language="text")


if __name__ == "__main__":
    main()
