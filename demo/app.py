"""
FinSight - Streamlit Interactive QA & Monitoring Dashboard (demo/app.py)

Provides two views:
  1. FinSight Q&A & Retrieval:
     * Query input + metadata filters (ticker / fiscal_year / section)
     * Dense-only vs. Hybrid (BM25+RRF) vs. + Cross-Encoder rerank
     * Citation-backed answer generation via FinSightGenerator
     * Retrieved passages with scores and provenance
  2. Monitoring & Evaluation Dashboard (Task J-7):
     * Live FastAPI /metrics polling (request latency p50/p95, error rate, citation rate)
     * CPU Latency Profile (Task J-6)
     * J-5 Comparative Matrix & N/D-1 Error Analysis summaries

Run:
    streamlit run demo/app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

# Ensure project root is importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "evaluation" / "results"
TICKERS = ["(any)", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA"]
YEARS = ["(any)", 2023, 2024]
SECTIONS = ["(any)", "item_1", "item_1a", "item_7", "item_8"]

st.set_page_config(page_title="FinSight - Financial RAG & Monitoring", layout="wide")


@st.cache_resource(show_spinner="Loading retrieval engine (embeddings + Qdrant)...")
def get_retriever():
    """Warms the FinSightRetriever singleton once per Streamlit session."""
    from retrieval.search import FinSightRetriever

    return FinSightRetriever.get_instance()


@st.cache_resource(show_spinner="Loading FinSight generator...")
def get_generator():
    """Warms the FinSightGenerator singleton once per Streamlit session."""
    from generation.inference import FinSightGenerator

    return FinSightGenerator.get_instance()


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def render_qa_tab() -> None:
    st.subheader("FinSight SEC 10-K Q&A & Retrieval")
    st.caption(
        "Exercises retrieval/search.py (Dense / Hybrid BM25+RRF / Cross-Encoder Rerank) "
        "and generation/inference.py (citation-enforced answer synthesis)."
    )

    with st.sidebar:
        st.header("Filters")
        ticker = st.selectbox("Ticker", TICKERS, index=0)
        fiscal_year = st.selectbox("Fiscal year", YEARS, index=0)
        section = st.selectbox("Section", SECTIONS, index=0)

        st.header("Retrieval mode")
        top_k = st.slider("Top-K", min_value=1, max_value=10, value=5)
        hybrid = st.checkbox("Hybrid (dense + BM25 RRF)", value=True)
        rerank = st.checkbox("Cross-encoder rerank", value=False)
        generate_ans = st.checkbox("Generate citation-backed answer", value=True)

    query = st.text_input(
        "Question",
        value="What were Apple's total net sales in fiscal 2024?",
    )

    if st.button("Search & Answer", type="primary") and query.strip():
        t0 = time.perf_counter()
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
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            st.info(
                "Check that `.env` has valid QDRANT_URL / QDRANT_API_KEY (or that a "
                "local index exists at data/processed/qdrant_db/), and that the "
                "retrieval dependencies from requirements.txt are installed."
            )
            return

        ret_ms = (time.perf_counter() - t0) * 1000.0
        mode = "dense" + (" + BM25" if hybrid else "") + (" + rerank" if rerank else "")
        st.success(f"Retrieved {len(chunks)} passage(s) in {ret_ms:.1f} ms  |  mode: {mode}")

        if not chunks:
            st.warning("No passages returned. Try removing filters or rephrasing.")
            return

        if generate_ans:
            generator = get_generator()
            gen_out = generator.generate(question=query, context_chunks=chunks)
            st.markdown("### Generated Answer")
            if gen_out.get("is_refusal"):
                st.warning(gen_out["answer"])
            else:
                st.info(gen_out["answer"])
            st.caption(
                f"Backend: `{gen_out.get('backend')}` | "
                f"Citations: `{gen_out.get('citations')}` | "
                f"Generation Latency: `{gen_out.get('latency_ms')} ms`"
            )

        st.markdown("### Retrieved 10-K Passages")
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


def render_monitoring_tab() -> None:
    st.subheader("System Monitoring, Latency & Evaluation Dashboard (Tasks J-6 / J-7)")

    # 1. Live FastAPI /metrics check
    st.markdown("### 1. Live FastAPI Telemetry (`/metrics`)")
    api_url = st.text_input("FastAPI Base URL", value="http://localhost:8000")
    if st.button("Refresh Live Metrics"):
        try:
            import urllib.request

            with urllib.request.urlopen(f"{api_url.rstrip('/')}/metrics", timeout=2.5) as resp:
                metrics = json.loads(resp.read().decode("utf-8"))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Requests", metrics.get("total_requests", 0))
            c2.metric("Error Rate", f"{metrics.get('error_rate_pct', 0.0):.2f}%")
            c3.metric("p95 Latency", f"{metrics.get('latency_ms', {}).get('p95', 0.0):.1f} ms")
            c4.metric(
                "Citation Rate",
                f"{metrics.get('rag_quality', {}).get('citation_rate_pct', 0.0):.1f}%",
            )
            if metrics.get("recent_requests"):
                st.dataframe(metrics["recent_requests"], use_container_width=True)
        except Exception as exc:
            st.warning(f"FastAPI server not reachable at {api_url} ({exc}). Showing offline benchmark telemetry below.")

    # 2. CPU Latency Profile (J-6)
    st.markdown("### 2. CPU Latency Profile (Task J-6)")
    lat_data = _load_json(RESULTS_DIR / "j6_latency_profile.json")
    if lat_data and "modes" in lat_data:
        rows = []
        for mode_name, stats in lat_data["modes"].items():
            rows.append({
                "Pipeline Mode": mode_name,
                "Mean (ms)": stats.get("mean_ms"),
                "p50 (ms)": stats.get("p50_ms"),
                "p95 (ms)": stats.get("p95_ms"),
                "Meets <5s SLA": "Yes" if stats.get("p95_ms", 9999) < 5000 else "No",
            })
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("Run `python scripts/profile_latency.py` to generate CPU latency profile.")

    # 3. J-5 Comparative Evaluation Matrix
    st.markdown("### 3. Comparative Evaluation Matrix (Task J-5)")
    j5_data = _load_json(RESULTS_DIR / "j5_comparative_matrix.json")
    if j5_data and "matrix_summary" in j5_data:
        st.dataframe(j5_data["matrix_summary"], use_container_width=True)
    else:
        st.info("Run `python eval/j5_comparative_eval.py` to generate comparative matrix.")

    # 4. N/D-1 Error Analysis
    st.markdown("### 4. Error Analysis Breakdown (Task N/D-1)")
    nd1_data = _load_json(RESULTS_DIR / "nd1_error_analysis.json")
    if nd1_data and "full_finsight_system" in nd1_data:
        full_err = nd1_data["full_finsight_system"]
        st.write("Full FinSight System Error Distribution:", full_err.get("error_counts", {}))
    else:
        st.info("Run `python eval/error_analysis.py` to generate error analysis.")


def main() -> None:
    st.title("FinSight — SEC 10-K Financial RAG")
    tab1, tab2 = st.tabs(["Q&A & Retrieval", "Monitoring & Benchmarks"])
    with tab1:
        render_qa_tab()
    with tab2:
        render_monitoring_tab()


if __name__ == "__main__":
    main()
