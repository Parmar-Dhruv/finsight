"""
FinSight - Interactive Financial QA & Monitoring UI (demo/app.py)

Polished Streamlit interface for FinSight SEC 10-K RAG:
  1. FinSight Q&A & Retrieval:
     * 1-click example financial queries
     * Optional auto-detection of company ticker & fiscal year from question
     * Dense / Hybrid (BM25+RRF) / Cross-Encoder Rerank controls
     * Citation-highlighted answer card with [SOURCE N] badges + KPI metrics
     * Source passage cards with provenance & relevance scores
     * Session query history
  2. Monitoring & Benchmarks (Tasks J-5, J-6, J-7, N/D-1):
     * Live FastAPI /metrics polling
     * Visual bar charts for J-6 CPU latency, J-5 5-config benchmark, and N/D-1 error breakdown

Run:
    python -m streamlit run demo/app.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Ensure project root is importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "evaluation" / "results"
TICKERS = ["(any)", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA"]
YEARS = ["(any)", 2023, 2024]
SECTIONS = ["(any)", "item_1", "item_1a", "item_7", "item_8"]

EXAMPLE_QUERIES = [
    "What were Apple's total net sales in fiscal 2024?",
    "What was Microsoft's Intelligent Cloud operating income in FY2024?",
    "How much revenue did AWS generate for Amazon in 2024?",
    "What are Alphabet's primary antitrust and regulatory risk factors?",
    "What capital expenditures did Meta report in 2024?",
]

COMPANY_PATTERNS: List[Tuple[str, str]] = [
    (r"\bapple\b|\baapl\b", "AAPL"),
    (r"\bmicrosoft\b|\bmsft\b", "MSFT"),
    (r"\bamazon\b|\bamzn\b", "AMZN"),
    (r"\balphabet\b|\bgoogle\b|\bgoogl\b", "GOOGL"),
    (r"\bmeta\b|\bfacebook\b", "META"),
    (r"\bnvidia\b|\bnvda\b", "NVDA"),
]

st.set_page_config(
    page_title="FinSight — SEC 10-K Financial RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle custom CSS for cleaner cards and citation badges
st.markdown(
    """
    <style>
    .citation-badge {
        display: inline-block;
        background-color: #1f6feb;
        color: #ffffff;
        padding: 1px 7px;
        border-radius: 6px;
        font-size: 0.82em;
        font-weight: 600;
        margin: 0 2px;
    }
    .answer-card {
        background-color: rgba(31, 111, 235, 0.08);
        border-left: 4px solid #1f6feb;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.75rem;
        font-size: 1.04rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def _auto_detect_filters(query: str) -> Tuple[Optional[str], Optional[int]]:
    q_lower = query.lower()
    ticker = None
    for pat, code in COMPANY_PATTERNS:
        if re.search(pat, q_lower):
            ticker = code
            break
    years = set(re.findall(r"\b(2023|2024)\b", q_lower))
    fiscal_year = int(next(iter(years))) if len(years) == 1 else None
    return ticker, fiscal_year


def _format_answer_html(answer: str) -> str:
    """Highlights [SOURCE N] tags as visual citation badges."""
    escaped = (
        answer.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return re.sub(
        r"(\[SOURCE\s+\d+\])",
        r'<span class="citation-badge">\1</span>',
        escaped,
    )


def render_qa_tab() -> None:
    if "query_input" not in st.session_state:
        st.session_state.query_input = EXAMPLE_QUERIES[0]
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    with st.sidebar:
        st.header("Retrieval & Filters")
        auto_filter = st.checkbox(
            "Auto-detect ticker & year from question",
            value=True,
            help="Automatically extracts company (e.g. Apple -> AAPL) and single fiscal year when set to (any).",
        )
        ticker = st.selectbox("Ticker", TICKERS, index=0)
        fiscal_year = st.selectbox("Fiscal year", YEARS, index=0)
        section = st.selectbox("Section", SECTIONS, index=0)

        st.divider()
        st.header("Pipeline Options")
        top_k = st.slider("Top-K Passages", min_value=1, max_value=10, value=5)
        hybrid = st.checkbox("Hybrid (Dense + BM25 RRF)", value=True)
        rerank = st.checkbox(
            "Cross-encoder rerank",
            value=False,
            help="Pure-NumPy MiniLM reranking adds ~3-7s on CPU.",
        )
        generate_ans = st.checkbox("Generate citation-backed answer", value=True)

        if st.session_state.qa_history:
            st.divider()
            st.header("Recent Queries")
            for item in reversed(st.session_state.qa_history[-5:]):
                st.caption(f"• {item['query'][:55]} ({item['latency_ms']:.0f} ms)")

    # Quick-start preset chips
    st.markdown("**Try an example question:**")
    chip_cols = st.columns(len(EXAMPLE_QUERIES))
    for idx, (col, ex) in enumerate(zip(chip_cols, EXAMPLE_QUERIES)):
        short_label = ex.split("?")[0][:28] + "..."
        if col.button(short_label, key=f"preset_{idx}", use_container_width=True):
            st.session_state.query_input = ex

    query = st.text_input(
        "Ask a financial question across SEC 10-K filings:",
        key="query_input",
        placeholder="e.g., What were Apple's total net sales in fiscal 2024?",
    )

    run_clicked = st.button("Search & Answer", type="primary")
    if run_clicked and query.strip():
        eff_ticker = None if ticker == "(any)" else ticker
        eff_year = None if fiscal_year == "(any)" else int(fiscal_year)
        eff_section = None if section == "(any)" else section

        if auto_filter:
            det_ticker, det_year = _auto_detect_filters(query)
            if eff_ticker is None:
                eff_ticker = det_ticker
            if eff_year is None:
                eff_year = det_year

        t0 = time.perf_counter()
        with st.spinner("Retrieving 10-K evidence and generating grounded answer..."):
            try:
                retriever = get_retriever()
                chunks = retriever.retrieve(
                    query=query,
                    k=top_k,
                    ticker=eff_ticker,
                    fiscal_year=eff_year,
                    section=eff_section,
                    hybrid=hybrid,
                    rerank=rerank,
                    rerank_pool=10,
                )
            except Exception as exc:
                st.error(f"Retrieval failed: {exc}")
                st.info(
                    "Check that `.env` has valid QDRANT_URL / QDRANT_API_KEY (or that a "
                    "local index exists at data/processed/qdrant_db/)."
                )
                return

            gen_out: Optional[Dict[str, Any]] = None
            if generate_ans and chunks:
                generator = get_generator()
                gen_out = generator.generate(question=query, context_chunks=chunks)

        total_ms = (time.perf_counter() - t0) * 1000.0
        mode = "dense" + (" + BM25" if hybrid else "") + (" + rerank" if rerank else "")

        st.session_state.qa_history.append({"query": query, "latency_ms": total_ms})

        # KPI Row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Passages Retrieved", len(chunks))
        k2.metric("End-to-End Latency", f"{total_ms:.0f} ms")
        k3.metric("Active Filters", f"{eff_ticker or 'All'} / {eff_year or 'All'}")
        k4.metric("Retrieval Mode", mode)

        if not chunks:
            st.warning("No passages returned. Try disabling filters or rephrasing your question.")
            return

        if gen_out is not None:
            st.markdown("### Grounded Answer")
            if gen_out.get("is_refusal"):
                st.warning(gen_out["answer"])
            else:
                st.markdown(
                    f'<div class="answer-card">{_format_answer_html(gen_out["answer"])}</div>',
                    unsafe_allow_html=True,
                )
            st.caption(
                f"Generator: `{gen_out.get('backend')}` · "
                f"Citations: `{', '.join(gen_out.get('citations', [])) or 'None'}` · "
                f"Synthesis Latency: `{gen_out.get('latency_ms')} ms`"
            )

        st.markdown("### Retrieved 10-K Evidence")
        for rank, chunk in enumerate(chunks, 1):
            header = (
                f"[SOURCE {rank}]  {chunk['ticker']} · FY{chunk['fiscal_year']} · "
                f"{chunk['section'].upper()}  |  Score: {chunk['score']:.4f}"
            )
            with st.expander(header, expanded=(rank == 1)):
                st.caption(f"Chunk ID: `{chunk.get('chunk_id', '')}`")
                st.write(chunk["text"])

        with st.expander("View Attributed LLM Prompt Context"):
            from retrieval.search import format_context_for_prompt

            st.code(format_context_for_prompt(chunks, max_tokens=2000), language="text")


def render_monitoring_tab() -> None:
    st.subheader("System Monitoring, Latency & Evaluation Dashboard (Tasks J-5 / J-6 / J-7 / N/D-1)")

    # 1. Live FastAPI /metrics check
    st.markdown("### 1. Live FastAPI Telemetry (`/metrics`)")
    col_url, col_btn = st.columns([4, 1])
    api_url = col_url.text_input("FastAPI Base URL", value="http://localhost:8000")
    if col_btn.button("Poll /metrics", use_container_width=True):
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
            st.info(f"FastAPI server is not running at `{api_url}` ({exc}). Offline benchmarks are shown below.")

    # 2. J-5 Comparative Evaluation Matrix
    st.markdown("### 2. Comparative Evaluation Matrix (Task J-5 · 100 Questions)")
    j5_data = _load_json(RESULTS_DIR / "j5_comparative_matrix.json")
    if j5_data and "matrix_summary" in j5_data:
        summary_rows = j5_data["matrix_summary"]
        st.dataframe(summary_rows, use_container_width=True)
        rag_rows = [r for r in summary_rows if r.get("use_rag")]
        if rag_rows:
            chart_data = {
                r["config_name"]: {
                    "Hit Rate @ 1 (%)": r["hit_rate_at_1"],
                    "Hit Rate @ 5 (%)": r["hit_rate_at_5"],
                }
                for r in rag_rows
            }
            st.bar_chart(chart_data)
    else:
        st.info("Run `python eval/j5_comparative_eval.py` to generate comparative matrix.")

    # 3. CPU Latency Profile (J-6) & Error Analysis (N/D-1)
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("### 3. CPU Latency Profile (Task J-6)")
        lat_data = _load_json(RESULTS_DIR / "j6_latency_profile.json")
        if lat_data and "modes" in lat_data:
            rows = [
                {
                    "Mode": mode_name,
                    "Mean (ms)": stats.get("mean_ms"),
                    "p50 (ms)": stats.get("p50_ms"),
                    "p95 (ms)": stats.get("p95_ms"),
                    "<5s SLA": "Yes" if stats.get("meets_5s_sla") else "No",
                }
                for mode_name, stats in lat_data["modes"].items()
            ]
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("Run `python scripts/profile_latency.py` to generate CPU latency profile.")

    with c_right:
        st.markdown("### 4. Error Analysis (Task N/D-1)")
        nd1_data = _load_json(RESULTS_DIR / "nd1_error_analysis.json")
        if nd1_data and "full_finsight_system" in nd1_data:
            full_err = nd1_data["full_finsight_system"].get("error_counts", {})
            st.bar_chart(full_err)
        else:
            st.info("Run `python eval/error_analysis.py` to generate error analysis.")


def main() -> None:
    st.title("📊 FinSight — SEC 10-K Financial RAG")
    st.caption(
        "Domain-Adapted Financial Q&A over Apple, Microsoft, Amazon, Alphabet, Meta & NVIDIA (FY2023–FY2024)"
    )
    tab1, tab2 = st.tabs(["🔍 Q&A & Retrieval", "📈 Monitoring & Benchmarks"])
    with tab1:
        render_qa_tab()
    with tab2:
        render_monitoring_tab()


if __name__ == "__main__":
    main()
