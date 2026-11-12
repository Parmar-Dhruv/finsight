"""
FinSight - Conversational Financial RAG Chatbot & Monitoring UI (demo/app.py)

Features:
  * User Login & Registration (SQLite + PBKDF2-SHA256) + 1-click Demo Analyst login
  * Persistent Multi-Session Chat History per user (+ New Chat, Switch, Delete, Export)
  * Native ChatGPT-style conversation bubbles (st.chat_message + st.chat_input)
  * Inline [SOURCE N] citation badges + expandable 10-K evidence cards per response
  * Thumbs up / down feedback per assistant response
  * Auto-detection of company ticker & fiscal year + manual sidebar overrides
  * Monitoring & Benchmarks dashboard (J-5, J-6, J-7, N/D-1 + user feedback stats)

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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.auth_store import AuthChatStore

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
    page_title="FinSight AI — SEC 10-K Financial Chatbot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .citation-badge {
        display: inline-block;
        background-color: #1f6feb;
        color: #ffffff;
        padding: 1px 7px;
        border-radius: 6px;
        font-size: 0.80em;
        font-weight: 600;
        margin: 0 2px;
    }
    .auth-box {
        max-width: 460px;
        margin: 2rem auto;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background-color: rgba(31, 111, 235, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_store() -> AuthChatStore:
    return AuthChatStore()


@st.cache_resource(show_spinner="Loading retrieval engine (embeddings + Qdrant)...")
def get_retriever():
    from retrieval.search import FinSightRetriever

    return FinSightRetriever.get_instance()


@st.cache_resource(show_spinner="Loading FinSight generator...")
def get_generator():
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
    escaped = answer.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(
        r"(\[SOURCE\s+\d+\])",
        r'<span class="citation-badge">\1</span>',
        escaped,
    )


def _export_markdown(messages: List[Dict[str, Any]], title: str) -> str:
    lines = [f"# FinSight Chat Export — {title}\n"]
    for m in messages:
        role_label = "**User**" if m["role"] == "user" else "**FinSight AI**"
        lines.append(f"### {role_label}\n{m['content']}\n")
        if m.get("sources"):
            lines.append("**Sources:**")
            for idx, src in enumerate(m["sources"], 1):
                lines.append(
                    f"- `[SOURCE {idx}]` {src.get('ticker')} FY{src.get('fiscal_year')} "
                    f"{src.get('section')} (Score: {src.get('score', 0.0):.4f})"
                )
            lines.append("")
    return "\n".join(lines)


def render_auth_screen(store: AuthChatStore) -> None:
    _, center_col, _ = st.columns([1, 1.4, 1])
    with center_col:
        st.markdown("## 📊 FinSight AI")
        st.caption(
            "Domain-Adapted SEC 10-K Financial Assistant · AAPL, MSFT, AMZN, GOOGL, META, NVDA"
        )

        tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="demo@finsight.ai")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)
                if submitted:
                    user = store.authenticate_user(email, password)
                    if user:
                        st.session_state.user = user
                        st.session_state.active_session_id = None
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")

            if st.button("⚡ One-Click Demo Analyst Login", use_container_width=True):
                user = store.authenticate_user("demo@finsight.ai", "finsight123")
                if user:
                    st.session_state.user = user
                    st.session_state.active_session_id = None
                    st.rerun()
            st.caption("Demo credentials: `demo@finsight.ai` / `finsight123`")

        with tab_signup:
            with st.form("signup_form"):
                name = st.text_input("Full Name", placeholder="Nilay Patel")
                new_email = st.text_input("Email", placeholder="you@example.com")
                new_pass = st.text_input("Password (min 6 chars)", type="password")
                reg_submit = st.form_submit_button("Create Account", type="primary", use_container_width=True)
                if reg_submit:
                    ok, msg = store.register_user(name, new_email, new_pass)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)


def _process_user_prompt(
    store: AuthChatStore,
    session_id: int,
    prompt_text: str,
    ticker: Optional[str],
    fiscal_year: Optional[int],
    section: Optional[str],
    auto_filter: bool,
    top_k: int,
    hybrid: bool,
    rerank: bool,
) -> None:
    eff_ticker = ticker
    eff_year = fiscal_year
    if auto_filter:
        det_t, det_y = _auto_detect_filters(prompt_text)
        if eff_ticker is None:
            eff_ticker = det_t
        if eff_year is None:
            eff_year = det_y

    # Update session title on first message
    existing_msgs = store.get_messages(session_id)
    if not existing_msgs:
        store.update_session_title(session_id, prompt_text[:48])

    store.add_message(session_id, "user", prompt_text)

    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("Searching 10-K filings and synthesizing cited answer..."):
            t0 = time.perf_counter()
            try:
                retriever = get_retriever()
                chunks = retriever.retrieve(
                    query=prompt_text,
                    k=top_k,
                    ticker=eff_ticker,
                    fiscal_year=eff_year,
                    section=section,
                    hybrid=hybrid,
                    rerank=rerank,
                    rerank_pool=10,
                )
            except Exception as exc:
                err_msg = f"Retrieval error: {exc}"
                st.error(err_msg)
                store.add_message(session_id, "assistant", err_msg)
                return

            generator = get_generator()
            gen_out = generator.generate(question=prompt_text, context_chunks=chunks)
            total_ms = (time.perf_counter() - t0) * 1000.0

            answer = gen_out["answer"]
            mode = "dense" + (" + BM25" if hybrid else "") + (" + rerank" if rerank else "")
            meta = {
                "latency_ms": round(total_ms, 1),
                "mode": mode,
                "filters": f"{eff_ticker or 'All'} / {eff_year or 'All'} / {section or 'All'}",
                "backend": gen_out.get("backend", "domain_synthesis"),
                "is_refusal": bool(gen_out.get("is_refusal")),
            }

            if meta["is_refusal"]:
                st.warning(answer)
            else:
                st.markdown(_format_answer_html(answer), unsafe_allow_html=True)

            st.caption(
                f"Latency: `{meta['latency_ms']} ms` · Mode: `{meta['mode']}` · "
                f"Filters: `{meta['filters']}` · Backend: `{meta['backend']}`"
            )

            if chunks:
                with st.expander(f"📚 View Retrieved 10-K Sources ({len(chunks)})"):
                    for idx, ch in enumerate(chunks, 1):
                        st.markdown(
                            f"**[SOURCE {idx}] {ch['ticker']} · FY{ch['fiscal_year']} · "
                            f"{ch['section'].upper()}** (Score: `{ch['score']:.4f}`)"
                        )
                        st.write(ch["text"])
                        st.divider()

            store.add_message(
                session_id,
                "assistant",
                answer,
                sources=chunks,
                metadata=meta,
            )


def render_chat_tab(store: AuthChatStore, user: Dict[str, Any]) -> None:
    sessions = store.list_sessions(user["id"])
    if not sessions:
        new_id = store.create_session(user["id"], title="New Chat")
        st.session_state.active_session_id = new_id
        sessions = store.list_sessions(user["id"])

    if (
        "active_session_id" not in st.session_state
        or st.session_state.active_session_id not in [s["id"] for s in sessions]
    ):
        st.session_state.active_session_id = sessions[0]["id"]

    active_sid = st.session_state.active_session_id
    active_session = next(s for s in sessions if s["id"] == active_sid)

    with st.sidebar:
        st.markdown(f"### 👤 {user['name']}")
        st.caption(user["email"])
        if st.button("Sign Out", use_container_width=True):
            st.session_state.pop("user", None)
            st.session_state.pop("active_session_id", None)
            st.rerun()

        st.divider()
        st.markdown("### 💬 Conversations")
        if st.button("➕ New Chat", type="primary", use_container_width=True):
            st.session_state.active_session_id = store.create_session(user["id"], title="New Chat")
            st.rerun()

        for sess in sessions:
            is_active = sess["id"] == active_sid
            label = f"🟢 {sess['title']}" if is_active else f"💬 {sess['title']}"
            if st.button(label, key=f"sess_{sess['id']}", use_container_width=True):
                st.session_state.active_session_id = sess["id"]
                st.rerun()

        msgs_for_export = store.get_messages(active_sid)
        col_del, col_exp = st.columns(2)
        if col_del.button("🗑️ Delete", use_container_width=True):
            store.delete_session(user["id"], active_sid)
            st.session_state.active_session_id = None
            st.rerun()
        col_exp.download_button(
            "⬇️ Export",
            data=_export_markdown(msgs_for_export, active_session["title"]),
            file_name=f"finsight_chat_{active_sid}.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.divider()
        st.markdown("### ⚙️ Retrieval Settings")
        auto_filter = st.checkbox("Auto-detect company & year", value=True)
        ticker = st.selectbox("Ticker", TICKERS, index=0)
        fiscal_year = st.selectbox("Fiscal Year", YEARS, index=0)
        section = st.selectbox("10-K Section", SECTIONS, index=0)
        top_k = st.slider("Top-K Passages", min_value=1, max_value=10, value=5)
        hybrid = st.checkbox("Hybrid (Dense + BM25 RRF)", value=True)
        rerank = st.checkbox("Cross-encoder rerank", value=False)

    # Render conversation messages
    messages = store.get_messages(active_sid)
    if not messages:
        st.info(
            "Welcome to **FinSight AI**! Ask any question about Apple, Microsoft, Amazon, "
            "Alphabet, Meta, or NVIDIA SEC 10-K filings (FY2023–FY2024), or click a starter prompt below:"
        )
        cols = st.columns(len(EXAMPLE_QUERIES))
        for idx, (col, ex) in enumerate(zip(cols, EXAMPLE_QUERIES)):
            if col.button(ex[:30] + "...", key=f"starter_{idx}", use_container_width=True):
                st.session_state.pending_prompt = ex
                st.rerun()

    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                meta = msg.get("metadata", {})
                if meta.get("is_refusal"):
                    st.warning(msg["content"])
                else:
                    st.markdown(_format_answer_html(msg["content"]), unsafe_allow_html=True)
                if meta:
                    st.caption(
                        f"Latency: `{meta.get('latency_ms', 0)} ms` · "
                        f"Mode: `{meta.get('mode', '-')}` · "
                        f"Filters: `{meta.get('filters', '-')}`"
                    )
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📚 View Retrieved 10-K Sources ({len(sources)})"):
                        for idx, ch in enumerate(sources, 1):
                            st.markdown(
                                f"**[SOURCE {idx}] {ch.get('ticker')} · FY{ch.get('fiscal_year')} · "
                                f"{str(ch.get('section', '')).upper()}** (Score: `{ch.get('score', 0.0):.4f}`)"
                            )
                            st.write(ch.get("text", ""))
                            st.divider()

                # Feedback buttons
                fb = msg.get("feedback")
                fb_c1, fb_c2, _ = st.columns([1, 1, 10])
                if fb_c1.button("👍" if fb != "up" else "👍✓", key=f"up_{msg['id']}"):
                    store.set_feedback(msg["id"], "up")
                    st.rerun()
                if fb_c2.button("👎" if fb != "down" else "👎✓", key=f"down_{msg['id']}"):
                    store.set_feedback(msg["id"], "down")
                    st.rerun()
            else:
                st.markdown(msg["content"])

    # Handle starter click or chat input
    eff_t = None if ticker == "(any)" else ticker
    eff_y = None if fiscal_year == "(any)" else int(fiscal_year)
    eff_s = None if section == "(any)" else section

    pending = st.session_state.pop("pending_prompt", None)
    user_prompt = st.chat_input(
        "Ask about revenue, margins, risks, segments, or capex in SEC 10-K filings..."
    )
    final_prompt = pending or user_prompt
    if final_prompt and final_prompt.strip():
        _process_user_prompt(
            store=store,
            session_id=active_sid,
            prompt_text=final_prompt.strip(),
            ticker=eff_t,
            fiscal_year=eff_y,
            section=eff_s,
            auto_filter=auto_filter,
            top_k=top_k,
            hybrid=hybrid,
            rerank=rerank,
        )
        st.rerun()


def render_monitoring_tab(store: AuthChatStore) -> None:
    st.subheader("System Monitoring, User Feedback & Evaluation Benchmarks")

    # User Feedback KPIs
    fb_stats = store.get_feedback_stats()
    f1, f2, f3 = st.columns(3)
    f1.metric("Assistant Responses", fb_stats["total_assistant_messages"])
    f2.metric("Helpful (👍)", fb_stats["thumbs_up"])
    f3.metric("Needs Improvement (👎)", fb_stats["thumbs_down"])

    st.divider()
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

    st.markdown("### 2. Comparative Evaluation Matrix (Task J-5 · 100 Questions)")
    j5_data = _load_json(RESULTS_DIR / "j5_comparative_matrix.json")
    if j5_data and "matrix_summary" in j5_data:
        summary_rows = j5_data["matrix_summary"]
        st.dataframe(summary_rows, use_container_width=True)
        rag_rows = [r for r in summary_rows if r.get("use_rag")]
        if rag_rows:
            st.bar_chart({
                r["config_name"]: {
                    "Hit Rate @ 1 (%)": r["hit_rate_at_1"],
                    "Hit Rate @ 5 (%)": r["hit_rate_at_5"],
                }
                for r in rag_rows
            })

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("### 3. CPU Latency Profile (Task J-6)")
        lat_data = _load_json(RESULTS_DIR / "j6_latency_profile.json")
        if lat_data and "modes" in lat_data:
            st.dataframe(
                [
                    {
                        "Mode": m,
                        "Mean (ms)": s.get("mean_ms"),
                        "p50 (ms)": s.get("p50_ms"),
                        "p95 (ms)": s.get("p95_ms"),
                        "<5s SLA": "Yes" if s.get("meets_5s_sla") else "No",
                    }
                    for m, s in lat_data["modes"].items()
                ],
                use_container_width=True,
            )

    with c_right:
        st.markdown("### 4. Error Analysis (Task N/D-1)")
        nd1_data = _load_json(RESULTS_DIR / "nd1_error_analysis.json")
        if nd1_data and "full_finsight_system" in nd1_data:
            st.bar_chart(nd1_data["full_finsight_system"].get("error_counts", {}))


def main() -> None:
    store = get_store()
    user = st.session_state.get("user")

    if not user:
        render_auth_screen(store)
        return

    st.title("📊 FinSight AI — SEC 10-K Financial Chatbot")
    tab_chat, tab_mon = st.tabs(["💬 Financial Assistant", "📈 Monitoring & Benchmarks"])
    with tab_chat:
        render_chat_tab(store, user)
    with tab_mon:
        render_monitoring_tab(store)


if __name__ == "__main__":
    main()
