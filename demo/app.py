"""
FinSight AI — SEC 10-K Financial Chatbot (demo/app.py)

A clean ChatGPT-style conversational interface for querying SEC 10-K filings.
Users log in, ask questions, and receive cited answers — no backend details shown.

Run:
    python -m streamlit run demo/app.py
"""

from __future__ import annotations

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

# ── Company detection patterns ──────────────────────────────────────────
COMPANY_PATTERNS: List[Tuple[str, str]] = [
    (r"\bapple\b|\baapl\b", "AAPL"),
    (r"\bmicrosoft\b|\bmsft\b", "MSFT"),
    (r"\bamazon\b|\bamzn\b", "AMZN"),
    (r"\balphabet\b|\bgoogle\b|\bgoogl\b", "GOOGL"),
    (r"\bmeta\b|\bfacebook\b", "META"),
    (r"\bnvidia\b|\bnvda\b", "NVDA"),
]

EXAMPLE_QUERIES = [
    "What were Apple's total net sales in fiscal 2024?",
    "What was Microsoft's Intelligent Cloud revenue in FY2024?",
    "How much revenue did AWS generate for Amazon in 2024?",
    "What are Alphabet's primary risk factors?",
    "What capital expenditures did Meta report in 2024?",
]

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — ChatGPT-style polish ───────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- Hide Streamlit Deploy button, hamburger menu & footer ---- */
    [data-testid="stAppDeployButton"],
    [data-testid="stToolbarActions"],
    .stDeployButton,
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
    }

    /* ---- Global ---- */
    .block-container { max-width: 860px; padding-top: 1.5rem; }

    /* ---- Citation badge ---- */
    .src-badge {
        display: inline-block;
        background: #1f6feb;
        color: #fff;
        padding: 1px 7px;
        border-radius: 6px;
        font-size: 0.78em;
        font-weight: 600;
        margin: 0 2px;
        cursor: default;
    }

    /* ---- Source card inside expander ---- */
    .source-card {
        border-left: 3px solid #1f6feb;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.7rem;
        border-radius: 4px;
        background: rgba(31,111,235,0.04);
    }
    .source-card .src-heading {
        font-weight: 600;
        font-size: 0.92em;
        margin-bottom: 0.3rem;
        color: #1f6feb;
    }
    .source-card .src-text {
        font-size: 0.85em;
        line-height: 1.5;
        color: inherit;
    }

    /* ---- Auth container ---- */
    .auth-wrapper {
        max-width: 440px;
        margin: 4rem auto 2rem auto;
    }
    .auth-wrapper h1 { text-align: center; }
    .auth-wrapper .tagline {
        text-align: center;
        color: #888;
        margin-bottom: 1.5rem;
    }

    /* ---- Welcome card starters ---- */
    .welcome-area { text-align: center; margin-top: 6rem; }
    .welcome-area h2 { font-weight: 400; }

    /* ---- Sidebar polish ---- */
    [data-testid="stSidebar"] .stButton > button {
        text-align: left;
    }

    /* ---- Feedback row ---- */
    .fb-row button { padding: 0.15rem 0.5rem; font-size: 0.82em; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════
#  Cached singletons
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource
def _get_store() -> AuthChatStore:
    return AuthChatStore()


@st.cache_resource(show_spinner="Loading FinSight retrieval engine…")
def _get_retriever():
    from retrieval.search import FinSightRetriever
    return FinSightRetriever.get_instance()


@st.cache_resource(show_spinner="Loading FinSight generator…")
def _get_generator():
    from generation.inference import FinSightGenerator
    return FinSightGenerator.get_instance()


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _auto_detect(query: str) -> Tuple[Optional[str], Optional[int]]:
    """Infer ticker & fiscal year from natural-language question."""
    q = query.lower()
    ticker = None
    for pat, code in COMPANY_PATTERNS:
        if re.search(pat, q):
            ticker = code
            break
    years = set(re.findall(r"(?:fy|fiscal\s*)?(202[234])\b", q))
    year = int(next(iter(years))) if len(years) == 1 else None
    return ticker, year


def _badge_html(answer: str) -> str:
    """Replace [SOURCE N] tokens with styled inline badges."""
    escaped = answer.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(
        r"(\[SOURCE\s+\d+\])",
        r'<span class="src-badge">\1</span>',
        escaped,
    )


def _format_source_excerpt(text: str) -> str:
    """Clean raw SEC text and pipe-delimited table artifacts for human display."""
    if not text:
        return ""
    t = text.replace("\ufffd", "'")
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", t)
    t = t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    lines = []
    for line in t.splitlines():
        line = line.strip()
        if not line or set(line).issubset({"|", "-", " ", ":"}):
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.split("|") if c.strip() and not set(c.strip()).issubset({"-", ":"})]
            if cells:
                lines.append(" · ".join(cells))
        else:
            lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned[:500] + ("…" if len(cleaned) > 500 else "")


def _render_sources(sources: List[Dict[str, Any]]) -> None:
    """Render a clean collapsible sources section (no scores / chunk IDs)."""
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)} passages)"):
        for idx, ch in enumerate(sources, 1):
            ticker = ch.get("ticker", "")
            fy = ch.get("fiscal_year", "")
            section = str(ch.get("section", "")).replace("_", " ").title()
            text = _format_source_excerpt(ch.get("text", "") or ch.get("excerpt", ""))
            st.markdown(
                f'<div class="source-card">'
                f'<div class="src-heading">[SOURCE {idx}]  {ticker} · FY{fy} · {section}</div>'
                f'<div class="src-text">{text}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════
#  AUTH SCREEN
# ═══════════════════════════════════════════════════════════════════════

def _render_auth(store: AuthChatStore) -> None:
    st.markdown('<div class="auth-wrapper">', unsafe_allow_html=True)

    st.markdown("# 📊 FinSight AI")
    st.markdown(
        '<p class="tagline">Your AI-powered SEC 10-K financial research assistant</p>',
        unsafe_allow_html=True,
    )

    tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="demo@finsight.ai")
            pwd = st.text_input("Password", type="password", placeholder="••••••••")
            submit = st.form_submit_button("Sign In", type="primary", use_container_width=True)
            if submit:
                user = store.authenticate_user(email, pwd)
                if user:
                    st.session_state.user = user
                    st.session_state.active_session = None
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

        if st.button("⚡  Try Demo Account", use_container_width=True):
            user = store.authenticate_user("demo@finsight.ai", "finsight123")
            if user:
                st.session_state.user = user
                st.session_state.active_session = None
                st.rerun()

        st.caption("Demo: `demo@finsight.ai` / `finsight123`")

    with tab_register:
        with st.form("register_form"):
            name = st.text_input("Full Name", placeholder="Your name")
            new_email = st.text_input("Email", placeholder="you@example.com")
            new_pwd = st.text_input("Password (min 6 characters)", type="password")
            reg = st.form_submit_button(
                "Create Account", type="primary", use_container_width=True
            )
            if reg:
                ok, msg = store.register_user(name, new_email, new_pwd)
                (st.success if ok else st.error)(msg)

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR — user info, conversations, sign out
# ═══════════════════════════════════════════════════════════════════════

def _render_sidebar(store: AuthChatStore, user: Dict[str, Any]) -> int:
    """Draw sidebar and return active session id."""
    sessions = store.list_sessions(user["id"])

    # Ensure at least one session
    if not sessions:
        new_id = store.create_session(user["id"], "New Chat")
        st.session_state.active_session = new_id
        sessions = store.list_sessions(user["id"])

    # Sync active session
    sid_list = [s["id"] for s in sessions]
    if (
        st.session_state.get("active_session") is None
        or st.session_state.active_session not in sid_list
    ):
        st.session_state.active_session = sid_list[0]

    active = st.session_state.active_session

    with st.sidebar:
        # ── User + sign-out ──
        st.markdown(f"### 👤 {user['name']}")
        st.caption(user["email"])
        if st.button("Sign Out", use_container_width=True):
            for k in ("user", "active_session", "pending_prompt"):
                st.session_state.pop(k, None)
            st.rerun()

        st.divider()

        # ── New Chat ──
        if st.button("➕  New Chat", type="primary", use_container_width=True):
            new_id = store.create_session(user["id"], "New Chat")
            st.session_state.active_session = new_id
            st.rerun()

        # ── Conversation list ──
        st.markdown("##### Recent Chats")
        for sess in sessions:
            is_active = sess["id"] == active
            icon = "▸" if is_active else " "
            label = f"{icon} {sess['title']}"
            if st.button(label, key=f"s_{sess['id']}", use_container_width=True):
                st.session_state.active_session = sess["id"]
                st.rerun()

        st.divider()

        # ── Delete current conversation ──
        if st.button("🗑️  Delete this chat", use_container_width=True):
            store.delete_session(user["id"], active)
            st.session_state.active_session = None
            st.rerun()

    return active


# ═══════════════════════════════════════════════════════════════════════
#  CHAT AREA
# ═══════════════════════════════════════════════════════════════════════

def _render_chat(store: AuthChatStore, user: Dict[str, Any], session_id: int) -> None:
    messages = store.get_messages(session_id)

    # ── Welcome screen (empty conversation) ──
    if not messages:
        st.markdown(
            '<div class="welcome-area">'
            "<h2>📊 What would you like to know?</h2>"
            "<p style='color:#888'>Ask about revenue, margins, risks, or any details "
            "from Apple, Microsoft, Amazon, Alphabet, Meta, or NVIDIA SEC 10-K filings.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(len(EXAMPLE_QUERIES))
        for idx, (col, q) in enumerate(zip(cols, EXAMPLE_QUERIES)):
            short = q[:32] + "…" if len(q) > 32 else q
            if col.button(short, key=f"starter_{idx}", use_container_width=True):
                st.session_state.pending_prompt = q
                st.rerun()
        # Still show chat_input even on welcome screen
        _handle_input(store, user, session_id, messages)
        return

    # ── Render history ──
    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                meta = msg.get("metadata", {})
                if meta.get("is_refusal"):
                    st.warning(msg["content"])
                else:
                    st.markdown(_badge_html(msg["content"]), unsafe_allow_html=True)
                    _render_sources(msg.get("sources", []))

                # Feedback buttons
                fb = msg.get("feedback")
                c1, c2, _ = st.columns([0.5, 0.5, 11])
                with c1:
                    if st.button(
                        "👍" if fb != "up" else "👍 ✓",
                        key=f"up_{msg['id']}",
                    ):
                        store.set_feedback(msg["id"], "up")
                        st.rerun()
                with c2:
                    if st.button(
                        "👎" if fb != "down" else "👎 ✓",
                        key=f"dn_{msg['id']}",
                    ):
                        store.set_feedback(msg["id"], "down")
                        st.rerun()
            else:
                st.markdown(msg["content"])

    # ── Input bar ──
    _handle_input(store, user, session_id, messages)


def _handle_input(
    store: AuthChatStore,
    user: Dict[str, Any],
    session_id: int,
    messages: List[Dict[str, Any]],
) -> None:
    """Render chat_input and process any pending or new prompt."""
    pending = st.session_state.pop("pending_prompt", None)
    typed = st.chat_input("Ask about SEC 10-K filings…")
    prompt = pending or typed
    if not prompt or not prompt.strip():
        return

    prompt = prompt.strip()

    # Title the session after the first user message
    if not messages:
        store.update_session_title(session_id, prompt[:50])

    # Persist & show user message
    store.add_message(session_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Generate answer ──
    with st.chat_message("assistant"):
        with st.spinner("Searching filings…"):
            ticker, fiscal_year = _auto_detect(prompt)

            try:
                retriever = _get_retriever()
                chunks = retriever.retrieve(
                    query=prompt,
                    k=5,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    hybrid=True,
                    rerank=False,
                )
            except Exception as exc:
                err = f"Sorry, I encountered a retrieval error: {exc}"
                st.error(err)
                store.add_message(session_id, "assistant", err, metadata={"is_refusal": True})
                return

            generator = _get_generator()
            result = generator.generate(question=prompt, context_chunks=chunks)

            answer = result["answer"]
            is_refusal = bool(result.get("is_refusal"))
            meta = {"is_refusal": is_refusal}

            if is_refusal:
                st.warning(answer)
                store.add_message(session_id, "assistant", answer, metadata=meta)
            else:
                st.markdown(_badge_html(answer), unsafe_allow_html=True)
                _render_sources(chunks)
                store.add_message(
                    session_id, "assistant", answer, sources=chunks, metadata=meta
                )

    st.rerun()


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    store = _get_store()
    user = st.session_state.get("user")

    if not user:
        _render_auth(store)
        return

    session_id = _render_sidebar(store, user)
    _render_chat(store, user, session_id)


if __name__ == "__main__":
    main()
