"""
Unit tests for FinSight UI Auth & Persistent Chat Store (demo/auth_store.py).
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from demo.auth_store import AuthChatStore


def test_demo_user_seeded_and_auth_works(tmp_path):
    store = AuthChatStore(db_path=tmp_path / "test_users.db")
    user = store.authenticate_user("demo@finsight.ai", "finsight123")
    assert user is not None
    assert user["name"] == "Demo Analyst"
    assert store.authenticate_user("demo@finsight.ai", "wrongpass") is None


def test_register_and_chat_persistence(tmp_path):
    store = AuthChatStore(db_path=tmp_path / "test_users.db")
    ok, _ = store.register_user("Nilay", "nilay@example.com", "secret123")
    assert ok is True
    # Duplicate email fails
    ok2, _ = store.register_user("Nilay2", "nilay@example.com", "secret123")
    assert ok2 is False

    user = store.authenticate_user("nilay@example.com", "secret123")
    assert user is not None

    sid = store.create_session(user["id"], title="Apple 10-K")
    sessions = store.list_sessions(user["id"])
    assert len(sessions) == 1 and sessions[0]["id"] == sid

    m1 = store.add_message(sid, "user", "What were Apple's net sales?")
    m2 = store.add_message(
        sid,
        "assistant",
        "Total net sales were $391.0B [SOURCE 1].",
        sources=[{"ticker": "AAPL", "fiscal_year": 2024, "section": "item_7", "score": 0.9, "text": "..."}],
        metadata={"latency_ms": 250.0},
    )
    assert m1 > 0 and m2 > m1

    msgs = store.get_messages(sid)
    assert len(msgs) == 2
    assert msgs[1]["sources"][0]["ticker"] == "AAPL"

    store.set_feedback(m2, "up")
    stats = store.get_feedback_stats()
    assert stats["thumbs_up"] == 1

    store.delete_session(user["id"], sid)
    assert store.list_sessions(user["id"]) == []

