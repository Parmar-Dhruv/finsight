"""
FinSight UI Authentication & Persistent Chat Store (demo/auth_store.py)

Provides local SQLite-backed user accounts (PBKDF2-SHA256 salted password hashing),
multi-conversation chat sessions, message history, and user feedback tracking.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "finsight_users.db"


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt_bytes = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 120_000)
    return f"{salt_bytes.hex()}:{digest.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, digest_hex = stored_hash.split(":", 1)
        salt_bytes = bytes.fromhex(salt_hex)
        expected = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 120_000).hex()
        return hmac.compare_digest(expected, digest_hex)
    except Exception:
        return False


class AuthChatStore:
    """SQLite persistence for FinSight chatbot users, sessions, and messages."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT,
                    metadata_json TEXT,
                    feedback TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
                )
                """
            )
            # Seed a ready-to-use demo account if not present
            existing = conn.execute(
                "SELECT id FROM users WHERE email = ?", ("demo@finsight.ai",)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    ("Demo Analyst", "demo@finsight.ai", _hash_password("finsight123"), time.time()),
                )

    def register_user(self, name: str, email: str, password: str) -> Tuple[bool, str]:
        clean_name = (name or "").strip()
        clean_email = (email or "").strip().lower()
        if not clean_name or not clean_email or "@" not in clean_email:
            return False, "Please enter a valid name and email address."
        if len(password or "") < 6:
            return False, "Password must be at least 6 characters long."

        with self._connect() as conn:
            existing = conn.execute("SELECT id FROM users WHERE email = ?", (clean_email,)).fetchone()
            if existing:
                return False, "An account with this email already exists."
            conn.execute(
                "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (clean_name, clean_email, _hash_password(password), time.time()),
            )
        return True, "Account created successfully. You can now sign in."

    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        clean_email = (email or "").strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, name, email, password_hash FROM users WHERE email = ?",
                (clean_email,),
            ).fetchone()
        if row and _verify_password(password or "", row["password_hash"]):
            return {"id": row["id"], "name": row["name"], "email": row["email"]}
        return None

    def create_session(self, user_id: int, title: str = "New Chat") -> int:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO chat_sessions (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (user_id, title, now, now),
            )
            return int(cur.lastrowid)

    def list_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_session_title(self, session_id: int, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title[:60], time.time(), session_id),
            )

    def delete_session(self, user_id: int, session_id: int) -> None:
        with self._connect() as conn:
            owner = conn.execute(
                "SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
            if owner:
                conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))

    def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, sources_json, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    json.dumps(sources or []),
                    json.dumps(metadata or {}),
                    now,
                ),
            )
            conn.execute("UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            return int(cur.lastrowid)

    def get_messages(self, session_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, role, content, sources_json, metadata_json, feedback, created_at "
                "FROM chat_messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        result: List[Dict[str, Any]] = []
        for r in rows:
            result.append({
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "sources": json.loads(r["sources_json"] or "[]"),
                "metadata": json.loads(r["metadata_json"] or "{}"),
                "feedback": r["feedback"],
                "created_at": r["created_at"],
            })
        return result

    def set_feedback(self, message_id: int, feedback: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chat_messages SET feedback = ? WHERE id = ?",
                (feedback, message_id),
            )

    def get_feedback_stats(self) -> Dict[str, int]:
        with self._connect() as conn:
            up = conn.execute("SELECT COUNT(*) FROM chat_messages WHERE feedback = 'up'").fetchone()[0]
            down = conn.execute("SELECT COUNT(*) FROM chat_messages WHERE feedback = 'down'").fetchone()[0]
            total_msgs = conn.execute("SELECT COUNT(*) FROM chat_messages WHERE role = 'assistant'").fetchone()[0]
        return {"thumbs_up": int(up), "thumbs_down": int(down), "total_assistant_messages": int(total_msgs)}

