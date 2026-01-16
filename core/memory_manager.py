# core/memory_manager.py
import sqlite3
import json
import os
from typing import Dict, Any, Optional, List

class MemoryManager:
    """Simple SQLite-backed persistent memory for runs and chat metadata."""

    def __init__(self, db_path: str = "memory_store.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as c:
            cur = c.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    dataset TEXT,
                    summary_json TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    user_text TEXT,
                    bot_text TEXT
                )
            """)
            c.commit()

    def add_run(self, summary: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as c:
            cur = c.cursor()
            cur.execute("INSERT INTO runs (timestamp, dataset, summary_json) VALUES (?, ?, ?)",
                        (summary.get("time"), summary.get("dataset"), json.dumps(summary)))
            c.commit()

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as c:
            cur = c.cursor()
            cur.execute("SELECT timestamp, dataset, summary_json FROM runs ORDER BY id DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        out = []
        for ts, ds, sj in rows:
            out.append({"timestamp": ts, "dataset": ds, "summary": json.loads(sj)})
        return out

    def add_chat(self, user_text: str, bot_text: str):
        with sqlite3.connect(self.db_path) as c:
            cur = c.cursor()
            cur.execute("INSERT INTO chat_history (timestamp, user_text, bot_text) VALUES (datetime('now'), ?, ?)",
                        (user_text, bot_text))
            c.commit()

    def last_run(self) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as c:
            cur = c.cursor()
            cur.execute("SELECT timestamp, dataset, summary_json FROM runs ORDER BY id DESC LIMIT 1")
            r = cur.fetchone()
            if r:
                ts, ds, sj = r
                return {"timestamp": ts, "dataset": ds, "summary": json.loads(sj)}
        return None
