"""Optional verdict cache for the reason filter.

The pipeline only caches one thing: the local LLM's yes/no/uncertain verdict for
a given detection ``reason`` string, so re-runs over the same trace are free.
This is an *acceleration* cache — every method fails open (returns None / no-ops)
so a storage hiccup never blocks asking the model again.

``Cache`` is SQLite-backed and exposes the ``get_llm`` / ``put_llm`` / ``close``
interface that :class:`sheerscan.ollama.Ollama` expects. A host app with its own
compatible cache (e.g. the host app's) can pass that instead — any object with the
same three methods works.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .runtime import get_local_video_dir


def default_cache_path() -> Path:
    return get_local_video_dir() / "cache.sqlite3"


class Cache:
    def __init__(self, db_path: Optional[Path] = None):
        db_path = db_path or default_cache_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def get_llm(self, key: str, model: str) -> Optional[Any]:
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT response FROM llm_cache WHERE key=? AND model=?",
                    (key, model),
                ).fetchone()
        except sqlite3.Error:
            return None
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def put_llm(self, key: str, model: str, response: Any) -> None:
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO llm_cache(key,model,response,created_at) VALUES (?,?,?,?)",
                    (key, model, json.dumps(response, ensure_ascii=False), time.time()),
                )
                self._conn.commit()
        except sqlite3.Error:
            pass

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


class MemoryCache:
    """In-process cache with the same interface; no persistence."""

    def __init__(self):
        self._d: dict[tuple[str, str], Any] = {}

    def get_llm(self, key: str, model: str) -> Optional[Any]:
        return self._d.get((key, model))

    def put_llm(self, key: str, model: str, response: Any) -> None:
        self._d[(key, model)] = response

    def close(self) -> None:
        self._d.clear()
