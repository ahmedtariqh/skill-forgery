import json
import sqlite3
import hashlib
import threading
from typing import List, Optional
from .models import VideoSnapshot, KnowledgeExtractionChunk


class PipelineCache:
    """
    Thread-safe and coroutine-safe SQLite checkpoint cache using WAL mode.
    Stores transcripts, video/ebook visual snapshots, and LLM knowledge extractions.
    """

    def __init__(self, db_path: str = "./skill_forge_cache.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        self._lock = threading.Lock()
        self._init_tables()

    def _init_tables(self):
        with self._lock, self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    video_id TEXT PRIMARY KEY,
                    title TEXT,
                    transcript_text TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS visual_snapshots (
                    source_id TEXT PRIMARY KEY,
                    snapshots_json TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS extractions (
                    cache_key TEXT PRIMARY KEY,
                    segment_id TEXT,
                    model_name TEXT,
                    detected_domain TEXT,
                    chunk_json TEXT
                )
            """)

    def get_transcript(self, video_id: str) -> Optional[str]:
        with self._lock:
            cur = self.conn.execute("SELECT transcript_text FROM transcripts WHERE video_id = ?", (video_id,))
            row = cur.fetchone()
            return row[0] if row else None

    def save_transcript(self, video_id: str, title: str, text: str):
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO transcripts (video_id, title, transcript_text) VALUES (?, ?, ?)",
                (video_id, title, text),
            )

    def get_snapshots(self, source_id: str) -> Optional[List[VideoSnapshot]]:
        with self._lock:
            cur = self.conn.execute("SELECT snapshots_json FROM visual_snapshots WHERE source_id = ?", (source_id,))
            row = cur.fetchone()
            if row:
                return [VideoSnapshot.model_validate(item) for item in json.loads(row[0])]
            return None

    def save_snapshots(self, source_id: str, snapshots: List[VideoSnapshot]):
        payload = json.dumps([s.model_dump() for s in snapshots])
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO visual_snapshots (source_id, snapshots_json) VALUES (?, ?)",
                (source_id, payload),
            )

    @staticmethod
    def _make_key(segment_id: str, text: str, model_tag: str, domain_hint: str) -> str:
        return hashlib.sha256(f"{segment_id}:{model_tag}:{domain_hint}:{text}".encode("utf-8")).hexdigest()

    def get_extraction(
        self, segment_id: str, text: str, model_tag: str, domain_hint: str
    ) -> Optional[KnowledgeExtractionChunk]:
        key = self._make_key(segment_id, text, model_tag, domain_hint)
        with self._lock:
            cur = self.conn.execute("SELECT chunk_json FROM extractions WHERE cache_key = ?", (key,))
            row = cur.fetchone()
            return KnowledgeExtractionChunk.model_validate_json(row[0]) if row else None

    def save_extraction(
        self, segment_id: str, text: str, model_tag: str, domain_hint: str, chunk: KnowledgeExtractionChunk
    ):
        key = self._make_key(segment_id, text, model_tag, domain_hint)
        with self._lock, self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO extractions 
                   (cache_key, segment_id, model_name, detected_domain, chunk_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (key, segment_id, model_tag, chunk.detected_domain, chunk.model_dump_json()),
            )
