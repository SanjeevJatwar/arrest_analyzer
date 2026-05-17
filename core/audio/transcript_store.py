"""
core/audio/transcript_store.py

Persists transcripts to disk as human-readable .txt and structured .json.
Each session creates a new timestamped file pair in the output directory.
"""
import os
import json
import time
import threading
from datetime import datetime
import config


class TranscriptStore:
    """Thread-safe transcript storage."""

    def __init__(self):
        self._lock = threading.Lock()
        self._entries = []
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs(config.TRANSCRIPT_DIR, exist_ok=True)

        self._txt_path  = os.path.join(
            config.TRANSCRIPT_DIR, f"transcript_{self._session_id}.txt")
        self._json_path = os.path.join(
            config.TRANSCRIPT_DIR, f"transcript_{self._session_id}.json")

        with open(self._txt_path, "w", encoding="utf-8") as f:
            f.write(f"=== ARREST ANALYZER TRANSCRIPT ===\n")
            f.write(f"Session: {self._session_id}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"{'=' * 50}\n\n")

        print(f"[TranscriptStore] Saving to: {self._txt_path}")

    def add(self, label: str, text: str):
        ts = datetime.now().isoformat()
        entry = {"timestamp": ts, "label": label, "text": text}

        with self._lock:
            self._entries.append(entry)
            with open(self._txt_path, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] [{label}] {text}\n")
            with open(self._json_path, "w", encoding="utf-8") as f:
                json.dump({"session_id": self._session_id,
                           "entries": self._entries}, f, indent=2,
                          ensure_ascii=False)

    def get_full_transcript(self) -> str:
        with self._lock:
            return "\n".join(
                f"[{e['label']}] {e['text']}" for e in self._entries)

    def get_entries(self) -> list:
        with self._lock:
            return list(self._entries)

    @property
    def txt_path(self):
        return self._txt_path

    @property
    def json_path(self):
        return self._json_path
