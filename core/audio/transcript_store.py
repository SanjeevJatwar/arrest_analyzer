import os
import json
import time
import threading
from datetime import datetime
import config


class TranscriptStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._entries = []
        self._emotion_history = []
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(config.TRANSCRIPT_DIR, exist_ok=True)
        self._txt_path = os.path.join(config.TRANSCRIPT_DIR, f"transcript_{self._session_id}.txt")
        self._json_path = os.path.join(config.TRANSCRIPT_DIR, f"transcript_{self._session_id}.json")
        with open(self._txt_path, "w", encoding="utf-8") as f:
            f.write(f"=== ARREST ANALYZER TRANSCRIPT ===\nSession: {self._session_id}\n{'='*50}\n\n")

    def add(self, label, text):
        ts = datetime.now().isoformat()
        entry = {"timestamp": ts, "label": label, "text": text}
        with self._lock:
            self._entries.append(entry)
            with open(self._txt_path, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] [{label}] {text}\n")
            with open(self._json_path, "w", encoding="utf-8") as f:
                json.dump({"session_id": self._session_id, "entries": self._entries}, f, indent=2, ensure_ascii=False)

    def add_emotion(self, dominant, scores):
        with self._lock:
            self._emotion_history.append({
                "timestamp": datetime.now().isoformat(),
                "dominant": dominant,
                "scores": dict(scores) if scores else {},
            })
            if len(self._emotion_history) > 200:
                self._emotion_history = self._emotion_history[-100:]

    def get_full_transcript(self):
        with self._lock:
            return "\n".join(f"[{e['label']}] {e['text']}" for e in self._entries)

    def get_entries(self):
        with self._lock:
            return list(self._entries)

    def get_emotion_history(self):
        with self._lock:
            return list(self._emotion_history)

    @property
    def txt_path(self):
        return self._txt_path
