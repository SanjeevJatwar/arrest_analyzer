"""
core/audio/transcription_worker.py

Dedicated transcription worker threads — one for LOCAL, one for REMOTE.
Each pulls utterances from its own queue and transcribes sequentially.
A shared lock ensures the Whisper model is accessed by one thread at a time.
"""
import threading
import queue
import numpy as np

_transcribe_lock = threading.Lock()


class TranscriptionWorker(threading.Thread):
    """
    Worker thread that pulls (audio_np, sample_rate) from a queue,
    transcribes via Whisper, and calls on_result(label, text).
    """

    def __init__(self, label: str, on_result, on_status):
        super().__init__(daemon=True, name=f"TranscriptionWorker-{label}")
        self.label     = label
        self.on_result = on_result
        self.on_status = on_status
        self._queue    = queue.Queue()
        self._running  = True

    def enqueue(self, audio_np: np.ndarray, sample_rate: int):
        self._queue.put((audio_np, sample_rate))

    def run(self):
        from core.audio import transcriber

        while self._running:
            try:
                audio_np, sample_rate = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self.on_status(f">> TRANSCRIBING [{self.label}]...", "#ffaa00")
                with _transcribe_lock:
                    text = transcriber.transcribe(audio_np, sample_rate)
                if text:
                    self.on_result(self.label, text)
                self.on_status(">> RECORDING", "#ff4444")
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.on_status(f"ERR [{self.label}]: {e}", "#ff4444")

    def stop(self):
        self._running = False
