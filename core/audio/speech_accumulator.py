"""
core/audio/speech_accumulator.py

Accumulates raw audio frames using WebRTC VAD.
Emits a complete utterance (as numpy array) only after detecting
a silence gap >= SILENCE_TIMEOUT_S.  This gives Whisper full
sentences rather than tiny chunks.
"""
import time
import threading
import numpy as np
import webrtcvad
import config


class SpeechAccumulator:
    """
    Usage:
        acc = SpeechAccumulator(label="LOCAL", on_utterance=my_callback)
        acc.feed(pcm_bytes)   # call this from your sounddevice callback
    
    on_utterance(label: str, audio: np.ndarray, sample_rate: int) is called
    on the accumulator's internal timer thread whenever an utterance ends.
    """

    def __init__(self, label: str, on_utterance):
        self.label        = label
        self.on_utterance = on_utterance
        self.vad          = webrtcvad.Vad(config.VAD_MODE)
        self.sample_rate  = config.SAMPLE_RATE
        self.frame_bytes  = int(self.sample_rate * config.FRAME_MS / 1000) * 2  # int16 → 2 bytes/sample

        self._lock        = threading.Lock()
        self._speech_buf  = []        # raw int16 bytes while speaking
        self._last_speech = None      # timestamp of last voiced frame
        self._is_speaking = False
        self._leftover    = b""       # incomplete frame carry-over

        # Background thread watches for silence timeout
        self._running = True
        self._watcher = threading.Thread(target=self._silence_watcher, daemon=True)
        self._watcher.start()

    # ── Public ────────────────────────────────────────────────────

    def feed(self, pcm_bytes: bytes):
        """Feed raw int16 PCM bytes (any length). Thread-safe."""
        data = self._leftover + pcm_bytes
        offset = 0
        with self._lock:
            while offset + self.frame_bytes <= len(data):
                frame = data[offset: offset + self.frame_bytes]
                offset += self.frame_bytes
                try:
                    voiced = self.vad.is_speech(frame, self.sample_rate)
                except Exception:
                    voiced = False

                if voiced:
                    self._speech_buf.append(frame)
                    self._last_speech = time.monotonic()
                    self._is_speaking = True
                elif self._is_speaking:
                    # keep a small trailing buffer during brief pauses
                    self._speech_buf.append(frame)

            self._leftover = data[offset:]

    def stop(self):
        self._running = False

    # ── Internal ──────────────────────────────────────────────────

    def _silence_watcher(self):
        """Polls for silence timeout and flushes complete utterances."""
        while self._running:
            time.sleep(0.1)
            with self._lock:
                if (
                    self._is_speaking
                    and self._last_speech is not None
                    and (time.monotonic() - self._last_speech) >= config.SILENCE_TIMEOUT_S
                ):
                    self._flush()

    def _flush(self):
        """Must be called with self._lock held."""
        if not self._speech_buf:
            return
        raw = b"".join(self._speech_buf)
        self._speech_buf.clear()
        self._is_speaking = False
        self._last_speech = None

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(audio) / self.sample_rate

        if duration < config.MIN_SPEECH_S:
            return  # too short — probably noise

        # Fire callback on a separate thread so we don't block the watcher
        threading.Thread(
            target=self.on_utterance,
            args=(self.label, audio, self.sample_rate),
            daemon=True,
        ).start()
