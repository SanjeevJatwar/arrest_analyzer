"""
core/audio/transcriber.py

Loads Whisper once at startup.
transcribe(audio_np, sample_rate) → str

Called from the speech accumulator callback (already on its own thread).
"""
import whisper
import numpy as np
import config

_model = None


def load_model():
    global _model
    print(f"[Whisper] Loading model: {config.WHISPER_MODEL} ...")
    _model = whisper.load_model(config.WHISPER_MODEL)
    print("[Whisper] Model ready.")


# core/audio/transcriber.py
def transcribe(audio: np.ndarray, sample_rate: int) -> str:
    if _model is None:
        raise RuntimeError("Call load_model() before transcribe()")

    # ---- GUARDS (prevents the 768/features crash) ----
    if audio is None or len(audio) == 0:
        print("[Whisper] skipped: empty audio")
        return ""

    if not np.isfinite(audio).all():
        print("[Whisper] skipped: non-finite audio")
        return ""

    # Ensure 1D mono
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if len(audio) < int(0.2 * sample_rate):  # <200ms
        print("[Whisper] skipped: too short")
        return ""