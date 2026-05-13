"""
core/audio/transcriber.py

Loads Whisper once at startup.
transcribe(audio_np, sample_rate) -> str

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


def transcribe(audio: np.ndarray, sample_rate: int) -> str:
    if _model is None:
        raise RuntimeError("Call load_model() before transcribe()")

    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size == 0:
        print("[Whisper] SKIP — empty audio")
        return ""

    # Whisper expects audio at 16000 Hz; warn if mismatch but don't crash
    if sample_rate != 16000:
        print(f"[Whisper] WARNING: sample_rate={sample_rate}, expected 16000")

    print(f"[Whisper] START samples={audio.size} ({audio.size/sample_rate:.2f}s)")

    result = _model.transcribe(
        audio,
        language=config.WHISPER_LANGUAGE,
        fp16=False,
        condition_on_previous_text=False,
    )

    text = result["text"].strip()
    print(f"[Whisper] DONE -> {text[:120]!r}")
    return text
