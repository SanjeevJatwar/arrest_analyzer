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

    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size == 0:
        print("[Whisper] SKIP empty audio")
        return ""

    print(f"[Whisper] START sr={sample_rate} samples={audio.size}")

    result = _model.transcribe(
        audio,
        language=config.WHISPER_LANGUAGE,
        fp16=False,
        condition_on_previous_text=False,
    )

    text = result["text"].strip()
    print(f"[Whisper] DONE text={text[:120]}")
    return text