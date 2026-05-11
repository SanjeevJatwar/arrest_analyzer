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


def transcribe(audio: np.ndarray, sample_rate: int) -> str:
    """
    audio   : float32 numpy array, values in [-1, 1]
    Returns : transcribed string (stripped)
    """
    if _model is None:
        raise RuntimeError("Call load_model() before transcribe()")

    # Whisper expects float32 @ 16kHz mono — already correct from accumulator
    result = _model.transcribe(
        audio,
        language=config.WHISPER_LANGUAGE,
        fp16=False,          # fp16=True only if CUDA available
        condition_on_previous_text=False,
    )
    print("Transcribing...")
    return result["text"].strip()
