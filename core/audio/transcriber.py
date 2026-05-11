"""
core/audio/transcriber.py
Two separate Whisper model instances — one per channel.
LOCAL and REMOTE transcribe in parallel without blocking each other.
"""
import whisper
import numpy as np
import config

_models = {}  # {"LOCAL": model, "REMOTE": model}


def load_models():
    """Load two separate Whisper model instances."""
    for label in ("LOCAL", "REMOTE"):
        print(f"[Whisper] Loading {label} model: {config.WHISPER_MODEL} ...")
        _models[label] = whisper.load_model(config.WHISPER_MODEL)
    print("[Whisper] Both models ready.")


def transcribe(audio: np.ndarray, sample_rate: int, label: str = "LOCAL") -> str:
    model = _models.get(label)
    if model is None:
        raise RuntimeError(f"Model for {label} not loaded. Call load_models() first.")

    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size == 0:
        return ""

    result = model.transcribe(
        audio,
        language=config.WHISPER_LANGUAGE,
        fp16=False,
        condition_on_previous_text=False,
    )
    return result["text"].strip()