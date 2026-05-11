"""
core/audio/transcriber.py
Two separate Whisper model instances — one per channel.
LOCAL and REMOTE transcribe in parallel without blocking each other.
"""
import whisper
import numpy as np
import config

_models = {}  # {"LOCAL": model, "REMOTE": model}


import threading

_models_lock = threading.Lock()
_loaded = False

def load_models():
    global _loaded
    with _models_lock:
        if _loaded:
            return
        for label in ("LOCAL", "REMOTE"):
            print(f"[Whisper] Loading {label} model: {config.WHISPER_MODEL} ...")
            _models[label] = whisper.load_model(config.WHISPER_MODEL)
        _loaded = True
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
    try:
        result = model.transcribe(...)
        return result["text"].strip()
    except Exception as e:
        print(f"[Whisper] {label} transcribe failed: {e}")
        return ""