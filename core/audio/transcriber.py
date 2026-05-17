"""
core/audio/transcriber.py

Loads Whisper once at startup on CUDA (if available).
transcribe(audio_np, sample_rate) -> str
"""
import whisper
import torch
import numpy as np
import config

_model = None
_device = None


def load_model():
    global _model, _device

    if config.WHISPER_DEVICE == "cuda" and torch.cuda.is_available():
        _device = "cuda"
    else:
        _device = "cpu"

    print(f"[Whisper] Loading model: {config.WHISPER_MODEL} on {_device} ...")
    _model = whisper.load_model(config.WHISPER_MODEL, device=_device)
    print(f"[Whisper] Model ready on {_device}.")


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

    use_fp16 = (_device == "cuda")

    result = _model.transcribe(
        audio,
        language=config.WHISPER_LANGUAGE,
        fp16=use_fp16,
        condition_on_previous_text=False,
    )

    text = result["text"].strip()
    try:
        print(f"[Whisper] DONE: {text[:120]}")
    except UnicodeEncodeError:
        print(f"[Whisper] DONE: {text[:120].encode('ascii', 'replace').decode()}")
    return text
