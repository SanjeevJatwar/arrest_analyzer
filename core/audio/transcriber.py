"""
core/audio/transcriber.py

Uses faster-whisper (CTranslate2) for accurate Hindi speech-to-text.
Loads the large-v3 model in int8 quantization to fit in 6 GB VRAM.
"""
import numpy as np
import config

_model = None
_device = None


def load_model():
    global _model, _device
    import torch
    from faster_whisper import WhisperModel

    if config.WHISPER_DEVICE == "cuda" and torch.cuda.is_available():
        _device = "cuda"
        compute_type = "int8_float16"  # best speed/accuracy on GPU
    else:
        _device = "cpu"
        compute_type = "int8"

    model_size = config.WHISPER_MODEL

    print(f"[Whisper] Loading faster-whisper model: {model_size} on {_device} ({compute_type}) ...")
    _model = WhisperModel(
        model_size,
        device=_device,
        compute_type=compute_type,
    )
    print(f"[Whisper] Model ready on {_device}.")


def transcribe(audio: np.ndarray, sample_rate: int) -> str:
    if _model is None:
        raise RuntimeError("Call load_model() before transcribe()")

    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size == 0:
        return ""

    if sample_rate != 16000:
        print(f"[Whisper] WARNING: sample_rate={sample_rate}, expected 16000")

    duration = audio.size / sample_rate
    print(f"[Whisper] START {duration:.2f}s ({audio.size} samples)")

    segments, info = _model.transcribe(
        audio,
        language=config.WHISPER_LANGUAGE,
        beam_size=5,
        vad_filter=True,           # built-in VAD for cleaner results
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
        condition_on_previous_text=False,
    )

    # Collect all segment texts
    texts = []
    for seg in segments:
        texts.append(seg.text.strip())

    text = " ".join(texts).strip()
    try:
        print(f"[Whisper] DONE ({info.language} {info.language_probability:.0%}): {text[:120]}")
    except (UnicodeEncodeError, Exception):
        print(f"[Whisper] DONE: transcription complete ({len(text)} chars)")
    return text
