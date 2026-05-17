"""
config.py — Adjust these before running.
Run `python -c "from core.audio.capture import list_devices; list_devices()"` to list devices.

NOTE: MIC_DEVICE_INDEX and REMOTE_DEVICE_INDEX are overridden at runtime
by the UI dropdowns — you don't need to change them here unless running headless.
Set to None to use the system default for that device.
"""

# ── Audio Device IDs ──────────────────────────────────────────────
# None = system default. Change only if running without UI.
MIC_DEVICE_INDEX    = None   # Your microphone
REMOTE_DEVICE_INDEX = None   # VB-Cable output / BlackHole

# ── Audio Settings ────────────────────────────────────────────────
SAMPLE_RATE = 16000   # Hz — required by Whisper & webrtcvad
CHANNELS    = 1       # Mono
FRAME_MS    = 30      # VAD frame duration: 10, 20, or 30 ms only
VAD_MODE    = 1       # 0 (lenient) -> 3 (aggressive)

# ── Utterance Boundary Detection ─────────────────────────────────
SILENCE_TIMEOUT_S = 1.6   # Seconds of silence before flushing utterance
MIN_SPEECH_S      = 0.25  # Ignore utterances shorter than this (noise filter)

# ── Whisper ───────────────────────────────────────────────────────
WHISPER_MODEL    = "large-v3"  # large-v3 via faster-whisper (int8, ~2.5 GB VRAM)
WHISPER_LANGUAGE = "hi"      # Hindi transcription
WHISPER_DEVICE   = "cuda"    # "cuda" | "cpu" — auto-falls back to cpu

# ── Vision (Emotion Detection) ───────────────────────────────────
WEBCAM_INDEX       = 0       # 0 = default webcam
EMOTION_INTERVAL   = 5       # Analyze every Nth frame

# ── Fraud Detection ──────────────────────────────────────────────
GEMINI_API_KEY         = None  # Set via env var GEMINI_API_KEY or here
FRAUD_CHECK_INTERVAL_S = 15   # Seconds between fraud checks (real-time)

# ── Transcript Storage ───────────────────────────────────────────
TRANSCRIPT_DIR = "output"

# ── UI ────────────────────────────────────────────────────────────
MAX_TRANSCRIPT_LINES = 200
