"""
config.py — Adjust these before running.
Run `python -c "import sounddevice as sd; print(sd.query_devices())"` to list devices.
"""

# ── Audio Device IDs ──────────────────────────────────────────────
# Set to None to auto-select system default
MIC_DEVICE_INDEX    = 29  # Your microphone (local speaker)
REMOTE_DEVICE_INDEX = 17   # VB-Cable output device index (remote/Zoom audio)
                              # e.g. on Windows: "CABLE Output (VB-Audio Virtual Cable)"

# ── Audio Settings ────────────────────────────────────────────────
SAMPLE_RATE   = 16000   # Hz — required by Whisper & webrtcvad
CHANNELS      = 1       # Mono
FRAME_MS      = 30      # VAD frame duration: 10, 20, or 30 ms only
VAD_MODE      = 3      # 0 (lenient) → 3 (aggressive); 3 filters most non-speech

# ── Utterance Boundary Detection ─────────────────────────────────
SILENCE_TIMEOUT_S = 2.2   # Seconds of silence before flushing utterance to Whisper
MIN_SPEECH_S      = 0.4   # Ignore utterances shorter than this (avoids noise bursts)

# ── Whisper ───────────────────────────────────────────────────────
WHISPER_MODEL = "small"   # Options: tiny, base, small, medium, large
                          # base = good balance of speed & accuracy on CPU
WHISPER_LANGUAGE = None  # None = auto-detect; set "en" to force English

# ── UI ────────────────────────────────────────────────────────────
MAX_TRANSCRIPT_LINES = 200  # Rolling buffer in the transcript panel
