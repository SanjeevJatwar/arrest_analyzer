"""
config.py — Adjust these before running.
Run `python -c "from core.audio.capture import list_devices; list_devices()"` to list devices.

NOTE: MIC_DEVICE_INDEX and REMOTE_DEVICE_INDEX are overridden at runtime
by the UI dropdowns — you don't need to change them here unless running headless.
Set to None to use the system default for that device.
"""

# ── Audio Device IDs ──────────────────────────────────────────────
# None = system default. Change only if running without UI.
MIC_DEVICE_INDEX    = 18  # Your microphone
REMOTE_DEVICE_INDEX = 27 # VB-Cable output / BlackHole

# ── Audio Settings ────────────────────────────────────────────────
SAMPLE_RATE = 16000   # Hz — required by Whisper & webrtcvad
CHANNELS    = 1       # Mono
FRAME_MS    = 30      # VAD frame duration: 10, 20, or 30 ms only
VAD_MODE    = 1       # 0 (lenient) -> 3 (aggressive)

# ── Utterance Boundary Detection ─────────────────────────────────
SILENCE_TIMEOUT_S = 1.6   # Seconds of silence before flushing utterance
MIN_SPEECH_S      = 0.25  # Ignore utterances shorter than this (noise filter)

# ── Whisper ───────────────────────────────────────────────────────
WHISPER_MODEL    = "small"   # tiny | base | small | medium | large
WHISPER_LANGUAGE = None      # None = auto-detect; "en" to force English

# ── UI ────────────────────────────────────────────────────────────
MAX_TRANSCRIPT_LINES = 200
