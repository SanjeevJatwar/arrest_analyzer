"""
core/audio/capture.py

Opens two sounddevice RawInputStream simultaneously:
  LOCAL  → your microphone (direct capture)
  REMOTE → VB-Cable Output / BlackHole (Zoom audio routed through virtual cable)

Common Windows setup:
  Zoom Settings → Audio → Speaker  =  CABLE Input (VB-Audio)
  App MIC dropdown                 =  your real microphone  (e.g. "Microphone Array")
  App REMOTE dropdown              =  CABLE Output (VB-Audio Virtual Cable)

Run test_devices.py first to confirm which index is which.
"""
import sounddevice as sd
import numpy as np
import config
from core.audio.speech_accumulator import SpeechAccumulator


class DualChannelCapture:
    """
    on_utterance(label, audio_np, sample_rate) fires when a full
    utterance is detected on either channel.
    """

    def __init__(self, on_utterance):
        self._on_utterance = on_utterance
        self._local_acc    = SpeechAccumulator("LOCAL",  on_utterance)
        self._remote_acc   = SpeechAccumulator("REMOTE", on_utterance)
        self._streams      = []

    def start(self):
        blocksize = int(config.SAMPLE_RATE * config.FRAME_MS / 1000)

        local_device  = config.MIC_DEVICE_INDEX
        remote_device = config.REMOTE_DEVICE_INDEX

        # Diagnose immediately — if these print wrong indices, fix the dropdowns
        print(f"\n[Capture] ── Starting dual-channel capture ──")
        print(f"[Capture]   LOCAL  (mic)    device={local_device!r}")
        print(f"[Capture]   REMOTE (vbcable) device={remote_device!r}")
        print(f"[Capture]   sample_rate={config.SAMPLE_RATE}  blocksize={blocksize}  vad_mode={config.VAD_MODE}")

        def _open_stream(device, label, callback):
            """Try opening at 16000 Hz. If device refuses, raise with clear message."""
            try:
                info = sd.query_devices(device) if device is not None else sd.query_devices(sd.default.device[0])
                print(f"[Capture]   {label}: opening '{info['name']}' (max_in={info['max_input_channels']})")
            except Exception:
                pass  # non-fatal, just for logging

            stream = sd.RawInputStream(
                samplerate=config.SAMPLE_RATE,
                blocksize=blocksize,
                device=device,
                channels=config.CHANNELS,
                dtype="int16",
                callback=callback,
            )
            return stream

        local_stream  = _open_stream(local_device,  "LOCAL ",  self._local_cb)
        remote_stream = _open_stream(remote_device, "REMOTE",  self._remote_cb)

        local_stream.start()
        remote_stream.start()
        self._streams = [local_stream, remote_stream]
        print(f"[Capture] Both streams started OK.\n")

    def stop(self):
        for s in self._streams:
            try:
                s.stop()
                s.close()
            except Exception as e:
                print(f"[Capture] Error closing stream: {e}")
        self._streams.clear()
        self._local_acc.stop()
        self._remote_acc.stop()
        print("[Capture] Stopped.")

    # ── Callbacks (called on sounddevice audio thread — keep minimal) ──

    def _local_cb(self, indata, frames, time_info, status):
        if status:
            print(f"[LOCAL  stream status] {status}")
        self._local_acc.feed(bytes(indata))

    def _remote_cb(self, indata, frames, time_info, status):
        if status:
            print(f"[REMOTE stream status] {status}")
        self._remote_acc.feed(bytes(indata))


# ── Utility ───────────────────────────────────────────────────────

def list_devices():
    """Print all audio input devices with their indices."""
    devices = sd.query_devices()
    default_in = sd.default.device[0]
    print("\n── Available Audio Input Devices ──")
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " ◄ DEFAULT" if i == default_in else ""
            print(f"  [{i:2d}] {d['name']:<52} ch={d['max_input_channels']} sr={int(d['default_samplerate'])}{marker}")
    print()


if __name__ == "__main__":
    list_devices()
