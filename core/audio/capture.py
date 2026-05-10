"""
core/audio/capture.py

Opens two sounddevice input streams simultaneously:
  1. MIC_DEVICE_INDEX   → local speaker (you)
  2. REMOTE_DEVICE_INDEX → VB-Cable (Zoom/remote audio)

Each stream feeds its SpeechAccumulator.
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
        self._local_acc  = SpeechAccumulator("LOCAL",  on_utterance)
        self._remote_acc = SpeechAccumulator("REMOTE", on_utterance)
        self._streams    = []

    def start(self):
        blocksize = int(config.SAMPLE_RATE * config.FRAME_MS / 1000)

        # ── Local mic ─────────────────────────────────────────────
        local_stream = sd.RawInputStream(
            samplerate=config.SAMPLE_RATE,
            blocksize=blocksize,
            device=config.MIC_DEVICE_INDEX,
            channels=config.CHANNELS,
            dtype="int16",
            callback=self._local_cb,
        )

        # ── Remote (VB-Cable) ─────────────────────────────────────
        remote_stream = sd.RawInputStream(
            samplerate=config.SAMPLE_RATE,
            blocksize=blocksize,
            device=config.REMOTE_DEVICE_INDEX,
            channels=config.CHANNELS,
            dtype="int16",
            callback=self._remote_cb,
        )

        local_stream.start()
        remote_stream.start()
        self._streams = [local_stream, remote_stream]

    def stop(self):
        for s in self._streams:
            s.stop()
            s.close()
        self._local_acc.stop()
        self._remote_acc.stop()

    # ── Callbacks (called on sounddevice audio thread) ────────────

    def _local_cb(self, indata, frames, time_info, status):
        self._local_acc.feed(bytes(indata))

    def _remote_cb(self, indata, frames, time_info, status):
        self._remote_acc.feed(bytes(indata))


# ── Utility: list available devices ──────────────────────────────

def list_devices():
    """Print all audio devices — run this to find your device indices."""
    devices = sd.query_devices()
    print("\n── Available Audio Devices ──")
    for i, d in enumerate(devices):
        ins  = d["max_input_channels"]
        outs = d["max_output_channels"]
        flag = " ◄ INPUT" if ins > 0 else ""
        print(f"  [{i:2d}] {d['name']:<45} in={ins} out={outs}{flag}")
    print()


if __name__ == "__main__":
    list_devices()
