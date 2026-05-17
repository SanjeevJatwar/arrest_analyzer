"""
core/audio/capture.py

Opens two sounddevice RawInputStream simultaneously:
  LOCAL  -> your microphone (direct capture)
  REMOTE -> VB-Cable Output / BlackHole (Zoom audio routed through virtual cable)

Handles sample rate mismatch: opens at device native rate and
resamples to 16 kHz in real-time before feeding the VAD accumulator.
"""
import sounddevice as sd
import numpy as np
import config
from core.audio.speech_accumulator import SpeechAccumulator


def _resample_int16(data_bytes, from_rate, to_rate):
    """Fast numpy-based resample of int16 PCM bytes."""
    if from_rate == to_rate:
        return data_bytes
    audio = np.frombuffer(data_bytes, dtype=np.int16)
    if len(audio) == 0:
        return data_bytes
    target_len = int(len(audio) * to_rate / from_rate)
    if target_len == 0:
        return b""
    resampled = np.interp(
        np.linspace(0, len(audio) - 1, target_len),
        np.arange(len(audio)),
        audio.astype(np.float64),
    ).astype(np.int16)
    return resampled.tobytes()


def _find_working_rate(device_index):
    """Find a sample rate the device actually supports."""
    if device_index is None:
        info = sd.query_devices(sd.default.device[0])
    else:
        info = sd.query_devices(device_index)
    native = int(info.get("default_samplerate", 48000))

    for rate in [config.SAMPLE_RATE, native, 48000, 44100, 32000, 16000]:
        try:
            sd.check_input_settings(
                device=device_index, samplerate=rate,
                channels=config.CHANNELS, dtype="int16",
            )
            return rate
        except Exception:
            continue
    return native


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
        self._local_native  = None
        self._remote_native = None

    def start(self):
        local_device  = config.MIC_DEVICE_INDEX
        remote_device = config.REMOTE_DEVICE_INDEX

        self._local_native  = _find_working_rate(local_device)
        self._remote_native = _find_working_rate(remote_device)

        print(f"\n[Capture] -- Starting dual-channel capture --")
        print(f"[Capture]   LOCAL  device={local_device!r}  native={self._local_native}Hz")
        print(f"[Capture]   REMOTE device={remote_device!r}  native={self._remote_native}Hz")
        print(f"[Capture]   target={config.SAMPLE_RATE}Hz  vad_mode={config.VAD_MODE}")

        def _open(device, label, native_rate, cb):
            blocksize = int(native_rate * config.FRAME_MS / 1000)
            try:
                info = sd.query_devices(device) if device is not None \
                    else sd.query_devices(sd.default.device[0])
                print(f"[Capture]   {label}: '{info['name']}' bs={blocksize}")
            except Exception:
                pass
            return sd.RawInputStream(
                samplerate=native_rate, blocksize=blocksize,
                device=device, channels=config.CHANNELS,
                dtype="int16", callback=cb,
            )

        local_s  = _open(local_device,  "LOCAL ", self._local_native,  self._local_cb)
        remote_s = _open(remote_device, "REMOTE", self._remote_native, self._remote_cb)

        local_s.start()
        remote_s.start()
        self._streams = [local_s, remote_s]
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

    # ── Callbacks (sounddevice audio thread — keep fast) ──────────

    def _local_cb(self, indata, frames, time_info, status):
        if status:
            print(f"[LOCAL  stream status] {status}")
        resampled = _resample_int16(bytes(indata), self._local_native, config.SAMPLE_RATE)
        self._local_acc.feed(resampled)

    def _remote_cb(self, indata, frames, time_info, status):
        if status:
            print(f"[REMOTE stream status] {status}")
        resampled = _resample_int16(bytes(indata), self._remote_native, config.SAMPLE_RATE)
        self._remote_acc.feed(resampled)


# ── Utility ───────────────────────────────────────────────────────

def list_devices():
    """Print all audio input devices with their indices."""
    devices = sd.query_devices()
    default_in = sd.default.device[0]
    print("\n-- Available Audio Input Devices --")
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " << DEFAULT" if i == default_in else ""
            print(f"  [{i:2d}] {d['name']:<52} ch={d['max_input_channels']} sr={int(d['default_samplerate'])}{marker}")
    print()


if __name__ == "__main__":
    list_devices()
