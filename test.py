import sounddevice as sd
import numpy as np
import time

devices = sd.query_devices()

print("\n===== CHECKING INPUT DEVICES =====\n")

for idx, dev in enumerate(devices):

    # Skip non-input devices
    if dev['max_input_channels'] <= 0:
        continue

    print(f"\nTesting Device {idx}: {dev['name']}")

    samplerate = int(dev['default_samplerate'])

    rms_values = []

    def callback(indata, frames, time_info, status):
        if status:
            print("Status:", status)

        rms = np.sqrt(np.mean(indata**2))
        rms_values.append(rms)

    try:
        with sd.InputStream(
            device=idx,
            samplerate=samplerate,
            channels=1,
            dtype='float32',
            callback=callback
        ):

            print("Listening for 3 seconds...")
            time.sleep(3)

        avg_rms = np.mean(rms_values) if rms_values else 0

        print(f"Average RMS: {avg_rms}")

        if avg_rms > 0.001:
            print(">>> ACTIVE AUDIO DETECTED <<<")
        else:
            print("No significant audio.")

    except Exception as e:
        print("FAILED:", e)

print("\n===== DONE =====")