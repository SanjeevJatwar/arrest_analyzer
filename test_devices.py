"""
test_devices.py — Run this FIRST to find your correct device indices.

Usage:
    python test_devices.py

It will:
  1. List every input device
  2. Record 3 seconds from EACH input device
  3. Show which ones actually picked up audio (RMS > threshold)
  4. Print the exact index to put in the UI dropdown for MIC and REMOTE

Run this while speaking normally, so your mic shows activity.
VB-Cable Output will show activity only if Zoom/audio is playing through it.
"""

import sounddevice as sd
import numpy as np
import time

TEST_SECONDS = 3
RMS_THRESHOLD = 0.001


def test_all_input_devices():

    devices = sd.query_devices()

    input_devices = [
        (i, d)
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]

    print("\n" + "=" * 75)
    print(" DEVICE TESTER — Speak now / keep Zoom audio playing")
    print("=" * 75)

    print(f"\nFound {len(input_devices)} input devices.\n")

    results = []

    for idx, dev in input_devices:

        name = dev["name"]

        # Use device native sample rate
        default_sr = int(dev["default_samplerate"] or 48000)

        print(f"[{idx:2d}] {name[:55]:<55}", end=" ", flush=True)

        rms_vals = []

        # =========================================================
        # CALLBACK
        # =========================================================
        def cb(indata, frames, time_info, status):

            try:
                # Raw bytes -> numpy array
                audio = np.frombuffer(indata, dtype=np.int16)

                if len(audio) == 0:
                    return

                # Normalize
                audio = audio.astype(np.float32) / 32768.0

                # RMS
                rms = np.sqrt(np.mean(audio ** 2))

                rms_vals.append(float(rms))

            except Exception as e:
                print(f"\nCallback error: {e}")

        worked = False
        used_sr = None

        # =========================================================
        # TRY MULTIPLE SAMPLE RATES
        # =========================================================
        sample_rates_to_try = [
            default_sr,
            48000,
            44100,
            32000,
            16000,
        ]

        # remove duplicates while preserving order
        sample_rates_to_try = list(dict.fromkeys(sample_rates_to_try))

        for try_sr in sample_rates_to_try:

            try:

                with sd.RawInputStream(
                    device=idx,
                    samplerate=try_sr,
                    channels=1,
                    dtype="int16",
                    blocksize=max(256, try_sr // 10),
                    callback=cb,
                ):

                    time.sleep(TEST_SECONDS)

                worked = True
                used_sr = try_sr
                break

            except Exception:
                rms_vals.clear()
                continue

        # =========================================================
        # FAILED
        # =========================================================
        if not worked:

            print("FAILED (unsupported format/sample rate)")

            results.append({
                "idx": idx,
                "name": name,
                "rms": 0.0,
                "active": False,
                "sr": None
            })

            continue

        # =========================================================
        # RESULTS
        # =========================================================
        avg_rms = float(np.mean(rms_vals)) if rms_vals else 0.0

        active = avg_rms > RMS_THRESHOLD

        bar = "#" * min(40, int(avg_rms * 800))

        status = "<< AUDIO DETECTED" if active else "(silent)"

        print(
            f"sr={used_sr:<6} "
            f"rms={avg_rms:.5f} "
            f"{bar} "
            f"{status}"
        )

        results.append({
            "idx": idx,
            "name": name,
            "rms": avg_rms,
            "active": active,
            "sr": used_sr
        })

    # =============================================================
    # SUMMARY
    # =============================================================
    print("\n" + "=" * 75)
    print(" SUMMARY")
    print("=" * 75)

    active_devices = [r for r in results if r["active"]]

    if active_devices:

        active_devices.sort(key=lambda x: -x["rms"])

        print(f"\nDevices with audio detected ({len(active_devices)}):\n")

        for r in active_devices:

            print(
                f"[{r['idx']:2d}] "
                f"sr={str(r['sr']):<6} "
                f"rms={r['rms']:.5f} "
                f"{r['name'][:50]}"
            )

    else:

        print("\nNo active audio detected.\n")

        print("Make sure:")
        print("  • You are speaking")
        print("  • Zoom/Meet audio is playing")
        print("  • VB-Cable is configured correctly")
        print("  • Windows privacy mic access is enabled")

    # =============================================================
    # HELP
    # =============================================================
    print("\n" + "=" * 75)

    print("\nHOW TO USE:\n")

    print("MIC (LOCAL)")
    print("  → Device where YOUR voice showed activity")

    print("\nREMOTE")
    print("  → VB-Cable Output / Stereo Mix / Virtual Cable")
    print("  → Should show activity only when call audio plays")

    print("\nUse those device indices inside your app dropdowns.")

    print("\n" + "=" * 75 + "\n")


if __name__ == "__main__":

    test_all_input_devices()