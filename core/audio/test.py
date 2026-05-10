from core.audio.capture import DualChannelCapture
import time


def on_utterance(label, audio_np, sample_rate):
    print(f"{label}: received audio {audio_np.shape}")


cap = DualChannelCapture(on_utterance)

print("Starting...")
cap.start()

try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopping...")
    cap.stop()