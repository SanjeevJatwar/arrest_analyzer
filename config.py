import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

MIC_DEVICE_INDEX    = None
REMOTE_DEVICE_INDEX = None

SAMPLE_RATE = 16000
CHANNELS    = 1
FRAME_MS    = 30
VAD_MODE    = 1

SILENCE_TIMEOUT_S = 1.6
MIN_SPEECH_S      = 0.25

WHISPER_MODEL    = "large-v3"
WHISPER_LANGUAGE = "hi"
WHISPER_DEVICE   = "cuda"

WEBCAM_INDEX       = 0
EMOTION_INTERVAL   = 5

GROQ_API_KEY           = os.getenv("GROQ_API_KEY")
FRAUD_CHECK_INTERVAL_S = 15

TRANSCRIPT_DIR = "output"
MAX_TRANSCRIPT_LINES = 200
