"""
core/vision/emotion_detector.py

Real-time facial emotion detection using DeepFace.
Runs on its own daemon thread, captures webcam frames, and emits
emotion predictions via callback.
"""
import threading
import time
import cv2
import numpy as np


class EmotionDetector(threading.Thread):
    """
    on_frame(frame_rgb, emotions_dict, dominant_emotion)
    on_status(msg, color)
    """

    def __init__(self, camera_index=0, analyze_every_n=5,
                 on_frame=None, on_status=None):
        super().__init__(daemon=True, name="EmotionDetector")
        self.camera_index    = camera_index
        self.analyze_every_n = analyze_every_n
        self.on_frame        = on_frame or (lambda *a: None)
        self.on_status       = on_status or (lambda *a: None)
        self._running        = False
        self._cap            = None

    def run(self):
        self._running = True

        try:
            from deepface import DeepFace
        except ImportError:
            self.on_status("CV ERROR: pip install deepface", "#ff4444")
            return

        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            self.on_status("CV ERROR: Cannot open webcam", "#ff4444")
            return

        self.on_status("CV: Webcam active", "#00e5ff")
        frame_count = 0
        last_emotions = {}
        last_dominant = "neutral"

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_count % self.analyze_every_n == 0:
                try:
                    results = DeepFace.analyze(
                        frame, actions=["emotion"],
                        enforce_detection=False,
                        silent=True,
                    )
                    if isinstance(results, list):
                        results = results[0]
                    last_emotions = results.get("emotion", {})
                    last_dominant = results.get("dominant_emotion", "neutral")
                except Exception:
                    pass

            self.on_frame(frame_rgb, last_emotions, last_dominant)
            time.sleep(0.03)  # ~30 fps cap

        if self._cap:
            self._cap.release()

    def stop(self):
        self._running = False
