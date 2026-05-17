"""
ui/main_window.py — Full integrated UI
Multi-threaded transcription, CV emotion, fraud analysis, transcript storage.
"""
import time
import subprocess
import sys
import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFrame, QSplitter,
    QComboBox, QMessageBox, QProgressBar, QGroupBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor, QImage, QPixmap
import numpy as np
import sounddevice as sd


class TranscriptSignals(QObject):
    new_entry      = pyqtSignal(str, str)
    status_update  = pyqtSignal(str, str)
    emotion_frame  = pyqtSignal(object, dict, str)
    fraud_result   = pyqtSignal(int, str, dict)


class ModelLoaderThread(QThread):
    finished = pyqtSignal()
    error    = pyqtSignal(str)
    def run(self):
        try:
            from core.audio import transcriber
            transcriber.load_model()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


STYLE = """
QMainWindow, QWidget#central { background: #0a0c0e; }
QLabel#title {
    font-family: 'Courier New', monospace; font-size: 13px; font-weight: bold;
    color: #c8ff00; letter-spacing: 4px;
}
QTextEdit {
    background: #060809; border: none; font-family: 'Courier New', monospace;
    font-size: 12px; padding: 14px; selection-background-color: #1a3040;
}
QPushButton#start_btn {
    background: #c8ff00; color: #0a0c0e; font-family: 'Courier New', monospace;
    font-size: 11px; font-weight: bold; letter-spacing: 2px; padding: 10px 28px;
    border: none; min-width: 110px;
}
QPushButton#start_btn:hover { background: #d4ff33; }
QPushButton#start_btn:pressed { background: #a8d900; }
QPushButton#start_btn:disabled { background: #2a2a2a; color: #555; }
QPushButton#stop_btn {
    background: transparent; color: #ff4444; font-family: 'Courier New', monospace;
    font-size: 11px; font-weight: bold; letter-spacing: 2px; padding: 10px 28px;
    border: 1px solid #ff4444; min-width: 110px;
}
QPushButton#stop_btn:hover { background: #1a0000; }
QPushButton#stop_btn:disabled { color: #333; border-color: #333; }
QPushButton#clear_btn, QPushButton#test_btn {
    background: transparent; color: #555; font-family: 'Courier New', monospace;
    font-size: 10px; letter-spacing: 2px; padding: 8px 16px; border: 1px solid #222;
}
QPushButton#clear_btn:hover { color: #888; border-color: #444; }
QPushButton#test_btn { color: #888; border-color: #333; }
QPushButton#test_btn:hover { color: #ffaa00; border-color: #ffaa00; }
QLabel#status_pill {
    font-family: 'Courier New', monospace; font-size: 10px;
    letter-spacing: 2px; padding: 4px 10px;
}
QLabel#timer_label {
    font-family: 'Courier New', monospace; font-size: 11px;
    color: #444; letter-spacing: 2px;
}
QFrame#header_bar { background: #0d1014; border-bottom: 1px solid #1a1d20; }
QFrame#footer_bar { background: #0d1014; border-top: 1px solid #1a1d20; }
QComboBox {
    background: #111416; color: #888; font-family: 'Courier New', monospace;
    font-size: 9px; border: 1px solid #222; padding: 3px 8px; min-width: 200px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #111416; color: #888; selection-background-color: #1a2a30;
}
QSplitter::handle { background: #1a1d20; width: 1px; }
QGroupBox {
    font-family: 'Courier New', monospace; font-size: 10px;
    color: #888; border: 1px solid #1a1d20; margin-top: 8px; padding-top: 14px;
}
QGroupBox::title { subcontrol-origin: margin; padding: 0 6px; }
QProgressBar {
    background: #111416; border: 1px solid #222; height: 14px;
    font-family: 'Courier New', monospace; font-size: 9px; color: #fff;
    text-align: center;
}
QProgressBar::chunk { background: qlineargradient(x1:0, x2:1, stop:0 #00c853, stop:1 #ff4444); }
"""


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.signals         = TranscriptSignals()
        self.capture         = None
        self._session_start  = None
        self._timer          = QTimer()
        self._timer.timeout.connect(self._tick_timer)
        self._model_loaded   = False
        self._local_worker   = None
        self._remote_worker  = None
        self._transcript_store = None
        self._emotion_detector = None
        self._fraud_thread     = None

        self.setWindowTitle("ARREST ANALYZER")
        self.resize(1280, 800)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(STYLE)

        self._build_ui()
        self._connect_signals()
        self._load_devices()
        self._start_model_load()

    # ── UI BUILD ─────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        root.addWidget(self._build_main_area(), stretch=1)
        root.addWidget(self._build_footer())

    def _build_header(self):
        bar = QFrame(); bar.setObjectName("header_bar"); bar.setFixedHeight(58)
        h = QHBoxLayout(bar); h.setContentsMargins(20, 0, 20, 0)
        title = QLabel(">> DIGITAL ARREST ANALYZER  //  LIVE")
        title.setObjectName("title"); h.addWidget(title); h.addStretch()

        self.status_pill = QLabel(">> LOADING MODEL")
        self.status_pill.setObjectName("status_pill")
        self.status_pill.setStyleSheet("color:#555; background:#111416; border:1px solid #222;")
        h.addWidget(self.status_pill); h.addSpacing(12)

        test_btn = QPushButton("TEST DEVICES"); test_btn.setObjectName("test_btn")
        test_btn.clicked.connect(self._run_device_test); h.addWidget(test_btn); h.addSpacing(8)

        self.clear_btn = QPushButton("CLEAR"); self.clear_btn.setObjectName("clear_btn")
        self.clear_btn.clicked.connect(self._clear_transcripts); h.addWidget(self.clear_btn); h.addSpacing(8)

        self.start_btn = QPushButton(">>  START"); self.start_btn.setObjectName("start_btn")
        self.start_btn.setEnabled(False); self.start_btn.clicked.connect(self._start_capture)
        h.addWidget(self.start_btn)

        self.stop_btn = QPushButton("||  STOP"); self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self._stop_capture)
        h.addWidget(self.stop_btn)
        return bar

    def _build_main_area(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # LEFT: Transcript panels
        transcript_w = QWidget()
        transcript_w.setStyleSheet("background: #060809;")
        tv = QVBoxLayout(transcript_w); tv.setContentsMargins(0,0,0,0); tv.setSpacing(0)

        inner_split = QSplitter(Qt.Orientation.Horizontal)
        inner_split.setHandleWidth(1)

        # LOCAL pane
        local_w = QWidget(); local_w.setStyleSheet("background: #060809;")
        lv = QVBoxLayout(local_w); lv.setContentsMargins(0,0,0,0); lv.setSpacing(0)
        local_lbl = QLabel(">> LOCAL  //  YOU  (microphone)")
        local_lbl.setStyleSheet(
            "font-family:'Courier New'; font-size:10px; font-weight:bold;"
            "letter-spacing:3px; color:#00e5ff; background:#001a1f;"
            "border-left:3px solid #00e5ff; padding:8px 14px;")
        lv.addWidget(local_lbl)
        self.local_text = QTextEdit(); self.local_text.setReadOnly(True)
        self.local_text.setPlaceholderText("[ waiting for speech ]")
        self.local_text.setStyleSheet("color: #00e5ff;"); lv.addWidget(self.local_text, stretch=1)

        # REMOTE pane
        remote_w = QWidget(); remote_w.setStyleSheet("background: #060809;")
        rv = QVBoxLayout(remote_w); rv.setContentsMargins(0,0,0,0); rv.setSpacing(0)
        remote_lbl = QLabel(">> REMOTE  //  VB-CABLE OUTPUT")
        remote_lbl.setStyleSheet(
            "font-family:'Courier New'; font-size:10px; font-weight:bold;"
            "letter-spacing:3px; color:#ff6b35; background:#1f0d00;"
            "border-left:3px solid #ff6b35; padding:8px 14px;")
        rv.addWidget(remote_lbl)
        self.remote_text = QTextEdit(); self.remote_text.setReadOnly(True)
        self.remote_text.setPlaceholderText("[ waiting for remote audio ]")
        self.remote_text.setStyleSheet("color: #ff6b35;"); rv.addWidget(self.remote_text, stretch=1)

        inner_split.addWidget(local_w); inner_split.addWidget(remote_w)
        inner_split.setSizes([400, 400])
        tv.addWidget(inner_split, stretch=1)
        splitter.addWidget(transcript_w)

        # RIGHT: CV + Fraud panel
        right_w = QWidget(); right_w.setStyleSheet("background: #0a0c0e;")
        rlay = QVBoxLayout(right_w); rlay.setContentsMargins(8,8,8,8); rlay.setSpacing(8)

        # Webcam feed
        cam_group = QGroupBox(">> WEBCAM  //  EMOTION DETECTION")
        cam_group.setStyleSheet(
            "QGroupBox { color: #00e5ff; border: 1px solid #002a33; }"
            "QGroupBox::title { color: #00e5ff; }")
        cam_lay = QVBoxLayout(cam_group)
        self.cam_label = QLabel("[ webcam off ]")
        self.cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_label.setMinimumSize(280, 210)
        self.cam_label.setStyleSheet("color: #333; font-family:'Courier New'; font-size:10px; background:#060809;")
        cam_lay.addWidget(self.cam_label)
        self.emotion_label = QLabel("Emotion: --")
        self.emotion_label.setStyleSheet("font-family:'Courier New'; font-size:12px; font-weight:bold; color:#00e5ff; padding:4px;")
        cam_lay.addWidget(self.emotion_label)
        rlay.addWidget(cam_group)

        # Fraud panel
        fraud_group = QGroupBox(">> FRAUD ANALYSIS")
        fraud_group.setStyleSheet(
            "QGroupBox { color: #ff4444; border: 1px solid #330000; }"
            "QGroupBox::title { color: #ff4444; }")
        flay = QVBoxLayout(fraud_group)
        self.fraud_score_bar = QProgressBar(); self.fraud_score_bar.setRange(0, 100)
        self.fraud_score_bar.setValue(0); self.fraud_score_bar.setFormat("FRAUD RISK: %p%")
        flay.addWidget(self.fraud_score_bar)
        self.fraud_verdict = QLabel("Verdict: WAITING FOR DATA...")
        self.fraud_verdict.setStyleSheet("font-family:'Courier New'; font-size:11px; color:#888; padding:4px;")
        self.fraud_verdict.setWordWrap(True)
        flay.addWidget(self.fraud_verdict)
        self.fraud_details = QTextEdit(); self.fraud_details.setReadOnly(True)
        self.fraud_details.setMaximumHeight(120)
        self.fraud_details.setStyleSheet("color:#666; font-size:10px; background:#060809;")
        self.fraud_details.setPlaceholderText("[ analysis details will appear here ]")
        flay.addWidget(self.fraud_details)
        rlay.addWidget(fraud_group)
        rlay.addStretch()

        splitter.addWidget(right_w)
        splitter.setSizes([800, 350])
        return splitter

    def _build_footer(self):
        bar = QFrame(); bar.setObjectName("footer_bar"); bar.setFixedHeight(42)
        h = QHBoxLayout(bar); h.setContentsMargins(20,0,20,0); h.setSpacing(8)
        h.addWidget(self._small_label("MIC:"))
        self.mic_combo = QComboBox(); h.addWidget(self.mic_combo); h.addSpacing(16)
        h.addWidget(self._small_label("REMOTE:"))
        self.remote_combo = QComboBox(); h.addWidget(self.remote_combo); h.addStretch()
        self.timer_label = QLabel("00:00:00"); self.timer_label.setObjectName("timer_label")
        h.addWidget(self.timer_label)
        return bar

    def _small_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("font-family:'Courier New'; font-size:9px; color:#555; letter-spacing:1px;")
        return lbl

    # ── DEVICE LOADING ───────────────────────────────────────────

    def _load_devices(self):
        self.mic_combo.clear(); self.remote_combo.clear()
        try: default_in = sd.default.device[0]
        except Exception: default_in = -1

        self.mic_combo.addItem("[ System Default ]", None)
        self.remote_combo.addItem("[ None ]", None)
        vbcable_idx = None

        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] <= 0: continue
            name = d["name"]; label = f"[{i}] {name[:46]}"
            if i == default_in: label += "  << default"
            self.mic_combo.addItem(label, i); self.remote_combo.addItem(label, i)
            if "cable output" in name.lower() or "vb-audio" in name.lower():
                if vbcable_idx is None: vbcable_idx = self.remote_combo.count() - 1

        if vbcable_idx is not None: self.remote_combo.setCurrentIndex(vbcable_idx)
        if default_in >= 0:
            for j in range(self.mic_combo.count()):
                if self.mic_combo.itemData(j) == default_in:
                    self.mic_combo.setCurrentIndex(j); break

    # ── MODEL LOAD ───────────────────────────────────────────────

    def _connect_signals(self):
        self.signals.new_entry.connect(self._append_transcript)
        self.signals.status_update.connect(self._set_status)
        self.signals.emotion_frame.connect(self._update_emotion)
        self.signals.fraud_result.connect(self._update_fraud)

    def _start_model_load(self):
        self.loader = ModelLoaderThread()
        self.loader.finished.connect(self._on_model_ready)
        self.loader.error.connect(lambda e: self._set_status(f"MODEL ERROR: {e}", "#ff4444"))
        self.loader.start()

    def _on_model_ready(self):
        self._model_loaded = True
        self._set_status(">> READY", "#c8ff00")
        self.start_btn.setEnabled(True)

    # ── CONTROLS ─────────────────────────────────────────────────

    def _start_capture(self):
        import config
        from core.audio.capture import DualChannelCapture
        from core.audio.transcription_worker import TranscriptionWorker
        from core.audio.transcript_store import TranscriptStore
        from core.vision.emotion_detector import EmotionDetector
        from core.fusion.fraud_analyzer import FraudAnalyzerThread

        config.MIC_DEVICE_INDEX    = self.mic_combo.currentData()
        config.REMOTE_DEVICE_INDEX = self.remote_combo.currentData()

        if (config.MIC_DEVICE_INDEX is not None
                and config.MIC_DEVICE_INDEX == config.REMOTE_DEVICE_INDEX):
            QMessageBox.warning(self, "Same Device", "MIC and REMOTE are the same device.")

        # Transcript store
        self._transcript_store = TranscriptStore()

        # Transcription workers (one per channel)
        self._local_worker = TranscriptionWorker(
            "LOCAL", self._on_transcription_result,
            lambda m, c: self.signals.status_update.emit(m, c))
        self._remote_worker = TranscriptionWorker(
            "REMOTE", self._on_transcription_result,
            lambda m, c: self.signals.status_update.emit(m, c))
        self._local_worker.start()
        self._remote_worker.start()

        # Audio capture
        try:
            self.capture = DualChannelCapture(on_utterance=self._on_utterance)
            self.capture.start()
        except Exception as e:
            QMessageBox.critical(self, "Capture Error",
                f"{e}\n\nTip: Run TEST DEVICES to verify device indices.")
            return

        # Emotion detector
        try:
            self._emotion_detector = EmotionDetector(
                camera_index=config.WEBCAM_INDEX,
                analyze_every_n=config.EMOTION_INTERVAL,
                on_frame=lambda f, e, d: self.signals.emotion_frame.emit(f, e, d),
                on_status=lambda m, c: self.signals.status_update.emit(m, c),
            )
            self._emotion_detector.start()
        except Exception as e:
            print(f"[UI] Emotion detector failed: {e}")

        # Fraud analyzer
        try:
            self._fraud_thread = FraudAnalyzerThread(
                get_transcript_fn=self._transcript_store.get_full_transcript,
                on_result=lambda s, v, d: self.signals.fraud_result.emit(s, v, d),
                interval_s=config.FRAUD_CHECK_INTERVAL_S,
            )
            self._fraud_thread.start()
        except Exception as e:
            print(f"[UI] Fraud analyzer failed: {e}")

        self._session_start = time.monotonic()
        self._timer.start(1000)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_status(">> RECORDING", "#ff4444")

    def _stop_capture(self):
        if self.capture:
            self.capture.stop(); self.capture = None
        if self._local_worker:
            self._local_worker.stop(); self._local_worker = None
        if self._remote_worker:
            self._remote_worker.stop(); self._remote_worker = None
        if self._emotion_detector:
            self._emotion_detector.stop(); self._emotion_detector = None
        if self._fraud_thread:
            self._fraud_thread.stop(); self._fraud_thread = None
        self._timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_status("|| STOPPED", "#555")
        self.cam_label.setText("[ webcam off ]")
        self.cam_label.setPixmap(QPixmap())

    def _clear_transcripts(self):
        self.local_text.clear(); self.remote_text.clear()

    def _run_device_test(self):
        script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "test_devices.py")
        if not os.path.exists(script):
            QMessageBox.warning(self, "Not Found", f"Not found: {script}"); return
        try:
            if sys.platform == "win32":
                subprocess.Popen(["cmd", "/c", "start", "cmd", "/k",
                     sys.executable, script], creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen([sys.executable, script])
            self._set_status(">> DEVICE TEST RUNNING", "#ffaa00")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ── UTTERANCE CALLBACK (audio daemon thread) ─────────────────

    def _on_utterance(self, label: str, audio_np, sample_rate: int):
        """Route utterance to the correct worker thread's queue."""
        if label == "LOCAL" and self._local_worker:
            self._local_worker.enqueue(audio_np, sample_rate)
        elif label == "REMOTE" and self._remote_worker:
            self._remote_worker.enqueue(audio_np, sample_rate)

    def _on_transcription_result(self, label: str, text: str):
        """Called by worker threads when transcription is done."""
        self.signals.new_entry.emit(label, text)
        if self._transcript_store:
            self._transcript_store.add(label, text)

    # ── QT SLOTS (main thread) ───────────────────────────────────

    def _append_transcript(self, label: str, text: str):
        ts     = time.strftime("%H:%M:%S")
        target = self.local_text if label == "LOCAL" else self.remote_text
        cursor = target.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt_ts = QTextCharFormat()
        fmt_ts.setForeground(QColor("#2a4a5a" if label == "LOCAL" else "#4a2a1a"))
        cursor.setCharFormat(fmt_ts); cursor.insertText(f"\n[{ts}]  ")
        fmt_txt = QTextCharFormat()
        fmt_txt.setForeground(QColor("#00e5ff" if label == "LOCAL" else "#ff6b35"))
        cursor.setCharFormat(fmt_txt); cursor.insertText(text)
        target.setTextCursor(cursor); target.ensureCursorVisible()

    def _set_status(self, msg: str, color: str = "#888"):
        self.status_pill.setText(msg)
        self.status_pill.setStyleSheet(
            f"font-family:'Courier New'; font-size:10px; letter-spacing:2px;"
            f"padding:4px 10px; color:{color}; background:#111416; border:1px solid #222;")

    def _update_emotion(self, frame_rgb, emotions: dict, dominant: str):
        if frame_rgb is None: return
        try:
            h, w, ch = frame_rgb.shape
            img = QImage(frame_rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
            scaled = QPixmap.fromImage(img).scaled(
                self.cam_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.cam_label.setPixmap(scaled)
        except Exception:
            pass
        color_map = {"angry":"#ff4444","fear":"#ff8800","sad":"#4488ff",
                     "happy":"#00ff88","surprise":"#ffff00","neutral":"#888888",
                     "disgust":"#aa44ff"}
        c = color_map.get(dominant, "#888")
        conf = emotions.get(dominant, 0)
        self.emotion_label.setText(f"Emotion: {dominant.upper()} ({conf:.0f}%)")
        self.emotion_label.setStyleSheet(
            f"font-family:'Courier New'; font-size:12px; font-weight:bold; color:{c}; padding:4px;")

    def _update_fraud(self, score: int, verdict: str, details: dict):
        self.fraud_score_bar.setValue(score)
        if score >= 70:
            bar_color = "#ff4444"; v_color = "#ff4444"
        elif score >= 40:
            bar_color = "#ff8800"; v_color = "#ff8800"
        else:
            bar_color = "#00c853"; v_color = "#00c853"
        self.fraud_score_bar.setStyleSheet(
            f"QProgressBar {{ background:#111416; border:1px solid #222; height:14px;"
            f"font-family:'Courier New'; font-size:9px; color:#fff; text-align:center; }}"
            f"QProgressBar::chunk {{ background: {bar_color}; }}")
        self.fraud_verdict.setText(f"Verdict: {verdict}")
        self.fraud_verdict.setStyleSheet(
            f"font-family:'Courier New'; font-size:11px; color:{v_color}; padding:4px;")
        method = details.get("method", "")
        matches = details.get("matches", [])
        analysis = details.get("analysis", "")
        info = f"Method: {method}\n"
        if matches: info += f"Keywords: {', '.join(matches[:10])}\n"
        if analysis: info += f"Analysis: {analysis}\n"
        self.fraud_details.setText(info)

    def _tick_timer(self):
        if self._session_start:
            elapsed = int(time.monotonic() - self._session_start)
            h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
            self.timer_label.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def closeEvent(self, event):
        self._stop_capture(); event.accept()
