"""
ui/main_window.py — Phase 1

Key fix in this version:
  - Status signal carries color (str, str) not just (str)
  - _on_model_ready connected only once
  - Footer MIC label fixed (original had broken chained setStyleSheet call)
  - Added "TEST DEVICES" button that runs test_devices.py in a terminal
  - Device dropdowns show current default device highlighted
  - Console output visible so you can see VAD / Whisper logs in real time
"""

import time
import subprocess
import sys
import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFrame, QSplitter,
    QComboBox, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor

import sounddevice as sd

# ── Signals ───────────────────────────────────────────────────────

class TranscriptSignals(QObject):
    new_entry     = pyqtSignal(str, str)   # (label, text)
    status_update = pyqtSignal(str, str)   # (message, color)


# ── Model loader thread ───────────────────────────────────────────

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


# ── Stylesheet ────────────────────────────────────────────────────

STYLE = """
QMainWindow, QWidget#central { background: #0a0c0e; }

QLabel#title {
    font-family: 'Courier New', monospace;
    font-size: 13px; font-weight: bold;
    color: #c8ff00; letter-spacing: 4px;
}
QTextEdit {
    background: #060809; border: none;
    font-family: 'Courier New', monospace;
    font-size: 12px; padding: 14px;
    selection-background-color: #1a3040;
}
QPushButton#start_btn {
    background: #c8ff00; color: #0a0c0e;
    font-family: 'Courier New', monospace;
    font-size: 11px; font-weight: bold;
    letter-spacing: 2px; padding: 10px 28px;
    border: none; min-width: 110px;
}
QPushButton#start_btn:hover    { background: #d4ff33; }
QPushButton#start_btn:pressed  { background: #a8d900; }
QPushButton#start_btn:disabled { background: #2a2a2a; color: #555; }

QPushButton#stop_btn {
    background: transparent; color: #ff4444;
    font-family: 'Courier New', monospace;
    font-size: 11px; font-weight: bold;
    letter-spacing: 2px; padding: 10px 28px;
    border: 1px solid #ff4444; min-width: 110px;
}
QPushButton#stop_btn:hover    { background: #1a0000; }
QPushButton#stop_btn:disabled { color: #333; border-color: #333; }

QPushButton#clear_btn, QPushButton#test_btn {
    background: transparent; color: #555;
    font-family: 'Courier New', monospace;
    font-size: 10px; letter-spacing: 2px;
    padding: 8px 16px; border: 1px solid #222;
}
QPushButton#clear_btn:hover { color: #888; border-color: #444; }
QPushButton#test_btn  { color: #888; border-color: #333; }
QPushButton#test_btn:hover  { color: #ffaa00; border-color: #ffaa00; }

QLabel#status_pill {
    font-family: 'Courier New', monospace;
    font-size: 10px; letter-spacing: 2px; padding: 4px 10px;
}
QLabel#timer_label {
    font-family: 'Courier New', monospace;
    font-size: 11px; color: #444; letter-spacing: 2px;
}
QFrame#header_bar { background: #0d1014; border-bottom: 1px solid #1a1d20; }
QFrame#footer_bar { background: #0d1014; border-top:    1px solid #1a1d20; }

QComboBox {
    background: #111416; color: #888;
    font-family: 'Courier New', monospace;
    font-size: 9px; border: 1px solid #222;
    padding: 3px 8px; min-width: 200px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #111416; color: #888;
    selection-background-color: #1a2a30;
}
QSplitter::handle { background: #1a1d20; width: 1px; }
"""


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.signals        = TranscriptSignals()
        self.capture        = None
        self._session_start = None
        self._timer         = QTimer()
        self._timer.timeout.connect(self._tick_timer)
        self._model_loaded  = False

        self.setWindowTitle("ARREST ANALYZER — PHASE 1")
        self.resize(1100, 720)
        self.setMinimumSize(800, 500)
        self.setStyleSheet(STYLE)

        self._build_ui()
        self._connect_signals()
        self._load_devices()
        self._start_model_load()

    # ── UI ────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        root.addWidget(self._build_transcript_area(), stretch=1)
        root.addWidget(self._build_footer())

    def _build_header(self):
        bar = QFrame()
        bar.setObjectName("header_bar")
        bar.setFixedHeight(58)
        h = QHBoxLayout(bar)
        h.setContentsMargins(20, 0, 20, 0)

        title = QLabel("◈  DIGITAL ARREST ANALYZER  //  PHASE-1  //  TRANSCRIPT")
        title.setObjectName("title")
        h.addWidget(title)
        h.addStretch()

        self.status_pill = QLabel("◌  LOADING MODEL")
        self.status_pill.setObjectName("status_pill")
        self.status_pill.setStyleSheet(
            "color:#555; background:#111416; border:1px solid #222;"
        )
        h.addWidget(self.status_pill)
        h.addSpacing(12)

        test_btn = QPushButton("TEST DEVICES")
        test_btn.setObjectName("test_btn")
        test_btn.setToolTip("Runs test_devices.py — speak while it runs to find your mic index")
        test_btn.clicked.connect(self._run_device_test)
        h.addWidget(test_btn)
        h.addSpacing(8)

        self.clear_btn = QPushButton("CLEAR")
        self.clear_btn.setObjectName("clear_btn")
        self.clear_btn.clicked.connect(self._clear_transcripts)
        h.addWidget(self.clear_btn)
        h.addSpacing(8)

        self.start_btn = QPushButton("▶  START")
        self.start_btn.setObjectName("start_btn")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start_capture)
        h.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■  STOP")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_capture)
        h.addWidget(self.stop_btn)

        return bar

    def _build_transcript_area(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # LOCAL
        local_w = QWidget()
        local_w.setStyleSheet("background: #060809;")
        lv = QVBoxLayout(local_w)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        local_lbl = QLabel("◈  LOCAL  //  YOU  (microphone)")
        local_lbl.setStyleSheet(
            "font-family:'Courier New'; font-size:10px; font-weight:bold;"
            "letter-spacing:3px; color:#00e5ff; background:#001a1f;"
            "border-left:3px solid #00e5ff; padding:8px 14px;"
        )
        lv.addWidget(local_lbl)
        self.local_text = QTextEdit()
        self.local_text.setReadOnly(True)
        self.local_text.setPlaceholderText("[ waiting for speech — select your mic in the footer dropdown ]")
        self.local_text.setStyleSheet("color: #00e5ff;")
        lv.addWidget(self.local_text, stretch=1)

        # REMOTE
        remote_w = QWidget()
        remote_w.setStyleSheet("background: #060809;")
        rv = QVBoxLayout(remote_w)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)
        remote_lbl = QLabel("◈  REMOTE  //  VB-CABLE OUTPUT")
        remote_lbl.setStyleSheet(
            "font-family:'Courier New'; font-size:10px; font-weight:bold;"
            "letter-spacing:3px; color:#ff6b35; background:#1f0d00;"
            "border-left:3px solid #ff6b35; padding:8px 14px;"
        )
        rv.addWidget(remote_lbl)
        self.remote_text = QTextEdit()
        self.remote_text.setReadOnly(True)
        self.remote_text.setPlaceholderText("[ waiting for remote audio — select CABLE Output in the footer dropdown ]")
        self.remote_text.setStyleSheet("color: #ff6b35;")
        rv.addWidget(self.remote_text, stretch=1)

        splitter.addWidget(local_w)
        splitter.addWidget(remote_w)
        splitter.setSizes([550, 550])
        return splitter

    def _build_footer(self):
        bar = QFrame()
        bar.setObjectName("footer_bar")
        bar.setFixedHeight(42)
        h = QHBoxLayout(bar)
        h.setContentsMargins(20, 0, 20, 0)
        h.setSpacing(8)

        h.addWidget(self._small_label("MIC (LOCAL):"))
        self.mic_combo = QComboBox()
        self.mic_combo.setToolTip(
            "Select YOUR microphone here.\n"
            "Run TEST DEVICES (while speaking) to find the right index."
        )
        h.addWidget(self.mic_combo)

        h.addSpacing(16)
        h.addWidget(self._small_label("REMOTE (VB-CABLE OUTPUT):"))
        self.remote_combo = QComboBox()
        self.remote_combo.setToolTip(
            "Select 'CABLE Output (VB-Audio Virtual Cable)' here.\n"
            "In Zoom: Settings → Audio → Speaker → CABLE Input (VB-Audio)"
        )
        h.addWidget(self.remote_combo)

        h.addStretch()

        self.timer_label = QLabel("00:00:00")
        self.timer_label.setObjectName("timer_label")
        h.addWidget(self.timer_label)

        return bar

    def _small_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "font-family:'Courier New'; font-size:9px; color:#555; letter-spacing:1px;"
        )
        return lbl

    # ── Device loading ────────────────────────────────────────────

    def _load_devices(self):
        """Populate dropdowns. Auto-selects VB-Cable if found."""
        self.mic_combo.clear()
        self.remote_combo.clear()

        try:
            default_in = sd.default.device[0]
        except Exception:
            default_in = -1

        self.mic_combo.addItem("[ System Default ]", None)
        self.remote_combo.addItem("[ None — select VB-Cable Output ]", None)

        vbcable_remote_idx = None
        vbcable_mic_idx    = None

        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] <= 0:
                continue
            name  = d["name"]
            label = f"[{i}] {name[:46]}"
            if i == default_in:
                label += "  ◄ default"

            self.mic_combo.addItem(label, i)
            self.remote_combo.addItem(label, i)

            name_lower = name.lower()
            # Auto-detect VB-Cable Output for REMOTE
            if "cable output" in name_lower or "vb-audio" in name_lower:
                if vbcable_remote_idx is None:
                    vbcable_remote_idx = self.remote_combo.count() - 1
            # Auto-detect BlackHole (macOS)
            if "blackhole" in name_lower:
                if vbcable_remote_idx is None:
                    vbcable_remote_idx = self.remote_combo.count() - 1

        # Auto-select VB-Cable in REMOTE if found
        if vbcable_remote_idx is not None:
            self.remote_combo.setCurrentIndex(vbcable_remote_idx)

        # Auto-select system default mic for LOCAL
        if default_in >= 0:
            for j in range(self.mic_combo.count()):
                if self.mic_combo.itemData(j) == default_in:
                    self.mic_combo.setCurrentIndex(j)
                    break

    # ── Model load ────────────────────────────────────────────────

    def _connect_signals(self):
        self.signals.new_entry.connect(self._append_transcript)
        self.signals.status_update.connect(self._set_status)

    def _start_model_load(self):
        self.loader = ModelLoaderThread()
        self.loader.finished.connect(self._on_model_ready)   # only once
        self.loader.error.connect(
            lambda e: self._set_status(f"MODEL ERROR: {e}", "#ff4444")
        )
        self.loader.start()

    def _on_model_ready(self):
        self._model_loaded = True
        self._set_status("●  READY", "#c8ff00")
        self.start_btn.setEnabled(True)

    # ── Controls ──────────────────────────────────────────────────

    def _start_capture(self):
        import config
        from core.audio.capture import DualChannelCapture

        config.MIC_DEVICE_INDEX    = self.mic_combo.currentData()
        config.REMOTE_DEVICE_INDEX = self.remote_combo.currentData()

        # Warn if both are the same device (will capture same audio on both channels)
        if (config.MIC_DEVICE_INDEX is not None
                and config.MIC_DEVICE_INDEX == config.REMOTE_DEVICE_INDEX):
            QMessageBox.warning(
                self, "Same Device Selected",
                "MIC and REMOTE are set to the same device.\n\n"
                "LOCAL and REMOTE will transcribe identical audio.\n"
                "Select your real microphone for MIC, and CABLE Output for REMOTE."
            )

        try:
            self.capture = DualChannelCapture(on_utterance=self._on_utterance)
            self.capture.start()
        except Exception as e:
            QMessageBox.critical(self, "Capture Error",
                f"{e}\n\nTip: Run TEST DEVICES to verify device indices.")
            return

        self._session_start = time.monotonic()
        self._timer.start(1000)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_status("●  RECORDING", "#ff4444")

    def _stop_capture(self):
        if self.capture:
            self.capture.stop()
            self.capture = None
        self._timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_status("■  STOPPED", "#555")

    def _clear_transcripts(self):
        self.local_text.clear()
        self.remote_text.clear()

    def _run_device_test(self):
        """Launch test_devices.py in a new console window."""
        script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "test_devices.py")
        if not os.path.exists(script):
            QMessageBox.warning(self, "Not Found", f"test_devices.py not found at:\n{script}")
            return

        try:
            # Windows: open a new cmd window so output is visible
            if sys.platform == "win32":
                subprocess.Popen(
                    ["cmd", "/c", "start", "cmd", "/k",
                     sys.executable, script],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )
            else:
                subprocess.Popen([sys.executable, script])
            self._set_status("⚙  DEVICE TEST RUNNING — check console", "#ffaa00")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ── Utterance callback (audio daemon thread) ──────────────────

    def _on_utterance(self, label: str, audio, sample_rate: int):
        try:
            from core.audio import transcriber
            self.signals.status_update.emit("⚙  TRANSCRIBING...", "#ffaa00")
            text = transcriber.transcribe(audio, sample_rate)
            if text:
                self.signals.new_entry.emit(label, text)
            self.signals.status_update.emit("●  RECORDING", "#ff4444")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.signals.status_update.emit(f"ERR: {e}", "#ff4444")

    # ── Qt slots (main thread) ────────────────────────────────────

    def _append_transcript(self, label: str, text: str):
        ts     = time.strftime("%H:%M:%S")
        target = self.local_text if label == "LOCAL" else self.remote_text

        cursor = target.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt_ts = QTextCharFormat()
        fmt_ts.setForeground(QColor("#2a4a5a" if label == "LOCAL" else "#4a2a1a"))
        cursor.setCharFormat(fmt_ts)
        cursor.insertText(f"\n[{ts}]  ")

        fmt_txt = QTextCharFormat()
        fmt_txt.setForeground(QColor("#00e5ff" if label == "LOCAL" else "#ff6b35"))
        cursor.setCharFormat(fmt_txt)
        cursor.insertText(text)

        target.setTextCursor(cursor)
        target.ensureCursorVisible()

    def _set_status(self, msg: str, color: str = "#888"):
        self.status_pill.setText(msg)
        self.status_pill.setStyleSheet(
            f"font-family:'Courier New'; font-size:10px; letter-spacing:2px;"
            f"padding:4px 10px; color:{color}; background:#111416; border:1px solid #222;"
        )

    def _tick_timer(self):
        if self._session_start:
            elapsed = int(time.monotonic() - self._session_start)
            h = elapsed // 3600
            m = (elapsed % 3600) // 60
            s = elapsed % 60
            self.timer_label.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def closeEvent(self, event):
        self._stop_capture()
        event.accept()
