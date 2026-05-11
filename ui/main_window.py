"""
ui/main_window.py

Phase 1 UI — Industrial surveillance aesthetic.
Dark, dense, utilitarian. Like a forensic monitoring station.

Layout:
  ┌─────────────────────────────────────────────────┐
  │  HEADER: title + status bar + controls           │
  ├──────────────────┬──────────────────────────────┤
  │  LOCAL           │  REMOTE                      │
  │  transcript      │  transcript                  │
  │  (scrolling)     │  (scrolling)                 │
  ├──────────────────┴──────────────────────────────┤
  │  FOOTER: device info + session timer             │
  └─────────────────────────────────────────────────┘
"""

import time
import threading
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFrame, QSplitter,
    QStatusBar, QComboBox, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QFont, QColor, QPalette, QTextCharFormat, QTextCursor

import sounddevice as sd

# ── Signals bridge (audio thread → Qt main thread) ────────────────

class TranscriptSignals(QObject):
    new_entry    = pyqtSignal(str, str)   # (label, text)
    status_update = pyqtSignal(str)
    model_ready  = pyqtSignal()


# ── Whisper loader thread ─────────────────────────────────────────

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


# ── Main Window ───────────────────────────────────────────────────

STYLE = """
QMainWindow, QWidget#central {
    background: #0a0c0e;
}

QLabel#title {
    font-family: 'Courier New', monospace;
    font-size: 13px;
    font-weight: bold;
    color: #c8ff00;
    letter-spacing: 4px;
    text-transform: uppercase;
}

QLabel#channel_label {
    font-family: 'Courier New', monospace;
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 3px;
    padding: 6px 12px;
}

QLabel#channel_local {
    color: #00e5ff;
    background: #001a1f;
    border-left: 3px solid #00e5ff;
}

QLabel#channel_remote {
    color: #ff6b35;
    background: #1f0d00;
    border-left: 3px solid #ff6b35;
}

QTextEdit {
    background: #060809;
    border: none;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.6;
    padding: 14px;
    selection-background-color: #1a3040;
}

QPushButton#start_btn {
    background: #c8ff00;
    color: #0a0c0e;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 2px;
    padding: 10px 28px;
    border: none;
    min-width: 110px;
}
QPushButton#start_btn:hover { background: #d4ff33; }
QPushButton#start_btn:pressed { background: #a8d900; }
QPushButton#start_btn:disabled { background: #2a2a2a; color: #555; }

QPushButton#stop_btn {
    background: transparent;
    color: #ff4444;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 2px;
    padding: 10px 28px;
    border: 1px solid #ff4444;
    min-width: 110px;
}
QPushButton#stop_btn:hover { background: #1a0000; }
QPushButton#stop_btn:disabled { color: #333; border-color: #333; }

QPushButton#clear_btn {
    background: transparent;
    color: #555;
    font-family: 'Courier New', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    padding: 8px 16px;
    border: 1px solid #222;
}
QPushButton#clear_btn:hover { color: #888; border-color: #444; }

QLabel#status_pill {
    font-family: 'Courier New', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    padding: 4px 10px;
}

QLabel#timer_label {
    font-family: 'Courier New', monospace;
    font-size: 11px;
    color: #444;
    letter-spacing: 2px;
}

QFrame#divider {
    background: #1a1d20;
    max-width: 1px;
}

QFrame#header_bar {
    background: #0d1014;
    border-bottom: 1px solid #1a1d20;
}

QFrame#footer_bar {
    background: #0d1014;
    border-top: 1px solid #1a1d20;
}

QComboBox {
    background: #111416;
    color: #666;
    font-family: 'Courier New', monospace;
    font-size: 9px;
    border: 1px solid #222;
    padding: 3px 8px;
    min-width: 180px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #111416;
    color: #888;
    selection-background-color: #1a2a30;
}

QSplitter::handle {
    background: #1a1d20;
    width: 1px;
}
"""


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.signals   = TranscriptSignals()
        self.capture   = None
        self._session_start = None
        self._timer    = QTimer()
        self._timer.timeout.connect(self._tick_timer)
        self._model_loaded = False

        self.setWindowTitle("ARREST ANALYZER — PHASE 1")
        self.resize(1100, 700)
        self.setMinimumSize(800, 500)
        self.setStyleSheet(STYLE)

        self._build_ui()
        self._connect_signals()
        self._load_devices()
        self._start_model_load()

    # ── UI Construction ───────────────────────────────────────────

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

        # Status pill
        self.status_pill = QLabel("◌  LOADING MODEL")
        self.status_pill.setObjectName("status_pill")
        self.status_pill.setStyleSheet("color: #555; background: #111416; border: 1px solid #222;")
        h.addWidget(self.status_pill)

        h.addSpacing(12)

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

        # ── LOCAL panel ───────────────────────────────────────────
        local_w = QWidget()
        local_w.setStyleSheet("background: #060809;")
        lv = QVBoxLayout(local_w)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)

        local_lbl = QLabel("◈  LOCAL  //  YOU")
        local_lbl.setObjectName("channel_label channel_local")
        local_lbl.setObjectName("channel_local")
        local_lbl.setStyleSheet(
            "font-family: 'Courier New'; font-size:10px; font-weight:bold;"
            "letter-spacing:3px; color:#00e5ff; background:#001a1f;"
            "border-left:3px solid #00e5ff; padding:8px 14px;"
        )
        lv.addWidget(local_lbl)

        self.local_text = QTextEdit()
        self.local_text.setReadOnly(True)
        self.local_text.setPlaceholderText("[ waiting for speech... ]")
        self.local_text.setStyleSheet("color: #00e5ff;")
        lv.addWidget(self.local_text, stretch=1)

        # ── REMOTE panel ──────────────────────────────────────────
        remote_w = QWidget()
        remote_w.setStyleSheet("background: #060809;")
        rv = QVBoxLayout(remote_w)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)

        remote_lbl = QLabel("◈  REMOTE  //  VB-CABLE")
        remote_lbl.setStyleSheet(
            "font-family: 'Courier New'; font-size:10px; font-weight:bold;"
            "letter-spacing:3px; color:#ff6b35; background:#1f0d00;"
            "border-left:3px solid #ff6b35; padding:8px 14px;"
        )
        rv.addWidget(remote_lbl)

        self.remote_text = QTextEdit()
        self.remote_text.setReadOnly(True)
        self.remote_text.setPlaceholderText("[ waiting for remote audio... ]")
        self.remote_text.setStyleSheet("color: #ff6b35;")
        rv.addWidget(self.remote_text, stretch=1)

        splitter.addWidget(local_w)
        splitter.addWidget(remote_w)
        splitter.setSizes([550, 550])
        return splitter

    def _build_footer(self):
        bar = QFrame()
        bar.setObjectName("footer_bar")
        bar.setFixedHeight(36)
        h = QHBoxLayout(bar)
        h.setContentsMargins(20, 0, 20, 0)

        # Device dropdowns
        h.addWidget(QLabel("MIC:").setStyleSheet("color:#333;") or self._small_label("MIC"))
        self.mic_combo = QComboBox()
        h.addWidget(self.mic_combo)

        h.addSpacing(20)
        h.addWidget(self._small_label("REMOTE (VB-CABLE):"))
        self.remote_combo = QComboBox()
        h.addWidget(self.remote_combo)

        h.addStretch()

        self.timer_label = QLabel("00:00:00")
        self.timer_label.setObjectName("timer_label")
        h.addWidget(self.timer_label)

        return bar

    def _small_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "font-family:'Courier New'; font-size:9px; color:#333; letter-spacing:1px;"
        )
        return lbl

    # ── Signals & Slots ───────────────────────────────────────────

    def _connect_signals(self):
        self.signals.new_entry.connect(self._append_transcript)
        self.signals.status_update.connect(self._set_status)
        self.signals.model_ready.connect(self._on_model_ready)

    def _load_devices(self):
        devices = sd.query_devices()
        self.mic_combo.addItem("Default", None)
        self.remote_combo.addItem("Default (set VB-Cable)", None)
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                label = f"[{i}] {d['name'][:40]}"
                self.mic_combo.addItem(label, i)
                self.remote_combo.addItem(label, i)

    def _start_model_load(self):
        self.loader = ModelLoaderThread()
        self.loader.finished.connect(self._on_model_ready)
        self.loader.error.connect(lambda e: self._set_status(f"MODEL ERROR: {e}"))
        self.loader.start()

    def _on_model_ready(self):
        self._model_loaded = True
        self._set_status("READY", color="#c8ff00")
        self.start_btn.setEnabled(True)

    # ── Controls ──────────────────────────────────────────────────

    def _start_capture(self):
        import config
        from core.audio.capture import DualChannelCapture

        # Apply device selections
        mic_idx    = self.mic_combo.currentData()
        remote_idx = self.remote_combo.currentData()
        config.MIC_DEVICE_INDEX    = mic_idx
        config.REMOTE_DEVICE_INDEX = remote_idx

        try:
            self.capture = DualChannelCapture(on_utterance=self._on_utterance)
            self.capture.start()
        except Exception as e:
            QMessageBox.critical(self, "Capture Error", str(e))
            return

        self._session_start = time.monotonic()
        self._timer.start(1000)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_status("● RECORDING", color="#ff4444")

    def _stop_capture(self):
        if self.capture:
            self.capture.stop()
            self.capture = None
        self._timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_status("STOPPED", color="#555")

    def _clear_transcripts(self):
        self.local_text.clear()
        self.remote_text.clear()

    # ── Utterance callback (called from audio thread) ─────────────

    def _on_utterance(self, label: str, audio, sample_rate: int):
        """Runs on a daemon thread — transcribes then emits signal to Qt."""
        try:
            from core.audio import transcriber
            self.signals.status_update.emit("⚙  TRANSCRIBING...")
            text = transcriber.transcribe(audio, sample_rate)
            if text:
                self.signals.new_entry.emit(label, text)
            self.signals.status_update.emit("● RECORDING")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.signals.status_update.emit(f"ERR: {e}")

    # ── Qt slots (main thread) ────────────────────────────────────

    def _append_transcript(self, label: str, text: str):
        import time as _t
        ts = _t.strftime("%H:%M:%S")
        target = self.local_text if label == "LOCAL" else self.remote_text

        cursor = target.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Timestamp in dim color
        fmt_ts = QTextCharFormat()
        fmt_ts.setForeground(QColor("#2a4a5a" if label == "LOCAL" else "#4a2a1a"))
        cursor.setCharFormat(fmt_ts)
        cursor.insertText(f"\n[{ts}]  ")

        # Transcript text in channel color
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
            h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
            self.timer_label.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def closeEvent(self, event):
        self._stop_capture()
        event.accept()
