"""
ui/main_window.py — Simplified dual-channel transcript UI
UPDATED:
- No concurrent transcribe() calls (one worker per channel)
- _on_utterance() only enqueues audio
- Clean STOP: no late transcripts after stop
- Prevents queue backlog (drops if full)
"""
import time
import threading
import queue

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QSplitter,
    QComboBox, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
import sounddevice as sd

STYLE = """
* { font-family: 'Courier New', monospace; }
QMainWindow, QWidget { background: #0d0d0d; color: #ccc; }
QLabel#title { font-size: 13px; color: #c8ff00; letter-spacing: 3px; }
QLabel#lbl { font-size: 9px; color: #444; letter-spacing: 2px; }
QTextEdit { background: #080808; border: 1px solid #1a1a1a; font-size: 12px; padding: 10px; }
QPushButton#start { background: #c8ff00; color: #000; font-weight: bold; padding: 8px 22px; border: none; }
QPushButton#start:disabled { background: #2a2a2a; color: #555; }
QPushButton#stop  { background: transparent; color: #ff4444; border: 1px solid #ff4444; padding: 8px 22px; }
QPushButton#stop:disabled  { color: #333; border-color: #333; }
QPushButton#clear { background: transparent; color: #444; border: 1px solid #222; padding: 6px 14px; }
QComboBox { background: #111; color: #666; border: 1px solid #222; padding: 3px 8px; min-width: 160px; font-size: 9px; }
QLabel#status { font-size: 10px; letter-spacing: 2px; color: #555; padding: 3px 8px; border: 1px solid #222; background: #111; }
QLabel#timer { font-size: 11px; color: #444; letter-spacing: 2px; }
QSplitter::handle { background: #1a1a1a; width: 1px; }
"""


class Signals(QObject):
    new_entry = pyqtSignal(str, str)
    status    = pyqtSignal(str, str)   # (text, color)
    ready     = pyqtSignal()


class ModelLoader(QThread):
    done  = pyqtSignal()
    error = pyqtSignal(str)

    def run(self):
        try:
            from core.audio import transcriber
            transcriber.load_models()
            self.done.emit()
        except Exception as e:
            self.error.emit(str(e))


class TranscribeWorker(threading.Thread):
    """
    One worker per channel.
    Guarantees:
    - FIFO ordering
    - no concurrent transcribe() for that channel
    - safe stop (won't emit after stop_event)
    """
    def __init__(self, label: str, emit_text, emit_status, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.label = label
        self.emit_text = emit_text      # function(label, text)
        self.emit_status = emit_status  # function(text, color)
        self.stop_event = stop_event
        self.q = queue.Queue(maxsize=8)

    def submit(self, audio, sr):
        try:
            self.q.put_nowait((audio, sr))
        except queue.Full:
            # Drop to prevent lag explosion
            pass

    def run(self):
        from core.audio import transcriber

        while not self.stop_event.is_set():
            try:
                audio, sr = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            if self.stop_event.is_set():
                break

            try:
                text = transcriber.transcribe(audio, sr, label=self.label)
                if text and (not self.stop_event.is_set()):
                    self.emit_text(self.label, text)
            except Exception as e:
                if not self.stop_event.is_set():
                    self.emit_status(f"ERR: {e}", "#f44")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sig = Signals()
        self.capture = None

        self._t0 = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)

        # NEW: stop flag + workers
        self._stop_event = threading.Event()
        self._local_worker = None
        self._remote_worker = None

        self.setWindowTitle("ARREST ANALYZER")
        self.resize(1100, 680)
        self.setStyleSheet(STYLE)
        self._build()
        self._load_devices()
        self._start_loader()

    def _build(self):
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        hdr = QWidget(); hdr.setFixedHeight(52)
        hdr.setStyleSheet("background:#0a0a0a; border-bottom:1px solid #1a1a1a;")
        h = QHBoxLayout(hdr); h.setContentsMargins(18, 0, 18, 0)
        t = QLabel("◈  ARREST ANALYZER  //  DUAL-CHANNEL TRANSCRIPT"); t.setObjectName("title"); h.addWidget(t)
        h.addStretch()
        self.status_lbl = QLabel("◌  LOADING"); self.status_lbl.setObjectName("status"); h.addWidget(self.status_lbl)
        h.addSpacing(10)
        clr = QPushButton("CLEAR"); clr.setObjectName("clear"); clr.clicked.connect(self._clear); h.addWidget(clr)
        h.addSpacing(6)
        self.start_btn = QPushButton("▶  START"); self.start_btn.setObjectName("start")
        self.start_btn.setEnabled(False); self.start_btn.clicked.connect(self._start); h.addWidget(self.start_btn)
        self.stop_btn = QPushButton("■  STOP"); self.stop_btn.setObjectName("stop")
        self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self._stop); h.addWidget(self.stop_btn)
        root.addWidget(hdr)

        # Transcripts
        sp = QSplitter(Qt.Orientation.Horizontal)
        for label, color, attr in [("LOCAL  //  YOU", "#00e5ff", "local_text"), ("REMOTE  //  VB-CABLE", "#ff6b35", "remote_text")]:
            w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0,0,0,0); v.setSpacing(0)
            lbl = QLabel(f"  ◈  {label}")
            lbl.setStyleSheet(f"font-size:9px; letter-spacing:3px; color:{color}; background:#0a0a0a; padding:7px 12px; border-bottom:1px solid #1a1a1a;")
            v.addWidget(lbl)
            txt = QTextEdit(); txt.setReadOnly(True)
            txt.setPlaceholderText("[ waiting... ]")
            txt.setStyleSheet(f"color:{color};")
            setattr(self, attr, txt); v.addWidget(txt, 1)
            sp.addWidget(w)
        sp.setSizes([550, 550])
        root.addWidget(sp, 1)

        # Footer
        ftr = QWidget(); ftr.setFixedHeight(34)
        ftr.setStyleSheet("background:#0a0a0a; border-top:1px solid #1a1a1a;")
        f = QHBoxLayout(ftr); f.setContentsMargins(18, 0, 18, 0)
        f.addWidget(self._lbl("MIC")); self.mic_cb = QComboBox(); f.addWidget(self.mic_cb)
        f.addSpacing(16)
        f.addWidget(self._lbl("REMOTE")); self.rem_cb = QComboBox(); f.addWidget(self.rem_cb)
        f.addStretch()
        self.timer_lbl = QLabel("00:00:00"); self.timer_lbl.setObjectName("timer"); f.addWidget(self.timer_lbl)
        root.addWidget(ftr)

        c = QWidget(); c.setLayout(root); self.setCentralWidget(c)
        self.sig.new_entry.connect(self._append)
        self.sig.status.connect(lambda t, c: self._setstatus(t, c))
        self.sig.ready.connect(self._on_ready)

    def _lbl(self, t):
        l = QLabel(t); l.setObjectName("lbl"); return l

    def _load_devices(self):
        self.mic_cb.addItem("Default", None); self.rem_cb.addItem("Default", None)
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                s = f"[{i}] {d['name'][:38]}"
                self.mic_cb.addItem(s, i); self.rem_cb.addItem(s, i)

    def _start_loader(self):
        self.loader = ModelLoader()
        self.loader.done.connect(self._on_ready)
        self.loader.error.connect(lambda e: self._setstatus(f"ERR: {e}", "#f44"))
        self.loader.start()

    def _on_ready(self):
        self._setstatus("READY", "#c8ff00")
        self.start_btn.setEnabled(True)

    def _start(self):
        import config
        from core.audio.capture import DualChannelCapture

        # reset stop flag for a new session
        self._stop_event.clear()

        # start per-channel workers
        self._local_worker = TranscribeWorker(
            "LOCAL",
            emit_text=lambda l, t: self.sig.new_entry.emit(l, t),
            emit_status=lambda t, c: self.sig.status.emit(t, c),
            stop_event=self._stop_event,
        )
        self._remote_worker = TranscribeWorker(
            "REMOTE",
            emit_text=lambda l, t: self.sig.new_entry.emit(l, t),
            emit_status=lambda t, c: self.sig.status.emit(t, c),
            stop_event=self._stop_event,
        )
        self._local_worker.start()
        self._remote_worker.start()

        config.MIC_DEVICE_INDEX    = self.mic_cb.currentData()
        config.REMOTE_DEVICE_INDEX = self.rem_cb.currentData()

        try:
            self.capture = DualChannelCapture(on_utterance=self._on_utterance)
            self.capture.start()
        except Exception as e:
            self._stop_event.set()
            QMessageBox.critical(self, "Error", str(e))
            return

        self._t0 = time.monotonic()
        self._timer.start(1000)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._setstatus("● RECORDING", "#ff4444")

    def _stop(self):
        # stop capture (no new audio)
        if self.capture:
            try:
                self.capture.stop()
            except Exception:
                pass
            self.capture = None

        # stop workers (prevents late UI emits)
        self._stop_event.set()

        self._timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._setstatus("STOPPED", "#555")

    def _clear(self):
        self.local_text.clear()
        self.remote_text.clear()

    def _on_utterance(self, label, audio, sr):
        """
        Runs on capture thread.
        Only enqueue work here (never transcribe here).
        """
        if self._stop_event.is_set():
            return

        if label == "LOCAL" and self._local_worker:
            self._local_worker.submit(audio, sr)
        elif label == "REMOTE" and self._remote_worker:
            self._remote_worker.submit(audio, sr)

    def _append(self, label, text):
        ts = time.strftime("%H:%M:%S")
        target = self.local_text if label == "LOCAL" else self.remote_text
        color  = "#00e5ff"        if label == "LOCAL" else "#ff6b35"
        dim    = "#1a3040"        if label == "LOCAL" else "#3a1a0a"

        cur = target.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        f1 = QTextCharFormat(); f1.setForeground(QColor(dim))
        cur.setCharFormat(f1); cur.insertText(f"\n[{ts}]  ")
        f2 = QTextCharFormat(); f2.setForeground(QColor(color))
        cur.setCharFormat(f2); cur.insertText(text)
        target.setTextCursor(cur); target.ensureCursorVisible()

    def _setstatus(self, t, c="#888"):
        self.status_lbl.setText(t)
        self.status_lbl.setStyleSheet(
            f"font-size:10px;letter-spacing:2px;padding:3px 8px;color:{c};background:#111;border:1px solid #222;"
        )

    def _tick(self):
        if self._t0:
            e = int(time.monotonic() - self._t0)
            self.timer_lbl.setText(f"{e//3600:02d}:{(e%3600)//60:02d}:{e%60:02d}")

    def closeEvent(self, ev):
        self._stop()
        ev.accept()