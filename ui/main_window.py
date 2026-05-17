import time, subprocess, sys, os, threading
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QFrame, QSplitter, QComboBox, QMessageBox, QProgressBar,
    QGroupBox, QDialog,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor, QImage, QPixmap
import numpy as np, sounddevice as sd


class TranscriptSignals(QObject):
    new_entry     = pyqtSignal(str, str)
    status_update = pyqtSignal(str, str)
    emotion_frame = pyqtSignal(object, dict, str)
    fraud_result  = pyqtSignal(int, str, dict)


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


class FraudAlertDialog(QDialog):
    def __init__(self, score, verdict, summary, recommendation, emotion_score, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FRAUD ANALYSIS REPORT")
        self.setMinimumSize(520, 380)
        self.setStyleSheet("""
            QDialog { background: #0c0e14; }
            QLabel { font-family: 'Segoe UI'; color: #ccc; }
            QTextEdit { background: #080a10; color: #ddd; border: 1px solid #1a1d24;
                        border-radius: 6px; font-family: 'Consolas'; font-size: 12px; padding: 10px; }
            QPushButton { background: #c8ff00; color: #000; font-weight: 700;
                          padding: 10px 30px; border: none; border-radius: 4px;
                          font-family: 'Segoe UI'; font-size: 12px; }
            QPushButton:hover { background: #d4ff33; }
        """)
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        if score >= 70: color = "#ff4444"
        elif score >= 40: color = "#ff8800"
        else: color = "#00c853"

        header = QLabel(f"FRAUD RISK: {score}%")
        header.setStyleSheet(f"font-size: 22px; font-weight: 800; color: {color}; padding: 8px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        verdict_lbl = QLabel(verdict)
        verdict_lbl.setStyleSheet(f"font-size: 14px; font-weight: 600; color: {color}; padding: 4px;")
        verdict_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(verdict_lbl)

        emotion_lbl = QLabel(f"Emotion Score (fear+sad avg): {emotion_score:.1f}/100")
        emotion_lbl.setStyleSheet("font-size: 11px; color: #888; padding: 2px;")
        emotion_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(emotion_lbl)

        layout.addWidget(QLabel("SUMMARY:"))
        summary_te = QTextEdit()
        summary_te.setReadOnly(True)
        summary_te.setText(summary)
        summary_te.setMaximumHeight(120)
        layout.addWidget(summary_te)

        layout.addWidget(QLabel("RECOMMENDATION:"))
        rec_te = QTextEdit()
        rec_te.setReadOnly(True)
        rec_te.setText(recommendation)
        rec_te.setMaximumHeight(80)
        layout.addWidget(rec_te)

        ok_btn = QPushButton("UNDERSTOOD")
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn, alignment=Qt.AlignmentFlag.AlignCenter)


STYLE = """
QMainWindow, QWidget#central {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #080a0e, stop:1 #0c1018);
}
QLabel#title {
    font-family: 'Segoe UI'; font-size: 14px; font-weight: 700;
    color: #c8ff00; letter-spacing: 3px;
}
QTextEdit {
    background: rgba(6,8,12,0.95); border: 1px solid rgba(255,255,255,0.04);
    border-radius: 6px; font-family: 'Consolas', monospace;
    font-size: 12px; padding: 12px;
}
QPushButton#start_btn {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #b8e600, stop:1 #c8ff00);
    color: #0a0c0e; font-family: 'Segoe UI'; font-size: 11px; font-weight: 700;
    letter-spacing: 2px; padding: 10px 28px; border: none; border-radius: 4px;
}
QPushButton#start_btn:hover { background: #d4ff33; }
QPushButton#start_btn:disabled { background: #1a1a1a; color: #444; }
QPushButton#stop_btn {
    background: transparent; color: #ff4444; font-family: 'Segoe UI';
    font-size: 11px; font-weight: 700; letter-spacing: 2px; padding: 10px 28px;
    border: 1px solid rgba(255,68,68,0.5); border-radius: 4px;
}
QPushButton#stop_btn:hover { background: rgba(255,68,68,0.1); }
QPushButton#stop_btn:disabled { color: #333; border-color: #222; }
QPushButton#clear_btn, QPushButton#test_btn {
    background: rgba(255,255,255,0.03); color: #666; font-family: 'Segoe UI';
    font-size: 10px; padding: 8px 16px;
    border: 1px solid rgba(255,255,255,0.06); border-radius: 4px;
}
QPushButton#clear_btn:hover { color: #999; }
QPushButton#test_btn:hover { color: #ffaa00; border-color: rgba(255,170,0,0.4); }
QLabel#status_pill {
    font-family: 'Consolas'; font-size: 10px; letter-spacing: 2px;
    padding: 5px 12px; border-radius: 12px;
}
QLabel#timer_label { font-family: 'Consolas'; font-size: 12px; color: #555; letter-spacing: 3px; }
QFrame#header_bar { background: rgba(13,16,20,0.95); border-bottom: 1px solid rgba(200,255,0,0.08); }
QFrame#footer_bar { background: rgba(13,16,20,0.95); border-top: 1px solid rgba(255,255,255,0.05); }
QComboBox {
    background: rgba(17,20,22,0.9); color: #888; font-family: 'Consolas';
    font-size: 9px; border: 1px solid rgba(255,255,255,0.06); border-radius: 3px;
    padding: 4px 8px; min-width: 200px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background: #111416; color: #888; }
QSplitter::handle { background: rgba(255,255,255,0.03); width: 1px; }
QGroupBox {
    font-family: 'Segoe UI'; font-size: 10px; font-weight: 600; color: #888;
    border: 1px solid rgba(255,255,255,0.05); border-radius: 8px;
    margin-top: 10px; padding: 16px 8px 8px 8px;
}
QGroupBox::title { subcontrol-origin: margin; padding: 0 8px; }
QProgressBar {
    background: rgba(17,20,22,0.9); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 7px; height: 16px; font-family: 'Consolas'; font-size: 9px;
    color: #fff; text-align: center;
}
QProgressBar::chunk { border-radius: 6px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.signals = TranscriptSignals()
        self.capture = None
        self._session_start = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick_timer)
        self._model_loaded = False
        self._local_worker = self._remote_worker = None
        self._transcript_store = None
        self._emotion_detector = None
        self._fraud_thread = None
        self._current_emotion = {"dominant": "neutral", "scores": {}}
        self._emotion_lock = threading.Lock()
        self._last_notif_time = 0
        self._alert_showing = False

        self.setWindowTitle("ARREST ANALYZER")
        self.resize(1360, 820)
        self.setMinimumSize(960, 600)
        self.setStyleSheet(STYLE)
        self._build_ui()
        self._connect_signals()
        self._load_devices()
        self._start_model_load()

    def _build_ui(self):
        central = QWidget(); central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        root.addWidget(self._build_header())
        root.addWidget(self._build_main_area(), stretch=1)
        root.addWidget(self._build_footer())

    def _build_header(self):
        bar = QFrame(); bar.setObjectName("header_bar"); bar.setFixedHeight(56)
        h = QHBoxLayout(bar); h.setContentsMargins(24,0,24,0)
        title = QLabel("DIGITAL ARREST ANALYZER")
        title.setObjectName("title"); h.addWidget(title); h.addStretch()
        self.status_pill = QLabel("LOADING MODEL...")
        self.status_pill.setObjectName("status_pill")
        self.status_pill.setStyleSheet("color:#555; background:rgba(17,20,22,0.9); border:1px solid rgba(255,255,255,0.06);")
        h.addWidget(self.status_pill); h.addSpacing(12)
        for name, oid, slot in [("TEST DEVICES","test_btn",self._run_device_test),("CLEAR","clear_btn",self._clear_transcripts)]:
            b = QPushButton(name); b.setObjectName(oid); b.clicked.connect(slot)
            h.addWidget(b); h.addSpacing(6)
        self.start_btn = QPushButton("START"); self.start_btn.setObjectName("start_btn")
        self.start_btn.setEnabled(False); self.start_btn.clicked.connect(self._start_capture)
        h.addWidget(self.start_btn); h.addSpacing(4)
        self.stop_btn = QPushButton("STOP"); self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self._stop_capture)
        h.addWidget(self.stop_btn)
        return bar

    def _build_main_area(self):
        splitter = QSplitter(Qt.Orientation.Horizontal); splitter.setHandleWidth(1)
        left = QWidget(); left.setStyleSheet("background:transparent;")
        lv = QVBoxLayout(left); lv.setContentsMargins(8,8,4,8); lv.setSpacing(6)
        tsplit = QSplitter(Qt.Orientation.Horizontal); tsplit.setHandleWidth(1)
        for lt, col, bg, attr in [
            ("LOCAL // YOU","#00e5ff","#001a1f","local_text"),
            ("REMOTE // CALLER","#ff6b35","#1f0d00","remote_text")]:
            w = QWidget(); w.setStyleSheet("background:transparent;")
            v = QVBoxLayout(w); v.setContentsMargins(0,0,0,0); v.setSpacing(0)
            lbl = QLabel(f"  {lt}")
            lbl.setStyleSheet(f"font-family:'Segoe UI'; font-size:10px; font-weight:700; letter-spacing:3px; color:{col}; background:{bg}; border-left:3px solid {col}; padding:9px 14px; border-radius:4px 4px 0 0;")
            v.addWidget(lbl)
            te = QTextEdit(); te.setReadOnly(True); te.setPlaceholderText("[ waiting... ]")
            te.setStyleSheet(f"color:{col}; border-radius:0 0 6px 6px;")
            v.addWidget(te, stretch=1); setattr(self, attr, te); tsplit.addWidget(w)
        tsplit.setSizes([400,400]); lv.addWidget(tsplit, stretch=1); splitter.addWidget(left)

        right = QWidget(); right.setStyleSheet("background:transparent;")
        rv = QVBoxLayout(right); rv.setContentsMargins(4,8,8,8); rv.setSpacing(8)

        cg = QGroupBox("WEBCAM // EMOTION")
        cg.setStyleSheet("QGroupBox{color:#00e5ff; border-color:rgba(0,229,255,0.15);} QGroupBox::title{color:#00e5ff;}")
        cl = QVBoxLayout(cg)
        self.cam_label = QLabel("[ webcam off ]")
        self.cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_label.setMinimumSize(300,220)
        self.cam_label.setStyleSheet("color:#333; font-family:'Consolas'; font-size:10px; background:rgba(6,8,12,0.9); border-radius:6px;")
        cl.addWidget(self.cam_label)
        self.emotion_label = QLabel("Emotion: --")
        self.emotion_label.setStyleSheet("font-family:'Segoe UI'; font-size:13px; font-weight:700; color:#00e5ff; padding:6px;")
        cl.addWidget(self.emotion_label)
        rv.addWidget(cg)

        fg = QGroupBox("FRAUD ANALYSIS (LLM)")
        fg.setStyleSheet("QGroupBox{color:#ff4444; border-color:rgba(255,68,68,0.15);} QGroupBox::title{color:#ff4444;}")
        fl = QVBoxLayout(fg)
        self.fraud_score_bar = QProgressBar()
        self.fraud_score_bar.setRange(0,100); self.fraud_score_bar.setValue(0)
        self.fraud_score_bar.setFormat("FRAUD RISK: %p%"); fl.addWidget(self.fraud_score_bar)
        self.fraud_verdict = QLabel("Waiting for data...")
        self.fraud_verdict.setStyleSheet("font-family:'Segoe UI'; font-size:11px; color:#666; padding:4px;")
        self.fraud_verdict.setWordWrap(True); fl.addWidget(self.fraud_verdict)
        self.fraud_summary = QTextEdit(); self.fraud_summary.setReadOnly(True)
        self.fraud_summary.setMaximumHeight(120)
        self.fraud_summary.setStyleSheet("color:#888; font-size:11px;")
        self.fraud_summary.setPlaceholderText("[ LLM analysis will appear here ]")
        fl.addWidget(self.fraud_summary)
        rv.addWidget(fg); rv.addStretch()
        splitter.addWidget(right); splitter.setSizes([820,400])
        return splitter

    def _build_footer(self):
        bar = QFrame(); bar.setObjectName("footer_bar"); bar.setFixedHeight(44)
        h = QHBoxLayout(bar); h.setContentsMargins(24,0,24,0); h.setSpacing(8)
        h.addWidget(self._lbl("MIC:"))
        self.mic_combo = QComboBox(); h.addWidget(self.mic_combo); h.addSpacing(16)
        h.addWidget(self._lbl("REMOTE:"))
        self.remote_combo = QComboBox(); h.addWidget(self.remote_combo); h.addStretch()
        self.timer_label = QLabel("00:00:00"); self.timer_label.setObjectName("timer_label")
        h.addWidget(self.timer_label)
        return bar

    def _lbl(self, t):
        l = QLabel(t); l.setStyleSheet("font-family:'Segoe UI'; font-size:9px; color:#555;"); return l

    def _load_devices(self):
        self.mic_combo.clear(); self.remote_combo.clear()
        try: di = sd.default.device[0]
        except: di = -1
        self.mic_combo.addItem("[ System Default ]", None)
        self.remote_combo.addItem("[ None ]", None)
        vb = None
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] <= 0: continue
            lab = f"[{i}] {d['name'][:46]}"
            if i == di: lab += " << default"
            self.mic_combo.addItem(lab, i); self.remote_combo.addItem(lab, i)
            if "cable output" in d["name"].lower() or "vb-audio" in d["name"].lower():
                if vb is None: vb = self.remote_combo.count() - 1
        if vb: self.remote_combo.setCurrentIndex(vb)
        if di >= 0:
            for j in range(self.mic_combo.count()):
                if self.mic_combo.itemData(j) == di: self.mic_combo.setCurrentIndex(j); break

    def _connect_signals(self):
        self.signals.new_entry.connect(self._append_transcript)
        self.signals.status_update.connect(self._set_status)
        self.signals.emotion_frame.connect(self._update_emotion)
        self.signals.fraud_result.connect(self._update_fraud)

    def _start_model_load(self):
        self.loader = ModelLoaderThread()
        self.loader.finished.connect(self._on_model_ready)
        self.loader.error.connect(lambda e: self._set_status(f"ERROR: {e}", "#ff4444"))
        self.loader.start()

    def _on_model_ready(self):
        self._model_loaded = True
        self._set_status("READY", "#c8ff00"); self.start_btn.setEnabled(True)

    def _start_capture(self):
        import config
        from core.audio.capture import DualChannelCapture
        from core.audio.transcription_worker import TranscriptionWorker
        from core.audio.transcript_store import TranscriptStore
        from core.vision.emotion_detector import EmotionDetector
        from core.fusion.fraud_analyzer import FraudAnalyzerThread

        config.MIC_DEVICE_INDEX = self.mic_combo.currentData()
        config.REMOTE_DEVICE_INDEX = self.remote_combo.currentData()
        self._transcript_store = TranscriptStore()
        self._local_worker = TranscriptionWorker("LOCAL", self._on_transcription_result,
            lambda m, c: self.signals.status_update.emit(m, c))
        self._remote_worker = TranscriptionWorker("REMOTE", self._on_transcription_result,
            lambda m, c: self.signals.status_update.emit(m, c))
        self._local_worker.start(); self._remote_worker.start()
        try:
            self.capture = DualChannelCapture(on_utterance=self._on_utterance)
            self.capture.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e)); return
        try:
            self._emotion_detector = EmotionDetector(
                camera_index=config.WEBCAM_INDEX, analyze_every_n=config.EMOTION_INTERVAL,
                on_frame=lambda f, e, d: self.signals.emotion_frame.emit(f, e, d),
                on_status=lambda m, c: self.signals.status_update.emit(m, c))
            self._emotion_detector.start()
        except Exception as e: print(f"[CV] {e}")
        try:
            self._fraud_thread = FraudAnalyzerThread(
                get_entries_fn=self._transcript_store.get_entries,
                get_emotion_history_fn=self._transcript_store.get_emotion_history,
                on_result=lambda s, v, d: self.signals.fraud_result.emit(s, v, d),
                interval_s=config.FRAUD_CHECK_INTERVAL_S)
            self._fraud_thread.start()
        except Exception as e: print(f"[Fraud] {e}")
        self._session_start = time.monotonic(); self._timer.start(1000)
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self._set_status("RECORDING", "#ff4444")

    def _stop_capture(self):
        for obj in [self.capture, self._local_worker, self._remote_worker,
                     self._emotion_detector, self._fraud_thread]:
            if obj:
                try: obj.stop()
                except: pass
        self.capture = self._local_worker = self._remote_worker = None
        self._emotion_detector = self._fraud_thread = None
        self._timer.stop()
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self._set_status("STOPPED", "#555")
        self.cam_label.setText("[ webcam off ]"); self.cam_label.setPixmap(QPixmap())

    def _clear_transcripts(self):
        self.local_text.clear(); self.remote_text.clear()

    def _run_device_test(self):
        script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_devices.py")
        if os.path.exists(script) and sys.platform == "win32":
            try: subprocess.Popen(["cmd","/c","start","cmd","/k",sys.executable,script], creationflags=subprocess.CREATE_NEW_CONSOLE)
            except: pass

    def _on_utterance(self, label, audio_np, sr):
        if label == "LOCAL" and self._local_worker: self._local_worker.enqueue(audio_np, sr)
        elif label == "REMOTE" and self._remote_worker: self._remote_worker.enqueue(audio_np, sr)

    def _on_transcription_result(self, label, text):
        self.signals.new_entry.emit(label, text)
        if self._transcript_store: self._transcript_store.add(label, text)

    def _append_transcript(self, label, text):
        ts = time.strftime("%H:%M:%S")
        target = self.local_text if label == "LOCAL" else self.remote_text
        cursor = target.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt_ts = QTextCharFormat()
        fmt_ts.setForeground(QColor("#1a3a4a" if label == "LOCAL" else "#3a2a1a"))
        cursor.setCharFormat(fmt_ts); cursor.insertText(f"\n[{ts}]  ")
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#00e5ff" if label == "LOCAL" else "#ff6b35"))
        cursor.setCharFormat(fmt); cursor.insertText(text)
        target.setTextCursor(cursor); target.ensureCursorVisible()

    def _set_status(self, msg, color="#888"):
        self.status_pill.setText(msg)
        self.status_pill.setStyleSheet(
            f"font-family:'Consolas'; font-size:10px; letter-spacing:2px; padding:5px 12px;"
            f"color:{color}; background:rgba(17,20,22,0.9); border:1px solid {color}40; border-radius:12px;")

    def _update_emotion(self, frame_rgb, emotions, dominant):
        if frame_rgb is not None:
            try:
                h, w, ch = frame_rgb.shape
                img = QImage(frame_rgb.data, w, h, w*ch, QImage.Format.Format_RGB888)
                self.cam_label.setPixmap(QPixmap.fromImage(img).scaled(
                    self.cam_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            except: pass
        with self._emotion_lock:
            self._current_emotion = {"dominant": dominant, "scores": emotions}
        if self._transcript_store:
            self._transcript_store.add_emotion(dominant, emotions)
        cmap = {"angry":"#ff4444","fear":"#ff8800","sad":"#4488ff","happy":"#00ff88",
                "surprise":"#ffff00","neutral":"#666","disgust":"#aa44ff"}
        c = cmap.get(dominant, "#666")
        conf = emotions.get(dominant, 0)
        self.emotion_label.setText(f"Emotion: {dominant.upper()} ({conf:.0f}%)")
        self.emotion_label.setStyleSheet(f"font-family:'Segoe UI'; font-size:13px; font-weight:700; color:{c}; padding:6px;")

    def _update_fraud(self, score, verdict, details):
        self.fraud_score_bar.setValue(score)
        if score >= 70: bc = "#ff4444"
        elif score >= 40: bc = "#ff8800"
        else: bc = "#00c853"
        self.fraud_score_bar.setStyleSheet(
            f"QProgressBar{{background:rgba(17,20,22,0.9); border:1px solid rgba(255,255,255,0.06);"
            f"border-radius:7px; height:16px; font-family:'Consolas'; font-size:9px; color:#fff; text-align:center;}}"
            f"QProgressBar::chunk{{background:{bc}; border-radius:6px;}}")
        eb = details.get("emotion_boost", 0)
        es = details.get("emotion_score", 0)
        extra = f" [emotion boost: +{eb}]" if eb > 0 else ""
        self.fraud_verdict.setText(f"{verdict}{extra}")
        self.fraud_verdict.setStyleSheet(f"font-family:'Segoe UI'; font-size:11px; color:{bc}; padding:4px;")
        summary = details.get("summary", "")
        rec = details.get("recommendation", "")
        self.fraud_summary.setText(f"{summary}\n\n{rec}" if rec else summary)

        if score >= 40:
            self._show_fraud_alert(score, verdict, summary, rec, es)

    def _show_fraud_alert(self, score, verdict, summary, recommendation, emotion_score):
        now = time.time()
        if now - self._last_notif_time < 30 or self._alert_showing:
            return
        self._last_notif_time = now

        try:
            from plyer import notification
            notification.notify(
                title="FRAUD ALERT" if score >= 70 else "Suspicious Activity",
                message=f"Risk: {score}% - {verdict}"[:256],
                app_name="Arrest Analyzer", timeout=10)
        except: pass

        self._alert_showing = True
        dlg = FraudAlertDialog(score, verdict, summary, recommendation, emotion_score, self)
        dlg.exec()
        self._alert_showing = False

    def _tick_timer(self):
        if self._session_start:
            e = int(time.monotonic() - self._session_start)
            self.timer_label.setText(f"{e//3600:02d}:{(e%3600)//60:02d}:{e%60:02d}")

    def closeEvent(self, ev):
        self._stop_capture(); ev.accept()
